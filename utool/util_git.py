#!/usr/bin/env python
"""
TODO: export from utool

        python -m utool.util_inspect check_module_usage --pat="util_git.py"
"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
import os
import re
import six
from six.moves import zip
from os.path import exists, join, dirname, isdir, basename
from utool import util_dev
from utool import util_inject
from utool import util_class
from utool import util_path
from utool import util_decor
from utool import util_list
print, rrr, profile = util_inject.inject2(__name__)


def _syscmd(cmdstr):
    print('RUN> ' + cmdstr)
    os.system(cmdstr)


def _cd(dir_):
    dir_ = util_path.truepath(dir_)
    print('> cd ' + dir_)
    os.chdir(dir_)


@six.add_metaclass(util_class.ReloadingMetaclass)
class RepoManager(util_dev.NiceRepr):
    """
    Batch git operations on multiple repos
    """
    def __init__(rman, repo_urls=None, code_dir=None, userid=None,
                 permitted_repos=None, label='', pythoncmd=None):
        if userid is None:
            userid = None
        if permitted_repos is None:
            permitted_repos = []
        rman.permitted_repos = permitted_repos
        rman.code_dir = code_dir
        rman.userid = userid
        rman.repos = []
        rman.label = label
        rman.pythoncmd = pythoncmd

        if repo_urls is not None:
            rman.add_repos(repo_urls)

    def union(rman, other):
        new = RepoManager()
        new.userid = rman.userid
        new.code_dir = rman.code_dir
        new.permitted_repos = rman.permitted_repos + other.permitted_repos
        new.repos = rman.repos + other.repos
        return new

    @property
    def repo_urls(rman):
        return [repo.url for repo in rman.repos]

    @property
    def repo_dirs(rman):
        return [repo.url for repo in rman.repos]

    def __nice__(rman):
        return '(num=%d)' % (len(rman.repo_urls))

    def __getitem__(rman, name):
        for repo in rman.repos:
            if name in repo.aliases:
                return repo
        raise KeyError(name)

    def ensure(rman):
        print('Ensuring that respos are checked out')
        for repo in rman.repos:
            if repo.url is not None:
                repo.clone()

    def add_repo(rman, repo):
        repo._fix_url(rman.userid, rman.permitted_repos)
        rman.repos.append(repo)

    def add_repos(rman, repo_urls=None, code_dir=None):
        if code_dir is None:
            code_dir = rman.code_dir
        assert code_dir is not None, 'Must specify the checkout code_dir'
        repos = [Repo(url, code_dir, pythoncmd=rman.pythoncmd)
                 for url in repo_urls]
        for repo in repos:
            repo._fix_url(rman.userid, rman.permitted_repos)
        rman.repos.extend(repos)

    def issue(rman, command, sudo=False):
        """ Runs a command on all of managed repos """
        print('+------- GG_COMMAND -------')
        print('| sudo=%s' % sudo)
        print('| command=%s' % command)
        for repo in rman.repos:
            if exists(repo.dpath):
                repo.issue(command, sudo=sudo)
            else:
                print('Repo %r not found' % (repo,))
        print('L___ FINISHED GG_COMMAND ___')

    def check_importable(rman):
        import utool as ut
        label = ' %s' % rman.label if rman.label else rman.label
        missing = []
        print('Checking if%s modules are importable' % (label,))
        msg_list = []
        recommended_fixes = []
        for repo in rman.repos:
            flag, msg, errors = repo.check_importable()
            if not flag:
                msg_list.append('  * !!!%s REPO %s HAS IMPORT ISSUES' % (label.upper(), repo,))
                if any([str(ex).find('undefined symbol') > -1 for ex in errors]):
                    recommended_fixes.append('rebuild')
                else:
                    recommended_fixes.append(None)
                if ut.VERBOSE:
                    msg_list.append(ut.indent(msg, '    '))
                missing.append(repo)
            else:
                if ut.VERBOSE:
                    msg_list.append(ut.indent(msg, '    '))
        print('\n'.join(msg_list))
        problems = list(zip(missing, recommended_fixes))
        return problems

    def check_installed(rman):
        import utool as ut
        label = ' %s' % rman.label if rman.label else rman.label
        missing = []
        msg_list = []
        print('Checking if%s modules are installed' % (label,))
        for repo in rman.repos:
            flag, msg = repo.check_installed()
            if not flag:
                msg_list.append('  * !!!%s REPO %s NEEDS TO BE INSTALLED' % (label.upper(), repo,))
                if ut.VERBOSE:
                    msg_list.append(ut.indent(msg, '    '))
                missing.append(repo)
            # else:
            #     print('  * found%s module = %s' % (label, repo,))
        print('\n'.join(msg_list))
        return missing

    def check_cpp_build(rman):
        import utool as ut
        label = ' %s' % rman.label if rman.label else rman.label
        missing = []
        print('Checking if%s modules are built' % (label,))
        for repo in rman.repos:
            flag, msg = repo.check_cpp_build()
            if not flag:
                print('  * !!!%s REPO %s NEEDS TO BE BUILT' % (label.upper(), repo,))
                if ut.VERBOSE:
                    print(ut.indent(msg, '    '))
                missing.append(repo)
        return missing

    def custom_build(rman):
        print('Custom Build')
        for repo in rman.repos:
            script = repo.get_script('build')
            if script is not None:
                script.exec_()

    def custom_install(rman):
        print('Custom Install')
        for repo in rman.repos:
            script = repo.get_script('install')
            if script is not None:
                script.exec_()

    def only_with_pysetup(rman):
        rman2 = RepoManager()
        rman2.code_dir = rman.code_dir
        rman2.permitted_repos = rman.permitted_repos
        rman2.code_dir = rman.code_dir
        rman2.userid = rman.userid
        rman2.label = rman.label
        rman2.repos = [repo for repo in rman.repos if repo.dpath and exists(join(repo.dpath, 'setup.py'))]
        return rman2


@six.add_metaclass(util_class.ReloadingMetaclass)
class Repo(util_dev.NiceRepr):
    """
    Handles a Python module repository
    """
    def __init__(repo, url=None, code_dir=None, dpath=None,
                 modname=None, pythoncmd=None):
        # modname might need to be called egg?
        import utool as ut
        if url is not None and '.git@' in url:
            # parse out specific branch
            repo.default_branch = url.split('@')[-1]
            url = '@'.join(url.split('@')[:-1])
        else:
            repo.default_branch = None
        repo.url = url
        repo._modname = None
        if modname is None:
            modname = []
        repo._modname_hints = ut.ensure_iterable(modname)
        repo.dpath = None
        repo.scripts = {}
        if pythoncmd is None:
            import sys
            pythoncmd = sys.executable
        repo.pythoncmd = pythoncmd

        if dpath is None and repo.url is not None and code_dir is not None:
            dpath = join(code_dir, repo.reponame)
        if dpath is not None:
            repo.dpath = util_path.unixpath(dpath)

    def infer_info(repo):
        if repo.url is None:
            repo.url = list(repo.as_gitpython().remotes[0].urls)[0]

    # --- GIT PYTHON STUFF ---

    @util_decor.memoize
    def as_gitpython(repo):
        """ pip install gitpython """
        import git
        gitrepo = git.Repo(repo.dpath)
        return gitrepo

    @property
    def remotes(repo):
        remotes = repo.as_gitpython().remotes
        remote_dict = {}
        for remote in remotes:
            pass
            remote_info = repo._remote_info(remote)
            if remote_info is not None:
                name = remote_info.pop('name')
                remote_dict[name] = remote_info
        return remote_dict

    def _remote_info(repo, remote):
        OLD = False
        if OLD:
            remote_details = remote.repo.git.remote("get-url", remote.name, '--push')
            # TODO push into gitpython
            urls = [line for line in remote_details.split('\n')]
        else:
            urls = list(remote.urls)

        # urls = list(remote.urls)
        if len(urls) == 0:
            print('[git] WARNING: repo %r has no remote urls' % (repo,))
            remote_info = None
        else:
            if len(urls) > 1:
                print('[git] WARNING: repo %r has multiple urls' % (repo,))
            url = urls[0]
            url = url.replace('github.com:/', 'github.com:')
            remote_info = {}
            url_parts = re.split('[@/:]', url)
            # TODO: parse what format the url is in, ssh/http/https
            idx = util_list.listfind(url_parts, 'github.com')
            remote_info['name'] = remote.name
            remote_info['url'] = url
            if idx is not None:
                username = url_parts[idx + 1]
                remote_info['host'] = 'github'
                remote_info['username'] = username
        return remote_info

    def _ensure_remote_exists(repo, remote_name, remote_url, fmt=None):
        # Remote the remote if it is not in the correct format
        if fmt is None:
            if remote_url.startswith('git@'):
                fmt = 'ssh'
            elif remote_url.startswith('https://'):
                fmt = 'https'
            else:
                raise ValueError('bad format')
        gitrepo = repo.as_gitpython()
        remotes = repo.remotes
        incorrect_version = False
        if remote_url in remotes:
            # Check correct version (SSH or HTTPS)
            wildme_remote_ = remotes[remote_url]
            wildme_url_ = wildme_remote_['url']
            is_ssh = '@' in wildme_url_
            incorrect_version = (is_ssh and fmt == 'https') or (not is_ssh and fmt == 'ssh')
            if incorrect_version:
                print('  * Deleting bad version remote %r: %r' % (remote_name, remote_url))
                gitrepo.delete_remote(remote_name)

        # Ensure there is a remote under the wildme name
        if remote_name not in repo.remotes or incorrect_version:
            print('  * Create remote %r: %r' % (remote_name, remote_url))
            gitrepo.create_remote(remote_name, remote_url)
        return incorrect_version

    def _new_remote_url(repo, host=None, user=None, reponame=None, fmt=None):
        import utool as ut
        if reponame is None:
            reponame = repo.reponame
        if host is None:
            host = 'github.com'
        if fmt is None:
            fmt = 'ssh'
        if host == 'github.com':
            assert user is not None, 'github needs a user'
        url_fmts = {
            'https': ('https://', '/'),
            'ssh':   ('git@', ':'),
        }
        prefix, sep = url_fmts[fmt]
        user_ = '' if user is None else user + '/'
        parts = [prefix, host, sep, user_, reponame, '.git']
        parts = ut.filter_Nones(parts)
        url = ''.join(parts)
        return url

    def reset_branch_to_remote(repo, branch, hard=True):
        """
        does a git reset --hard to whatever remote the branch is assigned to
        """
        remote = repo.get_branch_remote(branch)
        kw = dict(remote=remote, branch=branch)
        if hard:
            kw['flags'] = '--hard'
        repo.issue('git reset {flags} {remote}/{branch}'.format(**kw))

    def get_branch_remote(repo, branch):
        gitrepo = repo.as_gitpython()
        gitbranch = gitrepo.branches[branch]
        remote = gitbranch.tracking_branch().remote_name
        return remote

    def set_branch_remote(repo, branch, remote, remote_branch=None):
        if remote_branch is None:
            remote_branch = branch
        fmt = 'git branch --set-upstream-to={remote}/{remote_branch} {branch} '
        cmd = fmt.format(branch=branch, remote=remote,
                         remote_branch=remote_branch)
        repo.issue(cmd)

    @property
    def active_branch(repo):
        return repo.as_gitpython().active_branch.name

    @property
    def active_remote(repo):
        branch = repo.as_gitpython().active_branch
        tracking_branch = branch.tracking_branch()
        remote_info = None
        for remote in repo.as_gitpython().remotes:
            pass
            if remote.name == tracking_branch.remote_name:
                remote_info = repo._remote_info(remote)
                break
        return remote_info

    @property
    def active_tracking_remote_head(repo):
        branch = repo.as_gitpython().active_branch
        tracking_branch = branch.tracking_branch()
        return tracking_branch.remote_head

    @property
    def active_tracking_branch_name(repo):
        branch = repo.as_gitpython().active_branch
        tracking_branch = branch.tracking_branch()
        return tracking_branch.name

    @property
    def branches(repo):
        gitrepo = repo.as_gitpython()
        return [branch.name for branch in gitrepo.branches]

    # --- </GIT PYTHON STUFF> ---

    @property
    def aliases(repo):
        aliases = []
        if repo._modname is not None:
            aliases.append(repo._modname)
        aliases.extend(repo._modname_hints[:])
        # if repo.dpath and exists(repo.dpath):
        #     reponame = repo._find_modname_from_repo()
        #     if reponame is not None:
        #         aliases.append(reponame)
        aliases.append(repo.reponame)
        aliases.append(repo.reponame.lower())
        import utool as ut
        aliases = ut.unique(aliases)
        return aliases

    def __nice__(repo):
        reponame = repo.reponame
        modname = repo.modname
        # if modname is False:
        #     print(repo.__dict__)
        if modname is None or modname == reponame:
            return '(%s)' % (reponame,)
        else:
            return '(%s, %s)' % (reponame, modname)

    @property
    def modname(repo):
        # import utool as ut
        modname = None
        if repo._modname is not None:
            modname = repo._modname
        elif len(repo._modname_hints) == 1:
            modname = repo._modname_hints[0]
        else:
            modname = repo.aliases[0]
        return modname

    @property
    def reponame(repo):
        if repo.dpath is not None:
            reponame = basename(repo.dpath)
        elif repo.url is not None:
            url_parts = re.split('[/:]', repo.url)
            reponame = url_parts[-1].replace('.git', '')
        elif repo._modname_hints:
            reponame = repo._modname_hints[0]
        else:
            raise Exception('No way to infer (or even guess) repository name!')
        return reponame

    def _find_modname_from_repo(repo):
        import utool as ut
        packages = ut.get_submodules_from_dpath(repo.dpath, only_packages=True,
                                                recursive=False)
        if len(packages) == 1:
            modname = ut.get_modname_from_modpath(packages[0])
            return modname

    def add_script(repo, key, script):
        repo.scripts[key] = script

    def clone(repo, recursive=False):
        print('[git] check repo exists at %s' % (repo.dpath))
        if recursive:
            args = '--recursive'
        else:
            args = ''
        if not exists(repo.dpath):
            _cd(dirname(repo.dpath))
            print('repo.default_branch = %r' % (repo.default_branch,))
            if repo.default_branch is not None:
                args += ' -b {}'.format(repo.default_branch)
            _syscmd('git clone {args} {url}'.format(args=args, url=repo.url))

    def owner(repo):
        url_parts = re.split('[/:]', repo.url)
        owner = url_parts[-2]
        return owner

    def is_owner(repo, userid):
        return userid is not None and repo.owner.lower() == userid.lower()

    def _fix_url(repo, userid, permitted_repos=[]):
        is_owner = repo.is_owner(userid)
        is_contrib = repo.reponame in permitted_repos
        if is_owner or is_contrib:
            repo._ensure_ssh_format()

    def change_url_format(repo, out_type='ssh'):
        """ Changes the url format for committing """
        url = repo.url
        url_parts = re.split('[/:]', url)
        in_type = url_parts[0]
        url_fmts = {
            'https': ('.com/', 'https://'),
            'ssh':   ('.com:', 'git@'),
        }
        url_fmts['git'] = url_fmts['ssh']
        new_repo_url = url
        for old, new in zip(url_fmts[in_type], url_fmts[out_type]):
            new_repo_url = new_repo_url.replace(old, new)
        # Inplace change
        repo.url = new_repo_url
        print('new format repo.url = {!r}'.format(repo.url))

    def check_importable(repo):
        import utool as ut
        # import utool as ut
        found = False
        tried = []
        errors = []
        for modname in repo.aliases:
            tried.append(modname)
            try:
                ut.import_modname(modname)
            except ImportError as ex:  # NOQA
                tried[-1] += ' but got ImportError'
                errors.append(ex)
                pass
            except AttributeError as ex:  # NOQA
                tried[-1] += ' but got AttributeError'
                errors.append(ex)
            else:
                found = True
                errors.append(None)
                tried[-1] += ' and it worked'
                break
        msg = 'tried %s' % (', '.join(tried))
        return found, msg, errors

    def is_cloned(repo):
        from os.path import exists
        if not exists(repo.dpath):
            return False

    def check_installed(repo):
        import utool as ut
        found = None
        tried = []
        for modname in repo.aliases:
            tried.append(modname)
            found = ut.check_module_installed(modname)
            if found:
                break
        msg = 'tried %s' % (', '.join(tried))
        return found, msg

    def check_cpp_build(repo):
        import utool as ut
        script = repo.get_script('build')
        if script.is_fpath_valid():
            if repo.modname == 'pyflann':
                return True, 'cant detect flann cpp'
            # hack, this doesnt quite do it
            pat = '*' + ut.util_cplat.get_pylib_ext()
            dynlibs = ut.glob(repo.dpath + '/' + repo.modname, pat, recursive=True)
            msg = 'Could not find any dynamic libraries'
            flag = len(dynlibs) > 0
        else:
            flag = True
            msg = 'passed, but didnt expect anything'
        return flag, msg

    def get_script(repo, type_):
        class Script(object):
            def __init__(script):
                script.type_ = type_
                script.text = None
                script.fpath = None
                script.cmake = None

            def is_fpath_valid(script):
                return script.fpath is not None and exists(script.fpath)

            def is_valid(script):
                return script.text or script.is_fpath_valid()

            def exec_(script):
                import utool as ut
                print('+**** exec %s script *******' % (script.type_))
                print('repo = %r' % (repo,))
                with ut.ChdirContext(repo.dpath):
                    if script.is_fpath_valid():
                        normbuild_flag = '--no-rmbuild'
                        if ut.get_argflag(normbuild_flag):
                            ut.cmd(script.fpath + ' ' + normbuild_flag)
                        else:
                            ut.cmd(script.fpath)
                    else:
                        if script.text is not None:
                            print('ABOUT TO EXECUTE')
                            ut.print_code(script.text, 'bash')
                            if ut.are_you_sure('execute above script?'):
                                from os.path import join
                                scriptdir = ut.ensure_app_resource_dir('utool',
                                                                       'build_scripts')
                                script_path = join(scriptdir,
                                                   'script_' + script.type_ + '_' +
                                                   ut.hashstr27(script.text) + '.sh')
                                ut.writeto(script_path, script.text)
                                _ = ut.cmd('bash ', script_path)  # NOQA
                        else:
                            print("CANT QUITE EXECUTE THIS YET")
                            ut.print_code(script.text, 'bash')
                #os.system(scriptname)
                print('L**** exec %s script *******' % (script.type_))

        script = Script()
        script.text = repo.scripts.get(type_, None)

        if script.text is None and type_ == 'build' and repo.dpath:
            if sys.platform.startswith('win32'):
                # vtool --rebuild-sver didnt work with this line
                #scriptname = './mingw_build.bat'
                fpath = join(repo.dpath, 'mingw_build.bat')
            else:
                fpath = join(repo.dpath, 'unix_build.sh')
            if exists(fpath):
                script.fpath = fpath

            cmake = join(repo.dpath, 'CMakeLists.txt')
            if exists(cmake):
                script.cmake = cmake

        return script

    def has_script(repo, type_):
        return repo.get_script(type_).is_valid()

    def custom_build(repo):
        script = repo.get_script('build')
        if script is not None:
            script.exec_()

    def custom_install(repo):
        script = repo.get_script('install')
        if script is not None:
            script.exec_()
        # TODO:
        # import utool as ut
        # ut.print_code(repo.install_script, 'bash')

    def issue(repo, command, sudo=False, dry=False, error='raise', return_out=False):
        """
        issues a command on a repo

        CommandLine:
            python -m utool.util_git --exec-repocmd

        Example:
            >>> # DISABLE_DOCTEST
            >>> from utool.util_git import *  # NOQA
            >>> import utool as ut
            >>> repo = dirname(ut.get_modpath(ut, prefer_pkg=True))
            >>> command = 'git status'
            >>> sudo = False
            >>> result = repocmd(repo, command, sudo)
            >>> print(result)
        """
        import utool as ut
        if ut.WIN32:
            assert not sudo, 'cant sudo on windows'
        if command == 'short_status':
            return repo.short_status()
        command_list = ut.ensure_iterable(command)
        cmdstr = '\n        '.join([cmd_ for cmd_ in command_list])
        if not dry:
            print('+--- *** repocmd(%s) *** ' % (cmdstr,))
            print('repo=%s' % ut.color_text(repo.dpath, 'yellow'))
        verbose = True
        with repo.chdir_context():
            ret = None
            for count, cmd in enumerate(command_list):
                if dry:
                    print(cmd)
                    continue
                if not sudo or ut.WIN32:
                    # ret = os.system(cmd)
                    cmdinfo = ut.cmd2(cmd, verbout=True)
                    out, err, ret = ut.take(cmdinfo, ['out', 'err', 'ret'])
                else:
                    # cmdinfo = ut.cmd2('sudo ' + cmd, verbose=1)
                    out, err, ret = ut.cmd(cmd, sudo=True)
                if verbose > 1:
                    print('ret(%d) = %r' % (count, ret,))
                if ret != 0:
                    if error == 'raise':
                        raise Exception('Failed command %r' % (cmd,))
                    elif error == 'return':
                        return out
                    else:
                        raise ValueError('unknown flag error=%r' % (error,))
                if return_out:
                    return out
        if not dry:
            print('L____')

    def chdir_context(repo, verbose=False):
        import utool as ut
        return ut.ChdirContext(repo.dpath, verbose=verbose)

    def pull2(repo, overwrite=True):
        """
        Pulls and automatically overwrites conflict files.
        """
        cmd = 'git pull --no-edit'
        out = repo.issue(cmd, error='return')
        if overwrite and out is not None:
            repo._handle_overwrite_error(out)
            # Retry
            repo.issue(cmd)

    def checkout2(repo, branch, overwrite=True):
        """
        Checkout `branch` and automatically overwrites conflict files.
        """
        cmd = 'git checkout %s' % (branch,)
        out = repo.issue(cmd, error='return')
        if overwrite and out is not None:
            repo._handle_overwrite_error(out)
            repo._handle_abort_merge_rebase(out)
            # Retry
            repo.issue(cmd)

    def _parse_merge_conflict_fpaths(repo, out):
        fpaths = []
        if out is not None:
            for line in out.split('\n'):
                pref = 'CONFLICT (content): Merge conflict in '
                if line.startswith(pref):
                    fpaths.append(join(repo.dpath, line[len(pref):]))
        return fpaths

    def _handle_abort_merge_rebase(repo, out):
        if out.startswith('error: you need to resolve your current index first'):
            try:
                repo.issue('git merge --abort')
            except Exception:
                pass
            try:
                repo.issue('git rebase --abort')
            except Exception:
                pass

    def _handle_overwrite_error(repo, out):
        import utool as ut
        # parse stdout to handle the error
        if out.startswith('error: The following untracked working tree files would be overwritten'):
            print('[ut.git] handling overwrite error')
            lines = out.split('\n')[1:]
            fpaths = []
            for line in lines:
                if line.startswith('Please move or remove them before you can merge'):
                    break
                fpaths.append(join(repo.dpath, line.strip()))
                ut.remove_file_list(fpaths)

    def short_status(repo):
        r"""
        CommandLine:
            python -m utool.util_git short_status

        Example:
            >>> # DISABLE_DOCTEST
            >>> from utool.util_git import *  # NOQA
            >>> import utool as ut
            >>> repo = Repo(dpath=ut.truepath('.'))
            >>> result = repo.short_status()
            >>> print(result)
        """
        import utool as ut
        prefix = repo.dpath
        with ut.ChdirContext(repo.dpath, verbose=False):
            out, err, ret = ut.cmd('git status', verbose=False, quiet=True)
            # parse git status
            is_clean_msg1 = 'Your branch is up-to-date with'
            is_clean_msgs = [
                'nothing to commit, working directory clean',
                'nothing to commit, working tree clean',
            ]
            msg2 = 'nothing added to commit but untracked files present'

            needs_commit_msgs = [
                'Changes to be committed',
                'Changes not staged for commit',
                'Your branch is ahead of',
            ]

            suffix = ''
            if is_clean_msg1 in out and any(msg in out for msg in is_clean_msgs):
                suffix += ut.color_text('is clean', 'blue')
            if msg2 in out:
                suffix += ut.color_text('has untracked files', 'yellow')
            if any(msg in out for msg in needs_commit_msgs):
                suffix += ut.color_text('has changes', 'red')
        print(prefix + ' ' + suffix)

    def python_develop(repo):
        import utool as ut
        repo.issue('{pythoncmd} -m pip install -e {dpath}'.format(
            pythoncmd=repo.pythoncmd, dpath=repo.dpath),
            sudo=not ut.in_virtual_env())

    def is_gitrepo(repo):
        gitdir = join(repo.dpath, '.git')
        return exists(gitdir) and isdir(gitdir)

    def pull(repo, has_submods=False):
        print('Pulling: ' + repo.dpath)
        _cd(repo.dpath)
        assert repo.is_gitrepo(), 'cannot pull a nongit repo'
        _syscmd('git pull')
        if has_submods:
            _syscmd('git submodule init')
            _syscmd('git submodule update')

    def rename_branch(repo, old_branch_name, new_branch_name, remote='origin'):
        r"""
        References:
            http://stackoverflow.com/questions/1526794/rename?answertab=votes#tab-top
            http://stackoverflow.com/questions/9524933/renaming-a-branch-in-github

        CommandLine:
            python -m utool.util_git --test-rename_branch --old=mymaster --new=ibeis_master

        Example:
            >>> # DISABLE_DOCTEST
            >>> # SCRIPT
            >>> from utool.util_git import *  # NOQA
            >>> repo = ut.get_argval('--repo', str, '.')
            >>> remote = ut.get_argval('--remote', str, 'origin')
            >>> old_branch_name = ut.get_argval('--old', str, None)
            >>> new_branch_name = ut.get_argval('--new', str, None)
            >>> rename_branch(old_branch_name, new_branch_name, repo, remote)
        """
        assert repo.is_gitrepo(), 'cannot pull a nongit repo'
        fmtdict = dict(remote=remote,
                       old_branch_name=old_branch_name,
                       new_branch_name=new_branch_name)
        command_list = [
            'git checkout {old_branch_name}'.format(**fmtdict),
            # rename branch
            'git branch -m {old_branch_name} {new_branch_name}'.format(**fmtdict),
            # delete old branch
            'git push {remote} :{old_branch_name}'.format(**fmtdict),
            # push new branch
            'git push {remote} {new_branch_name}'.format(**fmtdict),
        ]
        repo.issue(command_list)

    @staticmethod
    def resolve_conflicts(fpath, strat, force=False, verbose=True):
        """
        Parses merge conflits and takes either version
        """
        import utool as ut
        import re
        top_pat = re.escape('<' * 7)
        mid_pat = re.escape('=' * 7)
        bot_pat = re.escape('>' * 7)
        flags = re.MULTILINE | re.DOTALL
        # Pattern to remove the top part
        theirs_pat1 = re.compile('^%s.*?%s.*?$\n' % (top_pat, mid_pat), flags=flags)
        theirs_pat2 = re.compile('^%s.*?$\n' % (bot_pat), flags=flags)
        # Pattern to remove the bottom part
        ours_pat1   = re.compile('^%s.*?%s.*?$\n' % (mid_pat, bot_pat), flags=flags)
        ours_pat2   = re.compile('^%s.*?$\n' % (top_pat), flags=flags)
        strat_pats = {
            'theirs': [theirs_pat1, theirs_pat2],
            'ours': [ours_pat1, ours_pat2],
        }

        text_in = ut.readfrom(fpath)
        text_out = text_in
        strat = 'ours'
        strat = 'theirs'
        for pat in strat_pats[strat]:
            text_out = pat.sub('', text_out)
        if verbose:
            ut.print_difftext(ut.difftext(text_in, text_out, num_context_lines=3))

        if force:
            ut.writeto(fpath, text_out)


def git_sequence_editor_squash(fpath):
    r"""
    squashes wip messages

    CommandLine:
        python -m utool.util_git --exec-git_sequence_editor_squash

    Example:
        >>> # DISABLE_DOCTEST
        >>> # SCRIPT
        >>> import utool as ut
        >>> from utool.util_git import *  # NOQA
        >>> fpath = ut.get_argval('--fpath', str, default=None)
        >>> git_sequence_editor_squash(fpath)

    Ignore:
        text = ut.codeblock(
            '''
            pick 852aa05 better doctest for tips
            pick 3c779b8 wip
            pick 02bc21d wip
            pick 1853828 Fixed root tablename
            pick 9d50233 doctest updates
            pick 66230a5 wip
            pick c612e98 wip
            pick b298598 Fixed tablename error
            pick 1120a87 wip
            pick f6c4838 wip
            pick 7f92575 wip
            ''')

    Ignore:
        def squash_consecutive_commits_with_same_message():
            # http://stackoverflow.com/questions/8226278/git-alias-to-squash-all-commits-with-a-particular-commit-message
            # Can do interactively with this. Can it be done automatically and pay attention to
            # Timestamps etc?
            git rebase --interactive HEAD~40 --autosquash
            git rebase --interactive $(git merge-base HEAD master) --autosquash

            # Lookbehind correct version
            %s/\([a-z]* [a-z0-9]* wip\n\)\@<=pick \([a-z0-9]*\) wip/squash \2 wip/gc

           # THE FULL NON-INTERACTIVE AUTOSQUASH SCRIPT
           # TODO: Dont squash if there is a one hour timedelta between commits

           GIT_EDITOR="cat $1" GIT_SEQUENCE_EDITOR="python -m utool.util_git --exec-git_sequence_editor_squash \
                   --fpath $1" git rebase -i $(git rev-list HEAD | tail -n 1) --autosquash --no-verify
           GIT_EDITOR="cat $1" GIT_SEQUENCE_EDITOR="python -m utool.util_git --exec-git_sequence_editor_squash \
                   --fpath $1" git rebase -i HEAD~10 --autosquash --no-verify

           GIT_EDITOR="cat $1" GIT_SEQUENCE_EDITOR="python -m utool.util_git --exec-git_sequence_editor_squash \
                   --fpath $1" git rebase -i $(git merge-base HEAD master) --autosquash --no-verify

           # 14d778fa30a93f85c61f34d09eddb6d2cafd11e2
           # c509a95d4468ebb61097bd9f4d302367424772a3
           # b0ffc26011e33378ee30730c5e0ef1994bfe1a90
           # GIT_SEQUENCE_EDITOR=<script> git rebase -i <params>
           # GIT_SEQUENCE_EDITOR="echo 'FOOBAR $1' " git rebase -i HEAD~40 --autosquash
           # git checkout master
           # git branch -D tmp
           # git checkout -b tmp
           # option to get the tail commit
           $(git rev-list HEAD | tail -n 1)
           # GIT_SEQUENCE_EDITOR="python -m utool.util_git --exec-git_sequence_editor_squash \
                   --fpath $1" git rebase -i HEAD~40 --autosquash
           # GIT_SEQUENCE_EDITOR="python -m utool.util_git --exec-git_sequence_editor_squash \
                   --fpath $1" git rebase -i HEAD~40 --autosquash --no-verify
           <params>
    """
    # print(sys.argv)
    import utool as ut
    text = ut.read_from(fpath)
    # print('fpath = %r' % (fpath,))
    print(text)
    # Doesnt work because of fixed witdth requirement
    # search = (ut.util_regex.positive_lookbehind('[a-z]* [a-z0-9]* wip\n') + 'pick ' +
    #           ut.reponamed_field('hash', '[a-z0-9]*') + ' wip')
    # repl = ('squash ' + ut.bref_field('hash') + ' wip')
    # import re
    # new_text = re.sub(search, repl, text, flags=re.MULTILINE)
    # print(new_text)
    prev_msg = None
    prev_dt = None
    new_lines = []

    def get_commit_date(hashid):
        out, err, ret = ut.cmd('git show -s --format=%ci ' + hashid, verbose=False, quiet=True, pad_stdout=False)
        # from datetime import datetime
        from dateutil import parser
        # print('out = %r' % (out,))
        stamp = out.strip('\n')
        # print('stamp = %r' % (stamp,))
        dt = parser.parse(stamp)
        # dt = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S %Z')
        # print('dt = %r' % (dt,))
        return dt

    for line in text.split('\n'):
        commit_line = line.split(' ')
        if len(commit_line) < 3:
            prev_msg = None
            prev_dt = None
            new_lines += [line]
            continue
        action = commit_line[0]
        hashid = commit_line[1]
        msg = ' ' .join(commit_line[2:])
        try:
            dt = get_commit_date(hashid)
        except ValueError:
            prev_msg = None
            prev_dt = None
            new_lines += [line]
            continue
        orig_msg = msg
        can_squash = action == 'pick' and msg == 'wip' and prev_msg == 'wip'
        if prev_dt is not None and prev_msg == 'wip':
            tdelta = dt - prev_dt
            # Only squash closely consecutive commits
            threshold_minutes = 45
            td_min = (tdelta.total_seconds() / 60.)
            # print(tdelta)
            can_squash &= td_min < threshold_minutes
            msg = msg + ' -- tdelta=%r' % (ut.get_timedelta_str(tdelta),)
        if can_squash:
            new_line = ' ' .join(['squash', hashid, msg])
            new_lines += [new_line]
        else:
            new_lines += [line]
        prev_msg = orig_msg
        prev_dt = dt
    new_text = '\n'.join(new_lines)

    def get_commit_date(hashid):
        out = ut.cmd('git show -s --format=%ci ' + hashid, verbose=False)
        print('out = %r' % (out,))

    # print('Dry run')
    # ut.dump_autogen_code(fpath, new_text)
    print(new_text)
    ut.write_to(fpath, new_text, n=None)


def std_build_command(repo='.'):
    """
    DEPRICATE
    My standard build script names.

    Calls mingw_build.bat on windows and unix_build.sh  on unix
    """
    import utool as ut
    print('+**** stdbuild *******')
    print('repo = %r' % (repo,))
    if sys.platform.startswith('win32'):
        # vtool --rebuild-sver didnt work with this line
        #scriptname = './mingw_build.bat'
        scriptname = 'mingw_build.bat'
    else:
        scriptname = './unix_build.sh'
    if repo == '':
        # default to cwd
        repo = '.'
    else:
        os.chdir(repo)
    ut.assert_exists(scriptname)
    normbuild_flag = '--no-rmbuild'
    if ut.get_argflag(normbuild_flag):
        scriptname += ' ' + normbuild_flag
    # Execute build
    ut.cmd(scriptname)
    #os.system(scriptname)
    print('L**** stdbuild *******')


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m utool.util_git
        python -m utool.util_git --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
