#!/usr/bin/env python2.7
"""
TODO: export from utool
"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
import os
from six.moves import zip
from os.path import exists, join, dirname, split, isdir
from utool._internal import meta_util_git as mu  # NOQA
from utool._internal.meta_util_git import get_repo_dirs, get_repo_dname  # NOQA
from utool import util_inject
from utool import util_arg
print, rrr, profile = util_inject.inject2(__name__, '[git]')

repo_list = mu.repo_list
set_userid = mu.set_userid

#try:
#    import __REPOS__
#    PROJECT_REPO_DIRS = __REPOS__.PROJECT_REPO_DIRS
#    PROJECT_REPO_URLS = __REPOS__.PROJECT_REPO_URLS
#except ImportError:
PROJECT_REPO_DIRS = []
PROJECT_REPO_URLS = []
CODE_DIR = None
DRY_RUN = util_arg.get_argflag('--dryrun')


def set_code_dir(code_dir):
    global CODE_DIR
    CODE_DIR = code_dir


def set_project_repos(repo_urls, repo_dirs):
    global PROJECT_REPO_DIRS
    global PROJECT_REPO_URLS
    PROJECT_REPO_URLS = repo_urls
    PROJECT_REPO_DIRS = repo_dirs


def get_project_repo_dirs():
    global PROJECT_REPO_DIRS
    return PROJECT_REPO_DIRS


def gitcmd(repo, command, sudo=False, dryrun=DRY_RUN):
    """
    TODO change name to repo command
    runs a command on a repo
    """
    import utool as ut
    if ut.WIN32:
        assert not sudo, 'cant sudo on windows'
    if not isinstance(command, (tuple, list)):
        command_list = [command]
    else:
        command_list = command
    cmdstr = '\n        '.join([cmd_ for cmd_ in command_list])
    print('+**** repocmd(%s) *******' % (cmdstr,))
    print('repo=%s' % ut.color_text(repo, 'yellow'))
    os.chdir(repo)
    #if command.find('git') != 0:
    #    command = 'git ' + command
    ret = None
    if not dryrun:
        for count, cmd in enumerate(command_list):
            #assert cmd.startswith('git '), 'invalid git command'
            if not sudo or sys.platform.startswith('win32'):
                ret = os.system(cmd)
            else:
                ret = os.system('sudo ' + cmd)
            verbose = True
            if verbose > 1:
                print('ret(%d) = %r' % (count, ret,))
            if ret != 0:
                raise Exception('Failed command %r' % (cmd,))
    print("L***********")


"""
def squash_consecutive_commits_with_same_message():
    http://stackoverflow.com/questions/8226278/git-alias-to-squash-all-commits-with-a-particular-commit-message

    # Can do interactively with this. Can it be done automatically and pay attention to
    # Timestamps etc?
    git rebase --interactive HEAD~40 --autosquash
    git rebase --interactive $(git merge-base HEAD master) --autosquash

    # Lookbehind correct version
    %s/\([a-z]* [a-z0-9]* wip\n\)\@<=pick \([a-z0-9]*\) wip/squash \2 wip/gc

   # THE FULL NON-INTERACTIVE AUTOSQUASH SCRIPT
   # TODO: Dont squash if there is a one hour timedelta between commits

   GIT_EDITOR="cat $1" GIT_SEQUENCE_EDITOR="python -m utool.util_git --exec-git_sequence_editor_squash --fpath $1" git rebase -i $(git rev-list HEAD | tail -n 1) --autosquash --no-verify
   GIT_EDITOR="cat $1" GIT_SEQUENCE_EDITOR="python -m utool.util_git --exec-git_sequence_editor_squash --fpath $1" git rebase -i HEAD~10 --autosquash --no-verify

   GIT_EDITOR="cat $1" GIT_SEQUENCE_EDITOR="python -m utool.util_git --exec-git_sequence_editor_squash --fpath $1" git rebase -i $(git merge-base HEAD master) --autosquash --no-verify



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
   # GIT_SEQUENCE_EDITOR="python -m utool.util_git --exec-git_sequence_editor_squash --fpath $1" git rebase -i HEAD~40 --autosquash
   # GIT_SEQUENCE_EDITOR="python -m utool.util_git --exec-git_sequence_editor_squash --fpath $1" git rebase -i HEAD~40 --autosquash --no-verify
   <params>
"""


def git_sequence_editor_squash(fpath):
    """
    squashes wip messages

    CommandLine:
        python -m utool.util_git --exec-git_sequence_editor_squash

    Example:
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
    """
    # print(sys.argv)
    import utool as ut
    text = ut.read_from(fpath)
    # print('fpath = %r' % (fpath,))
    print(text)

    # Doesnt work because of fixed witdth requirement
    # search = (ut.util_regex.positive_lookbehind('[a-z]* [a-z0-9]* wip\n') + 'pick ' +
    #           ut.named_field('hash', '[a-z0-9]*') + ' wip')
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
        # dt = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S %Z")
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
    print(new_text)
    ut.write_to(fpath, new_text, n=None)
    # ut.dump_autogen_code(fpath, new_text)


def rename_branch(old_branch_name, new_branch_name, repo='.', remote='origin', dryrun=DRY_RUN):
    r"""
    References:
        http://stackoverflow.com/questions/1526794/rename-master-branch-for-both-local-and-remote-git-repositories?answertab=votes#tab-top
        http://stackoverflow.com/questions/9524933/renaming-a-branch-in-github

    CommandLine:
        python -m utool.util_git --test-rename_branch --old=mymaster --new=ibeis_master --dryrun

    Example:
        >>> # SCRIPT
        >>> from utool.util_git import *  # NOQA
        >>> repo = ut.get_argval('--repo', str, '.')
        >>> remote = ut.get_argval('--remote', str, 'origin')
        >>> old_branch_name = ut.get_argval('--old', str, None)
        >>> new_branch_name = ut.get_argval('--new', str, None)
        >>> rename_branch(old_branch_name, new_branch_name, repo, remote)
    """
    import utool as ut
    repo = ut.truepath(repo)
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
    gitcmd(repo, command_list, dryrun=dryrun)


def std_build_command(repo='.'):
    """
    My standard build script names.

    Calls mingw_build.bat on windows and unix_build.sh  on unix
    """
    import utool as ut
    print("+**** stdbuild *******")
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
    print("L**** stdbuild *******")


def gg_command(command, sudo=False, repo_dirs=None):
    """ Runs a command on all of your PROJECT_REPO_DIRS """
    if repo_dirs is None:
        repo_dirs = PROJECT_REPO_DIRS
    print('+------- GG_COMMAND -------')
    print('| sudo=%s' % sudo)
    print('| command=%s' % command)
    for repo in repo_dirs:
        if exists(repo):
            gitcmd(repo, command, sudo=sudo)
    print('L___ FINISHED GG_COMMAND ___')


def checkout_repos(repo_urls, repo_dirs=None, checkout_dir=None):
    """ Checkout every repo in repo_urls into checkout_dir """
    # Check out any repo you dont have
    if checkout_dir is not None:
        repo_dirs = mu.get_repo_dirs(repo_urls, checkout_dir)
    assert repo_dirs is not None, 'specify checkout dir or repo_dirs'
    for repodir, repourl in zip(repo_dirs, repo_urls):
        print('[git] checkexist: ' + repodir)
        if not exists(repodir):
            mu.cd(dirname(repodir))
            mu.cmd('git clone ' + repourl)


def ensure_project_repos():
    ensure_repos(PROJECT_REPO_URLS, PROJECT_REPO_DIRS)


def ensure_repos(repo_urls, repo_dirs=None, checkout_dir=None):
    """ Checkout every repo in repo_urls into checkout_dir """
    # Check out any repo you dont have
    if checkout_dir is not None:
        repo_dirs = mu.get_repo_dirs(repo_urls, checkout_dir)
    assert repo_dirs is not None, 'specify checkout dir or repo_dirs'
    for repodir, repourl in zip(repo_dirs, repo_urls):
        print('[git] checkexist: ' + repodir)
        if not exists(repodir):
            mu.cd(dirname(repodir))
            mu.cmd('git clone ' + repourl)
    return repo_dirs


def setup_develop_repos(repo_dirs):
    """ Run python installs """
    for repodir in repo_dirs:
        print('\n[git] Setup Develop: ' + repodir)
        mu.cd(repodir)
        assert exists('setup.py'), 'cannot setup a nonpython repo'
        mu.cmd('python setup.py develop')


def pull_repos(repo_dirs, repos_with_submodules=[]):
    for repodir in repo_dirs:
        print('Pulling: ' + repodir)
        mu.cd(repodir)
        assert exists('.git'), 'cannot pull a nongit repo'
        mu.cmd('git pull')
        reponame = split(repodir)[1]
        if reponame in repos_with_submodules or\
           repodir in repos_with_submodules:
            repos_with_submodules
            mu.cmd('git submodule init')
            mu.cmd('git submodule update')


def is_gitrepo(repo_dir):
    gitdir = join(repo_dir, '.git')
    return exists(gitdir) and isdir(gitdir)


if __name__ == '__main__':
    import multiprocessing
    import utool as ut  # NOQA
    multiprocessing.freeze_support()  # for win32
    if ut.doctest_was_requested():
        ut.doctest_funcs()
    else:
        command = ' '.join(sys.argv[1:])
        # Apply command to all repos
        gg_command(command)
