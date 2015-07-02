#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import sys
import os
from six.moves import zip
from os.path import exists, join, dirname, split, isdir
from utool._internal import meta_util_git as mu  # NOQA
from utool._internal.meta_util_git import get_repo_dirs, get_repo_dname  # NOQA
from utool import util_inject
from utool import util_arg
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[git]')

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
    print("+****gitcmd*******")
    print('repo=%s' % repo)
    if not isinstance(command, (tuple, list)):
        command_list = [command]
    else:
        command_list = command
    print('command=%s' % '\n        '.join([cmd_ for cmd_ in command_list]))
    os.chdir(repo)
    #if command.find('git') != 0:
    #    command = 'git ' + command
    ret = None
    if not dryrun:
        for count, cmd in enumerate(command_list):
            assert cmd.startswith('git '), 'invalid git command'
            if not sudo or sys.platform.startswith('win32'):
                ret = os.system(cmd)
            else:
                ret = os.system('sudo ' + cmd)
            verbose = True
            if verbose:
                print('ret(%d) = %r' % (count, ret,))
            if ret != 0:
                raise Exception('Failed command %r' % (cmd,))
    print("L***********")


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
        #scriptname = './mingw_build.bat'  # vtool --rebuild-sver didnt work with this line
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


def gg_command(command, sudo=False):
    """ Runs a command on all of your PROJECT_REPO_DIRS """
    print('+------- GG_COMMAND -------')
    print('| sudo=%s' % sudo)
    print('| command=%s' % command)
    for repo in PROJECT_REPO_DIRS:
        if exists(repo):
            gitcmd(repo, command, sudo=sudo)
    print('L___ FINISHED GG_COMMAND ___')


def checkout_repos(repo_urls, repo_dirs=None, checkout_dir=None):
    """ Checkout every repo in repo_urls into checkout_dir """
    # Check out any repo you dont have
    if checkout_dir is not None:
        repo_dirs = mu.get_repo_dirs(checkout_dir)
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
        repo_dirs = mu.get_repo_dirs(checkout_dir)
    assert repo_dirs is not None, 'specify checkout dir or repo_dirs'
    for repodir, repourl in zip(repo_dirs, repo_urls):
        print('[git] checkexist: ' + repodir)
        if not exists(repodir):
            mu.cd(dirname(repodir))
            mu.cmd('git clone ' + repourl)


def setup_develop_repos(repo_dirs):
    """ Run python installs """
    for repodir in repo_dirs:
        print('Installing: ' + repodir)
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
