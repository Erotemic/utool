# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
from os.path import split, exists, join, dirname
from utool import util_inject
from utool._internal import meta_util_arg
print, rrr, profile = util_inject.inject2(__name__)


def in_virtual_env():
    """
    returns True if you are running inside a python virtual environment.
    (DOES NOT WORK IF IN IPYTHON AND USING A VIRTUALENV)

    sys.prefix gives the location of the virtualenv

    Notes:
        It seems IPython does not respect virtual environments properly.
        TODO: find a solution
        http://stackoverflow.com/questions/7335992/ipython-and-virtualenv-ignoring-site-packages

    References:
        http://stackoverflow.com/questions/1871549/python-determine-if-running-inside-virtualenv

    CommandLine:
        python -m utool.util_sysreq in_virtual_env

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_sysreq import *  # NOQA
        >>> import utool as ut
        >>> result = in_virtual_env()
        >>> print(result)
    """
    import sys
    has_venv = False
    if hasattr(sys, 'real_prefix'):
        # For virtualenv module
        has_venv = True
    elif hasattr(sys, 'base_prefix'):
        # For venv module
        has_venv = sys.base_prefix != sys.prefix
    return has_venv


def get_site_packages_dir():
    """
    CommandLine:
        python -m utool.util_sysreq get_site_packages_dir

    Notes:
        It seems IPython does not respect virtual environments properly.
        TODO: find a solution
        http://stackoverflow.com/questions/7335992/ipython-and-virtualenv-ignoring-site-packages
    """
    import distutils.sysconfig
    return distutils.sysconfig.get_python_lib()


def get_global_dist_packages_dir():
    """
    Attempts to work around virtualenvs and find the system dist_pacakges.
    Essentially this is implmenented as a lookuptable
    """
    import utool as ut
    if not ut.in_virtual_env():
        # Non venv case
        return get_site_packages_dir()
    else:
        candidates = []
        if ut.LINUX:
            import sys
            candidates += [
                '/usr/lib/python%s/dist-packages' % (sys.version[0:3],),
                '/usr/lib/python%s/dist-packages' % (sys.version[0:1],),
            ]
        else:
            raise NotImplementedError()
        for path in candidates:
            if ut.checkpath(path):
                return path


def get_local_dist_packages_dir():
    """
    Attempts to work around virtualenvs and find the system dist_pacakges.
    Essentially this is implmenented as a lookuptable
    """
    import utool as ut
    if not ut.in_virtual_env():
        # Non venv case
        return get_site_packages_dir()
    else:
        candidates = []
        if ut.LINUX:
            candidates += [
                '/usr/local/lib/python2.7/dist-packages',
            ]
        else:
            raise NotImplementedError()
        for path in candidates:
            if ut.checkpath(path):
                return path


def is_running_as_root():
    """
    References:
        http://stackoverflow.com/questions/5721529/running-python-script-as-root-with-sudo-what-is-the-username-of-the-effectiv
        http://stackoverflow.com/questions/2806897/what-is-the-best-practices-for-checking-if-the-user-of-a-python-script-has-root
    """
    return os.getenv('USER') == 'root'


def locate_path(dname, recurse_down=True):
    """ Search for a path """
    tried_fpaths = []
    root_dir = os.getcwd()
    while root_dir is not None:
        dpath = join(root_dir, dname)
        if exists(dpath):
            return dpath
        else:
            tried_fpaths.append(dpath)
        _new_root = dirname(root_dir)
        if _new_root == root_dir:
            root_dir = None
            break
        else:
            root_dir = _new_root
        if not recurse_down:
            break
    msg = 'Cannot locate dname=%r' % (dname,)
    msg = ('\n[sysreq!] Checked: '.join(tried_fpaths))
    print(msg)
    raise ImportError(msg)


def ensure_in_pythonpath(dname):
    dname_list = [split(dpath)[1] for dpath in sys.path]
    if dname not in dname_list:
        dpath = locate_path(dname)
        if meta_util_arg.VERBOSE:
            print('[sysreq] appending %r to PYTHONPATH' % dpath)
        sys.path.append(dpath)
    elif meta_util_arg.DEBUG:
        print('[sysreq] PYTHONPATH has %r' % dname)


def total_purge_developed_repo(repodir):
    r"""
    Outputs commands to help purge a repo

    Args:
        repodir (str): path to developed repository

    CommandLine:
        python -m utool.util_sysreq total_purge_installed_repo --show

    Ignore:
        repodir = ut.truepath('~/code/Lasagne')

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_sysreq import *  # NOQA
        >>> import utool as ut
        >>> repodir = ut.get_argval('--repodir', default=None)
        >>> result = total_purge_installed_repo(repodir)
    """
    assert repodir is not None
    import utool as ut
    import os
    repo = ut.util_git.Repo(dpath=repodir)

    user = os.environ['USER']

    fmtdict = dict(
        user=user,
        modname=repo.modname,
        reponame=repo.reponame,
        dpath=repo.dpath,
        global_site_pkgs=ut.get_global_dist_packages_dir(),
        local_site_pkgs=ut.get_local_dist_packages_dir(),
        venv_site_pkgs=ut.get_site_packages_dir(),
    )

    commands = [_.format(**fmtdict) for _ in [
        'pip uninstall {modname}',
        'sudo -H pip uninstall {modname}',
        'sudo pip uninstall {modname}',
        'easy_install -m {modname}',
        'cd {dpath} && python setup.py develop --uninstall',
        # If they still exist try chowning to current user
        'sudo chown -R {user}:{user} {dpath}',
    ]]
    print('Normal uninstall commands')
    print('\n'.join(commands))

    possible_link_paths = [_.format(**fmtdict) for _ in [
        '{dpath}/{modname}.egg-info',
        '{dpath}/build',
        '{venv_site_pkgs}/{reponame}.egg-info',
        '{local_site_pkgs}/{reponame}.egg-info',
        '{venv_site_pkgs}/{reponame}.egg-info',
    ]]
    from os.path import exists, basename
    existing_link_paths = [path for path in possible_link_paths]
    print('# Delete paths and eggs')
    for path in existing_link_paths:
        if exists(path):
            if ut.get_file_info(path)['owner'] != user:
                print('sudo /bin/rm -rf {path}'.format(path=path))
            else:
                print('/bin/rm -rf {path}'.format(path=path))
        #ut.delete(path)

    print('# Make sure nothing is in the easy install paths')
    easyinstall_paths = [_.format(**fmtdict) for _ in [
        '{venv_site_pkgs}/easy-install.pth',
        '{local_site_pkgs}/easy-install.pth',
        '{venv_site_pkgs}/easy-install.pth',
    ]]
    for path in easyinstall_paths:
        if exists(path):
            easy_install_list = ut.readfrom(path, verbose=False).strip().split('\n')
            easy_install_list_ = [basename(p) for p in easy_install_list]
            index1 = ut.listfind(easy_install_list_, repo.reponame)
            index2 = ut.listfind(easy_install_list_, repo.modname)
            if index1 is not None or index2 is not None:
                print('Found at index1=%r, index=%r' % (index1, index2))
                if ut.get_file_info(path)['owner'] != user:
                    print('sudo gvim {path}'.format(path=path))
                else:
                    print('gvim {path}'.format(path=path))

    checkcmds = [_.format(**fmtdict) for _ in [
        'python -c "import {modname}; print({modname}.__file__)"'
    ]]
    import sys
    assert repo.modname not in sys.modules
    print("# CHECK STATUS")
    for cmd in checkcmds:
        print(cmd)
        #ut.cmd(cmd, verbose=False)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m utool.util_sysreq
        python -m utool.util_sysreq --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
