from __future__ import absolute_import, division, print_function
import sys
import os
from os.path import split, exists, join, dirname
from utool import util_inject
from utool._internal import meta_util_arg
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[sysreq]')


def in_virtual_env():
    """
    returns True if you are running inside a python virtual environment

    References:
        http://stackoverflow.com/questions/1871549/python-determine-if-running-inside-virtualenv
    """
    import sys
    return hasattr(sys, 'real_prefix')


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
