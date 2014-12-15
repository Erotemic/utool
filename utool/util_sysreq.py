from __future__ import absolute_import, division, print_function
import sys
from os.path import split, exists, join, dirname
import os
from utool.util_inject import inject
from utool._internal import meta_util_arg
print, print_, printDBG, rrr, profile = inject(__name__, '[sysreq]')


VERBOSE = meta_util_arg.VERBOSE
DEBUG = meta_util_arg.DEBUG


def locate_path(dname, recurse_down=True):
    'Search for a path'
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
        if VERBOSE:
            print('[sysreq] appending %r to PYTHONPATH' % dpath)
        sys.path.append(dpath)
    elif DEBUG:
        print('[sysreq] PYTHONPATH has %r' % dname)
