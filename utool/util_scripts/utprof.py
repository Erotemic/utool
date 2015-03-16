#!/usr/bin/env python2.7
"""
provides a python script interface into utprof.sh
"""
from __future__ import absolute_import, division, print_function
import os

if __name__ == '__main__':
    import utool as ut
    # Allow for -m to be specified on the command line
    argv_tail = ut.get_argv_tail('utprof.py')
    cmd = 'utprof.sh ' + ' '.join(argv_tail)
    print(cmd)
    os.system(cmd)
