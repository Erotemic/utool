#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import os
import sys

if __name__ == '__main__':
    import utool as ut
    modname = ut.get_argval('-m')
    if modname is not None:
        modpath = ut.get_modpath_from_modname(modname)
        #print(modpath)
        argvx = sys.argv.index(modname) + 1
        argv = [modpath] + sys.argv[argvx:]
    else:
        argvx = sys.argv.index('profiler.py')
        argv = sys.argv[argvx:]
    cmd = 'profiler.sh ' + ' '.join(argv)
    print(cmd)
    os.system(cmd)
