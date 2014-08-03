#!/usr/bin/env python2.7
"""set PATH=%HOME%\code\utool\utool\util_scripts;%PATH%
cyth.py %HOME%/code/vtool/vtool/linalg_cython.pyx
python %HOME%/code/vtool/vtool/tests/test_linalg.py"""
from __future__ import absolute_import, division, print_function
import utool

if __name__ == '__main__':
    import sys
    print('[cyth.py] main()')
    fpath = sys.argv[1]
    ret = utool.compile_cython(fpath)
    print('[cyth.py] ret = %r' % (ret,))
    import os
    print('cwd = %r' % (os.getcwd(),))
    #if 'o' in sys.argv:
    #    if ret == 1:
    #        utool.editfile(fpath)
    print('[cyth.py] exit')
