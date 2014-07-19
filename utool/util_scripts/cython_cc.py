#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import utool

if __name__ == '__main__':
    import sys
    ret = utool.compile_cython(sys.argv[1])
    print(ret)
