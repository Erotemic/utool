#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script to open directory in current window manager
"""
import utool as ut


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = None
    ut.assertpath(path)
    if ut.checkpath(path, verbose=True):
        ut.view_directory(path)
    #F:\\data\\work\\PZ_MTEST\\_ibsdb\\
