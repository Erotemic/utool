#!/usr/bin/env python2.7
""" flake8: noqa
set PATH=%HOME%\code\utool\utool\util_scripts;%PATH%
"""
from __future__ import absolute_import, division, print_function
import utool
import sys

if __name__ == '__main__':
    fname = sys.argv[1]
    print('Classfuncs of %r' % fname)
    funcname_list = utool.list_class_funcnames(fname)
    print(utool.indentjoin(funcname_list, '\n *   '))
