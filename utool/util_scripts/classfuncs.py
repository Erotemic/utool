#!/usr/bin/env python2.7
""" flake8: noqa
set PATH=%HOME%\code\utool\utool\util_scripts;%PATH%
classfuncs.py %HOME%/code/ibeis/ibeis/control/IBEISControl.py %HOME%/code/ibeis
"""
from __future__ import absolute_import, division, print_function
import utool
import sys

if __name__ == '__main__':
    fname = utool.truepath(sys.argv[1])
    if len(sys.argv) >= 3:
        grep_dpath = utool.truepath(sys.argv[2])
    print('Classfuncs of %r' % fname)
    funcname_list = utool.list_class_funcnames(fname)
    print(utool.indentjoin(funcname_list, '\n *   '))

    # Check to see for function usage
    funcname_list = [r'\b%s\b' % (funcname.strip(),) for funcname in funcname_list if len(funcname) > 0]
    flagged_funcnames = []
    for funcname in funcname_list:
        found_filestr_list, found_lines_list, found_lxs_list = utool.grep([funcname], dpath_list=[grep_dpath])
        total = 0
        for lines in found_lines_list:
            total += len(lines)
        funcname_ = funcname.replace('\\b', '')
        print(funcname_ + ' ' + str(total))
        if total == 1:
            flagged_funcnames.append(funcname_)
    print('----------')
    print('flagged:')
    print('\n'.join(flagged_funcnames))
    utool.embed()
