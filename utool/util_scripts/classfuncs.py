#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""

flake8: noqa

script to help me figure out where things are used.
it is not named well

set PATH=%HOME%\code\utool\utool\util_scripts;%PATH%
classfuncs.py %HOME%/code/ibeis/ibeis/control/IBEISControl.py %HOME%/code/ibeis

classfuncs.py C:/Users/joncrall/code/ibeis/ibeis/algo/hots/query_request.py %HOME%/code/ibeis
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import sys


def show_function_usage(fname, funcname_list, dpath_list):
    # Check to see for function usage
    funcname_list = [r'\b%s\b' % (funcname.strip(),) for funcname in funcname_list if len(funcname) > 0]
    flagged_funcnames = []
    for funcname in funcname_list:
        found_filestr_list, found_lines_list, found_lxs_list = ut.grep([funcname], dpath_list=dpath_list)
        total = 0
        for lines in found_lines_list:
            total += len(lines)
        funcname_ = funcname.replace('\\b', '')
        print(funcname_ + ' ' + str(total))
        if total == 1:
            flagged_funcnames.append(funcname_)
        # See where external usage is
        isexternal_list = [fname == fname_ for fname_ in found_filestr_list]
        external_filestr_list = ut.compress(found_filestr_list, isexternal_list)
        external_lines_list = ut.compress(found_lines_list, isexternal_list)
        #external_lxs_list = ut.compress(found_lxs_list, isexternal_list)
        if len(external_filestr_list) == 0:
            print(' no external usage')
        else:
            for filename, lines in zip(external_filestr_list, external_lines_list):
                print(' * filename=%r' % (filename,))
                print(ut.repr4(lines))
            #print(ut.repr4(list(zip(external_filestr_list, external_lines_list))))
    print('----------')
    print('flagged:')
    print('\n'.join(flagged_funcnames))
    #ut.embed()


def get_codedir():
    from os.path import dirname, realpath
    import os
    if 'CODE_DIR' in os.environ:
        CODE_DIR = os.environ.get('CODE_DIR')
    else:
        CODE_DIR = dirname(dirname(realpath(__file__)))   # Home is where the .. is.  # '~/code'
    return CODE_DIR


if __name__ == '__main__':

    CODE_DIR = get_codedir()
    rman = ut.RepoManager(repo_urls=[
        'https://github.com/Erotemic/utool.git',
        'https://github.com/Erotemic/guitool.git',
        'https://github.com/Erotemic/plottool.git',
        'https://github.com/Erotemic/vtool.git',
        'https://github.com/bluemellophone/detecttools.git',
        'https://github.com/Erotemic/hesaff.git',
        'https://github.com/bluemellophone/pyrf.git',
        'https://github.com/Erotemic/ibeis.git',
        'https://github.com/aweinstock314/cyth.git',
        #'https://github.com/hjweide/pygist',
    ], code_dir=CODE_DIR)

    # (IBEIS_REPO_URLS, IBEIS_REPO_DIRS) = ut.repo_list(, forcessh=False)
    # ut.set_project_repos(IBEIS_REPO_URLS, IBEIS_REPO_DIRS)
    dpath_list = rman.repo_dirs
    # IBEIS_REPO_DIRS

    fname = ut.truepath(sys.argv[1])
    #if len(sys.argv) >= 3:
    #    grep_dpath = ut.truepath(sys.argv[2])
    print('Classfuncs of %r' % fname)
    funcname_list = ut.list_class_funcnames(fname)
    funcname_list = ut.list_global_funcnames(fname)
    print(ut.indentjoin(funcname_list, '\n *   '))
    show_function_usage(fname, funcname_list, dpath_list)
