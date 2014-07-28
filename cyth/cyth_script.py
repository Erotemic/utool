#!/usr/env/python2.7
"""
ut
python cyth/cyth_script.py ~/code/fpath
cyth_script.py ~/code/ibeis/ibeis/model/hots
cyth_script.py "~/code/vtool/vtool"

"""
from __future__ import absolute_import, division, print_function
import utool
import sys
from os.path import splitext, isfile


def find_cyth_tags(py_text):
    """
    Parses between the <CYTHE> </CYTHE> tags. Tags must be the first or last
    characters in the string so it doesn't pick up the ones in this docstr.
    Also returns line numbers so future parsing is less intensive.
    """
    tagstr_list = []
    lineno_list = []
    return tagstr_list, lineno_list


def parse_cythe_tags(tagstr_list, lineno_list, py_text):
    """
    creates new text for a pyx file
    """
    cython_text_blocks = []
    cython_text = ''.join(cython_text_blocks)
    return cython_text


def cythonize_fpath(py_fpath):
    print('[cyth] CYTHONIZE: py_fpath=%r' % py_fpath)
    cy_fpath = splitext(py_fpath)[0] + '_cyth.pyx'
    py_text = utool.read_from(py_fpath)
    lineno_list, tagstr_list = find_cyth_tags(py_text)
    if len(tagstr_list) > 0:
        cython_text = parse_cythe_tags(tagstr_list, lineno_list, py_fpath)
        utool.write_to(cy_fpath, cython_text)


if __name__ == '__main__':
    print('[cyth] main')
    input_path_list = utool.get_fpath_args(sys.argv[1:], pat='*.py')
    print('[cyth] nInput=%d' % (len(input_path_list,)))
    for fpath in input_path_list:
        if isfile(fpath):
            abspath = utool.unixpath(fpath)
            cythonize_fpath(abspath)
