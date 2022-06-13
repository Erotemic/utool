# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from os.path import join, normpath, pathsep, dirname  # NOQA


def get_regstr(regtype, var, val):
    regtype_map = {
        'REG_EXPAND_SZ': 'hex(2):',
        'REG_DWORD': 'dword:',
        'REG_BINARY': None,
        'REG_MULTI_SZ': None,
        'REG_SZ': '',
    }
    # It is not a good idea to write these variables...
    EXCLUDE = ['USERPROFILE', 'USERNAME', 'SYSTEM32']
    if var in EXCLUDE:
        return ''
    def quotes(str_):
        return '"' + str_.replace('"', r'\"') + '"'
    sanitized_var = quotes(var)
    if regtype == 'REG_EXPAND_SZ':
        # Weird encoding
        #bin_ = binascii.hexlify(hex_)
        #val_ = ','.join([''.join(hex2) for hex2 in hex2zip])
        #import binascii  # NOQA
        x = val
        ascii_ = x.encode("ascii")
        hex_ = ascii_.encode("hex")
        hex_ = x.encode("hex")
        hex2zip = zip(hex_[0::2], hex_[1::2])
        spacezip = [('0', '0')] * len(hex2zip)
        hex3zip = zip(hex2zip, spacezip)
        sanitized_val = ','.join([''.join(hex2) + ',' + ''.join(space) for hex2, space in hex3zip])
    elif regtype == 'REG_DWORD':
        sanitized_val = '%08d' % int(val)
    else:
        sanitized_val = quotes(val)
    # Comment with the human-readable nonhex version of the string
    comment = '; ' + var + '=' + val
    regstr = sanitized_var + '=' + regtype_map[regtype] + sanitized_val
    return comment + '\n' + regstr


def make_regfile_str(key, varval_list, rtype):
    # Input: list of (var, val) tuples
    # key to put varval list in
    # rtype - type of registry variables
    envtxt_list = ['Windows Registry Editor Version 5.00',
                   '',
                   key]
    print('\n'.join(map(repr, varval_list)))
    varval_list = filter(lambda x: isinstance(x, tuple), varval_list)
    vartxt_list = [get_regstr(rtype, var, val) for (var, val) in varval_list]
    envtxt_list.extend(vartxt_list)
    regfile_str = '\n'.join(envtxt_list)
    return regfile_str


def add_to_win32_PATH(script_fpath, *add_path_list):
    r"""
    Writes a registery script to update the PATH variable into the sync registry

    CommandLine:
        python -m utool.util_win32 --test-add_to_win32_PATH --newpath "C:\Program Files (x86)\Graphviz2.38\bin"

    Example:
        >>> # DISABLE_DOCTEST
        >>> # SCRIPT
        >>> from utool.util_win32 import *  # NOQA
        >>> script_fpath = join(ut.truepath('~'), 'Sync/win7/registry', 'UPDATE_PATH.reg')
        >>> new_path = ut.get_argval('--newpath', str, default=None)
        >>> result = add_to_win32_PATH(script_fpath, new_path)
        >>> print(result)
    """
    import utool as ut
    write_dir = dirname(script_fpath)
    key = r'[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment]'
    rtype = 'REG_EXPAND_SZ'
    # Read current PATH values
    win_pathlist = list(os.environ['PATH'].split(os.path.pathsep))
    new_path_list = ut.unique_ordered(win_pathlist + list(add_path_list))
    #new_path_list = unique_ordered(win_pathlist, rob_pathlist)
    print('\n'.join(new_path_list))
    pathtxt = pathsep.join(new_path_list)
    varval_list = [('Path', pathtxt)]
    regfile_str = make_regfile_str(key, varval_list, rtype)
    ut.view_directory(write_dir)
    print(regfile_str)
    ut.writeto(script_fpath, regfile_str, mode='wb')
    print('Please have an admin run the script. You may need to restart')


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_win32
        python -m utool.util_win32 --allexamples
        python -m utool.util_win32 --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
