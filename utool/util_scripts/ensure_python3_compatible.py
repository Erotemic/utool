"""
currently just checks invalidness
does not correct anything yet.



TODO: Fixes/Warns about the folloing problems:
    * incompatible 2/3 names
    * no future print
    * no utool inject
    * unreference tagged urls
    * ill formated docstrings
    * bad init files
    * no setup file
    * docstrings are prefixed with r

    * Unused functions?


rob sp DOCTEST_ENABLE ENABLE_DOCTEST
rob sp DOCTEST_DIABLE DISABLE_DOCTEST

"""
from __future__ import absolute_import, division, print_function
import utool as ut  # NOQA


def filter_comented_lines():
    pass


def ensure_future_compatible(mod_fpath):
    failed_future_list = []
    # Test for print function
    futureline = '^from __future__ import absolute_import, division, print_function$'
    lines, lineno = ut.grepfile(mod_fpath, futureline)
    if len(lines) == 0:
        print(mod_fpath)
        failed_future_list.append(mod_fpath)
        lines, lineno = ut.grepfile(mod_fpath, futureline)


def ensure_six_moves_compatible(mod_fpath):
    six_moves = ['zip', 'map', 'range', 'filter', 'cPickle', 'cStringio', 'zip_longest']
    for funcname in six_moves:
        funcname_regex = r'\b%s\b' % (funcname)
        lines, lineno = ut.grepfile(mod_fpath, funcname_regex)
        if len(lines) > 0:
            six_import = 'from six.moves import .*' + funcname
            lines_, lineno = ut.grepfile(mod_fpath, six_import)
            if len(lines_) == 0:
                print(mod_fpath + ' failed ' + funcname)
                print(lines)
                print(lines_)


def ensure_no_invalid_commands(mod_fpath):
    command_list = ['raw_input', '__builtins__', 'xrange', 'izip_longest']
    for funcname in command_list:
        funcname_regex = r'\b%s\b' % (funcname)
        lines, lineno = ut.grepfile(mod_fpath, funcname_regex)
        if len(lines) > 0:
            print(mod_fpath + ' failed ' + funcname)
            print(lines)


def ensure_utool_compatible(mod_fpath):
    ut_inject_line1 = r'print, print_, printDBG, rrr, profile ='
    ut_inject_line2 = r'\(print, print_, printDBG, rrr, profile\) ='
    ut_inject_line3 = r'inject\(__name__,'
    ut_inject_lines = (ut_inject_line1, ut_inject_line2, ut_inject_line3)
    #ut.inject(__name'
    lines, lineno = ut.grepfile(mod_fpath, ut_inject_lines)
    if len(lines) == 0:
        print(mod_fpath + ' does not have utool')


#def ensure_compatible_modfpath_list(mod_fpath_list):


if __name__ == '__main__':
    package_dir = ut.truepath('~/code/ibeis/ibeis')
    if 'module_fpath_list' not in vars():
        module_fpath_list = ut.glob_python_modules(package_dir)

    for mod_fpath in module_fpath_list:
        #ensure_compatible_modfpath(mod_fpath)
        #ensure_six_moves_compatible(mod_fpath)
        #ensure_utool_compatible(mod_fpath)
        ensure_no_invalid_commands(mod_fpath)
