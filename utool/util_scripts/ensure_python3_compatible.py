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
    * tests without any asserts or want lines

    * Unused functions?


rob sp DOCTEST_ENABLE ENABLE_DOCTEST
rob sp DOCTEST_DIABLE DISABLE_DOCTEST

rob gp "python .* --allexamples"

"""
from __future__ import absolute_import, division, print_function
import utool as ut  # NOQA


def filter_comented_lines():
    pass


def fix_bad_doctestcmds():
    # http://stackoverflow.com/questions/18737863/passing-a-function-to-re-sub-in-python
    # CANNOT USE [^ ] FOR SOME GOD DAMN REASON USE /S instead
    regex_list = ['python [A-Za-z_]+[\\/]\S* --allexamples']
    dpath_list = [
        ut.ensure_crossplat_path(ut.truepath('~/code/utool/utool')),
        ut.ensure_crossplat_path(ut.truepath('~/code/ibeis/ibeis')),
        ut.ensure_crossplat_path(ut.truepath('~/code/vtool/vtool')),
        ut.ensure_crossplat_path(ut.truepath('~/code/plottool/plottool')),
        ut.ensure_crossplat_path(ut.truepath('~/code/guitool/guitool')),
    ]
    #ut.named_field_repl(['python ', ('modrelpath',),])
    #['python ', ('modrelpath', 'utool[\\/].*'), '--allexamples'])
    res = ut.grep(regex_list, recursive=True, dpath_list=dpath_list, verbose=True)
    found_filestr_list, found_lines_list, found_lxs_list = res
    fpath = res[0][0]
    def replmodpath(matchobj):
        groupdict_ = matchobj.groupdict()
        relpath = groupdict_['modrelpath']
        prefix = groupdict_['prefix']
        suffix = groupdict_['suffix']
        modname = relpath
        modname = modname.replace('\\', '.')
        modname = modname.replace('/', '.')
        modname = modname.replace('.py', '')
        return prefix + '-m ' + modname + suffix

    for fpath in found_filestr_list:
        text = ut.read_from(fpath)
        import re
        keypat_list = [
            ('prefix', 'python\s*'),
            ('modrelpath', '[A-Za-z_]+[\\/]\S*'),
            ('suffix', '.*'),
        ]
        namedregex = ut.named_field_regex(keypat_list)
        #matchobj = re.search(namedregex, text, flags=re.MULTILINE)
        #print(text)
        for matchobj in re.finditer(namedregex, text):
            print(ut.get_match_text(matchobj))
            print('--')
        newtext = re.sub(namedregex, replmodpath, text)
        #print(newtext)


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
