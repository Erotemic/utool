# -*- coding: utf-8 -*-
"""

TODO: rename to something better.
this is pretty much just a mass file
editor / inspector

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
import os


if 'CODE_DIR' in os.environ:
    CODE_DIR = os.environ.get('CODE_DIR')

# Non local project repos
(IBEIS_REPO_URLS, IBEIS_REPO_DIRS) = ut.repo_list([
    'https://github.com/Erotemic/utool.git',
    'https://github.com/Erotemic/guitool.git',
    'https://github.com/Erotemic/plottool.git',
    'https://github.com/Erotemic/vtool.git',
    'https://github.com/bluemellophone/detecttools.git',
    'https://github.com/Erotemic/hesaff.git',
    'https://github.com/bluemellophone/pyrf.git',
    'https://github.com/Erotemic/ibeis.git',
    # 'https://github.com/aweinstock314/cyth.git',
    #'https://github.com/hjweide/pygist',
], CODE_DIR, forcessh=False)


ut.set_code_dir(CODE_DIR)
ut.set_project_repos(IBEIS_REPO_URLS, IBEIS_REPO_DIRS)


def ensure_ibeis_control_explicit_namespace(varname_list):
    # <input>
    import ibeis
    namespace = 'const'
    module = ibeis.control.IBEISControl
    fpath = module.__file__
    # </input>
    varname_list = [
        'ENCOUNTER_TABLE', 'EG_RELATION_TABLE', 'AL_RELATION_TABLE',
        'GL_RELATION_TABLE', 'CHIP_TABLE', 'FEATURE_TABLE',
        'LBLIMAGE_TABLE', 'CONTRIBUTOR_TABLE', 'LBLTYPE_TABLE',
        'METADATA_TABLE', 'VERSIONS_TABLE'
    ]
    ensure_explicit_namespace(fpath, namespace, varname_list)


def ensure_explicit_namespace(fpath, namespace, varname_list):
    import re
    import utool as ut

    text = ut.read_from(fpath)
    orig_text = text
    new_text = text

    for varname in varname_list:
        regex = ''.join((
            ut.named_field('prefix', '[^.]'),
            ut.named_field('var', ut.whole_word(varname)),
        ))
        repl = ''.join((
            ut.bref_field('prefix'),
            namespace, '.',
            ut.bref_field('var')
        ))

        new_text = re.sub(regex, repl, new_text)

    textdiff = ut.get_textdiff(orig_text, new_text)
    print(textdiff)
    if ut.user_cmdline_prompt('Does the text look good?'):
        # if diff looks good write
        ut.write_to(fpath, new_text)


def abstract_external_module_cv2():
    from os.path import join  # NOQA
    modname = 'cv2'
    repo_dirs = ut.get_project_repo_dirs()
    #exclude_dirs = [join(dpath, 'build') for dpath in repo_dirs]

    grepkw = dict(
        #exclude_dirs=exclude_dirs,
        dpath_list=repo_dirs,
        greater_exclude_dirs=ut.get_standard_exclude_dnames(),
        recursive=True,
    )
    modregex = r'\<' + modname + r'\>'
    tup = ut.grep(modregex, **grepkw)
    fpath_list, line_list, lineno_list = tup
    fpath_list = [fpath.replace('\\', '/') for fpath in fpath_list]
    print('\n'.join(fpath_list))


def filter_comented_lines():
    pass


def change_doctestcommand_to_use_dashm_flag():
    r"""
    VimRegex: # note sure how to execute replace command in vim in one lin
    %s/python\s*\([A-Za-z_]+[\\/]\S*\)\.py\(.*\)/python -m \1 \2

    """
    # http://stackoverflow.com/questions/18737863/passing-a-function-to-re-sub-in-python
    # CANNOT USE [^ ] FOR SOME GOD DAMN REASON USE /S instead
    regex_list = ['python [A-Za-z_]+[\\/]\\S* --allexamples']
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

    import re
    keypat_list = [
        ('prefix', 'python\\s*'),
        ('modrelpath', '[A-Za-z_]+[\\/]\\S*'),
        ('suffix', '.*'),
    ]
    namedregex = ut.named_field_regex(keypat_list)

    # Define function to pass to re.sub
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
        #matchobj = re.search(namedregex, text, flags=re.MULTILINE)
        #print(text)
        #for matchobj in re.finditer(namedregex, text):
        #    print(ut.get_match_text(matchobj))
        #    print('--')
        newtext = re.sub(namedregex, replmodpath, text)
        # Perform replacement
        ut.write_to(fpath, newtext)
        #print('\n'.join(newtext.splitlines()[-10:]))


def ensure_future_compatible(mod_fpath):
    failed_future_list = []
    # Test for print function
    futureline = '^from __future__ import absolute_import, division, print_function, unicode_literals$'
    lines, lineno = ut.grepfile(mod_fpath, futureline)
    if len(lines) == 0:
        print(mod_fpath)
        failed_future_list.append(mod_fpath)
        lines, lineno = ut.grepfile(mod_fpath, futureline)


def check_six_moves_compatibility(mod_fpath):
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
    ut_inject_line1 = r'print, rrr, profile ='
    ut_inject_line2 = r'\(print, rrr, profile\) ='
    ut_inject_line3 = r'inject2\(__name__,'
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
        #check_six_moves_compatibility(mod_fpath)
        #ensure_utool_compatible(mod_fpath)
        ensure_no_invalid_commands(mod_fpath)
