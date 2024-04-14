#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import sys


def run_tests():
    # Build module list and run tests
    import sys
    exclude_doctests_fnames = set(['__init__.py'])

    exclude_dirs = [
        '_broken',
        'old',
        'tests',
        'timeits',
        '_scripts',
        '_timeits',
        '_doc',
        'notebook',
    ]
    from os.path import dirname
    #dpath_list = ['vtool']
    if ut.in_pyinstaller_package():
        # HACK, find_doctestable_modnames does not work in pyinstaller
        """
        import utool as ut
        dpath_list = [dirname(ut.__file__)]
        doctest_modname_list = ut.find_doctestable_modnames(
            dpath_list, exclude_doctests_fnames, exclude_dirs)
        print(ut.indent('doctest_modname_list = ' + ut.repr4(doctest_modname_list), ' ' * 8))

        """
        doctest_modname_list = [
            'utool.util_class',
            'utool.util_distances',
            'utool.util_progress',
            'utool.util_cache',
            'utool.Preferences',
            'utool.util_list',
            'utool',
            'utool.util_inspect',
            'utool.util_dict',
            'utool.util_dev',
            'utool.util_time',
            'utool.util_type',
            'utool.util_csv',
            'utool.util_iter',
            'utool.util_print',
            'utool.util_arg',
            'utool.util_logging',
            'utool.util_import',
            'utool.util_parallel',
            'utool.util_cplat',
            'utool.util_str',
            'utool.util_gridsearch',
            'utool.util_numpy',
            'utool.util_dbg',
            'utool.util_io',
            'utool.util_decor',
            'utool.util_grabdata',
            'utool.util_path',
            'utool.util_git',
            'utool.util_inject',
            'utool.util_regex',
            'utool.util_assert',
            'utool.util_latex',
            'utool.util_tests',
            'utool.util_config',
            'utool.util_autogen',
            'utool.util_hash',
            'utool.util_alg',
            'utool.util_resources',
            'utool._internal.meta_util_iter',
        ]
    else:
        #dpath_list = ['utool']
        dpath_list = [dirname(ut.__file__)]
        doctest_modname_list = ut.find_doctestable_modnames(
            dpath_list, exclude_doctests_fnames, exclude_dirs)
    # Finding weird error
    # util cache and util inspect
    #doctest_modname_list = (doctest_modname_list[4:5] + doctest_modname_list[17:18])
    #doctest_modname_list = doctest_modname_list[17:18]

    modname_list2 = []
    for modname in doctest_modname_list:
        try:
            exec('import ' + modname, globals(), locals())
        except ImportError as ex:
            ut.printex(ex)
            if not ut.in_pyinstaller_package():
                raise
        else:
            modname_list2.append(modname)

    module_list = [sys.modules[name] for name in modname_list2]

    # Write to py.test / nose format
    if ut.get_argflag('--tonose'):
        convert_tests_from_utool_to_nose(module_list)
        return 0

    nPass, nTotal, failed_cmd_list = ut.doctest_module_list(module_list)
    if nPass != nTotal:
        return 1
    else:
        return 0
    #print(ut.repr4(doctest_modname_list))


def convert_tests_from_utool_to_nose(module_list):
    # PARSE OUT TESTABLE DOCTESTTUPS
    #import utool as ut
    testtup_list = []
    seen_ = set()

    topimport_list = []

    for module in module_list:
        mod_doctest_tup = ut.get_module_doctest_tup(module=module,
                                                    verbose=False,
                                                    allexamples=True)
        enabled_testtup_list, frame_fpath, all_testflags, module = mod_doctest_tup
        flags = [tup.src not in seen_ for tup in enabled_testtup_list]
        enabled_testtup_list = ut.compress(enabled_testtup_list, flags)
        testtup_list.extend(enabled_testtup_list)
        if len(enabled_testtup_list) > 0:
            topimport_list.append('from %s import *  # NOQA' % (module.__name__,))

    print('Found %d test tups' % (len(testtup_list)))

    autogen_test_src_funcs = []
    #import redbaron
    for testtup in testtup_list:
        name = testtup.name
        num  = testtup.num
        src  = testtup.src
        want = testtup.want
        import re
        src = re.sub('# ENABLE_DOCTEST\n', '', src)
        src = re.sub('from [^*]* import \\* *# NOQA\n', '', src)
        src = re.sub('from [^*]* import \\*\n', '', src)
        #flag = testtup.flag
        if want.endswith('\n'):
            want = want[:-1]
        if want:
            #src_node = redbaron.RedBaron(src)
            #if len(src_node.find_all('name', 'result')) > 0:
            #    src_node.append('assert result == %r' % (want,))
            if '\nresult = ' in src:
                src += '\nassert str(result) == %r' % (want,)
        func_src = 'def test_%s_%d():\n' % (name.replace('.', '_'), num,) + ut.indent(src)
        autogen_test_src_funcs.append(func_src)

    autogen_test_src = '\n'.join(topimport_list) + '\n\n\n' + '\n\n\n'.join(autogen_test_src_funcs) + '\n'
    from utool import tests
    from os.path import join
    moddir = ut.get_module_dir(tests)
    ut.writeto(join(moddir, 'test_autogen_nose_tests.py'), autogen_test_src)


if __name__ == '__main__':
    """
    python -m utool.tests.run_tests
    """
    import multiprocessing
    multiprocessing.freeze_support()
    retcode = run_tests()
    sys.exit(retcode)
