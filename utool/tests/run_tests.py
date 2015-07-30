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
        print(ut.indent('doctest_modname_list = ' + ut.list_str(doctest_modname_list), ' ' * 8))

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
    nPass, nTotal, failed_cmd_list = ut.doctest_module_list(module_list)
    if nPass != nTotal:
        return 1
    else:
        return 0
    #print(ut.list_str(doctest_modname_list))


if __name__ == '__main__':
    """
    python -m utool.tests.run_tests
    """
    import multiprocessing
    multiprocessing.freeze_support()
    retcode = run_tests()
    sys.exit(retcode)
