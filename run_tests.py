#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool as ut


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
    dpath_list = ['utool']
    doctest_modname_list = ut.find_doctestable_modnames(
        dpath_list, exclude_doctests_fnames, exclude_dirs)
    # Finding weird error
    # util cache and util inspect
    #doctest_modname_list = (doctest_modname_list[4:5] + doctest_modname_list[17:18])
    #doctest_modname_list = doctest_modname_list[17:18]

    for modname in doctest_modname_list:
        exec('import ' + modname, globals(), locals())
    module_list = [sys.modules[name] for name in doctest_modname_list]
    ut.doctest_module_list(module_list)
    #print(ut.list_str(doctest_modname_list))


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    run_tests()
