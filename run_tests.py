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
    nPass, nTotal, failed_cmd_list = ut.doctest_module_list(module_list)
    if nPass != nTotal:
        return 1
    else:
        return 0
    #print(ut.list_str(doctest_modname_list))


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    retcode = run_tests()
    sys.exit(retcode)
