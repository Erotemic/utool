#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool as ut


def run_tests():
    # Build module list and run tests
    import utool.util_alg
    import utool.util_arg
    import utool.util_autogen
    import utool.util_cache
    import utool.util_class
    import utool.util_cplat
    import utool.util_csv
    import utool.util_dbg
    import utool.util_decor
    import utool.util_dict
    import utool.util_distances
    import utool.util_grabdata
    import utool.util_inspect
    import utool.util_iter
    import utool.util_list
    import utool.util_numpy
    import utool.util_parallel
    import utool.util_path
    import utool.util_progress
    import utool.util_regex
    import utool.util_str
    import utool.util_tests
    import utool.util_time

    module_list = [
        utool.util_alg,
        utool.util_arg,
        utool.util_autogen,
        utool.util_cache,
        utool.util_class,
        utool.util_cplat,
        utool.util_csv,
        utool.util_dbg,
        utool.util_decor,
        utool.util_dict,
        utool.util_distances,
        utool.util_grabdata,
        utool.util_inspect,
        utool.util_iter,
        utool.util_list,
        utool.util_numpy,
        utool.util_parallel,
        utool.util_path,
        utool.util_progress,
        utool.util_regex,
        utool.util_str,
        utool.util_tests,
        utool.util_time,
    ]
    ut.doctest_module_list(module_list)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    run_tests()
