#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool as ut
import sys
from six.moves import range, map, filter, zip  # NOQA


def autogen_utool_runtest():
    """ special case to generate tests script for UTOOL

    Example:
        >>> from autogen_test_script import *  # NOQA
        >>> test_script = autogen_utool_runtest()
        >>> print(test_script)

    CommandLine:
        python -c "import utool; utool.autogen_utool_runtest()"
        python -c "import utool; print(utool.autogen_utool_runtest())"
        python -c "import utool; print(utool.autogen_utool_runtest())" > run_tests.sh
        chmod +x run_tests.sh
    """

    quick_tests = []

    test_argvs = '--quiet --noshow'

    repodir = '~/code/utool'

    exclude_list = []

    # Hacky, but not too bad way of getting in doctests
    # Test to see if doctest_funcs appears after main
    # Do not doctest these modules
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
    doctest_modname_list = ut.find_doctestable_modnames(dpath_list, exclude_doctests_fnames, exclude_dirs)

    for modname in doctest_modname_list:
        exec('import ' + modname, globals(), locals())
    module_list = [sys.modules[name] for name in doctest_modname_list]
    testcmds = ut.get_module_testlines(module_list, remove_pyc=True, verbose=False, pythoncmd='RUN_TEST')

    test_headers = [
        # title, default, module, testpattern
        ut.def_test('DOC', testcmds=testcmds, default=True)
    ]

    script_text = ut.make_run_tests_script_text(test_headers, test_argvs, quick_tests, repodir, exclude_list)

    return script_text

if __name__ == '__main__':
    """
    CommandLine:
        python autogen_test_script.py
        python autogen_test_script.py --verbose > run_tests.sh
        python autogen_test_script.py -o run_tests.sh && ./run_tests.sh
        chmod +x run_tests.sh
        run_tests.sh --testall
    """
    text = autogen_utool_runtest()

    runtests_fpath = ut.get_argval(('-o', '--outfile'), type_=str, default=None)
    if runtests_fpath is None and ut.get_argflag('-w'):
        runtests_fpath = 'run_tests.sh'

    if runtests_fpath is not None:
        ut.write_to(runtests_fpath, text)
    elif ut.get_argflag('--verbose'):
        print(text)
