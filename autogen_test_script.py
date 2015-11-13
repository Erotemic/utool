#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import sys  # NOQA
from six.moves import range, map, filter, zip  # NOQA


def autogen_ibeis_runtest():
    """ special case to generate tests script for IBEIS

    Example:
        >>> from autogen_test_script import *  # NOQA
        >>> test_script = autogen_ibeis_runtest()
        >>> print(test_script)

    CommandLine:
        python -c "import utool; utool.autogen_ibeis_runtest()"
        python -c "import utool; print(utool.autogen_ibeis_runtest())"

        python -c "import utool; print(utool.autogen_ibeis_runtest())" > run_tests.sh
        chmod +x run_tests.sh

    """

    quick_tests = [
        'ibeis/tests/assert_modules.py'
    ]

    #test_repos = [
    #    '~/code/ibeis'
    #    '~/code/vtool'
    #    '~/code/hesaff'
    #    '~/code/guitool'
    #]

    #test_pattern = [
    #    '~/code/ibeis/test_ibs*.py'
    #]

    test_argvs = '--quiet --noshow'

    misc_pats = [
        'test_utool_parallel.py',
        'test_pil_hash.py',
    ]

    repodir = '~/code/utool'

    exclude_list = []

    # Verbosity to show which modules at least have some tests
    #untested_modnames = ut.find_untested_modpaths(dpath_list, exclude_doctests_fnames, exclude_dirs)
    #print('\nUNTESTED MODULES:' + ut.indentjoin(untested_modnames))
    #print('\nTESTED MODULES:' + ut.indentjoin(doctest_modname_list))

    implicit_build_modlist_str = ut.codeblock(
        '''
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
        doctest_modname_list = ut.find_doctestable_modnames(dpath_list, exclude_doctests_fnames, exclude_dirs)

        for modname in doctest_modname_list:
            exec('import ' + modname, globals(), locals())
        module_list = [sys.modules[name] for name in doctest_modname_list]
        '''
    )
    globals_ = globals()
    locals_ = locals()
    exec(implicit_build_modlist_str, globals_, locals_)
    module_list = locals_['module_list']
    doctest_modname_list = locals_['doctest_modname_list']

    import_str = '\n'.join(['import ' + modname for modname in doctest_modname_list])
    modlist_str = ('module_list = [%s\n]' % ut.indentjoin([modname  + ',' for modname in doctest_modname_list]))
    explicit_build_modlist_str = '\n\n'.join((import_str, modlist_str))

    build_modlist_str = implicit_build_modlist_str
    #build_modlist_str = explicit_build_modlist_str

    pyscript_fmtstr = ut.codeblock(
        r'''
        #!/usr/bin/env python
        from __future__ import absolute_import, division, print_function
        import utool as ut


        def run_tests():
            # Build module list and run tests
            {build_modlist_str}
            ut.doctest_module_list(module_list)

        if __name__ == '__main__':
            import multiprocessing
            multiprocessing.freeze_support()
            run_tests()
        '''
    )

    pyscript_text = pyscript_fmtstr.format(build_modlist_str=ut.indent(build_modlist_str).strip())
    pyscript_text = ut.autofix_codeblock(pyscript_text)

    def def_test(header, pat=None, dpath=None, modname=None, default=False, testcmds=None):
        """ interface to make test tuple """
        return (header, default, modname, dpath, pat, testcmds)

    # BUILD OLD SHELL RUN TESTS HARNESS
    testcmds = ut.get_module_testlines(module_list, remove_pyc=True, verbose=False, pythoncmd='RUN_TEST')
    test_headers = [
        # title, default, module, testpattern
        def_test('DOC', testcmds=testcmds, default=True)
    ]

    shscript_text = ut.make_run_tests_script_text(test_headers, test_argvs, quick_tests, repodir, exclude_list)

    return shscript_text, pyscript_text

if __name__ == '__main__':
    """
    CommandLine:
        python autogen_test_script.py
        python autogen_test_script.py -w
    """
    shscript_text, pyscript_text = autogen_ibeis_runtest()
    runtest_fname = None

    if runtest_fname is None and ut.get_argflag('-w'):
        runtest_fname = 'run_tests'

    if runtest_fname is None and ut.get_argflag('-t'):
        runtest_fname = '_run_tests2'

    if runtest_fname is not None:
        ut.write_to('shell_' + runtest_fname + '.sh', shscript_text)
        ut.write_to(runtest_fname + '.py', pyscript_text)

    elif ut.get_argflag(('--verbose', '-v')):
        print(shscript_text)
        print('')
        print(pyscript_text)
