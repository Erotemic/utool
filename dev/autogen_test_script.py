#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import sys  # NOQA
from utool import util_path
from six.moves import range, map, filter, zip  # NOQA


def make_run_tests_script_text(test_headers, test_argvs, quick_tests=None,
                               repodir=None, exclude_list=[]):
    """
    Autogeneration function

    TODO move to util_autogen or just depricate

    Examples:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> import utool  # NOQA
        >>> testdirs = ['~/code/ibeis/test_ibs*.py']
    """
    import utool as ut
    from os.path import relpath, join, dirname  # NOQA

    exclude_list += ['__init__.py']

    # General format of the testing script

    script_fmtstr = ut.codeblock(
        r'''
        #!/bin/bash
        # Runs all tests
        # Win32 path hacks
        export CWD=$(pwd)
        export PYMAJOR="$(python -c "import sys; print(sys.version_info[0])")"

        # <CORRECT_PYTHON>
        # GET CORRECT PYTHON ON ALL PLATFORMS
        export SYSNAME="$(expr substr $(uname -s) 1 10)"
        if [ "$SYSNAME" = "MINGW32_NT" ]; then
            export PYEXE=python
        else
            if [ "$PYMAJOR" = "3" ]; then
                # virtual env?
                export PYEXE=python
            else
                export PYEXE=python2.7
            fi
        fi
        # </CORRECT_PYTHON>

        PRINT_DELIMETER()
        {{
            printf "\n#\n#\n#>>>>>>>>>>> next_test\n\n"
        }}

        export TEST_ARGV="{test_argvs} $@"

        {dirdef_block}

        # Default tests to run
        set_test_flags()
        {{
            export DEFAULT=$1
        {testdefault_block}
        }}
        set_test_flags OFF
        {testdefaulton_block}

        # Parse for bash commandline args
        for i in "$@"
        do
        case $i in --testall)
            set_test_flags ON
            ;;
        esac
        {testcmdline_block}
        done

        BEGIN_TESTS()
        {{
        cat <<EOF
        {runtests_bubbletext}
        EOF
            echo "BEGIN: TEST_ARGV=$TEST_ARGV"
            PRINT_DELIMETER
            num_passed=0
            num_ran=0
            export FAILED_TESTS=''
        }}

        RUN_TEST()
        {{
            echo "RUN_TEST: $@"
            export TEST="$PYEXE $@ $TEST_ARGV"
            $TEST
            export RETURN_CODE=$?
            echo "RETURN_CODE=$RETURN_CODE"
            PRINT_DELIMETER
            num_ran=$(($num_ran + 1))
            if [ "$RETURN_CODE" == "0" ] ; then
                num_passed=$(($num_passed + 1))
            fi
            if [ "$RETURN_CODE" != "0" ] ; then
                export FAILED_TESTS="$FAILED_TESTS\n$TEST"
            fi
        }}

        END_TESTS()
        {{
            echo "RUN_TESTS: DONE"
            if [ "$FAILED_TESTS" != "" ] ; then
                echo "-----"
                printf "Failed Tests:"
                printf "$FAILED_TESTS\n"
                printf "$FAILED_TESTS\n" >> failed_shelltests.txt
                echo "-----"
            fi
            echo "$num_passed / $num_ran tests passed"
        }}

        #---------------------------------------------
        # START TESTS
        BEGIN_TESTS

        {quicktest_block}

        {test_block}

        #---------------------------------------------
        # END TESTING
        END_TESTS
        ''')

    testcmdline_fmtstr = ut.codeblock(
        r'''
        case $i in --notest{header_lower})
            export {testflag}=OFF
            ;;
        esac
        case $i in --test{header_lower})
            export {testflag}=ON
            ;;
        esac
        ''')

    header_test_block_fmstr = ut.codeblock(
        r'''

        #---------------------------------------------
        #{header_text}
        if [ "${testflag}" = "ON" ] ; then
        cat <<EOF
        {header_bubble_text}
        EOF
        {testlines_block}
        fi
        ''')

    #specialargv = '--noshow'
    specialargv = ''
    testline_fmtstr = 'RUN_TEST ${dirvar}/{fpath} {specialargv}'
    testline_fmtstr2 = 'RUN_TEST {fpath} {specialargv}'

    def format_testline(fpath, dirvar):
        if dirvar is None:
            return testline_fmtstr2.format(fpath=fpath, specialargv=specialargv)
        else:
            return testline_fmtstr.format(dirvar=dirvar, fpath=fpath, specialargv=specialargv)

    default_flag_line_list = []
    defaulton_flag_line_list = []
    testcmdline_list = []
    dirdef_list = []
    header_test_block_list = []

    known_tests = ut.ddict(list)

    # Tests to always run
    if quick_tests is not None:
        quicktest_block = '\n'.join(
            ['# Quick Tests (always run)'] +
            ['RUN_TEST ' + testline for testline in quick_tests])
    else:
        quicktest_block = '# No quick tests'

    # Loop over different test types
    for testdef_tup in test_headers:
        header, default, modname, dpath, pats, testcmds = testdef_tup
        # Build individual test type information
        header_upper =  header.upper()
        header_lower = header.lower()
        testflag = header_upper + '_TEST'

        if modname is not None:
            dirvar = header_upper + '_DIR'
            dirdef = ''.join([
                'export {dirvar}=$($PYEXE -c "',
                'import os, {modname};',
                'print(str(os.path.dirname(os.path.dirname({modname}.__file__))))',
                '")']).format(dirvar=dirvar, modname=modname)
            dirdef_list.append(dirdef)
        else:
            dirvar = None

        # Build test dir
        #dirvar = header_upper + '_DIR'
        #dirdef = 'export {dirvar}={dirname}'.format(dirvar=dirvar, dirname=dirname)
        #dirdef_list.append(dirdef)

        # Build command line flags
        default_flag_line = 'export {testflag}=$DEFAULT'.format(testflag=testflag)

        if default:
            defaulton_flag_line = 'export {testflag}=ON'.format(testflag=testflag)
            defaulton_flag_line_list.append(defaulton_flag_line)

        testcmdline_fmtdict = dict(header_lower=header_lower,
                                        testflag=testflag,)
        testcmdline = testcmdline_fmtstr.format(**testcmdline_fmtdict)

        #ut.ls(dpath)

        # VERY HACK BIT OF CODE

        # Get list of tests from patterns
        if testcmds is None:
            if modname is not None:
                module = __import__(modname)
                repo_path = dirname(dirname(module.__file__))
            else:
                repo_path = repodir
            dpath_ = ut.unixpath(util_path.unixjoin(repo_path, dpath))

            if header_upper == 'OTHER':
                # Hacky way to grab any other tests not explicitly seen in this directory
                _testfpath_list = list(set(ut.glob(dpath_, '*.py')) - set(known_tests[dpath_]))
                #_testfpath_list = ut.glob(dpath_, '*.py')
                #set(known_tests[dpath_])
            else:
                _testfpath_list = ut.flatten([ut.glob(dpath_, pat) for pat in pats])

            def not_excluded(x):
                return not any([x.find(exclude) > -1 for exclude in exclude_list])

            _testfpath_list = list(filter(not_excluded, _testfpath_list))

            known_tests[dpath_].extend(_testfpath_list)
            #print(_testfpath_list)
            testfpath_list = [util_path.unixjoin(dpath, relpath(fpath, dpath_))
                              for fpath in _testfpath_list]

            testline_list = [format_testline(fpath, dirvar) for fpath in testfpath_list]
        else:
            testline_list = testcmds

        testlines_block = ut.indentjoin(testline_list).strip('\n')

        # Construct test block for this type
        header_text = header_upper + ' TESTS'
        headerfont = 'cybermedium'
        header_bubble_text =  ut.indent(ut.bubbletext(header_text, headerfont).strip())
        header_test_block_dict = dict(
            testflag=testflag,
            header_text=header_text,
            testlines_block=testlines_block,
            header_bubble_text=header_bubble_text,)
        header_test_block = header_test_block_fmstr.format(**header_test_block_dict)

        # Append to script lists
        header_test_block_list.append(header_test_block)
        default_flag_line_list.append(default_flag_line)
        testcmdline_list.append(testcmdline)

    runtests_bubbletext = ut.bubbletext('RUN TESTS', 'cyberlarge')

    test_block = '\n'.join(header_test_block_list)
    dirdef_block = '\n'.join(dirdef_list)
    testdefault_block = ut.indent('\n'.join(default_flag_line_list))
    testdefaulton_block = '\n'.join(defaulton_flag_line_list)
    testcmdline_block = '\n'.join(testcmdline_list)

    script_fmtdict = dict(
        quicktest_block=quicktest_block,
        runtests_bubbletext=runtests_bubbletext,
        test_argvs=test_argvs, dirdef_block=dirdef_block,
        testdefault_block=testdefault_block,
        testdefaulton_block=testdefaulton_block,
        testcmdline_block=testcmdline_block,
        test_block=test_block,)
    script_text = script_fmtstr.format(**script_fmtdict)

    return script_text


def autogen_ibeis_runtest():
    """ special case to generate tests script for IBEIS

    Example:
        >>> # DISABLE_DOCTEST
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
