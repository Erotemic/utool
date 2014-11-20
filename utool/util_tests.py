""" Helpers for tests """
from __future__ import absolute_import, division, print_function
import six
from six.moves import builtins
import inspect
import types
import parse  # NOQA
import traceback  # NOQA
import sys
from os.path import basename
from utool import util_print
from utool import util_dbg
from utool import util_arg
from utool import util_path
from utool import util_time
from utool.util_inject import inject
from utool._internal.meta_util_six import get_funcname
print, print_, printDBG, rrr, profile = inject(__name__, '[tests]')


VERBOSE_TEST = '--verb-test' in sys.argv or '--verbose-test' in sys.argv

HAPPY_FACE = r'''
               .-""""""-.
             .'          '.
            /   O      O   \
           :                :
           |                |
           : ',          ,' :
            \  '-......-'  /
             '.          .'
               '-......-'
                   '''


SAD_FACE = r'''
               .-""""""-.
             .'          '.
            /   O      O   \
           :           `    :
           |                |
           :    .------.    :
            \  '        '  /
             '.          .'
               '-......-'
                  '''


def _get_testable_name(testable):
    import utool
    try:
        testable_name = testable.func_name
    except AttributeError as ex1:
        try:
            testable_name = testable.__name__
        except AttributeError as ex2:
            utool.printex(ex1, utool.list_str(dir(testable)))
            utool.printex(ex2, utool.list_str(dir(testable)))
            raise
    return testable_name


def parse_docblocks_from_docstr(docstr):
    # FIXME Requires tags to be separated by two spaces
    import parse   # NOQA
    import utool as ut  # NOQA
    initial_docblocks = docstr.split('\n\n')
    #print('__________')
    #print('\n---\n'.join(initial_docblocks))
    docstr_blocks = []
    for docblock in initial_docblocks:
        docblock = docblock.strip('\n')
        indent = ' ' * ut.get_indentation(docblock)
        parse_result = parse.parse(indent + '{tag}:\n{rest}', docblock)
        if parse_result is not None:
            header = parse_result['tag']
        else:
            header = ''
        docstr_blocks.append((header, docblock))
    #print(docstr_blocks)
    return docstr_blocks


def parse_doctest_from_docstr(docstr):
    """
    because doctest itself doesnt do what I want it to do
    """
    import utool as ut
    docstr_blocks = parse_docblocks_from_docstr(docstr)
    example_docblocks = []
    for header, docblock in docstr_blocks:
        if header.startswith('Example'):
            example_docblocks.append((header, docblock))

    testheader_list = []
    testsrc_list    = []
    testwant_list   = []
    for header, docblock in example_docblocks:
        nonheader_src = ut.unindent('\n'.join(docblock.splitlines()[1:]))
        nonheader_lines = nonheader_src.splitlines()
        reversed_src_lines = []
        reversed_want_lines = []
        finished_want = False

        for line in reversed(nonheader_lines):
            if not finished_want:
                if line.startswith('>>> ') or line.startswith('... '):
                    finished_want = True
                else:
                    reversed_want_lines.append(line)
                    continue
            reversed_src_lines.append(line[4:])
        test_src = '\n'.join(reversed_src_lines[::-1])
        test_want = '\n'.join(reversed_want_lines[::-1])
        testheader_list.append(header)
        testsrc_list.append(test_src)
        testwant_list.append(test_want)
        #print('Parsed header=%r' % header)
        #print('Parsed src=%r' % test_src)
    return testheader_list, testsrc_list, testwant_list


def get_doctest_examples(func_or_class):
    """
    get_doctest_examples

    Args:
        func_or_class (function)

    Returns:
        tuple (list, list): example_list, want_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> func_or_class = get_doctest_examples
        >>> testsrc_list, testwant_list, docstr = get_doctest_examples(func_or_class)
        >>> result = str(len(example_list) + len(want_list))
        >>> print(result)
        2
    """
    import utool as ut
    if VERBOSE_TEST:
        print('[util_test] parsing %r for doctest' % (func_or_class))

    # Get the docstring
    try:
        docstr = func_or_class.func_doc
    except AttributeError:
        docstr = func_or_class.__doc__

    # Cache because my janky parser is slow
    docstr = ut.unindent(docstr)
    #with ut.GlobalShelfContext('utool') as shelf:
    #    if False and docstr in shelf:
    #        testsrc_list, testwant_list = shelf[docstr]
    #    else:
    testheader_list, testsrc_list, testwant_list = parse_doctest_from_docstr(docstr)
    #       shelf[docstr] = testsrc_list, testwant_list
    return testsrc_list, testwant_list, docstr
    # doctest doesnt do what i want. so I wrote my own primative but effective
    # parser.
    '''
    import doctest
    try:
        doctest_parser = doctest.DocTestParser()
        comment_iter = doctest_parser.parse(docstr)
        #comment_iter = doctest_parser.get_examples(docstr)
    except Exception as ex:
        import utool as ut
        ut.printex(ex, 'error parsing:\n%s\nin function %r' % (docstr, func_or_class))
        raise
    # Output lists
    example_list = []
    want_list = []
    # Accumulator lists
    current_example_lines = []
    current_want_lines = []
    def append_current_example():
        # next example defined if any examples lines have been found
        if len(current_example_lines) >= 1:
            example_list.append(''.join(current_example_lines))
            want_list.append(''.join(current_want_lines))
    # Loop over doctest lines
    for c in comment_iter:
        if isinstance(c, doctest.Example):
            # Append a docline or a wantline
            current_example_lines.append(c.source)
            current_want_lines.append(c.want)
        elif c == '':
            pass
        else:
            # Append current accumulators to output and reset
            # accumulators.
            append_current_example()
            current_example_lines = []
            current_want_lines = []
    append_current_example()
    import utool as ut
    ut.embed()
    return example_list, want_list
    '''


def doctest_modules(module_list):
    nPass = 0
    nTotal = 0
    failed_cmd_list = []
    for module in module_list:
        (nPass_, nTotal_, failed_cmd_list_) = doctest_funcs(module=module, allexamples=True)
        nPass  += nPass_
        nTotal += nTotal_
        failed_cmd_list.extend(failed_cmd_list_)

    print('+-------')
    print('| FINISHED TESTING MODULE LIST')
    print('| passed %d / %d' % (nPass, nTotal))
    print('L-------')
    if len(failed_cmd_list) > 0:
        print('Failed Tests:')
        print('\n'.join(failed_cmd_list))
    else:
        print('No Failures :)')


def get_module_testlines(module_list, remove_pyc=True, verbose=True, pythoncmd='python'):
    """
    Builds test commands for autogen tests
    """
    import utool as ut  # NOQA
    testcmd_list = []
    for module in module_list:
        enabled_testtup_list, frame_fpath, all_testflags, module_ = get_doctest_testtup_list(module=module, allexamples=True, verbose=verbose)
        for testtup in enabled_testtup_list:
            testflag = testtup[-1]
            if remove_pyc:
                # FIXME python 3
                frame_fpath = frame_fpath.replace('.pyc', '.py')
            frame_rel_fpath = ut.get_relative_modpath(frame_fpath)
            testcmd = ' '.join((pythoncmd, frame_rel_fpath, testflag))
            testcmd_list.append(testcmd)
    return testcmd_list


def get_doctest_testtup_list(testable_list=None, check_flags=True, module=None,
                             allexamples=None, needs_enable=None, N=0,
                             verbose=True):
    import utool as ut  # NOQA
    if needs_enable is None:
        needs_enable = not ut.get_argflag('--enableall')
        #needs_enable = True
    TEST_ALL_EXAMPLES = allexamples or ut.get_argflag(('--allexamples', '--all-examples'))
    testable_name_list = []
    if testable_list is None:
        testable_list = []
    if isinstance(testable_list, types.ModuleType):
        module = testable_list
        testable_list = []
    # ----------------------------------------
    # Inspect caller module for testable names
    # ----------------------------------------
    if module is None:
        frame_fpath = '???'
        try:
            frame = ut.get_caller_stack_frame(N=N)
            main_modname = '__main__'
            frame_name  = frame.f_globals['__name__']
            frame_fpath = frame.f_globals['__file__']
            if frame_name == main_modname:
                module = sys.modules[main_modname]
        except Exception as ex:
            print(frame.f_globals)
            ut.printex(ex, keys=['frame', 'module'])
            raise
        allexamples = False
    else:
        frame_fpath = module.__file__
        allexamples = True
    # ----------------------------------------
    # Get testable functions
    # ----------------------------------------
    try:
        if verbose or VERBOSE_TEST:
            print('[util_test] Iterating over module funcs')

        for key, val in ut.iter_module_funcs(module):
            docstr = inspect.getdoc(val)
            if docstr is not None and docstr.find('Example') >= 0:
                testable_name_list.append(key)
                testable_list.append(val)
                #else:
                #    if docstr.find('Example') >= 0:
                #        pass
                #        #print('[util_dev] DOCTEST DISABLED: %s' % key)
    except Exception as ex:
        ut.printex(ex, keys=['frame'])
        raise
    #if verbose:
    #    for val in testable_list:
    #        print('[util_dev] DOCTEST ENABLED: %s' % val)
    # ----------------------------------------
    # Get testable function examples
    # ----------------------------------------
    test_sentinals = [
        'ENABLE_DOCTEST',
        #'ENABLE_TEST',
        #'ENABLE_DOCTEST',
        #'ENABLE_UTOOL_DOCTEST',
        #'UTOOL_TEST',
        #'UTOOLTEST'
    ]
    sorted_testable = sorted(list(set(testable_list)), key=_get_testable_name)
    testtup_list = []
    # Append each testable example
    for testable in sorted_testable:
        testname = _get_testable_name(testable)
        examples, wants, docstr = get_doctest_examples(testable)
        if len(examples) > 0:
            for testno , srcwant_tup in enumerate(zip(examples, wants)):
                src, want = srcwant_tup
                src_ = ut.regex_replace('from __future__ import.*$', '', src)
                test_disabled = not any([src_.find(s) >= 0 for s in test_sentinals])
                if needs_enable and test_disabled:
                    #print('skipping: %r' % testname)
                    #print(src)
                    continue
                #ut.embed()
                testtup = (testname, testno, src_, want)
                testtup_list.append(testtup)
        else:
            print('WARNING: no examples in %r for testname=%r' % (frame_fpath, testname))
            if verbose:
                print(testable)
                print(examples)
                print(wants)
                print(docstr)
    # ----------------------------------------
    # Get enabled (requested) examples
    # ----------------------------------------
    all_testflags = []
    enabled_testtup_list = []
    distabled_testflags  = []
    subx = ut.get_argval('--subx', type_=int, default=None, help_='Only tests the subxth example')
    for testtup in testtup_list:
        name, num, src, want = testtup
        prefix = '--test-'
        flag1 = prefix + name + ':' + str(num)
        flag2 = prefix + name
        flag3 = prefix + name.replace('_', '-') + ':' + str(num)
        flag4 = prefix + name.replace('_', '-')
        testflag = ut.get_argflag((flag1, flag2, flag3, flag4))
        testenabled = TEST_ALL_EXAMPLES  or not check_flags or testflag
        if subx is not None and subx != num:
            continue
        all_testflags.append(flag4)
        if testenabled:
            new_testtup = (name, num, src, want, flag1)
            enabled_testtup_list.append(new_testtup)
        else:
            distabled_testflags.append(flag1)
    return enabled_testtup_list, frame_fpath, all_testflags, module


def doctest_funcs(testable_list=None, check_flags=True, module=None, allexamples=None,
                  needs_enable=None, strict=True, verbose=True):
    """
    Main entry point into utools main module doctest harness

    Args:
        testable_list (list):
        check_flags (bool):
        module (None):
        allexamples (None):
        needs_enable (None):

    Returns:
        tuple: (nPass, nTotal)

    CommandLine:
        python -c "import utool; utool.doctest_funcs(module=utool.util_tests, needs_enable=False)"
        python ibeis/model/preproc/preproc_chip.py --all-examples

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> testable_list = []
        >>> check_flags = True
        >>> module = None
        >>> allexamples = None
        >>> needs_enable = None
        >>> # careful might infinitely recurse
        >>> (nPass, nTotal) = doctest_funcs(testable_list, check_flags, module, allexamples, needs_enable)
        >>> print((nPass, nTotal))
    """
    import multiprocessing
    import utool as ut  # NOQA
    multiprocessing.freeze_support()  # just in case
    #
    if verbose:
        print('[util_test.doctest_funcs] Running doctest funcs')
    tup_ = get_doctest_testtup_list(testable_list, check_flags, module,
                                    allexamples, needs_enable, N=1,
                                    verbose=verbose)
    enabled_testtup_list, frame_fpath, all_testflags, module  = tup_
    # ----------------------------------------
    # Run enabled examles
    # ----------------------------------------
    nTotal = len(enabled_testtup_list)
    nPass = 0
    nFail = 0
    failed_flag_list = []
    for testtup in enabled_testtup_list:
        name, num, src, want, flag = testtup
        print('\n\n ---- DOCTEST ' + name.upper() + ':' + str(num) + '---')
        print(ut.msgblock('EXEC SRC', src))
        # --- EXEC STATMENT ---
        test_globals = module.__dict__.copy()
        try:
            test_locals = ut.run_test((name,  src, frame_fpath), globals=test_globals, want=want)
            nPass += (test_locals is not False)
        except Exception:
            nFail += 1
            failed_flag_list.append(flag)
            if strict:
                raise
            pass
    # -------
    # Print Results
    if nTotal == 0:
        print('No test flags sepcified. Please choose one of the following flags')
        print('Valid test argflags:\n' + '    --allexamples' + ut.indentjoin(all_testflags, '\n    '))
    print('+-------')
    print('| finished testing fpath=%r' % (frame_fpath,))
    print('| passed %d / %d' % (nPass, nTotal))
    print('L-------')
    failed_cmd_list = []
    if nFail > 0:
        failed_cmd_list = ['python %s %s' % (frame_fpath, flag_)
                            for flag_ in failed_flag_list]
        print('Failed Tests:')
        print('\n'.join(failed_cmd_list))
    return (nPass, nTotal, failed_cmd_list)


def run_test(func, *args, **kwargs):
    """
    Runs the test function with success / failure printing

    Input:
        Anything that needs to be passed to <func>
    """
    func_is_text = isinstance(func, types.TupleType)
    if func_is_text:
        (funcname, src, frame_fpath) = func
    else:
        funcname = get_funcname(func)
    upper_funcname = funcname.upper()
    with util_print.Indenter('[' + funcname + ']'):
        try:
            import utool as ut
            if ut.VERBOSE:
                printTEST('[TEST.BEGIN] %s ' % (sys.executable))
                printTEST('[TEST.BEGIN] %s ' % (funcname,))
            with util_time.Timer(upper_funcname) as timer:
                if func_is_text:
                    test_locals = {}
                    test_globals = kwargs.get('globals', {})
                    want = kwargs.get('want', None)
                    #test_globals['print'] = doctest_print
                    six.exec_(src, test_globals, test_locals)
                    if want is None or want == '':
                        print('warning test does not want anything')
                    else:
                        if want.endswith('\n'):
                            want = want[:-1]
                        result = str(test_locals.get('result', 'NO VARIABLE NAMED result'))
                        #print('!! RESULT LINES: ')
                        #print(result)
                        if result != want:
                            errmsg1 = ''
                            errmsg1 += ('GOT: result=%r\n' % (result))
                            errmsg1 += ('EXPECTED: want=%r\n' % (want))
                            raise AssertionError('result != want\n' + errmsg1)
                        #assert result == want, 'result is not the same as want'
                    #print('\n'.join(output_lines))
                else:
                    test_locals = func(*args, **kwargs)
                print('')
                # Write timings
            printTEST('[TEST.FINISH] %s -- SUCCESS' % (funcname,))
            print(HAPPY_FACE)
            with open('_test_times.txt', 'a') as file_:
                msg = '%.4fs in %s\n' % (timer.ellapsed, upper_funcname)
                file_.write(msg)
            return test_locals
        except Exception as ex:
            import utool as ut
            exc_type, exc_value, tb = sys.exc_info()
            # Get locals in the wrapped function
            util_dbg.printex(ex, tb=True)
            printTEST('[TEST.FINISH] %s -- FAILED: %s %s' % (funcname, type(ex), ex))
            if func_is_text:
                #ut.embed()
                print('Failed in module: %r' % frame_fpath)
                print(ut.msgblock('FAILED DOCTEST IN %s' % (funcname,), src))
                #failed_execline = traceback.format_tb(tb)[-1]
                #parse_str = 'File {fname}, line {lineno}, in {modname}'
                #parse_dict = parse.parse('{prefix_}' + parse_str + '{suffix_}', failed_execline)
                #if parse_dict['fname'] == '<string>':
                #    lineno = int(parse_dict['lineno'])
                #    failed_line = src.splitlines()[lineno - 1]
                #    print('Failed on line: %s' % failed_line)
            print(SAD_FACE)
            raise
            if util_arg.STRICT:
                # Remove this function from stack strace
                exc_type, exc_value, exc_traceback = sys.exc_info()
                exc_traceback = exc_traceback.tb_next
                # Python 2*3=6
                six.reraise(exc_type, exc_value, exc_traceback)
                # PYTHON 2.7 DEPRICATED:
                #raise exc_type, exc_value, exc_traceback.tb_next
                # PYTHON 3.3 NEW METHODS
                #ex = exc_type(exc_value)
                #ex.__traceback__ = exc_traceback.tb_next
                #raise ex
            return False


def printTEST(msg, wait=False):
    builtins.print('\n=============================')
    builtins.print('**' + msg)
    #if INTERACTIVE and wait:
    # raw_input('press enter to continue')


def tryimport(modname, pipiname):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tests import *   # NOQA
        >>> modname = 'pyfiglet'
        >>> pipiname = 'git+https://github.com/pwaller/pyfiglet'
        >>> pyfiglet = tryimport(modname, pipiname)
        >>> assert pyfiglet is None or isinstance(pyfiglet, types.ModuleType)
    """
    try:
        module = __import__(modname)
        return module
    except ImportError as ex:
        import utool
        if not utool.WIN32:
            pipcmd = 'sudo pip install %s' % pipiname
        else:
            pipcmd = 'pip install %s' % pipiname
        msg = 'unable to find module %r. Please install: %s' % (str(modname), str(pipcmd))
        utool.printex(ex, msg, iswarning=True)
        return None


def bubbletext(text, font='cybermedium'):
    r"""
    Other fonts include: cybersmall, cybermedium, and cyberlarge

    import pyfiglet

    References:
        http://www.figlet.org/

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> bubble_text1 = utool.bubbletext('TESTING', font='cyberlarge')
        >>> bubble_text2 = utool.bubbletext('BUBBLE', font='cybermedium')
        >>> bubble_text3 = utool.bubbletext('TEXT', font='cyberlarge')
        >>> print('\n'.join([bubble_text1, bubble_text2, bubble_text3]))
    """
    pyfiglet = tryimport('pyfiglet', 'git+https://github.com/pwaller/pyfiglet')
    if pyfiglet is None:
        return text
    else:
        bubble_text = pyfiglet.figlet_format(text, font=font)
        return bubble_text


def make_run_tests_script_text(test_headers, test_argvs, quick_tests=None,
                               repodir=None, exclude_list=[]):
    """
    Autogeneration function

    Examples:
        >>> from utool.util_tests import *  # NOQA
        >>> import utool
        >>> testdirs = ['~/code/ibeis/test_ibs*.py']
    """
    import utool
    from os.path import relpath, join, dirname  # NOQA

    exclude_list += ['__init__.py']

    # General format of the testing script

    script_fmtstr = utool.codeblock(
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
            PRINT_DELIMETER
            num_passed=$(($num_passed + (1 - $RETURN_CODE)))
            num_ran=$(($num_ran + 1))
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
                printf "$FAILED_TESTS\n" >> failed.txt
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

    testcmdline_fmtstr = utool.codeblock(
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

    header_test_block_fmstr = utool.codeblock(
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

    known_tests = utool.ddict(list)

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

        #utool.ls(dpath)

        # VERY HACK BIT OF CODE

        # Get list of tests from patterns
        if testcmds is None:
            if modname is not None:
                module = __import__(modname)
                repo_path = dirname(dirname(module.__file__))
            else:
                repo_path = repodir
            dpath_ = utool.unixpath(util_path.unixjoin(repo_path, dpath))

            if header_upper == 'OTHER':
                # Hacky way to grab any other tests not explicitly seen in this directory
                _testfpath_list = list(set(utool.glob(dpath_, '*.py')) - set(known_tests[dpath_]))
                #_testfpath_list = utool.glob(dpath_, '*.py')
                #set(known_tests[dpath_])
            else:
                _testfpath_list = utool.flatten([utool.glob(dpath_, pat) for pat in pats])

            def not_excluded(x):
                return not any([x.find(exclude) > -1 for exclude in exclude_list])

            _testfpath_list = list(filter(not_excluded, _testfpath_list))

            known_tests[dpath_].extend(_testfpath_list)
            #print(_testfpath_list)
            testfpath_list = [util_path.unixjoin(dpath, relpath(fpath, dpath_)) for fpath in _testfpath_list]

            testline_list = [format_testline(fpath, dirvar) for fpath in testfpath_list]
        else:
            testline_list = testcmds

        testlines_block = utool.indentjoin(testline_list).strip('\n')

        # Construct test block for this type
        header_text = header_upper + ' TESTS'
        headerfont = 'cybermedium'
        header_bubble_text =  utool.indent(utool.bubbletext(header_text, headerfont).strip())
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

    runtests_bubbletext = bubbletext('RUN TESTS', 'cyberlarge')

    test_block = '\n'.join(header_test_block_list)
    dirdef_block = '\n'.join(dirdef_list)
    testdefault_block = utool.indent('\n'.join(default_flag_line_list))
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


def find_doctestable_modnames(dpath_list=None, exclude_doctests_fnames=[], exclude_dirs=[]):
    import utool as ut
    fpath_list, lines_list, lxs_list = ut.grep('doctest_funcs',
                                               dpath_list=dpath_list,
                                               include_patterns=['*.py'],
                                               exclude_dirs=exclude_dirs,
                                               recursive=True)
    exclude_doctests_fnames = set(exclude_doctests_fnames)
    def is_not_excluded(fpath):
        return basename(fpath) not in exclude_doctests_fnames
    doctest_modpath_list = list(filter(is_not_excluded, fpath_list))
    doctest_modname_list = list(map(ut.get_modname_from_modpath, doctest_modpath_list))
    return doctest_modname_list


def find_untested_modpaths(dpath_list=None, exclude_doctests_fnames=[], exclude_dirs=[]):
    import utool as ut
    fpath_list, lines_list, lxs_list = ut.grep('>>> # ENABLE_DOCTEST',
                                               dpath_list=dpath_list,
                                               include_patterns=['*.py'],
                                               exclude_dirs=exclude_dirs,
                                               recursive=True,
                                               inverse=True)
    exclude_doctests_fnames = set(list(exclude_doctests_fnames) + ['__init__.py'])
    def is_not_excluded(fpath):
        fname = basename(fpath)
        return (not fname.startswith('_')) and fname not in exclude_doctests_fnames
    doctest_modpath_list = list(filter(is_not_excluded, fpath_list))
    #doctest_modname_list = list(map(ut.get_modname_from_modpath, doctest_modpath_list))
    return doctest_modpath_list


def def_test(header, pat=None, dpath=None, modname=None, default=False, testcmds=None):
    """ interface to make test tuple """
    return (header, default, modname, dpath, pat, testcmds)


if __name__ == '__main__':
    """
    python utool/util_tests.py
    python -c "import utool; utool.doctest_funcs(module=utool.util_tests, needs_enable=False)"
    /model/preproc/preproc_chip.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    doctest_funcs()
