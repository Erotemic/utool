""" Helpers for tests """
from __future__ import absolute_import, division, print_function
import six
from six.moves import builtins
import inspect
import types
import sys
from . import util_print
from . import util_dbg
from . import util_arg
from . import util_path
from . import util_time
from .util_inject import inject
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


def get_doctest_examples(func_or_class):
    """
    get_doctest_examples

    Args:
        func_or_class (function)

    Example:
        >>> from utool.util_tests import *  # NOQA
        >>> func_or_class = get_doctest_examples
        >>> result = get_doctest_examples(func_or_class)
        >>> print(result)
    """
    if VERBOSE_TEST:
        print('[util_test] parsing %r for doctest' % (func_or_class))
    import doctest
    try:
        docstr = func_or_class.func_doc
    except AttributeError:
        docstr = func_or_class.__doc__
    import textwrap
    docstr = textwrap.dedent(docstr)

    try:
        comment_iter = doctest.DocTestParser().parse(docstr)
    except Exception as ex:
        import utool as ut
        ut.printex(ex, 'error parsing:\n%s\nin function %r' % (docstr, func_or_class))
        raise
    current_example_lines = []
    example_list = []
    def append_current_example():
        # next example defined if any examples lines have been found
        if len(current_example_lines) >= 1:
            example_list.append(''.join(current_example_lines))
    # Loop over doctest lines
    for c in comment_iter:
        if isinstance(c, doctest.Example):
            current_example_lines.append(c.source)
        elif c == '':
            pass
        else:
            append_current_example()
            current_example_lines = []
    append_current_example()
    return example_list


def doctest_funcs(testable_list=[], check_flags=True, module=None, allexamples=None,
                  needs_enable=None):
    """
    Main entry point into utools main module doctest harness

    CommandLine:
        python -c "import utool; utool.doctest_funcs(module=utool.util_tests, needs_enable=False)"

    """
    import multiprocessing
    import utool
    import utool as ut
    multiprocessing.freeze_support()
    if needs_enable is None:
        needs_enable = not ut.get_argflag('--enableall')
    else:
        needs_enable = True
    TEST_ALL_EXAMPLES = allexamples or ut.get_argflag(('--allexamples', '--test-all-examples', '--testall', '--test-all'))

    print('[utool] Running doctest funcs')

    testable_name_list = []

    if isinstance(testable_list, types.ModuleType):
        module = testable_list
        testable_list = []

    frame_fpath = '???' if module is None else str(module)

    # Inspect module for testable names
    if module is None:
        try:
            frame = ut.get_caller_stack_frame(N=0)
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
        allexamples = True

    try:
        if VERBOSE_TEST:
            print('[util_test] Iterating over module funcs')

        #source = inspect.getsource(module)
        for key, val in ut.iter_module_funcs(module):
            docstr = inspect.getdoc(val)
            test_sentinals = [
                'ENABLE_DOCTEST',
                #'ENABLE_TEST',
                #'ENABLE_DOCTEST',
                #'ENABLE_UTOOL_DOCTEST',
                #'UTOOL_TEST',
                #'UTOOLTEST'
            ]
            if docstr is not None:
                #print('Inspect func ' + key)
                #print(key)
                docstr_upper = docstr.upper()
                test_enabled = any([docstr_upper.find(s) >= 0 for s in test_sentinals])
                if test_enabled or (not needs_enable):
                    testable_name_list.append(key)
                    testable_list.append(val)
                else:
                    if docstr.find('Example') >= 0:
                        print('[util_dev] DOCTEST DISABLED: %s' % key)
                        #print(docstr)
    except Exception as ex:
        ut.printex(ex, keys=['frame'])
        raise

    for val in testable_list:
        print('[util_dev] DOCTEST ENABLED: %s' % val)

    #ut.embed()
    wastested_list = []
    sorted_testnames = []
    sorted_testable = sorted(list(set(testable_list)), key=_get_testable_name)

    nPass = 0
    nTotal = 0

    subx = ut.get_argval('--subx', type_=int, default=None, help_='Only tests the subxth example')
    for testable in sorted_testable:
        # HACKy
        key = _get_testable_name(testable)
        sorted_testnames.append(key)
        flag1 = '--test-' + key.replace('_', '-')
        flag2 = '--test-' + key
        specific_test_flag = utool.get_argflag((flag1, flag2))
        if TEST_ALL_EXAMPLES or not check_flags or specific_test_flag:
            print('[utool] Doctest requested: %r' % key)
            examples = get_doctest_examples(testable)
            nExamples = len(examples)
            if nExamples == 0:
                print('WARNING: no examples for key=%r' % key)
                wastested_list.append(False)
            else:
                print('\n\n ---- TEST ' + key.upper() + '---')
                #with ut.Indenter('[TEST.%s]' % key):
                if subx is not None:
                    examples = examples[subx:subx + 1]
                nTotal += nExamples
                for testno , src in enumerate(examples):
                    src = ut.regex_replace('from __future__ import.*$', '', src)
                    print('\n Test #%d' % testno)
                    print(ut.msgblock('EXEC SRC', src))
                    # --- EXEC STATMENT ---
                    test_globals = module.__dict__.copy()
                    test_locals = ut.run_test((key,  src), globals=test_globals)
                    nPass += (test_locals is not False)
                    #exec(src)
                wastested_list.append(True)
        else:
            wastested_list.append(False)
    if not any(wastested_list):
        print('No test flags sepcified. Please choose one of the following flags')
        print('Valid test argflags:\n' + '    --allexamples' + utool.indentjoin(sorted_testnames, '\n    --test-'))
    print('+-------')
    print('| finished testing fpath=%r' % (frame_fpath,))
    print('| passed %d / %d' % (nPass, nTotal))
    print('L-------')
    return (nPass, nTotal)


def run_test(func, *args, **kwargs):
    """
    Runs the test function with success / failure printing

    Input:
        Anything that needs to be passed to <func>
    """
    func_is_text = isinstance(func, types.TupleType)
    if func_is_text:
        (key, src) = func
        funcname = key.upper()
    else:
        funcname = get_funcname(func)
    upper_funcname = funcname.upper()
    with util_print.Indenter('[' + upper_funcname + ']'):
        try:
            import utool as ut
            if ut.VERBOSE:
                printTEST('[TEST.BEGIN] %s ' % (sys.executable))
                printTEST('[TEST.BEGIN] %s ' % (funcname,))
            with util_time.Timer(upper_funcname) as timer:
                if func_is_text:
                    test_locals = {}
                    test_globals = kwargs.get('globals', {})
                    exec(src, test_globals, test_locals)
                else:
                    test_locals = func(*args, **kwargs)
                # Write timings
            printTEST('[TEST.FINISH] %s -- SUCCESS' % (funcname,))
            print(HAPPY_FACE)
            with open('_test_times.txt', 'a') as file_:
                msg = '%.4fs in %s\n' % (timer.ellapsed, upper_funcname)
                file_.write(msg)
            return test_locals
        except Exception as ex:
            # Get locals in the wrapped function
            util_dbg.printex(ex, tb=True)
            exc_type, exc_value, tb = sys.exc_info()
            printTEST('[TEST.FINISH] %s -- FAILED: %s %s' % (funcname, type(ex), ex))
            if func_is_text:
                import utool as ut
                print(ut.msgblock('FAILED DOCTEST IN %s' % (key,), src))
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
    """
    Other fonts include: cybersmall, cybermedium, and cyberlarge

    Example:
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


def autogen_run_tests(test_headers, test_argvs, quick_tests=None, repodir=None,
                      exclude_list=[]):
    """
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
    for header, default, modname, dpath, pats in test_headers:
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
        if modname is not None:
            module = __import__(modname)
            repo_path = dirname(dirname(module.__file__))
        else:
            repo_path = repodir
        #print(repo_path)
        dpath_ = utool.unixpath(util_path.unixjoin(repo_path, dpath))
        #print(dpath_)
        #print(pats)

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


def def_test(header, pat=None, dpath=None, modname=None, default=False):
    """ interface to make test tuple """
    return (header, default, modname, dpath, pat)


if __name__ == '__main__':
    """
    python utool/util_tests.py
    python -c "import utool; utool.doctest_funcs(module=utool.util_tests, needs_enable=False)"
    /model/preproc/preproc_chip.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    doctest_funcs()
