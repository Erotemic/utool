""" Helpers for tests """
from __future__ import absolute_import, division, print_function
from six.moves import builtins
import sys
from . import util_print
from . import util_dbg
from . import util_arg
from . import util_time
from .util_inject import inject
from utool._internal.meta_util_six import get_funcname
print, print_, printDBG, rrr, profile = inject(__name__, '[tests]')


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


def run_test(func, *args, **kwargs):
    """
    Runs the test function with success / failure printing

    Input:
        Anything that needs to be passed to <func>
    """
    upper_funcname = get_funcname(func).upper()
    with util_print.Indenter('[' + upper_funcname + ']'):
        try:
            import utool
            if utool.VERBOSE:
                printTEST('[TEST.BEGIN] %s ' % (sys.executable))
                printTEST('[TEST.BEGIN] %s ' % (get_funcname(func),))
            with util_time.Timer(upper_funcname) as timer:
                test_locals = func(*args, **kwargs)
                # Write timings
            printTEST('[TEST.FINISH] %s -- SUCCESS' % (get_funcname(func),))
            print(HAPPY_FACE)
            with open('_test_times.txt', 'a') as file_:
                msg = '%.4fs in %s\n' % (timer.ellapsed, upper_funcname)
                file_.write(msg)
            return test_locals
        except Exception as ex:
            # Get locals in the wrapped function
            util_dbg.printex(ex)
            exc_type, exc_value, tb = sys.exc_info()
            printTEST('[TEST.FINISH] %s -- FAILED: %s %s' % (get_funcname(func), type(ex), ex))
            print(SAD_FACE)
            if util_arg.STRICT:
                # Remove this function from stack strace
                exc_type, exc_value, exc_traceback = sys.exc_info()
                exc_traceback = exc_traceback.tb_next
                # Python 2*3=6
                import six
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
        >>> modname = 'pyfiglet'
        >>> pipiname = 'git+https://github.com/pwaller/pyfiglet'
        >>> pyfiglet = tryimport(modname, pipiname)
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
        msg = 'unable to find module %r. Please install: %s' (modname, pipcmd)
        utool.printex(ex, msg, iswarning=True)


def bubbletext(text, font='cybermedium'):
    """
    Other fonts include: cybersmall, cybermedium, and cyberlarge

    Example:
        >>> import utool
        >>> text = 'RUN TESTS'
        >>> font='cyberlarge'
        >>> bubble_text = utool.bubbletext(text, font='cyberlarge')
        >>> print(bubble_text)
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
    from os.path import relpath, join, dirname

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
        dpath_ = utool.truepath(join(repo_path, dpath))
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
        testfpath_list = [join(dpath, relpath(fpath, dpath_)) for fpath in _testfpath_list]

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


def autogen_ibeis_runtest():
    """ special case to generate tests script for IBEIS

    Example:
        >>> import utool
        >>> test_script = utool.autogen_ibeis_runtest()

    CommandLine:
        python -c "import utool; utool.autogen_ibeis_runtest()"
        python -c "import utool; print(utool.autogen_ibeis_runtest())"

        python -c "import utool; print(utool.autogen_ibeis_runtest())" > _run_tests2.sh
        chmod +x _run_tests2.sh

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

    repodir = '~/code/ibeis'
    testdir = 'ibeis/tests'

    exclude_list = [
    ]

    test_headers = [
        # title, default, module, testpattern
        def_test('VTOOL',  dpath='vtool/tests', pat=['test*.py'], modname='vtool'),
        def_test('GUI',    dpath=testdir, pat=['test_gui*.py']),
        def_test('IBEIS',  dpath=testdir, pat=['test_ibs*.py', 'test_delete*.py'], default=True),
        def_test('SQL',    dpath=testdir, pat=['test_sql*.py']),
        def_test('VIEW',   dpath=testdir, pat=['test_view*.py']),
        def_test('MISC',   dpath=testdir, pat=misc_pats),
        def_test('OTHER',  dpath=testdir, pat='OTHER'),
        def_test('HESAFF', dpath='pyhesaff/tests', pat=['test_*.py'], modname='pyhesaff'),
    ]

    script_text = autogen_run_tests(test_headers, test_argvs, quick_tests, repodir, exclude_list)
    #print(script_text)
    return script_text
