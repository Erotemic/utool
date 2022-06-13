# -*- coding: utf-8 -*-
"""
Helpers for tests

NOTE:
    DEPRICATE THIS IN FAVOR OF pytest and xdoctest

This module contains a more sane reimplementation of doctest functionality.
(I.E.  asserts work and you don't have to worry about stdout mucking things up)
The code isn't super clean though due to time constriaints.  Many functions
probably belong elsewhere and the parsers need a big cleanup.

TODO:
    * report the line of the doctest in the file when reporting errors as well as
     the relative line

    * restructure so there is a test collection step, a filtering step, and an
      execution step

    * Fix finding tests when running with @profile
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import inspect
import types
import sys
from collections import namedtuple
from os.path import basename
from utool import util_arg
from utool import util_time
from utool import util_inject
from utool import util_dbg
from utool import util_dev
from utool._internal.meta_util_six import get_funcname
print, rrr, profile = util_inject.inject2(__name__)


VERBOSE_TEST = util_arg.get_module_verbosity_flags('test')[0]

DEBUG_SRC = not util_arg.get_argflag('--nodbgsrc')
PRINT_SRC = util_arg.get_argflag(
    ('--printsrc', '--src', '--show-src', '--showsrc'),
    help_='show docstring source when running tests')
PRINT_FACE = util_arg.get_argflag(('--printface', '--face'))
BIGFACE = util_arg.get_argflag('--bigface')
SYSEXIT_ON_FAIL = util_arg.get_argflag(
    ('--sysexitonfail', '--fastfail'),
    help_='Force testing harness to exit on first test failure')
VERBOSE_TIMER = not util_arg.get_argflag('--no-time-tests')
INDENT_TEST   = False

ModuleDoctestTup = namedtuple('ModuleDoctestTup', ('enabled_testtup_list',
                                                   'frame_fpath',
                                                   'all_testflags', 'module'))


class TestTuple(util_dev.NiceRepr):
    """
    Simple container for test objects to replace old tuple format
    exec mode specifies if the test is being run as a script
    """
    def __init__(self, name, num, src, want, flag, tags=None, frame_fpath=None,
                 mode=None, nametup=None, test_namespace=None, shortname=None,
                 total=None):
        self._name = name  # function / class / testable name
        self.num = num    # doctest index
        self.src = src    # doctest src
        self.want = want  # doctest required result (optional)
        self.flag = flag  # doctest commandline flags
        self.frame_fpath = frame_fpath  # parent file fpath
        self.mode = mode      # flags if running as script
        self.nametup = nametup
        self.total = total
        self.test_namespace = test_namespace
        self.shortname = shortname

        if tags is None and src is not None:
            # hack to parse tag from source
            tagline = src.split('\n')[0].strip()
            if tagline.startswith('#'):
                tagline = tagline[1:].strip()
                tagline = tagline.replace('_DOCTEST', '')
                tags = tagline.split(',')
        self.tags = tags

    @property
    def name(self):
        # Hack for namespaced name
        return self.nametup[0]

    @property
    def full_name(self):
        return self.modname + '.' + self.name

    @property
    def modname(self):
        import utool as ut
        return ut.get_modname_from_modpath(self.frame_fpath)

    @property
    def namespace_levels(self):
        if self.test_namespace:
            return [self.modename, self.test_namespace]
        else:
            return [self.modename]

    @property
    def namespace(self):
        return '.'.join(self.namespace_levels)

    @property
    def exec_mode(self):
        return self.mode == 'exec'

    def __nice__(self):
        tagstr = ' ' + ','.join(self.tags) if self.tags is not None else ''
        return (' '  + self.name + ':' + str(self.num) + ' /' +
                str(self.total) + ' ' + tagstr + ' in ' + self.modname)


HAPPY_FACE_BIG = r'''
               .-""""""-.
             .'          '.
            /   O      O   \
           :                :
           |                |
           '  ,          ,' :
            \  '-......-'  /
             '.          .'
               '-......-'
                   '''

SAD_FACE_BIG = r'''
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

HAPPY_FACE_SMALL = r'''
     .""".
    | o o |
    | \_/ |
     ' = '
    '''

SAD_FACE_SMALL = r'''
     .""".
    | . . |
    |  ~  |
     ' = '
    '''

if BIGFACE:
    HAPPY_FACE = HAPPY_FACE_BIG
    SAD_FACE = SAD_FACE_BIG
else:
    HAPPY_FACE = HAPPY_FACE_SMALL
    #SAD_FACE = SAD_FACE_BIG
    SAD_FACE = SAD_FACE_SMALL


def get_package_testables(module=None, **tagkw):
    r"""
    New command that should eventually be used intead of old stuff?

    Args:
        module_list (list): (default = None)
        test_flags (None): (default = None)

    CommandLine:
        python -m utool get_package_testables --show --mod ibeis
        python -m utool get_package_testables --show --mod utool --tags SCRIPT
        python -m utool get_package_testables --show --mod utool --tags ENABLE

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> import utool as ut
        >>> has_any = ut.get_argval('--tags', type_=str, default=None)
        >>> module = ut.get_argval('--mod', default='utool')
        >>> test_tuples = get_package_testables(module, has_any=has_any)
        >>> result = ut.repr3(test_tuples)
        >>> print(result)
        >>> #print(ut.repr3(ut.list_getattr(test_tuples, 'tags')))
    """
    import utool as ut
    if isinstance(module, six.string_types):
        module = ut.import_modname(module)
    modname_list = ut.package_contents(module, ignore_prefix=[], ignore_suffix=[])
    module_list = []
    for modname in modname_list:
        try:
            module_list.append(ut.import_modname(modname))
        except Exception:
            pass

    test_tuples = []
    for module in module_list:
        old_testables = ut.get_module_doctest_tup(module=module,
                                                  needs_enable=False,
                                                  allexamples=True, verbose=False)
        test_tuples.extend(old_testables.enabled_testtup_list)
    if tagkw:
        tags_list = ut.list_getattr(test_tuples, 'tags')
        flags = ut.filterflags_general_tags(tags_list, **tagkw)
        test_tuples = ut.compress(test_tuples, flags)
    return test_tuples


def doctest_module_list(module_list):
    """
    Runs many module tests

    Entry point for batch run
    Depth 0)

    Ignore:
        :'<,'>!sort -n -k 2
    """
    import utool as ut
    nPass_list = []
    nTotal_list = []
    failed_cmds_list = []
    error_reports_list = []
    print('[util_test] Running doctests on module list')

    try:
        ut.write_to('timeings.txt', '\n\n --- begining doctest_module_list\n', mode='a')
    except IOError as ex:
        ut.printex(ex, '[util_test] IOWarning', iswarning=True)

    failed_doctest_fname = 'failed_doctests.txt'
    seen_ = set([])
    with open(failed_doctest_fname, 'a') as file_:
        file_.write('\n-------\n\n')
        file_.write(ut.get_timestamp(format_='printable') + '\n')
        file_.write('logfile (only present if logging) = %r\n' %
                    (ut.util_logging.get_current_log_fpath(),))
        testkw = dict(allexamples=True)
        with ut.Timer(verbose=False) as t:
            for module in module_list:
                (nPass, nTotal, failed_list, error_report_list) = ut.doctest_funcs(
                    module=module, seen_=seen_, **testkw)
                nPass_list.append(nPass)
                nTotal_list.append(nTotal)
                failed_cmds_list.append(failed_list)
                error_reports_list.append(error_report_list)
                # Write failed tests to disk
                for cmd in failed_list:
                    file_.write(cmd + '\n')
        total_time = t.ellapsed
        nPass = sum(nPass_list)
        nTotal = sum(nTotal_list)
        file_.write('PASSED %d / %d' % (nPass, nTotal))

    failed_cmd_list = ut.flatten(failed_cmds_list)
    error_report_list = ut.filter_Nones(ut.flatten(error_reports_list))
    if len(error_report_list) > 0:
        print('\nPrinting %d error reports' % (len(error_report_list),))
        for count, error_report in enumerate(error_report_list):
            print('\n=== Error Report %d / %d' % (count, len(error_report_list)))
            print(error_report)
        print('--- Done printing error reports ----')

    try:
        ut.write_to('timeings.txt', '\n\n --- finished doctest_module_list total_time=%.3fs\n' % (total_time), mode='a')
    except IOError as ex:
        ut.printex(ex, '[util_test] IOWarning', iswarning=True)

    print('')
    print('+========')
    print('| FINISHED TESTING %d MODULES' % (len(module_list),))
    print('| PASSED %d / %d' % (nPass, nTotal))
    print('L========')
    if len(failed_cmd_list) > 0:
        print('FAILED TESTS:')
        print('\n'.join(failed_cmd_list))
    return nPass, nTotal, failed_cmd_list


def doctest_funcs(testable_list=None, check_flags=True, module=None,
                  allexamples=None, needs_enable=None, strict=False,
                  verbose=True, return_error_report=True, seen_=None):
    r"""
    Main entry point into utools main module doctest harness
    Imports a module and checks flags for the function to run
    Depth 1)

    Args:
        testable_list (list):
        check_flags (bool): Force checking of the --test- and --exec- flags
        module (None):
        allexamples (None):
        needs_enable (None):

    Returns:
        tuple: (nPass, nTotal, failed_cmd_list)

    CommandLine:
        python -m ibeis.algo.preproc.preproc_chip --all-examples

    References:
        http://legacy.python.org/dev/peps/pep-0338/
        https://docs.python.org/2/library/runpy.html

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> testable_list = []
        >>> check_flags = True
        >>> module = None
        >>> allexamples = None
        >>> needs_enable = None
        >>> # careful might infinitely recurse
        >>> (nPass, nTotal) = doctest_funcs(testable_list, check_flags, module,
        ...                                 allexamples, needs_enable)
        >>> print((nPass, nTotal))
    """
    import multiprocessing
    import utool as ut  # NOQA
    #ut.start_logging()
    multiprocessing.freeze_support()  # just in case
    if ut.VERBOSE:
        print('[util_test] doctest_funcs')
    ut.inject_colored_exceptions()

    if (verbose or VERBOSE_TEST) and ut.NOT_QUIET:
        if VERBOSE_TEST:
            print('[util_test.doctest_funcs][DEPTH 1] doctest_funcs()')
        print('[util_test.doctest_funcs] Running doctest_funcs')
    if ut.is_developer():
        ut.change_term_title('DocTest ' + ' '.join(sys.argv))

    # PARSE OUT TESTABLE DOCTESTTUPS
    mod_doctest_tup = get_module_doctest_tup(
        testable_list, check_flags, module, allexamples, needs_enable, N=1,
        verbose=verbose)
    enabled_testtup_list, frame_fpath, all_testflags, module = mod_doctest_tup

    nPass = 0
    nFail = 0
    nTotal = len(enabled_testtup_list)

    #flags = [(tup.name, tup.num) in seen_ for tup in enabled_testtup_list]
    if seen_ is not None:
        flags = [tup.src not in seen_ for tup in enabled_testtup_list]
        enabled_testtup_list = ut.compress(enabled_testtup_list, flags)

    # Remove duplicate tests from previous parts of the batch run
    #print(sum(flags))

    EARLYEXIT = False
    if seen_ is not None:
        for tup in enabled_testtup_list:
            #seen_.add((tup.name, tup.num))
            seen_.add(tup.src)
    if EARLYEXIT:
        nPass = nTotal - sum(flags)
        if return_error_report:
            return (nPass, nTotal, [], [])
        else:
            return (nPass, nTotal, [])

    modname = ut.get_modname_from_modpath(frame_fpath)
    nTotal = len(enabled_testtup_list)

    # Run enabled examles
    failed_flag_list = []
    error_report_list = []
    if ut.get_argflag(('--edit-test-file', '--etf')):
        ut.editfile(frame_fpath)
    exec_mode = all([testtup.exec_mode for testtup in enabled_testtup_list])

    for testtup in enabled_testtup_list:
        name = testtup.name
        num  = testtup.num
        src  = testtup.src
        want = testtup.want
        flag = testtup.flag
        #if ut.is_developer():
        #    ut.change_term_title('DocTest ' + modname + ' ' + name)
        print('\n')
        fmtdict = dict(modname=modname, name=name, num=num)
        #      1          v12     v20       v30       v40       v50         v62
        print('+------------------------------------------------------------+')
        print('*  DOCTEST {modname:<20} {name:>26}:{num:d} '.format(**fmtdict))
        print('+------------------------------------------------------------+')

        if PRINT_SRC or VERBOSE_TEST:
            if ut.is_developer():
                print(ut.msgblock('EXEC SRC', ut.highlight_code(src), side='>>>'))
            else:
                print(ut.msgblock('EXEC SRC', src, side='>>>'))
        # Commented because it caused differences between
        # individual test runs and large test runs with ut
        # being imported
        # test_globals = module.__dict__.copy()
        test_globals = {}
        error_report = None
        try:
            testkw = dict(
                globals=test_globals,  # HACK
                want=want)
            assert testtup.frame_fpath == frame_fpath
            test_locals, error_report = ut.run_test(testtup, **testkw)
            pass_flag = (test_locals is not False)
            if pass_flag:
                if VERBOSE_TEST:
                    print('seems to pass')
                nPass += 1
            else:
                if VERBOSE_TEST:
                    print('raising failed exception')
                raise Exception('failed')
        except Exception:
            if VERBOSE_TEST:
                print('Seems to fail. ')
            nFail += 1
            failed_flag_list.append(flag)
            error_report_list.append(error_report)
            if strict or util_arg.SUPER_STRICT:
                raise
            else:
                if VERBOSE_TEST:
                    print('Silently Failing: '
                          'maybe adding the --super-strict flag would help debug?')
            pass
        except KeyboardInterrupt:
            print('[util_test] caught Ctrl+C')
        print('L_____________________________________________________________')
    #L__________________
    #+-------------------
    # Print Results
    if nTotal == 0 and not allexamples:
        valid_test_argflags = ['--allexamples'] + all_testflags
        modname = ut.get_modname_from_modpath(frame_fpath)
        # TODO: ensure that exename is in the PATH
        exename = basename(sys.executable)
        warning_msg = ut.codeblock(
            r'''
            [ut.doctest_funcs] No test flags specified
            Please choose one of the following flags or specify --enableall
            Valid test argflags:
            ''') + ut.indentjoin(valid_test_argflags, '\n %s -m %s ' % (exename, modname))
        # warning_msg = ut.indent(warning_msg, '[util_test.doctest_funcs]')
        ut.colorprint(warning_msg, 'yellow')

    if not exec_mode:
        print('+-------')
        print('| finished testing fpath=%r' % (frame_fpath,))
        print('| passed %d / %d' % (nPass, nTotal))
        print('L-------')
    failed_cmd_list = []
    if nFail > 0:
        #modname = module.__name__
        modname = ut.get_modname_from_modpath(frame_fpath)
        # TODO: ensure that exename is in the PATH
        exename = basename(sys.executable)
        failed_cmd_list = ['%s -m %s %s' % (exename, modname, flag_)
                            for flag_ in failed_flag_list]
        #failed_cmd_list = ['python %s %s' % (frame_fpath, flag_)
        #                    for flag_ in failed_flag_list]
        print('Failed sys.argv = %r' % (' '.join(sys.argv),))
        print('Failed Tests:')
        print('\n'.join(failed_cmd_list))
    #L__________________

    if ut.util_inject.PROFILING:
        ut.dump_profile_text()

    if return_error_report:
        return (nPass, nTotal, failed_cmd_list, error_report_list)
    else:
        return (nPass, nTotal, failed_cmd_list)


def run_test(func_or_testtup, *args, **kwargs):
    r"""
    Runs the test function. Prints a success / failure report.

    Args:
        func_or_testtup (func or tuple): function or doctest tuple
        args*: args to be forwarded to `func_or_testtup`
        kwargs*: keyword args to be forwarded to `func_or_testtup`
    """
    import utool as ut
    #func_is_testtup = isinstance(func_or_testtup, tuple)
    # NOTE: isinstance might not work here if ut.rrrr has been called
    func_is_testtup = isinstance(func_or_testtup, TestTuple)
    exec_mode = False
    dump_mode = False
    write_times = True
    if func_is_testtup:
        testtup = func_or_testtup
        src         = testtup.src
        funcname    = testtup.name
        frame_fpath = testtup.frame_fpath
        #(funcname, src, frame_fpath) = func_or_testtup
        exec_mode = testtup.exec_mode
        dump_mode = testtup.mode == 'dump'
    else:
        func_ = func_or_testtup
        funcname = get_funcname(func_)
        frame_fpath = ut.get_funcfpath(func_)
    upper_funcname = funcname.upper()
    if ut.VERBOSE:
        print('\n=============================')
        print('**[TEST.BEGIN] %s ' % (sys.executable))
        print('**[TEST.BEGIN] %s ' % (funcname,))

    verbose_timer = not exec_mode and VERBOSE_TIMER
    nocheckwant = True if exec_mode else None
    print_face = not exec_mode and PRINT_FACE
    error_report = None

    if dump_mode:
        print('testtup = %r' % (testtup,))
        print(ut.highlight_code(src))
        return None, None

    try:
        # RUN THE TEST WITH A TIMER
        with util_time.Timer(upper_funcname, verbose=verbose_timer) as timer:
            if func_is_testtup:
                test_locals = _exec_doctest(src, kwargs, nocheckwant)
            else:
                # TEST INPUT IS A LIVE PYTHON FUNCTION
                test_locals = func_(*args, **kwargs)
    except Exception as ex:
        import utool as ut
        # Get locals in the wrapped function
        ut.printex(ex, tb=True)
        error_report_lines = ['**[TEST.ERROR] %s -- FAILED:\n    type(ex)=%s' % (
            funcname, type(ex))]
        error_report_lines.append(ut.formatex(ex, tb=True))
        def print_report(msg):
            error_report_lines.append(msg)
            print(msg)
        print_report('\n=============================')
        print_report('**[TEST.FINISH] %s -- FAILED:\n    type(ex)=%s' % (funcname, type(ex)))
        exc_type, exc_value, tb = sys.exc_info()
        if PRINT_FACE:
            print_report(SAD_FACE)
        if func_is_testtup:
            print_report('Failed in module: %r' % frame_fpath)
            src_with_lineno = ut.number_text_lines(src)
            print_report(ut.msgblock('FAILED DOCTEST IN %s' % (funcname,), src_with_lineno))

        if util_arg.SUPER_STRICT:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if not func_is_testtup:
                # Remove this function from stack strace
                # dont do this for execed code
                exc_traceback = exc_traceback.tb_next
                pass
            # FIXME: make a _internal/py2_syntax_funcs version of this
            # function as well as a python3 syntax version due to the
            # reraise syntax issue to avoid an extra frame in the stack
            six.reraise(exc_type, exc_value, exc_traceback)
        if SYSEXIT_ON_FAIL:
            print('[util_test] SYSEXIT_ON_FAIL = True')
            print('[util_test] exiting with sys.exit(1)')
            sys.exit(ut.EXIT_FAILURE)
        #raise
        error_report = '\n'.join(error_report_lines)
        return False, error_report
    else:
        # LOG PASSING TEST
        if not exec_mode:
            print('\n=============================')
            print('**[TEST.FINISH] %s -- SUCCESS' % (funcname,))
            if print_face:
                print(HAPPY_FACE)
            if write_times:
                timemsg = '%.4fs in %s %s\n' % (
                    timer.ellapsed, funcname, frame_fpath)
                try:
                    ut.write_to('timeings.txt', timemsg, mode='a')
                except IOError as ex:
                    ut.printex(ex, '[util_test] IOWarning', iswarning=True)
        # RETURN VALID TEST LOCALS
        return test_locals, error_report


def _exec_doctest(src, kwargs, nocheckwant=None):
    """
    Helper for run_test

    block of code that runs doctest and was too big to be in run_test
    """
    # TEST INPUT IS PYTHON CODE TEXT
    #test_locals = {}
    test_globals = kwargs.get('globals', {}).copy()
    want = kwargs.get('want', None)
    #test_globals['print'] = doctest_print
    # EXEC FUNC
    #six.exec_(src, test_globals, test_locals)  # adds stack to debug trace
    import utool as ut
    # TODO RECTIFY WITH TF
    if ut.get_argflag(('--cmd', '--embed')):
        src = _cmd_modify_src(src)
    code = compile(src, '<string>', 'exec')
    try:
        # IN EXEC CONTEXT THERE IS NO DIFF BETWEEN LOCAL / GLOBALS.  ONLY PASS
        # IN ONE DICT. OTHERWISE TREATED ODDLY
        # References: https://bugs.python.org/issue13557
        #exec(code, test_globals, test_locals)
        test_locals = test_globals
        exec(code, test_globals)
    except ExitTestException:
        print('Test exited before show')
        pass
    if nocheckwant is None:
        nocheckwant = util_arg.get_argflag(
            '--no-checkwant', help_='Turns off checking for results')
    if nocheckwant or want is None or want == '':
        if not nocheckwant:
            print('warning test does not want anything')
    else:
        if want.endswith('\n'):
            want = want[:-1]
        result = six.text_type(test_locals.get('result',
                                               'NO VARIABLE NAMED result'))
        if result != want:
            from utool import util_str
            msglines = []
            # Even if they arent exactly the same, the difference might just be
            # some whitespace characters. Ignore in this case.
            difftext = util_str.get_textdiff(want, result,
                                             ignore_whitespace=True)
            if difftext:
                if util_dbg.COLORED_EXCEPTIONS:
                    difftext = ut.color_diff_text(difftext)
                msglines += [
                    'DIFF/GOT/EXPECTED',
                    difftext,
                ]
                if VERBOSE_TEST:
                    msglines += [
                        'REPR_GOT: result=',
                        repr(result),
                        'REPR_EXPECTED: want=',
                        repr(want),
                    ]
                msglines += [
                    'STR_GOT: result=',
                    str(result),
                    'STR_EXPECTED: want=',
                    str(want),
                ]
                errmsg1 = '\n'.join(msglines)
                raise AssertionError('result != want\n' + errmsg1)
    return test_locals


def get_module_testlines(module_list, remove_pyc=True, verbose=True,
                         pythoncmd=None):
    """
    Builds test commands for autogen tests
    called by autogen test scripts
    """
    import utool as ut  # NOQA
    if pythoncmd is None:
        pythoncmd = sys.executable
        #'python'
    testcmd_list = []
    for module in module_list:
        mod_doctest_tup = get_module_doctest_tup(
            module=module, allexamples=True, verbose=verbose)
        (enabled_testtup_list, frame_fpath,
                 all_testflags, module_) = mod_doctest_tup
        for testtup in enabled_testtup_list:
            #testflag = testtup[-1]
            testflag = testtup.flag
            if remove_pyc:
                # FIXME python 3 __pycache__/*.pyc
                frame_fpath = frame_fpath.replace('.pyc', '.py')
            frame_rel_fpath = ut.get_relative_modpath(frame_fpath)
            testcmd = ' '.join((pythoncmd, frame_rel_fpath, testflag))
            testcmd_list.append(testcmd)
    return testcmd_list


def _docblock_tester1():
    """testting docblocks
        foo
    # Depth 5)
    call:
        foo
    the next call is:
        bar
    called by parse_doctest_from_docstr
    TODO: move to util_inspect
        test what happens with weird indentation
    bla

    Args:
        docstr (str): a documentation string formatted in google style.
            Remember indentation matters.
              whoo
            this
              .. params
        new (Optional[bool]): does a new style of parsing
        **kwargs:

    Returns:
        list: docstr_blocks tuples
            [(blockname, blockstr, offset)]

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> import utool as ut
        >>> print('example1')

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> import utool as ut
        >>> print('dummy')
    """


def _docblock_tester2():
    """
    Args:
    Returns:
    Example:
    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
    """


def _docblock_tester3():
    """
    foo
    Args:

    Returns:

    Example:

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
    """


def _docblock_tester4():
    """foobar

    Args:
        *args:
        **kwargs:

    Returns:
        bool

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
    """


def _test_docblock_parser():
    import utool as ut
    basis = {
        'n_args': [None, 0, 1],
        'n_return': [None, 0, 1, 2, 3],
        'spacing': [1, 2],
        # 'example_lines': [0, 1, 2],
        'doc_lines': [0, 1, 2],
        'doc_indent': [0, 1, 4],
    }
    for config in ut.all_dict_combinations(basis):
        print('---')
        print(config)
        print('=====')
        docstr =  _make_test_docstr(config)
        docparts = ut.parse_docblocks_from_docstr(docstr)
        print(docstr)
        print('=====')
        if config['n_args']:
            assert 'Args:' in ut.take_column(docparts, 0)
        if config['n_return']:
            assert 'Returns:' in ut.take_column(docparts, 0)


def _make_test_docstr(config):
    import utool as ut

    blocks = []

    docpart = '\n'.join(ut.lorium_ipsum().strip().split('\n')[0:config['doc_lines']])
    docpart = (' ' * config['doc_indent']) + docpart
    blocks.append(docpart)

    if config['n_args'] is not None:
        argheader = 'Args'
        argname_list = ut.chr_range(config['n_args'], base='a')
        argtype_list = ['bool'] * config['n_args']
        argdesc_list = [''] * config['n_args']
        arg_docstr = ut.make_args_docstr(argname_list, argtype_list,
                                         argdesc_list, ismethod=False)
        argsblock = ut.make_docstr_block(argheader, arg_docstr)
    else:
        argsblock = ''

    if config['n_return'] is not None:
        if config['n_return'] == 0:
            returnblock = 'Returns:'
        else:
            return_name = 'outvar'
            return_type = 'int'
            return_desc = '\n'.join(ut.lorium_ipsum().strip().split('\n')[0:config['n_return']])
            return_doctr = ut.make_returns_or_yeilds_docstr(return_type, return_name, return_desc)
            returnblock = ut.make_docstr_block('Returns', return_doctr)
        blocks.append(returnblock)

    blocks.append(argsblock)

    docstr = ('\n' * config['spacing']).join(blocks)
    return docstr


def parse_docblocks_from_docstr(docstr, offsets=False):
    """
    # Depth 5)
    called by parse_doctest_from_docstr

    TODO: move to util_inspect

    Args:
        docstr (str): a documentation string formatted in google style.

    Returns:
        list: docstr_blocks tuples
            [(blockname, blockstr)]

    CommandLine:
        python -m utool.util_tests parse_docblocks_from_docstr

    Doctest:
        >>> from utool.util_tests import *  # NOQA
        >>> import utool as ut
        >>> #func_or_class = ut.flatten2
        >>> #func_or_class = ut.list_depth
        >>> func_or_class = ut.iter_module_doctestable
        >>> func_or_class = ut.all_multi_paths
        >>> docstr = ut.get_docstr(func_or_class)
        >>> docstr_blocks = parse_docblocks_from_docstr(docstr)
        >>> result = str(ut.repr4(docstr_blocks))
        >>> print(result)
    """
    if docstr is None:
        return []
    # import parse
    import utool as ut
    import re
    # import itertools as it
    docstr = ut.ensure_unicode(docstr)
    docstr = ut.unindent(docstr)

    # Parse out initial documentation lines
    # Then parse out the blocked lines.
    docstr_lines = docstr.split('\n')
    line_indent = [ut.get_indentation(line) for line in docstr_lines]
    line_len = [len(line) for line in docstr_lines]

    # The first line may not have the correct indentation if it starts
    # right after the triple quotes. Adjust it in this case to ensure that
    # base indent is always 0
    adjusted = False
    is_nonzero = [len_ > 0 for len_ in line_len]
    if len(line_indent) >= 2:
        if line_len[0] != 0:
            indents = ut.compress(line_indent, is_nonzero)
            if len(indents) >= 2:
                indent_adjust = min(indents[1:])
                line_indent[0] += indent_adjust
                line_len[0] += indent_adjust
                docstr_lines[0] = (' ' * indent_adjust) + docstr_lines[0]
                adjusted = True
    if adjusted:
        # Redo prepreocessing, but this time on a rectified input
        docstr = ut.unindent('\n'.join(docstr_lines))
        docstr_lines = docstr.split('\n')
        line_indent = [ut.get_indentation(line) for line in docstr_lines]
        line_len = [len(line) for line in docstr_lines]

    indents = ut.compress(line_indent, is_nonzero)
    if len(indents) >= 1:
        if indents[0] != 0:
            print('ERROR IN PARSING')
            print('adjusted = %r' % (adjusted,))
            print(docstr)
            raise ValueError('Google Style Docstring Missformat')

    base_indent = 0
    # We will group lines by their indentation.
    # Rectify empty lines by giving them their parent's indentation.
    true_indent = []
    prev_indent = None
    for indent_, len_ in zip(line_indent, line_len):
        if len_ == 0:
            # Empty lines take on their parents indentation
            indent_ = prev_indent
        true_indent.append(indent_)
        prev_indent = indent_

    # tag_pattern = ut.regex_or(['Args:', 'Return:', 'CommandLine':])
    tag_pattern = '[^\\s]+: *$'

    group_id = 0
    prev_indent = 0
    group_list = []
    in_tag = False
    for line_num, (line, indent_) in enumerate(zip(docstr_lines, true_indent)):
        if re.match(tag_pattern, line):
            # Check if we can look ahead
            if line_num + 1 < len(docstr_lines):
                # A tag is only valid if its next line is properly indented,
                # empty, or is a tag itself.
                if (true_indent[line_num + 1] > base_indent or line_len[line_num + 1] == 0 or re.match(tag_pattern, docstr_lines[line_num + 1])):
                    group_id += 1
                    in_tag = True
            else:
                group_id += 1
                in_tag = True
        # Check if this line belongs to a new group
        elif in_tag and indent_ != prev_indent and indent_ == base_indent:
            group_id += 1
            in_tag = False
        group_list.append(group_id)
        prev_indent = indent_

    groups_ = ut.group_items(docstr_lines, group_list)
    groups = []
    line_offset = 0
    for k, lines in groups_.items():
        if len(lines) == 0 or (len(lines) == 1 and len(lines[0]) == 0):
            continue
        elif len(lines) >= 1 and re.match(tag_pattern, lines[0]):
            # An encoded google sub-block
            key = lines[0]
            val = lines[1:]
            subblock = ut.unindent('\n'.join(val))
        else:
            # A top level text documentation block
            key = '__DOC__'
            val = lines[:]
            subblock = '\n'.join(val)

        if offsets:
            groups.append((key, subblock, line_offset))
        else:
            groups.append((key, subblock))
        line_offset += len(lines)
    # Ensure that no keys are duplicated
    # try:
    #     assert len(ut.find_duplicate_items(ut.take_column(groups, 0))) == 0, (
    #         'Duplicate google docblock keys are not allowed')
    # except Exception as ex:
    #     ut.printex(ex, iswarning=True)
    return groups


def read_exampleblock(docblock):
    # if False:
    #     no longer needed with new parser
    #     import utool as ut
    #     nonheader_src = ut.unindent('\n'.join(docblock.splitlines()[1:]))
    nonheader_src = docblock
    nonheader_lines = nonheader_src.splitlines()
    reversed_src_lines = []
    reversed_want_lines = []
    finished_want = False

    # Read the example block backwards to get the want string
    # and then the rest should all be source
    for line in reversed(nonheader_lines):
        if not finished_want:
            if line.startswith('>>> ') or line.startswith('... '):
                finished_want = True
            else:
                reversed_want_lines.append(line)
                if len(line.strip()) == 0:
                    reversed_want_lines = []
                continue
        reversed_src_lines.append(line[4:])
    test_src = '\n'.join(reversed_src_lines[::-1])
    test_want = '\n'.join(reversed_want_lines[::-1])
    return test_src, test_want


def parse_doctest_from_docstr(docstr):
    r"""
    because doctest itself doesnt do what I want it to do

    CAREFUL, IF YOU GET BACK WRONG RESULTS MAKE SURE YOUR DOCSTR IS PREFFIXED
    WITH R

    CommandLine:
        python -m utool.util_tests --exec-parse_doctest_from_docstr

    Doctest:
        >>> from utool.util_tests import *  # NOQA
        >>> import utool as ut
        >>> func_or_class = parse_doctest_from_docstr
        >>> func_or_class = ut.list_depth
        >>> #func_or_class = ut.util_depricated.cartesian
        >>> func_or_class = ut.iter_module_doctestable
        >>> docstr = ut.get_docstr(func_or_class)
        >>> testsrc_list, testwant_list, testlinenum_list, func_lineno, docstr = get_doctest_examples(func_or_class)
        >>> print('\n\n'.join(testsrc_list))
        >>> assert len(testsrc_list) == len(testwant_list)
    """
    import utool as ut
    docstr_blocks = parse_docblocks_from_docstr(docstr, offsets=True)
    # print('docstr_blocks = %r' % (docstr_blocks,))

    example_docblocks = []
    example_setups = []
    grid_example_docblock = []
    grid_setups = []
    param_grids = None

    for header, docblock, line_offset in docstr_blocks:
        if header.startswith('Example') or header.startswith('Doctest'):
            example_docblocks.append((header, docblock, line_offset))

        if header.startswith('Setup'):
            setup_src = read_exampleblock(docblock)[0]
            example_setups.append(setup_src)

        if header.startswith('GridParams'):
            paramgrid_src = read_exampleblock(docblock)[0]
            globals_ = {}
            six.exec_('import utool as ut\n' + paramgrid_src, globals_)
            assert 'combos' in globals_, 'param grid must define combos'
            combos = globals_['combos']
            param_grids = [ut.execstr_dict(combo, explicit=True)
                           for combo in combos]

        if header.startswith('GridExample'):
            grid_example_docblock.append((header, docblock, line_offset))

        if header.startswith('GridSetup'):
            setup_src = read_exampleblock(docblock)[0]
            grid_setups.append(setup_src)

    assert len(example_setups) <= 1, 'cant have more than 1 setup. %d' % (
        len(example_setups))

    if example_setups and not grid_setups:
        grid_setups = example_setups

    testheader_list     = []
    testsrc_list        = []
    testwant_list       = []
    testlineoffset_list = []

    # Place grid tests first
    for header, docblock, line_offset in grid_example_docblock:
        test_src, test_want = read_exampleblock(docblock)
        assert len(grid_setups) <= 1, 'need one grid setup'
        if len(grid_setups):
            grid_setup = grid_setups[0]
        else:
            grid_setup = ''
        hack_show_request = False
        if 'ut.show_if_requested()' in test_src:
            hack_show_request = True
            test_src = test_src.replace('ut.show_if_requested()', '')  # Megahack
        full_grid_testsrc = '\n'.join(
            [grid_setup] + [
                '\n'.join([
                    pgrid, 'print(\'Grid %d\')' % (count,), test_src
                    if count < (len(param_grids) - 1) else test_src
                ])
                for count, pgrid in enumerate(param_grids)])
        if hack_show_request:
            full_grid_testsrc += '\n' + 'ut.show_if_requested()'

        testsrc_list.append(full_grid_testsrc)
        testheader_list.append(header)
        testwant_list.append(test_want)
        testlineoffset_list.append(line_offset)

    for header, docblock, line_offset in example_docblocks:
        test_src, test_want = read_exampleblock(docblock)
        if header.startswith('Doctest'):
            # Enable anything labeled as a doctest
            test_src = '# ENABLE_DOCTEST\n' + test_src
        if len(example_setups) == 0:
            full_testsrc = test_src
        elif len(example_setups) == 1:
            # Hack: append setups to all sources
            full_testsrc = '\n'.join([example_setups[0], test_src])
        else:
            assert False, 'more than 1 setup'
        testheader_list.append(header)
        testsrc_list.append(full_testsrc)
        testwant_list.append(test_want)
        testlineoffset_list.append(line_offset)

    return testheader_list, testsrc_list, testwant_list, testlineoffset_list


#@debug_decor
def get_doctest_examples(func_or_class, modpath=None):
    """
    get_doctest_examples

    Depth 3)
    called by get_module_doctest_tup

    Args:
        func_or_class (function)

    Returns:
        tuple (list, list): example_list, want_list

    CommandLine:
        python -m utool.util_tests --test-get_doctest_examples

    Doctest:
        >>> from utool.util_tests import *  # NOQA
        >>> func_or_class = get_doctest_examples
        >>> tup  = get_doctest_examples(func_or_class)
        >>> testsrc_list, testwant_list, testlinenum_list, func_lineno, docstr = tup
        >>> result = str(len(testsrc_list) + len(testwant_list))
        >>> print(testsrc_list)
        >>> print(testlinenum_list)
        >>> print(func_lineno)
        >>> print(testwant_list)
        >>> print(result)
        6

    Doctest:
        >>> from utool.util_tests import *  # NOQA
        >>> import utool as ut
        >>> func_or_class = ut.tryimport
        >>> tup = get_doctest_examples(func_or_class)
        >>> testsrc_list, testwant_list, testlinenum_list, func_lineno, docstr = tup
        >>> result = str(len(testsrc_list) + len(testwant_list))
        >>> print(testsrc_list)
        >>> print(testlinenum_list)
        >>> print(func_lineno)
        >>> print(testwant_list)
        >>> print(result)
        4

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> import ibeis
        >>> func_or_class = ibeis.control.manual_annot_funcs.add_annots
        >>> tup = get_doctest_examples(func_or_class)
        >>> testsrc_list, testwant_list, testlinenum_list, func_lineno, docstr = tup
        >>> result = str(len(testsrc_list) + len(testwant_list))
        >>> print(testsrc_list)
        >>> print(testlinenum_list)
        >>> print(func_lineno)
        >>> print(testwant_list)
        >>> print(result)
        2
    """
    if isinstance(func_or_class, staticmethod):
        func_or_class = func_or_class.__func__
    import utool as ut
    if VERBOSE_TEST:
        print('[util_test][DEPTH 3] get_doctest_examples()')
        print('[util_test] + parsing %r for doctest' % (func_or_class))
        print('[util_test] - name = %r' % (func_or_class.__name__,))
        if hasattr(func_or_class, '__ut_parent_class__'):
            print('[util_test] - __ut_parent_class__ = %r' % (
                func_or_class.__ut_parent_class__,))
    try:
        raise NotImplementedError('FIXME')
        #func_or_class._utinfo['orig_func']
        func_lineno = func_or_class.func_code.co_firstlineno
        # FIXME: doesn't handle decorators well
        #
        # ~~FIXME doesn't account for multiline function definitions
        # actually parse this out~~
        # TODO: rectify with util_insepct get_funcsource with stip def line
        sourcecode = inspect.getsource(func_or_class)
        match = ut.regex_get_match('def [^)]*\\):\n', sourcecode)
        if match is not None:
            num_funcdef_lines = match.group().count('\n')
        else:
            num_funcdef_lines = 1
    except Exception as ex:
        func_lineno = 0
        num_funcdef_lines = 1
        if ut.DEBUG2:
            ut.printex(ex, '[util-test] error getting function line number')

    docstr = ut.get_docstr(func_or_class)

    (testheader_list, testsrc_list, testwant_list,
     testlineoffset_list) = parse_doctest_from_docstr(docstr)
    testlinenum_list = [
        func_lineno + num_funcdef_lines + offset
        for offset in testlineoffset_list
    ]
    if VERBOSE_TEST:
        print('[util_test] L found %d doctests' % (len(testsrc_list),))
    examptup = testsrc_list, testwant_list, testlinenum_list, func_lineno, docstr
    return examptup


def get_module_doctest_tup(testable_list=None, check_flags=True, module=None,
                           allexamples=None, needs_enable=None, N=0,
                           verbose=True, testslow=False):
    """
    Parses module for testable doctesttups
    Depth 2)

    Args:
        testable_list (list): a list of functions (default = None)
        check_flags (bool): (default = True)
        module (None): (default = None)
        allexamples (None): (default = None)
        needs_enable (None): (default = None)
        N (int): (default = 0)
        verbose (bool):  verbosity flag(default = True)
        testslow (bool): (default = False)

    Returns:
        ModuleDoctestTup : (enabled_testtup_list, frame_fpath, all_testflags, module)
            enabled_testtup_list (list): a list of testtup
                testtup (tuple): (name, num, src, want, flag) describes a valid doctest in the module
                    name  (str): test name
                    num   (str): test number of the module / function / class / method
                    src   (str): test source code
                    want  (str): expected test result
                    flag  (str): a valid commandline flag to enable this test
            frame_fpath (str):
                module fpath that will be tested
            module (module):
                the actual module that will be tested
            all_testflags (list):
                the command line arguments that will enable different tests
            exclude_inherited (bool): does not included tests defined in other modules

    CommandLine:
        python -m utool.util_tests --exec-get_module_doctest_tup

    Doctest:
        >>> from utool.util_tests import *  # NOQA
        >>> import utool as ut
        >>> #testable_list = [ut.util_import.package_contents]
        >>> testable_list = None
        >>> check_flags = False
        >>> module = ut.util_cplat
        >>> allexamples = False
        >>> needs_enable = None
        >>> N = 0
        >>> verbose = True
        >>> testslow = False
        >>> mod_doctest_tup = get_module_doctest_tup(testable_list, check_flags,
        >>>                                          module, allexamples,
        >>>                                          needs_enable, N, verbose,
        >>>                                          testslow)
        >>> result = ('mod_doctest_tup = %s' % (ut.repr4(mod_doctest_tup, nl=4),))
        >>> print(result)
    """
    #+------------------------
    if VERBOSE_TEST:
        print('[util_test.get_module_doctest tup][DEPTH 2] get_module_doctest tup()')
    import utool as ut  # NOQA
    if needs_enable is None:
        needs_enable = not ut.get_argflag('--enableall')
        #needs_enable = True
    TEST_ALL_EXAMPLES = allexamples or ut.get_argflag(('--allexamples', '--all-examples'))
    parse_testables = True
    if isinstance(testable_list, types.ModuleType):
        # hack
        module = testable_list
        testable_list = []
        testable_name_list = []
    elif testable_list is None:
        testable_list = []
        testable_name_list = []
    else:
        testable_name_list = [ut.get_funcname(func) for func in testable_list]
        parse_testables = False
    #L________________________
    #+------------------------
    # GET_MODULE_DOCTEST_TUP Step 1:
    # Inspect caller module for testable names
    if module is None:
        frame_fpath = '???'
        try:
            # This is a bit finky. Need to be exactly N frames under the main
            # module
            frame = ut.get_parent_frame(N=N)
            main_modname = '__main__'
            frame_name  = frame.f_globals['__name__']
            frame_fpath = frame.f_globals['__file__']
            if frame_name == main_modname:
                module = sys.modules[main_modname]
                entry_modname = ut.get_modname_from_modpath(module.__file__)
                if entry_modname in ['kernprof', 'kernprof-script']:
                    # kernprof clobbers the __main__ variable.
                    # workaround by reimporting the module name
                    import importlib
                    modname = ut.get_modname_from_modpath(frame_fpath)
                    module = importlib.import_module(modname)
        except Exception as ex:
            print(frame.f_globals)
            ut.printex(ex, keys=['frame', 'module'])
            raise
        allexamples = False
    else:
        frame_fpath = module.__file__
        allexamples = True
    #L________________________

    #+------------------------
    # GET_MODULE_DOCTEST_TUP Step 2:
    # --- PARSE TESTABLE FUNCTIONS ---
    # Get testable functions
    if parse_testables:
        try:
            if verbose or VERBOSE_TEST and ut.NOT_QUIET:
                print('[ut.test] Iterating over module funcs')
                print('[ut.test] module =%r' % (module,))

            _testableiter = ut.iter_module_doctestable(module,
                                                       include_inherited=False)
            for key, val in _testableiter:
                if isinstance(val, staticmethod):
                    docstr = inspect.getdoc(val.__func__)
                else:
                    docstr = inspect.getdoc(val)
                docstr = ut.ensure_unicode(docstr)
                if docstr is not None and (docstr.find('Example') >= 0 or docstr.find('Doctest') >= 0):
                    testable_name_list.append(key)
                    testable_list.append(val)
                    if VERBOSE_TEST and ut.NOT_QUIET:
                        print('[ut.test] Testable: %s' % (key,))
                else:
                    if VERBOSE_TEST and ut.NOT_QUIET:
                        if (docstr.find('Example') >= 0 or docstr.find('Doctest') >= 0):
                            print('[ut.test] Ignoring (disabled) : %s' % key)
                        else:
                            print('[ut.test] Ignoring (no Example) : %s' % key)
        except Exception as ex:
            print('FAILED')
            print(docstr)
            ut.printex(ex, keys=['frame'])
            raise

    # OUTPUTS: testable_list
    #L________________________
    #+------------------------
    # GET_MODULE_DOCTEST_TUP Step 3:
    # --- FILTER TESTABLES_---

    # Get testable function examples

    test_sentinals = [
        'ENABLE_DOCTEST',
        'ENABLE_GRID_DOCTEST',
    ]
    if testslow or ut.get_argflag(('--testall', '--testslow', '--test-slow')):
        test_sentinals.append('SLOW_DOCTEST')
    if testslow or ut.get_argflag(('--testall', '--testunstable')):
        test_sentinals.append('UNSTABLE_DOCTEST')

    # FIND THE TEST NAMES REQUESTED
    # Grab sys.argv enabled tests
    cmdline_varargs = ut.get_cmdline_varargs()
    force_enable_testnames_ = cmdline_varargs[:]
    valid_prefix_list = ['--test-', '--exec-', '--dump-']
    #if False:
    for arg in sys.argv:
        for prefix in valid_prefix_list:
            if arg.startswith(prefix):
                testname = arg[len(prefix):]
                #testname = testname.split(':')[0].replace('-', '_')
                force_enable_testnames_.append(testname)
                #break

    # PartA: Fixup names
    # TODO: parse out requested test number here
    # instead of later in the code. See PartB
    force_enable_testnames = []
    for testname in force_enable_testnames_:
        testname = testname.split(':')[0].replace('-', '_')
        testname.split(':')[0].replace('-', '_')
        force_enable_testnames.append(testname)

    def _get_testable_name(testable):
        import utool as ut
        if isinstance(testable, staticmethod):
            testable = testable.__func__
        try:
            testable_name = testable.func_name
        except AttributeError as ex1:
            try:
                testable_name = testable.__name__
            except AttributeError as ex2:
                ut.printex(ex1, ut.repr4(dir(testable)))
                ut.printex(ex2, ut.repr4(dir(testable)))
                raise
        return testable_name

    sorted_testable = sorted(list(set(testable_list)), key=_get_testable_name)
    # Append each testable example
    if VERBOSE_TEST:
        print('Vars:')
        print(' * needs_enable = %r' % (needs_enable,))
        print(' * force_enable_testnames = %r' % (force_enable_testnames,))
        print(' * len(sorted_testable) = %r' % (len(sorted_testable),))
        print(' * cmdline_varargs = %r' % (cmdline_varargs,))
        indenter = ut.Indenter('[FIND_AVAIL]')
        indenter.start()
    # PARSE OUT THE AVAILABLE TESTS FOR EACH REQUEST
    local_testtup_list = []
    for testable in sorted_testable:
        short_testname = _get_testable_name(testable)
        full_testname = None  # Namespaced classname (within module)
        if isinstance(testable, staticmethod):
            testable = testable.__func__
        if hasattr(testable, '__ut_parent_class__'):
            # HACK for getting classname.funcname
            test_namespace = testable.__ut_parent_class__.__name__
            full_testname = test_namespace + '.' + short_testname
        else:
            test_namespace = None
            full_testname = short_testname

        nametup = tuple(ut.unique([full_testname, short_testname]))
        # modpath = ut.get_modpath(module)
        examptup = get_doctest_examples(testable)
        examples, wants, linenums, func_lineno, docstr = examptup
        total_examples = len(examples)
        if total_examples > 0:
            for testno , srcwant_tup in enumerate(zip(examples, wants)):
                src, want = srcwant_tup
                src_ = ut.regex_replace('from __future__ import.*$', '', src)
                test_disabled = not any([src_.find(s) >= 0
                                         for s in test_sentinals])
                skip = (needs_enable and test_disabled and
                        ut.isdisjoint(nametup, force_enable_testnames))
                if not skip:
                    if VERBOSE_TEST:
                        print(' * HACK adding testname=%r to local_testtup_list' % (
                            full_testname,))
                    local_testtup = (nametup, testno, src_, want,
                                     test_namespace, short_testname,
                                     total_examples)
                    local_testtup_list.append(local_testtup)
                else:
                    if VERBOSE_TEST:
                        #print('force_enable_testnames = %r' % (force_enable_testnames,))
                        #print('nametup = %r' % (nametup,))
                        #print('needs_enable = %r' % (needs_enable,))
                        #print('test_disabled = %r' % (test_disabled,))
                        print(' * skipping: %r / %r' % (short_testname,
                                                        full_testname))
        else:
            print('WARNING: no examples in %r for testname=%r' % (frame_fpath,
                                                                  full_testname))
            if verbose:
                print(testable)
                print(examples)
                print(wants)
                print(docstr)
        if VERBOSE_TEST:
            print(' --')
    if VERBOSE_TEST:
        indenter.stop()
    #L________________________
    #+------------------------
    # Get enabled (requested) examples
    if VERBOSE_TEST:
        print('\n-----\n')
        indenter = ut.Indenter('[IS_ENABLED]')
        indenter.start()
        print('Finished parsing available doctests.')
        print('Now we need to find which examples are enabled')
        print('len(local_testtup_list) = %r' % (len(local_testtup_list),))
        print('local_testtup_list.T[0:2].T = %s' %
              ut.repr4(ut.take_column(local_testtup_list, [0, 1])))
        print('sys.argv = %r' % (sys.argv,))
    all_testflags = []
    enabled_testtup_list = []
    distabled_testflags  = []
    subx = ut.get_argval('--subx', type_=int, default=None,
                         help_='Only tests the subxth example')

    def make_valid_testnames(name, num, total):
        return [
            name + ':' + str(num),
            name,
            name + ':' + str(num - total),  # allow negative indices
            # prefix + name.replace('_', '-') + ':' + str(num),
            # prefix + name.replace('_', '-')
        ]

    def make_valid_test_argflags(prefix, name, num, total):
        valid_testnames = make_valid_testnames(name, num, total)
        return [prefix + testname for testname in valid_testnames]

    def check_if_test_requested(nametup, num, total, valid_prefix_list):
        #cmdline_varargs
        if VERBOSE_TEST:
            print('Checking cmdline for %r %r' % (nametup, num))
        valid_argflags = []

        # FIXME: PartB
        # should parse out test number above instead of here
        # See PartA
        mode = None
        veryverb = 0
        # First check positional args
        testflag = None
        for name in nametup:
            valid_testnames = make_valid_test_argflags('', name, num, total)
            if veryverb:
                print('Checking if positional* %r' % (valid_testnames[0:1],))
                print('name = %r' % (name,))
            if any([x in cmdline_varargs for x in valid_testnames]):
                # hack
                mode = 'exec'
                testflag = name
                flag1 = '--exec-' + name + ':' + str(num)
            if testflag is not None:
                if veryverb:
                    print('FOUND POSARG')
                    print(' * testflag = %r' % (testflag,))
                    print(' * num = %r' % (num,))
                break
        # Then check keyword-ish args
        if mode is None:
            for prefix, name in reversed(list(ut.iprod(valid_prefix_list, nametup))):
                valid_argflags = make_valid_test_argflags(prefix, name, num, total)
                if veryverb:
                    print('Checking for flags*: %r' % (valid_argflags[0],))
                flag1 = valid_argflags[0]
                testflag = ut.get_argflag(valid_argflags)
                mode = prefix.replace('-', '')
                if testflag:
                    if veryverb:
                        print("FOUND VARARG")
                    break
            else:
                # print('WARNING NO TEST IS ENABLED %r ' % (nametup,))
                pass
        checktup = flag1, mode, name, testflag
        return checktup

    for local_testtup in local_testtup_list:
        (nametup, num, src, want, shortname, test_namespace, total) = local_testtup
        checktup = check_if_test_requested(nametup, num, total,
                                           valid_prefix_list)
        flag1, mode, name, testflag = checktup
        testenabled = TEST_ALL_EXAMPLES  or not check_flags or testflag
        if subx is not None and subx != num:
            continue
        all_testflags.append(flag1)
        if testenabled:
            if VERBOSE_TEST:
                print('... enabling test')
            testtup = TestTuple(name, num, src, want, flag1,
                                frame_fpath=frame_fpath, mode=mode,
                                total=total, nametup=nametup,
                                shortname=shortname,
                                test_namespace=test_namespace)
            if VERBOSE_TEST:
                print('... ' + str(testtup))
            enabled_testtup_list.append(testtup)
        else:
            if VERBOSE_TEST:
                print('... disabling test')
            distabled_testflags.append(flag1)

    # Attempt to run test without any context
    # This will only work if the function exist and is self contained
    if len(force_enable_testnames_) > 0 and len(enabled_testtup_list) == 0:
        if VERBOSE_TEST:
            print('Forced test did not have a doctest example')
            print('Maybe it can be run without any context')
        import utool as ut
        # assert len(force_enable_testnames) == 1
        test_funcname_ = force_enable_testnames[0]
        if test_funcname_.find('.') != -1:
            test_classname, test_funcname = test_funcname_.split('.')
            class_ = getattr(module, test_classname, None)
            assert class_ is not None
            func_ = getattr(class_, test_funcname, None)
        else:
            test_funcname = test_funcname_
            func_ = getattr(module, test_funcname, None)
        if VERBOSE_TEST:
            print('test_funcname = %r' % (test_funcname,))
            print('func_ = %r' % (func_,))
        if func_ is not None:
            testno = 0
            modname = ut.get_modname_from_modpath(module.__file__)
            want = None
            try:
                if VERBOSE_TEST:
                    print('attempting xdoctest hack')
                # hack to get classmethods to read their example using
                # the xdoctest port
                from xdoctest import docscrape_google
                from xdoctest import core as xdoc_core
                from xdoctest import static_analysis as static
                if func_.__doc__ is None:
                    raise TypeError
                blocks = docscrape_google.split_google_docblocks(func_.__doc__)

                example_blocks = []
                for type_, block in blocks:
                    if type_.startswith('Example') or type_.startswith('Doctest'):
                        example_blocks.append((type_, block))

                if len(example_blocks) == 0:
                    if VERBOSE_TEST:
                        print('xdoctest found no blocks')
                    raise KeyError

                callname = test_funcname_
                hack_testtups = []
                for num, (type_, block) in enumerate(example_blocks):
                    # print('modname = %r' % (modname,))
                    # print('callname = %r' % (callname,))
                    # print('num = %r' % (num,))
                    modpath = static.modname_to_modpath(modname)
                    example = xdoc_core.DocTest(
                        modpath=modpath, callname=callname, docsrc=block, num=num)
                    src = example.format_src(colored=False, want=False,
                                             linenos=False)
                    want = '\n'.join(list(example.wants()))
                    testtup = TestTuple(test_funcname_, num, src, want=want,
                                        flag='--exec-' + test_funcname_,
                                        frame_fpath=frame_fpath, mode='exec',
                                        total=len(example_blocks),
                                        nametup=[test_funcname_])
                    hack_testtups.append(testtup)
                if VERBOSE_TEST:
                    print('hack_testtups = %r' % (hack_testtups,))
                enabled_testtup_list.extend(hack_testtups)
                # src = '\n'.join([line[4:] for line in src.split('\n')])
            except (ImportError, KeyError, TypeError):
                if VERBOSE_TEST:
                    print('xdoctest hack failed')
                # varargs = ut.get_cmdline_varargs()
                varargs = force_enable_testnames[1:]
                # Create dummy doctest
                src = ut.codeblock(
                    '''
                    # DUMMY_DOCTEST
                    from {modname} import *  # NOQA
                    args = {varargs}
                    result = {test_funcname_}(*args)
                    print(result)
                    ''').format(
                        modname=modname,
                        test_funcname_=test_funcname_,
                        varargs=repr(varargs))
                testtup = TestTuple(test_funcname_, testno, src, want=want,
                                    flag='--exec-' + test_funcname_,
                                    frame_fpath=frame_fpath, mode='exec',
                                    total=1, nametup=[test_funcname_])
                enabled_testtup_list.append(testtup)
        else:
            print('function %r was not found in %r' % (test_funcname_, module))

    if VERBOSE_TEST:
        indenter.stop()

    if ut.get_argflag('--list'):
        # HACK: Should probably just return a richer structure
        print('testable_name_list = %s' % (ut.repr4(testable_name_list),))

    mod_doctest_tup = ModuleDoctestTup(enabled_testtup_list, frame_fpath,
                                       all_testflags, module)
    #L________________________
    return mod_doctest_tup


def doctest_was_requested():
    """ lets a  __main__ codeblock know that util_test should do its thing """
    # FIXME; does not handle positinal doctest requests
    valid_prefix_list = ['--exec-', '--test-']
    return '--tf' in sys.argv or any([any([arg.startswith(prefix) for prefix in
                                           valid_prefix_list])
                                      for arg in sys.argv])


def find_doctestable_modnames(dpath_list=None, exclude_doctests_fnames=[],
                              exclude_dirs=[], allow_nonpackages=False):
    """
    Tries to find files with a call to ut.doctest_funcs in the __main__ part
    Implementation is very hacky. Should find a better heuristic

    Args:
        dpath_list (list): list of python package directories
        exclude_doctests_fnames (list): (default = [])
        exclude_dirs (list): (default = [])
        allow_nonpackages (bool): (default = False)

    Returns:
        list: list of filepaths

    CommandLine:
        python -m utool.util_tests find_doctestable_modnames --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> import utool as ut
        >>> from os.path import dirname
        >>> dpath_list = [ut.get_module_dir(ut)]
        >>> exclude_doctests_fnames = []
        >>> exclude_dirs = []
        >>> allow_nonpackages = False
        >>> result = find_doctestable_modnames(dpath_list, exclude_doctests_fnames, exclude_dirs, allow_nonpackages)
        >>> print(result)
    """
    import utool as ut
    from os.path import dirname, exists, join
    fpath_list = ut.grep(r'doctest_funcs\(', dpath_list=dpath_list,
                         include_patterns=['*.py'], exclude_dirs=exclude_dirs,
                         recursive=True)[0]
    exclude_doctests_fnames = set(exclude_doctests_fnames)

    def is_not_excluded(fpath):
        return basename(fpath) not in exclude_doctests_fnames

    def is_in_package(fpath):
        return exists(join(dirname(fpath), '__init__.py'))
    fpath_list = list(filter(is_in_package, fpath_list))
    fpath_list = list(filter(is_not_excluded, fpath_list))
    doctest_modname_list = list(map(ut.get_modname_from_modpath, fpath_list))
    doctest_modname_list = ut.unique(doctest_modname_list)
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


def show_was_requested():
    """
    returns True if --show is specified on the commandline or you are in
    IPython (and presumably want some sort of interaction
    """
    try:
        import plottool_ibeis as pt
    except ImportError:
        import plottool as pt
    return pt.show_was_requested()
    #import utool as ut
    #return ut.get_argflag('--show') or ut.inIPython()


try:
    # Use ExitTestException from xdoctest if possible
    from xdoctest import ExitTestException
except ImportError:
    try:
        from xdoctest.core import ExitTestException
    except ImportError:
        class ExitTestException(Exception):
            pass


def qt4ensure():
    try:
        import plottool_ibeis as pt
    except ImportError:
        import plottool as pt
    pt.qtensure()


def qtensure():
    try:
        import plottool_ibeis as pt
    except ImportError:
        import plottool as pt
    pt.qtensure()


def quit_if_noshow():
    import utool as ut
    saverequest = ut.get_argval('--save', default=None)
    if not (saverequest or ut.get_argflag(('--show', '--save')) or ut.inIPython()):
        raise ExitTestException('This should be caught gracefully by ut.run_test')


def show_if_requested():
    try:
        import plottool_ibeis as pt
    except ImportError:
        import plottool as pt
    pt.show_if_requested(N=2)


def find_testfunc(module, test_funcname, ignore_prefix=[], ignore_suffix=[],
                  func_to_module_dict={}, return_mod=False):
    import utool as ut
    if isinstance(module, six.string_types):
        module = ut.import_modname(module)
    modname_list = ut.package_contents(module, ignore_prefix=ignore_prefix,
                                       ignore_suffix=ignore_suffix)
    # Get only the modules already imported
    have_modnames = [modname_ for modname_ in modname_list
                     if modname_ in sys.modules]
    #missing_modnames = [modname for modname in modname_list
    #                    if modname not in sys.modules]
    module_list = ut.dict_take(sys.modules, have_modnames)
    # Search for the module containing the function
    test_func = None
    test_module = None
    test_classname = None
    if test_funcname.find('.') != -1:
        test_classname, test_funcname = test_funcname.split('.')
    if test_funcname.find(':') != -1:
        test_funcname, testno = test_funcname.split(':')
        testno = int(testno)
    else:
        testno = 0
    if test_classname is None:
        for module_ in module_list:
            #test_funcname = 'find_installed_tomcat'
            if test_funcname in module_.__dict__:
                test_module = module_
                test_func = test_module.__dict__[test_funcname]
                break
    else:
        for module_ in module_list:
            #test_funcname = 'find_installed_tomcat'
            if test_classname in module_.__dict__:
                test_module = module_
                test_class = test_module.__dict__[test_classname]
                test_func = getattr(test_class, test_funcname)
                # test_class.__dict__[test_funcname]

    if test_func is None:
        print('Did not find any function named %r ' % (test_funcname,))
        print('Searched ' + ut.repr4([mod.__name__ for mod in module_list]))
    if return_mod:
        return test_func, testno, test_module
    else:
        return test_func, testno


def get_module_completions(module):
    import utool as ut
    test_tuples = ut.get_package_testables(module)
    testnames = ut.instancelist(test_tuples).name
    return testnames


#def autocomplete_hook(module):
#    """
#    # https://argcomplete.readthedocs.io/en/latest/#activating-global-completion%20argcomplete
#    pip install argcomplete

#    Need to put
#    PYTHON_ARGCOMPLETE_OK at begining of file

#    pip install argcomplete
#    activate-global-python-argcomplete

#    eval "$(register-python-argcomplete your_script)"
#    register-python-argcomplete ibeis
#    eval "$(register-python-argcomplete ibeis)"
#    """
#    #if len(sys.argv) < 3:
#    #    sys.exit(1)
#    try:
#        # hook
#        import argparse as ap
#        import argcomplete
#    except ImportError:
#        pass
#    else:
#        parser = ap.ArgumentParser()
#        testnames = get_module_completions(module)
#        testnames = ['foo', 'foo2']
#        parser.add_argument('position1', choices=[testnames])
#        argcomplete.autocomplete(parser)
#        args = parser.parse_args()


def main_function_tester(module, ignore_prefix=[], ignore_suffix=[],
                         test_funcname=None, func_to_module_dict={}):
    """
    Allows a shorthand for __main__ packages of modules to run tests with
    unique function names
    """
    import utool as ut
    ut.colorprint('[utool] main_function_tester', 'yellow')

    if ut.get_argflag('--list-testfuncs'):
        print('Listing testfuncs')
        test_tuples = ut.get_package_testables(module)
        result = ut.repr3(test_tuples)
        print(result)

    #autocomplete_hook(module)

    if ut.get_argflag('--update-bashcomplete'):
        # http://stackoverflow.com/questions/427472/line-completion-with-custom-commands
        print('Listing testfuncs')
        testnames = get_module_completions(module)
        modname = module if isinstance(module, six.string_types) else module.__name__
        line = 'complete -W "%s" "%s"' % (' '.join(testnames), modname)
        bash_completer = ut.unixjoin(ut.ensure_app_resource_dir('ibeis'), 'ibeis_bash_complete.sh')
        ut.writeto(bash_completer, line)
        print('ADD TO BASHRC\nsource %s' % (bash_completer,))
        #print(line)
        sys.exit(ut.EXIT_SUCCESS)

    if ut.get_argflag('--make-bashcomplete'):
        # http://stackoverflow.com/questions/427472/line-completion-with-custom-commands
        print('Listing testfuncs')
        testnames = get_module_completions(module)
        modname = module if isinstance(module, six.string_types) else module.__name__
        line = 'complete -W "%s" "%s"' % (' '.join(testnames), modname)
        print('add the following line to your bashrc')
        print(line)
        sys.exit(ut.EXIT_SUCCESS)

    test_funcname = ut.get_argval(
        ('--test-func', '--tfunc', '--tf', '--testfunc'),
        type_=str, default=test_funcname,
        help_='specify a function to doctest')
    if test_funcname is None:
        cmdline_varags = ut.get_cmdline_varargs()
        if VERBOSE_TEST:
            print('Checking varargs')
            print('cmdline_varags = %r' % (cmdline_varags,))
        if len(cmdline_varags) > 0:
            test_funcname = cmdline_varags[0]
    print('test_funcname = %r' % (test_funcname,))

    if test_funcname in func_to_module_dict:
        modname = func_to_module_dict[test_funcname]
        ut.import_modname(modname)

    if test_funcname is not None:
        #locals_ = {}
        ut.inject_colored_exceptions()
        # print('[utool] __main__ Begin Function Test')
        print('[utool] __main__ Begin Function Test')
        test_func, testno, test_mod = find_testfunc(
            module, test_funcname, ignore_prefix, ignore_suffix,
            func_to_module_dict, return_mod=True)

        if test_func is not None:
            globals_ = {}
            try:
                func_globals = ut.get_funcglobals(test_func)
                globals_.update(func_globals)
            except AttributeError:
                pass
            tests = ut.get_doctest_examples(test_func)[0]
            if len(tests) > 0:
                testsrc = tests[testno]
            else:
                # Create dummy doctest
                modname = ut.get_modname_from_modpath(ut.get_modpath(test_mod))
                varargs = cmdline_varags[1:]
                testsrc = ut.codeblock(
                    '''
                    # DUMMY_DOCTEST
                    from {modname} import *  # NOQA
                    args = {varargs}
                    result = {test_funcname_}(*args)
                    print(result)
                    ''').format(
                        modname=modname,
                        test_funcname_=test_funcname,
                        varargs=repr(varargs))
                # testtup = TestTuple(test_funcname_, testno, src, want=want,
                #                     flag='--exec-' + test_funcname_,
                #                     frame_fpath=frame_fpath, mode='exec',
                #                     total=1, nametup=[test_funcname_])

            if ut.get_argflag(('--cmd', '--embed')):
                # TODO RECTIFY WITH EXEC DOCTEST
                testsrc = _cmd_modify_src(testsrc)
            # _exec_doctest(doctest_src)
            # doctest_src = ut.indent(testsrc, '>>> ')
            # Add line numbers
            doctest_src = ut.number_text_lines(testsrc)
            colored_src = ut.highlight_code(doctest_src)
            print('testsrc = \n%s' % (colored_src,))
            try:
                code = compile(testsrc, '<string>', 'exec')
                exec(code, globals_)  # , locals_)
            except ExitTestException:
                print('Test exited before show')
                pass
            retcode = ut.EXIT_SUCCESS
            print('Finished function test.')
        else:
            retcode = ut.EXIT_FAILURE
            print('Did not find any function named %r ' % (test_funcname,))
        if ut.util_inject.PROFILING:
            ut.dump_profile_text()
        print('...exiting')
        sys.exit(retcode)


def _cmd_modify_src(testsrc):
    # testsrc = 'import IPython; import utool as ut; ut.qtensure()\n' + testsrc
    # testsrc += '\nimport IPython; IPython.embed()'
    # testsrc += '\nimport utool as ut; ut.embed()'
    import utool as ut
    pline = 'print(%r)' % ut.highlight_code(ut.indent(testsrc, '>>> '))
    testsrc += '\nimport utool as ut; ' + pline + '; ut.qtensure(); ut.embed()'
    # testsrc += '\nimport utool as ut; ut.qtensure(); ut.embed()'
    return testsrc


def execute_doctest(func, testnum=0, module=None):
    """
    Execute a function doctest. Can optionaly specify func name and module to
    run from ipython notebooks.

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA

    IPython:
        import utool as ut
        ut.execute_doctest(func='dummy_example_depcacahe', module='dtool.example_depcache')
    """
    import utool as ut
    if isinstance(func, six.string_types):
        funcname = func
        if isinstance(module, six.string_types):
            modname = module
            module = ut.import_modname(modname)
        func, _testno = find_testfunc(module, funcname, ignore_prefix=[],
                                      ignore_suffix=[])
    # TODO RECTIFY WITH EXEC DOCTEST
    globals_ = {}
    testsrc = ut.get_doctest_examples(func)[0][testnum]
    # colored_src = ut.highlight_code(ut.indent(testsrc, '>>> '))
    doctest_src = ut.indent(testsrc, '>>> ')
    doctest_src = '\n'.join([
        '%3d %s' % (count, line)
        for count, line in enumerate(doctest_src.splitlines(), start=1)])
    colored_src = ut.highlight_code(doctest_src)
    print('testsrc = \n%s' % (colored_src,))
    try:
        code = compile(testsrc, '<string>', 'exec')
        exec(code, globals_)
    except ExitTestException:
        print('Test exited before show')


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_tests; utool.doctest_funcs(utool.util_tests)"
        python -m utool.util_tests
        python -m utool.util_tests --allexamples
        python -m utool.util_tests

        python -c "import utool; utool.doctest_funcs(module=utool.util_tests, needs_enable=False)"
        /model/preproc/preproc_chip.py --allexamples
    """
    import multiprocessing
    import utool as ut  # NOQA
    multiprocessing.freeze_support()
    #doctest_funcs()
    ut.doctest_funcs()
