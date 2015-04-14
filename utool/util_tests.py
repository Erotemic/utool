"""
Helpers for tests

This module contains a more sane reimplementation of doctest functionality.
(I.E. asserts work and you don't have to worry about standard out mucking things
up) The code isn't super clean though due to time constriaints. Many functions
probably belong elsewhere and the parsers need a big cleanup.

TODO:
    report the line of the doctest in the file when reporting errors as well as
    the relative line
"""
from __future__ import absolute_import, division, print_function
import six
from six.moves import builtins
from collections import namedtuple
import inspect
import types
import traceback  # NOQA
import sys
from os.path import basename
from utool import util_print  # NOQA
from utool import util_arg
from utool import util_path
from utool import util_time
from utool import util_inject
from utool._internal.meta_util_six import get_funcname
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[tests]')


VERBOSE_TEST = util_arg.get_argflag(('--verb-test', '--verbose-test'))
#PRINT_SRC = not util_arg.get_argflag(('--noprintsrc', '--nosrc'))
DEBUG_SRC = not util_arg.get_argflag('--nodbgsrc')
PRINT_SRC = util_arg.get_argflag(('--printsrc', '--src'))
PRINT_FACE = not util_arg.get_argflag(('--noprintface', '--noface'))
#BIGFACE = False
BIGFACE = util_arg.get_argflag('--bigface')
SYSEXIT_ON_FAIL = util_arg.get_argflag('--sysexitonfail')

ModuleDoctestTup = namedtuple('ModuleDoctestTup', ('enabled_testtup_list', 'frame_fpath', 'all_testflags', 'module'))

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


def _get_testable_name(testable):
    """
    Depth 3)
    called by get_module_doctest_tup
    """
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


def dev_ipython_copypaster(func):
    import utool as ut
    code_text = get_dev_paste_code(func)
    ut.copy_text_to_clipboard(code_text)


def get_dev_paste_code(func):
    import utool as ut
    example_texts = ut.get_doctest_examples(func)
    example_text = example_texts[0][0]
    assert isinstance(example_text, str), ut.list_str(example_text)
    assert isinstance(example_text, str), ut.list_str(example_text)
    source_text = ut.get_func_source(func)
    get_dev_code = '\n'.join((example_text, source_text))
    return get_dev_code


def get_func_source(func):
    r"""
    Args:
        func (function): live function

    Returns:
        src: source_text

    CommandLine:
        python -m utool.util_tests --test-get_func_source

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.devcases import *  # NOQA
        >>> func = myquery
        >>> deventer(func)
    """
    import utool as ut
    source_text = inspect.getsource(func)
    source_lines = source_text.splitlines()
    source_text = ut.unindent('\n'.join(source_lines[1:]))
    return source_text


def get_module_testlines(module_list, remove_pyc=True, verbose=True,
                         pythoncmd=None, **kwargs):
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
        mod_doctest_tup = get_module_doctest_tup(module=module, allexamples=True,
                                                 verbose=verbose, **kwargs)
        enabled_testtup_list, frame_fpath, all_testflags, module_ = mod_doctest_tup
        for testtup in enabled_testtup_list:
            testflag = testtup[-1]
            if remove_pyc:
                # FIXME python 3 __pycache__/*.pyc
                frame_fpath = frame_fpath.replace('.pyc', '.py')
            frame_rel_fpath = ut.get_relative_modpath(frame_fpath)
            testcmd = ' '.join((pythoncmd, frame_rel_fpath, testflag))
            testcmd_list.append(testcmd)
    return testcmd_list


def parse_docblocks_from_docstr(docstr):
    """
    parse_docblocks_from_docstr
    Depth 5)
    called by parse_doctest_from_docstr

    Args:
        docstr (str):

    Returns:
        list: docstr_blocks

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> import utool as ut
        >>> #import ibeis
        >>> #import ibeis.model.hots.query_request
        >>> #func_or_class = ibeis.model.hots.query_request.QueryParams
        >>> func_or_class = ut.parse_docblocks_from_docstr
        >>> docstr = ut.get_docstr(func_or_class)
        >>> docstr_blocks = parse_docblocks_from_docstr(docstr)
        >>> result = str(docstr_blocks)
        >>> print(result)
    """
    # FIXME Requires tags to be separated by two spaces
    import parse
    import utool as ut
    import itertools
    initial_docblocks = docstr.split('\n\n')
    docblock_len_list = [str_.count('\n') + 2 for str_ in initial_docblocks]
    offset_iter = itertools.chain([0], ut.cumsum(docblock_len_list)[:-1])
    initial_line_offsets = [offset for offset in offset_iter]

    if VERBOSE_TEST:
        if ut.VERBOSE:
            print('__________')
            print('__Initial Docblocks__')
            print('\n---\n'.join(initial_docblocks))
    docstr_blocks = []
    for docblock, line_offset in zip(initial_docblocks, initial_line_offsets):
        docblock = docblock.strip('\n')
        indent = ' ' * ut.get_indentation(docblock)
        parse_result = parse.parse(indent + '{tag}:\n{rest}', docblock)
        if parse_result is not None:
            header = parse_result['tag']
        else:
            header = ''
        docstr_blocks.append((header, docblock, line_offset))
    #print(docstr_blocks)

    docblock_headers = ut.get_list_column(docstr_blocks, 0)
    docblock_bodys = ut.get_list_column(docstr_blocks, 1)
    docblock_offsets = ut.get_list_column(docstr_blocks, 2)

    if VERBOSE_TEST:
        print('[util_test] * found %d docstr_blocks' % (len(docstr_blocks),))
        print('[util_test] * docblock_headers = %r' % (docblock_headers,))
        print('[util_test] * docblock_offsets = %r' % (docblock_offsets,))
        if ut.VERBOSE:
            print('[util_test] * docblock_bodys:')
            print('\n-=-\n'.join(docblock_bodys))
    return docstr_blocks


def parse_doctest_from_docstr(docstr):
    r"""
    because doctest itself doesnt do what I want it to do
    called by get_doctest_examples
    Depth 4)

    CAREFUL, IF YOU GET BACK WRONG RESULTS MAKE SURE YOUR DOCSTR IS PREFFIXED
    WITH R

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> from ibeis.model.hots import score_normalization
        >>> func_or_class = score_normalization.cached_ibeis_score_normalizer
        >>> docstr = ut.get_docstr(func_or_class)
        >>> testsrc_list, testwant_list, testlinenum_list, func_lineno, docstr = get_doctest_examples(func_or_class)
        >>> result = str(len(testsrc_list) + len(testwant_list))
        >>> print(result)
        2
    """
    import utool as ut
    docstr_blocks = parse_docblocks_from_docstr(docstr)

    example_docblocks = []

    for header, docblock, line_offset in docstr_blocks:
        if header.startswith('Example'):
            example_docblocks.append((header, docblock, line_offset))

    testheader_list     = []
    testsrc_list        = []
    testwant_list       = []
    testlineoffset_list = []
    for header, docblock, line_offset in example_docblocks:
        nonheader_src = ut.unindent('\n'.join(docblock.splitlines()[1:]))
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
                    continue
            reversed_src_lines.append(line[4:])
        test_src = '\n'.join(reversed_src_lines[::-1])
        test_want = '\n'.join(reversed_want_lines[::-1])
        testheader_list.append(header)
        testsrc_list.append(test_src)
        testwant_list.append(test_want)
        testlineoffset_list.append(line_offset)
        #print('Parsed header=%r' % header)
        #print('Parsed src=%r' % test_src)
    return testheader_list, testsrc_list, testwant_list, testlineoffset_list


def get_doctest_examples(func_or_class):
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

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> func_or_class = get_doctest_examples
        >>> testsrc_list, testwant_list, testlinenum_list, func_lineno, docstr = get_doctest_examples(func_or_class)
        >>> result = str(len(testsrc_list) + len(testwant_list))
        >>> print(testsrc_list)
        >>> print(testlinenum_list)
        >>> print(func_lineno)
        >>> print(testwant_list)
        >>> print(result)
        6

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> func_or_class = tryimport
        >>> testsrc_list, testwant_list, testlinenum_list, func_lineno, docstr = get_doctest_examples(func_or_class)
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
        >>> testsrc_list, testwant_list, testlinenum_list, func_lineno, docstr = get_doctest_examples(func_or_class)
        >>> result = str(len(testsrc_list) + len(testwant_list))
        >>> print(testsrc_list)
        >>> print(testlinenum_list)
        >>> print(func_lineno)
        >>> print(testwant_list)
        >>> print(result)
        2
    """
    import utool as ut
    if VERBOSE_TEST:
        print('[util_test] parsing %r for doctest' % (func_or_class))

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
    # Cache because my janky parser is slow
    #with ut.GlobalShelfContext('utool') as shelf:
    #    if False and docstr in shelf:
    #        testsrc_list, testwant_list = shelf[docstr]
    #    else:
    testheader_list, testsrc_list, testwant_list, testlineoffset_list = parse_doctest_from_docstr(docstr)
    testlinenum_list = [func_lineno + num_funcdef_lines + offset for offset in testlineoffset_list]
    #       shelf[docstr] = testsrc_list, testwant_list
    if VERBOSE_TEST:
        print('[util_test] * found %d doctests' % (len(testsrc_list),))
    return testsrc_list, testwant_list, testlinenum_list, func_lineno, docstr
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


def doctest_module_list(module_list):
    """
    Runs many module tests

    Entry point for batch run
    Depth 0)
    """
    import utool as ut
    nPass_list = []
    nTotal_list = []
    failed_cmds_list = []
    print('[util_test] Running doctests on module list')

    failed_doctest_fname = 'failed_doctests.txt'
    with open(failed_doctest_fname, 'a') as file_:
        file_.write('\n-------\n\n')
        file_.write(ut.get_printable_timestamp() + '\n')
        file_.write('logfile (only present if logging) = %r\n' % (ut.util_logging.get_current_log_fpath(),))
        testkw = dict(allexamples=True)
        for module in module_list:
            (nPass, nTotal, failed_list) = ut.doctest_funcs(module=module, **testkw)
            nPass_list.append(nPass)
            nTotal_list.append(nTotal)
            failed_cmds_list.append(failed_list)
            # Write failed tests to disk
            for cmd in failed_list:
                file_.write(cmd + '\n')

    nPass = sum(nPass_list)
    nTotal = sum(nTotal_list)
    failed_cmd_list = ut.flatten(failed_cmds_list)
    print('')
    print('+========')
    print('| FINISHED TESTING %d MODULES' % (len(module_list),))
    print('| PASSED %d / %d' % (nPass, nTotal))
    print('L========')
    if len(failed_cmd_list) > 0:
        print('FAILED TESTS:')
        print('\n'.join(failed_cmd_list))


def get_module_doctest_tup(testable_list=None, check_flags=True, module=None,
                             allexamples=None, needs_enable=None, N=0,
                             verbose=True, testslow=False):
    """
    Parses module for testable doctesttups
    Depth 2)
    called by doctest_funcs and get_module_testlines

    Returns:
        ModuleDoctestTup : (enabled_testtup_list, frame_fpath, all_testflags, module)
            enabled_testtup_list (list): a list of testtup
                testtup (tuple): (name, num, src, want, flag1) describes a valid doctest in the module
                    name  (str): test name
                    num   (str): test number of the module / function / class / method
                    src   (str): test source code
                    want  (str): expected test result
                    flag1 (str): a valid commandline flag to enable this test
            frame_fpath (str):
                module fpath that will be tested
            module (module):
                the actual module that will be tested
            all_testflags (list):
                the command line arguments that will enable different tests
    """
    #+------------------------
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
    #L________________________
    #+------------------------
    # Inspect caller module for testable names
    if module is None:
        frame_fpath = '???'
        try:
            frame = ut.get_caller_stack_frame(N=N)
            main_modname = '__main__'
            frame_name  = frame.f_globals['__name__']
            frame_fpath = frame.f_globals['__file__']
            if frame_name == main_modname:
                module = sys.modules[main_modname]
                entry_modname = ut.get_modname_from_modpath(module.__file__)
                #ut.embed()
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
    # Get testable functions
    try:
        if verbose or VERBOSE_TEST:
            print('[util_test] Iterating over module funcs')
            print('[util_test] module =%r' % (module,))

        for key, val in ut.iter_module_doctestable(module):
            docstr = inspect.getdoc(val)
            # FIXME:
            # BUG: We need to verify that this function actually belongs to this
            # module. In util_type ndarray is imported and we try to parse it
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
    #L________________________
    #+------------------------
    # Get testable function examples
    test_sentinals = [
        'ENABLE_DOCTEST',
        #'ENABLE_TEST',
        #'ENABLE_DOCTEST',
        #'ENABLE_UTOOL_DOCTEST',
        #'UTOOL_TEST',
        #'UTOOLTEST'
    ]
    if testslow or ut.get_argflag(('--testall', '--testslow')):
        test_sentinals.append('SLOW_DOCTEST')
    force_enable_testnames = []
    for arg in sys.argv:
        if arg.startswith('--test-'):
            testname = arg[7:].split(':')[0].replace('-', '_')
            force_enable_testnames.append(testname)
    #print(force_enable_testnames)
    sorted_testable = sorted(list(set(testable_list)), key=_get_testable_name)
    testtup_list = []
    # Append each testable example
    for testable in sorted_testable:
        testname = _get_testable_name(testable)
        examples, wants, linenums, func_lineno, docstr = get_doctest_examples(testable)
        if len(examples) > 0:
            for testno , srcwant_tup in enumerate(zip(examples, wants)):
                src, want = srcwant_tup
                src_ = ut.regex_replace('from __future__ import.*$', '', src)
                test_disabled = not any([src_.find(s) >= 0 for s in test_sentinals])
                if needs_enable and test_disabled:
                    if testname not in force_enable_testnames:
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
    #L________________________
    #+------------------------
    # Get enabled (requested) examples
    all_testflags = []
    enabled_testtup_list = []
    distabled_testflags  = []
    subx = ut.get_argval('--subx', type_=int, default=None,
                         help_='Only tests the subxth example')
    for testtup in testtup_list:
        (name, num, src, want) = testtup
        prefix = '--test-'
        flag1 = prefix + name + ':' + str(num)
        flag2 = prefix + name
        flag3 = prefix + name.replace('_', '-') + ':' + str(num)
        flag4 = prefix + name.replace('_', '-')
        testflag = ut.get_argflag((flag1, flag2, flag3, flag4))
        testenabled = TEST_ALL_EXAMPLES  or not check_flags or testflag
        if subx is not None and subx != num:
            continue
        all_testflags.append(flag3)
        if testenabled:
            new_testtup = (name, num, src, want, flag1)
            enabled_testtup_list.append(new_testtup)
        else:
            distabled_testflags.append(flag1)

    mod_doctest_tup = ModuleDoctestTup(enabled_testtup_list, frame_fpath, all_testflags, module)
    #L________________________
    return mod_doctest_tup


def doctest_funcs(testable_list=None, check_flags=True, module=None, allexamples=None,
                  needs_enable=None, strict=False, verbose=True):
    """
    Main entry point into utools main module doctest harness
    Depth 1)

    Args:
        testable_list (list):
        check_flags (bool):
        module (None):
        allexamples (None):
        needs_enable (None):

    Returns:
        tuple: (nPass, nTotal, failed_cmd_list)

    CommandLine:
        python -m ibeis.model.preproc.preproc_chip --all-examples

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
    multiprocessing.freeze_support()  # just in case
    #+-------------------
    if verbose:
        print('[util_test.doctest_funcs] Running doctest funcs')
    # parse out testable doctesttups
    mod_doctest_tup = get_module_doctest_tup(testable_list, check_flags, module,
                                             allexamples, needs_enable, N=1, verbose=verbose)
    enabled_testtup_list, frame_fpath, all_testflags, module  = mod_doctest_tup
    #L__________________
    #+-------------------
    # Run enabled examles
    nPass = 0
    nFail = 0
    failed_flag_list = []
    nTotal = len(enabled_testtup_list)
    for testtup in enabled_testtup_list:
        name, num, src, want, flag = testtup
        print('\n\n')
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')
        print(' ---- DOCTEST ' + name.upper() + ':' + str(num) + '---')
        if PRINT_SRC:
            print(ut.msgblock('EXEC SRC', src))
        test_globals = module.__dict__.copy()
        try:
            testkw = dict(globals=test_globals, want=want)
            testtup
            test_locals = ut.run_test((name,  src, frame_fpath), **testkw)
            is_pass = (test_locals is not False)
            if is_pass:
                nPass += 1
            else:
                raise Exception('failed')
        except Exception:
            nFail += 1
            failed_flag_list.append(flag)
            if strict:
                raise
            pass
    #L__________________
    #+-------------------
    # Print Results
    if nTotal == 0:
        print('No test flags sepcified.')
        print('Please choose one of the following flags or specify --enableall')
        print('Valid test argflags:\n' + '    --allexamples' +
                ut.indentjoin(all_testflags, '\n    '))
    print('+-------')
    print('| finished testing fpath=%r' % (frame_fpath,))
    print('| passed %d / %d' % (nPass, nTotal))
    print('L-------')
    failed_cmd_list = []
    if nFail > 0:
        #modname = module.__name__
        modname = ut.get_modname_from_modpath(frame_fpath)
        failed_cmd_list = ['python -m %s %s' % (modname, flag_)
                            for flag_ in failed_flag_list]
        #failed_cmd_list = ['python %s %s' % (frame_fpath, flag_)
        #                    for flag_ in failed_flag_list]
        print('Failed Tests:')
        print('\n'.join(failed_cmd_list))
    #L__________________
    return (nPass, nTotal, failed_cmd_list)


def run_test(func_or_doctesttup, *args, **kwargs):
    """
    Runs the test function with success / failure printing

    Args:
        func_or_doctesttup (func or tuple): function or doctest tuple

    Varargs/Kwargs:
        Anything that needs to be passed to <func_>
    """
    import utool as ut
    func_is_text = isinstance(func_or_doctesttup, types.TupleType)
    if func_is_text:
        (funcname, src, frame_fpath) = func_or_doctesttup
    else:
        func_ = func_or_doctesttup
        funcname = get_funcname(func_)
        frame_fpath = ut.get_funcfpath(func_)
    upper_funcname = funcname.upper()
    if ut.VERBOSE:
        printTEST('[TEST.BEGIN] %s ' % (sys.executable))
        printTEST('[TEST.BEGIN] %s ' % (funcname,))
    VERBOSE_TIMER = True
    INDENT_TEST   = False
    #print('  <funcname>  ')
    #print('  <' + funcname + '>  ')
    #short_funcname = ut.clipstr(funcname, 8)
    with util_print.Indenter('  <' + funcname + '>  ', enabled=INDENT_TEST):
        try:
            #+----------------
            # RUN THE TEST WITH A TIMER
            with util_time.Timer(upper_funcname, verbose=VERBOSE_TIMER) as timer:
                if func_is_text:
                    test_locals = _exec_doctest(src, kwargs)
                else:
                    # TEST INPUT IS A LIVE PYTHON FUNCTION
                    test_locals = func_(*args, **kwargs)
                print('')
            #L________________
            #+----------------
            # LOG PASSING TEST
            printTEST('[TEST.FINISH] %s -- SUCCESS' % (funcname,))
            if PRINT_FACE:
                print(HAPPY_FACE)
            msg = '%.4fs in %s %s\n' % (
                timer.ellapsed, funcname, frame_fpath)
            try:
                ut.write_to('test_times.txt', msg, mode='a')
                #with open('test_times.txt', 'a') as file_:
                #    file_.write(msg)
            except IOError as ex:
                ut.printex(ex, '[util_test] IOWarning')
            #L________________
            # RETURN VALID TEST LOCALS
            return test_locals

        except Exception as ex:
            import utool as ut
            exc_type, exc_value, tb = sys.exc_info()
            # Get locals in the wrapped function
            ut.printex(ex, tb=True)
            printTEST('[TEST.FINISH] %s -- FAILED:\n    type(ex)=%s\n    ex=%s' % (funcname, type(ex), ex))
            if PRINT_FACE:
                print(SAD_FACE)
            if func_is_text:
                print('Failed in module: %r' % frame_fpath)
                if DEBUG_SRC:
                    src_with_lineno = ut.number_text_lines(src)
                    print(ut.msgblock('FAILED DOCTEST IN %s' % (funcname,), src_with_lineno))
                #ut.embed()
                #print('\n... test encountered error. sys.exit(1)\n')
                #sys.exit(1)
                #failed_execline = traceback.format_tb(tb)[-1]
                #parse_str = 'File {fname}, line {lineno}, in {modname}'
                #parse_dict = parse.parse('{prefix_}' + parse_str + '{suffix_}', failed_execline)
                #if parse_dict['fname'] == '<string>':
                #    lineno = int(parse_dict['lineno'])
                #    failed_line = src.splitlines()[lineno - 1]
                #    print('Failed on line: %s' % failed_line)
            if util_arg.SUPER_STRICT:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                if not func_is_text:
                    # Remove this function from stack strace
                    # dont do this for execed code
                    exc_traceback = exc_traceback.tb_next
                # Python 2*3=6
                #six.reraise(exc_type, exc_value, exc_traceback)
                # PYTHON 2.7 DEPRICATED:
                if six.PY2:
                    raise exc_type, exc_value, exc_traceback.tb_next
                    #exec('raise exc_type, exc_value, exc_traceback.tb_next', globals(), locals())
                # PYTHON 3.3 NEW METHODS
                elif six.PY3:
                    ex = exc_type(exc_value)
                    ex.__traceback__ = exc_traceback.tb_next
                    raise ex
                else:
                    raise AssertionError('Weird python version')
            if SYSEXIT_ON_FAIL:
                print('[util_test] SYSEXIT_ON_FAIL = True')
                print('[util_test] exiting with sys.exit(1)')
                sys.exit(1)
            #raise
            return False


def _exec_doctest(src, kwargs):
    """
    Helper for run_test

    block of code that runs doctest and was too big to be in run_test
    """
    # TEST INPUT IS PYTHON CODE TEXT
    test_locals = {}
    test_globals = kwargs.get('globals', {})
    want = kwargs.get('want', None)
    #test_globals['print'] = doctest_print
    # EXEC FUNC
    #six.exec_(src, test_globals, test_locals)  # adds stack to debug trace
    try:
        exec(src, test_globals, test_locals)
    except ExitTestException:
        print('Test exited before show')
        pass
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
            try:
                import utool as ut
                errmsg1 += ('GOT/EXPECTED/DIFF\n' + ut.get_textdiff(result, want))
            except ImportError:
                errmsg1 += ('REPR_GOT: result=\n%r\n' % (result))
                errmsg1 += ('REPR_EXPECTED: want=\n%r\n' % (want))
                pass
            errmsg1 += ''
            errmsg1 += ('STR_GOT: result=\n%s\n' % (result))
            errmsg1 += ('STR_EXPECTED: want=\n%s\n' % (want))
            raise AssertionError('result != want\n' + errmsg1)
        assert result == want, 'result is not the same as want'
    return test_locals
    #print('\n'.join(output_lines))


def printTEST(msg, wait=False):
    builtins.print('\n=============================')
    builtins.print('**' + msg)
    #if INTERACTIVE and wait:
    # raw_input('press enter to continue')


def tryimport(modname, pipiname=None, ensure=False):
    """
    CommandLine:
        python -m utool.util_tests --test-tryimport

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tests import *   # NOQA
        >>> modname = 'pyfiglet'
        >>> pipiname = 'git+https://github.com/pwaller/pyfiglet'
        >>> pyfiglet = tryimport(modname, pipiname)
        >>> assert pyfiglet is None or isinstance(pyfiglet, types.ModuleType)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tests import *   # NOQA
        >>> modname = 'lru'
        >>> pipiname = 'git+https://github.com/amitdev/lru-dict'
        >>> lru = tryimport(modname, pipiname, ensure=True)
        >>> assert isinstance(lru, types.ModuleType)
    """
    if pipiname is None:
        pipiname = modname
    try:
        module = __import__(modname)
        return module
    except ImportError as ex:
        import utool
        base_pipcmd = 'pip install %s' % pipiname
        if not utool.WIN32:
            pipcmd = 'sudo ' + base_pipcmd
            sudo = True
        else:
            pipcmd = base_pipcmd
            sudo = False
        msg = 'unable to find module %r. Please install: %s' % (str(modname), str(pipcmd))
        print(msg)
        utool.printex(ex, msg, iswarning=True)
        if ensure:
            #raise NotImplementedError('not ensuring')
            utool.cmd(base_pipcmd, sudo=sudo)
            module = tryimport(modname, pipiname, ensure=False)
            if module is None:
                raise AssertionError('Cannot ensure modname=%r please install using %r'  % (modname, pipcmd))
            return module
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
    # TODO: move this function elsewhere
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
        >>> import utool  # NOQA
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


def show_was_requested():
    """
    returns True if --show is specified on the commandline or you are in
    IPython (and presumably want some sort of interaction
    """
    import plottool as pt
    return pt.show_was_requested()
    #import utool as ut
    #return ut.get_argflag('--show') or ut.inIPython()


class ExitTestException(Exception):
    pass


def quit_if_noshow():
    import utool as ut
    if not (ut.get_argflag(('--show', '--save')) or ut.inIPython()):
        raise ExitTestException('This should be caught gracefully by ut.run_test')


def show_if_requested():
    import plottool as pt
    pt.show_if_requested()


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_tests; utool.doctest_funcs(utool.util_tests, allexamples=True)"
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
