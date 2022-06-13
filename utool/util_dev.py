# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import types
import sys
import heapq
import six
import re
import os
import gc
import warnings
import weakref
import itertools as it
import functools
from collections import OrderedDict
from six.moves import input, zip, range, map, reduce
from utool import util_progress
from os.path import splitext, exists, join, split, relpath
from utool import util_inject
from utool import util_dict
from utool import util_const
from utool import util_arg
from utool import util_decor
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
# TODO: remove print_, or grab it dynamically from util_logger
print, rrr, profile = util_inject.inject2(__name__)
print_ = util_inject.make_module_write_func(__name__)


if HAVE_NUMPY:
    INDEXABLE_TYPES = (list, tuple, np.ndarray)
else:
    INDEXABLE_TYPES = (list, tuple)


def overrideable_partial(func, *args, **default_kwargs):
    """ like partial, but given kwargs can be overrideden at calltime """
    import functools
    @functools.wraps(func)
    def partial_wrapper(*given_args, **given_kwargs):
        kwargs = default_kwargs.copy()
        kwargs.update(given_kwargs)
        return func(*(args + given_args), **kwargs)
    return partial_wrapper


def ensure_str_list(input_):
    return [input_] if isinstance(input_, six.string_types) else input_


def _ensure_clipboard_backend():
    import pyperclip
    import utool as ut
    if ut.UNIX:
        backend_order = ['xclip', 'xsel', 'qt', 'gtk']
        for backend in backend_order:
            if getattr(pyperclip, '_ut_clipboard', 'no') == backend:
                break
            elif _check_clipboard_backend(backend):
                pyperclip.set_clipboard(backend)
                pyperclip._ut_clipboard = backend
                break
            else:
                print('warning %r not installed' % (backend,))


def _check_clipboard_backend(backend):
    import pyperclip
    if backend == 'qt':
        try:
            import PyQt4  # NOQA
            return True
        except ImportError:
            return False
    elif backend == 'gtk':
        try:
            import gtk  # NOQA
            return True
        except ImportError:
            return False
    else:
        return pyperclip._executable_exists(backend)


def set_clipboard(text):
    """ alias for copy_text_to_clipboard """
    return copy_text_to_clipboard(text)


def copy_text_to_clipboard(text):
    """
    Copies text to the clipboard

    CommandLine:
        pip install pyperclip
        sudo apt-get install xclip
        sudo apt-get install xsel

    References:
        http://stackoverflow.com/questions/11063458/python-script-to-copy-text-to-clipboard
        http://stackoverflow.com/questions/579687/how-do-i-copy-a-string-to-the-clipboard-on-windows-using-python

    Ignore:
        import pyperclip
        # Qt is by far the fastest, followed by xsel, and then xclip
        #
        backend_order = ['xclip', 'xsel', 'qt', 'gtk']
        backend_order = ['qt', 'xsel', 'xclip', 'gtk']
        for be in backend_order:
            print('be = %r' % (be,))
            pyperclip.set_clipboard(be)
            %timeit pyperclip.copy('a line of reasonable length text')
            %timeit pyperclip.paste()
    """
    import pyperclip
    _ensure_clipboard_backend()
    pyperclip.copy(text)
    # from Tkinter import Tk
    # tk_inst = Tk()
    # tk_inst.withdraw()
    # tk_inst.clipboard_clear()
    # tk_inst.clipboard_append(text)
    # tk_inst.destroy()


def get_clipboard():
    """
    References:
        http://stackoverflow.com/questions/11063458/python-script-to-copy-text-to-clipboard
    """
    # import utool as ut
    import pyperclip
    _ensure_clipboard_backend()
    text = pyperclip.paste()
    # from Tkinter import Tk
    # tk_inst = Tk()
    # tk_inst.withdraw()
    # text = tk_inst.clipboard_get()
    # tk_inst.destroy()
    return text


def get_nonconflicting_string(base_fmtstr, conflict_set, offset=0):
    """
    gets a new string that wont conflict with something that already exists

    Args:
        base_fmtstr (str):
        conflict_set (set):

    CommandLine:
        python -m utool.util_dev --test-get_nonconflicting_string

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> # build test data
        >>> base_fmtstr = 'somestring%d'
        >>> conflict_set = ['somestring0']
        >>> # execute function
        >>> result = get_nonconflicting_string(base_fmtstr, conflict_set)
        >>> # verify results
        >>> print(result)
        somestring1
    """
    # Infinite loop until we find a non-conflict
    conflict_set_ = set(conflict_set)
    for count in it.count(offset):
        base_str = base_fmtstr % count
        if base_str not in conflict_set_:
            return base_str


def get_nonconflicting_path_old(base_fmtstr, dpath, offset=0):
    r"""
    base_fmtstr must have a %d in it
    """
    import utool as ut
    from os.path import basename

    pattern = '*'
    dname_list = ut.glob(dpath, pattern, recursive=False,
                               with_files=True, with_dirs=True)
    conflict_set = set([basename(dname) for dname in dname_list])

    newname = ut.get_nonconflicting_string(base_fmtstr, conflict_set,
                                           offset=offset)
    newpath = join(dpath, newname)
    return newpath


def input_timeout(msg='Waiting for input...', timeout=30):
    """
    FIXME: Function does not work quite right yet.

    Args:
        msg (str):
        timeout (int):

    Returns:
        ?: ans

    References:
        http://stackoverflow.com/questions/1335507/keyboard-input-with-timeout-in-python
        http://home.wlu.edu/~levys/software/kbhit.py
        http://stackoverflow.com/questions/3471461/raw-input-and-timeout/3911560#3911560

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> msg = 'Waiting for input...'
        >>> timeout = 30
        >>> ans = input_timeout(msg, timeout)
        >>> print(ans)
    """
    import sys
    import select
    import time
    ans = None
    print('You have %d seconds to answer!' % timeout)
    print(msg)
    if sys.platform.startswith('win32'):
        import msvcrt
        start_time = time.time()
        instr = ''
        while True:
            if msvcrt.kbhit():
                chr_ = msvcrt.getche()
                if ord(chr_) == 13:  # enter_key
                    # Accept input
                    ans = instr
                    break
                elif ord(chr_) >= 32:  # space_char
                    # Append to input
                    instr += chr_
            ellapsed = time.time() - start_time
            if ellapsed > timeout:
                ans = None
        print('')  # needed to move to next line
    else:
        rlist, o, e = select.select([sys.stdin], [], [], timeout)
        if rlist:
            ans = sys.stdin.readline().strip()
    return ans


def strip_line_comments(code_text, comment_char='#'):
    import utool as ut
    comment_regex = comment_char + ' .[^\n]*$'
    # full line comments
    code_text = ut.regex_replace('^ *' + comment_regex + '\n', '', code_text)
    # inline comments
    code_text = ut.regex_replace('  ' + comment_regex, '', code_text)
    return code_text


def timeit_grid(stmt_list, setup='', iterations=10000, input_sizes=None,
                verbose=True, show=False):
    """
    Timeit::
        import utool as ut
        setup = ut.codeblock(
            '''
            import utool as ut
            from six.moves import range, zip
            import time
            def time_append(size):
                start_time    = time.time()
                last_time     = start_time
                list2 = []
                for x in range(size):
                    now_time    = time.time()
                    between = now_time - last_time
                    last_time   = now_time
                    list2.append(between)

            def time_assign(size):
                start_time    = time.time()
                last_time     = start_time
                list1 = ut.alloc_nones(size)
                for x in range(size):
                    now_time    = time.time()
                    between = now_time - last_time
                    last_time   = now_time
                    list1[x] = between

            def time_baseline(size):
                start_time    = time.time()
                last_time     = start_time
                for x in range(size):
                    now_time    = time.time()
                    between = now_time - last_time
                    last_time   = now_time

            def time_null(size):
                for x in range(size):
                    pass
            ''')

        input_sizes = [2 ** count for count in range(7, 12)]
        stmt_list = ['time_assign', 'time_append', 'time_baseline', 'time_null']
        input_sizes=[100, 1000, 10000]
        ut.timeit_grid(stmt_list, setup, input_sizes=input_sizes, show=True)
    """
    import timeit
    #iterations = timeit.default_number
    if input_sizes is None:
        input_sizes = [2 ** count for count in range(7, 14)]
    time_grid = []
    for size in input_sizes:
        time_list = []
        for stmt in stmt_list:
            stmt_ = stmt + '(' + str(size) + ')'
            if verbose:
                print('running stmt_=%r' % (stmt_,))
            time = timeit.timeit(stmt_, setup=setup, number=iterations)
            if verbose:
                print('... took %r seconds' % (time,))
            time_list.append(time)
        time_grid.append(time_list)

    if show:
        time_grid = np.array(time_grid)
        try:
            import plottool_ibeis as pt
        except ImportError:
            import plottool as pt
        color_list = pt.distinct_colors(len(stmt_list))
        for count, (stmt, color) in enumerate(zip(stmt_list, color_list)):
            pt.plot(input_sizes, time_grid.T[count], 'x-', color=color, label=stmt)
        pt.dark_background()
        pt.legend()
        pt.show_if_requested()
    return time_grid


def extract_timeit_setup(func, doctest_number, sentinal):
    import utool as ut
    # Extract Pre Setup
    testsrc = ut.get_doctest_examples(func)[0][doctest_number]
    srclines1 = testsrc.split('\n')
    srclines2 = []
    for line in srclines1:
        if line.find(ut.get_funcname(func)) > -1:
            break
        srclines2.append(line)
    # Extract Func
    for line in ut.get_func_sourcecode(func, stripdef=True, strip_docstr=True, strip_comments=True).split('\n'):
        if line.strip() == '':
            continue
        if line.find(sentinal) > -1:
            break
        srclines2.append(line)
    setup = '\n'.join(srclines2)
    return setup
    #print(setup)


def timeit_compare(stmt_list, setup='', iterations=100000, verbose=True,
                   strict=False, assertsame=True):
    """
    Compares several statments by timing them and also
    checks that they have the same return value

    Args:
        stmt_list (list) : list of statments to compare
        setup (str) :
        iterations (int) :
        verbose (bool) :
        strict (bool) :

    Returns:
        tuple (bool, list, list) : (passed, time_list, result_list)
            passed (bool): True if all results are the same
            time_list (list): list of times for each statment
            result_list (list): list of results values for each statment

    CommandLine:
        python -m utool.util_dev --exec-timeit_compare

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> import utool as ut
        >>> setup = ut.codeblock(
                '''
                import numpy as np
                rng = np.random.RandomState(0)
                invVR_mats = rng.rand(1000, 3, 3).astype(np.float64)
                ''')
        >>> stmt1 = 'invVR_mats[:, 0:2, 2].T'
        >>> stmt2 = 'invVR_mats.T[2, 0:2]'
        >>> iterations = 1000
        >>> verbose = True
        >>> stmt_list = [stmt1, stmt2]
        >>> ut.timeit_compare(stmt_list, setup=setup, iterations=iterations, verbose=verbose)
    """
    import timeit
    import utool as ut

    stmt_list = [s for s in stmt_list if not s.startswith('#')]

    for stmtx in range(len(stmt_list)):
        # Hacky way of removing assignment and just getting statement
        # We have to make sure it is ok when using it for kwargs
        stmt = stmt_list[stmtx]
        eqpos = stmt.find('=')
        lparen_pos = stmt.find('(')
        if eqpos > 0 and (lparen_pos == -1 or lparen_pos > eqpos):
            stmt = '='.join(stmt.split('=')[1:])
            stmt_list[stmtx] = stmt

    if verbose:
        print('+----------------')
        print('| TIMEIT COMPARE')
        print('+----------------')
        print('| iterations = %d' % (iterations,))
        print('| Input:')
        #print('|     +------------')
        print('|     | num | stmt')
        for count, stmt in enumerate(stmt_list):
            print('|     | %3d | %r' % (count, stmt))
        print('...')
        sys.stdout.flush()
        #print('+     L________________')

    if assertsame:
        result_list = [_testit(stmt, setup) for stmt in stmt_list]
    else:
        result_list = None
    time_list   = [timeit.timeit(stmt, setup=setup, number=iterations)
                   for stmt in stmt_list]

    def numpy_diff_tests(result_list):
        print('Testing numpy arrays')
        shape_list = [a.shape for a in result_list]
        print('shape_list = %r' % (shape_list,))
        sum_list = [a.sum() for a in result_list]
        diff_list = [np.abs(a - b) for a, b in ut.itertwo(result_list)]
        print('diff stats')
        for diffs in diff_list:
            print(ut.repr4(ut.get_stats(diffs, precision=2, use_median=True)))
        print('diff_list = %r' % (diff_list,))
        print('sum_list = %r' % (sum_list,))
        print('passed_list = %r' % (passed_list,))

    if assertsame:
        if ut.list_type(result_list) is np.ndarray:
            passed_list = [np.allclose(*tup) for tup in ut.itertwo(result_list)]
            passed = all(passed_list)
            is_numpy = True
        else:
            passed = ut.allsame(result_list, strict=False)
            is_numpy = False
    else:
        passed = True
    if verbose:
        print('| Output:')
        if not passed and assertsame:
            print('|    * FAILED: results differ between some statements')
            if is_numpy:
                numpy_diff_tests(result_list)
            print('| Results:')
            for result in result_list:
                for count, result in enumerate(result_list):
                    print('<Result %d>' % count)
                    print(result)
                    #print(ut.truncate_str(repr(result)))
                    print('</Result %d>' % count)
            if strict:
                raise AssertionError('Results are not valid')
        else:
            if assertsame:
                print('|    * PASSED: each statement produced the same result')
            else:
                print('|    * PASSED: each statement did not error')
            passed = True
        #print('|    +-----------------------------------')
        print('|    | num | total time | per loop | stmt')
        for count, tup in enumerate(zip(stmt_list, time_list)):
            stmt, time = tup
            print('|    | %3d | %10s | %8s | %s' %
                  (count, ut.seconds_str(time),
                   ut.seconds_str(time / iterations), stmt))
        #print('|    L___________________________________')
        if verbose:
            print('L_________________')
        return (passed, time_list, result_list)


def _testit(stmt, setup):
    # Make temporary locals/globals for a sandboxlike run
    _globals = {}
    try:
        exec(setup, _globals)
    except Exception as ex:
        import utool
        print('Setup Error')
        print(setup)
        print('---')
        utool.printex(ex, 'error executing setup', keys=['setup'])
        raise
    try:
        result = eval(stmt, _globals)
    except Exception as ex:
        import utool
        print('Statement Error')
        print(setup)
        print('---')
        print(stmt)
        utool.printex(ex, 'error executing statement', keys=['stmt'])
        raise
    return result


def memory_dump():
    """
    References:
       http://stackoverflow.com/questions/141351/how-do-i-find-what-is-using-memory-in-a-python-process-in-a-production-system
    """
    import cPickle
    dump = open("memory.pickle", 'w')
    for obj in gc.get_objects():
        i = id(obj)
        size = sys.getsizeof(obj, 0)
        #    referrers = [id(o) for o in gc.get_referrers(obj) if hasattr(o, '__class__')]
        referents = [id(o) for o in gc.get_referents(obj) if hasattr(o, '__class__')]
        if hasattr(obj, '__class__'):
            cls = str(obj.__class__)
            cPickle.dump({'id': i, 'class': cls, 'size': size, 'referents': referents}, dump)


def _disableable(func):
    def _wrp_disableable(self, *args, **kwargs):
        if self.disabled:
            return
        return func(self, *args, **kwargs)
    return _wrp_disableable


ENABLE_MEMTRACK = False
ENABLE_MEMTRACK = util_arg.get_argflag('--enable-memtrack')


class MemoryTracker(object):
    """
    A ``class`` for tracking memory usage.
    On initialization it logs the current available (free) memory.
    Calling the report method logs the current available memory as well
    as memory usage difference w.r.t the last report.

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool
        >>> import numpy as np
        >>> memtrack = utool.MemoryTracker('[ENTRY]')
        >>> memtrack.report('[BEFORE_CREATE]')
        >>> arr = np.ones(128 * (2 ** 20), dtype=np.uint8)
        >>> memtrack.report('[AFTER_CREATE]')
        >>> memtrack.track_obj(arr, 'arr')
        >>> memtrack.report_objs()
        >>> memtrack.report_largest()
        >>> del arr
        >>> memtrack.report('[DELETE]')
        #>>> memtrack.report_largest()
    """
    def __init__(self, lbl='Memtrack Init', disable=None,
                 avail=True, used=True, total_diff=True, abs_mag=True, peak=True):
        if disable is None:
            disable = ENABLE_MEMTRACK
        self.disabled = disable  # disable by default
        # self.init_peak_nBytes = self.get_peak_memory()
        # self.init_available_nBytes = self.get_available_memory()
        # self.init_used_nBytes = self.get_used_memory()

        self.keys = []
        if avail:
            self.keys.append('avail')
        if used:
            self.keys.append('used')
        if peak:
            self.keys.append('peak')

        # self.prev_available_nBytes = None
        # self.prev_peak_nBytes = None
        # self.prev_used_nBytes = None

        self.weakref_dict = {}  # weakref.WeakValueDictionary()
        self.weakref_dict2 = {}

        self.avail = avail
        self.used = used
        self.total_diff = total_diff
        self.abs_mag = abs_mag

        self.init_measures = self.measure_memory()
        self.prev_measures = None

        from collections import defaultdict
        self.records = defaultdict(list)
        self.report(lbl)

    def measure_memory(self):
        funcs = {
            'peak': self.get_peak_memory,
            'avail': self.get_available_memory,
            'used': self.get_used_memory,
        }
        return {key: funcs[key]() for key in self.keys}

    @_disableable
    def __call__(self, lbl=''):
        self.report(lbl=lbl)

    def record(self):
        import time
        self.collect()

        measures = self.measure_memory()
        for key in self.keys:
            now_nBytes = measures[key]
            if self.prev_measures is not None:
                diff = now_nBytes - self.init_measures[key]
            else:
                diff = None
            total_diff = now_nBytes - self.init_measures[key]
            self.records['diff_' + key].append(diff)
            self.records['total_diff_' + key].append(total_diff)
            self.records['nBytes_' + key].append(now_nBytes)

        # self.records['diff_avail'].append(diff_avail)
        # self.records['diff_used'].append(diff_used)
        # self.records['total_diff_avail'].append(total_diff_avail)
        # self.records['total_diff_used'].append(total_diff_used)
        # self.records['available_nBytes'].append(available_nBytes)
        # self.records['used_nBytes'].append(used_nBytes)
        self.records['time'].append(time.time())

        self.prev_measures = measures
        # self.prev_available_nBytes = available_nBytes
        # self.prev_used_nBytes = used_nBytes

    @_disableable
    def report(self, lbl=''):
        if self.disabled:
            return
        from utool.util_str import byte_str2
        self.record()

        def _byte(num):
            return 'None' if num is None else ut.byte_str2(num)
        try:
            import utool as ut
            row_data = [
                [_byte(self.records['diff_' + key][-1]) for key in self.keys],
                [_byte(self.records['total_diff_' + key][-1]) for key in self.keys],
                [_byte(self.records['nBytes_' + key][-1]) for key in self.keys],
            ]
            row_headers = ['prev_diff', 'total_diff', 'absolute']
            col_headers = self.keys
            table = ut.CSV(row_data, row_headers, col_headers)
            if lbl:
                print(lbl)
            print(table.transpose().tabulate())
            self.report_objs()
        except ImportError:
            diff_avail = self.records['diff_avail'][-1]
            diff_used = self.records['diff_used'][-1]
            total_diff_avail = self.records['total_diff_avail'][-1]
            total_diff_used = self.records['total_diff_used'][-1]
            available_nBytes = self.records['nBytes_avail'][-1]
            used_nBytes = self.records['nBytes_used'][-1]
            print('[memtrack] +----')
            lbl_ = '[%s] ' % (lbl,) if lbl else ''

            if diff_avail is not None:
                if self.avail:
                    print('[memtrack] | %sdiff(avail) = %s' % (lbl_, byte_str2(diff_avail)))
            else:
                print('[memtrack] | new MemoryTracker(%s)' % (lbl,))
            if diff_used is not None:
                if self.used:
                    print('[memtrack] | %sdiff(used) = %s' % (lbl_, byte_str2(diff_used)))

            if self.total_diff:
                if self.avail:
                    print('[memtrack] | Total diff(avail) = %s' % (byte_str2(total_diff_avail)))
                if self.used:
                    print('[memtrack] | Total diff(used) = %s' % (byte_str2(total_diff_used)))
            if self.abs_mag:
                if self.avail:
                    print('[memtrack] | Available Memory = %s' %  (byte_str2(available_nBytes),))
                if self.used:
                    print('[memtrack] | Used Memory      = %s' %  (byte_str2(used_nBytes),))
            self.report_objs()
            print('[memtrack] L----')

    @_disableable
    def get_peak_memory(self):
        from utool import util_resources
        return util_resources.peak_memory()

    @_disableable
    def get_available_memory(self):
        from utool import util_resources
        return util_resources.available_memory()

    @_disableable
    def get_used_memory(self):
        from utool import util_resources
        return util_resources.current_memory_usage()

    @_disableable
    def track_obj(self, obj, name):
        oid = id(obj)
        if not isinstance(obj, weakref.ref):
            obj = weakref.ref(obj)
        #obj_weakref = weakref.ref(obj)
        self.weakref_dict[oid] = obj
        self.weakref_dict2[oid] = name
        del obj

    @_disableable
    def report_obj(self, obj, name=None):
        if not isinstance(obj, weakref.ref):
            obj = weakref.ref(obj)
        report_memsize(obj, name)

    def report_type(self, class_, more=False):
        # Get existing objects of the requested type
        existing_objs = [obj for obj in gc.get_objects() if isinstance(obj, class_)]
        print('There are %d objects of type %s using %s' % (
            len(existing_objs), class_, get_object_size_str(existing_objs)))
        if more:
            for obj in existing_objs:
                self.report_obj(obj)

    @_disableable
    def report_objs(self):
        if len(self.weakref_dict) == 0:
            return
        import utool
        with utool.Indenter('[memtrack] '):
            #print('[memtrack] +----')
            for oid in self.weakref_dict.iterkeys():
                obj = self.weakref_dict[oid]
                if not isinstance(obj, weakref.ref):
                    obj = weakref.ref(obj)
                name = self.weakref_dict2[oid]
                report_memsize(obj, name)
                del obj
        #print('[memtrack] L----')

    @_disableable
    def report_largest(self):
        # Doesnt quite work yet
        import numpy as np
        import gc
        import utool
        print('reporting largest')
        obj_list = gc.get_objects()
        #simple_size_list = np.array([sys.getsizeof(obj) for obj in obj_list])
        #shortlist_size = 20
        #sortx = simple_size_list.argsort()[::-1][0:shortlist_size]
        #simple_size_sorted = simple_size_list[sortx]
        #obj_sorted = [obj_list[x] for x in sortx]
        #for obj, size in zip(obj_sorted, simple_size_sorted):
        #    print('size = %r, type(obj) = %r' % (utool.byte_str2(size), type(obj)))

        print('reporting largets ndarrays')
        ndarray_list = [obj for obj in obj_list if isinstance(obj, np.ndarray)]
        ndarray_list = [obj for obj in obj_list if str(type(obj)).find('array') > -1]
        size_list = np.array([utool.get_object_nbytes(obj) for obj in ndarray_list])
        sortx = size_list.argsort()[::-1]
        ndarray_sorted = [ndarray_list[x] for x in sortx]
        for obj, size in zip(ndarray_sorted, size_list):
            print('size = %r, type(obj) = %r' % (utool.byte_str2(size), type(obj)))

        #size_list = [utool.get_object_nbytes(obj) for obj in obj_list]
        pass

    @_disableable
    def collect(self):
        gc.collect()


def report_memsize(obj, name=None, verbose=True):
    #import types
    import utool as ut
    if name is None:
        name = 'obj'

    if not isinstance(obj, weakref.ref):
        obj = weakref.ref(obj)

    if obj() is None:
        with ut.Indenter('|   '):
            print('+----')
            print('Memsize: ')
            print('type(%s) = %r' % (name, type(obj())))
            print('%s has been deallocated' % name)
            print('L____')
            return

    referents = gc.get_referents(obj())
    referers  = gc.get_referrers(obj())
    with ut.Indenter('|   '):
        print('+----')
        print('Memsize: ')
        print('type(%s) = %r' % (name, type(obj())))
        print('%s is using: %s' % (name, ut.get_object_size_str(obj())))
        print('%s has %d referents' % (name, len(referents)))
        print('%s has %d referers' % (name, len(referers)))
        if verbose:
            if len(referers) > 0:
                for count, referer in enumerate(referers):
                    print('  <Referer %d>' % count)
                    print('    type(referer) = %r' % type(referer))
                    try:
                        #if isinstance(referer, frames.FrameType)
                        print('    frame(referer).f_code.co_name = %s' % (referer.f_code.co_name))
                    except Exception:
                        pass
                    try:
                        #if isinstance(referer, frames.FrameType)
                        print('    func(referer).func_name = %s' % (referer.func_name))
                    except Exception:
                        pass
                    try:
                        #if isinstance(referer, frames.FrameType)
                        print('    len(referer) = %s' % (len(referer)))
                    except Exception:
                        pass
                    if isinstance(referer, dict):
                        print('    len(referer) = %r' % len(referer))
                        if len(referer) < 30:
                            keystr = ut.packstr(repr(referer.keys()), 60, newline_prefix='        ')
                            print('    referer.keys = %s' % (keystr),)
                    print('    id(referer) = %r' % id(referer))
                    #print('referer = ' + ut.truncate_str(repr(referer)))
                    print('  </Referer %d>' % count)
        del obj
        del referents
        del referers
        print('L____')


LIVE_INTERACTIVE_ITER = None


class InteractivePrompt(object):
    def __init__(self, actions={}):
        """
        Args:
            actions (dict): mapping from function to
                tuple of keys / help dictionary
        """
        # for func in list(actions.keys()):
        #     val = actions[func]
        #     if isinstance(val
        self.actions = actions

    def handle_ans(self, ans_):
        """
        preforms an actionm based on a user answer
        """
        ans = ans_.strip(' ')
        def chack_if_answer_was(valid_keys):
            return any([ans == key or ans.startswith(key + ' ')
                        for key in valid_keys])
        # Custom interactions
        for func, tup in self.actions.items():
            valid_keys = tup[0]
            if chack_if_answer_was(valid_keys):
                func()

    def prompt(self):
        import utool as ut
        def _or_phrase(list_):
            return ut.conj_phrase(ut.lmap(repr, map(str, list_)), 'or')
        msg_list = ['enter %s to %s' % (_or_phrase(tup[0]), tup[1])
                    for tup in self.actions.values()]
        msg = ut.indentjoin(msg_list, '\n | * ')
        msg = ''.join([' +-----------', msg, '\n L-----------\n'])
        # TODO: timeout, help message
        print(msg)
        ans = input().strip()
        return ans

    def register(self, tup):
        def _wrp(func):
            self.actions[func] = tup
        return _wrp

    def loop(self):
        @self.register((['quit', 'q'], ''))
        def quit_action():
            raise StopIteration
        while True:
            ans = self.prompt()
            self.handle_ans(ans)


class InteractiveIter(object):
    """
    Choose next value interactively

    iterable should be a list, not a generator. sorry

    """
    def __init__(iiter, iterable=None, enabled=True, startx=0,
                 default_action='next', custom_actions=[], wraparound=False,
                 display_item=False, verbose=True):
        r"""
        Args:
            iterable (None): (default = None)
            enabled (bool): (default = True)
            startx (int): (default = 0)
            default_action (str): (default = 'next')
            custom_actions (list): list of 4-tuple (name, actions, help, func) (default = [])
            wraparound (bool): (default = False)
            display_item (bool): (default = True)
            verbose (bool):  verbosity flag(default = True)

        CommandLine:
            python -m utool.util_dev --test-InteractiveIter.__init__

        Example:
            >>> # DISABLE_DOCTEST
            >>> from utool.util_dev import *  # NOQA
            >>> iterable = [1, 2, 3]
            >>> enabled = True
            >>> startx = 0
            >>> default_action = 'next'
            >>> custom_actions = []
            >>> wraparound = False
            >>> display_item = True
            >>> verbose = True
            >>> iiter = InteractiveIter(iterable, enabled, startx, default_action, custom_actions, wraparound, display_item, verbose)
            >>> for _ in iiter:
            >>>     pass
        """
        import utool as ut
        iiter.wraparound = wraparound
        iiter.enabled = enabled
        iiter.iterable = iterable

        for actiontup in custom_actions:
            if isinstance(custom_actions, tuple):
                pass
            else:
                pass

        iiter.custom_actions = ut.get_list_column(custom_actions, [0, 1, 2])
        iiter.custom_funcs = ut.get_list_column(custom_actions, 3)
        iiter.action_tuples = [
            # (name, list, help)
            ('next',   ['n'], 'move to the next index'),
            ('prev',   ['p'], 'move to the previous index'),
            ('reload', ['r'], 'stay at the same index'),
            ('index',  ['x', 'i', 'index'], 'move to that index'),
            ('set',    ['set'], 'set current index value'),
            ('ipy',    ['ipy', 'ipython', 'cmd'], 'start IPython'),
            ('quit',   ['q', 'exit', 'quit'], 'quit'),
        ] + iiter.custom_actions
        default_action_index = ut.get_list_column(iiter.action_tuples, 0).index(default_action)
        iiter.action_tuples[default_action_index][1].append('')
        iiter.action_keys = {tup[0]: tup[1] for tup in iiter.action_tuples}
        iiter.index = startx
        iiter.display_item = display_item
        iiter.verbose = verbose
        pass

    @classmethod
    def eventloop(cls, custom_actions=[]):
        """
        For use outside of iteration wrapping. Makes an interactive event loop
        custom_actions should be specified in format
        [dispname, keys, desc, func]
        """
        iiter = cls([None], custom_actions=custom_actions, verbose=False)
        print('[IITER] Begining interactive main loop')
        for _ in iiter:
            pass
        return iiter

    def __iter__(iiter):
        global LIVE_INTERACTIVE_ITER
        LIVE_INTERACTIVE_ITER = iiter
        import utool as ut
        if not iiter.enabled:
            for item in ut.ProgressIter(iiter.iterable, lbl='nointeract: ', freq=1, adjust=False):
                yield item
            raise StopIteration()
        assert isinstance(iiter.iterable, INDEXABLE_TYPES), 'input is not iterable'
        iiter.num_items = len(iiter.iterable)
        if iiter.verbose:
            print('[IITER] Begin interactive iteration: %r items\n' % (iiter.num_items))
        if iiter.num_items == 0:
            raise StopIteration
        # TODO: replace with ub.ProgIter
        mark_, end_ = util_progress.log_progress(length=iiter.num_items,
                                                 lbl='interaction: ', freq=1)
        prompt_on_start = False
        if prompt_on_start:
            ans = iiter.prompt()
            action = iiter.handle_ans(ans)
            REFRESH_ON_BAD_INPUT = False
            if not REFRESH_ON_BAD_INPUT:
                while action is False:
                    ans = iiter.prompt()
                    action = iiter.handle_ans(ans)
            if action == 'IPython':
                ut.embed(N=1)

        while True:
            if iiter.verbose:
                print('')
            if iiter.wraparound:
                iiter.index = iiter.index % len(iiter.iterable)
            if iiter.index >= len(iiter.iterable):
                if iiter.verbose:
                    print('Got to end the end of the iterable')
                break
            mark_(iiter.index)
            item = iiter.iterable[iiter.index]
            if iiter.verbose:
                print('')
            yield item
            if iiter.verbose:
                print('')
            mark_(iiter.index)
            if iiter.verbose:
                print('')
                print('[IITER] current index=%r' % (iiter.index,))
                if iiter.display_item:
                    print('[IITER] current item=%r' % (item,))
            ans = iiter.prompt()
            action = iiter.handle_ans(ans)
            REFRESH_ON_BAD_INPUT = False
            if not REFRESH_ON_BAD_INPUT:
                while action is False:
                    ans = iiter.prompt()
                    action = iiter.handle_ans(ans)
            if action == 'IPython':
                ut.embed(N=1)
        end_()
        print('Ended interactive iteration')
        LIVE_INTERACTIVE_ITER = None

    def handle_ans(iiter, ans_):
        """
        preforms an actionm based on a user answer
        """
        ans = ans_.strip(' ')
        def parse_str_value(ans):
            return ' '.join(ans.split(' ')[1:])
        def chack_if_answer_was(valid_keys):
            return any([ans == key or ans.startswith(key + ' ') for key in valid_keys])
        # Handle standard actions
        if ans in iiter.action_keys['quit']:
            raise StopIteration()
        elif ans in iiter.action_keys['prev']:
            iiter.index -= 1
        elif ans in iiter.action_keys['next']:
            iiter.index += 1
        elif ans in iiter.action_keys['reload']:
            iiter.index += 0
        elif chack_if_answer_was(iiter.action_keys['index']):
            try:
                iiter.index = int(parse_str_value(ans))
            except ValueError:
                print('Unknown ans=%r' % (ans,))
        elif chack_if_answer_was(iiter.action_keys['set']):
            try:
                iiter.iterable[iiter.index] = eval(parse_str_value(ans))
            except ValueError:
                print('Unknown ans=%r' % (ans,))
        elif ans in iiter.action_keys['ipy']:
            return 'IPython'
        else:
            # Custom interactions
            for func, tup in zip(iiter.custom_funcs, iiter.custom_actions):
                key = tup[0]
                if chack_if_answer_was(iiter.action_keys[key]):
                    value  = parse_str_value(ans)
                    # cal custom function
                    print('Calling custom action func')
                    import utool as ut
                    argspec = ut.get_func_argspec(func)
                    if len(argspec.args) == 3:
                        # Forgot why I had custom functions take args in the first place
                        func(iiter, key, value)
                    else:
                        func()
                    # Custom funcs dont cause iteration
                    return False
            print('Unknown ans=%r' % (ans,))
            return False
        return True

    def prompt(iiter):
        import utool as ut
        def _or_phrase(list_):
            return ut.conj_phrase(ut.lmap(repr, map(str, list_)), 'or')
        msg_list = ['enter %s to %s' % (_or_phrase(tup[1]), tup[2])
                    for tup in iiter.action_tuples]
        msg = ut.indentjoin(msg_list, '\n | * ')
        msg = ''.join([' +-----------', msg, '\n L-----------\n'])
        # TODO: timeout, help message
        print(msg)
        ans = iiter.wait_for_input()
        return ans

    def wait_for_input(iiter):
        ans = input().strip()
        return ans

    def __call__(iiter, iterable=None):
        iiter.iterable = iterable


def user_cmdline_prompt(msg=''):
    #prompt_fmtstr = ut.codeblock(
    #    '''
    #    Accept system decision?
    #    ==========
    #    {decidemsg}
    #    ==========
    #    Enter {no_phrase} to reject
    #    Enter {embed_phrase} to embed into ipython
    #    Any other inputs accept system decision
    #    (input is case insensitive)
    #    '''
    #)
    #ans_list_embed = ['cmd', 'ipy', 'embed']
    #ans_list_no = ['no', 'n']
    ##ans_list_yes = ['yes', 'y']
    #prompt_str = prompt_fmtstr.format(
    #    no_phrase=ut.conj_phrase(ans_list_no),
    #    embed_phrase=ut.conj_phrase(ans_list_embed),
    #    decidemsg=decidemsg
    #)
    #prompt_block = ut.msgblock('USER_INPUT', prompt_str)
    msg += '\n... Enter yes to accept or anything else to reject\n'
    ans = input(msg)
    return ans == 'yes'


def are_you_sure(msg=''):
    r"""
    Prompts user to accept or checks command line for -y

    Args:
        msg (str):

    Returns:
        bool: accept or not
    """
    print(msg)
    from utool import util_arg
    from utool import util_str
    override = util_arg.get_argflag(('--yes', '--y', '-y'))
    if override:
        print('accepting based on command line flag')
        return True
    valid_ans = ['yes', 'y']
    valid_prompt = util_str.conj_phrase(valid_ans, 'or')
    ans = input('Are you sure?\n Enter %s to accept\n' % valid_prompt)
    return ans.lower() in valid_ans


def grace_period(msg='', seconds=10):
    """
    Gives user a window to stop a process before it happens
    """
    import time
    print(msg)
    override = util_arg.get_argflag(('--yes', '--y', '-y'))
    print('starting grace period')
    if override:
        print('ending based on command line flag')
        return True
    for count in reversed(range(1, seconds + 1)):
        time.sleep(1)
        print('%d' % (count,))
    print('%d' % (0,))
    print('grace period is over')
    return True


def delayed_retry_gen(delay_schedule=[.1, 1, 10], msg=None, timeout=None, raise_=True):
    """ template code for a infinte retry loop """
    import utool as ut
    import time
    if not ut.isiterable(delay_schedule):
        delay_schedule = [delay_schedule]
    tt = ut.tic()

    # First attempt is immediate
    yield 0

    for count in it.count(0):
        #print('count = %r' % (count,))
        if timeout is not None and ut.toc(tt) > timeout:
            if raise_:
                raise Exception('Retry loop timed out')
            else:
                raise StopIteration('Retry loop timed out')
        index = min(count, len(delay_schedule) - 1)
        delay = delay_schedule[index]
        time.sleep(delay)
        yield count + 1


def tuples_to_unique_scalars(tup_list):
    seen = {}
    def addval(tup):
        val = len(seen)
        seen[tup] = val
        return val
    scalar_list = [seen[tup] if tup in seen else addval(tup) for tup in tup_list]
    return scalar_list

STAT_KEY_ORDER = ['max', 'min', 'mean', 'sum', 'std', 'nMin', 'nMax', 'shape', 'num_nan']


def get_jagged_stats(arr_list, **kwargs):
    r"""
    Args:
        arr_list (list):

    Returns:
        dict: stats_dict

    CommandLine:
        python -m utool.util_dev --test-get_jagged_stats

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> import utool as ut
        >>> kwargs = dict(use_nan=True)
        >>> arr_list = [[1, 2, 3, 4], [3, 10], [np.nan, 3, 3, 3]]
        >>> stats_dict = get_jagged_stats(arr_list, **kwargs)
        >>> result = ut.align(str(ut.repr4(stats_dict)), ':')
        >>> print(result)
        {
            'mean'   : [2.5, 6.5, 3.0],
            'std'    : [1.118034, 3.5, 0.0],
            'max'    : [4.0, 10.0, 3.0],
            'min'    : [1.0, 3.0, 3.0],
            'nMin'   : [1, 1, 3],
            'nMax'   : [1, 1, 3],
            'shape'  : ['(4,)', '(2,)', '(4,)'],
            'num_nan': [0, 0, 1],
        }

    """
    import functools
    stats_dict_list = list(map(functools.partial(get_stats, **kwargs), arr_list))
    stats_dict = util_dict.dict_stack(stats_dict_list)
    # Fix order
    stats_dict = util_dict.order_dict_by(stats_dict, STAT_KEY_ORDER)
    return stats_dict


def get_stats(list_, axis=None, use_nan=False, use_sum=False, use_median=False,
              size=False):
    """
    Args:
        list_ (listlike): values to get statistics of
        axis (int): if `list_` is ndarray then this specifies the axis

    Returns:
        OrderedDict: stats: dictionary of common numpy statistics
            (min, max, mean, std, nMin, nMax, shape)

    SeeAlso:
        get_stats_str

    CommandLine:
        python -m utool.util_dev --test-get_stats
        python -m utool.util_dev --test-get_stats:1

    Examples0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> import numpy as np
        >>> import utool
        >>> axis = 0
        >>> np.random.seed(0)
        >>> list_ = np.random.rand(10, 2).astype(np.float32)
        >>> stats = get_stats(list_, axis, use_nan=False)
        >>> result = str(utool.repr4(stats, nl=1, precision=4, with_dtype=True))
        >>> print(result)
        {
            'mean': np.array([0.5206, 0.6425], dtype=np.float32),
            'std': np.array([0.2854, 0.2517], dtype=np.float32),
            'max': np.array([0.9637, 0.9256], dtype=np.float32),
            'min': np.array([0.0202, 0.0871], dtype=np.float32),
            'nMin': np.array([1, 1], dtype=np.int32),
            'nMax': np.array([1, 1], dtype=np.int32),
            'shape': (10, 2),
        }

    Examples1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> import numpy as np
        >>> import utool
        >>> axis = 0
        >>> rng = np.random.RandomState(0)
        >>> list_ = rng.randint(0, 42, size=100).astype(np.float32)
        >>> list_[4] = np.nan
        >>> stats = get_stats(list_, axis, use_nan=True)
        >>> result = str(utool.repr2(stats, precision=1, strkeys=True))
        >>> print(result)
        {mean: 20.0, std: 13.2, max: 41.0, min: 0.0, nMin: 7, nMax: 3, shape: (100,), num_nan: 1}
    """
    datacast = np.float32
    # Assure input is in numpy format
    if isinstance(list_, np.ndarray):
        nparr = list_
    elif isinstance(list_, list):
        nparr = np.array(list_)
    else:
        nparr = np.array(list(list_))
    # Check to make sure stats are feasible
    if len(nparr) == 0:
        stats = OrderedDict([('empty_list', True)])
        if size:
            stats['size'] = 0
    else:
        if use_nan:
            min_val = np.nanmin(nparr, axis=axis)
            max_val = np.nanmax(nparr, axis=axis)
            mean_ = np.nanmean(nparr, axis=axis)
            std_  = np.nanstd(nparr, axis=axis)
        else:
            min_val = nparr.min(axis=axis)
            max_val = nparr.max(axis=axis)
            mean_ = nparr.mean(axis=axis)
            std_  = nparr.std(axis=axis)
        # number of entries with min/max val
        nMin = np.sum(nparr == min_val, axis=axis)
        nMax = np.sum(nparr == max_val, axis=axis)
        stats = OrderedDict([
            ('mean',  datacast(mean_)),
            ('std',   datacast(std_)),
            ('max',   (max_val)),
            ('min',   (min_val)),
            ('nMin',  np.int32(nMin)),
            ('nMax',  np.int32(nMax)),
        ])
        if size:
            stats['size'] = nparr.size
        else:
            if isinstance(nparr.shape, tuple):
                # python 2.7 issue
                shape = tuple(map(int, nparr.shape))
            else:
                shape = nparr.shape
            stats['shape'] = shape
        if use_median:
            stats['med'] = np.nanmedian(nparr)
        if use_nan:
            stats['num_nan'] = np.isnan(nparr).sum()
        if use_sum:
            sumfunc = np.nansum if use_nan else np.sum
            stats['sum'] = sumfunc(nparr, axis=axis)
    return stats


get_statdict = get_stats
stats_dict = get_stats


def set_overlaps(set1, set2, s1='s1', s2='s2'):
    import utool as ut
    set1 = set(set1)
    set2 = set(set2)
    overlaps = ut.odict([
        (s1, len(set1)),
        (s2, len(set2)),
        ('isect', len(set1.intersection(set2))),
        ('union', len(set1.union(set2))),
        ('%s - %s' % (s1, s2), len(set1.difference(set2))),
        ('%s - %s' % (s2, s1), len(set2.difference(set1))),
    ])
    return overlaps


def set_overlap_items(set1, set2, s1='s1', s2='s2'):
    import utool as ut
    set1 = set(set1)
    set2 = set(set2)
    overlaps = ut.odict([
        ('len({})'.format(s1), len(set1)),
        ('len({})'.format(s2), len(set2)),
        ('isect', set1.intersection(set2)),
        ('union', set1.union(set2)),
        ('%s - %s' % (s1, s2), set1.difference(set2)),
        ('%s - %s' % (s2, s1), set2.difference(set1)),
    ])
    return overlaps

get_overlaps = set_overlaps
get_setdiff_info = set_overlaps
get_setdiff_items = set_overlap_items

# --- Info Strings ---


def get_stats_str(list_=None, newlines=False, keys=None, exclude_keys=[], lbl=None,
                  precision=None, axis=0, stat_dict=None, use_nan=False,
                  align=False, use_median=False, **kwargs):
    """
    Returns the string version of get_stats

    DEPRICATE in favor of ut.repr3(ut.get_stats(...))

    if keys is not None then it only displays chosen keys
    excluded keys are always removed


    CommandLine:
        python -m utool.util_dev --test-get_stats_str

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> list_ = [1, 2, 3, 4, 5]
        >>> newlines = False
        >>> keys = None
        >>> exclude_keys = []
        >>> lbl = None
        >>> precision = 2
        >>> stat_str = get_stats_str(list_, newlines, keys, exclude_keys, lbl, precision)
        >>> result = str(stat_str)
        >>> print(result)
        {'mean': 3, 'std': 1.41, 'max': 5, 'min': 1, 'nMin': 1, 'nMax': 1, 'shape': (5,)}

    SeeAlso:
        repr2
        get_stats
    """
    from utool.util_str import repr4
    import utool as ut
    # Get stats dict
    if stat_dict is None:
        stat_dict = get_stats(list_, axis=axis, use_nan=use_nan, use_median=use_median)
    else:
        stat_dict = stat_dict.copy()
    # Keep only included keys if specified
    if keys is not None:
        for key in list(six.iterkeys(stat_dict)):
            if key not in keys:
                del stat_dict[key]
    # Remove excluded keys
    for key in exclude_keys:
        if key in stat_dict:
            del stat_dict[key]
    # apply precision
    statstr_dict = stat_dict.copy()
    #precisionless_types =  (bool,) + six.string_types
    if precision is not None:
        assert ut.is_int(precision), 'precision must be an integer'
        float_fmtstr = '%.' + str(precision) + 'f'
        for key in list(six.iterkeys(statstr_dict)):
            val = statstr_dict[key]
            isfloat = ut.is_float(val)
            if not isfloat and isinstance(val, list):
                type_list = list(map(type, val))
                if len(type_list) > 0 and ut.allsame(type_list):
                    if ut.is_float(val[0]):
                        isfloat = True
                        val = np.array(val)
            if isfloat:
                if isinstance(val, np.ndarray):
                    strval = str([float_fmtstr % v for v in val]).replace('\'', '').lstrip('u')
                    #np.array_str((val), precision=precision)
                else:
                    strval = float_fmtstr % val
                if not strval.startswith('0'):
                    strval = strval.rstrip('0')
                    strval = strval.rstrip('.')
                statstr_dict[key] = strval
            else:
                if isinstance(val, np.ndarray):
                    strval = repr(val.tolist())
                else:
                    strval = str(val)
                statstr_dict[key] = strval

    # format the dictionary string
    stat_str  = repr4(statstr_dict, strvals=True, newlines=newlines)
    # add a label if requested
    if lbl is True:
        lbl = ut.get_varname_from_stack(list_, N=1)  # fancy
    if lbl is not None:
        stat_str = 'stats_' + lbl + ' = ' + stat_str
    if align:
        stat_str = ut.align(stat_str, ':')
    return stat_str


#expected_type = np.float32
#expected_dims = 5
def numpy_list_num_bits(nparr_list, expected_type, expected_dims):
    num_bits = 0
    num_items = 0
    num_elemt = 0
    bit_per_item = {
        np.float32: 32,
        np.uint8: 8
    }[expected_type]
    for nparr in iter(nparr_list):
        arr_len, arr_dims = nparr.shape
        if nparr.dtype.type is not expected_type:
            msg = 'Expected Type: ' + repr(expected_type)
            msg += 'Got Type: ' + repr(nparr.dtype)
            raise Exception(msg)
        if arr_dims != expected_dims:
            msg = 'Expected Dims: ' + repr(expected_dims)
            msg += 'Got Dims: ' + repr(arr_dims)
            raise Exception(msg)
        num_bits += len(nparr) * expected_dims * bit_per_item
        num_elemt += len(nparr) * expected_dims
        num_items += len(nparr)
    return num_bits,  num_items, num_elemt


def make_call_graph(func, *args, **kwargs):
    """ profile with pycallgraph

    Example:
        pycallgraph graphviz -- ./mypythonscript.py

    References:
        http://pycallgraph.slowchop.com/en/master/
    """
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    with PyCallGraph(output=GraphvizOutput):
        func(*args, **kwargs)


#def runprofile(cmd, globals_=globals(), locals_=locals()):
#    """ DEPRICATE. Tries to run a function and profile it """
#    # Meliae # from meliae import loader # om = loader.load('filename.json') # s = om.summarize();
#    #http://www.huyng.com/posts/python-performance-analysis/
#    #Once youve gotten your code setup with the <AT>profile decorator, use
#    kernprof.py to run your script.
#    #kernprof.py -l -v fib.py
#    import cProfile
#    print('[util] Profiling Command: ' + cmd)
#    cProfOut_fpath = 'OpenGLContext.profile'
#    cProfile.runctx( cmd, globals_, locals_, filename=cProfOut_fpath)
#    # RUN SNAKE
#    print('[util] Profiled Output: ' + cProfOut_fpath)
#    if sys.platform == 'win32':
#        rsr_fpath = 'C:/Python27/Scripts/runsnake.exe'
#    else:
#        rsr_fpath = 'runsnake'
#    view_cmd = rsr_fpath + ' "' + cProfOut_fpath + '"'
#    os.system(view_cmd)
#    return True


def _memory_profile(with_gc=False):
    """
    Helper for memory debugging. Mostly just a namespace where I experiment with
    guppy and heapy.

    References:
        http://stackoverflow.com/questions/2629680/deciding-between-subprocess-multiprocessing-and-thread-in-python

    Reset Numpy Memory::
        %reset out
        %reset array
    """
    import utool as ut
    if with_gc:
        garbage_collect()
    import guppy
    hp = guppy.hpy()
    print('[hpy] Waiting for heap output...')
    heap_output = hp.heap()
    print(heap_output)
    print('[hpy] total heap size: ' + ut.byte_str2(heap_output.size))
    ut.util_resources.memstats()
    # Graphical Browser
    #hp.pb()


def make_object_graph(obj, fpath='sample_graph.png'):
    """ memoryprofile with objgraph

    Examples:
        #import objgraph
        #objgraph.show_most_common_types()
        #objgraph.show_growth()
        #memtrack.report()
        #memtrack.report()
        #objgraph.show_growth()
        #import gc
        #gc.collect()
        #memtrack.report()
        #y = 0
        #objgraph.show_growth()
        #memtrack.report()
        #utool.embed()

    References:
        http://mg.pov.lt/objgraph/
    """
    import objgraph
    objgraph.show_most_common_types()
    #print(objgraph.by_type('ndarray'))
    #objgraph.find_backref_chain(
    #     random.choice(objgraph.by_type('ndarray')),
    #     objgraph.is_proper_module)
    objgraph.show_refs([obj], filename='ref_graph.png')
    objgraph.show_backrefs([obj], filename='backref_graph.png')


def disable_garbage_collection():
    gc.disable()


def enable_garbage_collection():
    gc.enable()


def garbage_collect():
    gc.collect()


def get_object_nbytes(obj, fallback_type=None, follow_pointers=False, exclude_modules=True, listoverhead=False):
    """
    CommandLine:
        python -m utool.util_dev --test-get_object_nbytes
        python -m utool.util_dev --test-get_object_nbytes:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> import numpy as np
        >>> import utool as ut
        >>> obj = [np.empty(1, dtype=np.uint8) for _ in range(8)]
        >>> nBytes = ut.get_object_nbytes(obj)
        >>> result = ('nBytes = %s' % (nBytes,))
        >>> print(result)
        nBytes = 8

    Example:
        >>> # xdoctest: +SKIP
        >>> from utool.util_dev import *  # NOQA
        >>> import ibeis
        >>> import utool as ut
        >>> species = ibeis.const.TEST_SPECIES.ZEB_PLAIN
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> qaids = ibs.get_valid_aids(species=species)
        >>> daids = ibs.get_valid_aids(species=species)
        >>> qreq_ = ibs.new_query_request(qaids, daids, verbose=True)
        >>> nBytes = ut.get_object_nbytes(qreq_)
        >>> result = (ut.byte_str2(nBytes))
        >>> print('result = %r' % (result,))

    Ignore:
        import sys
        sizedict = {key: sys.getsizeof(key()) for key in [dict, list, set, tuple, int, float]}
        ut.print_dict(sizedict)
        sizedict = {
            <type 'tuple'>: 56,
            <type 'set'>: 232,
            <type 'list'>: 72,
            <type 'float'>: 24,
            <type 'int'>: 24,
            <type 'dict'>: 280,
        }
    """
    import utool as ut
    import types
    seen = set([])
    def _object_nbytes(obj):
        object_id = id(obj)
        if object_id in seen:
            return 0
        if (obj is None or isinstance(obj, (int, bool, float))):
            return sys.getsizeof(obj)
        elif isinstance(obj, six.string_types):
            return sys.getsizeof(obj)
        seen.add(object_id)

        if listoverhead:
            totalsize = sys.getsizeof(obj)
        else:
            totalsize = 0
        try:
            if isinstance(obj, (ut.DynStruct, ut.Pref)):
                # dont deal with dynstruct shenanigans
                return
            elif exclude_modules and isinstance(obj, types.ModuleType):
                return 0
            elif isinstance(obj, np.ndarray):
                if not obj.flags['OWNDATA']:
                    # somebody else owns the data, returned size may be smaller than sobj.nbytes
                    # because sys.getsizeof will return the container size.
                    # if owndata is true sys.getsizeof returns the actual size
                    if follow_pointers:
                        totalsize += obj.nbytes
                    pass
                # TODO: check if a view or is memmapped
                # Does sys.getsizeof do the right thing ?
                totalsize = obj.nbytes
            elif (isinstance(obj, (tuple, list, set, frozenset))):
                for item in obj:
                    totalsize += _object_nbytes(item)
            elif isinstance(obj, dict):
                try:
                    for key, val in six.iteritems(obj):
                        totalsize += _object_nbytes(key)
                        totalsize += _object_nbytes(val)
                except RuntimeError as dictex:
                    ut.printex(dictex, 'RuntimeError in parsing dict nbytes',
                               keys=['key', (type, 'obj')], iswarning=True)
                    raise
            elif isinstance(obj, object) and hasattr(obj, '__dict__'):
                if hasattr(obj, 'used_memory') and not isinstance(obj, type):
                    # hack for flann objects
                    totalsize += obj.used_memory()
                totalsize += _object_nbytes(obj.__dict__)
                return totalsize
            elif isinstance(obj, type):
                # use zero for class definitions
                return 0
            elif isinstance(obj, np.int32):
                return obj.nbytes
            else:
                print('Unknown type %r for parsing size' % (type(obj),))
                return 0
        #except TypeError as ex:
        except Exception as ex:
            ut.printex(ex, 'may be an error in _object_nbytes',
                       keys=[(type, 'obj')], iswarning=True, tb=True)
            pass
            #import utool as ut
            #print('obj = %r' % (obj,))
            #ut.printex(ex)
            #ut.embed()
            #raise RuntimeError(str(ex))  # from ex
        return totalsize
    return _object_nbytes(obj)


def print_object_size_tree(obj, lbl='obj', maxdepth=None):
    """ Needs work """
    from utool import util_str
    from utool import util_type

    byte_str2 = util_str.byte_str2

    def _get_object_size_tree(obj, depth=0, lbl='obj', seen=None):
        indent = ' ' * (depth * 4)
        if maxdepth is not None and depth >= maxdepth:
            size_list = [get_object_nbytes(obj)]
            print(indent + str(size_list[0]))
            return size_list
        if (obj is None or isinstance(obj, (int, bool, float))):
            return [sys.getsizeof(obj)]
        elif isinstance(obj, six.string_types):
            return [sys.getsizeof(obj)]
        object_id = id(obj)
        if object_id in seen:
            print(indent + '%s ' % ('(seen) ' + lbl,))
            return []
        seen.add(object_id)
        size_list = [(lbl, sys.getsizeof(obj))]
        if isinstance(obj, np.ndarray):
            size_list.append(obj.nbytes)
            print('%s%s = %s ' % (indent, '(ndarray) %s' % (lbl,), byte_str2(obj.nbytes)))
        elif (isinstance(obj, (tuple, list, set, frozenset))):
            typestr = util_type.type_str(type(obj))
            print('%s(%s) %s = %s ' % (indent, typestr, lbl, byte_str2(sys.getsizeof(obj))))
            for item in obj:
                size_list += _get_object_size_tree(item, depth + 1, 'item', seen)
        elif isinstance(obj, dict):
            print('%s(dict) %s = %s ' % (indent, lbl, byte_str2(sys.getsizeof(obj))))
            try:
                for key, val in six.iteritems(obj):
                    size_list += _get_object_size_tree(key, depth + 1, key, seen)
                    size_list += _get_object_size_tree(val, depth + 1, key, seen)
            except RuntimeError as dictex:
                ut.printex(dictex, 'RuntimeError in parsing dict nbytes',
                           keys=['key', (type, 'obj')], iswarning=True)
                raise
        elif isinstance(obj, object) and hasattr(obj, '__dict__'):
            if hasattr(obj, 'used_memory'):
                size_ = obj.used_memory()
                print('(%sflann?) %s = %s ' % (indent, lbl, byte_str2(size_)))
                size_list += [size_]
            else:
                print('%s(object) %s = %s ' % (indent, lbl, byte_str2(sys.getsizeof(obj))))
            size_list += _get_object_size_tree(obj.__dict__,
                                               depth=depth + 1,
                                               lbl='__dict__', seen=seen)
        return size_list
    seen = set([])
    _get_object_size_tree(obj, depth=0, lbl=lbl, seen=seen)
    del seen


def get_object_size_str(obj, lbl='', unit=None):
    """

    CommandLine:
        python -m utool.util_dev --test-get_object_size_str --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> import numpy as np
        >>> import utool as ut
        >>> obj = [np.empty((512), dtype=np.uint8) for _ in range(10)]
        >>> nBytes = ut.get_object_size_str(obj)
        >>> result = ('result = %s' % (ut.repr2(nBytes),))
        >>> print(result)
        result = '5.00 KB'
    """

    from utool import util_str
    nBytes = get_object_nbytes(obj)
    if len(lbl) > 0 and not lbl.endswith(' '):
        lbl_ = lbl + ' '
    else:
        lbl_ = lbl
    if unit is None:
        sizestr = lbl_ + util_str.byte_str2(nBytes)
    else:
        sizestr = lbl_ + util_str.byte_str(nBytes, unit)
    return sizestr


def print_object_size(obj, lbl=''):
    print(get_object_size_str(obj, lbl=lbl))


def get_object_base():
    from .DynamicStruct import DynStruct
    from .util_classes import AutoReloader
    if '--min-base' in sys.argv:
        return object
    elif '--noreload-base' not in sys.argv:
        return AutoReloader
    elif '--dyn-base' in sys.argv:
        return DynStruct


def get_cython_exe():
    from utool import util_cplat
    if util_cplat.WIN32:
        cython_exe = r'C:\Python27\Scripts\cython.exe'
        if not exists(cython_exe):
            cython_exe = 'cython.py'
    else:
        cython_exe = 'cython'
    return cython_exe


def compile_cython(fpath, clean=True):
    r""" Compiles a cython pyx into a shared library

    This seems broken
    compiles pyx -> pyd/dylib/so

    Examples:
        REAL SETUP.PY OUTPUT
        cythoning vtool/linalg_cython.pyx to vtool\linalg_cython.c
        C:\MinGW\bin\gcc.exe -mdll -O -Wall ^
        -IC:\Python27\Lib\site-packages\numpy\core\include ^
        -IC:\Python27\include -IC:\Python27\PC ^
        -c vtool\linalg_cython.c ^
        -o build\temp.win32-2.7\Release\vtool\linalg_cython.o

        writing build\temp.win32-2.7\Release\vtool\linalg_cython.def

        C:\MinGW\bin\gcc.exe -shared \
        -s \
        build\temp.win32-2.7\Release\vtool\linalg_cython.o \
        build\temp.win32-2.7\Release\vtool\linalg_cython.def \
        -LC:\Python27\libs \
        -LC:\Python27\PCbuild \
        -lpython27 \
        -lmsvcr90 \
        -o build\lib.win32-2.7\vtool\linalg_cython.pyd

    """
    from utool import util_cplat
    from utool import util_path
    import utool

    # Get autogenerated filenames
    fpath = util_path.truepath(fpath)
    dpath_, fname_ = split(fpath)
    dpath = relpath(dpath_, os.getcwd())
    fname, ext = splitext(fname_)
    # Prefer pyx over py
    if exists(fname + '.pyx'):
        fpath = fname + '.pyx'
    fname_c  = join(dpath, fname + '.c')
    fname_lib = join(dpath, fname + util_cplat.get_pylib_ext())

    print('[utool.compile_cython] fpath=%r' % (fpath,))
    print(' --- PRECHECKS --- ')
    if clean:
        utool.delete(fname_c)
        utool.delete(fname_lib)

    utool.checkpath(fpath, verbose=True, n=4)
    utool.checkpath(fname_c, verbose=True, info=False, n=4)
    utool.checkpath(fname_lib, verbose=True, info=False, n=4)

    # Cython build arguments
    cython_exe = get_cython_exe()
    if util_cplat.WIN32:
        os.environ['LDFLAGS'] = '-march=i486'
        os.environ['CFLAGS'] = '-march=i486'
        cc_exe = r'C:\MinGW\bin\gcc.exe'
        pyinclude_list = [
            r'C:\Python27\Lib\site-packages\numpy\core\include',
            r'C:\Python27\include',
            r'C:\Python27\PC',
            np.get_include()]
        pylib_list     = [
            r'C:\Python27\libs',
            r'C:\Python27\PCbuild'
            #r'C:\Python27\DLLS',
        ]
        plat_gcc_flags = ' '.join([
            '-mdll',
            '-O',
            '-DNPY_NO_DEPRECATED_API',
            #'-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
            '-Wall',
            '-Wno-unknown-pragmas',
            '-Wno-format',
            '-Wno-unused-function',
            '-m32',
            '-shared',
            #'-fPIC',
            #'-fwrapv',
        ])
        #plat_gcc_flags = ' '.join([
        #    '-shared',
        #    '-m32',
        #    '-mdll',
        #    '-march=i486',
        #    '-O',
        #])
        plat_link_flags = '-lpython27 -lmsvcr90'

    #C:\MinGW\bin\gcc.exe -shared -s
    #build\temp.win32-2.7\Release\vtool\linalg_cython.o
    #build\temp.win32-2.7\Release\vtool\linalg_cython.def -LC:\Python27\libs
    #-LC:\Python27\PCbuild -lpython27 -lmsvcr90 -o
    #build\lib.win32-2.7\linalg_cython.pyd

    else:
        cc_exe = 'gcc'
        cython_exe = 'cython'
        pyinclude_list = [r'/usr/include/python2.7', np.get_include()]
        pylib_list     = []
        plat_gcc_flags = ' '.join([
            '-shared',
            '-pthread',
            '-fPIC',
            '-fwrapv',
            '-O2',
            '-Wall',
            '-fno-strict-aliasing',
        ])
    #C:\MinGW\bin\gcc.exe -mdll -O -Wall
    #-IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include
    #-IC:\Python27\PC -c vtool\linalg_cython.c -o
    #build\temp.win32-2.7\Release\vtool\linalg_cyth

    pyinclude = '' if len(pyinclude_list) == 0 else '-I' + ' -I'.join(pyinclude_list)
    pylib     = '' if len(pylib_list)     == 0 else '-L' + ' -L'.join(pylib_list)
    gcc_flag_list = [
        plat_gcc_flags,
        pyinclude,
        pylib,
        plat_link_flags,
    ]
    gcc_flags = ' '.join(filter(lambda x: len(x) > 0, gcc_flag_list))
    gcc_build_cmd = cc_exe + ' ' + gcc_flags + ' -o ' + fname_lib + ' -c ' + fname_c

    cython_build_cmd = cython_exe + ' ' + fpath

    # HACK
    print('\n --- CYTHON_COMMANDS ---')
    print(utool.pack_into(cython_build_cmd, textwidth=80, newline_prefix='  '))
    print('')
    print(utool.pack_into(gcc_build_cmd, textwidth=80, newline_prefix='  '))
    print(gcc_build_cmd)
    print('\n --- COMMAND_EXECUTION ---')

    def verbose_cmd(cmd):
        print('\n<CMD>')
        print(cmd)
        ret = os.system(cmd)
        print('> ret = %r' % ret)
        print('</CMD>\n')
        #print('-------------------')
        return ret

    ret = verbose_cmd(cython_build_cmd)
    assert utool.checkpath(fname_c, verbose=True, n=2), 'failed cython build'
    ret = verbose_cmd(gcc_build_cmd)
    assert utool.checkpath(fname_lib, verbose=True, n=2), 'failed gcc cython build'
    #try:
    #    #lib_dpath, lib_fname = split(fname_lib)
    #    #cwd = os.getcwd()
    #    #os.chdir(lib_dpath)
    #    ##exec('import ' + splitext(lib_fname)[0])
    #    #os.chdir(cwd)
    #    pass
    #except Exception:
    #    pass
    #    raise

    #out, err, ret = util_cplat.shell(cython_exe + ' ' + fpath)
    #out, err, ret = util_cplat.shell((cython_exe, fpath))
    #if ret == 0:
    #    out, err, ret = util_cplat.shell(cc_exe + ' ' + gcc_flags + ' -o ' +
    #    fname_so + ' ' + fname_c)
    return ret


def find_exe(name, path_hints=[], required=True):
    from utool import util_cplat
    if util_cplat.WIN32 and not name.endswith('.exe'):
        name += '.exe'

    for path in path_hints:
        exe_fpath = join(path, name)
        if exists(exe_fpath):
            return exe_fpath

    if required:
        raise AssertionError('cannot find ' + name)


def _on_ctrl_c(signal, frame):
    print('Caught ctrl+c')
    sys.exit(0)


def init_catch_ctrl_c():
    import signal
    signal.signal(signal.SIGINT, _on_ctrl_c)


def reset_catch_ctrl_c():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # reset ctrl+c behavior


USER_MODE      =  util_arg.get_argflag(('--user-mode', '--no-developer',
                                        '--nodev', '--nodeveloper'))
DEVELOPER_MODE =  util_arg.get_argflag(('--dev-mode', '--developer-mode'))
#USER_MODE = not DEVELOPER_MODE


def is_developer(mycomputers=None):
    import utool
    if USER_MODE:
        return False
    if DEVELOPER_MODE:
        return True
    if mycomputers is None:
        mycomputers = ['hyrule', 'ooo', 'bakerstreet']
    compname_lower = utool.get_computer_name().lower()
    return compname_lower in mycomputers


def iup():
    """ shortcut when pt is not imported """
    try:
        import plottool_ibeis as pt
    except ImportError:
        import plottool as pt
    pt.iup()


def make_at_least_n_items_valid(flag_list, n):
    """
    tries to make at least min(len(flag_list, n) items True in flag_list

    Args:
        flag_list (list): list of booleans
        n (int): number of items to ensure are True

    CommandLine:
        python -m utool.util_dev --test-make_at_least_n_items_valid

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> # build test data
        >>> flag_list = [False, True, False, False, False, False, False, True]
        >>> n = 5
        >>> # execute function
        >>> flag_list = make_at_least_n_items_valid(flag_list, n)
        >>> # verify results
        >>> result = str(flag_list)
        >>> print(result)
        [ True  True  True  True False False False  True]
    """
    flag_list = np.array(flag_list)
    num_valid = flag_list.sum()
    # Find how many places we need to make true
    num_extra = min(len(flag_list) - num_valid, n - num_valid)
    # make_at_least_n_items_valid
    # Add in some extra daids to show if there are not enough
    for index in range(len(flag_list)):
        if num_extra <= 0:
            break
        if not flag_list[index]:
            flag_list[index] = True
            num_extra -= 1
    return flag_list


# TODO: Generalize these functions
def inverable_unique_two_lists(item1_list, item2_list):
    """
    item1_list = aid1_list
    item2_list = aid2_list
    """
    import utool as ut
    unique_list1, inverse1 = np.unique(item1_list, return_inverse=True)
    unique_list2, inverse2 = np.unique(item2_list, return_inverse=True)

    flat_stacked, cumsum = ut.invertible_flatten2((unique_list1, unique_list2))
    flat_unique, inverse3 = np.unique(flat_stacked, return_inverse=True)
    reconstruct_tup = (inverse3, cumsum, inverse2, inverse1)
    return flat_unique, reconstruct_tup


def uninvert_unique_two_lists(flat_list, reconstruct_tup):
    """
    flat_list = thumb_list
    """
    import utool as ut
    (inverse3, cumsum, inverse2, inverse1) = reconstruct_tup
    flat_stacked_ = ut.take(flat_list, inverse3)
    unique_list1_, unique_list2_ = ut.unflatten2(flat_stacked_, cumsum)
    res_list1_ = ut.take(unique_list1_, inverse1)
    res_list2_ = ut.take(unique_list2_, inverse2)
    return res_list1_, res_list2_


def inverable_group_multi_list(item_lists):
    """
    aid_list1 = np.array([1, 1, 2, 2, 3, 3])
    aid2_list = np.array([4, 2, 1, 9, 8, 7])
    item_lists = (np.array(aid1_list), np.array(aid2_list))
    """
    #unique_list1, inverse1 = np.unique(item1_list, return_index=True, return_inverse=True)
    try:
        import vtool_ibeis as vt
    except ImportError:
        import vtool as vt
    import utool as ut
    # Find uniques and groups in each individual list
    unique_lists = []
    groupx_lists = []
    for item_list in item_lists:
        unique_items, groupxs = vt.group_indices(item_list)
        unique_lists.append(unique_items)
        groupx_lists.append(groupxs)
    # Merge all indexes into a signle long list
    groups_stacked = ut.flatten(groupx_lists)
    flat_stacked, cumsum = ut.invertible_flatten2(unique_lists)
    # Find uniques in those lists
    flat_unique, stack_groups = vt.group_indices(np.array(flat_stacked))
    # Get a list of corresonding group indicies from each input list
    flat_groupx_multilist = [ut.take(groups_stacked, groupx) for groupx in stack_groups]
    # flat_unique corresponds with the aids (hence chips) the flag_groupxs
    # multilist is a list where each item is a tuple who's nth item indexes
    # into the nth input list. Ie (1, 0) is a list of indexes into the 1st chip
    # the 0th keypoint list
    return flat_unique, flat_groupx_multilist


def autopep8_diff(fpath):
    r"""
    Args:
        fpath (str):  file path string

    CommandLine:
        python -m utool.util_dev --test-autopep8_diff --fpath ingest_data.py

    Example:
        >>> # xdoctest: +SKIP
        >>> from utool.util_dev import *  # NOQA
        >>> fpath = ut.get_argval('--fpath', type_=str, default='ingest_data.py')
        >>> result = autopep8_diff(fpath)
        >>> print(result)
    """
    import utool as ut
    args = ('autopep8', fpath, '--diff')
    res = ut.cmd(args, verbose=False)
    out, err, ret = res
    ut.print_difftext(out)


def get_partial_func_name(func, precision=3):
    float_fmtstr = '%.' + str(precision) + 'f'
    def repr_(v):
        if isinstance(v, float):
            return float_fmtstr % (v,)
        return repr(v)
    name_part = func.func.func_name
    args_part = ','.join(map(repr_, func.args))
    kwargs_part = '' if func.keywords is None else (
        ','.join(['%s=%s' % (key, repr_(val)) for key, val in func.keywords.items()])
    )
    name = name_part + '(' + args_part + kwargs_part + ')'
    return name


def dev_ipython_copypaster(func):
    import utool as ut
    code_text = get_dev_paste_code(func)
    ut.copy_text_to_clipboard(code_text)


def get_dev_paste_code(func):
    import utool as ut
    example_texts = ut.get_doctest_examples(func)
    example_text = example_texts[0][0]
    assert isinstance(example_text, str), ut.repr4(example_text)
    assert isinstance(example_text, str), ut.repr4(example_text)
    source_text = ut.get_func_sourcecode(func)
    get_dev_code = '\n'.join((example_text, source_text))
    return get_dev_code


def search_utool(pat):
    import utool as ut
    return search_module(ut, pat)


def search_module(mod, pat, ignore_case=True, recursive=False, _seen=None):
    r"""
    Searches module functions, classes, and constants for members matching a
    pattern.

    Args:
        mod (module): live python module
        pat (str): regular expression

    Returns:
        list: found_list

    CommandLine:
        python -m utool.util_dev --exec-search_module --mod=utool --pat=module
        python -m utool.util_dev --exec-search_module --mod=opengm --pat=cut
        python -m utool.util_dev --exec-search_module --mod=opengm --pat=multi
        python -m utool.util_dev --exec-search_module --mod=utool --pat=Levenshtein

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> import utool as ut
        >>> recursive = True
        >>> ignore_case = True
        >>> modname = ut.get_argval('--mod', type_=str, default='utool')
        >>> pat = ut.get_argval('--pat', type_=str, default='search')
        >>> mod = ut.import_modname(modname)
        >>> print('pat = %r' % (pat,))
        >>> print('mod = %r' % (mod,))
        >>> found_list = search_module(mod, pat, recursive=recursive)
        >>> result = ('found_list = %s' % (ut.repr2(found_list),))
        >>> print(result)

    Ignore:
        mod = cv2
        pat = 'freak'
    """
    if _seen is not None and mod in _seen:
        return []
    import utool as ut
    reflags = re.IGNORECASE * ignore_case
    found_list = [name for name in dir(mod) if re.search(pat, name, flags=reflags)]
    if recursive:
        if _seen is None:
            _seen = set()
        _seen.add(mod)
        module_attrs = [getattr(mod, name) for name in dir(mod)]
        submodules = [
            submod for submod in module_attrs
            if isinstance(submod, types.ModuleType) and submod not in _seen and
            ut.is_defined_by_module(submod, mod)
        ]
        for submod in submodules:
            found_list += search_module(submod, pat, ignore_case=ignore_case, recursive=recursive, _seen=_seen)
    # found_list = [name for name in dir(mod) if name.find(pat) >= 0]
    found_list = ut.unique_ordered(found_list)
    return found_list


def get_submodules_from_dpath(dpath, only_packages=False, recursive=True):
    r"""
    Args:
        dpath (str): directory path
        only_packages (bool): if True returns only package directories,
            otherwise returns module files. (default = False)

    Returns:
        list: submod_fpaths

    CommandLine:
        python -m utool.util_dev --exec-get_submodules_from_dpath --only_packages

    Example:
        >>> # xdoctest: +SKIP
        >>> # SCRIPT
        >>> from utool.util_dev import *  # NOQA
        >>> import utool as ut
        >>> dpath = ut.truepath_relative(ut.get_argval('--dpath', default='.'))
        >>> print(dpath)
        >>> only_packages = ut.get_argflag('--only_packages')
        >>> submod_fpaths = get_submodules_from_dpath(dpath, only_packages)
        >>> submod_fpaths = ut.lmap(ut.truepath_relative, submod_fpaths)
        >>> result = ('submod_fpaths = %s' % (ut.repr3(submod_fpaths),))
        >>> print(result)
    """
    import utool as ut
    submod_dpaths = [d for d in ut.ls_dirs(dpath) if ut.is_module_dir(d) ]
    if only_packages:
        submod_fpaths = submod_dpaths
    else:
        submod_fpaths = ut.ls_modulefiles(dpath)
    if recursive and len(submod_dpaths) > 0:
        recusive_results = [get_submodules_from_dpath(d, only_packages)
                            for d in submod_dpaths]
        submod_fpaths.extend(ut.flatten(recusive_results))
    return submod_fpaths


def pylab_qt4():
    try:
        import plottool_ibeis as pt
    except ImportError:
        import plottool as pt
    pt.ensureqt()


def ensureqt():
    try:
        import plottool_ibeis as pt
    except ImportError:
        import plottool as pt
    pt.ensureqt()


class DictLike_old(object):
    """
    DEPRICATE
    TODO: use util_dict.DictLike instead
    """
    def __repr__(self):
        return repr(dict(self.items()))

    def __str__(self):
        return str(dict(self.items()))

    def __len__(self):
        return len(self.keys())

    def __contains__(self, key):
        return key in self.keys()

    def copy(self):
        return dict(self.items())

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def iteritems(self):
        for key, val in zip(self.iterkeys(), self.itervalues()):
            yield key, val

    def itervalues(self):
        return (self[key] for key in self.keys())

    def values(self):
        return list(self.itervalues())

    def keys(self):
        return list(self.iterkeys())

    def items(self):
        return list(self.iteritems())


#class ClassAttrDictProxy(DictLike_old):
class ClassAttrDictProxy(util_dict.DictLike):
    """
    Allows class attributes to be specified using dictionary syntax

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> import utool as ut
        >>> class Foo(object):
        >>>     def __init__(self):
        >>>         self.attr1 = 'foo'
        >>>         self.attr2 = 'bar'
        >>>         self.proxy = ut.ClassAttrDictProxy(self, ['attr1', 'attr2'])
        >>> self = Foo()
        >>> self.proxy['attr2'] = 'baz'
        >>> assert self.attr2 == 'baz', 'should have changed in object'
    """
    def __init__(self, obj, keys, attrs=None):
        if attrs is None:
            attrs = keys
        self.obj = obj
        self._keys = keys
        self._attrs = attrs
        self.key2_attrs = dict(zip(keys, attrs))

    def keys(self):
        return self._keys

    def getitem(self, key):
        attrname = self.key2_attrs[key]
        return getattr(self.obj, attrname)

    def setitem(self, key, val):
        attrname = self.key2_attrs[key]
        setattr(self.obj, attrname, val)

    #def iterkeys(self):
    #    return iter(self._keys)

    #def __getitem__(self, key):
    #    return getattr(self.obj, self.key2_attrs[key])

    #def __setitem__(self, key, val):
    #    setattr(self.obj, self.key2_attrs[key], val)


class NiceRepr(object):
    """
    base class that defines a nice representation and string func
    for a class given that the user implements __nice__

    Rename to NiceObject?

    """
    def __repr__(self):
        try:
            classname = self.__class__.__name__
            devnice = self.__nice__()
            return '<%s(%s) at %s>' % (classname, devnice, hex(id(self)))
        except AttributeError:
            return object.__repr__(self)
            #return super(NiceRepr, self).__repr__()

    def __str__(self):
        try:
            classname = self.__class__.__name__
            devnice = self.__nice__()
            return '<%s(%s)>' % (classname, devnice)
        except AttributeError:
            warnings.warn('Error in __nice__ for %r' % (self.__class__,),
                          category=RuntimeWarning)
            if util_arg.SUPER_STRICT:
                raise
            else:
                return object.__str__(self)
            #return super(NiceRepr, self).__str__()


def execstr_funckw(func):
    """
    for doctests kwargs

    SeeAlso:
        ut.exec_func_src
        ut.argparse_funckw
    """
    import utool as ut
    funckw = ut.get_func_kwargs(func)
    return ut.execstr_dict(funckw, explicit=True)


def exec_argparse_funckw(func, globals_=None, defaults={}, **kwargs):
    """
    for doctests kwargs

    SeeAlso:
        ut.exec_func_src
        ut.argparse_funckw
    """
    import utool as ut
    funckw = ut.get_func_kwargs(func)
    funckw.update(defaults)
    parsekw = ut.argparse_dict(funckw, **kwargs)
    if globals_:
        globals_.update(parsekw)
    return parsekw
    # execstr = ut.execstr_dict(parsekw, explicit=True)
    # exec(execstr, globals_)


def exec_funckw(func, globals_):
    """
    SeeAlso:
        ut.argparse_funckw
    """
    exec(execstr_funckw(func), globals_)


def focusvim():
    import utool.util_ubuntu
    utool.util_ubuntu.xctrl.do(('focus', 'GVIM'),)


def ipcopydev():
    import IPython
    # ipy.history_manager.hist_file
    ipy = IPython.get_ipython()
    # lasttup_ = six.next(ipy.history_manager.get_tail())
    # lastline = lasttup_[2]
    lastline = ipy.history_manager.input_hist_parsed[-2]
    import utool as ut

    ut.copy_text_to_clipboard(lastline)

    # import utool as ut
    import utool.util_ubuntu
    utool.util_ubuntu.rrr(0)
    # utool.util_ubuntu.focus_window('gnome-terminal.Gnome-terminal')
    utool.util_ubuntu.xctrl.do(
        ('focus', 'GVIM'),
        ('key', 'ctrl+v'),
        ('focus', 'x-terminal-emulator.X-terminal-emulator')
    )
    # return lastline
    # .magic('history')
    # histfile = os.path.join(os.environ['HOME'], '.pythonhistory')
    pass


def instancelist(obj_list, check=False, shared_attrs=None):
    """
    Executes methods and attribute calls on a list of objects of the same type

    Bundles a list of object of the same type into a single object.
    The new object contains the same functions as each original object
    but applies them to each element of the list independantly when called.

    CommandLine:
        python -m utool.util_dev instancelist

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> import utool as ut
        >>> obj_list = ['hi', 'bye', 'foo']
        >>> self = ut.instancelist(obj_list, check=False)
        >>> print(self)
        >>> print(self.upper())
        >>> print(self.isalpha())
    """
    class InstanceList_(object):
        def __init__(self, obj_list, shared_attrs=None):
            self._obj_list = []
            self._shared_public_attrs = []
            self._example_type = None

            if len(obj_list) > 0:
                import utool as ut
                self._obj_list = obj_list

                example_obj = obj_list[0]
                example_type = type(example_obj)
                self._example_type = example_type

                if shared_attrs is None:
                    if check:
                        attrsgen = [set(dir(obj)) for obj in obj_list]
                        shared_attrs = list(reduce(set.intersection, attrsgen))
                    else:
                        shared_attrs = dir(example_obj)

                #allowed = ['__getitem__']  # TODO, put in metaclass
                allowed = []
                self._shared_public_attrs = [
                    a for a in shared_attrs
                    if a in allowed or not a.startswith('_')
                ]

                for attrname in self._shared_public_attrs:
                    attrtype = getattr(example_type, attrname, None)
                    if attrtype is not None and isinstance(attrtype, property):
                        # need to do this as metaclass
                        setattr(InstanceList_, attrname,
                                property(self._define_prop(attrname)))
                    else:
                        func = self._define_func(attrname)
                        ut.inject_func_as_method(self, func, attrname)

        def __nice__(self):
            if self._example_type is None:
                typename = 'object'
            else:
                typename = self._example_type.__name__
            return 'of %d %s(s)' % (len(self._obj_list), typename)

        def __repr__(self):
            classname = self.__class__.__name__
            devnice = self.__nice__()
            return '<%s(%s) at %s>' % (classname, devnice, hex(id(self)))

        def __str__(self):
            classname = self.__class__.__name__
            devnice = self.__nice__()
            return '<%s(%s)>' % (classname, devnice)

        def __getitem__(self, key):
            # TODO, put in metaclass
            return self._map_method('__getitem__', key)

        def _define_func(self, attrname):
            import utool as ut
            def _wrapper(self, *args, **kwargs):
                return self._map_method(attrname, *args, **kwargs)
            ut.set_funcname(_wrapper, attrname)
            return _wrapper

        def _map_method(self, attrname, *args, **kwargs):
            mapped_vals = [getattr(obj, attrname)(*args, **kwargs)
                           for obj in self._obj_list]
            return mapped_vals

        def _define_prop(self, attrname):
            import utool as ut
            def _getter(self):
                return self._map_property(attrname)
            ut.set_funcname(_getter, 'get_' + attrname)
            return _getter

        def _map_property(self, attrname):
            mapped_vals = [getattr(obj, attrname) for obj in self._obj_list]
            return mapped_vals
    return InstanceList_(obj_list, shared_attrs)


#from utool import util_class  # NOQA


#@util_class.reloadable_class
class ColumnLists(NiceRepr):
    r"""
    Way to work with column data

    Args:
        key_to_list (dict): (default = {})

    CommandLine:
        python -m utool.util_dev ColumnLists --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> import utool as ut
        >>> key_to_list = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> self = ColumnLists(key_to_list)
        >>> newself = self.take([1, 2])
        >>> print(self)
        >>> print(newself)
    """
    def __init__(self, key_to_list={}, _meta=None):
        import utool as ut
        self._key_to_list = key_to_list
        len_list = [len(vals) for vals in self._key_to_list.values()]
        if len(key_to_list) == 0:
            self._idxs = []
        else:
            self._idxs = list(range(len_list[0]))
        self._meta = {} if _meta is None else _meta
        assert ut.allsame(len_list)

    @classmethod
    def flatten(cls, list_):
        import utool as ut
        dict_list = [c._key_to_list for c in list_]
        stacked_dict = ut.dict_stack2(dict_list)
        key_to_list = ut.map_dict_vals(ut.flatten, stacked_dict)
        if len(list_) > 0:
            meta = list_[0]._meta.copy()
        else:
            meta = None
        self = cls(key_to_list, meta)
        return self

    def __add__(self, other):
        import utool as ut
        if len(other) == 0:
            return self.copy()
        elif len(self) == 0:
            return other.copy()
        key_to_list = ut.dict_union_combine(self._key_to_list,
                                            other._key_to_list)
        new = self.__class__(key_to_list, self._meta.copy())
        return new

    def copy(self):
        return self.__class__(self._key_to_list.copy(), self._meta.copy())

    @property
    def shape(self):
        shape = (len(self), len(self._key_to_list))
        return shape

    def __nice__(self):
        return 'rows=%r, cols=%r' % self.shape

    def __iter__(self):
        return iter(self._idxs)

    def __len__(self):
        return len(self._idxs)

    def keys(self):
        return self._key_to_list.keys()

    def values(self):
        return self._key_to_list.values()

    def __delitem__(self, key):
        if isinstance(key, list):
            import utool as ut
            ut.delete_dict_keys(self._key_to_list, key)
        else:
            del self._key_to_list[key]

    def __getitem__(self, key):
        return self._key_to_list[key]

    def __setitem__(self, key, val):
        assert len(val) == len(self), (
            'len(val)=%r does not correspond with len(self)=%r' % (
                len(val), len(self)))
        self._key_to_list[key] = val

    def asdict(self):
        return self._key_to_list

    def asdataframe(self):
        #import pandas as pd
        #pd.set_option("display.width", 1000)
        import pandas as pd
        df = pd.DataFrame(self.asdict())
        return df

    def aspandas(self):
        return self.asdataframe()

    def to_csv(self, **kwargs):
        import utool as ut
        column_lbls = list(self.keys())
        column_list = ut.take(self._key_to_list, column_lbls)
        kwargs['standardize'] = kwargs.get('standardize', True)
        csv_text = ut.util_csv.make_csv_table(column_list, column_lbls, **kwargs)
        #csv_text = ut.make_standard_csv(column_list, column_lbls)
        return csv_text.strip('\n')

    def print(self, ignore=None, keys=None):
        self_ = self
        if ignore is None:
            ignore = self._meta.get('ignore', [])

        if keys:
            self_ = self_.take_column(keys)
        if ignore:
            self_ = self_.copy()
            del self_[ignore]

        csv_text = self_.to_csv()
        lines = csv_text.split('\n')
        max_lines_start = self._meta.get('max_lines_start', 10)
        max_lines_end = self._meta.get('max_lines_end', 10)
        max_lines = max_lines_start + max_lines_end + 3
        if len(csv_text) > 100000 or len(lines) > max_lines:
            reduced_text = '\n'.join(
                lines[:max_lines_start] + ['...'] +
                lines[-max_lines_end:]
            )
            print(self_)
            print(reduced_text)
        else:
            print(self_)
            print(csv_text)

    def rrr(self):
        import utool as ut
        ut.reload_class(self)

    def reorder_columns(self, keys):
        import utool as ut
        self._key_to_list = ut.order_dict_by(self._key_to_list, keys)

    def take_column(self, keys, *extra_keys):
        """ Takes a subset of columns """
        import utool as ut
        keys = ut.ensure_iterable(keys) + list(extra_keys)
        key_to_list = ut.dict_subset(self._key_to_list, keys)
        newself = self.__class__(key_to_list, self._meta.copy())
        return newself

    def take(self, idxs):
        """ Takes a subset of rows """
        import utool as ut
        if False:
            key_to_list = ut.odict([
                (key, ut.take(val, idxs))
                for key, val in six.iteritems(self._key_to_list)
            ])
        else:
            import numpy as np
            key_to_list = ut.odict([
                (key, ut.take(val, idxs))
                if not isinstance(val, np.ndarray)
                else val.take(idxs, axis=0)
                for key, val in six.iteritems(self._key_to_list)
            ])
        newself = self.__class__(key_to_list, self._meta.copy())
        return newself

    def remove(self, idxs):
        """ Returns a copy with idxs removed """
        import utool as ut
        keep_idxs = ut.index_complement(idxs, len(self))
        return self.take(keep_idxs)

    def compress(self,  flags):
        import utool as ut
        idxs = ut.where(flags)
        return self.take(idxs)

    def chunks(self,  chunksize):
        import utool as ut
        for idxs in ut.ichunks(self, range(len(self))):
            yield self.take(idxs)

    def group_indicies(self, labels):
        import utool as ut
        if isinstance(labels, six.string_types):
            labels = self[labels]
        unique_labels, groupxs = ut.group_indices(labels)
        return unique_labels, groupxs

    def group_items(self, labels):
        """ group as dict """
        import utool as ut
        unique_labels, groups = self.group(labels)
        label_to_group = ut.odict(zip(unique_labels, groups))
        return label_to_group

    def group(self, labels):
        """ group as list """
        unique_labels, groupxs = self.group_indicies(labels)
        groups = [self.take(idxs) for idxs in groupxs]
        return unique_labels, groups

    def loc_by_key(self, key, vals):
        import utool as ut
        val_to_idx = ut.make_index_lookup(self[key])
        idx_list = ut.take(val_to_idx, vals)
        return self.take(idx_list)

    def get_singles(self, key):
        import utool as ut
        unique_labels, groupxs = self.group_indicies(key)
        single_xs = [xs for xs in groupxs if len(xs) == 1]
        singles = self.take(ut.flatten(single_xs))
        #groups = self.group(self[key])[1]
        #single_groups = [g for g in groups if len(g) == 1]
        #singles = self.__class__.flatten(single_groups)
        return singles

    def get_multis(self, key):
        import utool as ut
        unique_labels, groupxs = self.group_indicies(key)
        multi_xs = [xs for xs in groupxs if len(xs) > 1]
        multis = self.take(ut.flatten(multi_xs))
        #groups = self.group(self[key])[1]
        #multi_groups  = [g for g in groups if len(g) > 1]
        #multis = self.__class__.flatten(multi_groups)
        return multis

    def cast_column(self, keys, func):
        """ like map column but applies values inplace """
        import utool as ut
        for key in ut.ensure_iterable(keys):
            self[key] = [func(v) for v in self[key]]

    @util_decor.accepts_scalar_input2()
    def map_column(self, keys, func):
        """
        Args:
            keys (list or str): the column name(s) to apply the `func` to
            func (callable): applied to each element in the specified columns
        """
        return [[func(v) for v in self[key]] for key in keys]

    def merge_rows(self, key, merge_scalars=True):
        """
        Uses key as a unique index an merges all duplicates rows.  Use
        cast_column to modify types of columns before merging to affect
        behavior of duplicate rectification.

        Args:
            key: row to merge on
            merge_scalars: if True, scalar values become lists


        Example:
            >>> # DISABLE_DOCTEST
            >>> from utool.util_dev import *  # NOQA
            >>> import utool as ut
            >>> key_to_list = {
            >>>     'uuid': [1, 1, 2, 3, 4, 2, 1],
            >>>     'a':    [1, 2, 3, 4, 5, 6, 7],
            >>>     'b':    [[1], [2], [3], [4], [5], [6], [7]],
            >>>     'c':    [[1], [1], [2], [3], [4], [2], [1]],
            >>> }
            >>> self = ColumnLists(key_to_list)
            >>> key = 'uuid'
            >>> newself = self.merge_rows('uuid')
            >>> print(newself.to_csv())
            #             a,            c,            b,         uuid
                          4,          [3],          [4],            3
                          5,          [4],          [5],            4
                "[1, 2, 7]",  "[1, 1, 1]",  "[1, 2, 7]",  "[1, 1, 1]"
                   "[3, 6]",     "[2, 2]",     "[3, 6]",     "[2, 2]"

        """
        import utool as ut
        unique_labels, groupxs = self.group_indicies(key)
        single_xs = [xs for xs in groupxs if len(xs) == 1]
        multi_xs = [xs for xs in groupxs if len(xs) > 1]
        singles = self.take(ut.flatten(single_xs))
        multis = [self.take(idxs) for idxs in multi_xs]

        merged_groups = []
        for group in multis:
            newgroup = {}
            for key_ in group.keys():
                val = group[key_]
                if key_ == key:
                    # key_ was garuenteed unique
                    val_ = val[0]
                elif hasattr(val[0].__class__, 'union'):
                    # HACK
                    # Sets are unioned
                    val_ = ut.oset.union(*val)
                elif isinstance(val[0], (ut.oset,)):
                    # Sets are unioned
                    val_ = ut.oset.union(*val)
                elif isinstance(val[0], (set)):
                    # Sets are unioned
                    val_ = set.union(*val)
                elif isinstance(val[0], (tuple, list)):
                    # Lists are merged together
                    val_ = ut.flatten(val)
                    #val_ = ut.unique(ut.flatten(val))
                else:
                    if ut.allsame(val):
                        # Merge items that are the same
                        val_ = val[0]
                    else:
                        if merge_scalars:
                            # If mergeing scalars is ok, then
                            # Values become lists if they are different
                            val_ = val
                        else:
                            if True:
                                # If there is only one non-none value then use that.
                                other_vals = ut.filter_Nones(val)
                                if len(other_vals) == 1:
                                    val_ = val[0]
                                else:
                                    raise ValueError(
                                        'tried to merge a scalar in %r, val=%r' % (
                                            key_, val))
                            else:
                                # If merging scalars is not ok, then
                                # we must raise an error
                                raise ValueError(
                                    'tried to merge a scalar in %r, val=%r' % (
                                        key_, val))
                newgroup[key_] = [val_]
            merged_groups.append(ut.ColumnLists(newgroup))
        merged_multi = self.__class__.flatten(merged_groups)
        merged = singles + merged_multi
        return merged


# def fix_autoreload_classes():
#     """
#     http://stackoverflow.com/questions/15605925/last-exception-python-prompt
#     http://stackoverflow.com/questions/31363311/any-way-to-manually-fix-operation-of-super-after-ipython-reload-avoiding-ty
#     """
#     tb = sys.last_traceback
#     val = sys.last_value
#     next_tb = tb
#     while next_tb is not None:
#         this_tb = next_tb
#         next_tb = this_tb.tb_next
#     frame = this_tb.tb_frame
#     code = frame.f_code
#     fpath = code.co_filename
#     modname = frame.f_globals['__name__']
#     module = sys.modules[modname]
#     # ... not sure what to do now
#     # need to change local variables. seems not possible


class NamedPartial(functools.partial, NiceRepr):
    def __init__(self, func, *args, **kwargs):
        import utool as ut
        super(functools.partial, self).__init__(func, *args, **kwargs)
        self.__name__ = ut.get_funcname(func)

    def __nice__(self):
        return self.__name__


def super2(this_class, self):
    """
    Fixes an error where reload causes super(X, self) to raise an exception

    The problem is that reloading causes X to point to the wrong version of the
    class.  This function fixes the problem by searching and returning the
    correct version of the class. See example for proper usage.

    Args:
        this_class (class): class passed into super
        self (instance): instance passed into super

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> # If the parent module is reloaded, the super call may fail
        >>> # super(Foo, self).__init__()
        >>> # This will work around the problem most of the time
        >>> # ut.super2(Foo, self).__init__()
        >>> import utool as ut
        >>> class Parent(object):
        >>>     def __init__(self):
        >>>         self.parent_attr = 'bar'
        >>> class ChildSafe(Parent):
        >>>     def __init__(self):
        >>>         ut.super2(ChildSafe, self).__init__()
        >>> class ChildDanger(Parent):
        >>>     def __init__(self):
        >>>         super(ChildDanger, self).__init__()
        >>> # initial loading is fine
        >>> safe1 = ChildSafe()
        >>> danger1 = ChildDanger()
        >>> assert safe1.parent_attr == 'bar'
        >>> assert danger1.parent_attr == 'bar'
        >>> # But if we reload (via simulation), then there will be issues
        >>> Parent_orig = Parent
        >>> ChildSafe_orig = ChildSafe
        >>> ChildDanger_orig = ChildDanger
        >>> # reloading the changes the outer classname
        >>> # but the inner class is still bound via the closure
        >>> # (we simulate this by using the old functions)
        >>> # (note in reloaded code the names would not change)
        >>> class Parent_new(object):
        >>>     __init__ = Parent_orig.__init__
        >>> Parent_new.__name__ = 'Parent'
        >>> class ChildSafe_new(Parent_new):
        >>>     __init__ = ChildSafe_orig.__init__
        >>> ChildSafe_new.__name__ = 'ChildSafe'
        >>> class ChildDanger_new(Parent_new):
        >>>     __init__ = ChildDanger_orig.__init__
        >>> ChildDanger_new.__name__ = 'ChildDanger'
        >>> #
        >>> safe2 = ChildSafe_new()
        >>> assert safe2.parent_attr == 'bar'
        >>> import pytest
        >>> with pytest.raises(TypeError):
        >>>     danger2 = ChildDanger_new()
    """
    return super(fix_super_reload(this_class, self), self)


def fix_super_reload(this_class, self):
    """
    Fixes an error where reload causes super(X, self) to raise an exception

    The problem is that reloading causes X to point to the wrong version of the
    class.  This function fixes the problem by searching and returning the
    correct version of the class. See example for proper usage.

    USE `ut.super2` INSTEAD

    Args:
        this_class (class): class passed into super
        self (instance): instance passed into super

    DisableExample:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> class Parent(object):
        >>>     def __init__(self):
        >>>         self.parent_attr = 'bar'
        >>> #
        >>> class Foo(Parent):
        >>>     def __init__(self):
        >>>         # Dont do this, it will error if you reload
        >>>         # super(Foo, self).__init__()
        >>>         # Do this instead
        >>>         _Foo = ut.super2(Foo, self)
        >>>         super(_Foo, self).__init__()
        >>> self = Foo()
        >>> assert self.parent_attr == 'bar'
    """

    if isinstance(self, this_class):
        # Case where everything is ok
        this_class_now = this_class
    else:
        # Case where we need to search for the right class
        def find_parent_class(leaf_class, target_name):
            target_class = None
            from collections import deque
            queue = deque()
            queue.append(leaf_class)
            seen_ = set([])
            while len(queue) > 0:
                related_class = queue.pop()
                if related_class.__name__ != target_name:
                    for base in related_class.__bases__:
                        if base not in seen_:
                            queue.append(base)
                            seen_.add(base)
                else:
                    target_class = related_class
                    break
            return target_class
        # Find new version of class
        leaf_class = self.__class__
        target_name = this_class.__name__
        target_class = find_parent_class(leaf_class, target_name)

        this_class_now = target_class
        #print('id(this_class)     = %r' % (id(this_class),))
        #print('id(this_class_now) = %r' % (id(this_class_now),))
    assert isinstance(self, this_class_now), (
        'Failed to fix %r, %r, %r' % (self, this_class, this_class_now))
    return this_class_now


class Shortlist(NiceRepr):
    """
    Keeps an ordered collection of items.
    Removes smallest items if size grows to large.

    Example:
        >>> # DISABLE_DOCTEST
        >>> shortsize = 3
        >>> shortlist = Shortlist(shortsize)
        >>> print('shortlist = %r' % (shortlist,))
        >>> item = (10, 1)
        >>> shortlist.insert(item)
        >>> print('shortlist = %r' % (shortlist,))
        >>> item = (9, 1)
        >>> shortlist.insert(item)
        >>> print('shortlist = %r' % (shortlist,))
        >>> item = (4, 1)
        >>> shortlist.insert(item)
        >>> print('shortlist = %r' % (shortlist,))
        >>> item = (14, 1)
        >>> shortlist.insert(item)
        >>> print('shortlist = %r' % (shortlist,))
        >>> item = (1, 1)
        >>> shortlist.insert(item)
        >>> print('shortlist = %r' % (shortlist,))
    """
    def __init__(self, maxsize=None):
        self._items = []
        self._keys = []
        self.maxsize = maxsize

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __nice__(self):
        return str(self._items)

    def insert(self, item):
        import bisect
        k = item[0]
        idx = bisect.bisect_left(self._keys, k)
        self._keys.insert(idx, k)
        self._items.insert(idx, item)
        if self.maxsize is not None and len(self._keys) > self.maxsize:
            self._keys = self._keys[-self.maxsize:]
            self._items = self._items[-self.maxsize:]


def _heappush_max(heap, item):
    """ why is this not in heapq """
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap) - 1)


class PriorityQueue(NiceRepr):
    """
    abstracted priority queue for our needs

    Combines properties of dicts and heaps
    Uses a heap for fast minimum/maximum value search
    Uses a dict for fast read only operations

    CommandLine:
        python -m utool.util_dev PriorityQueue

    References:
        http://code.activestate.com/recipes/522995-priority-dict-a-priority-queue-with-updatable-prio/
        https://stackoverflow.com/questions/33024215/built-in-max-heap-api-in-python

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> items = dict(a=42, b=29, c=40, d=95, e=10)
        >>> self = ut.PriorityQueue(items)
        >>> print(self)
        >>> assert len(self) == 5
        >>> print(self.pop())
        >>> assert len(self) == 4
        >>> print(self.pop())
        >>> assert len(self) == 3
        >>> print(self.pop())
        >>> print(self.pop())
        >>> print(self.pop())
        >>> assert len(self) == 0


    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> items = dict(a=(1.0, (2, 3)), b=(1.0, (1, 2)), c=(.9, (3, 2)))
        >>> self = ut.PriorityQueue(items)

    Ignore:
        # TODO: can also use sortedcontainers to maintain priority queue
        import sortedcontainers
        queue = sortedcontainers.SortedListWithKey(items, key=lambda x: x[1])
        queue.add(('a', 1))
    """
    def __init__(self, items=None, ascending=True):
        # Use a heap for the priority queue aspect
        self._heap = []
        # Use a dict for very quick read only operations
        self._dict = {}
        if ascending:
            self._heapify = heapq.heapify
            self._heappush = heapq.heappush
            self._heappop = heapq.heappop
        else:
            self._heapify = heapq._heapify_max
            self._heappush = _heappush_max
            self._heappop = heapq._heappop_max
        if items is not None:
            self.update(items)

    def _rebuild(self):
        # Worst Case O(N)
        self._heap = [(v, k) for k, v in self._dict.items()]
        self._heapify(self._heap)

    def __len__(self):
        return len(self._dict)

    def __eq__(self, other):
        return self._dict == other._dict

    def __nice__(self):
        return 'size=%r' % (len(self),)

    def __contains__(self, key):
        return key in self._dict

    def __getitem__(self, key):
        # Worse Case O(1)
        return self._dict[key]

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def __setitem__(self, key, val):
        # Ammortized O(1)
        # assert not np.isnan(val), 'no nan in PQ: {}, {}'.format(key, val)
        self._dict[key] = val
        if len(self._heap) > 2 * len(self._dict):
            # When the heap grows larger than 2 * len(self), we rebuild it from
            # scratch to avoid wasting too much memory.
            self._rebuild()
        else:
            # Simply append the new value
            self._heappush(self._heap, (val, key))

    def clear(self):
        self._heap.clear()
        self._dict.clear()

    def __delitem__(self, key):
        del self._dict[key]

    def update(self, items):
        if isinstance(items, dict):
            items = items.items()
        items = list(items)
        if items:
            if len(items) > len(self._dict) / 2:
                self._dict.update(items)
                self._rebuild()
            else:
                for k, v in items:
                    self[k] = v

    def delete_items(self, key_list):
        for key in key_list:
            try:
                del self[key]
            except KeyError:
                pass

    def peek(self):
        """
        Peek at the next item in the queue
        """
        # Ammortized O(1)
        _heap = self._heap
        _dict = self._dict
        val, key = _heap[0]
        # Remove items marked for lazy deletion as they are encountered
        while key not in _dict or _dict[key] != val:
            self._heappop(_heap)
            val, key = _heap[0]
        return key, val

    def peek_many(self, n):
        """
        Actually this can be quite inefficient

        Example:
            >>> # DISABLE_DOCTEST
            >>> import utool as ut
            >>> items = list(zip(range(256), range(256)))
            >>> n = 32
            >>> ut.shuffle(items)
            >>> self = ut.PriorityQueue(items, ascending=False)
            >>> self.peek_many(56)
        """
        if n == 0:
            return []
        elif n == 1:
            return [self.peek()]
        else:
            items = list(self.pop_many(n))
            self.update(items)
            return items

    def pop_many(self, n):
        count = 0
        while len(self._dict) > 0 and count < n:
            yield self.pop()
            count += 1

    def pop(self, key=util_const.NoParam, default=util_const.NoParam):
        """
        Pop the next item off the queue
        """
        # Dictionary pop if key is specified
        if key is not util_const.NoParam:
            if default is util_const.NoParam:
                return (key, self._dict.pop(key))
            else:
                return (key, self._dict.pop(key, default))
        # Otherwise do a heap pop
        try:
            # Ammortized O(1)
            _heap = self._heap
            _dict = self._dict
            val, key = self._heappop(_heap)
            # Remove items marked for lazy deletion as they are encountered
            while key not in _dict or _dict[key] != val:
                val, key = self._heappop(_heap)
        except IndexError:
            if len(_heap) == 0:
                raise IndexError('queue is empty')
            else:
                raise
        del _dict[key]
        return key, val


def pandas_reorder(df, order):
    import utool as ut
    new_order = ut.partial_order(df.columns, order)
    new_df = df.reindex_axis(new_order, axis=1)
    return new_df


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_dev
        python -m utool.util_dev --allexamples
        python -m utool.util_dev --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
