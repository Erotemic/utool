from __future__ import absolute_import, division, print_function
import sys
import six
import re
import os
import gc
import warnings
import weakref
import itertools
from collections import OrderedDict
from six.moves import input
from utool import util_progress
from os.path import splitext, exists, join, split, relpath
from utool import util_inject
from utool import util_dict
from utool import util_arg
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError as ex:
    HAS_NUMPY = False
    pass
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[dev]')

if HAS_NUMPY:
    INDEXABLE_TYPES = (list, tuple, np.ndarray)
else:
    INDEXABLE_TYPES = (list, tuple)


class ClassNoParam(object):
    """
    # class used in place of None when None might be a valid value
    # probably should just make None not a valid value
    """
    def __init__(self):
        pass
    def __call__(self, default):
        pass

# Used instance of NoParam
NoParam = ClassNoParam()


def DEPRICATED(func):
    """ deprication decorator """
    warn_msg = 'Deprecated call to: %s' % func.__name__

    def __DEP_WRAPPER(*args, **kwargs):
        raise Exception('dep')
        warnings.warn(warn_msg, category=DeprecationWarning)
        #warnings.warn(warn_msg, category=DeprecationWarning)
        return func(*args, **kwargs)
    __DEP_WRAPPER.__name__ = func.__name__
    __DEP_WRAPPER.__doc__ = func.__doc__
    __DEP_WRAPPER.__dict__.update(func.__dict__)
    return __DEP_WRAPPER


#def ensure_vararg_list(varargs):
#    """
#    It is useful to have a function take a list of objects to act upon.
#    But sometimes you want just one. Varargs lets you pass in as many as you
#    want, and it lets you have just one if needbe.
#    But sometimes the function caller explicitly passes in the list. In that
#    case we parse it out
#    """
#    if len(varargs) == 1:
#        if isinstance(varargs[0], INDEXABLE_TYPES):
#            return varargs[0]
#    return varargs

def ensure_str_list(input_):
    return [input_] if isinstance(input_, six.string_types) else input_


def set_clipboard(text):
    """
    References:
        http://stackoverflow.com/questions/11063458/python-script-to-copy-text-to-clipboard
        http://stackoverflow.com/questions/579687/how-do-i-copy-a-string-to-the-clipboard-on-windows-using-python
    """
    import utool as ut
    pyperclip = ut.tryimport('pyperclip', ensure=True)
    if pyperclip is not None:
        pyperclip.copy(text)
    else:
        from Tkinter import Tk
        tk_inst = Tk()
        tk_inst.withdraw()
        tk_inst.clipboard_clear()
        tk_inst.clipboard_append(text)
        tk_inst.destroy()


def copy_text_to_clipboard(text):
    """ alias """
    return set_clipboard(text)


def get_clipboard():
    """
    References:
        http://stackoverflow.com/questions/11063458/python-script-to-copy-text-to-clipboard
    """
    import utool as ut
    pyperclip = ut.tryimport('pyperclip', ensure=True)
    if pyperclip is not None:
        text = pyperclip.paste()
    else:
        from Tkinter import Tk
        tk_inst = Tk()
        tk_inst.withdraw()
        text = tk_inst.clipboard_get()
        tk_inst.destroy()
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
    for count in itertools.count(offset):
        base_str = base_fmtstr % count
        if base_str not in conflict_set_:
            return base_str


def get_nonconflicting_path(base_fmtstr, dpath, offset=0):
    import utool as ut
    from os.path import basename
    pattern = '*'
    dname_list = ut.glob(dpath, pattern, recursive=False,
                               with_files=True, with_dirs=True)
    conflict_set = set([basename(dname) for dname in dname_list])

    newname = ut.get_nonconflicting_string(base_fmtstr, conflict_set, offset=offset)
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
        import plottool as pt
        color_list = pt.distinct_colors(len(stmt_list))
        for count, (stmt, color) in enumerate(zip(stmt_list, color_list)):
            pt.plot(input_sizes, time_grid.T[count], 'x-', color=color, label=stmt)
        pt.dark_background()
        pt.legend()
        pt.show_if_requested()
    return time_grid


def timeit_compare(stmt_list, setup='', iterations=100000, verbose=True,
                   strict=False):
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

    Example:
        >>> import utool
        >>> setup = utool.unindent(
            '''
            import numpy as np
            np.random.seed(0)
            invVR_mats = np.random.rand(1000, 3, 3).astype(np.float64)
            ''')
        >>> stmt1 = 'invVR_mats[:, 0:2, 2].T'
        >>> stmt2 = 'invVR_mats.T[2, 0:2]'
        >>> iterations = 100000
        >>> verbose = True
        >>> stmt_list = [stmt1, stmt2]
        >>> utool.timeit_compare(stmt_list, setup='', iterations=1000, verbose=True)
    """
    import timeit
    import utool

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

    result_list = [testit(stmt, setup) for stmt in stmt_list]
    time_list   = [timeit.timeit(stmt, setup=setup, number=iterations)
                   for stmt in stmt_list]

    passed = utool.util_list.list_allsame(result_list)
    if verbose:
        print('| Output:')
        if not passed:
            print('|    * FAILED: results differ between some statements')
            print('| Results:')
            for result in result_list:
                for count, result in enumerate(result_list):
                    print('<Result %d>' % count)
                    #print(ut.truncate_str(repr(result)))
                    print('</Result %d>' % count)
            if strict:
                raise AssertionError('Results are not valid')
        else:
            print('|    * PASSED: each statement produced the same result')
            passed = True
        #print('|    +-----------------------------------')
        print('|    | num | total time | per loop | stmt')
        for count, tup in enumerate(zip(stmt_list, time_list)):
            stmt, time = tup
            print('|    | %3d | %10s | %8s | %s' %
                  (count, utool.seconds_str(time),
                   utool.seconds_str(time / iterations), stmt))
        #print('|    L___________________________________')
        if verbose:
            print('L_________________')
        return (passed, time_list, result_list)


def testit(stmt, setup):
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
       from http://stackoverflow.com/questions/141351/how-do-i-find-what-is-using-memory-in-a-python-process-in-a-production-system
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


class MemoryTracker(object):
    """
    A ``class`` for tracking memory usage.
    On initialization it logs the current available (free) memory.
    Calling the report method logs the current available memory as well
    as memory usage difference w.r.t the last report.

    Example:
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
    def __init__(self, lbl='Memtrack Init', disable=True):
        self.disabled = disable  # disable by default
        self.init_nBytes = self.get_available_memory()
        self.prev_nBytes = None
        self.weakref_dict = {}  # weakref.WeakValueDictionary()
        self.weakref_dict2 = {}
        self.report(lbl)

    @_disableable
    def __call__(self, lbl=''):
        self.report(lbl=lbl)

    @_disableable
    def collect(self):
        gc.collect()

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
        size_list = np.array([utool.get_object_size(obj) for obj in ndarray_list])
        sortx = size_list.argsort()[::-1]
        ndarray_sorted = [ndarray_list[x] for x in sortx]
        for obj, size in zip(ndarray_sorted, size_list):
            print('size = %r, type(obj) = %r' % (utool.byte_str2(size), type(obj)))

        #size_list = [utool.get_object_size(obj) for obj in obj_list]
        pass

    @_disableable
    def report(self, lbl=''):
        from .util_str import byte_str2
        self.collect()
        nBytes = self.get_available_memory()
        print('[memtrack] +----')
        if self.prev_nBytes is not None:
            diff = self.prev_nBytes - nBytes
            print('[memtrack] | [%s] diff = %s' % (lbl, byte_str2(diff)))
        else:
            print('[memtrack] | new MemoryTracker(%s)' % (lbl,))

        total_diff = self.init_nBytes - nBytes
        print('[memtrack] | Total diff = %s' % (byte_str2(total_diff)))
        print('[memtrack] | Available Memory = %s' %  (byte_str2(nBytes),))
        self.report_objs()
        print('[memtrack] L----')
        self.prev_nBytes = nBytes

    @_disableable
    def get_available_memory(self):
        from .util_resources import available_memory
        return available_memory()

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


def report_memsize(obj, name=None, verbose=True):
    #import types
    import utool
    if name is None:
        name = 'obj'

    if not isinstance(obj, weakref.ref):
        obj = weakref.ref(obj)

    if obj() is None:
        with utool.Indenter('|   '):
            print('+----')
            print('Memsize: ')
            print('type(%s) = %r' % (name, type(obj())))
            print('%s has been deallocated' % name)
            print('L____')
            return

    referents = gc.get_referents(obj())
    referers  = gc.get_referrers(obj())
    with utool.Indenter('|   '):
        print('+----')
        print('Memsize: ')
        print('type(%s) = %r' % (name, type(obj())))
        print('%s is using: %s' % (name, utool.get_object_size_str(obj())))
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
                    if isinstance(referer, dict):
                        print('    len(referer) = %r' % len(referer))
                        if len(referer) < 30:
                            keystr = utool.packstr(repr(referer.keys()), 60, newline_prefix='        ')
                            print('    referer.keys = %s' % (keystr),)
                    print('    id(referer) = %r' % id(referer))
                    #print('referer = ' + utool.truncate_str(repr(referer)))
                    print('  </Referer %d>' % count)
        del obj
        del referents
        del referers
        print('L____')


class InteractiveIter(object):
    """
    Choose next value interactively

    iterable should be a list, not a generator. sorry
    """
    def __init__(iiter, iterable=None, enabled=True, startx=0,
                 default_action='next', custom_actions=[], wraparound=False):
        import utool as ut
        iiter.wraparound = wraparound
        iiter.enabled = enabled
        iiter.iterable = iterable
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
        pass

    def __iter__(iiter):
        import utool as ut
        if not iiter.enabled:
            for item in ut.ProgressIter(iiter.iterable, lbl='nointeract: '):
                yield item
            #raise StopIteration()
        assert isinstance(iiter.iterable, INDEXABLE_TYPES)
        iiter.num_items = len(iiter.iterable)
        print('[IITER] Begin interactive iteration: %r items\n' % (iiter.num_items))
        if iiter.num_items == 0:
            raise StopIteration
        mark_, end_ = util_progress.log_progress(total=iiter.num_items, lbl='interaction: ', freq=1)
        while True:
            print('')
            if iiter.wraparound:
                iiter.index = iiter.index % len(iiter.iterable)
            if iiter.index >= len(iiter.iterable):
                print('Got to end the end of the iterable')
                break
            mark_(iiter.index)
            item = iiter.iterable[iiter.index]
            yield item
            print('')
            mark_(iiter.index)
            print('')
            print('[IITER] current index=%r' % (iiter.index,))
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

    def handle_ans(iiter, ans_):
        ans = ans_.strip(' ')
        def parse_str_value(ans):
            return ' '.join(ans.split(' ')[1:])
        def chack_if_answer_was(valid_keys):
            return any([ans == key or ans.startswith(key + ' ') for key in valid_keys])
        # Quit
        if ans in iiter.action_keys['quit']:
            raise StopIteration()
        # Prev
        elif ans in iiter.action_keys['prev']:
            iiter.index -= 1
        # Next
        elif ans in iiter.action_keys['next']:
            iiter.index += 1
        # Reload
        elif ans in iiter.action_keys['reload']:
            iiter.index += 0
        # Index
        elif chack_if_answer_was(iiter.action_keys['index']):
            try:
                iiter.index = int(parse_str_value(ans))
            except ValueError:
                print('Unknown ans=%r' % (ans,))
        # Set
        elif chack_if_answer_was(iiter.action_keys['set']):
            try:
                iiter.iterable[iiter.index] = eval(parse_str_value(ans))
            except ValueError:
                print('Unknown ans=%r' % (ans,))
        # IPython
        elif ans in iiter.action_keys['ipy']:
            return 'IPython'
        else:
            # Custom interactions
            for func, tup in zip(iiter.custom_funcs, iiter.custom_actions):
                key = tup[0]
                if chack_if_answer_was(iiter.action_keys[key]):
                    value  = parse_str_value(ans)
                    func(iiter, key, value)
                    return True
            print('Unknown ans=%r' % (ans,))
            return False
        return True

    def prompt(iiter):
        import utool as ut
        def _or_phrase(list_):
            return ut.cond_phrase(list(map(repr, list_)), 'or')
        msg_list = ['enter %s to %s' % (_or_phrase(tup[1]), tup[2])
                    for tup in iiter.action_tuples]
        msg = ut.indentjoin(msg_list, '\n | * ')
        msg = ''.join([' +-----------', msg, '\n L-----------\n'])
        # TODO: timeout, help message
        ans = input(msg).strip()
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
    #    no_phrase=ut.cond_phrase(ans_list_no),
    #    embed_phrase=ut.cond_phrase(ans_list_embed),
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
    valid_prompt = util_str.cond_phrase(valid_ans, 'or')
    ans = input('Are you sure?\n Enter %s to accept\n' % valid_prompt)
    return ans.lower() in valid_ans


def tuples_to_unique_scalars(tup_list):
    seen = {}
    def addval(tup):
        val = len(seen)
        seen[tup] = val
        return val
    scalar_list = [seen[tup] if tup in seen else addval(tup) for tup in tup_list]
    return scalar_list

STAT_KEY_ORDER = ['max', 'min', 'mean', 'sum', 'std', 'nMin', 'nMax', 'shape', 'num_nan']


def find_interesting_stats(stat_dict, col_lbls=None, lbl=None):
    #argfind = np.argmax
    import utool as ut
    # select indices of interest
    sel_indices = []
    #statstr_kw = dict(precision=3, newlines=True, lbl=lbl, align=True)
    for key in ['max', 'mean', 'sum']:
        if key not in stat_dict:
            continue
        sortx  = np.argsort(stat_dict[key])
        if len(sortx) > 4:
            sel_sortx = sortx.take([0, 1, -2, -1])
        else:
            sel_sortx = sortx
        sel_indices.extend(sel_sortx)
    sel_indices = ut.unique_keep_order2(sel_indices)
    sel_stat_dict = ut.get_dict_column(stat_dict, sel_indices)
    sel_stat_dict = ut.order_dict_by(sel_stat_dict, STAT_KEY_ORDER)
    return sel_stat_dict, sel_indices


#def jagged_stats_str(arr, use_nan=True, lbl=None):
#    """
#    builds stats over all columns in arr. Can find interesting column labels

#    Args:
#        arr (?):  list of potentially non-parallel lists
#        use_nan (bool):
#        lbl (None):
#    """
#    stat_dict = get_jagged_stats(arr, use_nan=use_nan)
#    statstr_kw = dict(precision=3, newlines=True, lbl=lbl, align=True)
#    stat_str =  get_stats_str(stat_dict=stat_dict, **statstr_kw)
#    REPORT_INTERESTING = True
#    if REPORT_INTERESTING and col_lbls is not None:
#        pass

#    #import utool as ut
#    #ut.embed()

#    return stat_str


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
        >>> # build test data
        >>> kwargs = dict(use_nan=True)
        >>> arr_list = [[1, 2, 3, 4], [3, 10], [np.nan, 3, 3, 3]]
        >>> # execute function
        >>> stats_dict = get_jagged_stats(arr_list, **kwargs)
        >>> # verify results
        >>> result = ut.align(str(ut.dict_str(stats_dict)), ':')
        >>> print(result)
        {
            'max'    : [4.0, 10.0, 3.0],
            'min'    : [1.0, 3.0, 3.0],
            'mean'   : [2.5, 6.5, 3.0],
            'std'    : [1.118034, 3.5, 0.0],
            'nMin'   : [1, 1, 3],
            'nMax'   : [1, 1, 3],
            'shape'  : ['(4,)', '(2,)', '(4,)'],
            'num_nan': [0, 0, 1],
        }

    """
    import functools
    stats_dict_list = list(map(functools.partial(get_stats, **kwargs), arr_list))
    stats_dict_ = util_dict.dict_stack(stats_dict_list)
    # Fix order
    stats_dict = util_dict.order_dict_by(stats_dict_, STAT_KEY_ORDER)
    return stats_dict


def get_stats(list_, axis=None, use_nan=False, use_sum=False):
    """
    Args:
        list_ (listlike): values to get statistics of
        axis (int): if ``list_`` is ndarray then this specifies the axis

    Returns:
        OrderedDict: stat_dict - dictionary of common numpy statistics
            (min, max, mean, std, nMin, nMax, shape)


    CommandLine:
        python -m utool.util_dev --test-get_stats

    Examples:
        >>> # ENABLE_DOCTEST
        >>> import numpy as np
        >>> import utool
        >>> axis = 0
        >>> np.random.seed(0)
        >>> list_ = np.random.rand(10, 2)
        >>> stat_dict = get_stats(list_, axis, use_nan=False)
        >>> result = str(utool.dict_str(stat_dict))
        >>> print(result)
        {
            'max': np.array([ 0.96366274,  0.92559665], dtype=np.float32),
            'min': np.array([ 0.0202184,  0.0871293], dtype=np.float32),
            'mean': np.array([ 0.52056623,  0.64254338], dtype=np.float32),
            'std': np.array([ 0.28543401,  0.25168759], dtype=np.float32),
            'nMin': np.array([1, 1], dtype=np.int32),
            'nMax': np.array([1, 1], dtype=np.int32),
            'shape': (10, 2),
        }

    SeeAlso:
        print_stats
        get_stats_str
    """
    # Assure input is in numpy format
    if isinstance(list_, np.ndarray):
        nparr = list_
    elif isinstance(list_, list):
        nparr = np.array(list_)
    else:
        list_ = list(list_)
        nparr = np.array(list_)
    # Check to make sure stats are feasible
    if len(list_) == 0:
        stat_dict = {'empty_list': True}
    else:
        #import utool as ut
        #if np.any(np.isnan(nparr)):
        #    ut.embed()
        # Compute stats
        if use_nan:
            min_val = np.nanmin(nparr, axis=axis)
            max_val = np.nanmax(nparr, axis=axis)
            mean_ = np.nanmean(nparr, axis=axis)
            std_  = np.nanstd(nparr, axis=axis)
            # TODO report num nans
        else:
            min_val = nparr.min(axis=axis)
            max_val = nparr.max(axis=axis)
            mean_ = nparr.mean(axis=axis)
            std_  = nparr.std(axis=axis)
        # number of entries with min val
        nMin = np.sum(nparr == min_val, axis=axis)
        # number of entries with min val
        nMax = np.sum(nparr == max_val, axis=axis)
        stats_list = [
            ('max',   np.float32(max_val)),
            ('min',   np.float32(min_val)),
            ('mean',  np.float32(mean_)),
            ('std',   np.float32(std_)),
            ('nMin',  np.int32(nMin)),
            ('nMax',  np.int32(nMax)),
            ('shape', nparr.shape),  # repr(nparr.shape)),
        ]
        if use_nan:
            stats_list.append(('num_nan', np.isnan(nparr).sum()))
        if use_sum:
            sumfunc = np.nansum if use_nan else np.sum
            stats_list.append(('sum', sumfunc(nparr, axis=axis)))
        stat_dict = OrderedDict(stats_list)
    return stat_dict

# --- Info Strings ---


def get_stats_str(list_=None, newlines=False, keys=None, exclude_keys=[], lbl=None,
                  precision=None, axis=0, stat_dict=None, use_nan=False,
                  align=False):
    """
    Returns the string version of get_stats

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
        {'max': 5, 'min': 1, 'mean': 3, 'std': 1.41, 'nMin': 1, 'nMax': 1, 'shape': (5,),}

    SeeAlso:
        print_stats
        get_stats
    """
    from utool.util_str import dict_str
    import utool as ut
    # Get stats dict
    if stat_dict is None:
        stat_dict = get_stats(list_, axis=axis, use_nan=use_nan)
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
    #with ut.EmbedOnException():
    #precisionless_types =  (bool,) + six.string_types
    if precision is not None:
        float_fmtstr = '%.' + str(precision) + 'f'
        for key in list(six.iterkeys(statstr_dict)):
            val = statstr_dict[key]
            isfloat = ut.is_float(val)
            if not isfloat and isinstance(val, list):
                type_list = list(map(type, val))
                if len(type_list) > 0 and ut.list_allsame(type_list):
                    if ut.is_float(val[0]):
                        isfloat = True
                        val = np.array(val)
            if isfloat:
                if isinstance(val, np.ndarray):
                    strval = str([float_fmtstr % v for v in val]).replace('\'', '')
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
    stat_str  = dict_str(statstr_dict, strvals=True, newlines=newlines)
    # add a label if requested
    if lbl is True:
        lbl = ut.get_varname_from_stack(list_, N=1)  # fancy
    if lbl is not None:
        stat_str = 'stats_' + lbl + ' = ' + stat_str
    if align:
        stat_str = ut.align(stat_str, ':')
    return stat_str


def print_stats(list_, lbl=None, newlines=False, precision=2):
    """
    Prints string representation of stat of list_

    CommandLine:
        python -m utool.util_dev --test-print_stats

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> list_ = [1, 2, 3, 4, 5]
        >>> lbl = None
        >>> newlines = False
        >>> precision = 2
        >>> result = print_stats(list_, lbl, newlines, precision)
        {'max': 5, 'min': 1, 'mean': 3, 'std': 1.41, 'nMin': 1, 'nMax': 1, 'shape': (5,),}

    SeeAlso:
        get_stats_str
        get_stats

    """
    if lbl is not None:
        print('Stats for %s' % lbl)
    stat_str = get_stats_str(list_, newlines=newlines, precision=2)
    print(stat_str)
    return stat_str


def npArrInfo(arr):
    """
    OLD update and refactor
    """
    from utool.DynamicStruct import DynStruct
    info = DynStruct()
    info.shapestr  = '[' + ' x '.join([str(x) for x in arr.shape]) + ']'
    info.dtypestr  = str(arr.dtype)
    if info.dtypestr == 'bool':
        info.bittotal = 'T=%d, F=%d' % (sum(arr), sum(1 - arr))
    elif info.dtypestr == 'object':
        info.minmaxstr = 'NA'
    elif info.dtypestr[0] == '|':
        info.minmaxstr = 'NA'
    else:
        if arr.size > 0:
            info.minmaxstr = '(%r, %r)' % (arr.min(), arr.max())
        else:
            info.minmaxstr = '(None)'
    return info


def printableType(val, name=None, parent=None):
    """
    Tries to make a nice type string for a value.
    Can also pass in a Printable parent object
    """
    if parent is not None and hasattr(parent, 'customPrintableType'):
        # Hack for non - trivial preference types
        _typestr = parent.customPrintableType(name)
        if _typestr is not None:
            return _typestr
    if isinstance(val, np.ndarray):
        info = npArrInfo(val)
        _typestr = info.dtypestr
    elif isinstance(val, object):
        _typestr = val.__class__.__name__
    else:
        _typestr = str(type(val))
        _typestr = _typestr.replace('type', '')
        _typestr = re.sub('[\'><]', '', _typestr)
        _typestr = re.sub('  *', ' ', _typestr)
        _typestr = _typestr.strip()
    return _typestr


def printableVal(val, type_bit=True, justlength=False):
    """
    Very old way of doing pretty printing. Need to update and refactor.
    """
    from utool import util_dev
    # Move to util_dev
    # NUMPY ARRAY
    if type(val) is np.ndarray:
        info = npArrInfo(val)
        if info.dtypestr.startswith('bool'):
            _valstr = '{ shape:' + info.shapestr + ' bittotal: ' + info.bittotal + '}'  # + '\n  |_____'
        elif info.dtypestr.startswith('float'):
            _valstr = util_dev.get_stats_str(val)
        else:
            _valstr = '{ shape:' + info.shapestr + ' mM:' + info.minmaxstr + ' }'  # + '\n  |_____'
    # String
    elif isinstance(val, (str, unicode)):
        _valstr = '\'%s\'' % val
    # List
    elif isinstance(val, list):
        if justlength or len(val) > 30:
            _valstr = 'len=' + str(len(val))
        else:
            _valstr = '[ ' + (', \n  '.join([str(v) for v in val])) + ' ]'
    elif hasattr(val, 'get_printable') and type(val) != type:  # WTF? isinstance(val, AbstractPrintable):
        _valstr = val.get_printable(type_bit=type_bit)
    elif isinstance(val, dict):
        _valstr = '{\n'
        for val_key in val.keys():
            val_val = val[val_key]
            _valstr += '  ' + str(val_key) + ' : ' + str(val_val) + '\n'
        _valstr += '}'
    else:
        _valstr = str(val)
    if _valstr.find('\n') > 0:  # Indent if necessary
        _valstr = _valstr.replace('\n', '\n    ')
        _valstr = '\n    ' + _valstr
    _valstr = re.sub('\n *$', '', _valstr)  # Replace empty lines
    return _valstr


def myprint(input_=None, prefix='', indent='', lbl=''):
    """
    OLD PRINT FUNCTION USED WITH PRINTABLE VAL
    TODO: Refactor and update
    """
    if len(lbl) > len(prefix):
        prefix = lbl
    if len(prefix) > 0:
        prefix += ' '
    print_(indent + prefix + str(type(input_)) + ' ')
    if isinstance(input_, list):
        print(indent + '[')
        for item in iter(input_):
            myprint(item, indent=indent + '  ')
        print(indent + ']')
    elif isinstance(input_, six.string_types):
        print(input_)
    elif isinstance(input_, dict):
        print(printableVal(input_))
    else:
        print(indent + '{')
        attribute_list = dir(input_)
        for attr in attribute_list:
            if attr.find('__') == 0:
                continue
            val = str(input_.__getattribute__(attr))
            #val = input_[attr]
            # Format methods nicer
            #if val.find('built-in method'):
            #    val = '<built-in method>'
            print(indent + '  ' + attr + ' : ' + val)
        print(indent + '}')


def info(var, lbl):
    if isinstance(var, np.ndarray):
        return npinfo(var, lbl)
    if isinstance(var, list):
        return listinfo(var, lbl)


def npinfo(ndarr, lbl='ndarr'):
    info = ''
    info += (lbl + ': shape=%r ; dtype=%r' % (ndarr.shape, ndarr.dtype))
    return info


def listinfo(list_, lbl='ndarr'):
    if not isinstance(list_, list):
        raise Exception('!!')
    info = ''
    type_set = set([])
    for _ in iter(list_):
        type_set.add(str(type(_)))
    info += (lbl + ': len=%r ; types=%r' % (len(list_), type_set))
    return info


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
#    #Once youve gotten your code setup with the <AT>profile decorator, use kernprof.py to run your script.
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
    import utool
    import guppy
    if with_gc:
        garbage_collect()
    hp = guppy.hpy()
    print('[hpy] Waiting for heap output...')
    heap_output = hp.heap()
    print(heap_output)
    print('[hpy] total heap size: ' + utool.byte_str2(heap_output.size))
    utool.util_resources.memstats()
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


def get_object_size(obj):
    seen = set([])
    def _get_object_size(obj):
        if (obj is None or isinstance(obj, (str, int, bool, float))):
            return sys.getsizeof(obj)

        object_id = id(obj)
        if object_id in seen:
            return 0
        seen.add(object_id)

        totalsize = sys.getsizeof(obj)
        if isinstance(obj, np.ndarray):
            totalsize += obj.nbytes
        elif (isinstance(obj, (tuple, list, set, frozenset))):
            for item in obj:
                totalsize += _get_object_size(item)
        elif isinstance(obj, dict):
            try:
                for key, val in six.iteritems(obj):
                    totalsize += _get_object_size(key)
                    totalsize += _get_object_size(val)
            except RuntimeError:
                print(key)
                raise
        elif isinstance(obj, object) and hasattr(obj, '__dict__'):
            totalsize += _get_object_size(obj.__dict__)
            return totalsize
        return totalsize
    return _get_object_size(obj)


def print_object_size_tree(obj):
    """ Needs work """

    def _get_object_size_tree(obj, indent='', lbl='obj', seen=None):
        if (obj is None or isinstance(obj, (str, int, bool, float))):
            return [sys.getsizeof(obj)]
        object_id = id(obj)
        if object_id in seen:
            return []
        seen.add(object_id)
        size_list = [(lbl, sys.getsizeof(obj))]
        print(indent + '%s = %s ' % (lbl, str(sys.getsizeof(obj))))
        if isinstance(obj, np.ndarray):
            size_list.append(obj.nbytes)
            print(indent + '%s = %s ' % ('arr', obj.nbytes))
        elif (isinstance(obj, (tuple, list, set, frozenset))):
            for item in obj:
                size_list += _get_object_size_tree(item, indent + '   ', 'item', seen)
        elif isinstance(obj, dict):
            try:
                for key, val in six.iteritems(obj):
                    size_list += _get_object_size_tree(key, indent + '   ', key, seen)
                    size_list += _get_object_size_tree(val, indent + '   ', key, seen)
            except RuntimeError:
                print(key)
                raise
        elif isinstance(obj, object) and hasattr(obj, '__dict__'):
            size_list += _get_object_size_tree(obj.__dict__, indent + '   ', 'dict', seen)
            return size_list
        return size_list
    seen = set([])
    _get_object_size_tree(obj, '', 'obj', seen)
    del seen


def get_object_size_str(obj, lbl='', unit=None):
    from utool import util_str
    nBytes = get_object_size(obj)
    if unit is None:
        sizestr = lbl + util_str.byte_str2(nBytes)
    else:
        sizestr = lbl + util_str.byte_str(nBytes, unit)
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

#C:\MinGW\bin\gcc.exe -shared -s build\temp.win32-2.7\Release\vtool\linalg_cython.o build\temp.win32-2.7\Release\vtool\linalg_cython.def -LC:\Python27\libs -LC:\Python27\PCbuild -lpython27 -lmsvcr90 -o  build\lib.win32-2.7\linalg_cython.pyd

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
    #C:\MinGW\bin\gcc.exe -mdll -O -Wall -IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include -IC:\Python27\PC -c vtool\linalg_cython.c -o build\temp.win32-2.7\Release\vtool\linalg_cyth

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
    #    out, err, ret = util_cplat.shell(cc_exe + ' ' + gcc_flags + ' -o ' + fname_so + ' ' + fname_c)
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


USER_MODE      =  util_arg.get_argflag(('--user-mode', '--no-developer', '--nodev', '--nodeveloper'))
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
