"""
Module to executes the same function with different arguments in parallel.
"""
from __future__ import absolute_import, division, print_function
import multiprocessing
import atexit
import sys
import signal
import six
if six.PY2:
    import thread as _thread
elif six.PY3:
    import _thread
import threading
from utool._internal.meta_util_six import get_funcname
from utool import util_progress
from utool import util_time
from utool import util_arg
from utool import util_dbg
from utool import util_inject
from utool import util_cplat
util_inject.noinject('[parallel]')

QUIET   = util_arg.QUIET
SILENT  = util_arg.SILENT
PAR_VERBOSE = util_arg.VERBOSE or util_arg.get_argflag(('--verbose-par', '--verbpar'))
STRICT  = util_arg.STRICT

if SILENT:
    def print(msg):
        pass

__POOL__ = None
__EAGER_JOIN__      = not util_arg.get_argflag('--noclose-pool')
__TIME_GENERATE__   = util_arg.get_argflag('--time-generate')
__NUM_PROCS__       = util_arg.get_argval('--num-procs', int, default=None)
__FORCE_SERIAL__    = util_arg.get_argflag(('--utool-force-serial', '--force-serial', '--serial'))
__SERIAL_FALLBACK__ = not util_arg.get_argflag('--noserial-fallback')


MIN_PARALLEL_TASKS = 2
if util_cplat.WIN32:
    MIN_PARALLEL_TASKS = 16


BACKEND = 'multiprocessing'

#TODO:
#    http://dispy.sourceforge.net/

if BACKEND == 'gevent':
    raise NotImplementedError('gevent cannot run on multiple cpus')
    pass
elif BACKEND == 'zeromq':
    # TODO: Implement zeromq backend
    #http://zguide.zeromq.org/py:mtserver
    raise NotImplementedError('no zeromq yet')
    pass
elif BACKEND == 'multiprocessing':
    """
    expecting
    multiprocessing.__file__ = /usr/lib/python2.7/multiprocessing/__init__.pyc
    multiprocessing.__version__ >= 0.70a1

    BUT PIP SAYS:
        INSTALLED: 2.6.2.1 (latest)

    because multiprocessing on pip is: Backport of the multiprocessing package to Python 2.4 and 2.5

    ut.editfile(multiprocessing.__file__)
    from multiprocessing.pool import ThreadPool
    """
    def new_pool(num_procs, init_worker, maxtasksperchild):
        return multiprocessing.Pool(processes=num_procs,
                                    initializer=init_worker,
                                    maxtasksperchild=maxtasksperchild)
    pass


def set_num_procs(num_procs):
    global __NUM_PROCS__
    __NUM_PROCS__ = num_procs


def in_main_process():
    """ Returns if you are executing in a multiprocessing subprocess
    Usefull to disable init print messages on windows """
    return multiprocessing.current_process().name == 'MainProcess'


def get_default_numprocs():
    if __NUM_PROCS__ is not None:
        return __NUM_PROCS__
    #if WIN32:
    #    num_procs = 3  # default windows to 3 processes for now
    #else:
    #    num_procs = max(multiprocessing.cpu_count() - 2, 1)
    num_procs = max(multiprocessing.cpu_count() - 1, 1)
    return num_procs


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def init_pool(num_procs=None, maxtasksperchild=None):
    """ warning this might not be the right hting to do """
    global __POOL__
    if PAR_VERBOSE:
        print('[util_parallel] init_pool()')
    if num_procs is None:
        # Get number of cpu cores
        num_procs = get_default_numprocs()
    if not QUIET:
        print('[util_parallel.init_pool] initializing pool with %d processes' % num_procs)
    if num_procs == 1:
        print('[util_parallel.init_pool] num_procs=1, Will process in serial')
        __POOL__ = 1
        return
    if STRICT:
        #assert __POOL__ is None, 'pool is a singleton. can only initialize once'
        assert multiprocessing.current_process().name, 'can only initialize from main process'
    if __POOL__ is not None:
        print('[util_parallel.init_pool] close pool before reinitializing')
        return
    # Create the pool of processes
    #__POOL__ = multiprocessing.Pool(processes=num_procs, initializer=init_worker, maxtasksperchild=maxtasksperchild)
    __POOL__ = new_pool(num_procs, init_worker, maxtasksperchild)


@atexit.register
def close_pool(terminate=False):
    global __POOL__
    if PAR_VERBOSE:
        print('[util_parallel] close_pool()')

    if __POOL__ is not None:
        if not QUIET:
            if terminate:
                print('[util_parallel] terminating pool')
            else:
                print('[util_parallel] closing pool')
        if not isinstance(__POOL__, int):
            # Must join after close to avoid runtime errors
            if terminate:
                __POOL__.terminate()
            __POOL__.close()
            __POOL__.join()
        __POOL__ = None


def _process_serial(func, args_list, args_dict={}, nTasks=None):
    """
    Serial process map

    Use generate instead
    """
    if nTasks is None:
        nTasks = len(args_list)
    result_list = []
    mark_prog, end_prog = util_progress.progress_func(
        max_val=nTasks, lbl=get_funcname(func) + ': ')
    mark_prog(0)
    # Execute each task sequentially
    for count, args in enumerate(args_list):
        result = func(*args, **args_dict)
        result_list.append(result)
        mark_prog(count)
    end_prog()
    return result_list


def _process_parallel(func, args_list, args_dict={}, nTasks=None):
    """
    Parallel process map

    Use generate instead
    """
    # Define progress observers
    if nTasks is None:
        nTasks = len(args_list)
    num_tasks_returned_ptr = [0]
    mark_prog, end_prog = util_progress.progress_func(
        max_val=nTasks, lbl=get_funcname(func) + ': ')
    def _callback(result):
        mark_prog(num_tasks_returned_ptr[0])
        sys.stdout.flush()
        num_tasks_returned_ptr[0] += 1
    # Send all tasks to be executed asynconously
    apply_results = [__POOL__.apply_async(func, args, args_dict, _callback)
                     for args in args_list]
    # Wait until all tasks have been processed
    while num_tasks_returned_ptr[0] < nTasks:
        #print('Waiting: ' + str(num_tasks_returned_ptr[0]) + '/' + str(nTasks))
        pass
    end_prog()
    # Get the results
    result_list = [ap.get() for ap in apply_results]
    if __EAGER_JOIN__:
        close_pool()
    return result_list


def _generate_parallel(func, args_list, ordered=True, chunksize=1,
                       prog=True, verbose=True, nTasks=None, freq=None):
    """
    Parallel process generator
    """
    prog = prog and verbose
    if nTasks is None:
        nTasks = len(args_list)
    if chunksize is None:
        chunksize = max(1, nTasks // (__POOL__._processes ** 2))
    if verbose:
        prefix = '[util_parallel._generate_parallel]'
        fmtstr = prefix + 'executing %d %s tasks using %d processes with chunksize=%r'
        print(fmtstr % (nTasks, get_funcname(func), __POOL__._processes, chunksize))
    pmap_func = __POOL__.imap if ordered else __POOL__.imap_unordered
    raw_generator = pmap_func(func, args_list, chunksize)
    # Get iterator with or without progress
    result_generator = (
        util_progress.ProgressIter(raw_generator, nTotal=nTasks, lbl=get_funcname(func) + ': ', freq=freq)
        if prog else raw_generator
    )
    if __TIME_GENERATE__:
        tt = util_time.tic('_generate_parallel func=' + get_funcname(func))
    try:
        for result in result_generator:
            yield result
        if __EAGER_JOIN__:
            close_pool()
    except Exception as ex:
        util_dbg.printex(ex, 'Parallel Generation Failed!', '[utool]', tb=True)
        if __EAGER_JOIN__:
            close_pool()
        print('__SERIAL_FALLBACK__ = %r' % __SERIAL_FALLBACK__)
        if __SERIAL_FALLBACK__:
            print('Trying to handle error by falling back to serial')
            serial_generator = _generate_serial(
                func, args_list, prog=prog, verbose=verbose, nTasks=nTasks, freq=freq)
            for result in serial_generator:
                yield result
        else:
            raise
    if __TIME_GENERATE__:
        util_time.toc(tt)


def _generate_serial(func, args_list, prog=True, verbose=True, nTasks=None, freq=None):
    """ internal serial generator  """
    if nTasks is None:
        nTasks = len(args_list)
    if verbose:
        print('[util_parallel._generate_serial] executing %d %s tasks in serial' %
                (nTasks, get_funcname(func)))
    prog = prog and verbose and nTasks > 1
    # Get iterator with or without progress
    args_iter = (
        util_progress.ProgressIter(args_list, nTotal=nTasks, lbl=get_funcname(func) + ': ', freq=freq)
        if prog else args_list
    )
    if __TIME_GENERATE__:
        tt = util_time.tic('_generate_serial func=' + get_funcname(func))
    for args in args_iter:
        result = func(args)
        yield result
    if __TIME_GENERATE__:
        util_time.toc(tt)


def ensure_pool(warn=False):
    try:
        assert __POOL__ is not None, 'must init_pool() first'
    except AssertionError as ex:
        if warn:
            print('(WARNING) AssertionError: ' + str(ex))
        init_pool()


def generate(func, args_list, ordered=True, force_serial=__FORCE_SERIAL__,
             chunksize=1, prog=True, verbose=True, nTasks=None, freq=None):
    """

    Args:
        func (function): function to apply each argument to
        args_list (list or iter): sequence of tuples which are args for each function call
        ordered (bool):
        force_serial (bool):
        chunksize (int):
        prog (bool):
        verbose (bool):
        nTasks (int): optional (must be specified if args_list is an iterator)

    Returns:
        generator which yeilds result of applying func to args in args_list

    CommandLine:
        python -m utool.util_parallel --test-generate
        python -m utool.util_parallel --test-generate --verbose

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> #num = 8700  # parallel is slower for smaller numbers
        >>> num = 700  # parallel has an initial (~.1 second startup overhead)
        >>> print('TESTING SERIAL')
        >>> flag_generator0 = ut.generate(ut.is_prime, range(0, num), force_serial=True)
        >>> flag_list0 = list(flag_generator0)
        >>> print('TESTING PARALLEL')
        >>> flag_generator1 = ut.generate(ut.is_prime, range(0, num))
        >>> flag_list1 = list(flag_generator1)
        >>> print('ASSERTING')
        >>> assert flag_list0 == flag_list1

    """
    if nTasks is None:
        nTasks = len(args_list)
    if nTasks == 0:
        if PAR_VERBOSE or verbose:
            print('[util_parallel.generate] submitted 0 tasks')
        return iter([])
    if PAR_VERBOSE or verbose:
        print('[util_parallel.generate] ordered=%r' % ordered)
        print('[util_parallel.generate] force_serial=%r' % force_serial)
    # Check conditions under which we force serial
    force_serial_ = nTasks == 1 or nTasks < MIN_PARALLEL_TASKS or force_serial
    if not force_serial_:
        ensure_pool()
    if force_serial_ or isinstance(__POOL__, int):
        if PAR_VERBOSE or verbose:
            print('[util_parallel.generate] generate_serial')
        return _generate_serial(func, args_list, prog=prog, nTasks=nTasks)
    else:
        if PAR_VERBOSE or verbose:
            print('[util_parallel.generate] generate_parallel')
        return _generate_parallel(func, args_list, ordered=ordered,
                                  chunksize=chunksize, prog=prog,
                                  verbose=verbose, nTasks=nTasks, freq=freq)


def process(func, args_list, args_dict={}, force_serial=__FORCE_SERIAL__,
            nTasks=None):
    """
    Use ut.generate rather than ut.process

    Args:
        func (func):
        args_list (list or iter):
        args_dict (dict):
        force_serial (bool):

    Returns:
        result of parallel map(func, args_list)

    CommandLine:
        python -m utool.util_parallel --test-process

    Example:
        >>> # SLOW_DOCTEST
        >>> import utool as ut
        >>> num = 8700  # parallel is slower for smaller numbers
        >>> flag_generator0 = ut.process(ut.is_prime, zip(range(0, num)), force_serial=True)
        >>> flag_list0 = list(flag_generator0)
        >>> flag_generator1 = ut.process(ut.is_prime, zip(range(0, num)), force_serial=False)
        >>> flag_list1 = list(flag_generator1)
        >>> assert flag_list0 == flag_list1
    """

    ensure_pool()
    if nTasks is None:
        nTasks = len(args_list)
    if __POOL__ == 1 or force_serial:
        if not QUIET:
            print('[util_parallel] executing %d %s tasks in serial' %
                  (nTasks, get_funcname(func)))
        result_list = _process_serial(func, args_list, args_dict, nTasks=nTasks)
    else:
        if not QUIET:
            print('[util_parallel] executing %d %s tasks using %d processes' %
                  (nTasks, get_funcname(func), __POOL__._processes))
        result_list = _process_parallel(func, args_list, args_dict, nTasks=nTasks)
    return result_list


def spawn_background_process(func, *args, **kwargs):
    """
    Run a function in the background
    (like rebuilding some costly data structure)

    References:
        http://stackoverflow.com/questions/2046603/is-it-possible-to-run-function-in-a-subprocess-without-threading-or-writing-a-se
        http://stackoverflow.com/questions/1196074/starting-a-background-process-in-python
        http://stackoverflow.com/questions/15063963/python-is-thread-still-running

    Args:
        func (function):

    CommandLine:
        python -m utool.util_parallel --test-spawn_background_process

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_parallel import *  # NOQA
        >>> import utool as ut
        >>> import time
        >>> from os.path import join
        >>> # build test data
        >>> fname = 'test_bgfunc_output.txt'
        >>> dpath = ut.get_app_resource_dir('utool')
        >>> ut.ensuredir(dpath)
        >>> fpath = join(dpath, fname)
        >>> # ensure file is not around
        >>> sleep_time = 1
        >>> ut.delete(fpath)
        >>> assert not ut.checkpath(fpath, verbose=True)
        >>> def backgrond_func(fpath, sleep_time):
        ...     import utool as ut
        ...     import time
        ...     print('[BG] Background Process has started')
        ...     time.sleep(sleep_time)
        ...     print('[BG] Background Process is writing')
        ...     ut.write_to(fpath, 'background process')
        ...     print('[BG] Background Process has finished')
        ...     #raise AssertionError('test exception')
        >>> # execute function
        >>> func = backgrond_func
        >>> args = (fpath, sleep_time)
        >>> kwargs = {}
        >>> print('[FG] Spawning process')
        >>> threadid = ut.spawn_background_process(func, *args, **kwargs)
        >>> assert threadid.is_alive() is True, 'thread should be active'
        >>> print('[FG] Spawned process. threadid=%r' % (threadid,))
        >>> # background process should not have finished yet
        >>> assert not ut.checkpath(fpath, verbose=True)
        >>> print('[FG] Waiting to check')
        >>> time.sleep(sleep_time + .1)
        >>> print('[FG] Finished waiting')
        >>> # Now the file should be there
        >>> assert ut.checkpath(fpath, verbose=True)
        >>> assert threadid.is_alive() is False, 'process should have died'
    """
    proc_obj = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
    #proc_obj.isAlive = proc_obj.is_alive
    proc_obj.start()
    return proc_obj


def spawn_background_thread(func, *args, **kwargs):
    #threadobj = IMPLEMENTATION_NUM
    thread_obj = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread_obj.start()
    return thread_obj


def _spawn_background_thread0(func, *args, **kwargs):
    thread_id = _thread.start_new_thread(func, args, kwargs)
    return thread_id


if __name__ == '__main__':
    """
    Ignore:
       timing things
       python reset_dbs.py --time-generate
       python reset_dbs.py --time-generate --force-serial
       python reset_dbs.py --time-generate --preinit
       python reset_dbs.py --time-generate --force-serial

    CommandLine:
        python -m utool.util_parallel
        python -m utool.util_parallel --allexamples
    """
    #import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
