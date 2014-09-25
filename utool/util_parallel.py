"""
There are a lot of fancier things we can do here.
A good thing to do would be to keep similar function calls
and use multiprocessing.Queues for the backend.
This way we can print out progress.
"""
from __future__ import absolute_import, division, print_function
import multiprocessing
import atexit
import sys
import signal
from ._internal.meta_util_six import get_funcname
from .util_progress import progress_func
from .util_time import tic, toc
from . import util_arg
#from .util_cplat import WIN32
from .util_dbg import printex


QUIET   = util_arg.QUIET
VERBOSE = util_arg.VERBOSE
STRICT  = util_arg.STRICT

__POOL__ = None
__TIME__ = '--time' in sys.argv
__SERIAL_FALLBACK__ = '--noserial-fallback' not in sys.argv
__NUM_PROCS__ = util_arg.get_argval('--num-procs', int, default=None)
__FORCE_SERIAL__ = util_arg.get_argflag('--utool-force-serial')


BACKEND = 'multiprocessing'

if BACKEND == 'gevent':
    raise NotImplementedError('gevent cannot run on multiple cpus')
    pass
elif BACKEND == 'zeromq':
    #http://zguide.zeromq.org/py:mtserver
    raise NotImplementedError('no zeromq yet')
    pass
elif BACKEND == 'multiprocessing':
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
    global __POOL__
    if num_procs is None:
        # Get number of cpu cores
        num_procs = get_default_numprocs()
    if not QUIET:
        print('[parallel] initializing pool with %d processes' % num_procs)
    if num_procs == 1:
        print('[parallel] num_procs=1, Will process in serial')
        __POOL__ = 1
        return
    if STRICT:
        #assert __POOL__ is None, 'pool is a singleton. can only initialize once'
        assert multiprocessing.current_process().name, 'can only initialize from main process'
    if __POOL__ is not None:
        print('close pool before reinitializing')
        return
    # Create the pool of processes
    #__POOL__ = multiprocessing.Pool(processes=num_procs, initializer=init_worker, maxtasksperchild=maxtasksperchild)
    __POOL__ = new_pool(num_procs, init_worker, maxtasksperchild)


@atexit.register
def close_pool(terminate=False):
    global __POOL__
    if __POOL__ is not None:
        if not QUIET:
            if terminate:
                print('[parallel] terminating pool')
            else:
                print('[parallel] closing pool')
        if not isinstance(__POOL__, int):
            # Must join after close to avoid runtime errors
            if terminate:
                __POOL__.terminate()
            __POOL__.close()
            __POOL__.join()
        __POOL__ = None


def _process_serial(func, args_list, args_dict={}):
    """"  """
    num_tasks = len(args_list)
    result_list = []
    mark_prog, end_prog = progress_func(max_val=num_tasks,
                                        lbl=get_funcname(func) + ': ')
    mark_prog(0)
    # Execute each task sequentially
    for count, args in enumerate(args_list):
        result = func(*args, **args_dict)
        result_list.append(result)
        mark_prog(count)
    end_prog()
    return result_list


def _process_parallel(func, args_list, args_dict={}):
    """"  """
    # Define progress observers
    num_tasks = len(args_list)
    num_tasks_returned_ptr = [0]
    mark_prog, end_prog = progress_func(max_val=num_tasks,
                                        lbl=get_funcname(func) + ': ')
    def _callback(result):
        mark_prog(num_tasks_returned_ptr[0])
        sys.stdout.flush()
        num_tasks_returned_ptr[0] += 1
    # Send all tasks to be executed asynconously
    apply_results = [__POOL__.apply_async(func, args, args_dict, _callback)
                     for args in args_list]
    # Wait until all tasks have been processed
    while num_tasks_returned_ptr[0] < num_tasks:
        #print('Waiting: ' + str(num_tasks_returned_ptr[0]) + '/' + str(num_tasks))
        pass
    end_prog()
    # Get the results
    result_list = [ap.get() for ap in apply_results]
    return result_list


def _generate_parallel(func, args_list, ordered=True, chunksize=1,
                       prog=True, verbose=True):
    """  """
    prog = prog and verbose
    nTasks = len(args_list)
    if chunksize is None:
        chunksize = max(1, nTasks // (__POOL__._processes ** 2))
    if verbose:
        print('[parallel] executing %d %s tasks using %d processes with chunksize=%r' %
                (nTasks, get_funcname(func), __POOL__._processes, chunksize))
    if prog:
        mark_prog, end_prog = progress_func(max_val=len(args_list), lbl=get_funcname(func) + ': ')
    #assert isinstance(__POOL__, multiprocessing.Pool),\
    #        '%r __POOL__ = %r' % (type(__POOL__), __POOL__,)
    if ordered:
        generator = __POOL__.imap(func, args_list, chunksize)
    else:
        generator = __POOL__.imap_unordered(func, args_list, chunksize)
    try:
        for count, result in enumerate(generator):
            if prog:
                mark_prog(count)
            yield result
    except Exception as ex:
        printex(ex, 'Parallel Generation Failed!', '[utool]')
        print('__SERIAL_FALLBACK__ = %r' % __SERIAL_FALLBACK__)
        if __SERIAL_FALLBACK__:
            for result in _generate_serial(func, args_list, prog=prog, verbose=verbose):
                yield result
        else:
            raise
    if prog:
        end_prog()
    #close_pool()


def _generate_serial(func, args_list, prog=True, verbose=True):
    """  """
    if verbose:
        print('[parallel] executing %d %s tasks in serial' %
                (len(args_list), get_funcname(func)))
    prog = prog and verbose
    if prog:
        mark_prog, end_prog = progress_func(max_val=len(args_list), lbl=get_funcname(func) + ': ')
    for count, args in enumerate(args_list):
        if prog:
            mark_prog(count)
        result = func(args)
        yield result
    if prog:
        end_prog()


def ensure_pool(warn=False):
    try:
        assert __POOL__ is not None, 'must init_pool() first'
    except AssertionError as ex:
        if warn:
            print('(WARNING) AssertionError: ' + str(ex))
        init_pool()


def generate(func, args_list, ordered=True, force_serial=__FORCE_SERIAL__,
             chunksize=1, prog=True, verbose=True):
    """ Returns a generator which asynchronously returns results
     """
    num_tasks = len(args_list)
    if num_tasks == 0:
        if verbose:
            print('[parallel] submitted 0 tasks')
        return []
    if VERBOSE and verbose:
        print('[parallel.generate] ordered=%r' % ordered)
        print('[parallel.generate] force_serial=%r' % force_serial)
    force_serial_ = num_tasks == 1 or force_serial
    if not force_serial_:
        ensure_pool()
    if __TIME__:
        tt = tic(get_funcname(func))
    if force_serial_ or isinstance(__POOL__, int):
        if VERBOSE and verbose:
            print('[parallel.generate] generate_serial')
        return _generate_serial(func, args_list, prog=prog)
    else:
        if VERBOSE and verbose:
            print('[parallel.generate] generate_parallel')
        return _generate_parallel(func, args_list, ordered=ordered,
                                  chunksize=chunksize, prog=prog,
                                  verbose=verbose)
    if __TIME__:
        toc(tt)


def process(func, args_list, args_dict={}, force_serial=__FORCE_SERIAL__):
    """"  """

    ensure_pool()
    if __POOL__ == 1 or force_serial:
        if not QUIET:
            print('[parallel] executing %d %s tasks in serial' %
                  (len(args_list), get_funcname(func)))
        result_list = _process_serial(func, args_list, args_dict)
    else:
        print('[parallel] executing %d %s tasks using %d processes' %
              (len(args_list), get_funcname(func), __POOL__._processes))
        result_list = _process_parallel(func, args_list, args_dict)
    return result_list
