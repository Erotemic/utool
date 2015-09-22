# -*- coding: utf-8 -*-
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
VERBOSE_PARALLEL, VERYVERBOSE_PARALLEL = util_arg.get_module_verbosity_flags('par', 'parallel')
#VERBOSE_PARALLEL = util_arg.VERBOSE or util_arg.get_argflag(('--verbose-par', '--verbpar', '--verbose-parallel', '--verbparallel'))
#VERYVERBOSE_PARALLEL = util_arg.VERYVERBOSE or util_arg.get_argflag(('--veryverbose-par', '--veryverbpar', '--veryverbose-parallel', '--veryverbparallel'))
STRICT  = util_arg.STRICT

if SILENT:
    def print(msg):
        pass

__POOL__ = None
#__EAGER_JOIN__      = util_arg.get_argflag('--eager-join')
__EAGER_JOIN__      = not util_arg.get_argflag('--noclose-pool')

__NUM_PROCS__       = util_arg.get_argval('--num-procs', int, default=None)
__FORCE_SERIAL__    = util_arg.get_argflag(('--utool-force-serial', '--force-serial', '--serial'))
#__FORCE_SERIAL__    = True
__SERIAL_FALLBACK__ = not util_arg.get_argflag('--noserial-fallback')
__TIME_GENERATE__   = VERBOSE_PARALLEL or util_arg.get_argflag('--time-generate')

# Maybe global pooling is not correct?
USE_GLOBAL_POOL = util_arg.get_argflag('--use_global_pool')


# FIXME: running tests in IBEIS has errors when this number is low
# Due to the large number of parallel processes running?
MIN_PARALLEL_TASKS = 4
#MIN_PARALLEL_TASKS = 16
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


def init_pool(num_procs=None, maxtasksperchild=None, quiet=QUIET, **kwargs):
    """ warning this might not be the right hting to do """
    global __POOL__
    if VERBOSE_PARALLEL:
        print('[util_parallel] init_pool()')
    if num_procs is None:
        # Get number of cpu cores
        num_procs = get_default_numprocs()
    if not quiet:
        print('[util_parallel.init_pool] initializing pool with %d processes' % num_procs)
    if num_procs == 1:
        print('[util_parallel.init_pool] num_procs=1, Will process in serial')
        __POOL__ = 1
        return __POOL__
    if STRICT:
        #assert __POOL__ is None, 'pool is a singleton. can only initialize once'
        assert multiprocessing.current_process().name, 'can only initialize from main process'
    if __POOL__ is not None:
        print('[util_parallel.init_pool] close pool before reinitializing')
        return __POOL__
    # Create the pool of processes
    #__POOL__ = multiprocessing.Pool(processes=num_procs, initializer=init_worker, maxtasksperchild=maxtasksperchild)
    if not USE_GLOBAL_POOL:
        raise AssertionError('Global pool initialization is not allowed')
    __POOL__ = new_pool(num_procs, init_worker, maxtasksperchild)
    return __POOL__


@atexit.register
def close_pool(terminate=False, quiet=QUIET):
    global __POOL__

    if VERBOSE_PARALLEL:
        print('[util_parallel] close_pool()')

    if __POOL__ is not None:
        if not quiet:
            if terminate:
                print('[util_parallel] terminating pool')
            else:
                print('[util_parallel] closing pool')
        if not isinstance(__POOL__, int):
            # Must join after close to avoid runtime errors
            if not USE_GLOBAL_POOL:
                raise AssertionError('Global pools are not allowed. Should be impossible to call this')
            if terminate:
                __POOL__.terminate()
            __POOL__.close()
            __POOL__.join()
        __POOL__ = None


def _process_serial(func, args_list, args_dict={}, nTasks=None, quiet=QUIET):
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


def _process_parallel(func, args_list, args_dict={}, nTasks=None, quiet=QUIET):
    """
    Parallel process map

    Use generate instead
    """
    global __POOL__
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
        close_pool(quiet=quiet)
    return result_list


def _generate_parallel(func, args_list, ordered=True, chunksize=None,
                       prog=True, verbose=True, quiet=QUIET, nTasks=None,
                       freq=None):
    """
    Parallel process generator
    """
    global __POOL__
    if USE_GLOBAL_POOL:
        global __POOL__
        pool = __POOL__
    else:
        # Maybe global pools are bad?
        pool = new_pool(num_procs=get_default_numprocs(),
                        init_worker=init_worker,
                        maxtasksperchild=None)
        #pool = new_pool()

    prog = prog and verbose
    if nTasks is None:
        nTasks = len(args_list)
    if chunksize is None:
        chunksize = max(min(4, nTasks), min(8, nTasks // (pool._processes ** 2)))
    if verbose or VERBOSE_PARALLEL:
        prefix = '[util_parallel._generate_parallel]'
        fmtstr = prefix + 'executing %d %s tasks using %d processes with chunksize=%r'
        print(fmtstr % (nTasks, get_funcname(func), pool._processes, chunksize))

    #import utool as ut
    #buffered = ut.get_argflag('--buffered')
    #buffered = False
    #if buffered:
    #    # current tests indicate that normal pool.imap is faster than buffered
    #    # generation
    #    source_gen = (func(args) for args in args_list)
    #    raw_generator = buffered_generator(source_gen)
    #else:
    pmap_func = pool.imap if ordered else pool.imap_unordered
    raw_generator = pmap_func(func, args_list, chunksize)

    # Get iterator with or without progress
    if prog:
        result_generator = util_progress.ProgressIter(
            raw_generator, nTotal=nTasks, lbl=get_funcname(func) + ': ',
            freq=freq, adjust=False)
    else:
        result_generator = raw_generator

    if __TIME_GENERATE__:
        tt = util_time.tic('_generate_parallel func=' + get_funcname(func))
    try:
        # Start generating
        for result in result_generator:
            yield result
        if __EAGER_JOIN__:
            if USE_GLOBAL_POOL:
                close_pool(quiet=quiet)
            else:
                pool.close()
                pool.join()
    except Exception as ex:
        util_dbg.printex(ex, 'Parallel Generation Failed!', '[utool]', tb=True)
        if __EAGER_JOIN__:
            if USE_GLOBAL_POOL:
                close_pool(quiet=quiet)
            else:
                pool.close()
                pool.join()
        print('__SERIAL_FALLBACK__ = %r' % __SERIAL_FALLBACK__)
        if __SERIAL_FALLBACK__:
            print('Trying to handle error by falling back to serial')
            serial_generator = _generate_serial(
                func, args_list, prog=prog, verbose=verbose, nTasks=nTasks,
                freq=freq)
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
        util_progress.ProgressIter(args_list, nTotal=nTasks,
                                   lbl=get_funcname(func) + ': ', freq=freq,
                                   adjust=False)
        if prog else args_list
    )
    if __TIME_GENERATE__:
        tt = util_time.tic('_generate_serial func=' + get_funcname(func))
    for args in args_iter:
        result = func(args)
        yield result
    if __TIME_GENERATE__:
        util_time.toc(tt)


def ensure_pool(warn=False, quiet=QUIET):
    global __POOL__
    try:
        assert __POOL__ is not None, 'must init_pool() first'
    except AssertionError as ex:
        if warn:
            print('(WARNING) AssertionError: ' + str(ex))
        return init_pool(quiet=quiet)


def generate(func, args_list, ordered=True, force_serial=None,
             chunksize=None, prog=True, verbose=True, quiet=QUIET, nTasks=None,
             freq=None):
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
        python -m utool.util_parallel --test-generate:0
        python -m utool.util_parallel --test-generate:0 --use-global-pool
        python -m utool.util_parallel --test-generate:1
        python -m utool.util_parallel --test-generate:2
        python -m utool.util_parallel --test-generate:3
        python -m utool.util_parallel --test-generate --verbose

        python -c "import multiprocessing; print(multiprocessing.__version__)"
        python -c "import cv2; print(cv2.__version__)"

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> #num = 8700  # parallel is slower for smaller numbers
        >>> num = 40000  # parallel has an initial (~.1 second startup overhead)
        >>> print('TESTING SERIAL')
        >>> flag_generator0 = ut.generate(ut.is_prime, range(0, num), force_serial=True, freq=num / 4)
        >>> flag_list0 = list(flag_generator0)
        >>> print('TESTING PARALLEL')
        >>> flag_generator1 = ut.generate(ut.is_prime, range(0, num), freq=num / 10)
        >>> flag_list1 = list(flag_generator1)
        >>> print('ASSERTING')
        >>> assert len(flag_list1) == num
        >>> assert flag_list0 == flag_list1

    Example1:
        >>> # ENABLE_DOCTEST
        >>> # Trying to recreate the freeze seen in IBEIS
        >>> import utool as ut
        >>> print('TESTING SERIAL')
        >>> flag_generator0 = ut.generate(ut.is_prime, range(0, 1))
        >>> flag_list0 = list(flag_generator0)
        >>> flag_generator1 = ut.generate(ut.fibonacci_recursive, range(0, 1))
        >>> flag_list1 = list(flag_generator1)
        >>> print('TESTING PARALLEL')
        >>> flag_generator2 = ut.generate(ut.is_prime, range(0, 12))
        >>> flag_list2 = list(flag_generator2)
        >>> flag_generator3 = ut.generate(ut.fibonacci_recursive, range(0, 12))
        >>> flag_list3 = list(flag_generator3)
        >>> print('flag_list0 = %r' % (flag_list0,))
        >>> print('flag_list1 = %r' % (flag_list1,))
        >>> print('flag_list2 = %r' % (flag_list1,))
        >>> print('flag_list3 = %r' % (flag_list1,))

    Example2:
        >>> # UNSTABLE_DOCTEST
        >>> # Trying to recreate the freeze seen in IBEIS
        >>> import vtool as vt
        >>> #def gen_chip(tup):
        >>> #    import vtool as vt
        >>> #    cfpath, gfpath, bbox, theta, new_size, filter_list = tup
        >>> #    chipBGR = vt.compute_chip(gfpath, bbox, theta, new_size, filter_list)
        >>> #    height, width = chipBGR.shape[0:2]
        >>> #    vt.imwrite(cfpath, chipBGR)
        >>> #    return cfpath, width, height
        >>> import utool as ut
        >>> from ibeis.model.preproc.preproc_chip import gen_chip
        >>> #from ibeis.model.preproc.preproc_feat import gen_feat_worker
        >>> key_list = ['grace.jpg', 'easy1.png', 'ada2.jpg', 'easy3.png',
        >>>             'hard3.png', 'zebra.png', 'patsy.jpg', 'ada.jpg',
        >>>             'carl.jpg', 'lena.png', 'easy2.png']
        >>> img_fpath_list = [ut.grab_test_imgpath(key) for key in key_list]
        >>> arg_list1 = [(ut.augpath(img_fpath, '_testgen'), img_fpath, (0, 0, 100, 100), 0.0, (545, 372), []) for img_fpath in img_fpath_list[0:1]]
        >>> arg_list2 = [(ut.augpath(img_fpath, '_testgen'), img_fpath, (0, 0, 100, 100), 0.0, (545, 372), []) for img_fpath in img_fpath_list]
        >>> #arg_list3 = [(count, fpath, {}) for count, fpath in enumerate(ut.get_list_column(arg_list1, 0))]
        >>> #arg_list4 = [(count, fpath, {}) for count, fpath in enumerate(ut.get_list_column(arg_list2, 0))]
        >>> ut.remove_file_list(ut.get_list_column(arg_list2, 0))
        >>> chips1 = [x for x in ut.generate(gen_chip, arg_list1)]
        >>> chips2 = [y for y in ut.generate(gen_chip, arg_list2, force_serial=True)]
        >>> #feats3 = [z for z in ut.generate(gen_feat_worker, arg_list3)]
        >>> #feats4 = [w for w in ut.generate(gen_feat_worker, arg_list4)]

    Example3:
        >>> # UNSTABLE_DOCTEST
        >>> # Trying to recreate the freeze seen in IBEIS
        >>> # Extremely weird case: freezes only if dsize > (313, 313) AND __testwarp was called beforehand.
        >>> # otherwise the parallel loop works fine. Could be an opencv 3.0.0-dev issue.
        >>> import vtool as vt
        >>> import utool as ut
        >>> from ibeis.model.preproc.preproc_chip import gen_chip
        >>> import cv2
        >>> from utool.util_parallel import __testwarp
        >>> key_list = ['grace.jpg', 'easy1.png', 'ada2.jpg', 'easy3.png',
        >>>             'hard3.png', 'zebra.png', 'patsy.jpg', 'ada.jpg',
        >>>             'carl.jpg', 'lena.png', 'easy2.png']
        >>> img_fpath_list = [ut.grab_test_imgpath(key) for key in key_list]
        >>> arg_list1 = [(vt.imread(fpath),) for fpath in img_fpath_list[0:1]]
        >>> arg_list2 = [(vt.imread(fpath),) for fpath in img_fpath_list]
        >>> #new1 = [x for x in ut.generate(__testwarp, arg_list1)]
        >>> new1 =  __testwarp(arg_list1[0])
        >>> new2 = [y for y in ut.generate(__testwarp, arg_list2, force_serial=False)]
        >>> #print('new2 = %r' % (new2,))

    #Example4:
    #    >>> # Freakin weird. When IBEIS Runs generate it doesn't close the processes
    #    >>> # UNSTABLE_DOCTEST
    #    >>> # python -m utool.util_parallel --test-generate:4
    #    >>> # Trying to see if we can recreate the problem where there are
    #    >>> # defunct python processes
    #    >>> import utool as ut
    #    >>> #num = 8700  # parallel is slower for smaller numbers
    #    >>> num = 70000  # parallel has an initial (~.1 second startup overhead)
    #    >>> print('TESTING PARALLEL')
    #    >>> flag_generator1 = list(ut.generate(ut.is_prime, range(0, num)))
    #    >>> flag_generator1 = list(ut.generate(ut.is_prime, range(0, num)))
    #    >>> import time
    #    >>> time.sleep(10)

    """
    if force_serial is None:
        force_serial = __FORCE_SERIAL__
    if nTasks is None:
        nTasks = len(args_list)
    if nTasks == 0:
        if VERBOSE_PARALLEL or verbose:
            print('[util_parallel.generate] submitted 0 tasks')
        return iter([])
    if VERYVERBOSE_PARALLEL:
        print('[util_parallel.generate] ordered=%r' % ordered)
        print('[util_parallel.generate] force_serial=%r' % force_serial)
    # Check conditions under which we force serial
    force_serial_ = nTasks == 1 or nTasks < MIN_PARALLEL_TASKS or force_serial
    if USE_GLOBAL_POOL:
        if not force_serial_:
            ensure_pool(quiet=quiet)
    if force_serial_ or isinstance(__POOL__, int):
        if VERBOSE_PARALLEL or verbose:
            print('[util_parallel.generate] generate_serial')
        return _generate_serial(func, args_list, prog=prog, nTasks=nTasks, freq=freq)
    else:
        if VERBOSE_PARALLEL or verbose:
            print('[util_parallel.generate] generate_parallel')
        return _generate_parallel(func, args_list, ordered=ordered,
                                  chunksize=chunksize, prog=prog,
                                  verbose=verbose, quiet=quiet, nTasks=nTasks,
                                  freq=freq)


def __testwarp(tup):
    # THIS DOES NOT CAUSE A PROBLEM FOR SOME FREAKING REASON
    import cv2
    import numpy as np
    import vtool as vt
    img = tup[0]
    M = vt.rotation_mat3x3(.1)[0:2].dot(vt.translation_mat3x3(-10, 10))
    #new = cv2.warpAffine(img, M[0:2], (500, 500), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    # ONLY FAILS WHEN OUTPUT SIZE IS LARGE
    #dsize = (314, 314)  # (313, 313) does not cause the error
    dsize = (500, 500)  # (313, 313) does not cause the error
    dst = np.empty(dsize[::-1], dtype=img.dtype)
    #new = cv2.warpAffine(img, M[0:2], dsize)
    print('Warping?')
    new = cv2.warpAffine(img, M[0:2], dsize, dst)
    print(dst is new)
    return new


#class Sentinal(object):
#    """
#    Lightweight object that can be used as a sentinal in iter instead of None
#    Never generate this in a buffered generator
#    """
#
#    def __eq__(self, other):
#        return isinstance(other, self.__class__)
#
#    def __getstate__(self):
#        return {}
#
#    def __setstate__(self, state):
#        pass


def buffered_generator(source_gen, buffer_size=2):
    """
    Generator that runs a slow source generator in a separate process.

    My generate function still seems faster on test cases.
    However, this function is more flexible in its compatability.

    Args:
        source_gen (iterable): slow generator
        buffer_size (int): the maximal number of items to pre-generate
            (length of the buffer) (default = 2)

    References:
        Taken from Sander Dieleman's data augmentation pipeline
        https://github.com/benanne/kaggle-ndsb/blob/11a66cdbddee16c69514b9530a727df0ac6e136f/buffering.py

    CommandLine:
        python -m utool.util_parallel --test-buffered_generator:0
        python -m utool.util_parallel --test-buffered_generator:1

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> from utool.util_parallel import *  # NOQA
        >>> import utool as ut
        >>> num = 2 ** 14
        >>> func = ut.is_prime
        >>> data = [38873] * num
        >>> data = list(range(num))
        >>> with ut.Timer('serial') as t1:
        ...     result1 = list(map(func, data))
        >>> with ut.Timer('buffer') as t2:
        ...     result2 = list(ut.buffered_generator(map(func, data), buffer_size=2))
        >>> with ut.Timer('generate') as t3:
        ...     result3 = list(ut.generate(func, data, chunksize=2, quiet=True, verbose=False))
        >>> assert len(result1) == num and len(result2) == num and len(result3) == num
        >>> assert result3 == result2, 'inconsistent results'
        >>> assert result1 == result2, 'inconsistent results'

    Example1:
        >>> # UNSTABLE_DOCTEST
        >>> # This test confirms that generate seems to just be better
        >>> # Test the case when there is a expensive process between the generator calls
        >>> import utool as ut
        >>> import time
        >>> # Time it takes to compute x
        >>> import timeit
        >>> # 1.61 ms = 0.00161 seconds
        >>> #prime = 38873
        >>> prime = 346373
        >>> #gentime = timeit.timeit('ut.is_prime(' + str(prime) + ')', setup='import utool as ut', number=500) / 1000.0
        >>> with ut.Timer('gentime') as t:
        ...     ut.is_prime(prime)
        >>> gentime = t.ellapsed
        >>> #gentime   = 0.0016368601322174071
        >>> # sleeptime should be greater than the amount of time it takes to generate
        >>> # an xdata
        >>> #sleeptime = 0.01500
        >>> #sleeptime = gentime
        >>> #
        >>> #def sleepfunc(sleeptime=sleeptime):
        >>> #    import time
        >>> #    time.sleep(sleeptime)
        >>> def sleepfunc(prime=prime):
        >>>     import utool as ut
        >>>     ut.is_prime(prime)
        >>>     ut.is_prime(prime)
        >>>     ut.is_prime(prime)
        >>>     ut.is_prime(prime)
        >>> with ut.Timer('gentime') as t:
        ...     sleepfunc()
        >>> sleeptime = t.ellapsed
        >>> target_looptime = 4.0  # run each loop for about 2 seconds max
        >>> #thresh = 3.0
        >>> thresh = 10.
        >>> # compute amount of loops to run
        >>> num_loops = int(target_looptime / (gentime + sleeptime))
        >>> gentime = t.ellapsed
        >>> data = [prime] * num_loops
        >>> func = ut.is_prime
        >>> total_sleeptime = sleeptime * num_loops
        >>> total_gentime = gentime * num_loops
        >>> print('gentime = %r' % (gentime,))
        >>> print('sleeptime = %r' % (sleeptime,))
        >>> print('est time in sleep = %r' % (total_sleeptime,))
        >>> print('est time in gen = %r' % (total_gentime,))
        >>> if target_looptime < thresh:
        >>>     with ut.Timer('serial') as t1:
        >>>         for x in map(func, data):
        >>>             #time.sleep(sleeptime)
        >>>             sleepfunc()
        >>> with ut.Timer('buffer') as t2:
        >>>     for x in ut.buffered_generator(map(func, data), buffer_size=8):
        >>>         sleepfunc()
        >>>         #time.sleep(sleeptime)
        >>> with ut.Timer('generator') as t3:
        >>>     for x in ut.generate(func, data, chunksize=4, quiet=True, verbose=False):
        >>>         sleepfunc()
        >>>         #time.sleep(sleeptime)
        >>> def parallel_efficiency(t, total_sleeptime, total_gentime):
        >>>     return 1 - ((t.ellapsed - total_sleeptime) / total_gentime)
        >>> if target_looptime < thresh:
        >>>     print('efficiency 1 = %.3f%%' % (100 * parallel_efficiency(t1, total_sleeptime, total_gentime),))
        >>> print('efficiency 2 = %.3f%%' % (100 * parallel_efficiency(t2, total_sleeptime, total_gentime),))
        >>> print('efficiency 3 = %.2f%%' % (100 * parallel_efficiency(t3, total_sleeptime, total_gentime),))

    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer_ size is 2!")

    buffer_ = multiprocessing.Queue(maxsize=buffer_size - 1)
    # the effective buffer_ size is one less, because the generation process
    # will generate one extra element and block until there is room in the buffer_.

    # previously None was used as a sentinal, which fails when source_gen genrates None
    # need to make object that it will not be generated by the process
    sentinal = StopIteration  # mildly hacky use of StopIteration exception

    #print('\ncreate sentinal: ' + repr(sentinal))

    def _buffered_generation_process(source_gen, buffer_, sentinal):
        for data in source_gen:
            buffer_.put(data, block=True)
        buffer_.put(sentinal)  # sentinel: signal the end of the iterator
        buffer_.close()  # unfortunately this does not suffice as a signal: if buffer_.get()
        # was called and subsequently the buffer_ is closed, it will block forever.

    process = multiprocessing.Process(target=_buffered_generation_process, args=(source_gen, buffer_, sentinal))
    #__pool__ = ensure_pool(quiet=False)
    #if __pool__ is not None and not isinstance(__pool__, int):
    #    process = __pool__.Process(target=_buffered_generation_process, args=(source_gen, buffer_, sentinal))
    #    if __EAGER_JOIN__:
    #        close_pool(quiet=False)
    #else:
    #    process = multiprocessing.Process(target=_buffered_generation_process, args=(source_gen, buffer_, sentinal))
    process.start()

    for data in iter(buffer_.get, sentinal):
        yield data


def process(func, args_list, args_dict={}, force_serial=None,
            nTasks=None, quiet=QUIET):
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
    if force_serial is None:
        force_serial = __FORCE_SERIAL__

    if USE_GLOBAL_POOL:
        ensure_pool(quiet=quiet)
    if nTasks is None:
        nTasks = len(args_list)
    if __POOL__ == 1 or force_serial:
        if not QUIET:
            print('[util_parallel] executing %d %s tasks in serial' %
                  (nTasks, get_funcname(func)))
        result_list = _process_serial(func, args_list, args_dict, nTasks=nTasks,
                                      quiet=quiet)
    else:
        if not QUIET:
            print('[util_parallel] executing %d %s tasks using %d processes' %
                  (nTasks, get_funcname(func), __POOL__._processes))
        result_list = _process_parallel(func, args_list, args_dict, nTasks=nTasks,
                                        quiet=quiet)
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
