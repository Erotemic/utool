# -*- coding: utf-8 -*-
"""
Module to executes the same function with different arguments in parallel.
"""
from __future__ import absolute_import, division, print_function
import multiprocessing
# import atexit
#import sys
import signal
import ctypes
import six
import threading
from six.moves import map, range, zip  # NOQA
from utool._internal.meta_util_six import get_funcname
from utool import util_progress
from utool import util_time
from utool import util_arg
from utool import util_dbg
from utool import util_inject
from utool import util_cplat
if six.PY2:
    import thread as _thread
    import Queue as queue
elif six.PY3:
    import _thread
    import queue
util_inject.noinject('[parallel]')


FUTURE_ON = False
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
__FORCE_SERIAL__    = util_arg.get_argflag(
    ('--utool-force-serial', '--force-serial', '--serial'))
#__FORCE_SERIAL__    = True
# __SERIAL_FALLBACK__ = not util_arg.get_argflag('--noserial-fallback')
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

    because multiprocessing on pip is:
    Backport of the multiprocessing package to Python 2.4 and 2.5

    ut.editfile(multiprocessing.__file__)
    from multiprocessing.pool import ThreadPool
    """
    def new_pool(num_procs, init_worker, maxtasksperchild):
        if FUTURE_ON:
            raise AssertionError('USE FUTURES')
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


def get_sys_thread_limit():
    import utool as ut
    if ut.LINUX:
        out, err, ret = ut.cmd('ulimit', '-u', verbose=False, quiet=True,
                               shell=True)
    else:
        raise NotImplementedError('')


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
    if FUTURE_ON:
        raise AssertionError('USE FUTURES')
    if VERBOSE_PARALLEL:
        print('[util_parallel] init_pool()')
    if num_procs is None:
        # Get number of cpu cores
        num_procs = get_default_numprocs()
    if not quiet:
        print('[util_parallel.init_pool] initializing pool with %d processes' %
              num_procs)
    if num_procs == 1:
        print('[util_parallel.init_pool] num_procs=1, Will process in serial')
        __POOL__ = 1
        return __POOL__
    if STRICT:
        assert multiprocessing.current_process().name, (
            'can only initialize from main process')
    if __POOL__ is not None:
        print('[util_parallel.init_pool] close pool before reinitializing')
        return __POOL__
    # Create the pool of processes
    #__POOL__ = multiprocessing.Pool(processes=num_procs,
    # initializer=init_worker, maxtasksperchild=maxtasksperchild)
    if not USE_GLOBAL_POOL:
        raise AssertionError('Global pool initialization is not allowed')
    __POOL__ = new_pool(num_procs, init_worker, maxtasksperchild)
    return __POOL__


# @atexit.register
def close_pool(terminate=False, quiet=QUIET):
    global __POOL__
    if FUTURE_ON:
        raise AssertionError('USE FUTURES')

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
                raise AssertionError(
                    'Global pools are no longer allowed. '
                    'Should be impossible to call this')
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
    verbose = not quiet
    lbl = '(serproc) %s: ' % (get_funcname(func),)
    prog_iter = util_progress.ProgressIter(
        args_list, nTotal=nTasks, lbl=lbl, adjust=True, verbose=verbose)
    # Execute each task sequentially
    for args in prog_iter:
        result = func(*args, **args_dict)
        result_list.append(result)
    return result_list


def _process_parallel(func, args_list, args_dict={}, nTasks=None, quiet=QUIET, pool=None):
    """
    Parallel process map

    Use generate instead
    """
    if FUTURE_ON:
        raise AssertionError('USE FUTURES')
    # Define progress observers
    if nTasks is None:
        nTasks = len(args_list)
    verbose = not quiet
    lbl = '(parproc) %s: ' % (get_funcname(func),)
    _prog = util_progress.ProgressIter(
        range(nTasks), nTotal=nTasks, lbl=lbl,
        adjust=True, verbose=verbose)
    _prog_iter = iter(_prog)
    num_tasks_returned_ptr = [0]
    def _callback(result):
        six.next(_prog_iter)
        num_tasks_returned_ptr[0] += 1
    # Send all tasks to be executed asynconously
    apply_results = [pool.apply_async(func, args, args_dict, _callback)
                     for args in args_list]
    # Wait until all tasks have been processed
    while num_tasks_returned_ptr[0] < nTasks:
        #print('Waiting: ' + str(num_tasks_returned_ptr[0]) + '/' + str(nTasks))
        pass
    # Get the results
    result_list = [ap.get() for ap in apply_results]
    if __EAGER_JOIN__:
        if USE_GLOBAL_POOL:
            close_pool(quiet=quiet)
        else:
            pool.close()
            pool.join()
    return result_list


def _generate_parallel(func, args_list, ordered=True, chunksize=None,
                       prog=True, verbose=True, quiet=QUIET, nTasks=None,
                       **kwargs):
    """
    Parallel process generator
    """
    global __POOL__
    if FUTURE_ON:
        raise AssertionError('USE FUTURES')
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
        fmtstr = (prefix +
                  'executing %d %s tasks using %d processes with chunksize=%r')
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
        verbose = not quiet
        lbl = '(pargen) %s: ' % (get_funcname(func),)
        result_generator = util_progress.ProgressIter(
            raw_generator, nTotal=nTasks, lbl=lbl,
            freq=kwargs.get('freq', None),
            backspace=kwargs.get('backspace', True),
            adjust=kwargs.get('adjust', False),
            verbose=verbose
        )

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
        # DONT DO SERIAL FALLBACK IN GENERATOR CAN CAUSE ERRORS
        raise
        # print('__SERIAL_FALLBACK__ = %r' % __SERIAL_FALLBACK__)
        # if __SERIAL_FALLBACK__:
        #     print('Trying to handle error by falling back to serial')
        #     serial_generator = _generate_serial(
        #         func, args_list, prog=prog, verbose=verbose, nTasks=nTasks,
        #         **kwargs)
        #     for result in serial_generator:
        #         yield result
        # else:
        #     raise
    if __TIME_GENERATE__:
        util_time.toc(tt)


def _generate_serial(func, args_list, prog=True, verbose=True, nTasks=None,
                     quiet=QUIET, **kwargs):
    """ internal serial generator  """
    if nTasks is None:
        nTasks = len(args_list)
    if verbose and not quiet:
        print('[util_parallel._generate_serial] executing %d %s tasks in serial' %
                (nTasks, get_funcname(func)))
    prog = prog and verbose and nTasks > 1
    # Get iterator with or without progress
    verbose = verbose or not quiet
    lbl = '(sergen) %s: ' % (get_funcname(func),)
    args_iter = (
        util_progress.ProgressIter(args_list, nTotal=nTasks,
                                   lbl=lbl,
                                   freq=kwargs.get('freq', None),
                                   adjust=kwargs.get('adjust', False),
                                   verbose=verbose)
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
    if FUTURE_ON:
        raise AssertionError('USE FUTURES')
    try:
        assert __POOL__ is not None, 'must init_pool() first'
    except AssertionError as ex:
        if warn:
            print('(WARNING) AssertionError: ' + str(ex))
        return init_pool(quiet=quiet)


def generate(func, args_list, ordered=True, force_serial=None,
             chunksize=None, prog=True, verbose=True, quiet=QUIET, nTasks=None,
             freq=None, **kwargs):
    """
    Provides an interfaces to python's multiprocessing module.
    Esentially maps ``args_list`` onto ``func`` using pool.imap.
    Useful for embarrassingly parallel loops. Currently does not work with
    opencv3

    Args:
        func (function): function to apply each argument to
        args_list (list or iter): sequence of tuples which are args for each
                                  function call
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
        >>> num = 500  # parallel has an initial (~.1 second startup overhead)
        >>> print('TESTING SERIAL')
        >>> func = ut.is_prime
        >>> args_list = list(range(0, num))
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
        >>> from ibeis.algo.preproc.preproc_chip import gen_chip
        >>> #from ibeis.algo.preproc.preproc_feat import gen_feat_worker
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
        >>> # FAILING_DOCTEST
        >>> # Trying to recreate the freeze seen in IBEIS
        >>> # Extremely weird case: freezes only if dsize > (313, 313) AND __testwarp was called beforehand.
        >>> # otherwise the parallel loop works fine. Could be an opencv 3.0.0-dev issue.
        >>> import vtool as vt
        >>> import utool as ut
        >>> from ibeis.algo.preproc.preproc_chip import gen_chip
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
    if __FORCE_SERIAL__:
        force_serial = __FORCE_SERIAL__
    if nTasks is None:
        nTasks = len(args_list)
    if nTasks == 1 or nTasks < MIN_PARALLEL_TASKS:
        force_serial = True
    if nTasks == 0:
        if VERBOSE_PARALLEL or verbose:
            print('[util_parallel.generate] submitted 0 tasks')
        return iter([])
    if VERYVERBOSE_PARALLEL:
        print('[util_parallel.generate] ordered=%r' % ordered)
        print('[util_parallel.generate] force_serial=%r' % force_serial)
    # Check conditions under which we force serial
    if USE_GLOBAL_POOL:
        if not force_serial:
            ensure_pool(quiet=quiet)
    if force_serial or isinstance(__POOL__, int):
        if VERBOSE_PARALLEL or verbose:
            print('[util_parallel.generate] generate_serial')
        return _generate_serial(func, args_list, prog=prog, quiet=quiet,
                                nTasks=nTasks, freq=freq, **kwargs)
    else:
        if VERBOSE_PARALLEL or verbose:
            print('[util_parallel.generate] generate_parallel')
        return _generate_parallel(func, args_list, ordered=ordered,
                                  chunksize=chunksize, prog=prog,
                                  verbose=verbose, quiet=quiet, nTasks=nTasks,
                                  freq=freq, **kwargs)


def futures_generate(worker, args_gen, nTasks=None, freq=10, ordered=True,
                     force_serial=False, quiet=QUIET, verbose=None, prog=True,
                     **kwargs):
    from utool import util_resources
    # Check conditions under which we force serial
    if nTasks == 1 or nTasks < MIN_PARALLEL_TASKS:
        force_serial = True
    if verbose is None:
        verbose = True
    if VERBOSE_PARALLEL:
        verbose = 1
    if VERYVERBOSE_PARALLEL:
        verbose = 2
    if __FORCE_SERIAL__:
        force_serial = __FORCE_SERIAL__
    if nTasks is None:
        nTasks = len(args_gen)
    if nTasks == 0:
        if verbose:
            print('[util_parallel.generate] submitted 0 tasks')
        raise StopIteration
    if verbose > 1:
        print('[util_parallel.generate] ordered=%r' % ordered)
        print('[util_parallel.generate] force_serial=%r' % force_serial)

    nprocs = max(1, util_resources.num_cpus() - 1)

    if force_serial:
        if VERBOSE_PARALLEL or verbose:
            print('[util_parallel.generate] generate_serial')
        for result in _generate_serial(worker, args_gen, prog=prog,
                                       quiet=quiet, nTasks=nTasks, freq=freq,
                                       **kwargs):
            yield result
    else:
        from concurrent import futures
        func = worker
        prog = prog and verbose
        if verbose or VERBOSE_PARALLEL:
            prefix = '[util_parallel._generate_parallel]'
            fmtstr = (prefix +
                      'executing %d %s tasks using %d processes')
            print(fmtstr % (nTasks, get_funcname(func), nprocs))

        if prog:
            lbl = '(pargen) %s: ' % (get_funcname(func),)
            args_gen = util_progress.ProgIter(
                args_gen, nTotal=nTasks, lbl='submitting process',
                freq=kwargs.get('freq', freq), bs=kwargs.get('bs', True),
                adjust=kwargs.get('adjust', False)
            )
        # executor = futures.ProcessPoolExecutor(nprocs)
        # try:
        with futures.ProcessPoolExecutor(nprocs) as executor:
            fs_chunk = [executor.submit(worker, args) for args in args_gen]
            if prog:
                fs_chunk = util_progress.ProgIter(
                    fs_chunk, nTotal=nTasks, lbl=lbl,
                    freq=kwargs.get('freq', None), bs=kwargs.get('bs', True),
                    adjust=kwargs.get('adjust', False)
                )
            for fs in fs_chunk:
                yield fs.result()
        # except Exception:
        #     raise
        # finally:
        #     print('pool shutdown')
        #     executor.shutdown(wait=True)
        print('EXECUTOR SHUTDOWN')


# def futures_generate_(worker, args_gen):
#     import utool as ut
#     from concurrent import futures
#     nprocs = ut.num_unused_cpus(thresh=10) - 1
#     with futures.ProcessPoolExecutor(nprocs) as executor:
#         fs_chunk = [executor.submit(worker, args) for args in args_gen]
#         for fs in fs_chunk:
#             yield fs.result()


def __testwarp(tup):
    # THIS DOES NOT CAUSE A PROBLEM FOR SOME FREAKING REASON
    import cv2
    import numpy as np
    import vtool as vt
    img = tup[0]
    M = vt.rotation_mat3x3(.1)[0:2].dot(vt.translation_mat3x3(-10, 10))
    #new = cv2.warpAffine(img, M[0:2], (500, 500), flags=cv2.INTER_LANCZOS4,
    #                     borderMode=cv2.BORDER_CONSTANT)
    # ONLY FAILS WHEN OUTPUT SIZE IS LARGE
    #dsize = (314, 314)  # (313, 313) does not cause the error
    dsize = (500, 500)  # (313, 313) does not cause the error
    dst = np.empty(dsize[::-1], dtype=img.dtype)
    #new = cv2.warpAffine(img, M[0:2], dsize)
    print('Warping?')
    new = cv2.warpAffine(img, M[0:2], dsize, dst)
    print(dst is new)
    return new


def _test_buffered_generator():
    """
    Test for standard python calls

    CommandLine:
        python -m utool.util_parallel --test-_test_buffered_generator

    Example:
        >>> import utool as ut
        >>> from utool.util_parallel import *  # NOQA
        >>> from utool.util_parallel import _test_buffered_generator  # NOQA
        >>> _test_buffered_generator()
    """
    import utool as ut
    # ---- Func and Sleep Definitions
    args = [346373]  # 38873
    func = ut.is_prime
    def sleepfunc(prime=args[0]):
        #time.sleep(.1)
        import utool as ut
        [ut.is_prime(prime) for _ in range(2)]
    _test_buffered_generator_general(func, args, sleepfunc, 10.0)


def _test_buffered_generator2():
    """
    CommandLine:
        python -m utool.util_parallel --test-_test_buffered_generator2

    Looking at about time_thresh=15s or 350 iterations to get buffered over
    serial.

    Test for numpy calls

    Example:
        >>> from utool.util_parallel import *  # NOQA
        >>> _test_buffered_generator2()
    """
    import numpy as np
    #import utool as ut
    # ---- Func and Sleep Definitions
    from functools import partial
    rng = np.random.RandomState(0)
    args = [rng.rand(256, 256) for _ in range(32)]  # 38873
    func = partial(np.divide, 4.3)
    def sleepfunc(prime=346373):
        #time.sleep(.1)
        import utool as ut
        [ut.is_prime(prime) for _ in range(2)]
    _test_buffered_generator_general(func, args, sleepfunc, 15.0)


def _test_buffered_generator3():
    """
    CommandLine:
        python -m utool.util_parallel --test-_test_buffered_generator3

    This test suggests that a ut.buffered_generator is better for disk IO than
    ut.generate

    Example:
        >>> from utool.util_parallel import *  # NOQA
        >>> _test_buffered_generator3()
    """
    import vtool as vt
    import utool as ut
    # ---- Func and Sleep Definitions
    args = list(map(ut.grab_test_imgpath, ut.get_valid_test_imgkeys()))
    func = vt.imread
    def sleepfunc(prime=346373):
        #time.sleep(.1)
        import utool as ut
        [ut.is_prime(prime) for _ in range(2)]
    _test_buffered_generator_general(func, args, sleepfunc, 4.0)


def _test_buffered_generator_general(func, args, sleepfunc,
                                     target_looptime=1.0,
                                     serial_cheat=1, argmode=False,
                                     buffer_size=2):
    """
    # We are going to generate output of func in the background while sleep
    # func is running in the foreground
    # --- Hyperparams
    target_looptime = 1.5  # maximum time to run all loops
    """
    import utool as ut
    #serial_cheat = 1  # approx division factor to run serial less times
    show_serial = True  # target_looptime < 10.  # 3.0

    with ut.Timer('One* call to func') as t_fgfunc:
        results = [func(arg) for arg in args]
    functime = t_fgfunc.ellapsed / len(args)
    #sleepfunc = ut.is_prime
    with ut.Timer('One* call to sleep func') as t_sleep:
        if argmode:
            [sleepfunc(x) for x in results]
        else:
            [sleepfunc() for x in results]
    sleeptime = t_sleep.ellapsed / len(args)
    # compute amount of loops to run
    _num_loops = round(target_looptime // (functime + sleeptime))
    num_data = int(_num_loops // len(args))
    num_loops =  int(num_data * len(args))
    serial_cheat = min(serial_cheat, num_data)
    data = ut.flatten([args] * num_data)
    est_tsleep = sleeptime * num_loops
    est_tfunc = functime * num_loops
    est_needed_buffers =  sleeptime / functime
    print('Estimated stats' + ut.repr4(ut.dict_subset(locals(), [
        'num_loops',
        'functime', 'sleeptime', 'est_tsleep', 'est_tfunc', 'serial_cheat', 'buffer_size',
        'est_needed_buffers',
    ])))
    if show_serial:
        with ut.Timer('serial') as t1:
            # cheat for serial to make it go faster
            for x in map(func, data[:len(data) // serial_cheat]):
                if argmode:
                    sleepfunc(x)
                else:
                    sleepfunc()
        t_serial = serial_cheat * t1.ellapsed
        print('...toc(\'adjusted_serial\') = %r' % (t_serial))
    with ut.Timer('ut.buffered_generator') as t2:
        gen_ = ut.buffered_generator(map(func, data), buffer_size=buffer_size)
        for x in gen_:
            if argmode:
                sleepfunc(x)
            else:
                sleepfunc()
    with ut.Timer('ut.generate') as t3:
        gen_ = ut.generate(func, data, chunksize=buffer_size, quiet=1, verbose=0)
        for x in gen_:
            if argmode:
                sleepfunc(x)
            else:
                sleepfunc( )
    # Compare theoretical vs practical efficiency
    print('\n Theoretical Results')
    def parallel_efficiency(ellapsed, est_tsleep, est_tfunc):
        return (1 - ((ellapsed - est_tsleep) / est_tfunc)) * 100
    if show_serial:
        print('Theoretical gain (serial) = %.3f%%' % (
            parallel_efficiency(t_serial, est_tsleep, est_tfunc),))
    print('Theoretical gain (ut.buffered_generator) = %.3f%%' % (
        parallel_efficiency(t2.ellapsed, est_tsleep, est_tfunc),))
    print('Theoretical gain (ut.generate) = %.2f%%' % (
        parallel_efficiency(t3.ellapsed, est_tsleep, est_tfunc),))
    if show_serial:
        prac_tfunc = t_serial - est_tsleep
        print('\n Practical Results')
        print('Practical gain (serial) = %.3f%%' % (
            parallel_efficiency(t1.ellapsed, est_tsleep, prac_tfunc),))
        print('Practical gain (ut.buffered_generator) = %.3f%%' % (
            parallel_efficiency(t2.ellapsed, est_tsleep, prac_tfunc),))
        print('Practical gain (ut.generate) = %.2f%%' % (
            parallel_efficiency(t3.ellapsed, est_tsleep, prac_tfunc),))


def _test_buffered_generator_general2(bgfunc, bgargs, fgfunc,
                                      target_looptime=1.0, serial_cheat=1,
                                      buffer_size=2, show_serial=True):
    """
    # We are going to generate output of bgfunc in the background while
    # fgfunc is running in the foreground. fgfunc takes results of bffunc as
    # args.
    # --- Hyperparams
    target_looptime = 1.5  # maximum time to run all loops
    """
    import utool as ut
    with ut.Timer('One* call to bgfunc') as t_bgfunc:
        results = [bgfunc(arg) for arg in bgargs]
    bgfunctime = t_bgfunc.ellapsed / len(bgargs)
    #fgfunc = ut.is_prime
    with ut.Timer('One* call to fgfunc') as t_fgfunc:
        [fgfunc(x) for x in results]
    fgfunctime = t_fgfunc.ellapsed / len(bgargs)
    # compute amount of loops to run
    est_looptime = (bgfunctime + fgfunctime)
    _num_loops = round(target_looptime // est_looptime)
    num_data = int(_num_loops // len(bgargs))
    num_loops =  int(num_data * len(bgargs))
    serial_cheat = min(serial_cheat, num_data)
    data = ut.flatten([bgargs] * num_data)
    est_tfg = fgfunctime * num_loops
    est_tbg = bgfunctime * num_loops
    est_needed_buffers =  fgfunctime / bgfunctime
    print('Estimated stats' + ut.repr4(ut.dict_subset(locals(), [
        'num_loops',
        'bgfunctime', 'fgfunctime', 'est_tfg', 'est_tbg', 'serial_cheat',
        'buffer_size', 'est_needed_buffers',
    ])))
    if show_serial:
        with ut.Timer('serial') as t1:
            # cheat for serial to make it go faster
            for x in map(bgfunc, data[:len(data) // serial_cheat]):
                fgfunc(x)
        t_serial = serial_cheat * t1.ellapsed
        print('...toc(\'adjusted_serial\') = %r' % (t_serial))
    with ut.Timer('ut.buffered_generator') as t2:
        gen_ = ut.buffered_generator(map(bgfunc, data), buffer_size=buffer_size)
        for x in gen_:
            fgfunc(x)
    with ut.Timer('ut.generate') as t3:
        gen_ = ut.generate(bgfunc, data, chunksize=buffer_size, quiet=1, verbose=0)
        for x in gen_:
            fgfunc(x)
    # Compare theoretical vs practical efficiency
    print('\n Theoretical Results')
    def parallel_efficiency(ellapsed, est_tfg, est_tbg):
        return (1 - ((ellapsed - est_tfg) / est_tbg)) * 100
    if show_serial:
        print('Theoretical gain (serial) = %.3f%%' % (
            parallel_efficiency(t_serial, est_tfg, est_tbg),))
    print('Theoretical gain (ut.buffered_generator) = %.3f%%' % (
        parallel_efficiency(t2.ellapsed, est_tfg, est_tbg),))
    print('Theoretical gain (ut.generate) = %.2f%%' % (
        parallel_efficiency(t3.ellapsed, est_tfg, est_tbg),))
    if show_serial:
        prac_tbg = t_serial - est_tfg
        print('\n Practical Results')
        print('Practical gain (serial) = %.3f%%' % (
            parallel_efficiency(t1.ellapsed, est_tfg, prac_tbg),))
        print('Practical gain (ut.buffered_generator) = %.3f%%' % (
            parallel_efficiency(t2.ellapsed, est_tfg, prac_tbg),))
        print('Practical gain (ut.generate) = %.2f%%' % (
            parallel_efficiency(t3.ellapsed, est_tfg, prac_tbg),))


def bgfunc(path):
    # Test for /_test_buffered_generator_img
    #import utool as ut
    import vtool as vt
    for _ in range(1):
        img = vt.imread(path)
    img = img ** 1.1
    #[ut.is_prime(346373) for _ in range(2)]
    return img


def _test_buffered_generator_img():
    """
    Test for buffering image read calls

    CONCLUSIONS:
        Use buffer  when bgtime is bigger, but comparable to fgtime
        Use buffer  when fgtime < bgtime and (fgtime + bgtime) is large
        Use genrate when fgtime > bgtime and (fgtime + bgtime) is large
        Use serial when fgtime is bigger and all parts are comparitively small

        Buffer size should be roughly bgtime / fgtime

        Buffering also has a much more even and regular cpu demand.
        Also demands less cpus (I think)


    CommandLine:
        python -m utool.util_parallel --test-_test_buffered_generator_img

    Example:
        >>> import utool as ut
        >>> from utool.util_parallel import *  # NOQA
        >>> from utool.util_parallel import _test_buffered_generator_img  # NOQA
        >>> from utool.util_parallel import _test_buffered_generator_general2  # NOQA
        >>> _test_buffered_generator_img()
    """
    import utool as ut
    #import vtool as vt
    args = [ut.grab_test_imgpath(key) for key in ut.util_grabdata.get_valid_test_imgkeys()]
    #import cv2
    #import vtool as vt
    #func = cv2.imread
    #bffunc = vt.imread
    def sleepfunc_bufwin(x, niters=10):
        #import cv2
        for z in range(niters):
            # operate on image in some capacity
            x.cumsum()
        for z in range(2):
            x ** 1.1
        return x
    target_looptime = 60.0
    #target_looptime = 20.0
    #target_looptime = 10.0
    #target_looptime = 5.0
    serial_cheat = 1
    _test_buffered_generator_general2(bgfunc, args, sleepfunc_bufwin, target_looptime, serial_cheat, buffer_size=4, show_serial=False)
    #_test_buffered_generator_general2(bgfunc, args, sleepfunc_bufwin, target_looptime, serial_cheat, buffer_size=4, show_serial=True)


def buffered_generator(source_gen, buffer_size=2, use_multiprocessing=False):
    r"""
    Generator that runs a slow source generator in a separate process.

    My generate function still seems faster on test cases.
    However, this function is more flexible in its compatability.

    Args:
        source_gen (iterable): slow generator
        buffer_size (int): the maximal number of items to pre-generate
            (length of the buffer) (default = 2)
        use_multiprocessing (bool): if False uses GIL-hindered threading
            instead of multiprocessing (defualt = False).

    Note:
        use_multiprocessing = True seems to freeze if passed in a generator
        built by six.moves.map.

    References:
        Taken from Sander Dieleman's data augmentation pipeline
        https://github.com/benanne/kaggle-ndsb/blob/11a66cdbddee16c69514b9530a727df0ac6e136f/buffering.py

    CommandLine:
        python -m utool.util_parallel --test-buffered_generator:0
        python -m utool.util_parallel --test-buffered_generator:1

    Ignore:
        >>> #functime = timeit.timeit(
        >>> # 'ut.is_prime(' + str(prime) + ')', setup='import utool as ut',
        >>> # number=500) / 1000.0

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
        >>> with ut.Timer('ut.generate') as t3:
        ...     result3 = list(ut.generate(func, data, chunksize=2, quiet=1, verbose=0))
        >>> with ut.Timer('ut.buffered_generator') as t2:
        ...     result2 = list(ut.buffered_generator(map(func, data)))
        >>> assert len(result1) == num and len(result2) == num and len(result3) == num
        >>> assert result3 == result2, 'inconsistent results'
        >>> assert result1 == result2, 'inconsistent results'

    Example1:
        >>> # VERYSLLOOWWW_DOCTEST
        >>> from utool.util_parallel import _test_buffered_generator
        >>> _test_buffered_generator2()
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer_ size is 2!")

    if use_multiprocessing:
        print('WARNING seems to freeze if passed in a generator')
        #assert False, 'dont use this buffered multiprocessing'
        if False:
            if USE_GLOBAL_POOL:
                pool = __POOL__
            else:
                pool = new_pool(num_procs=get_default_numprocs(),
                                init_worker=init_worker,
                                maxtasksperchild=None)
            Process = pool.Process
        else:
            Process = multiprocessing.Process
        _Queue = multiprocessing.Queue
        target = _buffered_generation_process
    else:
        _Queue = queue.Queue
        Process = KillableThread
        target = _buffered_generation_thread

    # the effective buffer_ size is one less, because the generation process
    # will generate one extra element and block until there is room in the
    # buffer_.
    buffer_ = _Queue(maxsize=buffer_size - 1)

    # previously None was used as a sentinal, which fails when source_gen
    # genrates None need to make object that it will not be generated by the
    # process. A reasonable hack is to use the StopIteration exception instead
    sentinal = StopIteration

    process = Process(
        target=target,
        args=(iter(source_gen), buffer_, sentinal)
    )
    #if not use_multiprocessing:
    process.daemon = True

    process.start()

    while True:
        #output = buffer_.get(timeout=1.0)
        output = buffer_.get()
        if output is sentinal:
            raise StopIteration
        yield output

    #_iter = iter(buffer_.get, sentinal)
    #for data in _iter:
    #    if debug:
    #        print('Yeidling')
    #    yield data


def _buffered_generation_thread(source_gen, buffer_, sentinal):
    """ helper for buffered_generator """
    for data in source_gen:
        buffer_.put(data, block=True)
    # sentinel: signal the end of the iterator
    buffer_.put(sentinal)


def _buffered_generation_process(source_gen, buffer_, sentinal):
    """ helper for buffered_generator """
    for data in source_gen:
        buffer_.put(data, block=True)
    # sentinel: signal the end of the iterator
    buffer_.put(sentinal)
    # unfortunately this does not suffice as a signal: if buffer_.get() was
    # called and subsequently the buffer_ is closed, it will block forever.
    buffer_.close()


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
        >>> flag_generator0 = ut.process(ut.is_prime, list(zip(range(0, num))), force_serial=True)
        >>> flag_list0 = list(flag_generator0)
        >>> flag_generator1 = ut.process(ut.is_prime, list(zip(range(0, num))), force_serial=False)
        >>> flag_list1 = list(flag_generator1)
        >>> assert flag_list0 == flag_list1
    """
    if __FORCE_SERIAL__:
        force_serial = __FORCE_SERIAL__
    if FUTURE_ON:
        raise AssertionError('USE FUTURES')

    if USE_GLOBAL_POOL:
        ensure_pool(quiet=quiet)
    if nTasks is None:
        nTasks = len(args_list)
    if __POOL__ == 1 or force_serial:
        if not quiet:
            print('[util_parallel] executing %d %s tasks in serial' %
                  (nTasks, get_funcname(func)))
        result_list = _process_serial(func, args_list, args_dict, nTasks=nTasks,
                                      quiet=quiet)
    else:
        if __POOL__ is None:
            pool = new_pool(num_procs=get_default_numprocs(),
                            init_worker=init_worker,
                            maxtasksperchild=None)
        else:
            pool = __POOL__
        if not quiet:
            print('[util_parallel] executing %d %s tasks using %d processes' %
                  (nTasks, get_funcname(func), pool._processes))
        result_list = _process_parallel(func, args_list, args_dict, nTasks=nTasks,
                                        quiet=quiet, pool=pool)
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
        >>> # SLOW_DOCTEST
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
    import utool as ut
    func_name = ut.get_funcname(func)
    name = 'mp.Progress-' + func_name
    #proc_obj = multiprocessing.Process(target=func, name=name, args=args, kwargs=kwargs)
    proc_obj = KillableProcess(target=func, name=name, args=args, kwargs=kwargs)
    #proc_obj.daemon = True
    #proc_obj.isAlive = proc_obj.is_alive
    proc_obj.start()
    return proc_obj


class KillableProcess(multiprocessing.Process):
    """
    Simple subclass of multiprocessing.Process
    Gives an additional method to kill all children
    as well as itself. calls this function on delete.

    DEPRICATE, do not kill processes. It is not a good idea.
    It can cause deadlocks.
    """

    #def __del__(self):
    #    self.terminate2()
    #    super(KillableProcess, self).__del__()

    def terminate2(self):
        if self.is_alive():
            #print('[terminate2] Killing process')
            # Kill all children
            import psutil
            os_proc = psutil.Process(pid=self.pid)
            for child in os_proc.children():
                child.terminate()
            self.terminate()
        else:
            #print('[terminate2] Already dead')
            pass

#def _process_error_wraper(queue, func, args, kwargs):
#    pass


#def spawn_background_process2(func, *args, **kwargs):
#    multiprocessing_queue
#    import utool as ut
#    func_name = ut.get_funcname(func)
#    name = 'mp.Progress-' + func_name
#    proc_obj = multiprocessing.Process(target=func, name=name, args=args, kwargs=kwargs)
#    #proc_obj.isAlive = proc_obj.is_alive
#    proc_obj.start()


def _async_raise(tid, excobj):
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(excobj))
    if res == 0:
        raise ValueError('nonexistent thread id')
    elif res > 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError('PyThreadState_SetAsyncExc failed')


class KillableThread(threading.Thread):
    """
    DEPRICATE, do not kill threads. It is not a good idea.
    It can cause deadlocks.

    References:
        http://code.activestate.com/recipes/496960-thread2-killable-threads/
        http://tomerfiliba.com/recipes/Thread2/
    """
    def raise_exc(self, excobj):
        assert self.isAlive(), 'thread must be started'
        for tid, tobj in threading._active.items():
            if tobj is self:
                _async_raise(tid, excobj)
                return
        # the thread was alive when we entered the loop, but was not found
        # in the dict, hence it must have been already terminated. should we raise
        # an exception here? silently ignore?

    def terminate(self):
        # must raise the SystemExit type, instead of a SystemExit() instance
        # due to a bug in PyThreadState_SetAsyncExc
        try:
            self.raise_exc(SystemExit)
        except ValueError:
            pass


def spawn_background_thread(func, *args, **kwargs):
    #threadobj = IMPLEMENTATION_NUM
    thread_obj = KillableThread(target=func, args=args, kwargs=kwargs)
    thread_obj.start()
    return thread_obj


def spawn_background_daemon_thread(func, *args, **kwargs):
    #threadobj = IMPLEMENTATION_NUM
    thread_obj = KillableThread(target=func, args=args, kwargs=kwargs)
    thread_obj.daemon = True
    thread_obj.start()
    return thread_obj


def _spawn_background_thread0(func, *args, **kwargs):
    thread_id = _thread.start_new_thread(func, args, kwargs)
    return thread_id


def futures_map(func, args_list):
    # Requries python2.7
    # pip install futures
    from concurrent import futures
    with futures.ProcessPoolExecutor() as executor:
        result_generator = executor.map(func, args_list)
    return result_generator


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
        python -m utool.util_parallel --allexamples --testslow
        coverage run -m utool.util_parallel --allexamples
        coverage run -m utool.util_parallel --allexamples --testslow
        coverage report html -m utool/util_parallel.py
        coverage html

    """
    #import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool  # NOQA
    utool.doctest_funcs()
