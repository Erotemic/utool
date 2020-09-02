# -*- coding: utf-8 -*-
"""
Module to executes the same function with different arguments in parallel.
"""
from __future__ import absolute_import, division, print_function
import multiprocessing
from concurrent import futures
# import atexit
#import sys
import signal
import ctypes
import six
import threading
from six.moves import map, range, zip  # NOQA
from utool._internal.meta_util_six import get_funcname
from utool import util_progress
from utool import util_arg
from utool import util_inject
from utool import util_cplat
if six.PY2:
    # import thread as _thread
    import Queue as queue
elif six.PY3:
    # import _thread
    import queue
util_inject.noinject('[parallel]')


SILENT  = util_arg.SILENT

if SILENT:
    def print(msg):
        pass

# Default number of cores to use when doing parallel processing
__NUM_PROCS__ = util_arg.get_argval(
    ('--nprocs', '--num-procs'), type_=int, default=None)

# If true parallelism is disabled
__FORCE_SERIAL__ = util_arg.get_argflag(
    ('--utool-force-serial', '--force-serial', '--serial'))


# FIXME: running tests in IBEIS has errors when this number is low
# Due to the large number of parallel processes running?
__MIN_PARALLEL_TASKS__ = 4
if util_cplat.WIN32:
    __MIN_PARALLEL_TASKS__ = 16


def generate2(func, args_gen, kw_gen=None, ntasks=None, ordered=True,
              force_serial=False, use_pool=False, chunksize=None, nprocs=None,
              progkw={}, nTasks=None, verbose=None, use_futures_thread=False):
    r"""
    Interfaces to either multiprocessing or futures.
    Esentially maps ``args_gen`` onto ``func`` using pool.imap.
    However, args_gen must be a tuple of args that will be unpacked and send to
    the function. Thus, the function can take multiple args. Also specifing
    keyword args is supported.

    Useful for embarrassingly parallel loops. Currently does not work with
    opencv3

    CommandLine:
        python -m utool.util_parallel generate2

    Args:
        func (function):  live python function
        args_gen (?):
        kw_gen (None): (default = None)
        ntasks (None): (default = None)
        ordered (bool): (default = True)
        force_serial (bool): (default = False)
        verbose (bool):  verbosity flag(default = None)

    CommandLine:
        python -m utool.util_parallel generate2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_parallel import *  # NOQA
        >>> from utool.util_parallel import _kw_wrap_worker  # NOQA
        >>> import utool as ut
        >>> args_gen = list(zip(range(10000)))
        >>> kw_gen = [{}] * len(args_gen)
        >>> func = ut.is_prime
        >>> _ = list(generate2(func, args_gen))
        >>> _ = list(generate2(func, args_gen, ordered=False))
        >>> _ = list(generate2(func, args_gen, force_serial=True))
        >>> _ = list(generate2(func, args_gen, use_pool=True))
        >>> _ = list(generate2(func, args_gen, use_futures_thread=True))
        >>> _ = list(generate2(func, args_gen, ordered=False, verbose=False))

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> #num = 8700  # parallel is slower for smaller numbers
        >>> num = 500  # parallel has an initial (~.1 second startup overhead)
        >>> print('TESTING SERIAL')
        >>> func = ut.is_prime
        >>> args_list = list(range(0, num))
        >>> flag_generator0 = ut.generate2(ut.is_prime, zip(range(0, num)), force_serial=True)
        >>> flag_list0 = list(flag_generator0)
        >>> print('TESTING PARALLEL (PROCESS)')
        >>> flag_generator1 = ut.generate2(ut.is_prime, zip(range(0, num)))
        >>> flag_list1 = list(flag_generator1)
        >>> print('TESTING PARALLEL (THREAD)')
        >>> flag_generator2 = ut.generate2(ut.is_prime, zip(range(0, num)), use_futures_thread=True)
        >>> flag_list2 = list(flag_generator2)
        >>> print('ASSERTING')
        >>> assert len(flag_list1) == num
        >>> assert len(flag_list2) == num
        >>> assert flag_list0 == flag_list1
        >>> assert flag_list0 == flag_list2

    Example1:
        >>> # ENABLE_DOCTEST
        >>> # Trying to recreate the freeze seen in IBEIS
        >>> import utool as ut
        >>> print('TESTING SERIAL')
        >>> flag_generator0 = ut.generate2(ut.is_prime, zip(range(0, 1)))
        >>> flag_list0 = list(flag_generator0)
        >>> flag_generator1 = ut.generate2(ut.fibonacci_recursive, zip(range(0, 1)))
        >>> flag_list1 = list(flag_generator1)
        >>> print('TESTING PARALLEL')
        >>> flag_generator2 = ut.generate2(ut.is_prime, zip(range(0, 12)))
        >>> flag_list2 = list(flag_generator2)
        >>> flag_generator3 = ut.generate2(ut.fibonacci_recursive, zip(range(0, 12)))
        >>> flag_list3 = list(flag_generator3)
        >>> print('flag_list0 = %r' % (flag_list0,))
        >>> print('flag_list1 = %r' % (flag_list1,))
        >>> print('flag_list2 = %r' % (flag_list1,))
        >>> print('flag_list3 = %r' % (flag_list1,))

    Example2:
        >>> # DISABLE_DOCTEST
        >>> # UNSTABLE_DOCTEST
        >>> # Trying to recreate the freeze seen in IBEIS
        >>> try:
        >>>     import vtool_ibeis as vt
        >>> except ImportError:
        >>>     import vtool as vt
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
        >>> chips1 = [x for x in ut.generate2(gen_chip, arg_list1)]
        >>> chips2 = [y for y in ut.generate2(gen_chip, arg_list2, force_serial=True)]
        >>> #feats3 = [z for z in ut.generate2(gen_feat_worker, arg_list3)]
        >>> #feats4 = [w for w in ut.generate2(gen_feat_worker, arg_list4)]

    Example3:
        >>> # DISABLE_DOCTEST
        >>> # FAILING_DOCTEST
        >>> # Trying to recreate the freeze seen in IBEIS
        >>> # Extremely weird case: freezes only if dsize > (313, 313) AND __testwarp was called beforehand.
        >>> # otherwise the parallel loop works fine. Could be an opencv 3.0.0-dev issue.
        >>> try:
        >>>     import vtool_ibeis as vt
        >>> except ImportError:
        >>>     import vtool as vt
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
        >>> #new1 = [x for x in ut.generate2(__testwarp, arg_list1)]
        >>> new1 =  __testwarp(arg_list1[0])
        >>> new2 = [y for y in ut.generate2(__testwarp, arg_list2, force_serial=False)]
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
    #    >>> flag_generator1 = list(ut.generate2(ut.is_prime, range(0, num)))
    #    >>> flag_generator1 = list(ut.generate2(ut.is_prime, range(0, num)))
    #    >>> import time
    #    >>> time.sleep(10)
    """
    if verbose is None:
        verbose = 2
    if ntasks is None:
        ntasks = nTasks
    if ntasks is None:
        try:
            ntasks = len(args_gen)
        except TypeError:
            # Cast to a list
            args_gen = list(args_gen)
            ntasks = len(args_gen)
    if ntasks == 1 or ntasks < __MIN_PARALLEL_TASKS__:
        force_serial = True
    if __FORCE_SERIAL__:
        force_serial = __FORCE_SERIAL__
    if ntasks == 0:
        if verbose:
            print('[ut.generate2] submitted 0 tasks')
        raise StopIteration
    if nprocs is None:
        nprocs = min(ntasks, get_default_numprocs())
    if nprocs == 1:
        force_serial = True

    if kw_gen is None:
        kw_gen = [{}] * ntasks
    if isinstance(kw_gen, dict):
        # kw_gen can be a single dict applied to everything
        kw_gen = [kw_gen] * ntasks

    if force_serial:
        for result in _generate_serial2(func, args_gen, kw_gen,
                                        ntasks=ntasks, progkw=progkw,
                                        verbose=verbose):
            yield result
    else:
        if verbose:
            gentype = 'mp' if use_pool else 'futures'
            fmtstr = '[generate2] executing {} {} tasks using {} {} procs'
            print(fmtstr.format(ntasks, get_funcname(func), nprocs, gentype))

        if verbose > 1:
            lbl = '(pargen) %s: ' % (get_funcname(func),)
            progkw_ = dict(freq=None, bs=True, adjust=False, freq_est='absolute')
            progkw_.update(progkw)
            # print('progkw_.update = {!r}'.format(progkw_.update))
            progpart = util_progress.ProgPartial(length=ntasks, lbl=lbl, **progkw_)

        if use_pool:
            # Use multiprocessing
            if chunksize is None:
                chunksize = max(min(4, ntasks), min(8, ntasks // (nprocs ** 2)))

            try:
                pool = multiprocessing.Pool(nprocs)
                if ordered:
                    pmap_func = pool.imap
                else:
                    pmap_func = pool.imap_unordered

                wrapped_arg_gen = zip([func] * len(args_gen), args_gen, kw_gen)
                res_gen = pmap_func(_kw_wrap_worker, wrapped_arg_gen,
                                    chunksize)
                if verbose > 1:
                    res_gen = progpart(res_gen)
                for res in res_gen:
                    yield res
            finally:
                pool.close()
                pool.join()
        else:
            if use_futures_thread:
                executor_cls = futures.ThreadPoolExecutor
            else:
                executor_cls = futures.ProcessPoolExecutor
            # Use futures
            executor = executor_cls(nprocs)
            try:
                fs_list = [executor.submit(func, *a, **k)
                           for a, k in zip(args_gen, kw_gen)]
                fs_gen = fs_list
                if not ordered:
                    fs_gen = futures.as_completed(fs_gen)
                if verbose > 1:
                    fs_gen = progpart(fs_gen)
                for fs in fs_gen:
                    yield fs.result()
            finally:
                executor.shutdown(wait=True)


def _kw_wrap_worker(func_args_kw):
    func, args, kw = func_args_kw
    return func(*args, **kw)


def _generate_serial2(func, args_gen, kw_gen=None, ntasks=None, progkw={},
                      verbose=None, nTasks=None):
    """ internal serial generator  """
    if verbose is None:
        verbose = 2
    if ntasks is None:
        ntasks = nTasks
    if ntasks is None:
        ntasks = len(args_gen)
    if verbose > 0:
        print('[ut._generate_serial2] executing %d %s tasks in serial' %
                (ntasks, get_funcname(func)))

    # kw_gen can be a single dict applied to everything
    if kw_gen is None:
        kw_gen = [{}] * ntasks
    if isinstance(kw_gen, dict):
        kw_gen = [kw_gen] * ntasks

    # Get iterator with or without progress
    if verbose > 1:
        lbl = '(sergen) %s: ' % (get_funcname(func),)
        progkw_ = dict(freq=None, bs=True, adjust=False, freq_est='between')
        progkw_.update(progkw)
        args_gen = util_progress.ProgIter(args_gen, length=ntasks, lbl=lbl,
                                          **progkw_)

    for args, kw in zip(args_gen, kw_gen):
        result = func(*args, **kw)
        yield result


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


def __testwarp(tup):
    # THIS DOES NOT CAUSE A PROBLEM FOR SOME FREAKING REASON
    import cv2
    import numpy as np
    try:
        import vtool_ibeis as vt
    except ImportError:
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
        >>> # DISABLE_DOCTEST
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
        >>> # DISABLE_DOCTEST
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
        >>> # DISABLE_DOCTEST
        >>> from utool.util_parallel import *  # NOQA
        >>> _test_buffered_generator3()
    """
    try:
        import vtool_ibeis as vt
    except ImportError:
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
        gen_ = ut.generate2(func, zip(data), chunksize=buffer_size, quiet=1, verbose=0)
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
        gen_ = ut.generate2(bgfunc, zip(data), chunksize=buffer_size, quiet=1, verbose=0)
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
    try:
        import vtool_ibeis as vt
    except ImportError:
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
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> from utool.util_parallel import *  # NOQA
        >>> from utool.util_parallel import _test_buffered_generator_img  # NOQA
        >>> from utool.util_parallel import _test_buffered_generator_general2  # NOQA
        >>> _test_buffered_generator_img()
    """
    import utool as ut
    args = [ut.grab_test_imgpath(key) for key in ut.util_grabdata.get_valid_test_imgkeys()]
    #import cv2
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
        >>> # DISABLE_DOCTEST
        >>> # UNSTABLE_DOCTEST
        >>> from utool.util_parallel import *  # NOQA
        >>> import utool as ut
        >>> num = 2 ** 14
        >>> func = ut.is_prime
        >>> data = [38873] * num
        >>> data = list(range(num))
        >>> with ut.Timer('serial') as t1:
        ...     result1 = list(map(func, data))
        >>> with ut.Timer('ut.generate2') as t3:
        ...     result3 = list(ut.generate2(func, zip(data), chunksize=2, quiet=1, verbose=0))
        >>> with ut.Timer('ut.buffered_generator') as t2:
        ...     result2 = list(ut.buffered_generator(map(func, data)))
        >>> assert len(result1) == num and len(result2) == num and len(result3) == num
        >>> assert result3 == result2, 'inconsistent results'
        >>> assert result1 == result2, 'inconsistent results'

    Example1:
        >>> # DISABLE_DOCTEST
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
            pool = multiprocessing.Pool(processes=get_default_numprocs(),
                                        initializer=init_worker,
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
