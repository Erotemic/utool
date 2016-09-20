# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import itertools
import functools
import operator
from six.moves import zip, range, zip_longest, reduce
from itertools import chain, cycle
from utool import util_inject
from utool._internal import meta_util_iter
print, rrr, profile = util_inject.inject2(__name__, '[iter]')

ensure_iterable = meta_util_iter.ensure_iterable
isiterable = meta_util_iter.isiterable
isscalar = meta_util_iter.isscalar


def wrap_iterable(obj):
    """
    Returns:
        wrapped_obj, was_scalar
    """
    was_scalar = not isiterable(obj)
    wrapped_obj = [obj] if was_scalar else obj
    return wrapped_obj, was_scalar


def next_counter(start=0, step=1):
    r"""
    Args:
        start (int): (default = 0)
        step (int): (default = 1)

    Returns:
        func: next_

    CommandLine:
        python -m utool.util_iter --test-next_counter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> start = 1
        >>> step = 1
        >>> next_ = next_counter(start, step)
        >>> result = str([next_(), next_(), next_()])
        >>> print(result)
        [1, 2, 3]
    """
    count_gen = itertools.count(start, step)
    next_ = functools.partial(six.next, count_gen)
    return next_


def evaluate_generator(iter_):
    """
    for evaluating each item in a generator and ignoring output
    """
    for _ in iter_:  # NOQA
        pass
    # TODO: check if faster
    #try:
    #    while True:
    #        six.next(iter_)
    #except StopIteration:
    #    pass


def itake_column(list_, colx):
    """ iterator version of get_list_column """
    if isinstance(colx, list):
        # multi select
        return ([row[colx_] for colx_ in colx] for row in list_)
    else:
        return (row[colx] for row in list_)


iget_list_column = itake_column


def iget_list_column_slice(list_, start=None, stop=None, stride=None):
    """ iterator version of get_list_column """
    if isinstance(start, slice):
        slice_ = start
    else:
        slice_ = slice(start, stop, stride)
    return (row[slice_] for row in list_)


def iter_window(iterable, size=2, step=1, wrap=False):
    r"""
    iterates through iterable with a window size
    generalizeation of itertwo

    Args:
        iterable (iter): an iterable sequence
        size (int): window size (default = 2)
        wrap (bool): wraparound (default = False)

    Returns:
        iter: returns windows in a sequence

    CommandLine:
        python -m utool.util_iter --exec-iter_window

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> iterable = [1, 2, 3, 4, 5, 6]
        >>> size, step, wrap = 3, 1, True
        >>> window_iter = iter_window(iterable, size, step, wrap)
        >>> window_list = list(window_iter)
        >>> result = ('window_list = %r' % (window_list,))
        >>> print(result)
        window_list = [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 1), (6, 1, 2)]

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> iterable = [1, 2, 3, 4, 5, 6]
        >>> size, step, wrap = 3, 2, True
        >>> window_iter = iter_window(iterable, size, step, wrap)
        >>> window_list = list(window_iter)
        >>> result = ('window_list = %r' % (window_list,))
        >>> print(result)
        window_list = [(1, 2, 3), (3, 4, 5), (5, 6, 1)]
    """
    # itertools.tee may be slow, but works on all iterables
    iter_list = itertools.tee(iterable, size)
    if wrap:
        # Secondary iterables need to be cycled for wraparound
        iter_list = [iter_list[0]] + list(map(itertools.cycle, iter_list[1:]))
    # Step each iterator the approprate number of times
    try:
        for count, iter_ in enumerate(iter_list[1:], start=1):
            for _ in range(count):
                six.next(iter_)
    except StopIteration:
        return iter(())
    else:
        _window_iter = zip(*iter_list)
        # Account for the step size
        window_iter = itertools.islice(_window_iter, 0, None, step)
        return window_iter


def itertwo(iterable, wrap=False):
    r"""
    equivalent to iter_window(iterable, 2, 1, wrap)

    Args:
        iterable (iter): an iterable sequence
        wrap (bool): if True, returns with wraparound

    Returns:
        iter: returns edges in a sequence

    CommandLine:
        python -m utool.util_iter --test-itertwo

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> iterable = [1, 2, 3, 4]
        >>> wrap = False
        >>> edges = list(itertwo(iterable, wrap))
        >>> result = ('edges = %r' % (edges,))
        >>> print(result)
        edges = [(1, 2), (2, 3), (3, 4)]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> iterable = [1, 2, 3, 4]
        >>> wrap = True
        >>> edges = list(itertwo(iterable, wrap))
        >>> result = ('edges = %r' % (edges,))
        >>> print(result)
        edges = [(1, 2), (2, 3), (3, 4), (4, 1)]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> import utool as ut
        >>> iterable = iter([1, 2, 3, 4])
        >>> wrap = False
        >>> edge_iter = itertwo(iterable, wrap)
        >>> edges = list(edge_iter)
        >>> result = ('edges = %r' % (edges,))
        >>> ut.assert_eq(len(list(iterable)), 0, 'iterable should have been used up')
        >>> print(result)
        edges = [(1, 2), (2, 3), (3, 4)]
    """
    # itertools.tee may be slow, but works on all iterables
    iter1, iter2 = itertools.tee(iterable, 2)
    if wrap:
        iter2 = itertools.cycle(iter2)
    try:
        six.next(iter2)
    except StopIteration:
        return iter(())
    else:
        return zip(iter1, iter2)


def iter_compress(item_iter, flag_iter):
    """
    iter_compress - like numpy compress

    Args:
        item_iter (list):
        flag_iter (list): of bools

    Returns:
        list: true_items

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> item_iter = [1, 2, 3, 4, 5]
        >>> flag_iter = [False, True, True, False, True]
        >>> true_items = iter_compress(item_iter, flag_iter)
        >>> result = list(true_items)
        >>> print(result)
        [2, 3, 5]
    """
    # TODO: Just use itertools.compress
    true_items = (item for (item, flag) in zip(item_iter, flag_iter) if flag)
    return true_items


ifilter_items = iter_compress


def ifilterfalse_items(item_iter, flag_iter):
    """
    ifilterfalse_items

    Args:
        item_iter (list):
        flag_iter (list): of bools

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> item_iter = [1, 2, 3, 4, 5]
        >>> flag_iter = [False, True, True, False, True]
        >>> false_items = ifilterfalse_items(item_iter, flag_iter)
        >>> result = list(false_items)
        >>> print(result)
        [1, 4]
    """
    false_items = (item for (item, flag) in zip(item_iter, flag_iter) if not flag)
    return false_items


def ifilter_Nones(iter_):
    """ Removes any nones from the iterable """
    return (item for item in iter_ if item is not None)


def iflatten(list_):
    r""" flattens a list iteratively """
    # very fast flatten
    flat_iter = chain.from_iterable(list_)
    return flat_iter


def iter_multichunks(iterable, chunksizes, bordermode=None):
    """
    CommandLine:
        python -m utool.util_iter --test-iter_multichunks

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> import utool as ut
        >>> iterable = list(range(20))
        >>> chunksizes = (3, 2, 3)
        >>> bordermode = 'cycle'
        >>> genresult = iter_multichunks(iterable, chunksizes, bordermode)
        >>> multichunks = list(genresult)
        >>> depthprofile = ut.depth_profile(multichunks)
        >>> assert depthprofile[1:] == chunksizes, 'did not generate chunks correctly'
        >>> result = ut.list_str(map(str, multichunks), nobr=True)
        >>> print(result)
        '[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]]',
        '[[[18, 19, 0], [1, 2, 3]], [[4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15]]]',

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> import utool as ut
        >>> iterable = list(range(7))
        >>> # when chunksizes is len == 1, then equlivalent to ichunks
        >>> chunksizes = (3,)
        >>> bordermode = 'cycle'
        >>> genresult = iter_multichunks(iterable, chunksizes, bordermode)
        >>> multichunks = list(genresult)
        >>> depthprofile = ut.depth_profile(multichunks)
        >>> assert depthprofile[1:] == chunksizes, 'did not generate chunks correctly'
        >>> result = str(multichunks)
        >>> print(result)
        [[0, 1, 2], [3, 4, 5], [6, 0, 1]]
    """
    chunksize = reduce(operator.mul, chunksizes)
    for chunk in ichunks(iterable, chunksize, bordermode=bordermode):
        reshaped_chunk = chunk
        for d in chunksizes[1:][::-1]:
            reshaped_chunk = list(ichunks(reshaped_chunk, d))
        yield reshaped_chunk


def ichunks(iterable, chunksize, bordermode=None):
    r"""
    generates successive n-sized chunks from ``iterable``.

    Args:
        iterable (list): input to iterate over
        chunksize (int): size of sublist to return
        bordermode (str): None, 'cycle', or 'replicate'

    References:
        http://stackoverflow.com/questions/434287/iterate-over-a-list-in-chunks

    SeeAlso:
        util_progress.get_nTotalChunks

    CommandLine:
        python -m utool.util_iter --exec-ichunks --show

    Timeit:
        >>> import utool as ut
        >>> setup = ut.codeblock('''
                from utool.util_iter import *  # NOQA
                iterable = list(range(100))
                chunksize = 8
                ''')
        >>> stmt_list = [
        ...     'list(ichunks(iterable, chunksize))',
        ...     'list(ichunks_noborder(iterable, chunksize))',
        ...     'list(ichunks_list(iterable, chunksize))',
        ... ]
        >>> (passed, times, results) = ut.timeit_compare(stmt_list, setup)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> iterable = [1, 2, 3, 4, 5, 6, 7]
        >>> chunksize = 3
        >>> genresult = ichunks(iterable, chunksize)
        >>> result = list(genresult)
        >>> print(result)
        [[1, 2, 3], [4, 5, 6], [7]]

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> iterable = (1, 2, 3, 4, 5, 6, 7)
        >>> chunksize = 3
        >>> bordermode = 'cycle'
        >>> genresult = ichunks(iterable, chunksize, bordermode)
        >>> result = list(genresult)
        >>> print(result)
        [[1, 2, 3], [4, 5, 6], [7, 1, 2]]

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> iterable = (1, 2, 3, 4, 5, 6, 7)
        >>> chunksize = 3
        >>> bordermode = 'replicate'
        >>> genresult = ichunks(iterable, chunksize, bordermode)
        >>> result = list(genresult)
        >>> print(result)
        [[1, 2, 3], [4, 5, 6], [7, 7, 7]]
    """
    if bordermode is None:
        return ichunks_noborder(iterable, chunksize)
    elif bordermode == 'cycle':
        return ichunks_cycle(iterable, chunksize)
    elif bordermode == 'replicate':
        return ichunks_replicate(iterable, chunksize)
    else:
        raise ValueError('unknown bordermode=%r' % (bordermode,))


def ichunks_noborder(iterable, chunksize):
    # feed the same iter to zip_longest multiple times, this causes it to
    # consume successive values of the same sequence rather than striped values
    sentinal = object()
    copied_iterators = [iter(iterable)] * chunksize
    chunks_with_sentinals = zip_longest(*copied_iterators, fillvalue=sentinal)
    # Yeild smaller chunks without sentinals
    for chunk in chunks_with_sentinals:
        if len(chunk) > 0:
            yield [item for item in chunk if item is not sentinal]


def ichunks_cycle(iterable, chunksize):
    # feed the same iter to zip_longest multiple times, this causes it to
    # consume successive values of the same sequence rather than striped values
    sentinal = object()
    copied_iterators = [iter(iterable)] * chunksize
    chunks_with_sentinals = zip_longest(*copied_iterators, fillvalue=sentinal)
    bordervalues = cycle(iter(iterable))
    # Yeild smaller chunks without sentinals
    for chunk in chunks_with_sentinals:
        if len(chunk) > 0:
            yield [item if item is not sentinal else six.next(bordervalues)
                   for item in chunk]


def ichunks_replicate(iterable, chunksize):
    # feed the same iter to zip_longest multiple times, this causes it to
    # consume successive values of the same sequence rather than striped values
    sentinal = object()
    copied_iterators = [iter(iterable)] * chunksize
    chunks_with_sentinals = zip_longest(*copied_iterators, fillvalue=sentinal)
    # Yeild smaller chunks without sentinals
    for chunk in chunks_with_sentinals:
        if len(chunk) > 0:
            filtered_chunk = [item for item in chunk if item is not sentinal]
            if len(filtered_chunk) == chunksize:
                yield filtered_chunk
            else:
                sizediff = (chunksize - len(filtered_chunk))
                padded_chunk = filtered_chunk + [filtered_chunk[-1]] * sizediff
                yield padded_chunk


def ichunks_list(list_, chunksize):
    """
    input must be a list.

    SeeAlso:
        ichunks

    References:
        http://stackoverflow.com/questions/434287/iterate-over-a-list-in-chunks
    """
    return (list_[ix:ix + chunksize] for ix in range(0, len(list_), chunksize))


def interleave(args):
    r"""
    zip followed by flatten

    Args:
        args (tuple): tuple of lists to interleave

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> import utool as ut
        >>> args = ([1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        >>> genresult = interleave(args)
        >>> result = ut.list_str(list(genresult), nl=False)
        >>> print(result)
        [1, 'A', 2, 'B', 3, 'C', 4, 'D', 5, 'E']
    """
    arg_iters = list(map(iter, args))
    cycle_iter = cycle(arg_iters)
    for iter_ in cycle_iter:
        yield six.next(iter_)


def and_iters(*args):
    return (all(tup) for tup in zip(*args))


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_iter
        python -m utool.util_iter --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
