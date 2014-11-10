from __future__ import absolute_import, division, print_function
try:
    import numpy as np
except ImportError as ex:
    pass
import six
from six.moves import zip, range
from itertools import chain, cycle, islice, izip_longest
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[iter]')


def itertwo(iterable):
    iter1 = iter(iterable)
    iter2 = iter(iterable)
    six.next(iter2)
    return zip(iter1, iter2)


def ensure_iterable(obj):
    if np.iterable(obj):
        return obj
    else:
        return [obj]


def isiterable(obj):
    return np.iterable(obj) and not isinstance(obj, six.string_types)


def ifilter_items(item_iter, flag_iter):
    true_items = (item for (item, flag) in zip(item_iter, flag_iter) if flag)
    return true_items


def ifilterfalse_items(item_iter, flag_iter):
    false_items = (item for (item, flag) in zip(item_iter, flag_iter) if not flag)
    return false_items


def ifilter_Nones(iter_):
    """ Removes any nones from the iterable """
    return (item for item in iter_ if item is not None)


def iflatten(list_):
    """ flattens a list iteratively """
    flat_iter = chain.from_iterable(list_)  # very fast flatten
    return flat_iter


def iflatten_scalars(list_):
    [item for item in list_]


def ichunks(iterable, chunksize):
    """
    generates successive n-sized chunks from ``iterable``.

    Args:
        iterable (list): input to iterate over
        chunksize (int): size of sublist to return

    References:
        http://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks

    Timeit:
        >>> import utool as ut
        >>> setup = ut.codeblock('''
                from utool.util_iter import *  # NOQA
                iterable = list(range(100))
                chunksize = 8
                ''')
        >>> stmt_list = [
        ...     'list(ichunks(iterable, chunksize))',
        ...     'list(ichunks_list(iterable, chunksize))'
        ... ]
        >>> (passed, time_list, result_list) = ut.timeit_compare(stmt_list, setup)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> iterable = [1, 2, 3, 4, 5, 6, 7]
        >>> chunksize = 3
        >>> result = ichunks(iterable, chunksize)
        >>> print(list(result))
        [[1, 2, 3], [4, 5, 6], [7]]
    """
    # feed the same iter to izip_longest multiple times, this causes it to
    # consume successive values of the same sequence rather than striped values
    sentinal = object()
    chunks_with_sentinals = izip_longest(*[iter(iterable)] * chunksize, fillvalue=sentinal)
    # Yeild smaller chunks without sentinals
    for chunk in chunks_with_sentinals:
        yield [item for item in chunk if item is not sentinal]


def ichunks_list(list_, chunksize):
    """ input must be a list. SeeAlso ichunks

    References:
        http://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
    """
    return (list_[ix:ix + chunksize] for ix in range(0, len(list_), chunksize))
    #for ix in range(0, len(list_), chunksize):
    #    yield list_[ix: ix + chunksize]


def interleave(args):
    """
    interleave

    Args:
        args (tuple):

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_iter import *  # NOQA
        >>> args = ([1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        >>> result = interleave(args)
        >>> print(list(result))
        [1, 'A', 2, 'B', 3, 'C', 4, 'D', 5, 'E']
    """
    arg_iters = list(map(iter, args))
    cycle_iter = cycle(arg_iters)
    for iter_ in cycle_iter:
        yield six.next(iter_)


def interleave2(*iterables):
    raise NotImplementedError('not sure if this implementation is correct')
    return chain.from_iterable(zip(*iterables))


def interleave3(*args):
    cycle_iter = zip(*args)
    raise NotImplementedError('not sure if this implementation is correct')
    if six.PY2:
        for iter_ in cycle_iter:
            yield iter_.next()
    else:
        for iter_ in cycle_iter:
            yield next(iter_)


def roundrobin(*iterables):
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    raise NotImplementedError('not sure if this implementation is correct')
    # http://stackoverflow.com/questions/11125212/interleaving-lists-in-python
    #sentinel = object()
    #return (x for x in chain(*zip_longest(fillvalue=sentinel, *iterables)) if x is not sentinel)
    pending = len(iterables)
    if six.PY2:
        nexts = cycle(iter(it).next for it in iterables)
    else:
        nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))
