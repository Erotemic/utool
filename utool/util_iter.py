from __future__ import absolute_import, division, print_function
import numpy as np
from itertools import chain, cycle, izip
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[iter]')


def ensure_iterable(obj):
    """ <CYTH> """
    if np.iterable(obj):
        return obj
    else:
        return [obj]


def ifilter_items(item_iter, flag_iter):
    """ <CYTH> """
    true_items = (item for (item, flag) in izip(item_iter, flag_iter) if flag)
    return true_items


def ifilterfalse_items(item_iter, flag_iter):
    """ <CYTH> """
    false_items = (item for (item, flag) in izip(item_iter, flag_iter) if not flag)
    return false_items


def ifilter_Nones(iter_):
    """ Removes any nones from the iterable
    <CYTH> """
    return (item for item in iter_ if item is not None)


def isiterable(obj):
    """ <CYTH> """
    return np.iterable(obj) and not isinstance(obj, (str, unicode))


def iflatten(list_):
    """ flatten <CYTH> """
    flat_iter = chain.from_iterable(list_)  # very fast flatten
    return flat_iter


def iflatten_scalars(list_):
    """ <CYTH> """
    [item for item in list_]


def ichunks(list_, chunksize):
    """ Yield successive n-sized chunks from list_.
    <CYTH> """
    for ix in xrange(0, len(list_), chunksize):
        yield list_[ix: ix + chunksize]


def interleave(args):
    """ <CYTH> """
    arg_iters = map(iter, args)
    cycle_iter = cycle(arg_iters)
    for iter_ in cycle_iter:
        yield iter_.next()
