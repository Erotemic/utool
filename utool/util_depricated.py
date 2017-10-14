# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def find_std_inliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)


def cartesian(arrays, out=None):
    r"""
    Generate a cartesian product of input arrays.

    Args:
        arrays (list of array-like): 1-D arrays to form the cartesian product of
        out (ndarray): Outvar which is modified in place if specified

    Returns:
        out (ndarray): cartesian products formed of input arrays
            2-D array of shape (M, len(arrays))

    References:
        gist.github.com/hernamesbarbara/68d073f551565de02ac5

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_depricated import *  # NOQA
        >>> arrays = ([1, 2, 3], [4, 5], [6, 7])
        >>> out = cartesian(arrays)
        >>> result = repr(out.T)
        array([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
               [4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5],
               [6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7]])

    Timeit:
        >>> # DISABLE_DOCTEST
        >>> # Use itertools product instead
        >>> setup = 'import utool as ut\n' + ut.get_doctest_examples(ut.cartesian)[0][0]
        >>> statements = [
        >>>     'cartesian(arrays)',
        >>>     'np.array(list(ut.iprod(*arrays)))',
        >>> ]
        >>> ut.timeit_compare(statements, setup=setup)
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out

#def unpack_items_sorted(dict_, sortfn, reverse=True):
#    """ Unpacks and sorts the dictionary by sortfn
#    """
#    items = dict_.items()
#    sorted_items = sorted(items, key=sortfn, reverse=reverse)
#    sorted_keys, sorted_vals = list(zip(*sorted_items))
#    return sorted_keys, sorted_vals


#def unpack_items_sorted_by_lenvalue(dict_, reverse=True):
#    """ Unpacks and sorts the dictionary by key
#    """
#    def sort_lenvalue(item):
#        return len(item[1])
#    return unpack_items_sorted(dict_, sort_lenvalue)


#def unpack_items_sorted_by_value(dict_, reverse=True):
#    """ Unpacks and sorts the dictionary by key
#    """
#    def sort_value(item):
#        return item[1]
#    return unpack_items_sorted(dict_, sort_value)

#def inbounds(arr, min_, max_):
#    if min_ > 0 and max_ is not None:
#        #if max_ is not None and min
#        islt_max = np.less_equal(arr, max_)
#        isgt_min = np.greater_equal(arr, min_)
#        is_inbounds = np.logical_and(islt_max, isgt_min)
#    elif min_ == 0:
#        is_inbounds = np.less_equal(arr, max_)
#    elif max_ is None:
#        is_inbounds = np.greater_equal(arr, min_)
#    else:
#        assert False
#    return is_inbounds

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m utool.util_depricated
        python -m utool.util_depricated --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
