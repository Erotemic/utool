# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import six


def ensure_iterable(obj):
    r"""
    Args:
        obj (scalar or iterable):

    Returns:
        it3erable: obj if it was iterable otherwise [obj]

    CommandLine:
        python -m utool._internal.meta_util_iter --test-ensure_iterable

    Timeit:
        %timeit ut.ensure_iterable([1])
        %timeit ut.ensure_iterable(1)
        %timeit ut.ensure_iterable(np.array(1))
        %timeit ut.ensure_iterable([1])
        %timeit [1]


    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool._internal.meta_util_iter import *  # NOQA
        >>> # build test data
        >>> obj_list = [3, [3], '3', (3,), [3,4,5]]
        >>> # execute function
        >>> result = [ensure_iterable(obj) for obj in obj_list]
        >>> # verify results
        >>> result = str(result)
        >>> print(result)
        [[3], [3], ['3'], (3,), [3, 4, 5]]
    """
    if isiterable(obj):
        return obj
    else:
        return [obj]


def isiterable(obj):
    """
    Returns if the object can be iterated over and is NOT a string
    # TODO: implement isscalar similar to numpy

    Args:
        obj (scalar or iterable):

    Returns:
        bool:

    CommandLine:
        python -m utool._internal.meta_util_iter --test-isiterable

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool._internal.meta_util_iter import *  # NOQA
        >>> # build test data
        >>> obj_list = [3, [3], '3', (3,), [3,4,5]]
        >>> # execute function
        >>> result = [isiterable(obj) for obj in obj_list]
        >>> # verify results
        >>> print(result)
        [False, True, False, True, True]
    """
    try:
        iter(obj)
        return not isinstance(obj, six.string_types)
    except Exception:
        return False
    #return np.iterable(obj) and not isinstance(obj, six.string_types)


def isscalar(obj):
    return not isiterable(obj)


#def isscalar(obj):
if __name__ == '__main__':
    """
    CommandLine:
        python -m utool._internal.meta_util_iter
        python -m utool._internal.meta_util_iter --allexamples
        python -m utool._internal.meta_util_iter --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
