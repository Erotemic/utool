# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import operator
import six
import itertools
import warnings
import functools
from math import floor, ceil
from six.moves import zip, map, zip_longest, range, filter, reduce
from utool import util_iter
from utool import util_inject
from utool import util_str
from utool import util_type
from utool._internal.meta_util_six import get_funcname, set_funcname
print, rrr, profile = util_inject.inject2(__name__)

if util_type.HAVE_NUMPY:
    import numpy as np


# --- List Allocations ---


def emap(func, iter_, **kwargs):
    """
    Eager version of the builtin map function.
    This provides the same functionality as python2 map.

    Note this is inefficient and should only be used when prototyping and
    debugging.

    Extended functionality supports passing common kwargs to all functions
    """
    return [func(arg, **kwargs) for arg in iter_]
    # return list(map(func, iter_))


def estarmap(func, iter_, **kwargs):
    """
    Eager version of it.starmap from itertools

    Note this is inefficient and should only be used when prototyping and
    debugging.
    """
    return [func(*arg, **kwargs) for arg in iter_]


def ezip(*args):
    """
    Eager version of the builtin zip function.
    This provides the same functionality as python2 zip.

    Note this is inefficient and should only be used when prototyping and
    debugging.
    """
    return list(zip(*args))

lmap = emap
lzip = ezip
lstarmap = estarmap


def maplen(iter_):
    return list(map(len, iter_))


def rebase_labels(label_list):
    counter = itertools.count(0)
    orig_to_new = {}
    for label in sorted(label_list):
        if label not in orig_to_new:
            orig_to_new[label] = six.next(counter)
    import utool as ut
    rebased_labels = ut.take(orig_to_new, label_list)
    return rebased_labels


def replace_nones(list_, repl=-1):
    r"""
    Recursively removes Nones in all lists and sublists and replaces them with
    the repl variable

    Args:
        list_ (list):
        repl (obj): replacement value

    Returns:
        list

    CommandLine:
        python -m utool.util_list --test-replace_nones

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> # build test data
        >>> list_ = [None, 0, 1, 2]
        >>> repl = -1
        >>> # execute function
        >>> repl_list = replace_nones(list_, repl)
        >>> # verify results
        >>> result = str(repl_list)
        >>> print(result)
        [-1, 0, 1, 2]

    """
    repl_list = [
        repl if item is None else (
            replace_nones(item, repl) if isinstance(item, list) else item
        )
        for item in list_
    ]
    return repl_list


def recursive_replace(list_, target, repl=-1):
    r"""
    Recursively removes target in all lists and sublists and replaces them with
    the repl variable
    """
    repl_list = [
        recursive_replace(item, target, repl) if isinstance(item, (list, np.ndarray)) else
        (repl if item == target else item)
        for item in list_
    ]
    return repl_list


def list_replace(list_, target, repl):
    r"""
    alias

    Recursively removes target in all lists and sublists and replaces them with
    the repl variable
    """
    return recursive_replace(list_, target, repl)


def alloc_lists(num_alloc):
    """ allocates space for a ``list`` of lists """
    return [[] for _ in range(num_alloc)]


def alloc_nones(num_alloc):
    """ allocates space for a ``list`` of Nones """
    return [None] * num_alloc
    #return [None for _ in range(num_alloc)]


def ensure_list_size(list_, size_):
    """ Allocates more space if needbe.

    Ensures len(``list_``) == ``size_``.

    Args:
        list_ (list): ``list`` to extend
        size_ (int): amount to exent by
    """
    lendiff = (size_) - len(list_)
    if lendiff > 0:
        extension = [None for _ in range(lendiff)]
        list_.extend(extension)


# --- List Searching --- #

def get_list_column_slice(list_, start=None, stop=None, stride=None):
    return list(util_iter.iget_list_column_slice(list_, start, stop, stride))


def take_column(list_, colx):
    r"""
    accepts a list of (indexables) and returns a list of indexables
    can also return a list of list of indexables if colx is a list

    Args:
        list_ (list):  list of lists
        colx (int or list): index or key in each sublist get item

    Returns:
        list: list of selected items

    CommandLine:
        python -m utool.util_list --test-take_column

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [['a', 'b'], ['c', 'd']]
        >>> colx = 0
        >>> result = take_column(list_, colx)
        >>> import utool as ut
        >>> result = ut.repr4(result, nl=False)
        >>> print(result)
        ['a', 'c']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [['a', 'b'], ['c', 'd']]
        >>> colx = [1, 0]
        >>> result = take_column(list_, colx)
        >>> import utool as ut
        >>> result = ut.repr4(result, nl=False)
        >>> print(result)
        [['b', 'a'], ['d', 'c']]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [{'spam': 'EGGS', 'ham': 'SPAM'}, {'spam': 'JAM', 'ham': 'PRAM'}]
        >>> # colx can be a key or list of keys as well
        >>> colx = ['spam']
        >>> result = take_column(list_, colx)
        >>> import utool as ut
        >>> result = ut.repr4(result, nl=False)
        >>> print(result)
        [['EGGS'], ['JAM']]
    """
    #return list(util_iter.iget_list_column(list_, colx))
    return list(util_iter.itake_column(list_, colx))
    #if isinstance(colx, list):
    #    # multi select
    #    return [[row[colx_] for colx_ in colx] for row in list_]
    #else:
    #    return [row[colx] for row in list_]


get_list_column = take_column
#def get_list_row(list_, rowx):
#    return list_[rowx]


def safeapply(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return None


def safelen(list_):
    return safeapply(len, list_)
    #return None if list_ is None else len(list_)


def safe_listget(list_, index, default='?'):
    """ depricate """
    if index >= len(list_):
        return default
    ret = list_[index]
    if ret is None:
        return default
    return ret


def listclip(list_, num, fromback=False):
    r"""
    DEPRICATE: use slices instead

    Args:
        list_ (list):
        num (int):

    Returns:
        sublist:

    CommandLine:
        python -m utool.util_list --test-listclip

    Example1:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> # build test data
        >>> list_ = [1, 2, 3, 4, 5]
        >>> result_list = []
        >>> # execute function
        >>> num = 3
        >>> result_list += [ut.listclip(list_, num)]
        >>> num = 9
        >>> result_list += [ut.listclip(list_, num)]
        >>> # verify results
        >>> result = ut.repr4(result_list)
        >>> print(result)
        [
            [1, 2, 3],
            [1, 2, 3, 4, 5],
        ]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> # build test data
        >>> list_ = [1, 2, 3, 4, 5]
        >>> result_list = []
        >>> # execute function
        >>> num = 3
        >>> result = ut.listclip(list_, num, fromback=True)
        >>> print(result)
        [3, 4, 5]
    """
    if num is None:
        num_ = len(list_)
    else:
        num_ = min(len(list_), num)
    if fromback:
        sublist = list_[-num_:]
    else:
        sublist = list_[:num_]
    return sublist


def find_list_indexes(list_, items):
    """
    Args:
        list_ (list): list of items to be searched
        items (list): list of items to find

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = ['a', 'b', 'c']
        >>> items = ['d', 'c', 'b', 'f']
        >>> index_list = find_list_indexes(list_, items)
        >>> result = ('index_list = %r' % (index_list,))
        >>> print(result)
        index_list = [None, 2, 1, None]
    """
    index_lookup = make_index_lookup(list_)
    index_list = [index_lookup.get(item) for item in items]
    # index_list = [listfind(list_, item) for item in items]
    return index_list


def listfind(list_, tofind):
    r"""
    get the position of item ``tofind`` in ``list_`` if it exists
    otherwise returns None

    Args:
        list_ (?):
        tofind (?):

    Returns:
        int: index of ``tofind`` in ``list_``

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = ['a', 'b', 'c']
        >>> tofind = 'b'
        >>> result = listfind(list_, tofind)
        >>> print(result)
        1
    """
    try:
        return list_.index(tofind)
    except ValueError:
        return None


def search_list(text_list, pattern, flags=0):
    """
    CommandLine:
        python -m utool.util_list --test-search_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> text_list = ['ham', 'jam', 'eggs', 'spam']
        >>> pattern = '.am'
        >>> flags = 0
        >>> (valid_index_list, valid_match_list) = ut.search_list(text_list, pattern, flags)
        >>> result = str(valid_index_list)
        >>> print(result)
        [0, 1, 3]
    """
    import re
    import utool as ut
    match_list = [re.search(pattern, text, flags=flags) for text in text_list]
    valid_index_list = [index for index, match in enumerate(match_list) if match is not None]
    valid_match_list = ut.take(match_list, valid_index_list)
    return valid_index_list, valid_match_list


# --- List Modification --- #

def multi_replace(instr, search_list=[], repl_list=None):
    """
    Does a string replace with a list of search and replacements

    TODO: rename
    """
    repl_list = [''] * len(search_list) if repl_list is None else repl_list
    for ser, repl in zip(search_list, repl_list):
        instr = instr.replace(ser, repl)
    return instr


def flatten(list_):
    r"""
    Args:
        list_ (list): list of lists

    Returns:
        list: flat list

    CommandLine:
        python -m utool.util_list --test-flatten

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> list_ = [['a', 'b'], ['c', 'd']]
        >>> unflat_list2 = flatten(list_)
        >>> result = ut.repr4(unflat_list2, nl=False)
        >>> print(result)
        ['a', 'b', 'c', 'd']
    """
    return list(util_iter.iflatten(list_))


def invertible_flatten1(unflat_list):
    r"""
    Flattens `unflat_list` but remember how to reconstruct the `unflat_list`
    Returns `flat_list` and the `reverse_list` with indexes into the
    `flat_list`

    Args:
        unflat_list (list): list of nested lists that we will flatten.

    Returns:
        tuple : (flat_list, reverse_list)

    CommandLine:
        python -m utool.util_list --exec-invertible_flatten1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> unflat_list = [[1, 2, 3], [4, 5], [6, 6]]
        >>> flat_list, reverse_list = invertible_flatten1(unflat_list)
        >>> result = ('flat_list = %s\n' % (ut.repr2(flat_list),))
        >>> result += ('reverse_list = %s' % (ut.repr2(reverse_list),))
        >>> print(result)
        flat_list = [1, 2, 3, 4, 5, 6, 6]
        reverse_list = [[0, 1, 2], [3, 4], [5, 6]]
    """
    nextnum = functools.partial(six.next, itertools.count(0))
    # Build an unflat list of flat indexes
    reverse_list = [[nextnum() for _ in tup] for tup in unflat_list]
    flat_list = flatten(unflat_list)
    return flat_list, reverse_list


def unflatten1(flat_list, reverse_list):
    """ Rebuilds unflat list from invertible_flatten1

    Args:
        flat_list (list): the flattened list
        reverse_list (list): the list which undoes flattenting

    Returns:
        unflat_list2: original nested list


    SeeAlso:
        invertible_flatten1
        invertible_flatten2
        unflatten2

    """
    unflat_list2 = [[flat_list[index] for index in tup]
                    for tup in reverse_list]
    return unflat_list2


def accumulate(iterator):
    """
    Notice:
        use itertools.accumulate in python > 3.2
    """
    total = 0
    for item in iterator:
        total += item
        yield total


def total_flatten(unflat_list):
    """
    unflat_list = [1, 2, [3, 4], [5, [9]]]
    Args:
        unflat_list (list):

    Returns:
        list: flat_list

    CommandLine:
        python -m utool.util_list --exec-total_flatten --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> unflat_list = [[[1, 2, 3], 4, 5], 9, [2, 3], [1, [2, 3, 4]], 1, 2, 3]
        >>> flat_list = total_flatten(unflat_list)
        >>> result = ('flat_list = %s' % (ut.repr2(flat_list),))
        >>> print(result)
    """
    import utool as ut
    next_list = unflat_list
    scalar_flags = [not ut.isiterable(item) for item in next_list]
    while not all(scalar_flags):
        unflatenized = [[item] if flag else item for flag, item in zip(scalar_flags, next_list)]
        flatter_list = ut.flatten(unflatenized)
        next_list = flatter_list
        scalar_flags = [not ut.isiterable(item) for item in next_list]
    flat_list = next_list
    return flat_list


def invertible_total_flatten(unflat_list):
    r"""
    Args:
        unflat_list (list):

    Returns:
        tuple: (flat_list, invert_levels)

    CommandLine:
        python -m utool.util_list --exec-invertible_total_flatten --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> unflat_list = [0, [[1, 2, 3], 4, 5], 9, [2, 3], [1, [2, 3, 4]], 1, 2, 3]
        >>> print('unflat_list = %r' % (unflat_list,))
        >>> (flat_list, invert_levels) = invertible_total_flatten(unflat_list)
        >>> print('flat_list = %r' % (flat_list,))
        >>> unflat_list2 = total_unflatten(flat_list, invert_levels)
        >>> print('unflat_list2 = %r' % (unflat_list2,))
        >>> assert unflat_list2 == unflat_list
        >>> assert ut.depth_profile(flat_list) == 16

    """
    import utool as ut
    next_list = unflat_list
    scalar_flags = [not ut.isiterable(item) for item in next_list]
    invert_stack = []
    # print('unflat_list = %r' % (unflat_list,))
    while not all(scalar_flags):
        unflattenized = [[item] if flag else item
                         for flag, item in zip(scalar_flags, next_list)]
        flatter_list, invert_part = ut.invertible_flatten1(unflattenized)
        # print('flatter_list = %r' % (flatter_list,))
        for idx in ut.where(scalar_flags):
            invert_part[idx] = invert_part[idx][0]
        invert_stack.append(invert_part)
        next_list = flatter_list
        scalar_flags = [not ut.isiterable(item) for item in next_list]
    # invert_part = [None] * len(scalar_flags)
    # invert_stack.append(invert_part)
    invert_levels = invert_stack[::-1]
    flat_list = next_list
    return flat_list, invert_levels


def total_unflatten(flat_list, invert_levels):
    import utool as ut
    less_flat_list = flat_list
    # print('less_flat_list = %r' % (less_flat_list,))
    for lx, level in enumerate(invert_levels):
        needs_unflatten = [ut.isiterable(x) for x in level]
        is_alreadyflat = ut.not_list(needs_unflatten)
        needs_unflatxs = ut.where(needs_unflatten)
        already_flatxs = ut.where(is_alreadyflat)
        invertinfo = ut.compress(level, needs_unflatten)
        unflat_part = ut.unflatten1(less_flat_list, invertinfo)

        flat_sortx = ut.take(level, already_flatxs)
        flat_part = ut.take(less_flat_list, flat_sortx)
        maxsize = len(already_flatxs) + len(needs_unflatxs) - 1
        # (len(invertinfo) + sum(is_alreadyflat) + 1)
        groups = [flat_part, unflat_part]
        groupxs = [already_flatxs, needs_unflatxs]
        less_flat_list_ = ut.ungroup(groups, groupxs, maxsize)
        less_flat_list = less_flat_list_
    unflat_list = less_flat_list
    return unflat_list


def invertible_flatten2(unflat_list):
    """
    An alternative to invertible_flatten1 which uses cumsum

    Flattens ``list`` but remember how to reconstruct the unflat ``list``
    Returns flat ``list`` and the unflat ``list`` with indexes into the flat
    ``list``

    Args:
        unflat_list (list):

    Returns:
        tuple: flat_list, cumlen_list

    SeeAlso:
        invertible_flatten1
        unflatten1
        unflatten2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool
        >>> utool.util_list
        >>> unflat_list = [[5], [2, 3, 12, 3, 3], [9], [13, 3], [5]]
        >>> flat_list, cumlen_list = invertible_flatten2(unflat_list)
        >>> unflat_list2 = unflatten2(flat_list, cumlen_list)
        >>> assert unflat_list2 == unflat_list
        >>> result = ((flat_list, cumlen_list))
        >>> print(result)
        ([5, 2, 3, 12, 3, 3, 9, 13, 3, 5], [1, 6, 7, 9, 10])

    TODO: This flatten is faster fix it to be used everywhere

    Timeit:
        unflat_list = [[random.random() for _ in range(int(random.random() * 1000))] for __ in range(200)]
        unflat_arrs = list(map(np.array, unflat_list))

        %timeit invertible_flatten2(unflat_list)
        %timeit invertible_flatten2_numpy(unflat_list)
        %timeit invertible_flatten2_numpy(unflat_arrs)

    Timeits:
        import utool
        unflat_list = aids_list1
        flat_aids1, reverse_list = utool.invertible_flatten1(unflat_list)
        flat_aids2, cumlen_list = utool.invertible_flatten2(unflat_list)
        unflat_list1 = utool.unflatten1(flat_aids1, reverse_list)
        unflat_list2 = utool.unflatten2(flat_aids2, cumlen_list)
        assert list(map(list, unflat_list1)) == unflat_list2
        print(utool.get_object_size_str(unflat_list,  'unflat_list  '))
        print(utool.get_object_size_str(flat_aids1,   'flat_aids1   '))
        print(utool.get_object_size_str(flat_aids2,   'flat_aids2   '))
        print(utool.get_object_size_str(reverse_list, 'reverse_list '))
        print(utool.get_object_size_str(cumlen_list,  'cumlen_list  '))
        print(utool.get_object_size_str(unflat_list1, 'unflat_list1 '))
        print(utool.get_object_size_str(unflat_list2, 'unflat_list2 '))
        print('Timings 1:)
        %timeit utool.invertible_flatten1(unflat_list)
        %timeit utool.unflatten1(flat_aids1, reverse_list)
        print('Timings 2:)
        %timeit utool.invertible_flatten2(unflat_list)
        %timeit utool.unflatten2(flat_aids2, cumlen_list)
    """
    sublen_list = list(map(len, unflat_list))
    if not util_type.HAVE_NUMPY:
        cumlen_list = np.cumsum(sublen_list)
        # Build an unflat list of flat indexes
    else:
        cumlen_list = list(accumulate(sublen_list))
    flat_list = flatten(unflat_list)
    return flat_list, cumlen_list


def invertible_flatten2_numpy(unflat_arrs, axis=0):
    """ more numpy version

    TODO: move to vtool

    Args:
        unflat_arrs (list):  list of ndarrays

    Returns:
        tuple: (flat_list, cumlen_list)

    CommandLine:
        python -m utool.util_list --test-invertible_flatten2_numpy

    Ignore:
        >>> # ENABLE_DOCTET
        >>> from utool.util_list import *  # NOQA
        >>> unflat_arrs = [np.array([1, 2, 1]), np.array([5, 9]), np.array([4])]
        >>> (flat_list, cumlen_list) = invertible_flatten2_numpy(unflat_arrs)
        >>> result = str((flat_list, cumlen_list))
        >>> print(result)
        (array([1, 2, 1, 5, 9, 4]), array([3, 5, 6]))
    """
    cumlen_list = np.cumsum([arr.shape[axis] for arr in unflat_arrs])
    flat_list = np.concatenate(unflat_arrs, axis=axis)
    return flat_list, cumlen_list


def unflatten2(flat_list, cumlen_list):
    """ Rebuilds unflat list from invertible_flatten1

    Args:
        flat_list (list): the flattened list
        cumlen_list (list): the list which undoes flattenting

    Returns:
        unflat_list2: original nested list

    SeeAlso:
        invertible_flatten1
        invertible_flatten2
        unflatten2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool
        >>> utool.util_list
        >>> flat_list = [5, 2, 3, 12, 3, 3, 9, 13, 3, 5]
        >>> cumlen_list = [ 1,  6,  7,  9, 10]
        >>> unflat_list2 = unflatten2(flat_list, cumlen_list)
        >>> result = (unflat_list2)
        >>> print(result)
        [[5], [2, 3, 12, 3, 3], [9], [13, 3], [5]]
    """
    unflat_list2 = [flat_list[low:high] for low, high in
                    zip(itertools.chain([0], cumlen_list), cumlen_list)]
    return unflat_list2


def unflat_unique_rowid_map(func, unflat_rowids, **kwargs):
    """
    performs only one call to the underlying func with unique rowids the func
    must be some lookup function

    TODO: move this to a better place.

    CommandLine:
        python -m utool.util_list --test-unflat_unique_rowid_map:0
        python -m utool.util_list --test-unflat_unique_rowid_map:1

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> kwargs = {}
        >>> unflat_rowids = [[1, 2, 3], [2, 5], [1], []]
        >>> num_calls0 = [0]
        >>> num_input0 = [0]
        >>> def func0(rowids, num_calls0=num_calls0, num_input0=num_input0):
        ...     num_calls0[0] += 1
        ...     num_input0[0] += len(rowids)
        ...     return [rowid + 10 for rowid in rowids]
        >>> func = func0
        >>> unflat_vals = unflat_unique_rowid_map(func, unflat_rowids, **kwargs)
        >>> result = [arr.tolist() for arr in unflat_vals]
        >>> print(result)
        >>> ut.assert_eq(num_calls0[0], 1)
        >>> ut.assert_eq(num_input0[0], 4)
        [[11, 12, 13], [12, 15], [11], []]

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> import numpy as np
        >>> kwargs = {}
        >>> unflat_rowids = [[1, 2, 3], [2, 5], [1], []]
        >>> num_calls1 = [0]
        >>> num_input1 = [0]
        >>> def func1(rowids, num_calls1=num_calls1, num_input1=num_input1, np=np):
        ...     num_calls1[0] += 1
        ...     num_input1[0] += len(rowids)
        ...     return [np.array([rowid + 10, rowid, 3]) for rowid in rowids]
        >>> func = func1
        >>> unflat_vals = unflat_unique_rowid_map(func, unflat_rowids, **kwargs)
        >>> result = [arr.tolist() for arr in unflat_vals]
        >>> print(result)
        >>> ut.assert_eq(num_calls1[0], 1)
        >>> ut.assert_eq(num_input1[0], 4)
        [[[11, 1, 3], [12, 2, 3], [13, 3, 3]], [[12, 2, 3], [15, 5, 3]], [[11, 1, 3]], []]

    """
    import utool as ut
    # First flatten the list, and remember the original dimensions
    flat_rowids, reverse_list = ut.invertible_flatten2(unflat_rowids)
    # Then make the input unique
    flat_rowids_arr = np.array(flat_rowids)
    unique_flat_rowids, inverse_unique = np.unique(flat_rowids_arr, return_inverse=True)
    # Then preform the lookup / implicit mapping
    unique_flat_vals = func(unique_flat_rowids, **kwargs)
    # Then broadcast unique values back to original flat positions
    flat_vals_ = np.array(unique_flat_vals)[inverse_unique]
    #flat_vals_ = np.array(unique_flat_vals).take(inverse_unique, axis=0)
    output_shape = tuple(list(flat_rowids_arr.shape) + list(flat_vals_.shape[1:]))
    flat_vals = np.array(flat_vals_).reshape(output_shape)
    # Then _unflatten the results to the original input dimensions
    unflat_vals = ut.unflatten2(flat_vals, reverse_list)
    return unflat_vals


def unpack_iterables(list_):
    import utool as ut
    if ut.isiterable(list_):
        return [new_item for item in list_ for new_item in unpack_iterables(item)]
    else:
        return [list_]


def safe_slice(list_, *args):
    """ safe_slice(list_, [start], stop, [end], [step])
        Slices list and truncates if out of bounds
    """
    if len(args) == 3:
        start = args[0]
        stop  = args[1]
        step  = args[2]
    else:
        step = 1
        if len(args) == 2:
            start = args[0]
            stop  = args[1]
        else:
            start = 0
            stop = args[0]
    len_ = len(list_)
    if stop > len_:
        stop = len_
    return list_[slice(start, stop, step)]


# --- List Queries --- #


def allsame(list_, strict=True):
    """
    checks to see if list is equal everywhere

    Args:
        list_ (list):

    Returns:
        True if all items in the list are equal
    """
    if len(list_) == 0:
        return True
    first_item = list_[0]
    return list_all_eq_to(list_, first_item, strict)


def list_all_eq_to(list_, val, strict=True):
    """
    checks to see if list is equal everywhere to a value

    Args:
        list_ (list):
        val : value to check against

    Returns:
        True if all items in the list are equal to val
    """
    if util_type.HAVE_NUMPY and isinstance(val, np.ndarray):
        return all([np.all(item == val) for item in list_])
    try:
        # FUTURE WARNING
        # FutureWarning: comparison to `None` will result in an elementwise object comparison in the future.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            flags = [item == val for item in list_]
            return all([np.all(flag) if hasattr(flag, '__array__') else flag
                        for flag in flags])
        #return all([item == val for item in list_])
    except ValueError:
        if not strict:
            return all([repr(item) == repr(val) for item in list_])
        else:
            raise


def get_dirty_items(item_list, flag_list):
    """
    Returns each item in item_list where not flag in flag_list

    Args:
        item_list (list):
        flag_list (list):

    Returns:
        dirty_items
    """
    assert len(item_list) == len(flag_list)
    dirty_items = [item for (item, flag) in
                   zip(item_list, flag_list)
                   if not flag]
    #print('num_dirty_items = %r' % len(dirty_items))
    #print('item_list = %r' % (item_list,))
    #print('flag_list = %r' % (flag_list,))
    return dirty_items


def compress(item_list, flag_list):
    """
    like np.compress but for lists

    Returns items in item list where the corresponding item in flag list is
    True

    Args:
        item_list (list): list of items to mask
        flag_list (list): list of booleans used as a mask

    Returns:
        list : filtered_items - masked items
    """
    assert len(item_list) == len(flag_list), (
        'lists should correspond. len(item_list)=%r len(flag_list)=%r' %
        (len(item_list), len(flag_list)))
    filtered_items = list(util_iter.iter_compress(item_list, flag_list))
    return filtered_items


def zipflat(*args):
    return [flatten(tup) for tup in zip(*args)]


def ziptake(items_list, indexes_list):
    """
    SeeAlso:
        vt.ziptake
    """
    return [take(list_, index_list)
            for list_, index_list in zip(items_list, indexes_list)]


def zipcompress(items_list, flags_list):
    """
    SeeAlso:
        vt.zipcompress
    """
    return [compress(list_, flags)
            for list_, flags in zip(items_list, flags_list)]


def list_zipflatten(*items_lists):
    return [flatten(items) for items in zip(*items_lists)]


def list_compresstake(items_list, flags_list):
    return [compress(list_, flags) for list_, flags in zip(items_list, flags_list)]


def filter_items(item_list, flag_list):
    """
    Returns items in item list where the corresponding item in flag list is True

    Args:
        item_list (list):
        flag_list (list):

    Returns:
        filtered_items

    SeeAlso:
        util_iter.iter_compress
    """
    return compress(item_list, flag_list)


def filterfalse_items(item_list, flag_list):
    """
    Returns items in item list where the corresponding item in flag list is true

    Args:
        item_list (list): list of items
        flag_list (list): list of truthy values

    Returns:
        filtered_items : items where the corresponding flag was truthy

    SeeAlso:
        util_iter.ifilterfalse_items
    """
    assert len(item_list) == len(flag_list)
    filtered_items = list(util_iter.ifilterfalse_items(item_list, flag_list))
    return filtered_items


def filter_Nones(item_list):
    """
    Removes any nones from the list

    Args:
        item_list (list):

    Returns:
        sublist which does not contain Nones
    """
    return list(util_iter.ifilter_Nones(item_list))


# --- List combinations --- #


def isect(list1, list2):
    r"""
    returns list1 elements that are also in list2. preserves order of list1

    intersect_ordered

    Args:
        list1 (list):
        list2 (list):

    Returns:
        list: new_list

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list1 = ['featweight_rowid', 'feature_rowid', 'config_rowid', 'featweight_forground_weight']
        >>> list2 = [u'featweight_rowid']
        >>> result = intersect_ordered(list1, list2)
        >>> print(result)
        ['featweight_rowid']

    Timeit:
        def timeit_func(func, *args):
            niter = 10
            times = []
            for count in range(niter):
                with ut.Timer(verbose=False) as t:
                    _ = func(*args)
                times.append(t.ellapsed)
            return sum(times) / niter

        grid = {
            'size1': [1000, 5000, 10000, 50000],
            'size2': [1000, 5000, 10000, 50000],
            #'overlap': [0, 1],
        }
        data = []
        for kw in ut.all_dict_combinations(grid):
            pool = np.arange(kw['size1'] * 2)
            size2 = size1 = kw['size1']
            size2 = kw['size2']
            list1 = (np.random.rand(size1) * size1).astype(np.int32).tolist()
            list1 = ut.random_sample(pool, size1).tolist()
            list2 = ut.random_sample(pool, size2).tolist()
            list1 = set(list1)
            list2 = set(list2)
            kw['ut'] = timeit_func(ut.isect, list1, list2)
            #kw['np1'] = timeit_func(np.intersect1d, list1, list2)
            #kw['py1'] = timeit_func(lambda a, b: set.intersection(set(a), set(b)), list1, list2)
            kw['py2'] = timeit_func(lambda a, b: sorted(set.intersection(set(a), set(b))), list1, list2)
            data.append(kw)

        import pandas as pd
        pd.options.display.max_rows = 1000
        pd.options.display.width = 1000
        df = pd.DataFrame.from_dict(data)
        data_keys = list(grid.keys())
        other_keys = ut.setdiff(df.columns, data_keys)
        df = df.reindex_axis(data_keys + other_keys, axis=1)
        df['abs_change'] = df['ut'] - df['py2']
        df['pct_change'] = df['abs_change'] / df['ut'] * 100
        #print(df.sort('abs_change', ascending=False))

        print(str(df).split('\n')[0])
        for row in df.values:
            argmin = row[len(data_keys):len(data_keys) + len(other_keys)].argmin() + len(data_keys)
            print('    ' + ', '.join([
            '%6d' % (r) if x < len(data_keys) else (
                ut.color_text('%8.6f' % (r,), 'blue')
                    if x == argmin else '%8.6f' % (r,))
            for x, r in enumerate(row)
            ]))

        %timeit ut.isect(list1, list2)
        %timeit np.intersect1d(list1, list2, assume_unique=True)
        %timeit set.intersection(set(list1), set(list2))

        #def highlight_max(s):
        #    '''
        #    highlight the maximum in a Series yellow.
        #    '''
        #    is_max = s == s.max()
        #    return ['background-color: yellow' if v else '' for v in is_max]
        #df.style.apply(highlight_max)
    """
    set2 = set(list2)
    return [item for item in list1 if item in set2]


def union_ordered(*lists):
    return unique_ordered(flatten(lists))


def union(*lists, **kwargs):
    """
    Ignore:
        %timeit len(reduce(set.union, map(set, x)))
        %timeit len(ut.union(*x))
        %timeit len(ut.unique(ut.flatten(ut.lmap(np.unique, x))))
        %timeit len(ut.unique(ut.flatten(x)))
        %timeit len(ut.union(*x))
        %timeit len(ut.list_union(*x))
        %timeit len(set.union(*[set(list_) for list_ in lists]))
        %timeit len(set.union(*(set(list_) for list_ in lists)))
    """
    if kwargs.get('ordered', True):
        return union_ordered(*lists)
    else:
        return list_union(*lists)


def list_intersection(list1, list2):
    return set(list1).intersection(set(list2))


def list_issubset(list1, list2):
    return set(list1).issubset(set(list2))


def list_issuperset(list1, list2):
    return set(list1).issuperset(set(list2))


def list_isdisjoint(list1, list2):
    return set(list1).isdisjoint(set(list2))


def list_union(*lists):
    return set.union(*(set(list_) for list_ in lists))
    # return set.union(*[set(list_) for list_ in lists])


def isect_indices(items1, items2):
    set1_ = set(items1)
    set2_ = set(items2)
    items_isect = set1_.intersection(set2_)
    idxs1 = find_list_indexes(items1, items_isect)
    idxs2 = find_list_indexes(items2, items_isect)
    return idxs1, idxs2


intersect_ordered = isect

is_subset = list_issubset
is_superset = list_issuperset

issubset = list_issubset
issuperset = list_issuperset
isdisjoint = list_isdisjoint


def list_set_equal(list1, list2):
    return set(list1) == set(list2)


def is_subset_of_any(set_, other_sets):
    """
    returns True if set_ is a subset of any set in other_sets

    Args:
        set_ (set):
        other_sets (list of sets):

    Returns:
        bool: flag

    CommandLine:
        python -m utool.util_list --test-is_subset_of_any

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> # build test data
        >>> set_ = {1, 2}
        >>> other_sets = [{1, 4}, {3, 2, 1}]
        >>> # execute function
        >>> result = is_subset_of_any(set_, other_sets)
        >>> # verify results
        >>> print(result)
        True

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> # build test data
        >>> set_ = {1, 2}
        >>> other_sets = [{1, 4}, {3, 2}]
        >>> # execute function
        >>> result = is_subset_of_any(set_, other_sets)
        >>> # verify results
        >>> print(result)
        False
    """
    set_ = set(set_)
    other_sets = map(set, other_sets)
    return any([set_.issubset(other_set) for other_set in other_sets])


def priority_sort(list_, priority):
    r"""
    Args:
        list_ (list):
        priority (list): desired order of items

    Returns:
        list: reordered_list

    CommandLine:
        python -m utool.util_list --test-priority_argsort

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [2, 4, 6, 8, 10]
        >>> priority = [8, 2, 6, 9]
        >>> reordered_list = priority_sort(list_, priority)
        >>> result = str(reordered_list)
        >>> print(result)
        [8, 2, 6, 4, 10]
    """
    # remove requested priority items not in the list
    priority_ = setintersect_ordered(priority, list_)
    reordered_list = unique_ordered(priority_ + list_)
    return reordered_list


def priority_argsort(list_, priority):
    r"""
    Args:
        list_ (list):
        priority (list): desired order of items

    Returns:
        list: reordered_index_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> list_ = [2, 4, 6, 8, 10]
        >>> priority = [8, 2, 6, 9]
        >>> sortx = priority_argsort(list_, priority)
        >>> reordered_list = priority_sort(list_, priority)
        >>> assert ut.take(list_, sortx) == reordered_list
        >>> result = str(sortx)
        >>> print(result)
        [3, 0, 2, 1, 4]
    """
    reordered_list = priority_sort(list_, priority)
    # FIXME: inefficient
    sortx = [list_.index(item) for item in reordered_list]
    return sortx


def flag_unique_items(list_):
    """
    Returns a list of flags corresponding to the first time an item is seen

    Args:
        list_ (list): list of items

    Returns:
        flag_list

    Timing:
        import random
        import utool as ut

        def random_items(n, m):
            rng = random.Random(0)
            return [rng.randint(0, n) for _ in range(m)]

        m = 1000

        def method1(list_):
            seen = set()
            def unseen(item):
                if item in seen:
                    return False
                seen.add(item)
                return True
            flag_list = [unseen(item) for item in list_]
            return flag_list

        def method2(list_):
            return ut.index_to_boolmask([list_.index(x) for x in set(list_)], len(list_))

        def method3(list_):
            return ut.index_to_boolmask(dict(zip(reversed(list_), reversed(range(len(list_))))).values(), len(list_))


        import ubelt as ub
        ub.Timerit.DEFAULT_VERBOSE = False

        ut.qtensure()
        exps = [0, .25, .5, .75, 1, 2]
        pnum_ = pt.make_pnum_nextgen(nSubplots=len(exps))
        current = ut.flag_unique_items

        for count, exp in ut.ProgIter(list(enumerate(exps, start=1))):
            ydatas = ut.ddict(list)
            xdata = []
            for m in ut.ProgIter(list(range(0, 10000, 100)), freq=1):
                xdata.append(m)
                num = 10
                n = int(m ** exp)
                list_ = random_items(n=n, m=m)
                ydatas['method1'].append(ub.Timerit(num).call(method1, list_))
                ydatas['method2'].append(ub.Timerit(num).call(method2, list_))
                ydatas['method3'].append(ub.Timerit(num).call(method3, list_))
                ydatas['current'].append(ub.Timerit(num).call(current, list_))

                # assert method1(list_) == method3(list_)
                # assert method1(list_) == current(list_)

            pt.multi_plot(
                xdata, list(ydatas.values()), label_list=list(ydatas.keys()),
                ylabel='time', title=str(exp), fnum=1, pnum=pnum_())

    """
    len_ = len(list_)
    item_to_index = dict(zip(reversed(list_), reversed(range(len_))))
    flag_list = index_to_boolmask(item_to_index.values(), len_)
    return flag_list


unique_flags = flag_unique_items


def iflag_unique_items(list_):
    """
    Returns a list of flags corresponding to the first time an item is seen

    Args:
        list_ (list): list of items

    Returns:
        flag_iter
    """
    seen = set()
    def unseen(item):
        if item in seen:
            return False
        seen.add(item)
        return True
    flag_iter = (unseen(item) for item in list_)
    return flag_iter


def unique_ordered(list_):
    """
    Returns unique items in ``list_`` in the order they were seen.

    Args:
        list_ (list):

    Returns:
        list: unique_list - unique list which maintains order

    CommandLine:
        python -m utool.util_list --exec-unique_ordered

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [4, 6, 6, 0, 6, 1, 0, 2, 2, 1]
        >>> unique_list = unique_ordered(list_)
        >>> result = ('unique_list = %s' % (str(unique_list),))
        >>> print(result)
        unique_list = [4, 6, 0, 1, 2]
    """
    list_ = list(list_)
    flag_list = flag_unique_items(list_)
    unique_list = compress(list_, flag_list)
    return unique_list


def unique_unordered(list_):
    """
    wrapper around list(set(list_))
    """
    return list(set(list_))


def unique_indices(list_):
    return where(flag_unique_items(list_))


def unique(list_, ordered=True):
    """
    Returns unique items in ``list_``.
    Generally, unordered (*should be) faster.
    """
    if ordered:
        return unique_ordered(list_)
    else:
        return unique_unordered(list_)


def flat_unique(*lists_, **kwargs):
    """ returns items unique across all lists """
    return unique(flatten(lists_), **kwargs)


def setdiff(list1, list2):
    """
    returns list1 elements that are not in list2. preserves order of list1

    Args:
        list1 (list):
        list2 (list):

    Returns:
        list: new_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> list1 = ['featweight_rowid', 'feature_rowid', 'config_rowid', 'featweight_forground_weight']
        >>> list2 = [u'featweight_rowid']
        >>> new_list = setdiff_ordered(list1, list2)
        >>> result = ut.repr4(new_list, nl=False)
        >>> print(result)
        ['feature_rowid', 'config_rowid', 'featweight_forground_weight']
    """
    set2 = set(list2)
    return [item for item in list1 if item not in set2]


def setdiff_flags(list1, list2):
    return list(isetdiff_flags(list1, list2))


def isetdiff_flags(list1, list2):
    """
    move to util_iter
    """
    set2 = set(list2)
    return (item not in set2 for item in list1)


setdiff_ordered = setdiff


def setintersect_ordered(list1, list2):
    """
    returns list1 elements that are in list2. preserves order of list1

    setintersect_ordered

    Args:
        list1 (list):
        list2 (list):

    Returns:
        list: new_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list1 = [1, 2, 3, 5, 8, 13, 21]
        >>> list2 = [6, 4, 2, 21, 8]
        >>> new_list = setintersect_ordered(list1, list2)
        >>> result = new_list
        >>> print(result)
        [2, 8, 21]
    """
    return [item for item in list1 if item in set(list2)]


setintersect = setintersect_ordered


def sortedby(item_list, key_list, reverse=False):
    """ sorts ``item_list`` using key_list

    Args:
        list_ (list): list to sort
        key_list (list): list to sort by
        reverse (bool): sort order is descending (largest first)
                        if reverse is True else acscending (smallest first)

    Returns:
        list : ``list_`` sorted by the values of another ``list``. defaults to
        ascending order

    SeeAlso:
        sortedby2

    Examples:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> list_    = [1, 2, 3, 4, 5]
        >>> key_list = [2, 5, 3, 1, 5]
        >>> result = utool.sortedby(list_, key_list, reverse=True)
        >>> print(result)
        [5, 2, 3, 1, 4]

    """
    assert len(item_list) == len(key_list), (
        'Expected same len. Got: %r != %r' % (len(item_list), len(key_list)))
    sorted_list = [item for (key, item) in
                   sorted(list(zip(key_list, item_list)), reverse=reverse)]
    return sorted_list


def sortedby2(item_list, *args, **kwargs):
    """ sorts ``item_list`` using key_list

    Args:
        item_list (list): list to sort
        *args: multiple lists to sort by
        **kwargs:
            reverse (bool): sort order is descending if True else acscending

    Returns:
        list : ``list_`` sorted by the values of another ``list``. defaults to
        ascending order

    CommandLine:
        python -m utool.util_list --exec-sortedby2 --show

    Examples:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> item_list = [1, 2, 3, 4, 5]
        >>> key_list1 = [1, 1, 2, 3, 4]
        >>> key_list2 = [2, 1, 4, 1, 1]
        >>> args = (key_list1, key_list2)
        >>> kwargs = dict(reverse=False)
        >>> result = ut.sortedby2(item_list, *args, **kwargs)
        >>> print(result)
        [2, 1, 3, 4, 5]

    Examples:
        >>> # ENABLE_DOCTEST
        >>> # Python 3 Compatibility Test
        >>> import utool as ut
        >>> item_list = [1, 2, 3, 4, 5]
        >>> key_list1 = ['a', 'a', 2, 3, 4]
        >>> key_list2 = ['b', 'a', 4, 1, 1]
        >>> args = (key_list1, key_list2)
        >>> kwargs = dict(reverse=False)
        >>> result = ut.sortedby2(item_list, *args, **kwargs)
        >>> print(result)
        [3, 4, 5, 2, 1]
    """
    assert all([len(item_list) == len_ for len_ in map(len, args)])
    reverse = kwargs.get('reverse', False)
    key = operator.itemgetter(*range(1, len(args) + 1))
    tup_list = list(zip(item_list, *args))
    #print(tup_list)
    try:
        sorted_tups = sorted(tup_list, key=key, reverse=reverse)
    except TypeError:
        # Python 3 does not allow sorting mixed types
        def keyfunc(tup):
            return tuple(map(str, tup[1:]))
        sorted_tups = sorted(tup_list, key=keyfunc, reverse=reverse)
    sorted_list = [tup[0] for tup in sorted_tups]
    return sorted_list


def unflat_take(items_list, unflat_index_list):
    r"""
    Returns nested subset of items_list

    Args:
        items_list (list):
        unflat_index_list (list): nested list of indices

    CommandLine:
        python -m utool.util_list --exec-unflat_take

    SeeAlso:
        ut.take

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> items_list = [1, 2, 3, 4, 5]
        >>> unflat_index_list = [[0, 1], [2, 3], [0, 4]]
        >>> result = unflat_take(items_list, unflat_index_list)
        >>> print(result)
        [[1, 2], [3, 4], [1, 5]]
    """
    return [unflat_take(items_list, xs)
            if isinstance(xs, list) else
            take(items_list, xs)
            for xs in unflat_index_list]


def argsort(*args, **kwargs):
    """
    like np.argsort but for lists

    Args:
        *args: multiple lists to sort by
        **kwargs:
            reverse (bool): sort order is descending if True else acscending

    CommandLine:
        python -m utool.util_list argsort

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> result = ut.argsort({'a': 3, 'b': 2, 'c': 100})
        >>> print(result)
    """
    if len(args) == 1 and isinstance(args[0], dict):
        dict_ = args[0]
        index_list = list(dict_.keys())
        value_list = list(dict_.values())
        return sortedby2(index_list, value_list)
    else:
        index_list = list(range(len(args[0])))
        return sortedby2(index_list, *args, **kwargs)


def argsort2(indexable, key=None, reverse=False):
    """
    Returns the indices that would sort a indexable object.

    This is similar to np.argsort, but it is written in pure python and works
    on both lists and dictionaries.

    Args:
        indexable (list or dict): indexable to sort by

    Returns:
        list: indices: list of indices such that sorts the indexable

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> # argsort works on dicts
        >>> dict_ = indexable = {'a': 3, 'b': 2, 'c': 100}
        >>> indices = ut.argsort2(indexable)
        >>> assert list(ut.take(dict_, indices)) == sorted(dict_.values())
        >>> # argsort works on lists
        >>> indexable = [100, 2, 432, 10]
        >>> indices = ut.argsort2(indexable)
        >>> assert list(ut.take(indexable, indices)) == sorted(indexable)
        >>> # argsort works on iterators
        >>> indexable = reversed(range(100))
        >>> indices = ut.argsort2(indexable)
        >>> assert indices[0] == 99
    """
    # Create an iterator of value/key pairs
    if isinstance(indexable, dict):
        vk_iter = ((v, k) for k, v in indexable.items())
    else:
        vk_iter = ((v, k) for k, v in enumerate(indexable))
    # Sort by values and extract the keys
    if key is None:
        indices = [k for v, k in sorted(vk_iter, reverse=reverse)]
    else:
        indices = [k for v, k in sorted(vk_iter, key=lambda vk: key(vk[0]),
                                        reverse=reverse)]
    return indices


def argmax(input_, multi=False):
    """
    Returns index / key of the item with the largest value.

    Args:
        input_ (dict or list):

    References:
        http://stackoverflow.com/questions/16945518/python-argmin-argmax

    Ignore:
        list_ = np.random.rand(10000).tolist()
        %timeit list_.index(max(list_))
        %timeit max(enumerate(list_), key=operator.itemgetter(1))[0]
        %timeit max(enumerate(list_), key=lambda x: x[1])[0]
        %timeit max(range(len(list_)), key=list_.__getitem__)

        input_ = dict_
        list_ = np.random.rand(100000).tolist()
        dict_ = {str(ut.random_uuid()): x for x in list_}
        %timeit list(input_.keys())[ut.argmax(list(input_.values()))]
        %timeit max(input_.items(), key=operator.itemgetter(1))[0]

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *
        >>> import utool as ut
        >>> input_ = [1, 2, 3, 3, 2, 3, 2, 1]
        >>> ut.argmax(input_, multi=True)
        >>> input_ = {1: 4, 2: 2, 3: 3, 3: 4}
        >>> ut.argmax(input_, multi=True)
    """
    if multi:
        if isinstance(input_, dict):
            keys = list(input_.keys())
            values = list(input_.values())
            return [keys[idx] for idx in argmax(values, multi=multi)]
        else:
            return where(equal([max(input_)], input_))
    else:
        if isinstance(input_, dict):
            # its crazy, but this is faster
            # max(input_.items(), key=operator.itemgetter(1))[0]
            return list(input_.keys())[argmax(list(input_.values()))]
        elif hasattr(input_, 'index'):
            return input_.index(max(input_))
        else:
            return max(enumerate(input_), key=operator.itemgetter(1))[0]


def argmin(input_, key=None):
    """
    Returns index / key of the item with the smallest value.

    Args:
        input_ (dict or list):

    Note:
        a[argmin(a, key=key)] == min(a, key=key)
    """
    # if isinstance(input_, dict):
    #     return list(input_.keys())[argmin(list(input_.values()))]
    # elif hasattr(input_, 'index'):
    #     return input_.index(min(input_))
    # else:
    #     return min(enumerate(input_), key=operator.itemgetter(1))[0]
    if isinstance(input, dict):
        return list(input.keys())[argmin(list(input.values()), key=key)]
    else:
        if key is None:
            def _key(item):
                return item[1]
        else:
            def _key(item):
                return key(item[1])
        return min(enumerate(input), key=_key)[0]


def index_complement(index_list, len_=None):
    """
    Returns the other indicies in a list of length ``len_``
    """
    mask1 = index_to_boolmask(index_list, len_)
    mask2 = not_list(mask1)
    index_list_bar = list_where(mask2)
    return index_list_bar


def take_complement(list_, index_list):
    """ Returns items in ``list_`` not indexed by index_list """
    mask = not_list(index_to_boolmask(index_list, len(list_)))
    return compress(list_, mask)


def none_take(list_, index_list):
    """
    Like take but indices can be None

    SeeAlso:
        ut.take
    """
    return [None if index is None else list_[index] for index in index_list]


def take(list_, index_list):
    """
    Selects a subset of a list based on a list of indices.
    This is similar to np.take, but pure python.

    Args:
        list_ (list): some indexable object
        index_list (list, slice, int): some indexing object

    Returns:
        list or scalar: subset of the list

    CommandLine:
        python -m utool.util_list --test-take

    SeeAlso:
        ut.dict_take
        ut.dict_subset
        ut.none_take
        ut.compress

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [0, 1, 2, 3]
        >>> index_list = [2, 0]
        >>> result = take(list_, index_list)
        >>> print(result)
        [2, 0]

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [0, 1, 2, 3]
        >>> index = 2
        >>> result = take(list_, index)
        >>> print(result)
        2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [0, 1, 2, 3]
        >>> index = slice(1, None, 2)
        >>> result = take(list_, index)
        >>> print(result)
        [1, 3]
    """
    try:
        return [list_[index] for index in index_list]
    except TypeError:
        return list_[index_list]
    #if util_iter.isiterable(index_list):
    #else:

# def take

# def take2(item_list, indicies, axis):
#     def _get_axes(list_, axis);
#     pass


def take_percentile(arr, percent):
    """ take the top `percent` items in a list rounding up """
    size = len(arr)
    stop = min(int(size * percent), len(arr))
    return arr[0:stop]


def snapped_slice(size, frac, n):
    r"""
    Creates a slice spanning `n` items in a list of length `size` at position
    `frac`.

    Args:
        size (int): length of the list
        frac (float): position in the range [0, 1]
        n (int): number of items in the slice

    Returns:
        slice: slice object that best fits the criteria

    SeeAlso:
        take_percentile_parts

    Example:

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> print(snapped_slice(0, 0, 10))
        >>> print(snapped_slice(1, 0, 10))
        >>> print(snapped_slice(100, 0, 10))
        >>> print(snapped_slice(9, 0, 10))
        >>> print(snapped_slice(100, 1, 10))
        pass
    """
    if size < n:
        n = size
    start = int(size * frac - ceil(n / 2)) + 1
    stop  = int(size * frac + floor(n / 2)) + 1
    # slide to the front or the back
    buf = 0
    if stop >= size:
        buf = (size - stop)
    elif start < 0:
        buf = 0 - start
    stop += buf
    start += buf
    assert stop <= size, 'out of bounds [%r, %r]' % (stop, start)
    sl = slice(start, stop)
    return sl


def take_around_percentile(arr, frac, n):
    return arr[snapped_slice(len(arr), frac, n)]


def flag_percentile_parts(arr, front=None, mid=None, back=None):
    slices = []
    if front:
        slices += [snapped_slice(len(arr), 0.0, front)]
    if mid:
        slices += [snapped_slice(len(arr), 0.5, mid)]
    if back:
        slices += [snapped_slice(len(arr), 1.0, back)]

    import itertools as it
    indices = sorted(set(it.chain.from_iterable(
        range(*sl.indices(len(arr))) for sl in slices)))
    flags = index_to_boolmask(indices, len(arr))
    return flags


def take_percentile_parts(arr, front=None, mid=None, back=None):
    r"""
    Take parts from front, back, or middle of a list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> arr = list(range(20))
        >>> front = 3
        >>> mid = 3
        >>> back = 3
        >>> result = take_percentile_parts(arr, front, mid, back)
        >>> print(result)
        [0, 1, 2, 9, 10, 11, 17, 18, 19]
    """
    slices = []
    if front:
        slices += [snapped_slice(len(arr), 0.0, front)]
    if mid:
        slices += [snapped_slice(len(arr), 0.5, mid)]
    if back:
        slices += [snapped_slice(len(arr), 1.0, back)]
    parts = flatten([arr[sl] for sl in slices])
    return parts


def list_inverse_take(list_, index_list):
    r"""
    Args:
        list_ (list): list in sorted domain
        index_list (list): index list of the unsorted domain

    Note:
        Seems to be logically equivalent to
        ut.take(list_, ut.argsort(index_list)), but faster

    Returns:
        list: output_list_ - the input list in the unsorted domain

    CommandLine:
        python -m utool.util_list --test-list_inverse_take

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> # build test data
        >>> rank_list = [3, 2, 4, 1, 9, 2]
        >>> prop_list = [0, 1, 2, 3, 4, 5]
        >>> index_list = ut.argsort(rank_list)
        >>> sorted_prop_list = ut.take(prop_list, index_list)
        >>> # execute function
        >>> list_ = sorted_prop_list
        >>> output_list_  = list_inverse_take(list_, index_list)
        >>> output_list2_ = ut.take(list_, ut.argsort(index_list))
        >>> assert output_list_ == prop_list
        >>> assert output_list2_ == prop_list
        >>> # verify results
        >>> result = str(output_list_)
        >>> print(result)

    Timeit::
        %timeit list_inverse_take(list_, index_list)
        %timeit ut.take(list_, ut.argsort(index_list))
    """
    output_list_ = [None] * len(index_list)
    for item, index in zip(list_, index_list):
        output_list_[index] = item
    return output_list_


def broadcast_zip(list1, list2):
    r"""
    Zips elementwise pairs between list1 and list2. Broadcasts
    the first dimension if a single list is of length 1.

    Aliased as bzip

    Args:
        list1 (list):
        list2 (list):

    Returns:
        list: list of pairs

    SeeAlso:
        util_dict.dzip

    Raises:
        ValueError: if the list dimensions are not broadcastable

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> assert list(bzip([1, 2, 3], [4])) == [(1, 4), (2, 4), (3, 4)]
        >>> assert list(bzip([1, 2, 3], [4, 4, 4])) == [(1, 4), (2, 4), (3, 4)]
        >>> assert list(bzip([1], [4, 4, 4])) == [(1, 4), (1, 4), (1, 4)]
        >>> ut.assert_raises(ValueError, bzip, [1, 2, 3], [])
        >>> ut.assert_raises(ValueError, bzip, [], [4, 5, 6])
        >>> ut.assert_raises(ValueError, bzip, [], [4])
        >>> ut.assert_raises(ValueError, bzip, [1, 2], [4, 5, 6])
        >>> ut.assert_raises(ValueError, bzip, [1, 2, 3], [4, 5])
    """
    try:
        len(list1)
    except TypeError:
        list1 = list(list1)
    try:
        len(list2)
    except TypeError:
        list2 = list(list2)
    # if len(list1) == 0 or len(list2) == 0:
    #     # Corner case where either list is empty
    #     return []
    if len(list1) == 1 and len(list2) > 1:
        list1 = list1 * len(list2)
    elif len(list1) > 1 and len(list2) == 1:
        list2 = list2 * len(list1)
    elif len(list1) != len(list2):
        raise ValueError('out of alignment len(list1)=%r, len(list2)=%r' % (
            len(list1), len(list2)))
    # return list(zip(list1, list2))
    return zip(list1, list2)


def where(flag_list):
    """ takes flags returns indexes of True values """
    return [index for index, flag in enumerate(flag_list) if flag]


def equal(list1, list2):
    """ takes flags returns indexes of True values """
    return [item1 == item2 for item1, item2 in broadcast_zip(list1, list2)]


def where_not_None(item_list):
    """ returns list of indexes of non None values

    SeeAlso:
        flag_None_items
    """
    return [index for index, item in enumerate(item_list) if item is not None]


def flag_None_items(list_):
    return [item is None for item in list_]


def flag_not_None_items(list_):
    return [item is not None for item in list_]


def scalar_input_map(func, input_):
    """
    Map like function

    Args:
        func: function to apply
        input_ : either an iterable or scalar value

    Returns:
        If ``input_`` is iterable this function behaves like map
        otherwise applies func to ``input_``
    """
    if util_iter.isiterable(input_):
        return list(map(func, input_))
    else:
        return func(input_)


def partial_imap_1to1(func, si_func):
    """ a bit messy

    DEPRICATE
    """
    @functools.wraps(si_func)
    def wrapper(input_):
        if not util_iter.isiterable(input_):
            return func(si_func(input_))
        else:
            return list(map(func, si_func(input_)))
    set_funcname(wrapper, util_str.get_callable_name(func) + '_mapper_' + get_funcname(si_func))
    return wrapper


def sample_zip(items_list, num_samples, allow_overflow=False, per_bin=1):
    """ Helper for sampling

    Given a list of lists, samples one item for each list and bins them into
    num_samples bins. If all sublists are of equal size this is equivilent to a
    zip, but otherewise consecutive bins will have monotonically less
    elemements

    # Doctest doesn't work with assertionerror
    #util_list.sample_zip(items_list, 2)
    #...
    #AssertionError: Overflow occured

    Args:
        items_list (list):
        num_samples (?):
        allow_overflow (bool):
        per_bin (int):

    Returns:
        tuple : (samples_list, overflow_samples)

    Examples:
        >>> # DISABLE_DOCTEST
        >>> from utool import util_list
        >>> items_list = [[1, 2, 3, 4, 0], [5, 6, 7], [], [8, 9], [10]]
        >>> util_list.sample_zip(items_list, 5)
        ...
        [[1, 5, 8, 10], [2, 6, 9], [3, 7], [4], [0]]
        >>> util_list.sample_zip(items_list, 2, allow_overflow=True)
        ...
        ([[1, 5, 8, 10], [2, 6, 9]], [3, 7, 4])
        >>> util_list.sample_zip(items_list, 4, allow_overflow=True, per_bin=2)
        ...
        ([[1, 5, 8, 10, 2, 6, 9], [3, 7, 4], [], []], [0])
    """
    # Prealloc a list of lists
    samples_list = [[] for _ in range(num_samples)]
    # Sample the ix-th value from every list
    samples_iter = zip_longest(*items_list)
    sx = 0
    for ix, samples_ in zip(range(num_samples), samples_iter):
        samples = filter_Nones(samples_)
        samples_list[sx].extend(samples)
        # Put per_bin from each sublist into a sample
        if (ix + 1) % per_bin == 0:
            sx += 1
    # Check for overflow
    if allow_overflow:
        overflow_samples = flatten([filter_Nones(samples_) for samples_ in samples_iter])
        return samples_list, overflow_samples
    else:
        try:
            samples_iter.next()
        except StopIteration:
            pass
        else:
            raise AssertionError('Overflow occured')
        return samples_list


def sample_lists(items_list, num=1, seed=None):
    r"""
    Args:
        items_list (list):
        num (int): (default = 1)
        seed (None): (default = None)

    Returns:
        list: samples_list

    CommandLine:
        python -m utool.util_list --exec-sample_lists

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> items_list = [[], [1, 2, 3], [4], [5, 6], [7, 8, 9, 10]]
        >>> num = 2
        >>> seed = 0
        >>> samples_list = sample_lists(items_list, num, seed)
        >>> result = ('samples_list = %s' % (str(samples_list),))
        >>> print(result)
        samples_list = [[], [3, 2], [4], [5, 6], [10, 9]]
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    def random_choice(items, num):
        size = min(len(items), num)
        return rng.choice(items, size, replace=False).tolist()
    samples_list = [random_choice(items, num)
                    if len(items) > 0 else []
                    for items in items_list]
    return samples_list


def strided_sample(items, num, offset=0):
    r"""
    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> # build test data
        >>> items = [1, 2, 3, 4, 5]
        >>> num = 3
        >>> offset = 0
        >>> # execute function
        >>> sample_items = strided_sample(items, num, offset)
        >>> # verify results
        >>> result = str(sample_items)
        >>> print(result)
    """
    import math
    stride = max(int(math.ceil(len(items) / num)), 1)
    sample_items = items[offset::stride]
    return sample_items


def issorted(list_, op=operator.le):
    """
    Determines if a list is sorted

    Args:
        list_ (list):
        op (func): sorted operation (default=operator.le)

    Returns:
        bool : True if the list is sorted
    """
    return all(op(list_[ix], list_[ix + 1]) for ix in range(len(list_) - 1))


def find_nonconsec_values(values, min_=None, max_=None):
    """
    Determines if a list of values is consecutive (ascending)

    Args:
        values (list): list of values, sorted and unique
        min_(int): minimum value in range defaults min(values)
        max_(int): maximum value in range defaults max(values)

    Returns:
        missing_values: missing values that would make the list consecutive

    CommandLine:
        python -m utool.util_list --test-find_nonconsec_values

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import numpy as np
        >>> values = np.array([-2, 1,  2, 10])
        >>> result = find_nonconsec_values(values)
        >>> print(result)
        [-1, 0, 3, 4, 5, 6, 7, 8, 9]
    """
    # values = sorted(set(values))
    if min_ is None:
        min_ = values[0]
    if max_ is None:
        max_ = values[-1]
    valx   = 0
    missing_values = []
    for check in range(min_, max_ + 1):
        if values[valx] != check:
            missing_values.append(check)
        else:
            valx += 1
    return missing_values


def group_consecutives(data, stepsize=1):
    """
    Return list of consecutive lists of numbers from data (number list).

    References:
        http://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
    """
    run = []
    result = [run]
    expect = None
    for item in data:
        if (item == expect) or (expect is None):
            run.append(item)
        else:
            run = [item]
            result.append(run)
        expect = item + stepsize
    return result


def group_consecutives_numpy(data, stepsize=1):
    """

    Args:
        data (?):
        stepsize (int):

    Returns:
        list: list of ndarrays

    References:
        http://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy

    CommandLine:
        python -m utool.util_list --test-group_consecutives

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> # build test data
        >>> data = np.array([  0,   1,   2,   3,   4, 320, 636, 637, 638, 639])
        >>> stepsize = 1
        >>> # execute function
        >>> result = group_consecutives(data, stepsize)
        >>> # verify results
        >>> print(result)
        [array([0, 1, 2, 3, 4]), array([320]), array([636, 637, 638, 639])]

    Timeit::
        %timeit group_consecutives_numpy(data, stepsize) #  14.8 µs per loop
        %timeit group_consecutives(data, stepsize) # 4.47 µs per loop

    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def debug_consec_list(list_):
    """
    Returns:
        tuple of (missing_items, missing_indices, duplicate_items)
    """
    if not issorted(list_):
        print('warning list is not sorted. indices will not match')
    sortedlist = sorted(list_)
    start = sortedlist[0]
    last = start - 1
    missing_vals = []
    missing_indices = []
    duplicate_items = []
    for count, item in enumerate(sortedlist):
        diff = item - last
        if diff > 1:
            missing_indices.append(count)
            for miss in range(last + 1, last + diff):
                missing_vals.append(miss)
        elif diff == 0:
            duplicate_items.append(item)
        elif diff == 1:
            # Expected case
            pass
        else:
            raise AssertionError('We sorted the list. diff can not be negative')
        last = item
    return missing_vals, missing_indices, duplicate_items


def find_duplicate_items(items, k=2):
    r"""
    Args:
        items (list):

    Returns:
        dict: duplicate_map of indexes

    CommandLine:
        python -m utool.util_list --test-find_duplicate_items

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> items = [0, 1, 2, 3, 3, 0, 12, 2, 9]
        >>> duplicate_map = find_duplicate_items(items)
        >>> result = str(duplicate_map)
        >>> print(result)
    """
    import utool as ut
    # Build item histogram
    duplicate_map = ut.ddict(list)
    for count, item in enumerate(items):
        duplicate_map[item].append(count)
    # remove singleton items
    singleton_keys = []
    for key in six.iterkeys(duplicate_map):
        if len(duplicate_map[key]) == 1:
            singleton_keys.append(key)
    for key in singleton_keys:
        del duplicate_map[key]
    duplicate_map = dict(duplicate_map)
    return duplicate_map


#get_non_consecutive_positions = debug_consec_list


def duplicates_exist(items):
    """ returns if list has duplicates """
    return len(items) - len(set(items)) != 0


def isunique(items):
    return not duplicates_exist(items)


def print_duplicate_map(duplicate_map, *args, **kwargs):
    # args are corresponding lists
    import utool as ut
    printfn = kwargs.get('printfn', print)
    printfn('There are %d duplicates' % (len(duplicate_map)))
    for key, index_list in six.iteritems(duplicate_map):
        printfn('item=%s appears %d times at indices: %r' % (key, len(index_list), index_list))
        for argx, arg in enumerate(args):
            #argname = 'arg%d' % (argx)
            argname = ut.get_varname_from_stack(arg, N=2)
            for index in index_list:
                printfn(' * %s[%d] = %r' % (argname, index, arg[index]))
    return duplicate_map


def debug_duplicate_items(items, *args, **kwargs):
    import utool as ut
    pad_stdout = kwargs.get('pad_stdout', True)
    if pad_stdout:
        print('')

    varname = ut.get_varname_from_locals(items, ut.get_parent_frame().f_locals)
    print('[util_list] +--- DEBUG DUPLICATE ITEMS  %r ---' % (varname,))
    def printfn(msg):
        print('[util_list] |' + msg)
    #with ut.Indenter('[util_list] | '):
    duplicate_map = ut.find_duplicate_items(items)
    printkw = {'printfn': printfn}
    ut.print_duplicate_map(duplicate_map, *args, **printkw)
    print('[util_list] L--- FINISH DEBUG DUPLICATE ITEMS ---')
    if pad_stdout:
        print('')
    return duplicate_map


def list_depth(list_, func=max, _depth=0):
    """
    Returns the deepest level of nesting within a list of lists

    Args:
       list_  : a nested listlike object
       func   : depth aggregation strategy (defaults to max)
       _depth : internal var

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [[[[[1]]], [3]], [[1], [3]], [[1], [3]]]
        >>> result = (list_depth(list_, _depth=0))
        >>> print(result)

    """
    depth_list = [list_depth(item, func=func, _depth=_depth + 1)
                  for item in  list_ if util_type.is_listlike(item)]
    if len(depth_list) > 0:
        return func(depth_list)
    else:
        return _depth


def depth(sequence, func=max, _depth=0):
    """
    Find the nesting depth of a nested sequence
    """
    if isinstance(sequence, dict):
        sequence = list(sequence.values())
    depth_list = [depth(item, func=func, _depth=_depth + 1)
                  for item in sequence if (isinstance(item, dict) or util_type.is_listlike(item))]
    if len(depth_list) > 0:
        return func(depth_list)
    else:
        return _depth


def list_deep_types(list_):
    """
    Returns all types in a deep list
    """
    type_list = []
    for item in list_:
        if util_type.is_listlike(item):
            type_list.extend(list_deep_types(item))
        else:
            type_list.append(type(item))
    return type_list


def depth_profile(list_, max_depth=None, compress_homogenous=True, compress_consecutive=False, new_depth=False):
    r"""
    Returns a nested list corresponding the shape of the nested structures
    lists represent depth, tuples represent shape. The values of the items do
    not matter. only the lengths.

    Args:
        list_ (list):
        max_depth (None):
        compress_homogenous (bool):
        compress_consecutive (bool):  experimental

    CommandLine:
        python -m utool.util_list --test-depth_profile

    Setup:
        >>> from utool.util_list import *  # NOQA

    Example0:
        >>> # ENABLE_DOCTEST
        >>> list_ = [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
        >>> result = depth_profile(list_)
        >>> print(result)
        (2, 3, 4)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> list_ = [[[[[1]]], [3, 4, 33]], [[1], [2, 3], [4, [5, 5]]], [1, 3]]
        >>> result = depth_profile(list_)
        >>> print(result)
        [[(1, 1, 1), 3], [1, 2, [1, 2]], 2]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> list_ = [[[[[1]]], [3, 4, 33]], [[1], [2, 3], [4, [5, 5]]], [1, 3]]
        >>> result = depth_profile(list_, max_depth=1)
        >>> print(result)
        [[(1, '1'), 3], [1, 2, [1, '2']], 2]

    Example3:
        >>> # ENABLE_DOCTEST
        >>> list_ = [[[1, 2], [1, 2, 3]], None]
        >>> result = depth_profile(list_, compress_homogenous=True)
        >>> print(result)
        [[2, 3], 1]

    Example4:
        >>> # ENABLE_DOCTEST
        >>> list_ = [[3, 2], [3, 2], [3, 2], [3, 2], [3, 2], [3, 2], [9, 5, 3], [2, 2]]
        >>> result = depth_profile(list_, compress_homogenous=True, compress_consecutive=True)
        >>> print(result)
        [2] * 6 + [3, 2]

    Example5:
        >>> # ENABLE_DOCTEST
        >>> list_ = [[[3, 9], 2], [[3, 9], 2], [[3, 9], 2], [[3, 9], 2]]  #, [3, 2], [3, 2]]
        >>> result = depth_profile(list_, compress_homogenous=True, compress_consecutive=True)
        >>> print(result)
        (4, [2, 1])

    Example6:
        >>> # ENABLE_DOCTEST
        >>> list_ = [[[[1, 2]], [1, 2]], [[[1, 2]], [1, 2]], [[[0, 2]], [1]]]
        >>> result1 = depth_profile(list_, compress_homogenous=True, compress_consecutive=False)
        >>> result2 = depth_profile(list_, compress_homogenous=True, compress_consecutive=True)
        >>> result = str(result1) + '\n' + str(result2)
        >>> print(result)
        [[(1, 2), 2], [(1, 2), 2], [(1, 2), 1]]
        [[(1, 2), 2]] * 2 + [[(1, 2), 1]]

    Example7:
        >>> # ENABLE_DOCTEST
        >>> list_ = [[{'a': [1, 2], 'b': [3, 4, 5]}, [1, 2, 3]], None]
        >>> result = depth_profile(list_, compress_homogenous=True)
        >>> print(result)

    Example8:
        >>> # ENABLE_DOCTEST
        >>> list_ = [[[1]], [[[1, 1], [1, 1]]], [[[[1, 3], 1], [[1, 3, 3], 1, 1]]]]
        >>> result = depth_profile(list_, compress_homogenous=True)
        >>> print(result)

    Example9:
        >>> # ENABLE_DOCTEST
        >>> list_ = []
        >>> result = depth_profile(list_)
        >>> print(result)

        # THIS IS AN ERROR???
        SHOULD BE
        #[1, 1], [1, 2, 2], (1, ([1, 2]), (

    Example10:
        >>> # ENABLE_DOCTEST
        >>> fm1 = [[0, 0], [0, 0]]
        >>> fm2 = [[0, 0], [0, 0], [0, 0]]
        >>> fm3 = [[0, 0], [0, 0], [0, 0], [0, 0]]
        >>> list_ = [0, 0, 0]
        >>> list_ = [fm1, fm2, fm3]
        >>> max_depth = 0
        >>> new_depth = True
        >>> result = depth_profile(list_, max_depth=max_depth, new_depth=new_depth)
        >>> print(result)
    """
    if isinstance(list_, dict):
        list_ = list(list_.values())   # handle dict
    level_shape_list = []
    # For a pure bottom level list return the length
    if not any(map(util_type.is_listlike, list_)):
        return len(list_)

    if False and new_depth:
        pass
        # max_depth_ = None if max_depth is None else max_depth - 1
        # if max_depth_ is None or max_depth_ > 0:
        #     pass
        # else:
        #     for item in list_:
        #         if isinstance(item, dict):
        #             item = list(item.values())  # handle dict
        #         if util_type.is_listlike(item):
        #             if max_depth is None:
        #                 level_shape_list.append(depth_profile(item, None))
        #             else:
        #                 if max_depth >= 0:
        #                     level_shape_list.append(depth_profile(item, max_depth - 1))
        #                 else:
        #                     level_shape_list.append(str(len(item)))
        #         else:
        #             level_shape_list.append(1)
    else:
        for item in list_:
            if isinstance(item, dict):
                item = list(item.values())  # handle dict
            if util_type.is_listlike(item):
                if max_depth is None:
                    level_shape_list.append(depth_profile(item, None))
                else:
                    if max_depth >= 0:
                        level_shape_list.append(depth_profile(item, max_depth - 1))
                    else:
                        level_shape_list.append(str(len(item)))
            else:
                level_shape_list.append(1)

    if compress_homogenous:
        # removes redudant information by returning a shape duple
        if allsame(level_shape_list):
            dim_ = level_shape_list[0]
            len_ = len(level_shape_list)
            if isinstance(dim_, tuple):
                level_shape_list = tuple([len_] + list(dim_))
            else:
                level_shape_list = tuple([len_, dim_])

    if compress_consecutive:
        hash_list = list(map(hash, map(str, level_shape_list)))
        consec_list = group_consecutives(hash_list, 0)
        if len(consec_list) != len(level_shape_list):
            len_list = list(map(len, consec_list))
            cumsum_list = np.cumsum(len_list)
            consec_str = '['
            thresh = 1
            for len_, cumsum in zip(len_list, cumsum_list):
                value = level_shape_list[cumsum - 1]
                if len_ > thresh:
                    consec_str += str(value) + '] * ' + str(len_)
                    consec_str += ' + ['
                else:
                    consec_str += str(value) + ', '
            if consec_str.endswith(', '):
                consec_str = consec_str[:-2]
                #consec_str += ']'
            #consec_str = consec_str.rstrip(', ').rstrip(']')
            #consec_str = consec_str.rstrip(', ')
            #if consec_str.endswith(']'):
            #    consec_str = consec_str[:-1]
            consec_str += ']'
            level_shape_list = consec_str
    return level_shape_list


def list_type(list_):
    types =  unique_ordered(list(map(type, list_)))
    if len(types) == 1:
        return types[0]
    else:
        return types


def list_type_profile(sequence, compress_homogenous=True, with_dtype=True):
    """
    similar to depth_profile but reports types

    Args:
        sequence (?):
        compress_homogenous (bool): (default = True)

    Returns:
        str: level_type_str

    CommandLine:
        python -m utool.util_list --test-list_type_profile
        python3 -m utool.util_list --test-list_type_profile

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import numpy as np
        >>> sequence = [[1, 2], np.array([1, 2, 3], dtype=np.int32), (np.array([1, 2, 3], dtype=np.int32),)]
        >>> compress_homogenous = True
        >>> level_type_str = list_type_profile(sequence, compress_homogenous)
        >>> result = ('level_type_str = %s' % (str(level_type_str),))
        >>> print(result)
        level_type_str = list(list(int*2), ndarray[int32], tuple(ndarray[int32]*1))
    """
    # For a pure bottom level list return the length
    #if not any(map(util_type.is_listlike, sequence)) or (isinstance(sequence, np.ndarray) and sequence.dtype != object):
    if not util_type.is_listlike(sequence) or (isinstance(sequence, np.ndarray) and sequence.dtype != object):
        typename = str(type(sequence)).replace('<type \'', '').replace('\'>', '')
        if six.PY3:
            typename = str(type(sequence)).replace('<class \'', '').replace('\'>', '')
        if with_dtype and typename == 'numpy.ndarray':
            typename = typename.replace('numpy.', '')
            typename += '[%s]' % (sequence.dtype,)

        level_type_str = typename
        return level_type_str
    if len(sequence) == 0:
        return ''

    level_type_list = []
    for item in sequence:
        #if util_type.is_listlike(item):
        item_type_profile = list_type_profile(item, with_dtype=with_dtype)
        level_type_list.append(item_type_profile)

    if compress_homogenous:
        # removes redudant information by returning a type and number
        if allsame(level_type_list):
            type_ = level_type_list[0]
            level_type_str = str(type_) + '*' + str(len(level_type_list))
        else:
            level_type_str = ', '.join(level_type_list)
    typename = str(type(sequence)).replace('<type \'', '').replace('\'>', '')
    if six.PY3:
        typename = str(type(sequence)).replace('<class \'', '').replace('\'>', '')
    level_type_str = typename + '(' + str(level_type_str) + ')'
    return level_type_str


def type_profile2(sequence, TypedSequence=None):
    """
    similar to depth_profile but reports types

    Args:
        sequence (?):
        compress_homogenous (bool): (default = True)

    Returns:
        str: level_type_str

    CommandLine:
        python -m utool.util_list --exec-type_profile2

    Example:
        >>> # DISABLE_DOCTEST
        >>> sequence = []
        >>> from utool.util_list import *  # NOQA
        >>> self = typeprof = type_profile2(sequence, type_sequence_factory())
        >>> result = ('level_type_str = %s' % (str(level_type_str),))
        >>> print(result)
    """
    if TypedSequence is None:
        TypedSequence = type_sequence_factory()
    # For a pure bottom level list return the length
    #if not any(map(util_type.is_listlike, sequence)) or (isinstance(sequence, np.ndarray) and sequence.dtype != object):
    if not util_type.is_listlike(sequence) or (isinstance(sequence, np.ndarray) and sequence.dtype != object):
        # Scalar / ndarray type
        if type(sequence) == 'numpy.ndarray':
            subtype_list = '[%s]' % (sequence.dtype,)
        else:
            subtype_list = None
        return TypedSequence(type(sequence), subtype_list)
    elif util_type.is_listlike(sequence):
        # Sequence type
        sequence_type = type(sequence)

        subtype_list = []
        for item in sequence:
            item_type_profile = type_profile2(item, TypedSequence=TypedSequence)
            subtype_list.append(item_type_profile)
        sequence_type_profile = TypedSequence(sequence_type, subtype_list)
        return sequence_type_profile
        #level_type_str = typename + '(' + str(level_type_str) + ')'
        #return level_type_str


def type_sequence_factory():
    from utool import util_dev
    class TypedSequence(util_dev.NiceRepr):
        def __init__(self, type_, subtype_list=None):
            self.type_ = type_
            self.subtype_list = subtype_list

        def __eq__(self, other):
            return str(self) == str(other)

        def type_str(self):
            type_str = six.text_type(self.type_)
            type_str = type_str.replace('\'>', '')
            type_str = type_str.replace('<type \'', '')
            type_str = type_str.replace('<class \'', '')
            type_str = type_str.replace('numpy.', '')

            return type_str

        def subtype_str(self):
            import utool as ut
            if self.subtype_list is None:
                return ''
            elif isinstance(self.subtype_list, six.string_types):
                return self.subtype_list
            else:
                prev = None
                grouped = []
                group = []
                for item in self.subtype_list:
                    if item == prev or prev is None:
                        group.append(item)
                    else:
                        grouped.append(group)
                        group = [item]
                    prev = item
                grouped.append(group)

                if len(grouped) == len(self.subtype_list):
                    toret = '(' + ', '.join([sub.__nice__()[1:] for sub in self.subtype_list]) + ')'
                else:
                    compressed_types = ut.take_column(grouped, 0)
                    compressed_lens = ut.lmap(len, grouped)
                    zip(compressed_types, compressed_lens)
                    groupstrs = [
                        sub.__nice__()[1:] + '*' + str(num) if num > 1 else
                        sub.__nice__()[1:]
                        for sub, num in zip(compressed_types, compressed_lens)]
                    toret = '(' + ', '.join(groupstrs) + ')'
                return toret

        def __nice__(self):
            type_str = self.type_str()
            if self.subtype_list is not None:
                type_str += self.subtype_str()
            return ' ' + type_str
    return TypedSequence


type_profile = list_type_profile


def list_cover(list1, list2):
    r"""
    returns boolean for each position in list1 if it is in list2

    Args:
        list1 (list):
        list2 (list):

    Returns:
        list: incover_list - true where list1 intersects list2

    CommandLine:
        python -m utool.util_list --test-list_cover

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> # build test data
        >>> list1 = [1, 2, 3, 4, 5, 6]
        >>> list2 = [2, 3, 6]
        >>> # execute function
        >>> incover_list = list_cover(list1, list2)
        >>> # verify results
        >>> result = str(incover_list)
        >>> print(result)
        [False, True, True, False, False, True]
    """
    set2 = set(list2)
    incover_list = [item1 in set2 for item1 in list1]
    return incover_list


def and_lists(*args):
    #[all(tup) for tup in zip(*args)]
    return list(util_iter.and_iters(*args))


def xor_lists(*args):
    r"""
    Returns:
        list:

    CommandLine:
        python -m utool.util_list --test-xor_lists

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> args = ([True, False, False, True], [True, True, False, False])
        >>> result = xor_lists(*args)
        >>> print(result)
        [False, True, False, True]
    """
    return [reduce(operator.xor, tup) for tup in zip(*args)]


def not_list(flag_list):
    return [not flag for flag in flag_list]


def or_lists(*args):
    return [any(tup) for tup in zip(*args)]


def make_sortby_func(item_list, reverse=False):
    sortxs_ = argsort(item_list)
    sortxs = sortxs_[::-1] if reverse else sortxs_
    def sortby_func(list_):
        return take(list_, sortxs)
    return sortby_func


def filter_startswith(list_, str_):
    def item_startswith(item):
        return item.startswith(str_)
    return list(filter(item_startswith, list_))


def list_roll(list_, n):
    """
    Like numpy.roll for python lists

    Args:
        list_ (list):
        n (int):

    Returns:
        list:

    References:
        http://stackoverflow.com/questions/9457832/python-list-rotation

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [1, 2, 3, 4, 5]
        >>> n = 2
        >>> result = list_roll(list_, n)
        >>> print(result)
        [4, 5, 1, 2, 3]

    Ignore:
        np.roll(list_, n)
    """
    return list_[-n:] + list_[:-n]


def list_argmax(list_):
    return np.argmax(np.array(list_))


def list_argmaxima(list_):
    r"""
    Args:
        list_ (list):

    Returns:
        list: argmaxima

    CommandLine:
        python -m utool.util_list --exec-list_argmaxima

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = np.array([1, 2, 3, 3, 3, 2, 1])
        >>> argmaxima = list_argmaxima(list_)
        >>> result = ('argmaxima = %s' % (str(argmaxima),))
        >>> print(result)
        argmaxima = [2 3 4]
    """
    argmax = list_argmax(list_)
    maxval = list_[argmax]
    argmaxima = np.where((np.isclose(maxval, list_)))[0]
    return argmaxima


def make_index_lookup(list_, dict_factory=dict):
    r"""
    Args:
        list_ (list): assumed to have unique items

    Returns:
        dict: mapping from item to index

    CommandLine:
        python -m utool.util_list --exec-make_index_lookup

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> list_ = [5, 3, 8, 2]
        >>> idx2_item = ut.make_index_lookup(list_)
        >>> result = ut.repr2(idx2_item, nl=False)
        >>> assert ut.dict_take(idx2_item, list_) == list(range(len(list_)))
        >>> print(result)
        {2: 3, 3: 1, 5: 0, 8: 2}
    """
    return dict_factory(zip(list_, range(len(list_))))


def list_alignment(list1, list2, missing=False):
    """
    Assumes list items are unique

    Args:
        list1 (list): a list of unique items to be aligned
        list2 (list): a list of unique items in a desired ordering
        missing (bool): True if list2 can contain items not in list1

    Returns:
        list: sorting that will map list1 onto list2

    CommandLine:
        python -m utool.util_list list_alignment

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> list1 = ['b', 'c', 'a']
        >>> list2 = ['a', 'b', 'c']
        >>> sortx = list_alignment(list1, list2)
        >>> list1_aligned = take(list1, sortx)
        >>> assert list1_aligned == list2

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> list1 = ['b', 'c', 'a']
        >>> list2 = ['a', 'a2', 'b', 'c', 'd']
        >>> sortx = ut.list_alignment(list1, list2, missing=True)
        >>> print('sortx = %r' % (sortx,))
        >>> list1_aligned = ut.none_take(list1, sortx)
        >>> result = ('list1_aligned = %s' % (ut.repr2(list1_aligned),))
        >>> print(result)
        list1_aligned = ['a', None, 'b', 'c', None]
    """
    import utool as ut
    item1_to_idx = make_index_lookup(list1)
    if missing:
        sortx = ut.dict_take(item1_to_idx, list2, None)
    else:
        sortx = ut.take(item1_to_idx, list2)
    return sortx


def unique_inverse(item_list):
    """
    Like np.unique(item_list, return_inverse=True)
    """
    import utool as ut
    unique_items = ut.unique(item_list)
    inverse = list_alignment(unique_items, item_list)
    return unique_items, inverse


def list_transpose(list_, shape=None):
    r"""
    Swaps rows and columns.
    nCols should be specified if the initial list is empty.

    Args:
        list_ (list):

    Returns:
        list:

    CommandLine:
        python -m utool.util_list --test-list_transpose

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [[1, 2], [3, 4]]
        >>> result = list_transpose(list_)
        >>> print(result)
        [(1, 3), (2, 4)]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = []
        >>> result = list_transpose(list_, shape=(0, 5))
        >>> print(result)
        [[], [], [], [], []]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [[], [], [], [], []]
        >>> result = list_transpose(list_)
        >>> print(result)
        []

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> list_ = [[1, 2, 3], [3, 4]]
        >>> ut.assert_raises(ValueError, list_transpose, list_)
    """
    num_cols_set = unique([len(x) for x in list_])
    if shape is None:
        if len(num_cols_set) == 0:
            raise ValueError('listT does not support empty transpose without shapes')
    else:
        assert len(shape) == 2, 'shape must be a 2-tuple'
        if len(num_cols_set) == 0:
            return [[] for _ in range(shape[1])]
        elif num_cols_set[0] == 0:
            return []
    if len(num_cols_set) != 1:
        raise ValueError('inconsistent column lengths=%r' % (num_cols_set,))
    return list(zip(*list_))


listT = list_transpose


def delete_items_by_index(list_, index_list, copy=False):
    """
    Remove items from ``list_`` at positions specified in ``index_list``
    The original ``list_`` is preserved if ``copy`` is True

    Args:
        list_ (list):
        index_list (list):
        copy (bool): preserves original list if True

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [8, 1, 8, 1, 6, 6, 3, 4, 4, 5, 6]
        >>> index_list = [2, -1]
        >>> result = delete_items_by_index(list_, index_list)
        >>> print(result)
        [8, 1, 1, 6, 6, 3, 4, 4, 5]
    """
    if copy:
        list_ = list_[:]
    # Rectify negative indicies
    index_list_ = [(len(list_) + x if x < 0 else x) for x in index_list]
    # Remove largest indicies first
    index_list_ = sorted(index_list_, reverse=True)
    for index in index_list_:
        del list_[index]
    return list_


def delete_list_items(list_, item_list, copy=False):
    r"""
    Remove items in ``item_list`` from ``list_``.
    The original ``list_`` is preserved if ``copy`` is True
    """
    if copy:
        list_ = list_[:]
    for item in item_list:
        list_.remove(item)
    return list_


def unflat_map(func, unflat_items, vectorized=False, **kwargs):
    r"""
    Uses an ibeis lookup function with a non-flat rowid list.
    In essence this is equivilent to [list(map(func, _items)) for _items in unflat_items].
    The utility of this function is that it only calls method once.
    This is more efficient for calls that can take a list of inputs

    Args:
        func (func): function
        unflat_items (list): list of rowid lists

    Returns:
        list of values: unflat_vals

    CommandLine:
        python -m utool.util_list --test-unflat_map

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> vectorized = False
        >>> kwargs = {}
        >>> def func(x):
        >>>     return x + 1
        >>> unflat_items = [[], [1, 2, 3], [4, 5], [6, 7, 8, 9], [], []]
        >>> unflat_vals = unflat_map(func, unflat_items)
        >>> result = str(unflat_vals)
        >>> print(result)
        [[], [2, 3, 4], [5, 6], [7, 8, 9, 10], [], []]
    """
    import utool as ut
    # First flatten the list, and remember the original dimensions
    flat_items, reverse_list = ut.invertible_flatten2(unflat_items)
    # Then preform the lookup / implicit mapping
    if vectorized:
        # func is vectorized
        flat_vals = func(flat_items, **kwargs)
    else:
        flat_vals = [func(item, **kwargs) for item in flat_items]
    if True:
        assert len(flat_vals) == len(flat_items), (
            'flat lens not the same, len(flat_vals)=%d len(flat_items)=%d' %
            (len(flat_vals), len(flat_items),))
    # Then ut.unflatten2 the results to the original input dimensions
    unflat_vals = ut.unflatten2(flat_vals, reverse_list)
    if True:
        assert len(unflat_vals) == len(unflat_items), (
            'unflat lens not the same, len(unflat_vals)=%d len(unflat_rowids)=%d' %
            (len(unflat_vals), len(unflat_items),))
    return unflat_vals


def unflat_vecmap(func, unflat_items, vectorized=False, **kwargs):
    """ unflat map for vectorized functions """
    import utool as ut
    # First flatten the list, and remember the original dimensions
    flat_items, reverse_list = ut.invertible_flatten2(unflat_items)
    # Then preform the lookup / implicit mapping
    flat_vals = func(flat_items, **kwargs)
    if True:
        assert len(flat_vals) == len(flat_items), (
            'flat lens not the same, len(flat_vals)=%d len(flat_items)=%d' %
            (len(flat_vals), len(flat_items),))
    # Then ut.unflatten2 the results to the original input dimensions
    unflat_vals = ut.unflatten2(flat_vals, reverse_list)
    if True:
        assert len(unflat_vals) == len(unflat_items), (
            'unflat lens not the same, len(unflat_vals)=%d len(unflat_rowids)=%d' %
            (len(unflat_vals), len(unflat_items),))
    return unflat_vals


def list_getattr(list_, attrname):
    return list(map(operator.attrgetter(attrname), list_))


def list_reshape(list_, new_shape, trail=False):
    r"""
    reshapes leaving trailing dimnsions in front if prod(new_shape) != len(list_)

    Args:
        list_ (list):
        new_shape (tuple):

    Returns:
        list: list_

    CommandLine:
        python -m utool.util_list --exec-list_reshape --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool as ut
        >>> import numpy as np
        >>> list_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> new_shape = (2, 2, 3)
        >>> newlist = list_reshape(list_, new_shape)
        >>> depth = ut.depth_profile(newlist)
        >>> result = ('list_ = %s' % (ut.repr2(newlist, nl=1),))
        >>> print('depth = %r' % (depth,))
        >>> print(result)
        >>> newlist2 = np.reshape(list_, depth).tolist()
        >>> ut.assert_eq(newlist, newlist2)
    """
    if not trail:
        total = reduce(operator.mul, new_shape)
        assert total == len(list_)
    newlist = list_
    for dim in reversed(new_shape):
        slice_ = (newlist[i::dim] for i in range(dim))
        newlist = list(map(list, zip(*slice_)))
    if not trail:
        newlist = newlist[0]
    return newlist


def index_to_boolmask(indices, maxval=None):
    r"""
    Constructs a list of booleans where an item is True if its position is in
    `indices` otherwise it is False.

    Args:
        indices (list): list of integer indices
        maxval (int): length of the returned list. If not specified
            this is inverred from `indices`

    Returns:
        list: mask: list of booleans. mask[idx] is True if idx in indices

    SeeAlso:
        vt.index_to_boolmask numpy version

    CommandLine:
        python -m vtool.other index_to_boolmask

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> indices = [0, 1, 4]
        >>> maxval = 5
        >>> mask = ut.index_to_boolmask(indices, maxval)
        >>> assert mask == [True, True, False, False, True]
    """
    if maxval is None:
        indices = list(indices)
        maxval = max(indices) + 1
    mask = [False] * maxval
    for index in indices:
        mask[index] = True
    return mask


# Aliases
list_compress = compress
list_ziptake = ziptake
list_zipcompress = zipcompress
list_where = where
list_take = take
list_argsort = argsort
bzip = broadcast_zip


def list_strip(list_, to_strip, left=True, right=True):
    """
    list_ = [1, 2, 1, 3, 1, 1]
    to_strip = 1
    stripped_list = ut.list_strip(list_, to_strip)
    """
    import utool as ut
    flags = [item != to_strip for item in list_]
    flag_lists = []
    if right:
        rstrip_flags = ut.cumsum(flags[::-1])[::-1]
        flag_lists.append(rstrip_flags)
    if left:
        lstrip_flags = ut.cumsum(flags)
        flag_lists.append(lstrip_flags)
    strip_flags = ut.and_lists(*flag_lists)
    stripped_list = ut.compress(list_, strip_flags)
    return stripped_list


def insert_values(list_, index, values, inplace=False):
    if inplace:
        assert False
        del list_[index]
        for new in values[::-1]:
            list_.insert(index, new)
    else:
        left_part = list_[:index]
        right_part = list_[index + 1:]
        new_list = left_part + values + right_part
    return new_list


def aslist(sequence):
    r"""
    Ensures that the sequence object is a Python list.
    Handles, numpy arrays, and python sequences (e.g. tuples, and iterables).

    Args:
        sequence (sequence): a list-like object

    Returns:
        list: list_ - `sequence` as a Python list

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> s1 = [1, 2, 3]
        >>> s2 = (1, 2, 3)
        >>> assert aslist(s1) is s1
        >>> assert aslist(s2) is not s2
        >>> aslist(np.array([[1, 2], [3, 4], [5, 6]]))
        [[1, 2], [3, 4], [5, 6]]
        >>> aslist(range(3))
        [0, 1, 2]
    """
    if isinstance(sequence, list):
        return sequence
    elif util_type.HAVE_NUMPY and isinstance(sequence, np.ndarray):
        list_ = sequence.tolist()
    else:
        list_ = list(sequence)
    return list_


#def partition2(list_, idxs1, idxs2):
#    list1_ = ut.take(list_, idxs1)
#    list2_ = list(zip(ut.take(list_, idxs2)))
#    partitioned_items = [list1_, list2_]
#    return partitioned_items


#class ListLike(object):
#    """
#    similar to DictLike
#    """

#    def getitem(self, index):
#        raise NotImplementedError('abstract getitem function')

#    def setitem(self, index, value):
#        raise NotImplementedError('abstract setitem function')

#    def append(self, value):
#        raise NotImplementedError('Unfinished')

#    def insert(self, index, value):
#        raise NotImplementedError('Unfinished')

#    def extend(self, values):
#        raise NotImplementedError('Unfinished')

#    def pop(self, value):
#        raise NotImplementedError('Unfinished')

#    def remove(self, value):
#        raise NotImplementedError('Unfinished')

#    def reverse(self, value):
#        raise NotImplementedError('Unfinished')

#    def sort(self, value):
#        raise NotImplementedError('Unfinished')

#    def aslist(self):
#        return [self[idx] for idx in range(len(self))]

#    def __repr__(self):
#        return repr(self.aslist())

#    def __str__(self):
#        return str(self.aslist())

#    def __len__(self):
#        return len(list(self.keys()))

#    def __contains__(self, key):
#        return key in self.aslist()

#    def __getitem__(self, key):
#        return self.getitem(key)

#    def __setitem__(self, key, value):
#        return self.setitem(key, value)


def length_hint(obj, default=0):
    """
    Return an estimate of the number of items in obj.

    This is the PEP 424 implementation.
    If the object supports len(), the result will be
    exact. Otherwise, it may over- or under-estimate by an
    arbitrary amount. The result will be an integer >= 0.
    """
    try:
        return len(obj)
    except TypeError:
        try:
            get_hint = type(obj).__length_hint__
        except AttributeError:
            return default
        try:
            hint = get_hint(obj)
        except TypeError:
            return default
        if hint is NotImplemented:
            return default
        if not isinstance(hint, int):
            raise TypeError("Length hint must be an integer, not %r" %
                            type(hint))
        if hint < 0:
            raise ValueError("__length_hint__() should return >= 0")
        return hint


def partial_order(list_, part):
    list_items = set(list_)
    part_items = set(part)
    begin = [p for p in part if p in list_items]
    end = [item for item in list_ if item not in part_items]
    return begin + end


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_list; utool.doctest_funcs(utool.util_list, allexamples=True)"
        python -c "import utool, utool.util_list; utool.doctest_funcs(utool.util_list)"
        python -m utool.util_list
        python -m utool.util_list --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
