from __future__ import absolute_import, division, print_function
import operator
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
import six
import itertools
from six.moves import zip, map, zip_longest, range, filter
from utool import util_iter
from utool import util_inject
from utool.util_str import get_callable_name
from utool import util_type
from utool._internal.meta_util_six import get_funcname, set_funcname
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[list]')


# --- List Allocations ---

def lmap(func, iter_):
    """
    list map - eagerly evaulates map like in python2
    (but you aren't using that right?)
    """
    return list(map(func, iter_))


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

def get_list_column(list_, colx):
    r"""
    accepts a list of (indexables) and returns a list of indexables
    can also return a list of list of indexables if colx is a list

    Args:
        list_ (list):  list of lists
        colx (int or list): index or key in each sublist get item

    Returns:
        list: list of selected items

    CommandLine:
        python -m utool.util_list --test-get_list_column

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [['a', 'b'], ['c', 'd']]
        >>> colx = 0
        >>> result = get_list_column(list_, colx)
        >>> print(result)
        ['a', 'c']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [['a', 'b'], ['c', 'd']]
        >>> colx = [1, 0]
        >>> result = get_list_column(list_, colx)
        >>> print(result)
        [['b', 'a'], ['d', 'c']]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [{'spam': 'EGGS', 'ham': 'SPAM'}, {'spam': 'JAM', 'ham': 'PRAM'},]
        >>> # colx can be a key or list of keys as well
        >>> colx = ['spam']
        >>> result = get_list_column(list_, colx)
        >>> print(result)
        [['EGGS'], ['JAM']]
    """
    return list(util_iter.iget_list_column(list_, colx))
    #if isinstance(colx, list):
    #    # multi select
    #    return [[row[colx_] for colx_ in colx] for row in list_]
    #else:
    #    return [row[colx] for row in list_]


#def get_list_row(list_, rowx):
#    return list_[rowx]


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
        >>> result = ut.list_str(result_list)
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
    num_ = min(len(list_), num)
    if fromback:
        sublist = list_[-num_:]
    else:
        sublist = list_[:num_]
    return sublist


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

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = ['a', 'b', 'c']
        >>> tofind = 'd'
        >>> result = listfind(list_, tofind)
        >>> print(result)
        None
    """
    try:
        return list_.index(tofind)
    except ValueError:
        return None


# --- List Modification --- #

def list_replace(instr, search_list=[], repl_list=None):
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

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [['a', 'b'], ['c', 'd']]
        >>> unflat_list2 = flatten(list_)
        >>> result = str(unflat_list2)
        >>> print(result)
        ['a', 'b', 'c', 'd']
    """
    return list(util_iter.iflatten(list_))


def invertible_flatten(unflat_list):
    """
    Flattens ``list`` but remember how to reconstruct the unflat ``list``
    Returns flat ``list`` and the unflat ``list`` with indexes into the flat
    ``list``

    Args:
        unflat_list (list): list of nested lists that we will flatten.

    Returns:
        tuple : (flat_list, reverse_list)
    """

    def nextnum(trick_=[0]):
        num = trick_[0]
        trick_[0] += 1
        return num
    # Build an unflat list of flat indexes
    reverse_list = [[nextnum() for _ in tup] for tup in unflat_list]
    flat_list = flatten(unflat_list)
    return flat_list, reverse_list


@profile
def unflatten(flat_list, reverse_list):
    """ Rebuilds unflat list from invertible_flatten

    Args:
        flat_list (list): the flattened list
        reverse_list (list): the list which undoes flattenting

    Returns:
        unflat_list2: original nested list


    SeeAlso:
        invertible_flatten
        invertible_flatten2
        unflatten2

    """
    unflat_list2 = [[flat_list[index] for index in tup] for tup in reverse_list]
    return unflat_list2


@profile
def accumulate(iterator):
    """
    Notice:
        use itertools.accumulate in python > 3.2
    """
    total = 0
    for item in iterator:
        total += item
        yield total


@profile
def invertible_flatten2(unflat_list):
    """
    An alternative to invertible_flatten which uses cumsum

    Flattens ``list`` but remember how to reconstruct the unflat ``list``
    Returns flat ``list`` and the unflat ``list`` with indexes into the flat
    ``list``

    Args:
        unflat_list (list):

    Returns:
        tuple: flat_list, cumlen_list

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

    SeeAlso:
        invertible_flatten
        unflatten
        unflatten2

    Timeits:
        import utool
        unflat_list = aids_list1
        flat_aids1, reverse_list = utool.invertible_flatten(unflat_list)
        flat_aids2, cumlen_list = utool.invertible_flatten2(unflat_list)
        unflat_list1 = utool.unflatten(flat_aids1, reverse_list)
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
        %timeit utool.invertible_flatten(unflat_list)
        %timeit utool.unflatten(flat_aids1, reverse_list)
        print('Timings 2:)
        %timeit utool.invertible_flatten2(unflat_list)
        %timeit utool.unflatten2(flat_aids2, cumlen_list)
    """
    sublen_list = list(map(len, unflat_list))
    if not HAS_NUMPY:
        cumlen_list = np.cumsum(sublen_list)
        # Build an unflat list of flat indexes
    else:
        cumlen_list = list(accumulate(sublen_list))
    flat_list = flatten(unflat_list)
    return flat_list, cumlen_list


@profile
def invertible_flatten2_numpy(unflat_arrs):
    """ more numpy version """
    sublen_list = [arr.shape[0] for arr in unflat_arrs]
    cumlen_list = np.cumsum(sublen_list)
    flat_list = np.hstack(unflat_arrs)
    return flat_list, cumlen_list


def unflatten2(flat_list, cumlen_list):
    """ Rebuilds unflat list from invertible_flatten

    Args:
        flat_list (list): the flattened list
        cumlen_list (list): the list which undoes flattenting

    Returns:
        unflat_list2: original nested list


    SeeAlso:
        invertible_flatten
        invertible_flatten2
        unflatten2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import numpy as np
        >>> import utool
        >>> utool.util_list
        >>> flat_list = [5, 2, 3, 12, 3, 3, 9, 13, 3, 5]
        >>> cumlen_list = np.array([ 1,  6,  7,  9, 10])
        >>> unflat_list2 = unflatten2(flat_list, cumlen_list)
        >>> result = (unflat_list2)
        >>> print(result)
        [[5], [2, 3, 12, 3, 3], [9], [13, 3], [5]]
    """
    unflat_list2 = [flat_list[low:high] for low, high in zip(itertools.chain([0], cumlen_list), cumlen_list)]
    return unflat_list2


@profile
def unflat_unique_rowid_map(func, unflat_rowids, **kwargs):
    """
    performs only one call to the underlying func with unique rowids the func
    must be some lookup function

    TODO: move this to a better place.

    CommandLine:
        python -m utool.util_list --test-unflat_unique_rowid_map:0
        python -m utool.util_list --test-unflat_unique_rowid_map:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> from utool.util_list import *  # NOQA
        >>> kwargs = {}
        >>> unflat_rowids = [[1, 2, 3], [2, 5], [1], []]
        >>> num_calls0 = [0]
        >>> num_input0 = [0]
        >>> def func0(rowids, num_calls0=num_calls0, num_input0=num_input0):
        ...    num_calls0[0] += 1
        ...    num_input0[0] += len(rowids)
        ...    return [rowid + 10 for rowid in rowids]
        >>> func = func0
        >>> unflat_vals = unflat_unique_rowid_map(func, unflat_rowids, **kwargs)
        >>> result = [arr.tolist() for arr in unflat_vals]
        >>> print(result)
        >>> ut.assert_eq(num_calls0[0], 1)
        >>> ut.assert_eq(num_input0[0], 4)
        [[11, 12, 13], [12, 15], [11], []]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> from utool.util_list import *  # NOQA
        >>> kwargs = {}
        >>> unflat_rowids = [[1, 2, 3], [2, 5], [1], []]
        >>> num_calls1 = [0]
        >>> num_input1 = [0]
        >>> def func1(rowids, num_calls1=num_calls1, num_input1=num_input1):
        ...    num_calls1[0] += 1
        ...    num_input1[0] += len(rowids)
        ...    return [np.array([rowid + 10, rowid, 3]) for rowid in rowids]
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


def tuplize(list_):
    """ Converts each scalar item in a list to a dimension-1 tuple
    """
    tup_list = [item if util_iter.isiterable(item) else (item,) for item in list_]
    return tup_list


def flattenize(list_):
    """ maps flatten to a tuplized list

    Weird function. DEPRICATE

    Example:
        >>> list_ = [[1, 2, 3], [2, 3, [4, 2, 1]], [3, 2], [[1, 2], [3, 4]]]
        >>> import utool
        >>> from itertools import zip
        >>> val_list1 = [(1, 2), (2, 4), (5, 3)]
        >>> id_list1  = [(1,),     (2,),   (3,)]
        >>> out_list1 = utool.flattenize(zip(val_list1, id_list1))

        >>> val_list2 = [1, 4, 5]
        >>> id_list2  = [(1,),     (2,),   (3,)]
        >>> out_list2 = utool.flattenize(zip(val_list2, id_list2))

        >>> val_list3 = [1, 4, 5]
        >>> id_list3  = [1, 2, 3]
        >>> out_list3 = utool.flattenize(zip(val_list3, id_list3))

        out_list4 = list(zip(val_list3, id_list3))
        %timeit utool.flattenize(zip(val_list1, id_list1))
        %timeit utool.flattenize(zip(val_list2, id_list2))
        %timeit utool.flattenize(zip(val_list3, id_list3))
        %timeit list(zip(val_list3, id_list3))

        100000 loops, best of 3: 14 us per loop
        100000 loops, best of 3: 16.5 us per loop
        100000 loops, best of 3: 18 us per loop
        1000000 loops, best of 3: 1.18 us per loop
    """

    #return map(iflatten, list_)
    #if not isiterable(list_):
    #    list2_ = (list_,)
    #else:
    #    list2_ = list_
    tuplized_iter   = list(map(tuplize, list_))
    flatenized_list = list(map(flatten, tuplized_iter))
    return flatenized_list


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


def list_allsame(list_):
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
    return list_all_eq_to(list_, first_item)


def list_all_eq_to(list_, val):
    """
    checks to see if list is equal everywhere to a value

    Args:
        list_ (list):
        val : value to check against

    Returns:
        True if all items in the list are equal to val
    """
    if HAS_NUMPY and isinstance(val, np.ndarray):
        return all([np.all(item == val) for item in list_])
    return all([item == val for item in list_])


def flag_None_items(list_):
    return [item is None for item in list_]


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


def filter_items(item_list, flag_list):
    """
    Returns items in item list where the corresponding item in flag list is true

    Args:
        item_list (list):
        flag_list (list):

    Returns:
        filtered_items

    SeeAlso:
        util_iter.ifilter_items
    """

    assert len(item_list) == len(flag_list)
    filtered_items = list(util_iter.ifilter_items(item_list, flag_list))
    return filtered_items


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


def intersect_ordered(list1, list2):
    """
    returns list1 elements that are also in list2. preserves order of list1

    intersect_ordered

    Args:
        list1 (list):
        list2 (list):

    Returns:
        list: new_list

    Example:
        >>> from utool.util_list import *  # NOQA
        >>> list1 = ['featweight_rowid', 'feature_rowid', 'config_rowid', 'featweight_forground_weight']
        >>> list2 = [u'featweight_rowid']
        >>> result = intersect_ordered(list1, list2)
        >>> print(result)
        ['featweight_rowid']
    """
    return [item for item in list1 if item in set(list2)]


def list_intersection(list1, list2):
    return set(list1).intersection(set(list2))


def list_issubset(list1, list2):
    return set(list1).issubset(set(list2))


def list_issuperset(list1, list2):
    return set(list1).issuperset(set(list2))


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


def setdiff_ordered(list1, list2):
    """
    returns list1 elements that are not in list2. preserves order of list1

    setdiff_ordered

    Args:
        list1 (list):
        list2 (list):

    Returns:
        list: new_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list1 = ['featweight_rowid', 'feature_rowid', 'config_rowid', 'featweight_forground_weight']
        >>> list2 = [u'featweight_rowid']
        >>> new_list = setdiff_ordered(list1, list2)
        >>> result = new_list
        >>> print(result)
        ['feature_rowid', 'config_rowid', 'featweight_forground_weight']
    """
    return [item for item in list1 if item not in set(list2)]


def flag_unique_items(list_):
    """
    Returns a list of flags corresponding to the first time an item is seen

    Args:
        list_ (list): list of items

    Returns:
        flag_list
    """
    seen = set()
    def unseen(item):
        if item in seen:
            return False
        seen.add(item)
        return True
    flag_list = [unseen(item) for item in list_]
    return flag_list


def unique_keep_order2(list_):
    """
    pure python version of unique_keep_ordered

    Args:
        list_ (list):

    Returns:
        unique_list : unique list which maintains order
    """
    seen = set()
    def unseen(item):
        if item in seen:
            return False
        seen.add(item)
        return True
    unique_list = [item for item in list_ if unseen(item)]
    return unique_list

unique_ordered = unique_keep_order2


def unique_unordered(list_):
    return list(set(list_))


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

    Examples:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> list_    = [1, 2, 3, 4, 5]
        >>> key_list = [2, 5, 3, 1, 5]
        >>> result = utool.sortedby(list_, key_list, reverse=True)
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

    Varargs:
        *args (list): multiple lists to sort by

    Kwargs:
        reverse (bool): sort order is descending if True else acscending

    Returns:
        list : ``list_`` sorted by the values of another ``list``. defaults to
        ascending order

    Examples:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> item_list = [1, 2, 3, 4, 5]
        >>> key_list1 = [1, 1, 2, 3, 4]
        >>> key_list2 = [2, 1, 4, 1, 1]
        >>> args = (key_list1, key_list2)
        >>> kwargs = dict(reverse=False)
        >>> result = utool.sortedby2(item_list, *args, **kwargs)
        >>> print(result)
        [2, 1, 3, 4, 5]

    """
    import operator
    assert all([len(item_list) == len_ for len_ in map(len, args)])
    reverse = kwargs.get('reverse', False)
    key = operator.itemgetter(*range(1, len(args) + 1))
    tup_list = list(zip(item_list, *args))
    #print(tup_list)
    sorted_tups = sorted(tup_list, key=key, reverse=reverse)
    sorted_list = [tup[0] for tup in sorted_tups]
    return sorted_list


def list_argsort(*args, **kwargs):
    """ like np.argsort but for lists

    Varargs:
        *args (list): multiple lists to sort by

    Kwargs:
        reverse (bool): sort order is descending if True else acscending
    """
    index_list = list(range(len(args[0])))
    return sortedby2(index_list, *args, **kwargs)


def list_compress(list_, flag_list):
    """ like np.compress but for lists """
    return filter_items(list_, flag_list)


def list_take(list_, index_list):
    """ like np.take but for lists

    Args:
        list_ (list):
        index_list (list):

    Returns:
        list or scalar:

    CommandLine:
        python -m utool.util_list --test-list_take

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [0, 1, 2, 3]
        >>> index_list = [2, 0]
        >>> result = list_take(list_, index_list)
        >>> print(result)
        [2, 0]

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [0, 1, 2, 3]
        >>> index = 2
        >>> result = list_take(list_, index)
        >>> print(result)
        2
    """
    try:
        return [list_[index] for index in index_list]
    except TypeError:
        return list_[index_list]
    #if util_iter.isiterable(index_list):
    #else:


def list_inverse_take(list_, index_list):
    r"""
    Args:
        list_ (list): list in sorted domain
        index_list (list): index list of the unsorted domain

    Note:
        Seems to be logically equivalent to
        ut.list_take(list_, ut.list_argsort(index_list)), but faster

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
        >>> index_list = ut.list_argsort(rank_list)
        >>> sorted_prop_list = ut.list_take(prop_list, index_list)
        >>> # execute function
        >>> list_ = sorted_prop_list
        >>> output_list_  = list_inverse_take(list_, index_list)
        >>> output_list2_ = ut.list_take(list_, ut.list_argsort(index_list))
        >>> assert output_list_ == prop_list
        >>> assert output_list2_ == prop_list
        >>> # verify results
        >>> result = str(output_list_)
        >>> print(result)

    Timeit::
        %timeit list_inverse_take(list_, index_list)
        %timeit ut.list_take(list_, ut.list_argsort(index_list))
    """
    output_list_ = [None] * len(index_list)
    for item, index in zip(list_, index_list):
        output_list_[index] = item
    return output_list_


def list_where(flag_list):
    """ takes flags returns indexes of True values """
    return [index for index, flag in enumerate(flag_list) if flag]


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
    """ a bit messy """
    from functools import wraps
    @wraps(si_func)
    def wrapper(input_):
        if not util_iter.isiterable(input_):
            return func(si_func(input_))
        else:
            return list(map(func, si_func(input_)))
    set_funcname(wrapper, get_callable_name(func) + '_mapper_' + get_funcname(si_func))
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
    if seed is not None:
        np.random.seed(seed)
    def random_choice(items, num):
        size = min(len(items), num)
        return np.random.choice(items, size, replace=False).tolist()
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
    Args:
        list_ (list):
        op (builtin_function_or_method):

    Returns:
        bool : True if the list is sorted
    """
    return all(op(list_[ix], list_[ix + 1]) for ix in range(len(list_) - 1))


def find_nonconsec_indices(unique_vals, consec_vals):
    """
    # TODO: rectify with above function

    Args:
        unique_vals (list):
        consec_vals (list):

    Returns:
        missing_ixs

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import numpy as np
        >>> unique_vals = np.array([-2, -1,  1,  2, 10])
        >>> max_ = unique_vals.max()
        >>> min_ = unique_vals.min()
        >>> range_ = max_ - min_
        >>> consec_vals = np.linspace(min_, max_ + 1, range_ + 2)
        >>> missing_ixs = find_nonconsec_indices(unique_vals, consec_vals)
        >>> result = (consec_vals[missing_ixs])
        [ 0.  3.  4.  5.  6.  7.  8.  9.]
    """
    missing_ixs = []
    valx   = 0
    consecx = 0
    while valx < len(unique_vals) and consecx < len(consec_vals):
        if unique_vals[valx] != consec_vals[consecx]:
            missing_ixs.append(consecx)
        else:
            valx += 1
        consecx += 1
    return missing_ixs


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


def find_duplicate_items(items):
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

    print('[util_list] +--- DEBUG DUPLICATE ITEMS  %r ---' % ut.get_varname_from_locals(items, ut.get_caller_locals()))
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


def depth_profile(list_, max_depth=None, compress_homogenous=True):
    """
    Returns a nested list corresponding the shape of the nested structures
    lists represent depth, tuples represent shape. The values of the items do
    not matter. only the lengths.

    CommandLine:
        python -m utool.util_list --test-depth_profile

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
        >>> result = depth_profile(list_)
        >>> print(result)
        (2, 3, 4)

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [[[[[1]]], [3, 4, 33]], [[1], [2, 3], [4, [5, 5]]], [1, 3]]
        >>> result = depth_profile(list_)
        >>> print(result)
        [[(1, 1, 1), 3], [1, 2, [1, 2]], 2]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [[[[[1]]], [3, 4, 33]], [[1], [2, 3], [4, [5, 5]]], [1, 3]]
        >>> result = depth_profile(list_, 1)
        >>> print(result)
        [[(1, '1'), 3], [1, 2, [1, '2']], 2]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [[[1, 2], [1, 2, 3]], None]
        >>> result = depth_profile(list_, compress_homogenous=True)
        >>> print(result)
        [[2, 3], 1]

    """
    level_shape_list = []
    # For a pure bottom level list return the length
    if not any(map(util_type.is_listlike, list_)):
        return len(list_)
    for item in list_:
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
        if list_allsame(level_shape_list):
            dim_ = level_shape_list[0]
            len_ = len(level_shape_list)
            if isinstance(dim_, tuple):
                level_shape_list = tuple([len_] + list(dim_))
            else:
                level_shape_list = tuple([len_, dim_])

    return level_shape_list


def list_type_profile(sequence, compress_homogenous=True):
    """ similar to depth_profile but reports types """
    # For a pure bottom level list return the length
    #if not any(map(util_type.is_listlike, sequence)) or (isinstance(sequence, np.ndarray) and sequence.dtype != object):
    if not util_type.is_listlike(sequence) or (isinstance(sequence, np.ndarray) and sequence.dtype != object):
        typename = str(type(sequence)).replace('<type \'', '').replace('\'>', '')
        level_type_str = typename
        return level_type_str

    level_type_list = []
    for item in sequence:
        #if util_type.is_listlike(item):
        level_type_list.append(list_type_profile(item))

    if compress_homogenous:
        # removes redudant information by returning a type and number
        if list_allsame(level_type_list):
            type_ = level_type_list[0]
            level_type_str = str(type_) + '*' + str(len(level_type_list))
        else:
            level_type_str = ', '.join(level_type_list)
    typename = str(type(sequence)).replace('<type \'', '').replace('\'>', '')
    level_type_str = typename + '(' + str(level_type_str) + ')'
    return level_type_str


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
    sortxs_ = list_argsort(item_list)
    sortxs = sortxs_[::-1] if reverse else sortxs_
    def sortby_func(list_):
        return list_take(list_, sortxs)
    return sortby_func


def find_first_true_indices(flags_list):
    """
    returns a list of indexes where the index is the first True position
    in the corresponding sublist or None if it does not exist

    in other words: for each row finds the smallest True column number or None

    Args:
        flags_list (list): list of lists of booleans

    CommandLine:
        python -m utool.util_list --test-find_first_true_indices

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> # build test data
        >>> flags_list = [[True, False, True],
        ...               [False, False, False],
        ...               [False, True, True],
        ...               [False, False, True]]
        >>> # execute function
        >>> index_list = find_first_true_indices(flags_list)
        >>> # verify results
        >>> result = str(index_list)
        >>> print(result)
        [0, None, 1, 2]
    """
    def tryget_fisrt_true(flags):
        index_list = np.where(flags)[0]
        index = None if len(index_list) == 0 else index_list[0]
        return index
    index_list = [tryget_fisrt_true(flags) for flags in flags_list]
    return index_list


def find_next_true_indices(flags_list, offset_list):
    """
    Uses output of either this function or find_first_true_indices
    to find the next index of true flags

    Args:
        flags_list (list): list of lists of booleans

    CommandLine:
        python -m utool.util_list --test-find_next_true_indices

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> # build test data
        >>> flags_list = [[True, False, True],
        ...               [False, False, False],
        ...               [False, True, True],
        ...               [False, False, True]]
        >>> offset_list = find_first_true_indices(flags_list)
        >>> # execute function
        >>> index_list = find_next_true_indices(flags_list, offset_list)
        >>> # verify results
        >>> result = str(index_list)
        >>> print(result)
        [2, None, 2, None]
    """
    def tryget_next_true(flags, offset_):
        offset = offset_ + 1
        relative_flags = flags[offset:]
        rel_index_list = np.where(relative_flags)[0]
        index = None if len(rel_index_list) == 0 else rel_index_list[0] + offset
        return index
    index_list = [None if offset is None else tryget_next_true(flags, offset)
                  for flags, offset in zip(flags_list, offset_list)]
    return index_list


def filter_startswith(list_, str_):
    def item_startswith(item):
        return item.startswith(str_)
    return list(filter(item_startswith, list_))


def list_rotate(list_, n):
    """
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
        >>> result = list_rotate(list_, n)
        >>> print(result)
        [3, 4, 5, 1, 2]
    """
    return list_[n:] + list_[:n]


def list_argmax(list_):
    return np.argmax(np.array(list_))


def make_index_lookup(list_):
    return dict(zip(list_, range(len(list_))))


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
