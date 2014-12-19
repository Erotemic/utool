from __future__ import absolute_import, division, print_function
import operator
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
import six
import itertools
from six.moves import zip, map, zip_longest, range
from utool import util_iter
from utool import util_inject
from utool.util_str import get_callable_name
from utool.util_type import is_listlike
from utool._internal.meta_util_six import get_funcname, set_funcname
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[list]')


# --- List Allocations ---

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
    Args:
        list_ (list):  list of lists
        colx (int):  position in each sublist get item

    Returns:
        list: list of selected items

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [['a', 'b'], ['c', 'd']]
        >>> colx = 0
        >>> result = get_list_column(list_, colx)
        >>> print(result)
        ['a', 'c']
    """
    return [row[colx] for row in list_]


def list_getat(list_, index_list):
    return [list_[index] for index in index_list]


def safe_listget(list_, index, default='?'):
    if index >= len(list_):
        return default
    ret = list_[index]
    if ret is None:
        return default
    return ret


def listclip(list_, index):
    r"""
    Args:
        list_ (list):
        index (int):

    Returns:
        sublist:

    CommandLine:
        python -m utool.util_list --test-listclip

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> # build test data
        >>> list_ = [1, 2, 3, 4, 5]
        >>> result_list = []
        >>> # execute function
        >>> index = 3
        >>> result_list += [ut.listclip(list_, index)]
        >>> index = 9
        >>> result_list += [ut.listclip(list_, index)]
        >>> # verify results
        >>> result = ut.list_str(result_list)
        >>> print(result)
        [
            [1, 2, 3],
            [1, 2, 3, 4, 5],
        ]
    """
    sublist = list_[:min(len(list_), index)]
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


def invertable_flatten(unflat_list):
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


def unflatten(flat_list, reverse_list):
    """ Rebuilds unflat list from invertable_flatten

    Args:
        flat_list (list): the flattened list
        reverse_list (list): the list which undoes flattenting

    Returns:
        unflat_list2: original nested list


    SeeAlso:
        invertable_flatten
        invertable_flatten2
        unflatten2

    """
    unflat_list2 = [[flat_list[index] for index in tup] for tup in reverse_list]
    return unflat_list2


def invertable_flatten2(unflat_list):
    """
    An alternative to invertable_flatten which uses cumsum

    TODO: This flatten is faster fix it to be used everywhere

    Flattens ``list`` but remember how to reconstruct the unflat ``list``
    Returns flat ``list`` and the unflat ``list`` with indexes into the flat
    ``list``

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> import utool
        >>> utool.util_list
        >>> unflat_list = [[5], [2, 3, 12, 3, 3], [9], [13, 3], [5]]
        >>> flat_list, cumlen_list = invertable_flatten2(unflat_list)
        >>> unflat_list2 = unflatten2(flat_list, cumlen_list)
        >>> assert unflat_list2 == unflat_list
        >>> result = ((flat_list, cumlen_list))
        >>> print(result)
        ([5, 2, 3, 12, 3, 3, 9, 13, 3, 5], [1, 6, 7, 9, 10])

    SeeAlso:
        invertable_flatten
        unflatten
        unflatten2

    Timeits:
        import utool
        unflat_list = aids_list1
        flat_aids1, reverse_list = utool.invertable_flatten(unflat_list)
        flat_aids2, cumlen_list = utool.invertable_flatten2(unflat_list)
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
        %timeit utool.invertable_flatten(unflat_list)
        %timeit utool.unflatten(flat_aids1, reverse_list)
        print('Timings 2:)
        %timeit utool.invertable_flatten2(unflat_list)
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


def accumulate(iterator):
    """
    Notice:
        use itertools.accumulate in python > 3.2
    """
    total = 0
    for item in iterator:
        total += item
        yield total


def unflatten2(flat_list, cumlen_list):
    """ Rebuilds unflat list from invertable_flatten

    Args:
        flat_list (list): the flattened list
        cumlen_list (list): the list which undoes flattenting

    Returns:
        unflat_list2: original nested list


    SeeAlso:
        invertable_flatten
        invertable_flatten2
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
        reverse (bool): sort order is descending if True else acscending

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
    assert len(item_list) == len(key_list), 'Expected same length. Got: %r != %r' % (
        len(item_list), len(key_list))
    sorted_list = [item for (key, item) in
                   sorted(list(zip(key_list, item_list)), reverse=reverse)]
    return sorted_list


def sortedby2(item_list, *args, **kwargs):
    """ sorts ``item_list`` using key_list

    Args:
        item_list (list): list to sort
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
    """ like np.argsort but for lists """
    index_list = list(range(len(args[0])))
    return sortedby2(index_list, *args, **kwargs)


def list_take(list_, index_list):
    """ like np.take but for lists """
    return [list_[ix] for ix in index_list]


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


def issorted(list_, op=operator.le):
    """
    Args:
        list_ (list):
        op (builtin_function_or_method):

    Returns:
        bool : True if the list is sorted
    """
    return all(op(list_[ix], list_[ix + 1]) for ix in range(len(list_) - 1))


def find_nonconsec_indicies(unique_vals, consec_vals):
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
        >>> missing_ixs = find_nonconsec_indicies(unique_vals, consec_vals)
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
        tuple of (missing_items, missing_indicies, duplicate_items)
    """
    if not issorted(list_):
        print('warning list is not sorted. indicies will not match')
    sortedlist = sorted(list_)
    start = sortedlist[0]
    last = start - 1
    missing_vals = []
    missing_indicies = []
    duplicate_items = []
    for count, item in enumerate(sortedlist):
        diff = item - last
        if diff > 1:
            missing_indicies.append(count)
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
    return missing_vals, missing_indicies, duplicate_items


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
        printfn('item=%s appears %d times at indicies: %r' % (key, len(index_list), index_list))
        for argx, arg in enumerate(args):
            #argname = 'arg%d' % (argx)
            argname = ut.get_varname_from_stack(arg, N=2)
            for index in index_list:
                printfn(' * %s[%d] = %r' % (argname, index, arg[index]))
    return duplicate_map


def debug_duplicate_items(items, *args, **kwargs):
    import utool as ut
    separate = kwargs.get('separate', True)
    if separate:
        print('')

    print('[util_list] +--- DEBUG DUPLICATE ITEMS  %r ---' % ut.get_varname_from_locals(items, ut.get_caller_locals()))
    def printfn(msg):
        print('[util_list] |' + msg)
    #with ut.Indenter('[util_list] | '):
    duplicate_map = ut.find_duplicate_items(items)
    printkw = {'printfn': printfn}
    ut.print_duplicate_map(duplicate_map, *args, **printkw)
    print('[util_list] L--- FINISH DEBUG DUPLICATE ITEMS ---')
    if separate:
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
                  for item in  list_ if is_listlike(item)]
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
        if is_listlike(item):
            type_list.extend(list_deep_types(item))
        else:
            type_list.append(type(item))
    return type_list


def depth_profile(list_):
    """
    Returns a nested list corresponding the shape of the nested structure

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [[[[[1]]], [3, 4, 33]], [[1], [3]], [[1], [3]]]
        >>> result = (depth_profile(list_))
        >>> print(result)
        [[[[1]], 3], [1, 1], [1, 1]]
    """
    level_shape_list = []
    # For a pure bottom level list return the length
    if not any(map(is_listlike, list_)):
        return len(list_)
    for item in list_:
        if is_listlike(item):
            level_shape_list.append(depth_profile(item))
    return level_shape_list


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
