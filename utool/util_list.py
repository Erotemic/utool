from __future__ import absolute_import, division, print_function
import sys
from six.moves import zip, map, zip_longest, range
from .util_iter import iflatten, isiterable, ifilter_Nones, ifilter_items, ifilterfalse_items
from .util_inject import inject
from .util_str import get_callable_name
from ._internal.meta_util_six import get_funcname, set_funcname
print, print_, printDBG, rrr, profile = inject(__name__, '[list]')

USE_ASSERT = not ('--no-assert' in sys.argv)


# --- List Allocations ---

def alloc_lists(num_alloc):
    """ allocates space for a list of lists """
    return [[] for _ in range(num_alloc)]


def alloc_nones(num_alloc):
    """ allocates space for a list of Nones """
    return [None for _ in range(num_alloc)]


def ensure_list_size(list_, size_):
    """ extend list length to size_ """
    lendiff = (size_) - len(list_)
    if lendiff > 0:
        extension = [None for _ in range(lendiff)]
        list_.extend(extension)


# --- List Searching --- #

def get_list_column(list_, colx):
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


def listfind(list_, tofind):
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
    """ <CYTHE> """
    return list(iflatten(list_))


def assert_unflat_level(unflat_list, level=1, basetype=None):
    if USE_ASSERT:
        return
    num_checked = 0
    for item in unflat_list:
        if level == 1:
            for x in item:
                num_checked += 1
                assert not isinstance(x, (tuple, list)), \
                    'list is at an unexpected unflat level, x=%r' % (x,)
                if basetype is not None:
                    assert isinstance(x, basetype), \
                        'x=%r, type(x)=%r is not basetype=%r' % (x, type(x), basetype)
        else:
            assert_unflat_level(item, level - 1)
    #print('checked %r' % num_checked)
    #assert num_checked > 0, 'num_checked=%r' % num_checked


def invertable_flatten(unflat_list):
    """
    Flattens list but remember how to reconstruct the unflat list
    Returns flat list and the unflat list with indexes into the flat list
    </CYTHE> """

    def nextnum(trick_=[0]):
        num = trick_[0]
        trick_[0] += 1
        return num
    # Build an unflat list of flat indexes
    reverse_list = [tuple([nextnum() for _ in tup]) for tup in unflat_list]
    flat_list = flatten(unflat_list)
    return flat_list, reverse_list


def unflatten(flat_list, reverse_list):
    """ Rebuilds unflat list from invertable_flatten
    </CYTHE> """
    unflat_list2 = [tuple([flat_list[index] for index in tup]) for tup in reverse_list]
    return unflat_list2


def tuplize(list_):
    """ Converts each scalar item in a list to a dimension-1 tuple
    </CYTHE> """
    tup_list = [item if isiterable(item) else (item,) for item in list_]
    return tup_list


def flattenize(list_):
    """ maps flatten to a tuplized list
    list_ = [[1, 2, 3], [2, 3, [4, 2, 1]], [3, 2], [[1, 2], [3, 4]]]
    #
    # test flattenize
    import utool
    from itertools import zip
    val_list1 = [(1, 2), (2, 4), (5, 3)]
    id_list1  = [(1,),     (2,),   (3,)]
    out_list1 = utool.flattenize(zip(val_list1, id_list1))

    val_list2 = [1, 4, 5]
    id_list2  = [(1,),     (2,),   (3,)]
    out_list2 = utool.flattenize(zip(val_list2, id_list2))

    val_list3 = [1, 4, 5]
    id_list3  = [1, 2, 3]
    out_list3 = utool.flattenize(zip(val_list3, id_list3))

    out_list4 = list(zip(val_list3, id_list3))
    %timeit utool.flattenize(zip(val_list1, id_list1))
    %timeit utool.flattenize(zip(val_list2, id_list2))
    %timeit utool.flattenize(zip(val_list3, id_list3))
    %timeit list(zip(val_list3, id_list3))

    100000 loops, best of 3: 14 us per loop
    100000 loops, best of 3: 16.5 us per loop
    100000 loops, best of 3: 18 us per loop
    1000000 loops, best of 3: 1.18 us per loop
    </CYTHE> """

    #return map(iflatten, list_)
    tuplized_iter   = map(tuplize, list_)
    flatenized_list = list(map(flatten, tuplized_iter))
    return flatenized_list


def safe_slice(list_, *args):
    """ safe_slice(list_, [start], stop, [end], [step])
        Slices list and truncates if out of bounds
    </CYTHE> """
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


def list_eq(list_):
    # checks to see if list is equal everywhere
    if len(list_) == 0:
        return True
    item0 = list_[0]
    return all([item == item0 for item in list_])


def assert_all_not_None(list_, list_name='some_list', key_list=[]):
    if USE_ASSERT:
        return
    if any([item is None for count, item in enumerate(list_)]):
        msg = ((list_name + '[%d] = %r') % (count, item))
        ex = AssertionError(msg)
        from .util_dbg import printex
        printex(ex, msg, key_list=key_list, N=1)
        raise ex


def get_dirty_items(item_list, flag_list):
    """ Returns each item in item_list where not flag in flag_list
    </CYTHE> """
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
    </CYTHE> """

    assert len(item_list) == len(flag_list)
    filtered_items = list(ifilter_items(item_list, flag_list))
    return filtered_items


def filterfalse_items(item_list, flag_list):
    """
    Returns items in item list where the corresponding item in flag list is true
    </CYTHE> """
    assert len(item_list) == len(flag_list)
    filtered_items = list(ifilterfalse_items(item_list, flag_list))
    return filtered_items


def filter_Nones(list_):
    """ Removes any nones from the list </CYTHE> """
    return list(ifilter_Nones(list_))


# --- List combinations --- #


def intersect_ordered(list1, list2):
    """
    returns list1 elements that are also in list2 preserves order of list1
    </CYTHE> """
    set2 = set(list2)
    new_list = [item for item in iter(list1) if item in set2]
    #new_list =[]
    #for item in iter(list1):
    #    if item in set2:
    #        new_list.append(item)
    return new_list


def flag_unique_items(list_):
    seen = set()
    def unseen(item):
        if item in seen:
            return False
        seen.add(item)
        return True
    flag_list = [unseen(item) for item in list_]
    return flag_list


def unique_keep_order2(list_):
    """ pure python version """
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
    return tuple(set(list_))


def sortedby(list_, sortable, reverse=False):
    """ Returns a list sorted by the values of another list
    </CYTHE>
    >>> import utool
    >>> list_    = [1, 2, 3, 4, 5]
    >>> sortable = [2, 5, 3, 1, 5]
    >>> utool.sortedby(list_, sortable, reverse=True)
    [5, 2, 3, 1, 4]
    """
    assert len(list_) == len(sortable), 'must be same len'
    sorted_list = [item for (key, item) in sorted(list(zip(sortable, list_)), reverse=reverse)]
    return sorted_list


def scalar_input_map(func, input_):
    """
    If input_ is iterable this function behaves like map
    otherwise applies func to input
    </CYTHE>
    """
    if isiterable(input_):
        return list(map(func, input_))
    else:
        return func(input_)


def partial_imap_1to1(func, si_func):
    """ a bit messy """
    from functools import wraps
    @wraps(si_func)
    def wrapper(input_):
        if not isiterable(input_):
            return func(si_func(input_))
        else:
            return list(map(func, si_func(input_)))
    set_funcname(wrapper, get_callable_name(func) + '_mapper_' + get_funcname(si_func))
    return wrapper


def sample_zip(items_list, num_samples, allow_overflow=False, per_bin=1):
    """ Given a list of lists, samples one item for each list and bins them into
    num_samples bins. If all sublists are of equal size this is equivilent to a
    zip, but otherewise consecutive bins will have monotonically less
    elemements

    # Doctest doesn't work with assertionerror
    #util_list.sample_zip(items_list, 2)
    #...
    #AssertionError: Overflow occured
    </CYTHE>

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
