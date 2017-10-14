# -*- coding: utf-8 -*-
""" convinience functions for dictionaries """
from __future__ import absolute_import, division, print_function, unicode_literals
import operator as op
import itertools as it
from collections import defaultdict, OrderedDict
from functools import partial
from six.moves import zip, range, map
from utool import util_inject
from utool import util_list
from utool import util_const
from utool import util_iter
import copy
import six
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    pass
print, rrr, profile = util_inject.inject2(__name__)


@profile
def dzip(list1, list2):
    r"""
    Zips elementwise pairs between list1 and list2 into a dictionary. Values
    from list2 can be broadcast onto list1.

    Args:
        list1 (sequence): full sequence
        list2 (sequence): can either be a sequence of one item or a sequence of
            equal length to `list1`

    SeeAlso:
        util_list.broadcast_zip

    Returns:
        dict: similar to dict(zip(list1, list2))

    CommandLine:
        python -m utool.util_dict dzip

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> assert dzip([1, 2, 3], [4]) == {1: 4, 2: 4, 3: 4}
        >>> assert dzip([1, 2, 3], [4, 4, 4]) == {1: 4, 2: 4, 3: 4}
        >>> ut.assert_raises(ValueError, dzip, [1, 2, 3], [])
        >>> ut.assert_raises(ValueError, dzip, [], [4, 5, 6])
        >>> ut.assert_raises(ValueError, dzip, [], [4])
        >>> ut.assert_raises(ValueError, dzip, [1, 2], [4, 5, 6])
        >>> ut.assert_raises(ValueError, dzip, [1, 2, 3], [4, 5])
    """
    try:
        len(list1)
    except TypeError:
        list1 = list(list1)
    try:
        len(list2)
    except TypeError:
        list2 = list(list2)
    if len(list1) == 0 and len(list2) == 1:
        # Corner case:
        # allow the first list to be empty and the second list to broadcast a
        # value. This means that the equality check wont work for the case
        # where list1 and list2 are supposed to correspond, but the length of
        # list2 is 1.
        list2 = []
    if len(list2) == 1 and len(list1) > 1:
        list2 = list2 * len(list1)
    if len(list1) != len(list2):
        raise ValueError('out of alignment len(list1)=%r, len(list2)=%r' % (
            len(list1), len(list2)))
    return dict(zip(list1, list2))


def map_dict_vals(func, dict_):
    """ applies a function to each of the keys in a dictionary

    Args:
        func (callable): a function
        dict_ (dict): a dictionary

    Returns:
        newdict: transformed dictionary

    CommandLine:
        python -m utool.util_dict --test-map_dict_vals

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {'a': [1, 2, 3], 'b': []}
        >>> func = len
        >>> newdict = map_dict_vals(func, dict_)
        >>> result = ut.repr2(newdict)
        >>> print(result)
        {'a': 3, 'b': 0}
    """
    if not hasattr(func, '__call__'):
        func = func.__getitem__
    keyval_list = [(key, func(val)) for key, val in six.iteritems(dict_)]
    dictclass = OrderedDict if isinstance(dict_, OrderedDict) else dict
    newdict = dictclass(keyval_list)
    # newdict = type(dict_)(keyval_list)
    return newdict


def map_dict_keys(func, dict_):
    """ applies a function to each of the keys in a dictionary

    Args:
        func (callable): a function
        dict_ (dict): a dictionary

    Returns:
        newdict: transformed dictionary

    CommandLine:
        python -m utool.util_dict --test-map_dict_keys

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {'a': [1, 2, 3], 'b': []}
        >>> func = ord
        >>> newdict = map_dict_keys(func, dict_)
        >>> result = ut.repr2(newdict)
        >>> ut.assert_raises(AssertionError, map_dict_keys, len, dict_)
        >>> print(result)
        {97: [1, 2, 3], 98: []}
    """
    if not hasattr(func, '__call__'):
        func = func.__getitem__
        # op.itemgetter(func)
    keyval_list = [(func(key), val) for key, val in six.iteritems(dict_)]
    # newdict = type(dict_)(keyval_list)
    dictclass = OrderedDict if isinstance(dict_, OrderedDict) else dict
    newdict = dictclass(keyval_list)
    assert len(newdict) == len(dict_), (
        'multiple input keys were mapped to the same output key')
    return newdict


map_vals = map_dict_vals
map_keys = map_dict_keys


class AutoVivification(dict):
    """
    Implementation of perl's autovivification feature.

    An AutoVivification is an infinitely nested default dict of dicts.

    References:
        http://stackoverflow.com/questions/651794/best-way-to-init-dict-of-dicts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = AutoVivification()
        >>> # Notice that there is no KeyError
        >>> dict_[0][10][100] = None
        >>> result = ('dict_ = %r' % (dict_,))
        >>> print(result)
        dict_ = {0: {10: {100: None}}}
    """
    def __getitem__(self, key):
        try:
            # value = super(AutoVivification, self).__getitem__(key)
            value = dict.__getitem__(self, key)
        except KeyError:
            value = self[key] = type(self)()
        return value


class OrderedAutoVivification(OrderedDict):
    """
    Implementation of perl's autovivification feature.

    An OrderedAutoVivification is an infinitely nested default dict of ordered
    dicts.

    References:
        http://stackoverflow.com/questions/651794/best-way-to-init-dict-of-dicts

    Doctest:
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = AutoOrderedDict()
        >>> # Notice that there is no KeyError
        >>> dict_[0][10][100] = None
        >>> dict_[0][10][1] = None
        >>> result = ('dict_ = %r' % (dict_,))
        >>> print(result)
        dict_ = {0: {10: {100: None, 1: None}}}
    """
    def __getitem__(self, key):
        try:
            # value = super(OrderedAutoVivification, self).__getitem__(key)
            value = OrderedDict.__getitem__(self, key)
        except KeyError:
            value = self[key] = type(self)()
        return value

    def __repr__(self):
        import utool as ut
        return ut.repr2(self)

    __str__ = __repr__

AutoDict = AutoVivification
AutoOrderedDict = OrderedAutoVivification


def count_dict_vals(dict_of_lists):
    count_dict = {'len(%s)' % (key,): len(val) for key, val in six.iteritems(dict_of_lists)}
    return count_dict


def dict_keysubset(dict_, keys):
    return [key for key in keys if key in dict_]


def get_dict_hashid(dict_):
    r"""
    Args:
        dict_ (dict):

    Returns:
        int: id hash

    References:
        http://stackoverflow.com/questions/5884066/hashing-a-python-dictionary

    CommandLine:
        python -m utool.util_dict --test-get_dict_hashid
        python3 -m utool.util_dict --test-get_dict_hashid

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = {}
        >>> dict_ = {'a': 'b'}
        >>> dict_ = {'a': {'c': 'd'}}
        >>> #dict_ = {'a': {'c': 'd'}, 1: 143, dict: set}
        >>> #dict_ = {'a': {'c': 'd'}, 1: 143 } non-determenism
        >>> hashid = get_dict_hashid(dict_)
        >>> result = str(hashid)
        >>> print(result)
        mxgkepoboqjerkhb

        oegknoalkrkojumi
    """
    import utool as ut
    raw_text = ut.repr4(dict_, sorted_=True, strvals=True, nl=2)
    #print('raw_text = %r' % (raw_text,))
    hashid = ut.hashstr27(raw_text)
    #from utool import util_hash
    #hashid = hash(frozenset(dict_.items()))
    #hashid = util_hash.make_hash(dict_)
    return hashid


class hashdict(dict):
    """
    hashable dict implementation, suitable for use as a key into
    other dicts.

    Example:
        >>> # DISABLE_DOCTEST
        >>> h1 = hashdict({"apples": 1, "bananas":2})
        >>> h2 = hashdict({"bananas": 3, "mangoes": 5})
        >>> h1+h2
        hashdict(apples=1, bananas=3, mangoes=5)
        >>> d1 = {}
        >>> d1[h1] = "salad"
        >>> d1[h1]
        'salad'
        >>> d1[h2]
        Traceback (most recent call last):
        ...
        KeyError: hashdict(bananas=3, mangoes=5)

    References:
       http://stackoverflow.com/questions/1151658/python-hashable-dicts
       http://stackoverflow.com/questions/1151658/python-hashable-dicts
    """
    def __key(self):
        return tuple(sorted(self.items()))

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join("{0}={1}".format(
                                     str(i[0]), repr(i[1])) for i in self.__key()))

    def __hash__(self):
        return hash(self.__key())
    def __setitem__(self, key, value):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def __delitem__(self, key):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def clear(self):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def pop(self, *args, **kwargs):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def popitem(self, *args, **kwargs):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def setdefault(self, *args, **kwargs):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def update(self, *args, **kwargs):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    # update is not ok because it mutates the object
    # __add__ is ok because it creates a new object
    # while the new object is under construction, it's ok to mutate it
    def __add__(self, right):
        result = hashdict(self)
        dict.update(result, right)
        return result


def dict_stack(dict_list, key_prefix=''):
    r"""
    stacks values from two dicts into a new dict where the values are list of
    the input values. the keys are the same.

    DEPRICATE in favor of dict_stack2

    Args:
        dict_list (list): list of dicts with similar keys

    Returns:
        dict dict_stacked

    CommandLine:
        python -m utool.util_dict --test-dict_stack
        python -m utool.util_dict --test-dict_stack:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict1_ = {'a': 1, 'b': 2}
        >>> dict2_ = {'a': 2, 'b': 3, 'c': 4}
        >>> dict_stacked = dict_stack([dict1_, dict2_])
        >>> result = ut.repr2(dict_stacked, sorted_=True)
        >>> print(result)
        {'a': [1, 2], 'b': [2, 3], 'c': [4]}

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> # Get equivalent behavior with dict_stack2?
        >>> # Almost, as long as None is not part of the list
        >>> dict1_ = {'a': 1, 'b': 2}
        >>> dict2_ = {'a': 2, 'b': 3, 'c': 4}
        >>> dict_stacked_ = dict_stack2([dict1_, dict2_])
        >>> dict_stacked = {key: ut.filter_Nones(val) for key, val in dict_stacked_.items()}
        >>> result = ut.repr2(dict_stacked, sorted_=True)
        >>> print(result)
        {'a': [1, 2], 'b': [2, 3], 'c': [4]}
    """
    dict_stacked_ = defaultdict(list)
    for dict_ in dict_list:
        for key, val in six.iteritems(dict_):
            dict_stacked_[key_prefix + key].append(val)
    dict_stacked = dict(dict_stacked_)
    return dict_stacked


def dict_stack2(dict_list, key_suffix=None, default=None):
    """
    Stacks vals from a list of dicts into a dict of lists. Inserts Nones in
    place of empty items to preserve order.

    Args:
        dict_list (list): list of dicts
        key_suffix (str): (default = None)

    Returns:
        dict: stacked_dict

    Example:
        >>> # ENABLE_DOCTEST
        >>> # Usual case: multiple dicts as input
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict1_ = {'a': 1, 'b': 2}
        >>> dict2_ = {'a': 2, 'b': 3, 'c': 4}
        >>> dict_list = [dict1_, dict2_]
        >>> dict_stacked = dict_stack2(dict_list)
        >>> result = ut.repr2(dict_stacked)
        >>> print(result)
        {'a': [1, 2], 'b': [2, 3], 'c': [None, 4]}

    Example1:
        >>> # ENABLE_DOCTEST
        >>> # Corner case: one dict as input
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict1_ = {'a': 1, 'b': 2}
        >>> dict_list = [dict1_]
        >>> dict_stacked = dict_stack2(dict_list)
        >>> result = ut.repr2(dict_stacked)
        >>> print(result)
        {'a': [1], 'b': [2]}

    Example2:
        >>> # ENABLE_DOCTEST
        >>> # Corner case: zero dicts as input
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_list = []
        >>> dict_stacked = dict_stack2(dict_list)
        >>> result = ut.repr2(dict_stacked)
        >>> print(result)
        {}

    Example3:
        >>> # ENABLE_DOCTEST
        >>> # Corner case: empty dicts as input
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_list = [{}]
        >>> dict_stacked = dict_stack2(dict_list)
        >>> result = ut.repr2(dict_stacked)
        >>> print(result)
        {}

    Example4:
        >>> # ENABLE_DOCTEST
        >>> # Corner case: one dict is empty
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict1_ = {'a': [1, 2], 'b': [2, 3]}
        >>> dict2_ = {}
        >>> dict_list = [dict1_, dict2_]
        >>> dict_stacked = dict_stack2(dict_list)
        >>> result = ut.repr2(dict_stacked)
        >>> print(result)
        {'a': [[1, 2], None], 'b': [[2, 3], None]}

    Example5:
        >>> # ENABLE_DOCTEST
        >>> # Corner case: disjoint dicts
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict1_ = {'a': [1, 2], 'b': [2, 3]}
        >>> dict2_ = {'c': 4}
        >>> dict_list = [dict1_, dict2_]
        >>> dict_stacked = dict_stack2(dict_list)
        >>> result = ut.repr2(dict_stacked)
        >>> print(result)
        {'a': [[1, 2], None], 'b': [[2, 3], None], 'c': [None, 4]}

    Example6:
        >>> # ENABLE_DOCTEST
        >>> # Corner case: 3 dicts
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_list = [{'a': 1}, {'b': 1}, {'c': 1}, {'b': 2}]
        >>> default = None
        >>> dict_stacked = dict_stack2(dict_list, default=default)
        >>> result = ut.repr2(dict_stacked)
        >>> print(result)
        {'a': [1, None, None, None], 'b': [None, 1, None, 2], 'c': [None, None, 1, None]}
    """
    if len(dict_list) > 0:
        dict_list_ = [map_dict_vals(lambda x: [x], kw) for kw in dict_list]
        # Reduce does not handle default quite correctly
        default1 = []
        default2 = [default]
        accum_ = dict_list_[0]
        for dict_ in dict_list_[1:]:
            default1.append(default)
            accum_ = dict_union_combine(accum_, dict_, default=default1,
                                        default2=default2)
        stacked_dict = accum_
        # stacked_dict = reduce(partial(dict_union_combine, default=[default]), dict_list_)
    else:
        stacked_dict = {}
    # Augment keys if requested
    if key_suffix is not None:
        stacked_dict = map_dict_keys(lambda x: x + key_suffix, stacked_dict)
    return stacked_dict


def invert_dict(dict_, unique_vals=True):
    """
    Reverses the keys and values in a dictionary. Set unique_vals to False if
    the values in the dict are not unique.

    Args:
        dict_ (dict_): dictionary
        unique_vals (bool): if False, inverted keys are returned in a list.

    Returns:
        dict: inverted_dict

    CommandLine:
        python -m utool.util_dict --test-invert_dict

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {'a': 1, 'b': 2}
        >>> inverted_dict = invert_dict(dict_)
        >>> result = ut.repr4(inverted_dict, nl=False)
        >>> print(result)
        {1: 'a', 2: 'b'}

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = OrderedDict([(2, 'good',), (1, 'ok',), (0, 'junk',), (None, 'UNKNOWN',)])
        >>> inverted_dict = invert_dict(dict_)
        >>> result = ut.repr4(inverted_dict, nl=False)
        >>> print(result)
        {'good': 2, 'ok': 1, 'junk': 0, 'UNKNOWN': None}

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {'a': 1, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 2}
        >>> inverted_dict = invert_dict(dict_, unique_vals=False)
        >>> inverted_dict = ut.map_dict_vals(sorted, inverted_dict)
        >>> result = ut.repr4(inverted_dict, nl=False)
        >>> print(result)
        {0: ['b', 'c', 'd', 'e'], 1: ['a'], 2: ['f']}
    """
    if unique_vals:
        inverted_items = [(val, key) for key, val in six.iteritems(dict_)]
        inverted_dict = type(dict_)(inverted_items)
    else:
        inverted_dict = group_items(dict_.keys(), dict_.values())
    return inverted_dict


def iter_all_dict_combinations_ordered(varied_dict):
    """
    Same as all_dict_combinations but preserves order
    """
    tups_list = [[(key, val) for val in val_list]
                 for (key, val_list) in six.iteritems(varied_dict)]
    dict_iter = (OrderedDict(tups) for tups in it.product(*tups_list))
    return dict_iter


def all_dict_combinations_ordered(varied_dict):
    """
    Same as all_dict_combinations but preserves order
    """
    dict_list = list(iter_all_dict_combinations_ordered)
    return dict_list


def all_dict_combinations(varied_dict):
    """
    all_dict_combinations

    Args:
        varied_dict (dict):  a dict with lists of possible parameter settings

    Returns:
        list: dict_list a list of dicts correpsonding to all combinations of params settings

    CommandLine:
        python -m utool.util_dict --test-all_dict_combinations

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> varied_dict = {'logdist_weight': [0.0, 1.0], 'pipeline_root': ['vsmany'], 'sv_on': [True, False, None]}
        >>> dict_list = all_dict_combinations(varied_dict)
        >>> result = str(ut.repr4(dict_list))
        >>> print(result)
        [
            {'logdist_weight': 0.0, 'pipeline_root': 'vsmany', 'sv_on': True},
            {'logdist_weight': 0.0, 'pipeline_root': 'vsmany', 'sv_on': False},
            {'logdist_weight': 0.0, 'pipeline_root': 'vsmany', 'sv_on': None},
            {'logdist_weight': 1.0, 'pipeline_root': 'vsmany', 'sv_on': True},
            {'logdist_weight': 1.0, 'pipeline_root': 'vsmany', 'sv_on': False},
            {'logdist_weight': 1.0, 'pipeline_root': 'vsmany', 'sv_on': None},
        ]
    """
    #tups_list = [[(key, val) for val in val_list]
    #             if isinstance(val_list, (list, tuple))
    #             else [(key, val_list)]
    #             for (key, val_list) in six.iteritems(varied_dict)]
    tups_list = [[(key, val) for val in val_list]
                 if isinstance(val_list, (list))
                 #if isinstance(val_list, (list, tuple))
                 else [(key, val_list)]
                 for (key, val_list) in iteritems_sorted(varied_dict)]
    dict_list = [dict(tups) for tups in it.product(*tups_list)]
    #dict_list = [{key: val for (key, val) in tups} for tups in it.product(*tups_list)]
    #from collections import OrderedDict
    #dict_list = [OrderedDict([(key, val) for (key, val) in tups]) for tups in it.product(*tups_list)]
    return dict_list


def all_dict_combinations_lbls(varied_dict, remove_singles=True, allow_lone_singles=False):
    """
    returns a label for each variation in a varydict.

    It tries to not be oververbose and returns only what parameters are varied
    in each label.

    CommandLine:
        python -m utool.util_dict --test-all_dict_combinations_lbls
        python -m utool.util_dict --exec-all_dict_combinations_lbls:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> from utool.util_dict import *  # NOQA
        >>> varied_dict = {'logdist_weight': [0.0, 1.0], 'pipeline_root': ['vsmany'], 'sv_on': [True, False, None]}
        >>> comb_lbls = utool.all_dict_combinations_lbls(varied_dict)
        >>> result = (utool.repr4(comb_lbls))
        >>> print(result)
        [
            'logdist_weight=0.0,sv_on=True',
            'logdist_weight=0.0,sv_on=False',
            'logdist_weight=0.0,sv_on=None',
            'logdist_weight=1.0,sv_on=True',
            'logdist_weight=1.0,sv_on=False',
            'logdist_weight=1.0,sv_on=None',
        ]

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> from utool.util_dict import *  # NOQA
        >>> varied_dict = {'logdist_weight': [0.0], 'pipeline_root': ['vsmany'], 'sv_on': [True]}
        >>> allow_lone_singles = True
        >>> comb_lbls = ut.all_dict_combinations_lbls(varied_dict, allow_lone_singles=allow_lone_singles)
        >>> result = (ut.repr4(comb_lbls))
        >>> print(result)
        [
            'logdist_weight=0.0,pipeline_root=vsmany,sv_on=True',
        ]
    """
    is_lone_single = all([
        isinstance(val_list, (list, tuple)) and len(val_list) == 1
        for key, val_list in iteritems_sorted(varied_dict)
    ])
    if not remove_singles or (allow_lone_singles and is_lone_single):
        # all entries have one length
        multitups_list = [
            [(key, val) for val in val_list]
            for key, val_list in iteritems_sorted(varied_dict)
        ]
    else:
        multitups_list = [
            [(key, val) for val in val_list]
            for key, val_list in iteritems_sorted(varied_dict)
            if isinstance(val_list, (list, tuple)) and len(val_list) > 1]
    combtup_list = list(it.product(*multitups_list))
    combtup_list2 = [
        [(key, val) if isinstance(val, six.string_types) else (key, repr(val))
         for (key, val) in combtup]
        for combtup in combtup_list]
    comb_lbls = [','.join(['%s=%s' % (key, val) for (key, val) in combtup])
                 for combtup in combtup_list2]
    #comb_lbls = list(map(str, comb_pairs))
    return comb_lbls


def dict_union2(dict1, dict2):
    return dict(list(dict1.items()) + list(dict2.items()))


def dict_union(*args):
    return dict([item for dict_ in iter(args) for item in six.iteritems(dict_)])


def build_conflict_dict(key_list, val_list):
    """
    Builds dict where a list of values is associated with more than one key

    Args:
        key_list (list):
        val_list (list):

    Returns:
        dict: key_to_vals

    CommandLine:
        python -m utool.util_dict --test-build_conflict_dict

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> key_list = [  1,   2,   2,   3,   1]
        >>> val_list = ['a', 'b', 'c', 'd', 'e']
        >>> key_to_vals = build_conflict_dict(key_list, val_list)
        >>> result = ut.repr4(key_to_vals)
        >>> print(result)
        {
            1: ['a', 'e'],
            2: ['b', 'c'],
            3: ['d'],
        }
    """
    key_to_vals = defaultdict(list)
    for key, val in zip(key_list, val_list):
        key_to_vals[key].append(val)
    return key_to_vals


def assert_keys_are_subset(dict1, dict2):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> dict1 = {1:1, 2:2, 3:3}
        >>> dict2 = {2:3, 3:3}
        >>> assert_keys_are_subset(dict1, dict2)
        >>> #dict2 = {4:3, 3:3}
    """
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    unknown_keys = keys2.difference(keys1)
    assert len(unknown_keys) == 0, 'unknown_keys=%r' % (unknown_keys,)


def augdict(dict1, dict2=None, **kwargs):
    dict1_ = copy.deepcopy(dict1)
    if dict2 is not None:
        dict1_ = update_existing(dict1_, dict2, assert_exists=True)
    if len(kwargs) > 0:
        dict1_ = update_existing(dict1_, kwargs, assert_exists=True)
    return dict1_


def update_existing(dict1, dict2, copy=False, assert_exists=False,
                    iswarning=False, alias_dict=None):
    r"""
    updates vals in dict1 using vals from dict2 only if the
    key is already in dict1.

    Args:
        dict1 (dict):
        dict2 (dict):
        copy (bool): if true modifies dictionary in place (default = False)
        assert_exists (bool): if True throws error if new key specified (default = False)
        alias_dict (dict): dictionary of alias keys for dict2 (default = None)

    Returns:
        dict - updated dictionary

    CommandLine:
        python -m utool.util_dict --test-update_existing

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict1 = {'a': 1, 'b': 2, 'c': 3}
        >>> dict2 = {'a': 2, 'd': 3}
        >>> dict1_ = update_existing(dict1, dict2)
        >>> assert 'd' not in dict1
        >>> assert dict1['a'] == 2
        >>> assert dict1_ is dict1
    """
    if assert_exists:
        try:
            assert_keys_are_subset(dict1, dict2)
        except AssertionError as ex:
            from utool import util_dbg
            util_dbg.printex(ex, iswarning=iswarning, N=1)
            if not iswarning:
                raise
    if copy:
        dict1 = dict(dict1)
    if alias_dict is None:
        alias_dict = {}
    for key, val in six.iteritems(dict2):
        key = alias_dict.get(key, key)
        if key in dict1:
            dict1[key] = val
    return dict1


def update_dict(dict1, dict2, copy=False, alias_dict=None):
    if copy:
        dict1 = dict(dict1)
    if alias_dict is None:
        alias_dict = {}
    for key, val in six.iteritems(dict2):
        key = alias_dict.get(key, key)
        dict1[key] = val
    return dict1


def dict_update_newkeys(dict_, dict2):
    """ Like dict.update, but does not overwrite items """
    for key, val in six.iteritems(dict2):
        if key not in dict_:
            dict_[key] = val


def is_dicteq(dict1_, dict2_, almosteq_ok=True, verbose_err=True):
    """ Checks to see if dicts are the same. Performs recursion. Handles numpy """
    import utool as ut
    assert len(dict1_) == len(dict2_), 'dicts are not of same length'
    try:
        for (key1, val1), (key2, val2) in zip(dict1_.items(), dict2_.items()):
            assert key1 == key2, 'key mismatch'
            assert type(val1) == type(val2), 'vals are not same type'
            if HAVE_NUMPY and np.iterable(val1):
                if almosteq_ok and ut.is_float(val1):
                    assert np.all(ut.almost_eq(val1, val2)), 'float vals are not within thresh'
                else:
                    assert all([np.all(x1 == x2) for (x1, x2) in zip(val1, val2)]), 'np vals are different'
            elif isinstance(val1, dict):
                is_dicteq(val1, val2, almosteq_ok=almosteq_ok, verbose_err=verbose_err)
            else:
                assert val1 == val2, 'vals are different'
    except AssertionError as ex:
        if verbose_err:
            ut.printex(ex)
        return False
    return True


def dict_subset(dict_, keys, default=util_const.NoParam):
    r"""
    Args:
        dict_ (dict):
        keys (list):

    Returns:
        dict: subset dictionary

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {'K': 3, 'dcvs_clip_max': 0.2, 'p': 0.1}
        >>> keys = ['K', 'dcvs_clip_max']
        >>> d = tuple([])
        >>> subdict_ = dict_subset(dict_, keys)
        >>> result = ut.repr4(subdict_, sorted_=True, newlines=False)
        >>> print(result)
        {'K': 3, 'dcvs_clip_max': 0.2}
    """
    if default is util_const.NoParam:
        items = dict_take(dict_, keys)
    else:
        items = dict_take(dict_, keys, default)
    subdict_ = OrderedDict(list(zip(keys, items)))
    #item_sublist = [(key, dict_[key]) for key in keys]
    ##subdict_ = type(dict_)(item_sublist)  # maintain old dict format
    #subdict_ = OrderedDict(item_sublist)
    return subdict_


def dict_to_keyvals(dict_):
    return list(six.iteritems(dict_))


def dict_setdiff(dict_, negative_keys):
    r"""
    returns a copy of dict_ without keys in the negative_keys list

    Args:
        dict_ (dict):
        negative_keys (list):
    """
    keys = [key for key in six.iterkeys(dict_)
            if key not in set(negative_keys)]
    subdict_ = dict_subset(dict_, keys)
    return subdict_


def delete_dict_keys(dict_, key_list):
    r"""
    Removes items from a dictionary inplace. Keys that do not exist are
    ignored.

    Args:
        dict_ (dict): dict like object with a __del__ attribute
        key_list (list): list of keys that specify the items to remove

    CommandLine:
        python -m utool.util_dict --test-delete_dict_keys

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {'bread': 1, 'churches': 1, 'cider': 2, 'very small rocks': 2}
        >>> key_list = ['duck', 'bread', 'cider']
        >>> delete_dict_keys(dict_, key_list)
        >>> result = ut.repr4(dict_, nl=False)
        >>> print(result)
        {'churches': 1, 'very small rocks': 2}

    """
    invalid_keys = set(key_list) - set(dict_.keys())
    valid_keys = set(key_list) - invalid_keys
    for key in valid_keys:
        del dict_[key]
    return dict_


delete_keys = delete_dict_keys


def dict_take_gen(dict_, keys, *d):
    r"""
    generate multiple values from a dictionary

    Args:
        dict_ (dict):
        keys (list):

    Varargs:
        d: if specified is default for key errors

    CommandLine:
        python -m utool.util_dict --test-dict_take_gen

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {1: 'a', 2: 'b', 3: 'c'}
        >>> keys = [1, 2, 3, 4, 5]
        >>> result = list(dict_take_gen(dict_, keys, None))
        >>> result = ut.repr4(result, nl=False)
        >>> print(result)
        ['a', 'b', 'c', None, None]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = {1: 'a', 2: 'b', 3: 'c'}
        >>> keys = [1, 2, 3, 4, 5]
        >>> try:
        >>>     print(list(dict_take_gen(dict_, keys)))
        >>>     result = 'did not get key error'
        >>> except KeyError:
        >>>     result = 'correctly got key error'
        >>> print(result)
        correctly got key error
    """
    if isinstance(keys, six.string_types):
        # hack for string keys that makes copy-past easier
        keys = keys.split(', ')
    if len(d) == 0:
        # no default given throws key error
        dictget = dict_.__getitem__
    elif len(d) == 1:
        # default given does not throw key erro
        dictget = dict_.get
    else:
        raise ValueError('len(d) must be 1 or 0')
    for key in keys:
        if HAVE_NUMPY and isinstance(key, np.ndarray):
            # recursive call
            yield list(dict_take_gen(dict_, key, *d))
        else:
            yield dictget(key, *d)


def dict_take(dict_, keys, *d):
    """ get multiple values from a dictionary """
    try:
        return list(dict_take_gen(dict_, keys, *d))
    except TypeError:
        return list(dict_take_gen(dict_, keys, *d))[0]
    #return [dict_[key] for key in keys]

dict_take_list = dict_take
#def dict_take(dict_, keys, *d):
#    """ alias """
#    try:
#        return dict_take_list(dict_, keys, *d)
#    except TypeError:
#        return dict_take_list(dict_, [keys], *d)[0]


#def dict_unflat_take(dict_, unflat_key_list, *d):
#    return [dict_unflat_take(dict_, xs, *d)
#            if isinstance(xs, list) else
#            dict_take(dict_, xs, *d)
#            for xs in unflat_key_list]


def dict_take_asnametup(dict_, keys, name='_NamedTup'):
    from collections import namedtuple
    values = dict_take(dict_, keys)
    _NamedTup = namedtuple(name, keys)
    tup = _NamedTup(*values)
    return tup


def dict_take_pop(dict_, keys, *d):
    """ like dict_take but pops values off

    CommandLine:
        python -m utool.util_dict --test-dict_take_pop

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {1: 'a', 'other': None, 'another': 'foo', 2: 'b', 3: 'c'}
        >>> keys = [1, 2, 3, 4, 5]
        >>> print('before: ' + ut.repr4(dict_))
        >>> result = list(dict_take_pop(dict_, keys, None))
        >>> result = ut.repr4(result, nl=False)
        >>> print('after: ' + ut.repr4(dict_))
        >>> assert len(dict_) == 2
        >>> print(result)
        ['a', 'b', 'c', None, None]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {1: 'a', 2: 'b', 3: 'c'}
        >>> keys = [1, 2, 3, 4, 5]
        >>> print('before: ' + ut.repr4(dict_))
        >>> try:
        >>>     print(list(dict_take_pop(dict_, keys)))
        >>>     result = 'did not get key error'
        >>> except KeyError:
        >>>     result = 'correctly got key error'
        >>> assert len(dict_) == 0
        >>> print('after: ' + ut.repr4(dict_))
        >>> print(result)
        correctly got key error
    """
    if len(d) == 0:
        return [dict_.pop(key) for key in keys]
    elif len(d) == 1:
        default = d[0]
        return [dict_.pop(key, default) for key in keys]
    else:
        raise ValueError('len(d) must be 1 or 0')


def dict_assign(dict_, keys, vals):
    """ simple method for assigning or setting values with a similar interface
    to dict_take """
    for key, val in zip(keys, vals):
        dict_[key] = val


def dict_where_len0(dict_):
    """
    Accepts a dict of lists. Returns keys that have vals with no length
    """
    keys = np.array(dict_.keys())
    flags = np.array(list(map(len, dict_.values()))) == 0
    indices = np.where(flags)[0]
    return keys[indices]


def get_dict_column(dict_, colx):
    r"""
    Args:
        dict_ (dict_): a dictionary of lists
        colx (int):

    CommandLine:
        python -m utool.util_dict --test-get_dict_column

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {'a': [0, 1, 2], 'b': [3, 4, 5], 'c': [6, 7, 8]}
        >>> colx = [2, 0]
        >>> retdict_ = get_dict_column(dict_, colx)
        >>> result = ut.repr2(retdict_)
        >>> print(result)
        {'a': [2, 0], 'b': [5, 3], 'c': [8, 6]}
    """
    retdict_ = {key: util_list.list_take(val, colx)
                for key, val in six.iteritems(dict_)}
    return retdict_


def dict_take_column(list_of_dicts_, colkey, default=None):
    return [dict_.get(colkey, default) for dict_ in list_of_dicts_]


def dict_set_column(list_of_dicts_, colkey, value_list):
    for dict_, value in zip(list_of_dicts_, value_list):
        dict_[colkey] = value


def dictinfo(dict_):
    """
    dictinfo

    In depth debugging info

    Args:
        dict_ (dict):

    Returns:
        str

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = {}
        >>> result = dictinfo(dict_)
        >>> print(result)
    """
    import utool as ut
    if not isinstance(dict_, dict):
        return 'expected dict got %r' % type(dict_)

    keys = list(dict_.keys())
    vals = list(dict_.values())
    num_keys  = len(keys)
    key_types = list(set(map(type, keys)))
    val_types = list(set(map(type, vals)))

    fmtstr_ = '\n' + ut.unindent('''
    * num_keys  = {num_keys}
    * key_types = {key_types}
    * val_types = {val_types}
    '''.strip('\n'))

    if len(val_types) == 1:
        if val_types[0] == np.ndarray:
            # each key holds an ndarray
            val_shape_stats = ut.get_stats(set(map(np.shape, vals)), axis=0)
            val_shape_stats_str = ut.repr4(val_shape_stats, strvals=True, newlines=False)
            val_dtypes = list(set([val.dtype for val in vals]))
            fmtstr_ += ut.unindent('''
            * val_shape_stats = {val_shape_stats_str}
            * val_dtypes = {val_dtypes}
            '''.strip('\n'))
        elif val_types[0] == list:
            # each key holds a list
            val_len_stats =  ut.get_stats(set(map(len, vals)))
            val_len_stats_str = ut.repr4(val_len_stats, strvals=True, newlines=False)
            depth = ut.list_depth(vals)
            deep_val_types = list(set(ut.list_deep_types(vals)))
            fmtstr_ += ut.unindent('''
            * list_depth = {depth}
            * val_len_stats = {val_len_stats_str}
            * deep_types = {deep_val_types}
            '''.strip('\n'))
            if len(deep_val_types) == 1:
                if deep_val_types[0] == np.ndarray:
                    deep_val_dtypes = list(set([val.dtype for val in vals]))
                    fmtstr_ += ut.unindent('''
                    * deep_val_dtypes = {deep_val_dtypes}
                    ''').strip('\n')
        elif val_types[0] in [np.uint8, np.int8, np.int32, np.int64, np.float16, np.float32, np.float64]:
            # each key holds a scalar
            val_stats = ut.get_stats(vals)
            fmtstr_ += ut.unindent('''
            * val_stats = {val_stats}
            ''').strip('\n')

    fmtstr = fmtstr_.format(**locals())
    return ut.indent(fmtstr)


def dict_find_keys(dict_, val_list):
    r"""
    Args:
        dict_ (dict):
        val_list (list):

    Returns:
        dict: found_dict

    CommandLine:
        python -m utool.util_dict --test-dict_find_keys

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {'default': 1, 'hierarchical': 5, 'linear': 0, 'kdtree': 1,
        ...          'composite': 3, 'autotuned': 255, 'saved': 254, 'kmeans': 2,
        ...          'lsh': 6, 'kdtree_single': 4}
        >>> val_list = [1]
        >>> found_dict = dict_find_keys(dict_, val_list)
        >>> result = ut.repr2(ut.map_vals(sorted, found_dict))
        >>> print(result)
        {1: ['default', 'kdtree']}
    """
    found_dict = {
        search_val: [key for key, val in six.iteritems(dict_)
                     if val == search_val]
        for search_val in val_list
    }
    return found_dict


def dict_find_other_sameval_keys(dict_, key):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = {'default': 1, 'hierarchical': 5, 'linear': 0, 'kdtree': 1,
        ...          'composite': 3, 'autotuned': 255, 'saved': 254, 'kmeans': 2,
        ...          'lsh': 6, 'kdtree_single': 4}
        >>> key = 'default'
        >>> found_dict = dict_find_keys(dict_, val_list)
    """
    value = dict_[key]
    found_dict = dict_find_keys(dict_, [value])
    other_keys = found_dict[value]
    other_keys.remove(key)
    return other_keys


@profile
def dict_hist(item_list, weight_list=None, ordered=False, labels=None):
    r"""
    Builds a histogram of items in item_list

    Args:
        item_list (list): list with hashable items (usually containing duplicates)

    Returns:
        dict : dictionary where the keys are items in item_list, and the values
          are the number of times the item appears in item_list.

    CommandLine:
        python -m utool.util_dict --test-dict_hist

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> item_list = [1, 2, 39, 900, 1232, 900, 1232, 2, 2, 2, 900]
        >>> hist_ = dict_hist(item_list)
        >>> result = ut.repr2(hist_)
        >>> print(result)
        {1: 1, 2: 4, 39: 1, 900: 3, 1232: 2}
    """
    if labels is None:
        # hist_ = defaultdict(lambda: 0)
        hist_ = defaultdict(int)
    else:
        hist_ = {k: 0 for k in labels}
    if weight_list is None:
        # weight_list = it.repeat(1)
        for item in item_list:
            hist_[item] += 1
    else:
        for item, weight in zip(item_list, weight_list):
            hist_[item] += weight
    # hist_ = dict(hist_)
    if ordered:
        # import utool as ut
        # key_order = ut.sortedby(list(hist_.keys()), list(hist_.values()))
        getval = op.itemgetter(1)
        key_order = [key for (key, value) in sorted(hist_.items(), key=getval)]
        hist_ = order_dict_by(hist_, key_order)
    return hist_


def range_hist(items, bins):
    """
    Bins items into a discrete histogram by values and/or ranges.

        items = [1, 2, 3, 4, 5, 6, 7]
        bins = [0, 1, 2, (3, float('inf'))]
        ut.range_hist(items, bins)
    """
    big_hist = ut.dict_hist(items)
    hist = ut.odict([(b, 0) for b in bins])

    for k, v in big_hist.items():
        for b in bins:
            if isinstance(b, (list, tuple)):
                if k >= b[0] and k < b[1]:
                    hist[b] += v
            elif k == b:
                hist[b] += v
    return hist


def dict_hist_cumsum(hist_, reverse=True):
    """ VERY HACKY """
    import utool as ut
    items = hist_.items()
    if reverse:
        items = sorted(items)[::-1]
    else:
        items = sorted(items)
    key_list = ut.get_list_column(items, 0)
    val_list = ut.get_list_column(items, 1)
    cumhist_ = dict(zip(key_list, np.cumsum(val_list)))
    return cumhist_


def merge_dicts(*args):
    r"""
    add / concatenate / union / join / merge / combine dictionaries

    Copies the first dictionary given and then repeatedly calls update using
    the rest of the dicts given in args. Duplicate keys will receive the last
    value specified the list of dictionaries.

    Returns:
        dict: mergedict_

    CommandLine:
        python -m utool.util_dict --test-merge_dicts

    References:
        http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> x = {'a': 1, 'b': 2}
        >>> y = {'b': 3, 'c': 4}
        >>> mergedict_ = merge_dicts(x, y)
        >>> result = ut.repr4(mergedict_, sorted_=True, newlines=False)
        >>> print(result)
        {'a': 1, 'b': 3, 'c': 4}

    """
    iter_ = iter(args)
    mergedict_ = six.next(iter_).copy()
    for dict_ in iter_:
        mergedict_.update(dict_)
    return mergedict_


def dict_union3(dict1, dict2, combine_op=op.add):
    r"""
    Args:
        dict1 (dict):
        dict2 (dict):
        combine_op (func): (default=op.add)

    Returns:
        dict: mergedict_

    CommandLine:
        python -m utool.util_dict --exec-dict_union3

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        >>> dict2 = {'b': 2, 'c': 3, 'd': 5, 'e': 21, 'f': 42}
        >>> combine_op = op.add
        >>> mergedict_ = dict_union3(dict1, dict2, combine_op)
        >>> result = ('mergedict_ = %s' % (ut.repr4(mergedict_, nl=False),))
        >>> print(result)
        mergedict_ = {'a': 1, 'b': 4, 'c': 6, 'd': 9, 'e': 21, 'f': 42}
    """
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    # Combine common keys
    keys3 = keys1.intersection(keys2)
    if len(keys3) > 0 and combine_op is None:
        raise AssertionError('Can only combine disjoint dicts when combine_op is None')
    dict3 = {key: combine_op(dict1[key], dict2[key]) for key in keys3}
    # Combine unique keys
    for key in keys1.difference(keys3):
        dict3[key] = dict1[key]
    for key in keys2.difference(keys3):
        dict3[key] = dict2[key]
    return dict3


def dict_intersection(dict1, dict2, combine=False, combine_op=op.add):
    r"""
    Args:
        dict1 (dict):
        dict2 (dict):
        combine (bool): Combines keys only if the values are equal if False else
            values are combined using combine_op (default = False)
        combine_op (func): (default = op.add)

    Returns:
        dict: mergedict_

    CommandLine:
        python -m utool.util_dict --exec-dict_intersection

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        >>> dict2 = {'b': 2, 'c': 3, 'd': 5, 'e': 21, 'f': 42}
        >>> combine = False
        >>> mergedict_ = dict_intersection(dict1, dict2, combine)
        >>> result = ('mergedict_ = %s' % (ut.repr4(mergedict_, nl=False),))
        >>> print(result)
        mergedict_ = {'b': 2, 'c': 3}
    """
    isect_keys = set(dict1.keys()).intersection(set(dict2.keys()))
    if combine:
        # TODO: depricate this
        dict_isect = {k: combine_op(dict1[k], dict2[k]) for k in isect_keys}
    else:
        # maintain order if possible
        if isinstance(dict1, OrderedDict):
            isect_keys_ = [k for k in dict1.keys() if k in isect_keys]
            _dict_cls = OrderedDict
        else:
            isect_keys_ = isect_keys
            _dict_cls = dict
        dict_isect = _dict_cls(
            (k, dict1[k]) for k in isect_keys_ if dict1[k] == dict2[k]
        )
    return dict_isect


def dict_isect_combine(dict1, dict2, combine_op=op.add):
    """ Intersection of dict keys and combination of dict values """
    keys3 = set(dict1.keys()).intersection(set(dict2.keys()))
    dict3 = {key: combine_op(dict1[key], dict2[key]) for key in keys3}
    return dict3


def dict_union_combine(dict1, dict2, combine_op=op.add,
                       default=util_const.NoParam,
                       default2=util_const.NoParam):
    """
    Combine of dict keys and uses dfault value when key does not exist

    CAREFUL WHEN USING THIS WITH REDUCE. Use dict_stack2 instead
    """
    keys3 = set(dict1.keys()).union(set(dict2.keys()))
    if default is util_const.NoParam:
        dict3 = {key: combine_op(dict1[key], dict2[key]) for key in keys3}
    else:
        if default2 is util_const.NoParam:
            default2 = default
        dict3 = {key: combine_op(dict1.get(key, default), dict2.get(key, default2))
                 for key in keys3}
    return dict3


def dict_accum(*dict_list):
    accumulator = defaultdict(list)
    for dict_ in dict_list:
        for key, val in dict_.items():
            accumulator[key].append(val)
    return accumulator

dict_isect = dict_intersection


def dict_filter_nones(dict_):
    r"""
    Removes None values

    Args:
        dict_ (dict):  a dictionary

    Returns:
        dict:

    CommandLine:
        python -m utool.util_dict --exec-dict_filter_nones

    Example:
        >>> # DISABLE_DOCTEST
        >>> # UNSTABLE_DOCTEST
        >>> # fails on python 3 because of dict None order
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {1: None, 2: 'blue', 3: 'four', None: 'fun'}
        >>> dict2_ = dict_filter_nones(dict_)
        >>> result = ut.repr4(dict2_, nl=False)
        >>> print(result)
        {None: 'fun', 2: 'blue', 3: 'four'}
    """
    dict2_ = {
        key: val
        for key, val in six.iteritems(dict_)
        if val is not None
    }
    return dict2_


def groupby_tags(item_list, tags_list):
    r"""
    case where an item can belong to multiple groups

    Args:
        item_list (list):
        tags_list (list):

    Returns:
        dict: groupid_to_items

    CommandLine:
        python -m utool.util_dict --test-groupby_tags

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> tagged_item_list = {
        >>>     'spam': ['meat', 'protein', 'food'],
        >>>     'eggs': ['protein', 'food'],
        >>>     'cheese': ['dairy', 'protein', 'food'],
        >>>     'jam':  ['fruit', 'food'],
        >>>     'banana': ['weapon', 'fruit', 'food'],
        >>> }
        >>> item_list = list(tagged_item_list.keys())
        >>> tags_list = list(tagged_item_list.values())
        >>> groupid_to_items = groupby_tags(item_list, tags_list)
        >>> groupid_to_items = ut.map_vals(sorted, groupid_to_items)
        >>> result = ('groupid_to_items = %s' % (ut.repr4(groupid_to_items),))
        >>> print(result)
        groupid_to_items = {
            'dairy': ['cheese'],
            'food': ['banana', 'cheese', 'eggs', 'jam', 'spam'],
            'fruit': ['banana', 'jam'],
            'meat': ['spam'],
            'protein': ['cheese', 'eggs', 'spam'],
            'weapon': ['banana'],
        }

    """
    groupid_to_items = defaultdict(list)
    for tags, item in zip(tags_list, item_list):
        for tag in tags:
            groupid_to_items[tag].append(item)
    return groupid_to_items


def groupby_attr(item_list, attrname):
    return group_items(item_list,
                       map(op.attrgetter(attrname), item_list))


def group_pairs(pair_list):
    """
    Groups a list of items using the first element in each pair as the item and
    the second element as the groupid.

    Args:
        pair_list (list): list of 2-tuples (item, groupid)

    Returns:
        dict: groupid_to_items: maps a groupid to a list of items

    SeeAlso:
        group_items
    """
    # Initialize dict of lists
    groupid_to_items = defaultdict(list)
    # Insert each item into the correct group
    for item, groupid in pair_list:
        groupid_to_items[groupid].append(item)
    return groupid_to_items


def group_items(items, by=None, sorted_=True):
    """
    Groups a list of items by group id.

    Args:
        items (list): a list of the values to be grouped.
            if `by` is None, then each item is assumed to be a
            (groupid, value) pair.
        by (list): a corresponding list to group items by.
            if specified, these are used as the keys to group values
            in `items`
        sorted_ (bool): if True preserves the ordering of items within groups
            (default = True) FIXME. the opposite is true

    Returns:
        dict: groupid_to_items: maps a groupid to a list of items

    SeeAlso:
        group_indices - first part of a a more fine grained grouping algorithm
        apply_gropuing - second part of a more fine grained grouping algorithm

    CommandLine:
        python -m utool.util_dict --test-group_items

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> items = ['ham',     'jam',   'spam',     'eggs',    'cheese', 'bannana']
        >>> by    = ['protein', 'fruit', 'protein',  'protein', 'dairy',  'fruit']
        >>> groupid_to_items = ut.group_items(items, iter(by))
        >>> result = ut.repr2(groupid_to_items)
        >>> print(result)
        {'dairy': ['cheese'], 'fruit': ['jam', 'bannana'], 'protein': ['ham', 'spam', 'eggs']}
    """
    if by is not None:
        pairs = list(zip(by, items))
        if sorted_:
            # Sort by groupid for cache efficiency (does this even do anything?)
            # I forgot why this is needed? Determenism?
            try:
                pairs = sorted(pairs, key=op.itemgetter(0))
            except TypeError:
                # Python 3 does not allow sorting mixed types
                pairs = sorted(pairs, key=lambda tup: str(tup[0]))
    else:
        pairs = items

    # Initialize a dict of lists
    groupid_to_items = defaultdict(list)
    # Insert each item into the correct group
    for groupid, item in pairs:
        groupid_to_items[groupid].append(item)
    return groupid_to_items


def hierarchical_group_items(item_list, groupids_list):
    """
    Generalization of group_item. Convert a flast list of ids into a heirarchical dictionary.

    TODO: move to util_dict

    Reference:
        http://stackoverflow.com/questions/10193235/python-translate-a-table-to-a-hierarchical-dictionary

    Args:
        item_list (list):
        groupids_list (list):

    CommandLine:
        python -m utool.util_dict --exec-hierarchical_group_items

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> item_list     = [1, 2, 3, 4]
        >>> groupids_list = [[1, 1, 2, 2]]
        >>> tree = hierarchical_group_items(item_list, groupids_list)
        >>> result = ('tree = ' + ut.repr4(tree, nl=len(groupids_list) - 1))
        >>> print(result)
        tree = {1: [1, 2], 2: [3, 4]}

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> item_list     = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> groupids_list = [[1, 2, 1, 2, 1, 2, 1, 2], [3, 2, 2, 2, 3, 1, 1, 1]]
        >>> tree = hierarchical_group_items(item_list, groupids_list)
        >>> result = ('tree = ' + ut.repr4(tree, nl=len(groupids_list) - 1))
        >>> print(result)
        tree = {
            1: {1: [7], 2: [3], 3: [1, 5]},
            2: {1: [6, 8], 2: [2, 4]},
        }

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> item_list     = [1, 2, 3, 4]
        >>> groupids_list = [[1, 1, 1, 2], [1, 2, 2, 2], [1, 3, 1, 1]]
        >>> tree = hierarchical_group_items(item_list, groupids_list)
        >>> result = ('tree = ' + ut.repr4(tree, nl=len(groupids_list) - 1))
        >>> print(result)
        tree = {
            1: {
                1: {1: [1]},
                2: {1: [3], 3: [2]},
            },
            2: {
                2: {1: [4]},
            },
        }

    """
    # Construct a defaultdict type with the appropriate number of levels
    num_groups = len(groupids_list)
    leaf_type = partial(defaultdict, list)
    if num_groups > 1:
        node_type = leaf_type
        for _ in range(len(groupids_list) - 2):
            node_type = partial(defaultdict, node_type)
        root_type = node_type
    elif num_groups == 1:
        root_type = list
    else:
        raise ValueError('must suply groupids')
    tree = defaultdict(root_type)
    #
    groupid_tuple_list = list(zip(*groupids_list))
    for groupid_tuple, item in zip(groupid_tuple_list, item_list):
        node = tree
        for groupid in groupid_tuple:
            node = node[groupid]
        node.append(item)
    return tree


def iflatten_dict_values(node, depth=0):
    """
        >>> from utool.util_dict import *  # NOQA
    """
    if isinstance(node, dict):
        _iter = (iflatten_dict_values(value) for value in six.itervalues(node))
        return util_iter.iflatten(_iter)
    else:
        return node


#def iflatten_dict_items(node, depth=0):
#    if isinstance(node, dict):
#        six.iteritems(node)
#        _iter = ((key, iflatten_dict_items(value)) for key, value in six.iteritems(node))
#        return util_iter.iflatten(_iter)
#    else:
#        return node


#def iflatten_dict_keys(node, depth=0):
#    if isinstance(node, dict):
#        _iter = (iflatten_dict_keys(value) for key, value in six.iteritems(node))
#        return util_iter.iflatten(_iter)
#    else:
#        return node


def hierarchical_map_vals(func, node, max_depth=None, depth=0):
    """
    node is a dict tree like structure with leaves of type list

    TODO: move to util_dict

    CommandLine:
        python -m utool.util_dict --exec-hierarchical_map_vals

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> item_list     = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> groupids_list = [[1, 2, 1, 2, 1, 2, 1, 2], [3, 2, 2, 2, 3, 1, 1, 1]]
        >>> tree = ut.hierarchical_group_items(item_list, groupids_list)
        >>> len_tree = ut.hierarchical_map_vals(len, tree)
        >>> result = ('len_tree = ' + ut.repr4(len_tree, nl=1))
        >>> print(result)
        len_tree = {
            1: {1: 1, 2: 1, 3: 2},
            2: {1: 2, 2: 2},
        }

    Example1:
        >>> # DISABLE_DOCTEST
        >>> # UNSTABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> depth = 4
        >>> item_list = list(range(2 ** (depth + 1)))
        >>> num = len(item_list) // 2
        >>> groupids_list = []
        >>> total = 0
        >>> for level in range(depth):
        ...     num2 = len(item_list) // int((num * 2))
        ...     #nonflat_levelids = [([total + 2 * x + 1] * num + [total + 2 * x + 2] * num) for x in range(num2)]
        ...     nonflat_levelids = [([1] * num + [2] * num) for x in range(num2)]
        ...     levelids = ut.flatten(nonflat_levelids)
        ...     groupids_list.append(levelids)
        ...     total += num2 * 2
        ...     num //= 2
        >>> print('groupids_list = %s' % (ut.repr4(groupids_list, nl=1),))
        >>> print('depth = %r' % (len(groupids_list),))
        >>> tree = ut.hierarchical_group_items(item_list, groupids_list)
        >>> print('tree = ' + ut.repr4(tree, nl=None))
        >>> flat_tree_values = list(ut.iflatten_dict_values(tree))
        >>> assert sorted(flat_tree_values) == sorted(item_list)
        >>> print('flat_tree_values = ' + str(flat_tree_values))
        >>> #print('flat_tree_keys = ' + str(list(ut.iflatten_dict_keys(tree))))
        >>> #print('iflatten_dict_items = ' + str(list(ut.iflatten_dict_items(tree))))
        >>> len_tree = ut.hierarchical_map_vals(len, tree, max_depth=4)
        >>> result = ('len_tree = ' + ut.repr4(len_tree, nl=None))
        >>> print(result)

    """
    #if not isinstance(node, dict):
    if not hasattr(node, 'items'):
        return func(node)
    elif max_depth is not None and depth >= max_depth:
        #return func(node)
        return map_dict_vals(func, node)
        #return {key: func(val) for key, val in six.iteritems(node)}
    else:
        # recursion
        #return {key: hierarchical_map_vals(func, val, max_depth, depth + 1) for key, val in six.iteritems(node)}
        #keyval_list = [(key, hierarchical_map_vals(func, val, max_depth, depth + 1)) for key, val in six.iteritems(node)]
        keyval_list = [(key, hierarchical_map_vals(func, val, max_depth, depth + 1)) for key, val in node.items()]
        if isinstance(node, OrderedDict):
            return OrderedDict(keyval_list)
        else:
            return dict(keyval_list)


def move_odict_item(odict, key, newpos):
    """
    References:
        http://stackoverflow.com/questions/22663966/changing-order-of-ordered-dictionary-in-python

    CommandLine:
        python -m utool.util_dict --exec-move_odict_item

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> odict = OrderedDict()
        >>> odict['a'] = 1
        >>> odict['b'] = 2
        >>> odict['c'] = 3
        >>> odict['e'] = 5
        >>> print(ut.repr4(odict, nl=False))
        >>> move_odict_item(odict, 'c', 1)
        >>> print(ut.repr4(odict, nl=False))
        >>> move_odict_item(odict, 'a', 3)
        >>> print(ut.repr4(odict, nl=False))
        >>> move_odict_item(odict, 'a', 0)
        >>> print(ut.repr4(odict, nl=False))
        >>> move_odict_item(odict, 'b', 2)
        >>> result = ut.repr4(odict, nl=False)
        >>> print(result)
        {'a': 1, 'c': 3, 'b': 2, 'e': 5}
    """
    odict[key] = odict.pop(key)
    for i, otherkey in enumerate(list(odict.keys())):
        if otherkey != key and i >= newpos:
            odict[otherkey] = odict.pop(otherkey)
    return odict


hmap_vals = hierarchical_map_vals


#def hierarchical_map_nodes(func, node, max_depth=None, depth=0):
#    """
#    applies function to non-leaf nodes
#    """
#    if not isinstance(node, dict):
#        return node
#    elif max_depth is not None and depth >= max_depth:
#        #return func(node)
#        return func(node)
#    else:
#        # recursion
#        return {key: func(hierarchical_map_vals(func, val, max_depth, depth + 1)) for key, val in six.iteritems(node)}


class DictLike(object):
    """
    move to util_dict rectify with util_dev

    An inherited class must specify the ``getitem``, ``setitem``, and
      ``keys`` methods.


    """

    def getitem(self, key):
        raise NotImplementedError('abstract getitem function')

    def setitem(self, key, value):
        raise NotImplementedError('abstract setitem function')

    def delitem(self, key):
        raise NotImplementedError('abstract delitem function')

    def keys(self):
        raise NotImplementedError('abstract keys function')

    def __repr__(self):
        return repr(self.asdict())

    def __str__(self):
        return str(self.asdict())

    def __len__(self):
        return len(list(self.keys()))

    def __contains__(self, key):
        return key in self.keys()

    def __delitem__(self, key):
        return self.delitem(key)

    def __getitem__(self, key):
        return self.getitem(key)

    def __setitem__(self, key, value):
        return self.setitem(key, value)

    def items(self):
        if six.PY2:
            return list(self.iteritems())
        else:
            return self.iteritems()

    def values(self):
        if six.PY2:
            return [self[key] for key in self.keys()]
        else:
            return (self[key] for key in self.keys())

    def copy(self):
        return dict(self.items())

    def asdict(self):
        return dict(self.items())

    def iteritems(self):
        for key, val in zip(self.iterkeys(), self.itervalues()):
            yield key, val

    def itervalues(self):
        return (self[key] for key in self.keys())

    def iterkeys(self):
        return (key for key in self.keys())

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def sort_dict(dict_, part='keys', key=None, reverse=False):
    """
    sorts a dictionary by its values or its keys

    Args:
        dict_ (dict_):  a dictionary
        part (str): specifies to sort by keys or values
        key (Optional[func]): a function that takes specified part
            and returns a sortable value
        reverse (bool): (Defaults to False) - True for descinding order. False
            for ascending order.

    Returns:
        OrderedDict: sorted dictionary

    CommandLine:
        python -m utool.util_dict sort_dict

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {'a': 3, 'c': 2, 'b': 1}
        >>> results = []
        >>> results.append(sort_dict(dict_, 'keys'))
        >>> results.append(sort_dict(dict_, 'vals'))
        >>> results.append(sort_dict(dict_, 'vals', lambda x: -x))
        >>> result = ut.repr4(results)
        >>> print(result)
        [
            {'a': 3, 'b': 1, 'c': 2},
            {'b': 1, 'c': 2, 'a': 3},
            {'a': 3, 'c': 2, 'b': 1},
        ]
    """
    if part == 'keys':
        index = 0
    elif part in {'vals', 'values'}:
        index = 1
    else:
        raise ValueError('Unknown method part=%r' % (part,))
    if key is None:
        _key = op.itemgetter(index)
    else:
        def _key(item):
            return key(item[index])
    sorted_items = sorted(six.iteritems(dict_), key=_key, reverse=reverse)
    sorted_dict = OrderedDict(sorted_items)
    return sorted_dict


def order_dict_by(dict_, key_order):
    r"""
    Reorders items in a dictionary according to a custom key order

    Args:
        dict_ (dict_):  a dictionary
        key_order (list): custom key order

    Returns:
        OrderedDict: sorted_dict

    CommandLine:
        python -m utool.util_dict --exec-order_dict_by

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {1: 1, 2: 2, 3: 3, 4: 4}
        >>> key_order = [4, 2, 3, 1]
        >>> sorted_dict = order_dict_by(dict_, key_order)
        >>> result = ('sorted_dict = %s' % (ut.repr4(sorted_dict, nl=False),))
        >>> print(result)
        >>> assert result == 'sorted_dict = {4: 4, 2: 2, 3: 3, 1: 1}'

    """
    dict_keys = set(dict_.keys())
    other_keys = dict_keys - set(key_order)
    key_order = it.chain(key_order, other_keys)
    sorted_dict = OrderedDict(
        (key, dict_[key]) for key in key_order if key in dict_keys
    )
    return sorted_dict


def iteritems_sorted(dict_):
    """ change to iteritems ordered """
    if isinstance(dict_, OrderedDict):
        return six.iteritems(dict_)
    else:
        return iter(sorted(six.iteritems(dict_)))


def keys_sorted_by_value(dict_):
    sorted_keys = sorted(dict_, key=lambda key: dict_[key])
    return sorted_keys


def flatten_dict_vals(dict_):
    """
    Flattens only values in a heirarchical dictionary, keys are nested.
    """
    if isinstance(dict_, dict):
        return dict([
            ((key, augkey), augval)
            for key, val in dict_.items()
            for augkey, augval in flatten_dict_vals(val).items()
        ])
    else:
        return {None: dict_}


def flatten_dict_items(dict_):
    """
    Flattens keys / values in a heirarchical dictionary

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> item_list     = [1, 2, 3, 4]
        >>> groupids_list = [[1, 1, 1, 2], [1, 2, 2, 2], [1, 3, 1, 1]]
        >>> dict_ = hierarchical_group_items(item_list, groupids_list)
        >>> flatter_dict = flatten_dict_items(dict_)
        >>> result = ('flatter_dict = ' + ut.repr4(flatter_dict, nl=1))
        >>> print(result)
        flatter_dict = {
            (1, 1, 1): [1],
            (1, 2, 1): [3],
            (1, 2, 3): [2],
            (2, 2, 1): [4],
        }
    """
    import utool as ut
    flat_dict = ut.flatten_dict_vals(dict_)
    flatter_dict = dict([(tuple(ut.unpack_iterables(key)[:-1]), val)
                         for key, val in flat_dict.items()])
    return flatter_dict


def depth_atleast(list_, depth):
    r"""
    Returns if depth of list is at least ``depth``

    Args:
        list_ (list):
        depth (int):

    Returns:
        bool: True

    CommandLine:
        python -m utool.util_dict --exec-depth_atleast --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> list_ = [[[[0]]], [[0]]]
        >>> depth = 0
        >>> result = [depth_atleast(list_, depth) for depth in range(0, 7)]
        >>> print(result)
    """
    if depth == 0:
        return True
    else:
        try:
            return all([depth_atleast(item, depth - 1) for item in list_])
        except TypeError:
            return False


class DefaultValueDict(dict):
    """
    picklable default dictionary that can store scalar values.

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> self = ut.DefaultValueDict(0)
        >>> print(self[4])
        >>> self[4] = 4
        >>> print(self[4])
        0
        4
    """
    def __init__(self, default, other=None, **kwargs):
        self.default = default
        if other:
            self.update(other)
        if kwargs:
            self.update(kwargs)

    def __getitem__(self, item):
        return self.get(item, self.default)


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_dict
        python -m utool.util_dict --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
