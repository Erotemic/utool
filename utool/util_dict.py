""" convinience functions for dictionaries """
from __future__ import absolute_import, division, print_function
from collections import defaultdict
from itertools import product as iprod
from six.moves import zip
from collections import OrderedDict
from utool import util_inject
from utool import util_list
import six
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    pass
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[dict]')


def count_dict_vals(dict_of_lists):
    count_dict = {'len(%s)' % (key,): len(val) for key, val in six.iteritems(dict_of_lists)}
    return count_dict


def dict_keysubset(dict_, keys):
    return [key for key in keys if key in dict_]


def get_dict_hashid(dict_):
    r"""
    Args:
        dict_ (?):

    Returns:
        ?: inverted_dict

    CommandLine:
        python -m utool.util_dict --test-get_dict_hashid

    References:
        http://stackoverflow.com/questions/5884066/hashing-a-python-dictionary

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> # build test data
        >>> dict_ = {}
        >>> dict_ = {'a': 'b'}
        >>> dict_ = {'a': {'c': 'd'}}
        >>> # execute function
        >>> hashid = get_dict_hashid(dict_)
        >>> # verify results
        >>> result = str(hashid)
        >>> print(result)
    """
    from utool import util_hash
    #hashid = hash(frozenset(dict_.items()))
    hashid = util_hash.make_hash(dict_)
    return hashid


def dict_stack(dict_list, key_prefix=''):
    r"""
    stacks values from two dicts into a new dict where the values are list of
    the input values. the keys are the same.

    Args:
        dict_list (list): list of dicts with similar keys

    Returns:
        dict dict_stacked

    CommandLine:
        python -m utool.util_dict --test-dict_stack

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> # build test data
        >>> dict1_ = {'a': 1, 'b': 2}
        >>> dict2_ = {'a': 2, 'b': 3, 'c': 4}
        >>> # execute function
        >>> dict_stacked = dict_stack([dict1_, dict2_])
        >>> # verify results
        >>> result = str(dict_stacked)
        >>> print(result)
        {'a': [1, 2], 'c': [4], 'b': [2, 3]}

    """
    dict_stacked_ = defaultdict(list)
    for dict_ in dict_list:
        for key, val in six.iteritems(dict_):
            dict_stacked_[key_prefix + key].append(val)
    dict_stacked = dict(dict_stacked_)
    return dict_stacked


def invert_dict(dict_):
    """
    invert_dict

    Args:
        dict_ (dict_):

    Returns:
        dict: inverted_dict

    CommandLine:
        python -m utool.util_dict --test-invert_dict

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = {'a': 1, 'b': 2}
        >>> inverted_dict = invert_dict(dict_)
        >>> result = inverted_dict
        >>> print(result)
        {1: 'a', 2: 'b'}

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = OrderedDict([(2, 'good',), (1, 'ok',), (0, 'junk',), (None, 'UNKNOWN',),])
        >>> inverted_dict = invert_dict(dict_)
        >>> result = inverted_dict
        >>> print(result)
        OrderedDict([('good', 2), ('ok', 1), ('junk', 0), ('UNKNOWN', None)])

    """
    inverted_items = [(val, key) for key, val in six.iteritems(dict_)]
    inverted_dict = type(dict_)(inverted_items)
    return inverted_dict


def iter_all_dict_combinations_ordered(varied_dict):
    """
    Same as all_dict_combinations but preserves order
    """
    tups_list = [[(key, val) for val in val_list]
                 for (key, val_list) in six.iteritems(varied_dict)]
    dict_iter = (OrderedDict(tups) for tups in iprod(*tups_list))
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
        >>> result = str(ut.list_str(dict_list))
        >>> print(result)
        [
            {'pipeline_root': 'vsmany', 'sv_on': True, 'logdist_weight': 0.0,},
            {'pipeline_root': 'vsmany', 'sv_on': True, 'logdist_weight': 1.0,},
            {'pipeline_root': 'vsmany', 'sv_on': False, 'logdist_weight': 0.0,},
            {'pipeline_root': 'vsmany', 'sv_on': False, 'logdist_weight': 1.0,},
            {'pipeline_root': 'vsmany', 'sv_on': None, 'logdist_weight': 0.0,},
            {'pipeline_root': 'vsmany', 'sv_on': None, 'logdist_weight': 1.0,},
        ]

    Ignore:
        print(x)
        print(y)

        print(ut.hz_str('\n'.join(list(x)), '\n'.join(list(y))))
    """
    tups_list = [[(key, val) for val in val_list] if isinstance(val_list, (list, tuple)) else [(key, val_list)]
                 for (key, val_list) in six.iteritems(varied_dict)]
    dict_list = [dict(tups) for tups in iprod(*tups_list)]
    #dict_list = [{key: val for (key, val) in tups} for tups in iprod(*tups_list)]
    #from collections import OrderedDict
    #dict_list = [OrderedDict([(key, val) for (key, val) in tups]) for tups in iprod(*tups_list)]
    return dict_list


def all_dict_combinations_lbls(varied_dict):
    """ returns a label for each variation in a varydict.
    It tries to not be oververbose and returns only what parameters are varied
    in each label.

    CommandLine:
        python -m utool.util_dict --test-all_dict_combinations_lbls

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> varied_dict = {'logdist_weight': [0.0, 1.0], 'pipeline_root': ['vsmany'], 'sv_on': [True, False, None]}
        >>> comb_lbls = utool.all_dict_combinations_lbls(varied_dict)
        >>> result = (utool.list_str(comb_lbls))
        >>> print(result)
        [
            "(('sv_on', True), ('logdist_weight', 0.0))",
            "(('sv_on', True), ('logdist_weight', 1.0))",
            "(('sv_on', False), ('logdist_weight', 0.0))",
            "(('sv_on', False), ('logdist_weight', 1.0))",
            "(('sv_on', None), ('logdist_weight', 0.0))",
            "(('sv_on', None), ('logdist_weight', 1.0))",
        ]

    #ut.list_type_profile(comb_lbls)
    """
    multitups_list = [[(key, val) for val in val_list]
                      for key, val_list in six.iteritems(varied_dict)
                      if len(val_list) > 1]
    comb_lbls = list(map(str, list(iprod(*multitups_list))))
    return comb_lbls


def dict_union2(dict1, dict2):
    return dict(list(dict1.items()) + list(dict2.items()))


def dict_union(*args):
    return dict([item for dict_ in iter(args) for item in six.iteritems(dict_)])


def items_sorted_by_value(dict_):
    sorted_items = sorted(six.iteritems(dict_), key=lambda k, v: v[1])
    return sorted_items


def keys_sorted_by_value(dict_):
    sorted_keys = sorted(dict_, key=lambda key: dict_[key])
    return sorted_keys


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
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> # build test data
        >>> key_list = [  1,   2,   2,   3,   1]
        >>> val_list = ['a', 'b', 'c', 'd', 'e']
        >>> # execute function
        >>> key_to_vals = build_conflict_dict(key_list, val_list)
        >>> # verify results
        >>> result = ut.dict_str(key_to_vals)
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


def updateif_haskey(dict1, dict2):
    r"""
    updates vals in dict1 using vals from dict2 only if the
    key is already in dict1.

    Args:
        dict1 (dict)
        dict2 (dict):

    CommandLine:
        python -m utool.util_dict --test-updateif_haskey

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> # build test data
        >>> dict1 = {'a': 1, 'b': 2, 'c': 3}
        >>> dict2 = {'a': 2, 'd': 3}
        >>> # execute function
        >>> result = updateif_haskey(dict1, dict2)
        >>> assert 'd' not in dict1
        >>> assert dict1['a'] == 2
        >>> assert result is dict1
        >>> # verify results
        >>> print(result)
    """
    for key, val in six.iteritems(dict2):
        if key in dict1:
            dict1[key] = val
    return dict1


def dict_update_newkeys(dict_, dict2):
    """ Like dict.update, but does not overwrite items """
    for key, val in six.iteritems(dict2):
        if key not in dict_:
            dict_[key] = val


def is_dicteq(dict1_, dict2_, almosteq_ok=True, verbose_err=True):
    """ Checks to see if dicts are the same. Performs recursion. Handles numpy """
    import utool
    from utool import util_alg
    from utool import util_dbg
    assert len(dict1_) == len(dict2_), 'dicts are not of same length'
    try:
        for (key1, val1), (key2, val2) in zip(dict1_.items(), dict2_.items()):
            assert key1 == key2, 'key mismatch'
            assert type(val1) == type(val2), 'vals are not same type'
            if HAS_NUMPY and np.iterable(val1):
                if almosteq_ok and utool.is_float(val1):
                    assert np.all(util_alg.almost_eq(val1, val2)), 'float vals are not within thresh'
                else:
                    assert all([np.all(x1 == x2) for (x1, x2) in zip(val1, val2)]), 'np vals are different'
            elif isinstance(val1, dict):
                is_dicteq(val1, val2, almosteq_ok=almosteq_ok, verbose_err=verbose_err)
            else:
                assert val1 == val2, 'vals are different'
    except AssertionError as ex:
        if verbose_err:
            util_dbg.printex(ex)
        return False
    return True


def dict_subset(dict_, keys):
    r"""
    Args:
        dict_ (dict):
        keys (list):

    Returns:
        dict: subset dictionary

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = {'K': 3, 'dcvs_clip_max': 0.2, 'p': 0.1}
        >>> keys = ['K', 'dcvs_clip_max']
        >>> result = dict_subset(dict_, keys)
        >>> print(result)
        {'K': 3, 'dcvs_clip_max': 0.2}
    """
    subdict_ = {key: dict_[key] for key in keys}
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
    in place deletion if keys exist

    Args:
        dict_ (?):
        key_list (list):

    CommandLine:
        python -m utool.util_dict --test-delete_dict_keys

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> # build test data
        >>> dict_ = {'bread': 1, 'churches': 1, 'cider': 2, 'very small rocks':2}
        >>> key_list = ['duck', 'bread', 'cider']
        >>> # execute function
        >>> delete_dict_keys(dict_, key_list)
        >>> # verify results
        >>> result = ut.dict_str(dict_, newlines=False)
        >>> print(result)
        {'churches': 1, 'very small rocks': 2,}

    """
    invalid_keys = set(key_list) - set(six.iterkeys(dict_))
    valid_keys = set(key_list) - invalid_keys
    for key in valid_keys:
        del dict_[key]
    return dict_


def dict_take_gen(dict_, keys, *d):
    r"""
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
        >>> dict_ = {1: 'a', 2: 'b', 3: 'c'}
        >>> keys = [1, 2, 3, 4, 5]
        >>> result = list(dict_take_gen(dict_, keys, None))
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
    if len(d) == 0:
        # no default given throws key error
        dictget = dict_.__getitem__
    elif len(d) == 1:
        # default given does not throw key erro
        dictget = dict_.get
    else:
        raise ValueError('len(d) must be 1 or 0')
    for key in keys:
        if HAS_NUMPY and isinstance(key, np.ndarray):
            # recursive call
            yield list(dict_take_gen(dict_, key, *d))
        else:
            yield dictget(key, *d)


def dict_take_list(dict_, keys, *d):
    return list(dict_take_gen(dict_, keys, *d))
    #return [dict_[key] for key in keys]


def dict_take(dict_, keys, *d):
    """ alias """
    try:
        return dict_take_list(dict_, keys, *d)
    except TypeError:
        return dict_take_list(dict_, [keys], *d)[0]


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
        >>> print('before: ' + ut.dict_str(dict_))
        >>> result = list(dict_take_pop(dict_, keys, None))
        >>> print('after: ' + ut.dict_str(dict_))
        >>> assert len(dict_) == 2
        >>> print(result)
        ['a', 'b', 'c', None, None]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> dict_ = {1: 'a', 2: 'b', 3: 'c'}
        >>> keys = [1, 2, 3, 4, 5]
        >>> print('before: ' + ut.dict_str(dict_))
        >>> try:
        >>>     print(list(dict_take_pop(dict_, keys)))
        >>>     result = 'did not get key error'
        >>> except KeyError:
        >>>     result = 'correctly got key error'
        >>> assert len(dict_) == 0
        >>> print('after: ' + ut.dict_str(dict_))
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


def order_dict_by(dict_, key_order):
    sorted_item_list = [(key, dict_[key]) for key in key_order if key in dict_]
    sorted_dict = OrderedDict(sorted_item_list)
    return sorted_dict


def get_dict_column(dict_, colx):
    r"""
    Args:
        dict_ (dict_):  a dictionary
        colx (?):

    CommandLine:
        python -m utool.util_dict --test-get_dict_column

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = {'a': [0, 1, 2], 'b': [3, 4, 5], 'c': [6, 7, 8]}
        >>> colx = [2, 0]
        >>> retdict_ = get_dict_column(dict_, colx)
        >>> result = str(retdict_)
        >>> print(result)
        {'a': [2, 0], 'c': [8, 6], 'b': [5, 3]}
    """
    retdict_ = {key: util_list.list_take(val, colx) for key, val in six.iteritems(dict_)}
    return retdict_


def dictinfo(dict_):
    """
    dictinfo

    In depth debugging info

    Args:
        dict_ (dict):

    Returns:
        str

    Example:
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
            val_shape_stats_str = ut.dict_str(val_shape_stats, strvals=True, newlines=False)
            val_dtypes = list(set([val.dtype for val in vals]))
            fmtstr_ += ut.unindent('''
            * val_shape_stats = {val_shape_stats_str}
            * val_dtypes = {val_dtypes}
            '''.strip('\n'))
        elif val_types[0] == list:
            # each key holds a list
            val_len_stats =  ut.get_stats(set(map(len, vals)))
            val_len_stats_str = ut.dict_str(val_len_stats, strvals=True, newlines=False)
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
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> # build test data
        >>> dict_ = {'default': 1, 'hierarchical': 5, 'linear': 0, 'kdtree': 1,
        ...          'composite': 3, 'autotuned': 255, 'saved': 254, 'kmeans': 2,
        ...          'lsh': 6, 'kdtree_single': 4}
        >>> val_list = [1]
        >>> # execute function
        >>> found_dict = dict_find_keys(dict_, val_list)
        >>> # verify results
        >>> result = str(found_dict)
        >>> print(result)
        {1: ['kdtree', 'default']}
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
        >>> # build test data
        >>> dict_ = {'default': 1, 'hierarchical': 5, 'linear': 0, 'kdtree': 1,
        ...          'composite': 3, 'autotuned': 255, 'saved': 254, 'kmeans': 2,
        ...          'lsh': 6, 'kdtree_single': 4}
        >>> key = 'default'
        >>> # execute function
        >>> found_dict = dict_find_keys(dict_, val_list)
    """
    value = dict_[key]
    found_dict = dict_find_keys(dict_, [value])
    other_keys = found_dict[value]
    other_keys.remove(key)
    return other_keys


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_dict; utool.doctest_funcs(utool.util_dict, allexamples=True)"
        python -c "import utool, utool.util_dict; utool.doctest_funcs(utool.util_dict)"
        python -m utool.util_dict
        python -m utool.util_dict --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
