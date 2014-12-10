from __future__ import absolute_import, division, print_function
from itertools import product as iprod
from six.moves import zip
from collections import defaultdict
from utool.util_inject import inject
import six
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    pass
print, print_, printDBG, rrr, profile = inject(__name__, '[dict]')


def invert_dict(dict_):
    """
    invert_dict

    Args:
        dict_ (dict_):

    Returns:
        dict: inverted_dict

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = {'a': 1, 'b': 2}
        >>> inverted_dict = invert_dict(dict_)
        >>> result = inverted_dict
        >>> print(result)
        {1: 'a', 2: 'b'}
    """
    inverted_dict = {val: key for key, val in six.iteritems(dict_)}
    return inverted_dict


def all_dict_combinations_ordered(varied_dict):
    """
    Same as all_dict_combinations but preserves order
    """
    from collections import OrderedDict
    tups_list = [[(key, val) for val in val_list]
                 for (key, val_list) in six.iteritems(varied_dict)]
    dict_list = [OrderedDict(tups) for tups in iprod(*tups_list)]
    return dict_list


def all_dict_combinations(varied_dict):
    """
    all_dict_combinations

    Args:
        varied_dict (dict):  a dict with lists of possible parameter settings

    Returns:
        list: dict_list a list of dicts correpsonding to all combinations of params settings

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> varied_dict = {'logdist_weight': [0.0, 1.0], 'pipeline_root': ['vsmany'], 'sv_on': [True, False, None]}
        >>> dict_list = all_dict_combinations(varied_dict)
        >>> result = str(ut.list_str(dict_list))
        >>> print(result)
        [
            {'pipeline_root': 'vsmany', 'sv_on': True, 'logdist_weight': 0.0},
            {'pipeline_root': 'vsmany', 'sv_on': True, 'logdist_weight': 1.0},
            {'pipeline_root': 'vsmany', 'sv_on': False, 'logdist_weight': 0.0},
            {'pipeline_root': 'vsmany', 'sv_on': False, 'logdist_weight': 1.0},
            {'pipeline_root': 'vsmany', 'sv_on': None, 'logdist_weight': 0.0},
            {'pipeline_root': 'vsmany', 'sv_on': None, 'logdist_weight': 1.0},
        ]
    """
    tups_list = [[(key, val) for val in val_list]
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

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> varied_dict = {'logdist_weight': [0.0, 1.0], 'pipeline_root': ['vsmany'], 'sv_on': [True, False, None]}
        >>> comb_lbls = utool.all_dict_combinations_lbls(varied_dict)
        >>> result = (utool.list_str(comb_lbls))
        >>> print(result)
        [
            (('sv_on', True), ('logdist_weight', 0.0)),
            (('sv_on', True), ('logdist_weight', 1.0)),
            (('sv_on', False), ('logdist_weight', 0.0)),
            (('sv_on', False), ('logdist_weight', 1.0)),
            (('sv_on', None), ('logdist_weight', 0.0)),
            (('sv_on', None), ('logdist_weight', 1.0)),
        ]

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
    Builds dict where a list of values is associated with a key
    """
    key_to_vals = defaultdict(list)
    for key, val in zip(key_list, val_list):
        key_to_vals[key].append(val)
    return key_to_vals


def updateif_haskey(dict1, dict2):
    for key, val in six.iteritems(dict2):
        if key in dict1:
            dict1[key] = val


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
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = '?'
        >>> keys = '?'
        >>> result = dict_subset(dict_, keys)
        >>> print(result)
    """
    return {key: dict_[key] for key in keys}


def dict_take_gen(dict_, keys):
    hasnumpy = 'np' in vars()
    for key in keys:
        if hasnumpy and isinstance(key, np.ndarray):
            yield list(dict_take_gen(dict_, key))
        else:
            yield dict_[key]


def dict_take_list(dict_, keys):
    return list(dict_take_gen(dict_, keys))
    #return [dict_[key] for key in keys]


def dict_where_len0(dict_):
    keys = np.array(dict_.keys())
    flags = np.array(list(map(len, dict_.values()))) == 0
    indices = np.where(flags)[0]
    return keys[indices]


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
