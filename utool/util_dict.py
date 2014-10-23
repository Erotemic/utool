from __future__ import absolute_import, division, print_function
from itertools import product as iprod
from six.moves import zip
from collections import defaultdict
from .util_inject import inject
import six
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    pass
print, print_, printDBG, rrr, profile = inject(__name__, '[dict]')


def all_dict_combinations(varied_dict):
    """
    Input: a dict with lists of possible parameter settings
    Output: a list of dicts correpsonding to all combinations of params settings
     """

    tups_list = [[(key, val) for val in val_list]
                 for (key, val_list) in six.iteritems(varied_dict)]
    dict_list = [{key: val for (key, val) in tups} for tups in iprod(*tups_list)]
    return dict_list


def all_dict_combinations_lbls(varied_dict):
    """ returns a label for each variation in a varydict.
    It tries to not be oververbose and returns only what parameters are varied
    in each label.

    Example:
        >>> import utool
        >>> varied_dict = {'logdist_weight': [0.0, 1.0], 'pipeline_root': ['vsmany'], 'sv_on': [True, False, None]}
        >>> comb_lbls = utool.all_dict_combinations_lbls(varied_dict)
        >>> print(utool.list_str(comb_lbls))
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
    from . import util_alg
    from . import util_dbg
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
