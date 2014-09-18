from __future__ import absolute_import, division, print_function
from itertools import product as iprod
from six.moves import zip
from collections import defaultdict
from .util_inject import inject
import six
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
    """ returns what parameters are varied"""
    multitups_list = [[(key, val) for val in val_list] for key, val_list in six.iteritems(varied_dict) if len(val_list) > 1]
    comb_lbls = map(str, list(iprod(*multitups_list)))
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
