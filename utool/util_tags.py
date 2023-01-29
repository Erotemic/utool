# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import re
import operator
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)


def modify_tags(tags_list, direct_map=None, regex_map=None, regex_aug=None,
                delete_unmapped=False, return_unmapped=False,
                return_map=False):
    import utool as ut
    tag_vocab = ut.unique(ut.flatten(tags_list))
    alias_map = ut.odict()
    if regex_map is not None:
        alias_map.update(**ut.build_alias_map(regex_map, tag_vocab))
    if direct_map is not None:
        alias_map.update(ut.odict(direct_map))

    new_tags_list = tags_list
    new_tags_list = ut.alias_tags(new_tags_list, alias_map)

    if regex_aug is not None:
        alias_aug = ut.build_alias_map(regex_aug, tag_vocab)
        aug_tags_list = ut.alias_tags(new_tags_list, alias_aug)
        new_tags_list = [ut.unique(t1 + t2) for t1, t2 in zip(new_tags_list, aug_tags_list)]

    unmapped = list(set(tag_vocab) - set(alias_map.keys()))
    if delete_unmapped:
        new_tags_list = [ut.setdiff(tags, unmapped) for tags in new_tags_list]

    toreturn = None
    if return_map:
        toreturn = (alias_map,)

    if return_unmapped:
        toreturn = toreturn + (unmapped,)

    if toreturn is None:
        toreturn = new_tags_list
    else:
        toreturn = (new_tags_list,) + toreturn
    return toreturn


def tag_coocurrence(tags_list):
    import utool as ut
    co_occur_list = []
    for tags in tags_list:
        for combo in ut.combinations(tags, 2):
            key = tuple(sorted(combo))
            co_occur_list.append(key)
    co_occur = ut.dict_hist(co_occur_list, ordered=True)
    #        co_occur[key] += 1
    #co_occur = ut.odict(co_occur)
    return co_occur


def tag_hist(tags_list):
    import utool as ut
    return ut.dict_hist(ut.flatten(tags_list), ordered=True)


def build_alias_map(regex_map, tag_vocab):
    """
    Constructs explicit mapping. Order of items in regex map matters.
    Items at top are given preference.

    Example:
        >>> # DISABLE_DOCTEST
        >>> tags_list = [['t1', 't2'], [], ['t3'], ['t4', 't5']]
        >>> tag_vocab = ut.flat_unique(*tags_list)
        >>> regex_map = [('t[3-4]', 'A9'), ('t0', 'a0')]
        >>> unmapped = list(set(tag_vocab) - set(alias_map.keys()))
    """
    import utool as ut
    import re
    alias_map = ut.odict([])
    for pats, new_tag in reversed(regex_map):
        pats = ut.ensure_iterable(pats)
        for pat in pats:
            flags = [re.match(pat, t) for t in tag_vocab]
            for old_tag in ut.compress(tag_vocab, flags):
                alias_map[old_tag] = new_tag
    identity_map = ut.take_column(regex_map, 1)
    for tag in ut.filter_Nones(identity_map):
        alias_map[tag] = tag
    return alias_map


def alias_tags(tags_list, alias_map):
    """
    update tags to new values

    Args:
        tags_list (list):
        alias_map (list): list of 2-tuples with regex, value

    Returns:
        list: updated tags

    CommandLine:
        python -m utool.util_tags alias_tags --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_tags import *  # NOQA
        >>> import utool as ut
        >>> tags_list = [['t1', 't2'], [], ['t3'], ['t4', 't5']]
        >>> ut.build_alias_map()
        >>> result = alias_tags(tags_list, alias_map)
        >>> print(result)
    """
    def _alias_dict(tags):
        tags_ = [alias_map.get(t, t) for t in tags]
        return list(set([t for t in tags_ if t is not None]))
    tags_list_ = [_alias_dict(tags) for tags in tags_list]
    return tags_list_
    # def _fix_tags(tags):
    #     return {six.text_type(t.lower()) for t in tags}
    # tags_list_ = list(map(_fix_tags, tags_list))
    # re_list = [re.compile(pat) for pat, val in alias_map]
    # val_list = ut.take_column(alias_map, 0)
    # def _alias_regex(tags):
    #     new_tags = 0
    #     for t in tags:
    #         matched = [re_.match(t) is not None for re_ in re_list]
    #         matched_idx = ut.where(matched)
    #         assert len(matched_idx) <= 1, 'more than one tag in %r matched pattern' % (tags,)
    #         if len(matched_idx) > 0:
    #             repl_tags = ut.take(val_list, matched_idx)
    #             new_tags.extend(repl_tags)
    #         else:
    #             new_tags.append(t)
    #     return new_tags
    # # tags_list_ = [_alias_regex(tags) for tags in tags_list_]
    # return tags_list_


def filterflags_general_tags(tags_list, has_any=None, has_all=None,
                             has_none=None, min_num=None, max_num=None,
                             any_startswith=None, any_endswith=None,
                             in_any=None, any_match=None, none_match=None,
                             logic='and', ignore_case=True):
    r"""
    maybe integrate into utool? Seems pretty general

    Args:
        tags_list (list):
        has_any (None): (default = None)
        has_all (None): (default = None)
        min_num (None): (default = None)
        max_num (None): (default = None)

    Notes:
        in_any should probably be ni_any

    TODO: make this function more natural

    CommandLine:
        python -m utool.util_tags --exec-filterflags_general_tags
        python -m utool.util_tags --exec-filterflags_general_tags:0  --helpx
        python -m utool.util_tags --exec-filterflags_general_tags:0
        python -m utool.util_tags --exec-filterflags_general_tags:0  --none_match n
        python -m utool.util_tags --exec-filterflags_general_tags:0  --has_none=n,o
        python -m utool.util_tags --exec-filterflags_general_tags:1
        python -m utool.util_tags --exec-filterflags_general_tags:2

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tags import *  # NOQA
        >>> import utool as ut
        >>> tags_list = [['v'], [], ['P'], ['P', 'o'], ['n', 'o'], [], ['n', 'N'], ['e', 'i', 'p', 'b', 'n'], ['q', 'v'], ['n'], ['n'], ['N']]
        >>> kwargs = ut.argparse_dict(ut.get_kwdefaults2(filterflags_general_tags), type_hint=list)
        >>> print('kwargs = %r' % (kwargs,))
        >>> flags = filterflags_general_tags(tags_list, **kwargs)
        >>> print(flags)
        >>> result = ut.compress(tags_list, flags)
        >>> print('result = %r' % (result,))

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tags import *  # NOQA
        >>> import utool as ut
        >>> tags_list = [['v'], [], ['P'], ['P'], ['n', 'o'], [], ['n', 'N'], ['e', 'i', 'p', 'b', 'n'], ['n'], ['n'], ['N']]
        >>> has_all = 'n'
        >>> min_num = 1
        >>> flags = filterflags_general_tags(tags_list, has_all=has_all, min_num=min_num)
        >>> result = ut.compress(tags_list, flags)
        >>> print('result = %r' % (result,))

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tags import *  # NOQA
        >>> import utool as ut
        >>> tags_list = [['vn'], ['vn', 'no'], ['P'], ['P'], ['n', 'o'], [], ['n', 'N'], ['e', 'i', 'p', 'b', 'n'], ['n'], ['n', 'nP'], ['NP']]
        >>> kwargs = {
        >>>     'any_endswith': 'n',
        >>>     'any_match': None,
        >>>     'any_startswith': 'n',
        >>>     'has_all': None,
        >>>     'has_any': None,
        >>>     'has_none': None,
        >>>     'max_num': 3,
        >>>     'min_num': 1,
        >>>     'none_match': ['P'],
        >>> }
        >>> flags = filterflags_general_tags(tags_list, **kwargs)
        >>> filtered = ut.compress(tags_list, flags)
        >>> result = ('result = %s' % (ut.repr2(filtered),))
        result = [['vn', 'no'], ['n', 'o'], ['n', 'N'], ['n'], ['n', 'nP']]
    """
    import numpy as np
    import utool as ut

    def _fix_tags(tags):
        if ignore_case:
            return set([]) if tags is None else {six.text_type(t.lower()) for t in tags}
        else:
            return set([]) if tags is None else {six.text_type() for t in tags}

    if logic is None:
        logic = 'and'

    logic_func = {
        'and': np.logical_and,
        'or': np.logical_or,
    }[logic]

    default_func = {
        'and': np.ones,
        'or': np.zeros,
    }[logic]

    tags_list_ = [_fix_tags(tags_) for tags_ in tags_list]
    flags = default_func(len(tags_list_), dtype=bool)

    if min_num is not None:
        flags_ = [len(tags_) >= min_num for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if max_num is not None:
        flags_ = [len(tags_) <= max_num for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if has_any is not None:
        has_any = _fix_tags(set(ut.ensure_iterable(has_any)))
        flags_ = [len(has_any.intersection(tags_)) > 0 for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if has_none is not None:
        has_none = _fix_tags(set(ut.ensure_iterable(has_none)))
        flags_ = [len(has_none.intersection(tags_)) == 0 for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if has_all is not None:
        has_all = _fix_tags(set(ut.ensure_iterable(has_all)))
        flags_ = [len(has_all.intersection(tags_)) == len(has_all) for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    def _test_item(tags_, fields, op, compare):
        t_flags = [any([compare(t, f) for f in fields]) for t in tags_]
        num_passed = sum(t_flags)
        flag = op(num_passed, 0)
        return flag

    def _flag_tags(tags_list, fields, op, compare):
        flags = [_test_item(tags_, fields, op, compare) for tags_ in tags_list_]
        return flags

    def _exec_filter(flags, tags_list, fields, op, compare):
        if fields is not None:
            fields = ut.ensure_iterable(fields)
            if ignore_case:
                fields = [f.lower() for f in fields]
            flags_ = _flag_tags(tags_list, fields, op, compare)
            logic_func(flags, flags_, out=flags)
        return flags

    flags = _exec_filter(
        flags, tags_list, any_startswith,
        operator.gt, six.text_type.startswith)

    flags = _exec_filter(
        flags, tags_list, in_any,
        operator.gt, operator.contains)

    flags = _exec_filter(
        flags, tags_list, any_endswith,
        operator.gt, six.text_type.endswith)

    flags = _exec_filter(
        flags, tags_list, any_match,
        operator.gt, lambda t, f: re.match(f, t))

    flags = _exec_filter(
        flags, tags_list, none_match,
        operator.eq, lambda t, f: re.match(f, t))
    return flags


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m utool.util_tags
        python -m utool.util_tags --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
