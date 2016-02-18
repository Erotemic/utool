# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import re
import operator
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__, '[util_tags]')


def filterflags_general_tags(tags_list, has_any=None, has_all=None,
                             has_none=None, min_num=None, max_num=None,
                             any_startswith=None, any_endswith=None,
                             any_match=None, none_match=None, logic='and',
                             ignore_case=True):
    r"""
    maybe integrate into utool? Seems pretty general

    Args:
        tags_list (list):
        has_any (None): (default = None)
        has_all (None): (default = None)
        min_num (None): (default = None)
        max_num (None): (default = None)

    CommandLine:
        python -m utool.util_tags --exec-filterflags_general_tags
        python -m utool.util_tags --exec-filterflags_general_tags:0  --helpx
        python -m utool.util_tags --exec-filterflags_general_tags:0
        python -m utool.util_tags --exec-filterflags_general_tags:0  --none_match n
        python -m utool.util_tags --exec-filterflags_general_tags:0  --has_none=n,o
        python -m utool.util_tags --exec-filterflags_general_tags:1
        python -m utool.util_tags --exec-filterflags_general_tags:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tags import *  # NOQA
        >>> import utool as ut
        >>> tags_list = [['v'], [], ['P'], ['P', 'o'], ['n', 'o',], [], ['n', 'N'], ['e', 'i', 'p', 'b', 'n'], ['q', 'v'], ['n'], ['n'], ['N']]
        >>> kwargs = ut.argparse_dict(ut.get_kwdefaults2(filterflags_general_tags), type_hint=list)
        >>> print('kwargs = %r' % (kwargs,))
        >>> flags = filterflags_general_tags(tags_list, **kwargs)
        >>> print(flags)
        >>> result = ut.compress(tags_list, flags)
        >>> print('result = %r' % (result,))

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tags import *  # NOQA
        >>> import utool as ut
        >>> tags_list = [['v'], [], ['P'], ['P'], ['n', 'o',], [], ['n', 'N'], ['e', 'i', 'p', 'b', 'n'], ['n'], ['n'], ['N']]
        >>> has_all = 'n'
        >>> min_num = 1
        >>> flags = filterflags_general_tags(tags_list, has_all=has_all, min_num=min_num)
        >>> result = ut.compress(tags_list, flags)
        >>> print('result = %r' % (result,))

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tags import *  # NOQA
        >>> import utool as ut
        >>> tags_list = [['vn'], ['vn', 'no'], ['P'], ['P'], ['n', 'o',], [], ['n', 'N'], ['e', 'i', 'p', 'b', 'n'], ['n'], ['n', 'nP'], ['NP']]
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

    def fix_tags(tags):
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

    tags_list_ = [fix_tags(tags_) for tags_ in tags_list]
    flags = default_func(len(tags_list_), dtype=np.bool)

    if min_num is not None:
        flags_ = [len(tags_) >= min_num for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if max_num is not None:
        flags_ = [len(tags_) <= max_num for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if has_any is not None:
        has_any = fix_tags(set(ut.ensure_iterable(has_any)))
        flags_ = [len(has_any.intersection(tags_)) > 0 for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if has_none is not None:
        has_none = fix_tags(set(ut.ensure_iterable(has_none)))
        flags_ = [len(has_none.intersection(tags_)) == 0 for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if has_all is not None:
        has_all = fix_tags(set(ut.ensure_iterable(has_all)))
        flags_ = [len(has_all.intersection(tags_)) == len(has_all) for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    def test_item(tags_, fields, op, compare):
        t_flags = [any([compare(t, f) for f in fields]) for t in tags_]
        num_passed = sum(t_flags)
        flag = op(num_passed, 0)
        return flag

    def flag_tags(tags_list, fields, op, compare):
        flags = [test_item(tags_, fields, op, compare) for tags_ in tags_list_]
        return flags

    def execute_filter(flags, tags_list, fields, op, compare):
        if fields is not None:
            fields = ut.ensure_iterable(fields)
            if ignore_case:
                fields = [f.lower() for f in fields]
            flags_ = flag_tags(tags_list, fields, op, compare)
            logic_func(flags, flags_, out=flags)
        return flags

    flags = execute_filter(
        flags, tags_list, any_startswith,
        operator.gt, six.text_type.startswith)

    flags = execute_filter(
        flags, tags_list, any_endswith,
        operator.gt, six.text_type.endswith)

    flags = execute_filter(
        flags, tags_list, any_match,
        operator.gt, lambda t, f: re.match(f, t))

    flags = execute_filter(
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
