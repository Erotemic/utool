# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    # TODO remove numpy
    pass
import operator
from six.moves import zip
from utool import util_iter
from utool import util_alg
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)
from utool import util_arg  # NOQA


def get_first_None_position(list_):
    for index, item in enumerate(list_):
        if item is None:
            return index
    return None


def assert_raises(ex_type, func, *args, **kwargs):
    r"""
    Checks that a function raises an error when given specific arguments.

    Args:
        ex_type (Exception): exception type
        func (callable): live python function

    CommandLine:
        python -m utool.util_assert assert_raises --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_assert import *  # NOQA
        >>> import utool as ut
        >>> ex_type = AssertionError
        >>> func = len
        >>> # Check that this raises an error when something else does not
        >>> assert_raises(ex_type, assert_raises, ex_type, func, [])
        >>> # Check this does not raise an error when something else does
        >>> assert_raises(ValueError, [].index, 0)
    """
    try:
        func(*args, **kwargs)
    except Exception as ex:
        assert isinstance(ex, ex_type), (
            'Raised %r but type should have been %r' % (ex, ex_type))
        return True
    else:
        raise AssertionError('No error was raised')


def assert_unique(item_list, ignore=[], name='list', verbose=None):
    import utool as ut
    dups = ut.find_duplicate_items(item_list)
    ut.delete_dict_keys(dups, ignore)
    if len(dups) > 0:
        raise AssertionError(
            'Found duplicate items in %s: %s' % (
                name, ut.repr4(dups)))
    if verbose:
        print('No duplicates found in %s' % (name,))


def assert_all_in(key_list, valid_list, msg=''):
    missing_keys = set(key_list).difference(set(valid_list))
    assert len(missing_keys) == 0, 'missing_keys = %r. %s' % (missing_keys, msg)


def assert_all_not_None(list_, list_name='some_list', key_list=[], verbose=not
                        util_arg.QUIET, veryverbose=False):
    if util_arg.NO_ASSERTS:
        return
    try:
        index = get_first_None_position(list_)
        assert index is None, 'index=%r in %s is None' % (index, list_name)
        if veryverbose:
            print('PASSED: %s has no Nones' % (list_name))
    except AssertionError as ex:
        from utool import util_dbg
        item = list_[index]
        msg = (list_name + '[%d] = %r') % (index, item)
        if verbose:
            msg += '\n len(list_) = %r' % (len(list_))
        util_dbg.printex(ex, msg, keys=key_list, N=1)
        raise


def assert_unflat_level(unflat_list, level=1, basetype=None):
    if util_arg.NO_ASSERTS:
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


def assert_scalar_list(list_):
    if util_arg.NO_ASSERTS:
        return
    for count, item in enumerate(list_):
        assert not util_iter.isiterable(item), 'count=%r, item=%r is iterable!' % (count, item)


def assert_same_len(list1, list2, additional_msg=''):
    if util_arg.NO_ASSERTS:
        return
    assert len(list1) == len(list2), (
        'unequal lens. len(list1)=%r, len(list2)=%r%s' % (
            len(list1), len(list2), additional_msg))


assert_eq_len = assert_same_len


def lists_eq(list1, list2):
    """ recursive """
    if len(list1) != len(list2):
        return False
    for count, (item1, item2) in enumerate(zip(list1, list2)):
        if isinstance(item1, np.ndarray) or isinstance(item2, np.ndarray):
            failed = not np.all(item1 == item2)  # lists_eq(item1, item2)
        else:
            failed = item1 != item2
        if failed:
            return False
    return True


def assert_lists_eq(list1, list2, failmsg='', verbose=False):
    if util_arg.NO_ASSERTS:
        return
    msg = ''
    if len(list1) != len(list2):
        msg += ('LENGTHS ARE UNEQUAL: len(list1)=%r, len(list2)=%r\n' % (len(list1), len(list2)))

    difflist = []
    for count, (item1, item2) in enumerate(zip(list1, list2)):
        if item1 != item2:
            difflist.append('count=%r, item1=%r, item2=%r' % (count, item1, item2))

    length = max(len(list1), len(list2))

    if verbose or len(difflist) < 10:
        msg += '\n'.join(difflist)
    else:
        if len(difflist) > 0:
            msg += 'There are %d/%d different ordered items\n' % (len(difflist), length)

    if len(msg) > 0:
        intersecting_items = set(list1).intersection(set(list2))
        missing_items1 = set(list2).difference(intersecting_items)
        missing_items2 = set(list1).difference(intersecting_items)
        num_intersect = len(intersecting_items)
        isect_msg = 'There are %d/%d intersecting unordered items' % (num_intersect, length)
        msg = failmsg + '\n' + msg + isect_msg
        if len(missing_items1) > 0:
            msg += '\n %d items are missing from list1' % (len(missing_items1))
            msg += '\n missing_items1 = %r' % (missing_items1,)
        if len(missing_items2) > 0:
            msg += '\n %d items are missing from list2' % (len(missing_items2))
            msg += '\n missing_items2 = %r' % (missing_items2,)
        ex = AssertionError(msg)
        if verbose:
            print(msg)
        raise ex


def assert_inbounds(num, low, high, msg='', eq=False, verbose=not util_arg.QUIET):
    r"""
    Args:
        num (scalar):
        low (scalar):
        high (scalar):
        msg (str):
    """
    from utool import util_str
    if util_arg.NO_ASSERTS:
        return
    passed = util_alg.inbounds(num, low, high, eq=eq)
    if isinstance(passed, np.ndarray):
        passflag = np.all(passed)
    else:
        passflag = passed
    if not passflag:
        failednum = num.compress(~passed) if isinstance(num, np.ndarray) else num
        failedlow = low.compress(~passed) if isinstance(low, np.ndarray) else low
        failedhigh = high.compress(~passed) if isinstance(high, np.ndarray) else high
        msg_ = 'num=%r is out of bounds=(%r, %r)' % (failednum, failedlow, failedhigh)
        raise AssertionError(msg_ + '\n' + msg)
    else:
        if verbose:
            op = '<=' if eq else '<'
            fmtstr = 'Passed assert_inbounds: {low} {op} {num} {op} {high}'
            print(fmtstr.format(low=low, op=op, num=util_str.truncate_str(str(num)), high=high))


def assert_almost_eq(arr_test, arr_target, thresh=1E-11):
    r"""
    Args:
        arr_test (ndarray or list):
        arr_target (ndarray or list):
        thresh (scalar or ndarray or list):
    """
    if util_arg.NO_ASSERTS:
        return
    import utool as ut
    arr1 = np.array(arr_test)
    arr2 = np.array(arr_target)
    passed, error = ut.almost_eq(arr1, arr2, thresh, ret_error=True)
    if not np.all(passed):
        failed_xs = np.where(np.logical_not(passed))
        failed_error = error.take(failed_xs)
        failed_arr_test = arr1.take(failed_xs)
        failed_arr_target = arr2.take(failed_xs)

        msg_list = [
            'FAILED ASSERT ALMOST EQUAL',
            '  * failed_xs = %r' % (failed_xs,),
            '  * failed_error = %r' % (failed_error,),
            '  * failed_arr_test   = %r' % (failed_arr_test,),
            '  * failed_arr_target = %r' % (failed_arr_target,),
        ]
        msg = '\n'.join(msg_list)
        raise AssertionError(msg)
    return error


def assert_lessthan(arr_test, arr_max, msg=''):
    r"""
    Args:
        arr_test (ndarray or list):
        arr_target (ndarray or list):
        thresh (scalar or ndarray or list):
    """
    if util_arg.NO_ASSERTS:
        return
    arr1 = np.array(arr_test)
    arr2 = np.array(arr_max)
    error = arr_max - arr_test
    passed = error >= 0
    if not np.all(passed):
        failed_xs = np.where(np.logical_not(passed))
        failed_error = error.take(failed_xs)
        failed_arr_test = arr1.take(failed_xs)
        failed_arr_target = arr2.take(failed_xs)

        msg_list = [
            'FAILED ASSERT LESSTHAN',
            msg,
            '  * failed_xs = %r' % (failed_xs,),
            '  * failed_error = %r' % (failed_error,),
            '  * failed_arr_test   = %r' % (failed_arr_test,),
            '  * failed_arr_target = %r' % (failed_arr_target,),
        ]
        msg = '\n'.join(msg_list)
        raise AssertionError(msg)
    return error


def assert_all_eq(item_list, eq_=operator.eq):
    if len(item_list) == 0:
        return True
    import six
    item_iter = iter(item_list)
    item0 = six.next(item_iter)
    for count, item in enumerate(item_iter, start=1):
        flag = eq_(item0, item)
        if not flag:
            print('Error:')
            print('count = %r' % (count,))
            print('item = %r' % (item,))
            print('item0 = %r' % (item0,))
            msg = 'The %d-th item was not equal to item 0' % (count,)
            raise AssertionError(msg)


def assert_eq(var1, var2, msg='', var1_name=None, var2_name=None,
              verbose=None):
    import utool as ut
    if verbose is None:
        verbose = not util_arg.QUIET
    failed = var1 != var2
    if var1_name is None:
        var1_name = ut.get_varname_from_stack(var1, N=1, default='var1')
    if var2_name is None:
        var2_name = ut.get_varname_from_stack(var2, N=1, default='var2')
    fmtdict = dict(
        msg=msg,
        var1_name=var1_name,
        var2_name=var2_name,
        var1_repr=repr(var1),
        var2_repr=repr(var2))
    if failed:
        msg_fmtstr = ut.codeblock('''
            +=====
            ERROR {var1_name} != {var2_name}
            msg = {msg}
            ---
            {var1_name} = {var1_repr}
            ---
            {var2_name} = {var2_repr}
            L_____
            ''')
        msg = msg_fmtstr.format(**fmtdict)
        raise AssertionError(msg)
    else:
        if verbose:
            print('ASSERT_EQ_PASSED: {var1_name} == {var2_name} == {var1_repr}'.format(**fmtdict))


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_assert
        python -m utool.util_assert --allexamples
        python -m utool.util_assert --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
