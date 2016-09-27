# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
try:
    import numpy as np
except ImportError as ex:
    pass
import decimal
from utool import util_type
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)


def order_of_magnitude_ceil(num):
    nDigits = np.ceil(np.log10(num))
    scalefactor = 10 ** (nDigits - 1)
    return np.ceil(num / scalefactor) * scalefactor


def float_to_decimal(f):
    """
    Convert a floating point number to a Decimal with no loss of information

    References:
        http://docs.python.org/library/decimal.html#decimal-faq
    """
    n, d = f.as_integer_ratio()
    numerator, denominator = decimal.Decimal(n), decimal.Decimal(d)
    ctx = decimal.Context(prec=60)
    result = ctx.divide(numerator, denominator)
    while ctx.flags[decimal.Inexact]:
        ctx.flags[decimal.Inexact] = False
        ctx.prec *= 2
        result = ctx.divide(numerator, denominator)
    return result


def sigfig_str(number, sigfig):
    """
    References:
        http://stackoverflow.com/questions/2663612/nicely-repr-float-in-python
    """
    assert(sigfig > 0)
    try:
        d = decimal.Decimal(number)
    except TypeError:
        d = float_to_decimal(float(number))
    sign, digits, exponent = d.as_tuple()
    if len(digits) < sigfig:
        digits = list(digits)
        digits.extend([0] * (sigfig - len(digits)))
    shift = d.adjusted()
    result = int(''.join(map(str, digits[:sigfig])))
    # Round the result
    if len(digits) > sigfig and digits[sigfig] >= 5:
        result += 1
    result = list(str(result))
    # Rounding can change the length of result
    # If so, adjust shift
    shift += len(result) - sigfig
    # reset len of result to sigfig
    result = result[:sigfig]
    if shift >= sigfig - 1:
        # Tack more zeros on the end
        result += ['0'] * (shift - sigfig + 1)
    elif 0 <= shift:
        # Place the decimal point in between digits
        result.insert(shift + 1, '.')
    else:
        # Tack zeros on the front
        assert(shift < 0)
        result = ['0.'] + ['0'] * (-shift - 1) + result
    if sign:
        result.insert(0, '-')
    return ''.join(result)


def num2_sigfig(num):
    return int(np.ceil(np.log10(num)))


def num_fmt(num, max_digits=None):
    r"""
    Weird function. Not very well written. Very special case-y

    Args:
        num (int or float):
        max_digits (int):

    Returns:
        str:

    CommandLine:
        python -m utool.util_num --test-num_fmt

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_num import *  # NOQA
        >>> # build test data
        >>> num_list = [0, 0.0, 1.2, 1003232, 41431232., .0000000343, -.443243]
        >>> max_digits = None
        >>> # execute function
        >>> result = [num_fmt(num, max_digits) for num in num_list]
        >>> # verify results
        >>> print(result)
        ['0', '0.0', '1.2', '1,003,232', '41431232.0', '0.0', '-0.443']
    """
    if num is None:
        return 'None'
    def num_in_mag(num, mag):
        return mag > num and num > (-1 * mag)
    if max_digits is None:
        # TODO: generalize
        if num_in_mag(num, 1):
            if num_in_mag(num, .1):
                max_digits = 4
            else:
                max_digits = 3
        else:
            max_digits = 1
    if util_type.is_float(num):
        num_str = ('%.' + str(max_digits) + 'f') % num
        # Handle trailing and leading zeros
        num_str = num_str.rstrip('0').lstrip('0')
        if num_str.startswith('.'):
            num_str = '0' + num_str
        if num_str.endswith('.'):
            num_str = num_str + '0'
        return num_str
    elif util_type.is_int(num):
        return int_comma_str(num)
    else:
        return '%r'


def int_comma_str(num):
    int_str = ''
    reversed_digits = decimal.Decimal(num).as_tuple()[1][::-1]
    for i, digit in enumerate(reversed_digits):
        if (i) % 3 == 0 and i != 0:
            int_str += ','
        int_str += str(digit)
    return int_str[::-1]


def get_sys_maxint():
    return sys.maxint


def get_sys_minint():
    return sys.maxint + 1


def get_sys_maxfloat():
    return sys.float_info.max
