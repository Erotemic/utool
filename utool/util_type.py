# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import six
import functools
import re
import types
from utool import util_inject
from utool._internal.meta_util_six import IntType, LongType, FloatType, BooleanType
from utool._internal import meta_util_six
#import warnings
print, rrr, profile = util_inject.inject2(__name__)


__STR__ = meta_util_six.__STR__


if not six.PY2:
    def type_str(type_):
        return str(type_).replace('<class \'', '').replace('\'>', '')
    VALID_STRING_TYPES = (str,)
    unicode = None
    basestring = None
else:
    def type_str(type_):
        str_ = str(type_)
        str_ = str_.replace('<type \'', '').replace('\'>', '')
        str_ = str_.replace('<class \'', '').replace('\'>', '')
        return str_
    VALID_STRING_TYPES = (str, unicode, basestring)


# Very odd that I have to put in dtypes in two different ways.
try:
    import numpy as np
    HAVE_NUMPY = True
    NUMPY_SCALAR_NAMES = sorted(list(set(
        (str_.replace('numpy.', '')
         for str_ in (type_str(type_) for type_ in np.ScalarType)
         if str_.startswith('numpy.')
         ))))

    try:
        sctypeDict = np.sctypeDict
    except Exception:
        sctypeDict = np.typeDict

    VALID_INT_TYPES = (IntType, LongType,
                       sctypeDict['int64'],
                       sctypeDict['int32'],
                       sctypeDict['uint8'],
                       np.dtype('int32'),
                       np.dtype('uint8'),
                       np.dtype('int64'),)

    VALID_FLOAT_TYPES = (FloatType,
                         sctypeDict['float64'],
                         sctypeDict['float32'],
                         sctypeDict['float16'],
                         np.dtype('float64'),
                         np.dtype('float32'),
                         np.dtype('float16'),)

    VALID_BOOL_TYPES = (BooleanType, np.bool_)
    LISTLIKE_TYPES = (tuple, list, np.ndarray)
    NUMPY_TYPE_TUPLE = (
        tuple([np.ndarray] + list(set(sctypeDict.values()))))

    try:
        import pandas as pd  # NOQA
        HAVE_PANDAS = True
    except ImportError:
        HAVE_PANDAS = False

except (ImportError, AttributeError):
    # TODO remove numpy
    HAVE_NUMPY = False
    HAVE_PANDAS = False
    VALID_INT_TYPES = (IntType, LongType,)
    VALID_FLOAT_TYPES = (FloatType,)
    VALID_BOOL_TYPES = (BooleanType,)
    LISTLIKE_TYPES = (tuple, list)
    NUMPY_TYPE_TUPLE = tuple()


COMPARABLE_TYPES = {
    # Ensure that type_list can be used by isinstance
    type_: tuple([other for other in type_list if isinstance(other, type)])
    for type_list in [
        VALID_INT_TYPES,
        VALID_FLOAT_TYPES,
        VALID_BOOL_TYPES,
        VALID_STRING_TYPES,
    ]
    for type_ in type_list
}
for int_type_ in VALID_INT_TYPES:
    COMPARABLE_TYPES[int_type_] = COMPARABLE_TYPES[int_type_] + tuple([other for other in VALID_BOOL_TYPES if isinstance(other, type)])
for float_type_ in VALID_FLOAT_TYPES:
    COMPARABLE_TYPES[float_type_] = (COMPARABLE_TYPES[float_type_] +
                                     tuple([other for other in VALID_BOOL_TYPES if isinstance(other, type)]) +
                                     tuple([other for other in VALID_INT_TYPES if isinstance(other, type)]))


PRIMATIVE_TYPES = (
    tuple(six.string_types) + (bytes, list, dict, set, frozenset, int, float,
                               bool, type(None))
)


def is_comparable_type(var, type_):
    """
    Check to see if `var` is an instance of known compatible types for `type_`

    Args:
        var (?):
        type_ (?):

    Returns:
        bool:

    CommandLine:
        python -m utool.util_type is_comparable_type --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_type import *  # NOQA
        >>> import utool as ut
        >>> flags = []
        >>> flags += [is_comparable_type(0, float)]
        >>> flags += [is_comparable_type(0, np.float32)]
        >>> flags += [is_comparable_type(0, np.int32)]
        >>> flags += [is_comparable_type(0, int)]
        >>> flags += [is_comparable_type(0.0, int)]
        >>> result = ut.repr2(flags)
        >>> print(result)
        [True, True, True, True, False]
    """
    other_types = COMPARABLE_TYPES.get(type_, type_)
    return isinstance(var, other_types)


def is_valid_floattype(type_):
    """
    Args:
        type_ (``type``): type to check

    Returns:
        bool: if a ``type_`` is a valid float ``type_`` (not variable)
    """
    return type_ in VALID_FLOAT_TYPES
    #try:
    #    #flags = [type_ == float_type for float_type in VALID_FLOAT_TYPES]
    #    #return any(flags)
    #    tried = []
    #    for float_type in VALID_FLOAT_TYPES:
    #        tried.append(float_type)
    #        if type_ == float_type:
    #            return True
    #    return False
    #except Exception:
    #    print('tried=%r' % (tried,))
    #    print('type_=%r' % (type_,))
    #    print('float_type=%r' % (float_type,))


def try_cast(var, type_, default=None):
    if type_ is None:
        return var
    try:
        return smart_cast(var, type_)
    except Exception:
        return default


def smart_cast(var, type_):
    """
    casts var to type, and tries to be clever when var is a string

    Args:
        var (object): variable to cast
        type_ (type or str): type to attempt to cast to

    Returns:
        object:

    CommandLine:
        python -m utool.util_type --exec-smart_cast

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_type import *  # NOQA
        >>> var = '1'
        >>> type_ = 'fuzzy_subset'
        >>> cast_var = smart_cast(var, type_)
        >>> result = repr(cast_var)
        >>> print(result)
        [1]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_type import *  # NOQA
        >>> import utool as ut
        >>> cast_var = smart_cast('1', None)
        >>> result = ut.repr2(cast_var)
        >>> print(result)
        '1'

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_type import *  # NOQA
        >>> cast_var = smart_cast('(1,3)', 'eval')
        >>> result = repr(cast_var)
        >>> print(result)
        (1, 3)

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_type import *  # NOQA
        >>> cast_var = smart_cast('(1,3)', eval)
        >>> result = repr(cast_var)
        >>> print(result)
        (1, 3)

    Example4:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_type import *  # NOQA
        >>> cast_var = smart_cast('1::3', slice)
        >>> result = repr(cast_var)
        >>> print(result)
        slice(1, None, 3)
    """
    #if isinstance(type_, tuple):
    #    for trytype in type_:
    #        try:
    #            return trytype(var)
    #        except Exception:
    #            pass
    #    raise TypeError('Cant figure out type=%r' % (type_,))
    if type_ is None or var is None:
        return var
    #if not isinstance(type_, six.string_types):
    try:
        if issubclass(type_, type(None)):
            return var
    except TypeError:
        pass
    if is_str(var):
        if type_ in VALID_BOOL_TYPES:
            return bool_from_str(var)
        elif type_ is slice:
            args = [None if len(arg) == 0 else int(arg) for arg in var.split(':')]
            return slice(*args)
        elif type_ is list:
            # need more intelligent parsing here
            subvar_list = var.split(',')
            return [smart_cast2(subvar) for subvar in subvar_list]
        elif isinstance(type_, six.string_types):
            if type_ == 'fuzzy_subset':
                return fuzzy_subset(var)
            if type_ == 'eval':
                return eval(var, {}, {})
            #elif type_ == 'fuzzy_int':
            #    return fuzzy_subset(var)
            else:
                raise NotImplementedError('Uknown smart type_=%r' % (type_,))
    return type_(var)


def smart_cast2(var):
    r"""
    if the variable is a string tries to cast it to a reasonable value.
    maybe can just use eval. FIXME: funcname

    Args:
        var (unknown):

    Returns:
        unknown: some var

    CommandLine:
        python -m utool.util_type --test-smart_cast2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_type import *  # NOQA
        >>> import utool as ut
        >>> # build test data
        >>> var_list = ['?', 1, '1', '1.0', '1.2', 'True', None, 'None']
        >>> # execute function
        >>> castvar_list = [smart_cast2(var) for var in var_list]
        >>> # verify results
        >>> result = ut.repr4(castvar_list, nl=False)
        >>> print(result)
        ['?', 1, 1, 1.0, 1.2, True, None, None]
    """
    if var is None:
        return None
    if isinstance(var, six.string_types):
        castvar = None
        lower = var.lower()
        if lower == 'true':
            return True
        elif lower == 'false':
            return False
        elif lower == 'none':
            return None
        if var.startswith('[') and var.endswith(']'):
            #import re
            #subvar_list = re.split(r',\s*' + ut.negative_lookahead(r'[^\[\]]*\]'), var[1:-1])
            return smart_cast(var[1:-1], list)
        elif var.startswith('(') and var.endswith(')'):
            #import re
            #subvar_list = re.split(r',\s*' + ut.negative_lookahead(r'[^\[\]]*\]'), var[1:-1])
            return tuple(smart_cast(var[1:-1], list))
        type_list = [int, float]
        for type_ in type_list:
            castvar = try_cast(var, type_)
            if castvar is not None:
                break
        if castvar is None:
            castvar = var
    else:
        castvar = var
    return castvar


def bool_from_str(str_):
    lower = str_.lower()
    if lower == 'true':
        return True
    elif lower == 'false':
        return False
    else:
        raise TypeError('string does not represent boolean')


def fuzzy_subset(str_):
    """
    converts a string into an argument to list_take
    """
    if str_ is None:
        return str_
    if ':' in str_:
        return smart_cast(str_, slice)
    if str_.startswith('['):
        return smart_cast(str_[1:-1], list)
    else:
        return smart_cast(str_, list)


def fuzzy_int(str_):
    """
    lets some special strings be interpreted as ints
    """
    try:
        ret = int(str_)
        return ret
    except Exception:
        # Parse comma separated values as ints
        if re.match(r'\d*,\d*,?\d*', str_):
            return tuple(map(int, str_.split(',')))
        # Parse range values as ints
        if re.match(r'\d*:\d*:?\d*', str_):
            return tuple(range(*map(int, str_.split(':'))))
        raise


def assert_int(var, lbl='var'):
    from utool.util_arg import NO_ASSERTS
    if NO_ASSERTS:
        return
    try:
        assert is_int(var), 'type(%s)=%r is not int' % (lbl, get_type(var))
    except AssertionError:
        print('[tools] %s = %r' % (lbl, var))
        print('[tools] VALID_INT_TYPES: %r' % VALID_INT_TYPES)
        raise

# if HAVE_NUMPY:
_WIN32 = (sys.platform == 'win32')


def get_type(var):
    """
    Gets types accounting for numpy

    Ignore:
        import utool as ut
        import pandas as pd
        var = np.array(['a', 'b', 'c'])
        ut.get_type(var)
        var = pd.Index(['a', 'b', 'c'])
        ut.get_type(var)
    """
    if HAVE_NUMPY and isinstance(var, np.ndarray):
        if _WIN32:
            # This is a weird system specific error
            # https://github.com/numpy/numpy/issues/3667
            type_ = var.dtype
        else:
            type_ = var.dtype.type
    elif HAVE_PANDAS and isinstance(var, pd.Index):
        if _WIN32:
            type_ = var.dtype
        else:
            type_ = var.dtype.type
    else:
        type_ = type(var)
    return type_


def is_type(var, valid_types):
    """ Checks for types accounting for numpy """
    #printDBG('checking type var=%r' % (var,))
    #var_type = type(var)
    #printDBG('type is type(var)=%r' % (var_type,))
    #printDBG('must be in valid_types=%r' % (valid_types,))
    #ret = var_type in valid_types
    #printDBG('result is %r ' % ret)
    return get_type(var) in valid_types


def is_int(var):
    """

    Returns:
        bool: True if var is an integer.

    Note:
        Yuck, isinstance(True, int) returns True. This function does not have
        that flaw.

    References:
        http://www.peterbe.com/plog/bool-is-int

    CommandLine:
        python -m utool.util_type --test-is_int

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_type import *  # NOQA
        >>> var1 = 1
        >>> var2 = np.array([1, 2, 3])
        >>> var3 = True
        >>> var4 = np.array([True, True, False])
        >>> result = [is_int(var) for var in [var1, var2, var3, var4]]
        >>> print(result)
        [True, True, False, False]
    """
    #if _newbehavior:
    #    if is_bool(var):
    #        msg = 'Comparing bool to int. Make sure legacy code does is updated accordingly.'
    #        print('Warning: ' + msg)
    #        warnings.warn(msg)
    #        return False
    #    else:
    #        return is_type(var, VALID_INT_TYPES)
    #else:
    return is_type(var, VALID_INT_TYPES)


def is_float(var):
    r"""
    Args:
        var (ndarray or scalar):

    Returns:
        var:

    CommandLine:
        python -m utool.util_type --test-is_float

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_type import *  # NOQA
        >>> # build test data
        >>> var = np.array([1.0, 2.0, 3.0])
        >>> # execute function
        >>> assert is_float(var) is True, 'var is a float'
        >>> # verify results
        >>> print(result)
    """
    return is_type(var, VALID_FLOAT_TYPES)


def is_str(var):
    return isinstance(var, six.string_types)
    #return is_type(var, VALID_STRING_TYPES)


def is_bool(var):
    return isinstance(var, VALID_BOOL_TYPES)


def is_dict(var):
    return isinstance(var, dict)


def is_list(var):
    return isinstance(var, list)


def is_listlike(var):
    return isinstance(var, LISTLIKE_TYPES)


def is_tuple(var):
    return isinstance(var, tuple)


def is_method(var):
    return isinstance(var, (types.MethodType,))


def is_func_or_method(var):
    return isinstance(var, (types.MethodType, types.FunctionType))


def is_func_or_method_or_partial(var):
    return isinstance(var, (types.MethodType, types.FunctionType,
                            functools.partial))


def is_funclike(var):
    return hasattr(var, '__call__')


#def get_list_type(list_):
#    if isinstance(list_, np.ndarray):
#        return list_.dtype
#        pass


def get_homogenous_list_type(list_):
    """
    Returns the best matching python type even if it is an ndarray assumes all
    items in the list are of the same type. does not check this
    """
    # TODO Expand and make work correctly
    if HAVE_NUMPY and isinstance(list_, np.ndarray):
        item = list_
    elif isinstance(list_, list) and len(list_) > 0:
        item = list_[0]
    else:
        item = None
    if item is not None:
        if is_float(item):
            type_ = float
        elif is_int(item):
            type_ = int
        elif is_bool(item):
            type_ = bool
        elif is_str(item):
            type_ = str
        else:
            type_ = get_type(item)
    else:
        type_ = None
    return type_


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_type
        python -m utool.util_type --allexamples
        python -m utool.util_type --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
