from __future__ import absolute_import, division, print_function
import sys
import six
import functools
import types
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    # TODO remove numpy
    HAS_NUMPY = False
    pass
from utool import util_inject
from utool._internal.meta_util_six import IntType, LongType, FloatType, BooleanType
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[type]')


def type_str(type_):
    return str(type_).replace('<type \'', '').replace('\'>', '')


# Very odd that I have to put in dtypes in two different ways.
if HAS_NUMPY:
    NUMPY_SCALAR_NAMES = sorted(list(set(
        [str_.replace('numpy.', '')
         for str_ in (type_str(type_) for type_ in np.ScalarType)
         if str_.startswith('numpy.')
         ])))

    VALID_INT_TYPES = (IntType, LongType,
                       np.typeDict['int64'],
                       np.typeDict['int32'],
                       np.typeDict['uint8'],
                       np.dtype('int32'),
                       np.dtype('uint8'),
                       np.dtype('int64'),)

    VALID_FLOAT_TYPES = (FloatType,
                         np.typeDict['float64'],
                         np.typeDict['float32'],
                         np.typeDict['float16'],
                         np.dtype('float64'),
                         np.dtype('float32'),
                         np.dtype('float16'),)

    VALID_BOOL_TYPES = (BooleanType, np.bool_)
    NP_NDARRAY = np.ndarray
else:
    VALID_INT_TYPES = (IntType, LongType,)
    VALID_FLOAT_TYPES = (FloatType,)
    VALID_BOOL_TYPES = (BooleanType,)
    NP_NDARRAY = None


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
    if type_ in VALID_BOOL_TYPES and is_str(var):
        return bool_from_str(var)
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
        >>> # build test data
        >>> var_list = ['?', 1, '1', '1.0', '1.2', 'True', None, 'None']
        >>> # execute function
        >>> castvar_list = [smart_cast2(var) for var in var_list]
        >>> # verify results
        >>> result = str(castvar_list)
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

if sys.platform == 'win32':
    # Well this is a weird system specific error
    # https://github.com/numpy/numpy/issues/3667
    def get_type(var):
        """Gets types accounting for numpy"""
        return var.dtype if isinstance(var, NP_NDARRAY) else type(var)
else:
    def get_type(var):
        """Gets types accounting for numpy"""
        return var.dtype.type if isinstance(var, NP_NDARRAY) else type(var)


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
    return isinstance(var, (list, tuple, NP_NDARRAY))


def is_tuple(var):
    return isinstance(var, tuple)


def is_func_or_method(var):
    return isinstance(var, (types.MethodType, types.FunctionType))


def is_func_or_method_or_partial(var):
    return isinstance(var, (types.MethodType, types.FunctionType,
                            functools.partial))


def is_funclike(var):
    return hasattr(var, '__call__')


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
