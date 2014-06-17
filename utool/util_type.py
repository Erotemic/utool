from __future__ import absolute_import, division, print_function
import sys
# Science
import numpy as np
import functools
import types
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[type]')


# Very odd that I have to put in dtypes in two different ways.
VALID_INT_TYPES = (types.IntType,
                   types.LongType,
                   np.typeDict['int64'],
                   np.typeDict['int32'],
                   np.typeDict['uint8'],
                   np.dtype('int32'),
                   np.dtype('uint8'),
                   np.dtype('int64'),)

VALID_FLOAT_TYPES = (types.FloatType,
                     np.typeDict['float64'],
                     np.typeDict['float32'],
                     np.typeDict['float16'],
                     np.dtype('float64'),
                     np.dtype('float32'),
                     np.dtype('float16'),)

VALID_BOOL_TYPES = (types.BooleanType, np.bool_)


def is_valid_floattype(type_):
    """ Returns if a type_ is a valid float type_ (not variable) """
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


def bool_from_str(str_):
    lower = str_.lower()
    if lower == 'true':
        return True
    elif lower == 'false':
        return False
    else:
        raise TypeError('string does not represent boolean')


def assert_int(var, lbl='var'):
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
        'Gets types accounting for numpy'
        return var.dtype if isinstance(var, np.ndarray) else type(var)
else:
    def get_type(var):
        'Gets types accounting for numpy'
        return var.dtype.type if isinstance(var, np.ndarray) else type(var)


def is_type(var, valid_types):
    'Checks for types accounting for numpy'
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
    return is_type(var, VALID_FLOAT_TYPES)


def is_str(var):
    return isinstance(var, (str, unicode))
    #return is_type(var, VALID_STRING_TYPES)


def is_bool(var):
    return isinstance(var, VALID_BOOL_TYPES)


def is_dict(var):
    return isinstance(var, dict)


def is_list(var):
    return isinstance(var, list)


def type_str(type_):
    return str(type_).replace('<type \'', '').replace('\'>', '')


def is_func_or_method(var):
    return isinstance(var, (types.MethodType, types.FunctionType))


def is_func_or_method_or_partial(var):
    return isinstance(var, (types.MethodType, types.FunctionType,
                            functools.partial))


def is_funclike(var):
    return hasattr(var, '__call__')
