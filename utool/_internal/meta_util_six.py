from __future__ import absolute_import, division, print_function
import six

if six.PY2:
    import types
    def get_funcname(func):
        return getattr(func, 'func_name')
    IntType  = types.IntType
    LongType = types.LongType
    BooleanType = types.BooleanType
    FloatType = types.FloatType
elif six.PY3:
    IntType  = int
    LongType = int
    BooleanType = bool
    FloatType = float
    def get_funcname(func):
        return getattr(func, '__name__')
