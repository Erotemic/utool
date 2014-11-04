from __future__ import absolute_import, division, print_function
import six

if six.PY2:
    import types
    def get_funcname(func):
        return getattr(func, 'func_name')
    def set_funcname(func, newname):
        return setattr(func, 'func_name', newname)
    def get_funcglobals(func):
        return getattr(func, 'func_globals')
    def get_funcdoc(func):
        return getattr(func, 'func_doc')
    def get_imfunc(func):
        return getattr(func, 'im_func')
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
    def set_funcname(func, newname):
        return setattr(func, '__name__', newname)
    def get_funcglobals(func):
        return getattr(func, '__globals__')
    def get_funcdoc(func):
        return getattr(func, '__doc__')
    def get_imfunc(func):
        return getattr(func, '__func__')
else:
    raise AssertionError('python4 ?!!')
