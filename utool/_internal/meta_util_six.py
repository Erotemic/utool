from __future__ import absolute_import, division, print_function
import six

if six.PY2:
    import functools
    import types
    def get_funcname(func):
        try:
            return getattr(func, 'func_name')
        except AttributeError:
            if isinstance(func, functools.partial):
                return get_funcname(func.func)
        #except Exception as ex:
        #    import utool as ut
        #    with ut.embed_on_exception_context:
        #        raise

    def set_funcname(func, newname):
        return setattr(func, 'func_name', newname)
    #
    def get_funcglobals(func):
        return getattr(func, 'func_globals')
    #
    def get_funcdoc(func):
        return getattr(func, 'func_doc')
    def set_funcdoc(func, newdoc):
        return setattr(func, 'func_doc', newdoc)
    #
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
    #
    def get_funcglobals(func):
        return getattr(func, '__globals__')
    #
    def get_funcdoc(func):
        return getattr(func, '__doc__')
    def set_funcdoc(func, newdoc):
        return setattr(func, '__doc__', newdoc)
    #
    def get_imfunc(func):
        return getattr(func, '__func__')
else:
    raise AssertionError('python4 ?!!')
