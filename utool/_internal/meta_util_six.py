# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import six


if six.PY2:
    import types
    __STR__ = unicode

    # TODO: use six.text_type
    def get_funcname(func):
        """
        Weird behavior for classes
        I dont know why this returns type / None
        import lasagne
        lasagne.layers.InputLayer
        lasagne.layers.InputLayer.__module__
        lasagne.layers.InputLayer.__class__.__name__ == 'type'
        lasagne.layers.InputLayer.__class__ is type
        wtf
        """
        try:
            return getattr(func, 'func_name')
        except AttributeError:
            if isinstance(func, functools.partial):
                return get_funcname(func.func)
            elif isinstance(func, six.class_types):
                return str(func).replace('<class \'', '').replace('\'>', '')
            else:
                raise
        #except Exception as ex:
        #    import utool as ut
        #    with ut.embed_on_exception_context:
        #        raise

    def set_funcname(func, newname):
        return setattr(func, 'func_name', str(newname))
    #
    def get_funcglobals(func):
        return getattr(func, 'func_globals')
    #
    def get_funcdoc(func):
        return getattr(func, 'func_doc')
    #
    def set_funcdoc(func, newdoc):
        return setattr(func, 'func_doc', newdoc)
    #
    def get_imfunc(func):
        return getattr(func, 'im_func')

    def get_funccode(func):
        return getattr(func, 'func_code')
    IntType  = types.IntType
    LongType = types.LongType
    BooleanType = types.BooleanType
    FloatType = types.FloatType
elif six.PY3:

    __STR__ = str
    IntType  = int
    LongType = int
    BooleanType = bool
    FloatType = float
    def get_funcname(func):
        try:
            return getattr(func, '__name__')
        except AttributeError:
            if isinstance(func, functools.partial):
                return get_funcname(func.func)
            else:
                raise
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

    def get_funccode(func):
        return getattr(func, '__code__')
else:
    raise AssertionError('python4 ?!!')


def ensure_unicode(str_):
    """
    TODO:
        rob gp "isinstance\\(.*\\\\bstr\\\\b\\)"
    """
    if isinstance(str_, __STR__):
        return str_
    else:
        try:
            return __STR__(str_)
        except UnicodeDecodeError:
            return str_.decode('utf-8')
    #if not isinstance(str_, __STR__) and is_byte_encoded_unicode(str_):
    #    return str_.decode('utf-8')
    #else:
    #    return __STR__(str_)
