import functools
import codecs
import types

__STR__ = str
IntType  = int
LongType = int
BooleanType = bool
FloatType = float


def get_funcname(func):
    try:
        return getattr(func, '__name__')
    except AttributeError as original:
        if isinstance(func, functools.partial):
            return get_funcname(func.func)
        if isinstance(func, types.BuiltinFunctionType):
            # for cv2.imread
            #return str(cv2.imread).replace('>', '').replace('<built-in function', '')
            return str(func).replace('<built-in function', '<')
        else:
            try:
                return str(getattr(func, '__class__')).strip("<class '").strip("'>").split('.')[-1]
            except Exception:
                raise original


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


def get_funccode(func):
    return getattr(func, '__code__')


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
            if str_.startswith(codecs.BOM_UTF8):
                # Can safely remove the utf8 marker
                # http://stackoverflow.com/questions/12561063/python-extract-data-from-file
                str_ = str_[len(codecs.BOM_UTF8):]
            return str_.decode('utf-8')
    #if not isinstance(str_, __STR__) and is_byte_encoded_unicode(str_):
    #    return str_.decode('utf-8')
    #else:
    #    return __STR__(str_)
