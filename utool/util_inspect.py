from __future__ import absolute_import, division, print_function
import inspect
import types
import six
from . import util_inject
from ._internal import meta_util_six
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[alg]')


def iter_module_funcs(module):
    r"""
    Example:
        >>> import utool
        >>> func_names = utool.get_list_column(list(iter_module_funcs(utool.util_tests)), 0)
        >>> print('\n'.join(func_names))
    """
    valid_func_types = (types.FunctionType, types.MethodType,
                        types.BuiltinFunctionType, types.BuiltinMethodType,
                        types.ClassType)
    for key, val in six.iteritems(module.__dict__):
        if isinstance(val, valid_func_types):
            yield key, val
        elif isinstance(val, types.InstanceType):
            pass
        elif isinstance(val, types.ModuleType):
            pass
        elif isinstance(val, six.string_types):
            pass
        else:
            import utool as ut
            if ut.VERBOSE:
                print('Unknown if testable %r' % type(val))


def list_class_funcnames(fname, blank_pats=['    #']):
    """
    list_class_funcnames

    Args:
        fname (str): filepath
        blank_pats (list): defaults to '    #'

    Returns:
        list: funcname_list

    Example:
        >>> from utool.util_inspect import *  # NOQA
        >>> fname = 'util_class.py'
        >>> blank_pats = ['    #']
        >>> funcname_list = list_class_funcnames(fname, blank_pats)
        >>> print(funcname_list)
    """
    with open(fname, 'r') as file_:
        lines = file_.readlines()
    funcname_list = []

    #full_line_ = ''
    for lx, line in enumerate(lines):
        #full_line_ += line
        if any([line.startswith(pat) for pat in blank_pats]):
            funcname_list.append('')
        if line.startswith('    def '):
            def_x    = line.find('def')
            rparen_x = line.find('(')
            funcname = line[(def_x + 3):rparen_x]
            #print(funcname)
            funcname_list.append(funcname)
    return funcname_list


# grep is in util_path. Thats pretty inspecty


def get_funcname(func):
    return meta_util_six.get_funcname(func)


def set_funcname(func, newname):
    return meta_util_six.set_funcname(func, newname)


def get_imfunc(func):
    return meta_util_six.get_imfunc(func)


def get_funcglobals(func):
    return meta_util_six.get_funcglobals(func)


def get_funcdoc(func):
    return meta_util_six.get_funcdoc(func)


def set_funcdoc(func, newdoc):
    return meta_util_six.set_funcdoc(func, newdoc)


def get_func_argspec(func):
    argspec = inspect.getargspec(func)
    return argspec
