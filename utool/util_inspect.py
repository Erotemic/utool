from __future__ import absolute_import, division, print_function
import inspect
import types
import six
import functools
from utool import util_inject
from utool._internal import meta_util_six
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[alg]')


def iter_module_doctestable(module, include_funcs=True, include_classes=True, include_methods=True):
    r"""
    iter_module_doctestable

    Args:
        module (?):
        include_funcs (bool):
        include_classes (bool):
        include_methods (bool):

    Example1:
        >>> from utool.util_inspect import *   # NOQA
        >>> import utool as ut
        >>> module = ut.util_tests
        >>> doctestable_list = list(iter_module_doctestable(module))
        >>> func_names = ut.get_list_column(doctestable_list, 0)
        >>> print('\n'.join(func_names))

    Example2:
        >>> from utool.util_inspect import *   # NOQA
        >>> import utool as ut
        >>> import ibeis
        >>> import ibeis.control.IBEISControl
        >>> module = ibeis.control.IBEISControl
        >>> doctestable_list = list(iter_module_doctestable(module))
        >>> func_names = ut.get_list_column(doctestable_list, 0)
        >>> print('\n'.join(func_names))
    """
    valid_func_types = (types.FunctionType, types.BuiltinFunctionType,
                        #types.MethodType, types.BuiltinMethodType,
                        )
    valid_class_types = (types.ClassType,  types.TypeType,)

    scalar_types = [dict, list, tuple, set, frozenset, bool, float, int] + list(six.string_types)
    scalar_types += list(six.string_types)
    other_types = [types.InstanceType, functools.partial, types.ModuleType]
    invalid_types = tuple(scalar_types + other_types)

    for key, val in six.iteritems(module.__dict__):
        if val is None:
            pass
        elif isinstance(val, valid_func_types):
            if include_funcs:
                yield key, val
        elif isinstance(val, valid_class_types):
            class_ = val
            if include_classes:
                yield key, val
            if include_methods:
                for subkey, subval in six.iteritems(class_.__dict__):
                    # Unbound methods are still typed as functions
                    if isinstance(subval, valid_func_types):
                        yield subkey, subval
        elif isinstance(val, invalid_types):
            pass
        else:
            #import utool as ut
            #if ut.VERBOSE:
            print('[util_inspect] WARNING:')
            print(' * Unknown if testable val=%r' % (val))
            print(' * Unknown if testable type(val)=%r' % type(val))


def list_class_funcnames(fname, blank_pats=['    #']):
    """
    list_class_funcnames

    Args:
        fname (str): filepath
        blank_pats (list): defaults to '    #'

    Returns:
        list: funcname_list

    Example:
        >>>
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


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_inspect; utool.doctest_funcs(utool.util_inspect, allexamples=True)"
        python -c "import utool, utool.util_inspect; utool.doctest_funcs(utool.util_inspect)"
        python utool/util_inspect.py --enableall
        python utool/util_inspect.py --enableall --test-iter-module-doctestable:1
        python utool/util_inspect.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
