# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import functools
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)


def compose_functions(*func_list):
    """
    Referenes:
        https://mathieularose.com/function-composition-in-python/
    """
    def apply_composition(f, g):
        def compose(x):
            return f(g(x))
        return compose
    composed_func = functools.reduce(apply_composition, func_list)
    return composed_func


def identity(input_):
    """ identity function """
    return input_

# DEPRICATE EVERYTHING BELOW HERE


#def uinput_1to1(func, input_):
#    """ universal input (really just accept list or tuple as input to a list
#    only function)

#    Move to guitool
#    """
#    if isinstance(input_, (tuple, list)):
#        output_ = list(map(func, input_))
#    else:
#        output_ = func(input_)
#    return output_


#def general_get(getter, index, **kwargs):
#    """ Works with getter funcs or indexable read/write arrays """
#    if hasattr(getter, '__getitem__'):
#        val = getter[index]
#    else:
#        val = getter(index, **kwargs)
#    return val


#def general_set(setter, index, val, **kwargs):
#    """ Works with setter funcs or indexable read/write arrays """
#    if hasattr(setter, '__setitem__'):
#        setter[index] = val
#    else:
#        setter(index, val, **kwargs)
