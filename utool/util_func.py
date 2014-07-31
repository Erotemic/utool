from __future__ import absolute_import, division, print_function
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[func]')


def uinput_1to1(func, input_):
    """ universal input (really just accept list or tuple as input to a list
    only function) """
    if isinstance(input_, (tuple, list)):
        output_ = tuple(map(func, input_))
    else:
        output_ = func(input_)
    return output_


def general_get(getter, index):
    """ Works with getter funcs or indexable read/write arrays """
    if hasattr(getter, '__getitem__'):
        val = getter[index]
    else:
        val = getter(index)
    return val


def general_set(setter, index, val):
    """ Works with setter funcs or indexable read/write arrays """
    if hasattr(setter, '__setitem__'):
        setter[index] = val
    else:
        setter(index, val)
