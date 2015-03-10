from __future__ import absolute_import, division, print_function
from utool import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[func]')


# DEPRICATE WHOLE FILE

def uinput_1to1(func, input_):
    """ universal input (really just accept list or tuple as input to a list
    only function) """
    if isinstance(input_, (tuple, list)):
        output_ = list(map(func, input_))
    else:
        output_ = func(input_)
    return output_


def general_get(getter, index, **kwargs):
    """ Works with getter funcs or indexable read/write arrays """
    if hasattr(getter, '__getitem__'):
        val = getter[index]
    else:
        val = getter(index, **kwargs)
    return val


def general_set(setter, index, val, **kwargs):
    """ Works with setter funcs or indexable read/write arrays """
    if hasattr(setter, '__setitem__'):
        setter[index] = val
    else:
        setter(index, val, **kwargs)


def identity(input_):
    """ identity function """
    return input_
