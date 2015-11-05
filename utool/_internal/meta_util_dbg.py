# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import inspect
from os.path import split, splitext, dirname, basename


def get_stack_frame(N=0, strict=True):
    frame_level0 = inspect.currentframe()
    frame_cur = frame_level0
    for _ix in range(N + 1):
        frame_next = frame_cur.f_back
        if frame_next is None:
            if strict:
                raise AssertionError('Frame level %r is root' % _ix)
            else:
                break
        frame_cur = frame_next
    return frame_cur


def get_caller_lineno(N=0, strict=True):
    parent_frame = get_stack_frame(N=N + 1, strict=strict)
    lineno =  parent_frame.f_lineno
    return lineno


def get_caller_name(N=0, strict=True):
    """ Standalone version of get_caller_name """
    if isinstance(N, (list, tuple)):
        name_list = []
        for N_ in N:
            try:
                name_list.append(get_caller_name(N_))
            except AssertionError:
                name_list.append('X')
        return '[' + ']['.join(name_list) + ']'
    # <get_parent_frame>
    parent_frame = get_stack_frame(N=N + 2, strict=strict)
    # </get_parent_frame>
    caller_name = parent_frame.f_code.co_name
    if caller_name == '<module>':
        co_filename = parent_frame.f_code.co_filename
        caller_name = splitext(split(co_filename)[1])[0]
    if caller_name == '__init__':
        co_filename = parent_frame.f_code.co_filename
        caller_name = basename(dirname(co_filename)) + '.' + caller_name
    return caller_name
