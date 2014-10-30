from __future__ import absolute_import, division, print_function
import inspect
from os.path import split, splitext, dirname, basename


def get_caller_name(N=0):
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
    frame_level0 = inspect.currentframe()
    frame_cur = frame_level0
    for _ix in range(N + 1):
        frame_next = frame_cur.f_back
        if frame_next is None:
            raise AssertionError('Frame level %r is root' % _ix)
        frame_cur = frame_next
    parent_frame = frame_cur
    # </get_parent_frame>
    caller_name = parent_frame.f_code.co_name
    if caller_name == '<module>':
        co_filename = parent_frame.f_code.co_filename
        caller_name = splitext(split(co_filename)[1])[0]
    if caller_name == '__init__':
        caller_name = basename(dirname(co_filename)) + '.' + caller_name
    return caller_name
