#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool
from utool import util_io

from utool.util_io import write_to
write_to2 = utool.write_to


def print_ids():
    print('<')
    print('utool.write_to:         %r' % id(utool.write_to))  # WHY GOOD?
    print('utool.util_io.write_to: %r' % id(utool.util_io.write_to))  # GOOD
    print('util_io.write_to:       %r' % id(util_io.write_to))  # GOOD
    print('write_to:               %r' % id(write_to))   # BAD
    print('write_to2:              %r' % id(write_to2))  # BAD
    print('>')


def print_docstr():
    print('<')
    print('utool.write_to:         %r' % (utool.write_to.__doc__))
    print('utool.util_io.write_to: %r' % (utool.util_io.write_to.__doc__))
    print('util_io.write_to:       %r' % (util_io.write_to.__doc__))
    print('write_to:               %r' % (write_to.__doc__))
    print('write_to2:              %r' % (write_to2.__doc__))
    print('>')


def reloading_test1():
    print('TEST1')
    print_ids()
    utool.rrrr()
    print_ids()


def reloading_test2():
    print('TEST2')
    print_ids()
    import imp
    # Causes utool.write_to to be bad
    # because utool doesnt do a rrr itself
    imp.reload(utool.util_io)
    print_ids()


def docstr_test1():
    print_ids()
    utool.rrr()
    print_ids()


def docstr_test2():
    print_ids()
    utool.rrr()
    print_ids()


if __name__ == '__main__':
    utool.print_object_size(utool)
    reloading_test1()
    #reloading_test2()
    reloading_test1()
    reloading_test1()
    #reloading_test2()
    utool.print_object_size(utool)
    pass
