#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from six.moves import builtins
import utool
print, rrr, profile = utool.inject2(__name__)


@utool.indent_func
def func1():
    print('enter func1')
    print('exit  func1')


@utool.indent_func
def func2():
    print('enter func2')
    func1()
    print('exit  func2')


def remove_timestamp(string):
    import re
    return re.sub(r'\[\d\d:\d\d:\d\d\]', '', string)


@utool.indent_func
def test():
    print('enter test')
    log_fpath1 = utool.get_app_resource_dir('utool', 'test_logfile1.txt')
    log_fpath2 = utool.get_app_resource_dir('utool', 'test_logfile2.txt')

    utool.start_logging(log_fpath1, 'w')
    func1()
    func2()
    utool.stop_logging()

    print('\n\n')
    print('This line is NOT logged')
    print('\n\n')

    utool.start_logging(log_fpath2, 'w')
    print('This line is logged')
    utool.stop_logging()

    log1 = utool.read_from(log_fpath1, verbose=False)
    log2 = utool.read_from(log_fpath2, verbose=False)

    target1 = utool.unindent('''
    <__LOG_START__>
    logging to log_fpath=%r
    [test][func1]enter func1
    [test][func1]exit  func1
    [test][func2]enter func2
    [test][func2][func1]enter func1
    [test][func2][func1]exit  func1
    [test][func2]exit  func2
    <__LOG_STOP__>''' % log_fpath1).strip()

    target2 = utool.unindent('''
    <__LOG_START__>
    logging to log_fpath=%r
    [test]This line is logged
    <__LOG_STOP__>''' % log_fpath2).strip()

    output1 = remove_timestamp(log1).strip()
    output2 = remove_timestamp(log2).strip()

    try:
        assert target1 == output1, 'target1 failed'
        assert target2 == output2, 'target2 failed'
        builtins.print('TEST PASSED')
    except AssertionError:
        builtins.print('\n<!!! TEST FAILED !!!>')

        builtins.print('\ntarget1:')
        builtins.print(target1)
        builtins.print('\noutput1:')
        builtins.print(output1)

        builtins.print('\ntarget2:')
        builtins.print(target2)
        builtins.print('\noutput2:')
        builtins.print(output2)

        builtins.print('</!!! TEST FAILED !!!>\n')
        raise


if __name__ == '__main__':
    test()
