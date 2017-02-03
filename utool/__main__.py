#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


def utool_main():
    ignore_prefix = []
    ignore_suffix = []
    import utool as ut
    # allows for --tf
    ut.main_function_tester('utool', ignore_prefix, ignore_suffix)

if __name__ == '__main__':
    """
    python -m utool --tf infer_function_info:0
    """
    print('Checking utool main')
    utool_main()
