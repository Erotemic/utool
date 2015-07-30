#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import sys


if __name__ == '__main__':
    import utool.tests.run_tests
    import multiprocessing
    multiprocessing.freeze_support()
    ut.change_term_title('RUN UTOOL TESTS')
    retcode = utool.tests.run_tests.run_tests()
    sys.exit(retcode)
