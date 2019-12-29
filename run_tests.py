#!/usr/bin/env python
# -*- coding: utf-8 -*-
if __name__ == '__main__':
    import pytest
    import sys
    package_name = 'utool'
    pytest_args = [
        '--cov-config', '.coveragerc',
        '--cov-report', 'html',
        '--cov-report', 'term',
        '--xdoctest',
        '--cov=' + package_name,
    ]
    pytest_args = pytest_args + sys.argv[1:]
    pytest.main(pytest_args)


# OLD UTOOL TESTER (still works)
# if __name__ == '__main__':
#     import utool.tests.run_tests
#     import multiprocessing
#     multiprocessing.freeze_support()
#     ut.change_term_title('RUN UTOOL TESTS')
#     retcode = utool.tests.run_tests.run_tests()
#     sys.exit(retcode)
