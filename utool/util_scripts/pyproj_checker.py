# -*- coding: utf-8 -*-
"""
Checks python files in a project for consistent header patterns
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import re


if __name__ == '__main__':
    """
    python -m utool.util_scripts.pyproj_checker
    python ~/code/utool/utool/util_scripts/pyproj_checker.py
    """
    exclude_dirs = ['_broken', '_doc', 'build']
    fpath_list = ut.glob('.', '*.py', exclude_dirs=exclude_dirs, recursive=True)

    pattern_items = [
        '# -*- coding: utf-8 -*-',
        #'from __future__ import absolute_import, division, print_function',
        #'from __future__ import absolute_import, division, print_function, unicode_literals',
    ]

    for pat in pattern_items:
        print('Checking for pattern: %r' % (pat,))
        for fpath in fpath_list:
            pattern = re.escape(pat)
            found_lines, found_lxs = ut.grepfile(fpath, pattern)
            if len(found_lines) == 0:
                print(fpath)
