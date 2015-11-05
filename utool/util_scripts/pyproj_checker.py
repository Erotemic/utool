# -*- coding: utf-8 -*-
"""
Checks python files in a project for consistent header patterns
"""
from __future__ import absolute_import, division, print_function, unicode_literals


if __name__ == '__main__':
    """
    python -m utool.util_scripts.pyproj_checker
    python ~/code/utool/utool/util_scripts/pyproj_checker.py
    """
    import utool as ut
    import re
    exclude_dirs = ['_broken', '_doc', 'build']
    fpath_list = ut.glob('.', '*.py', exclude_dirs=exclude_dirs, recursive=True)

    encoding_line = '# -*- coding: utf-8 -*-'

    pattern_items = [
        encoding_line,
        #'from __future__ import absolute_import, division, print_function',
        #'from __future__ import absolute_import, division, print_function, unicode_literals',
    ]

    for pat in pattern_items:
        print('Checking for pattern: %r' % (pat,))
        for fpath in fpath_list:
            pattern = re.escape(pat)
            found_lines, found_lxs = ut.grepfile(fpath, pattern)
            if len(found_lines) == 0:
                print('\n-----------------\nFound file without encodeing line: ' + fpath)
                line_list = ut.read_lines_from(fpath)
                for linenum, line in enumerate(line_list):
                    # Ensure encoding line is after shebang
                    if line.startswith('#!'):
                        continue
                    break
                if linenum >= len(line_list) - 1:
                    print(' ! could not find approprate position')
                else:
                    #print(' * linenum = %r' % (linenum,))
                    new_lines = line_list[:linenum] + [encoding_line + '\n'] + line_list[linenum:]
                    new_text = ''.join(new_lines)
                    if True:
                        old_text = ''.join(line_list)
                        textdiff = ut.get_textdiff(old_text, new_text, num_context_lines=1)
                        print('Diff:')
                        ut.print_difftext(textdiff)
                    dry = True
                    if not dry:
                        pass
