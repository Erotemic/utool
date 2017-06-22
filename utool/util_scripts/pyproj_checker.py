# -*- coding: utf-8 -*-
"""
Checks python files in a project for consistent header patterns

Rectify with ensure_python3_compatible
"""
from __future__ import absolute_import, division, print_function, unicode_literals


def find_encoding_insert_position(line_list):
    linenum = None
    for linenum, line in enumerate(line_list):
        # Ensure encoding line is after shebang
        if line.startswith('#!'):
            continue
        break
    if len(line_list) == 0 or linenum >= len(line_list) - 1:
        print(' ! could not find approprate position')
        linenum = None
    return linenum


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

    show_diff = ut.get_argflag('--diff')
    do_write = ut.get_argflag('--write')

    need_encoding_fpaths = []

    for pat in pattern_items:
        print('Checking for pattern: %r' % (pat,))
        for fpath in fpath_list:
            pattern = re.escape(pat)
            found_lines, found_lxs = ut.grepfile(fpath, pattern)
            # DID NOT FIND ENCODING LINE
            if len(found_lines) == 0:
                need_encoding_fpaths.append(fpath)

    print('The following fpaths need encoding lines: ' + ut.repr4(need_encoding_fpaths, strvals=True))

    if do_write or show_diff:
        for fpath in need_encoding_fpaths:
            print('\n-----------------\nFound file without encodeing line: ' + fpath)
            line_list = ut.read_lines_from(fpath)
            linenum = find_encoding_insert_position(line_list)
            if linenum is not None:
                #print(' * linenum = %r' % (linenum,))
                new_lines = line_list[:linenum] + [encoding_line + '\n'] + line_list[linenum:]
                new_text = ''.join(new_lines)
                if show_diff:
                    old_text = ''.join(line_list)
                    textdiff = ut.get_textdiff(old_text, new_text, num_context_lines=1)
                    print('Diff:')
                    ut.print_difftext(textdiff)
                if do_write:
                    ut.writeto(fpath, new_text)
                    pass
