# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function


def permit_gitrepo(config_fpath, writeback=False):
    """
    Changes https:// in .git/config files to git@ and makes
    appropriate changes to colons and slashses
    """
    # Define search replace patterns
    username_regex = utool.named_field('username', utool.REGEX_VARNAME)
    username_repl = utool.backref_field('username')
    regexpat = r'https://github.com/' + username_regex + '/'
    replpat = r'git@github.com:' + username_repl + '/'
    # Read and replace
    lines = utool.read_from(config_fpath, aslines=True)
    newlines = utool.regex_replace_lines(lines, regexpat, replpat)
    # Writeback or print
    if not WRITEBACK:
        print(''.join(newlines))
    else:
        utool.write_to(config_fpath, newlines, aslines=True)


if __name__ == '__main__':
    import utool
    WRITEBACK = utool.get_argflag('-i')
    if not WRITEBACK:
        print('specify -i to write changes')
    config_fpath = '.git/config'
    permit_gitrepo(config_fpath, writeback=WRITEBACK)
