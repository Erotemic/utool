from __future__ import absolute_import, division, print_function
import re
import os
from os.path import split, relpath
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[str]')


RE_FLAGS = re.MULTILINE | re.DOTALL
RE_KWARGS = {'flags': RE_FLAGS}


REGEX_VARNAME = '[A-Za-z_][A-Za-z0-9_]*'


def get_match_text(match):
    if match is not None:
        start, stop = match.start(), match.end()
        return match.string[start:stop]
    else:
        return None


def regex_search(regex, text):
    if text is None:
        return None
    match = regex_get_match(regex, text, fromstart=False)
    match_text = get_match_text(match)
    return match_text


def regex_split(regex, text):
    return re.split(regex, text, **RE_KWARGS)


def named_field(key, regex):
    if key is None:
        return regex
    return r'(?P<%s>%s)' % (key, regex)


def repl_field(key):
    return r'\g<%s>' % (key)


def regex_replace(regex, repl, text):
    return re.sub(regex, repl, text, **RE_KWARGS)


def named_field_regex(keypat_tups):
    named_fields = [named_field(key, pat) for key, pat in keypat_tups]
    regex = ''.join(named_fields)
    return regex


def regex_get_match(regex, text, fromstart=False):
    if fromstart:
        match = re.match(regex, text, **RE_KWARGS)
    else:
        match = re.search(regex, text, **RE_KWARGS)
    return match


def regex_parse(regex, text, fromstart=True):
    match = regex_get_match(regex, text, fromstart=fromstart)
    if match is not None:
        parse_dict = match.groupdict()
        return parse_dict
    return None


def regex_replace_lines(lines, regexpat, replpat):
    newlines = [regex_replace(regexpat, replpat, line) for line in lines]
    return newlines


def sed(regexpr, repl, force=False, recursive=False, dpath_list=None):
    """
    Python implementation of sed. NOT FINISHED

    searches and replaces text in files

    Args:
        regexpr (str): regx patterns to find
        repl (str): text to replace
        force (bool):
        recursive (bool):
        dpath_list (list): directories to search (defaults to cwd)
    """
    from . import util_path
    #_grep(r, [repl], dpath_list=dpath_list, recursive=recursive)
    include_patterns = ['*.py', '*.cxx', '*.cpp', '*.hxx', '*.hpp', '*.c', '*.h']
    if dpath_list is None:
        dpath_list = [os.getcwd()]
    print('sed-ing %r' % (dpath_list,))
    print(' * regular expression : %r' % (regexpr,))
    print(' * replacement        : %r' % (repl,))
    print(' * recursive: %r' % (recursive,))
    print(' * force: %r' % (force,))
    regexpr = util_path.extend_regex(regexpr)
    if '\x08' in regexpr:
        print('Remember \\x08 != \\b')
        print('subsituting for you for you')
        regexpr = regexpr.replace('\x08', '\\b')
        print(' * regular expression : %r' % (regexpr,))

    # Walk through each directory recursively
    for fpath in util_path._matching_fnames(dpath_list, include_patterns, recursive=recursive):
        sedfile(fpath, regexpr, repl, force)


def sedfile(fpath, regexpr, repl, force=False, verbose=True):
    """
    Executes sed on a specific file
    """
    # TODO: move to util_edit
    path, name = split(fpath)
    new_file_lines = []
    with open(fpath, 'r') as file:
        file_lines = file.readlines()
        # Search each line for the desired regexpr
        new_file_lines = [re.sub(regexpr, repl, line) for line in file_lines]

    changed_lines = [(newline, line)
                     for newline, line in zip(new_file_lines, file_lines)
                     if  newline != line]
    nChanged = len(changed_lines)
    if nChanged > 0:
        rel_fpath = relpath(fpath, os.getcwd())
        print(' * %s changed %d lines in %r ' %
              (['(dry-run)', '(real-run)'][force], nChanged, rel_fpath))
        print(' * --------------------')
        new_file = ''.join(new_file_lines)
        #print(new_file.replace('\n','\n))
        if verbose:
            changed_new, changed_old = zip(*changed_lines)
            prefixold = ' * old (%d, %r):  \n | ' % (nChanged, name)
            prefixnew = ' * new (%d, %r):  \n | ' % (nChanged, name)
            print(prefixold + (' | '.join(changed_old)).strip('\n'))
            print(' * ____________________')
            print(prefixnew + (' | '.join(changed_new)).strip('\n'))
            print(' * --------------------')
            print(' * =====================================================')
        # Write back to file
        if force:
            print(' ! WRITING CHANGES')
            with open(fpath, 'w') as file:
                file.write(new_file)
        return changed_lines
    return None
