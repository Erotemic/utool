"""
Change to util_parse?
"""
from __future__ import absolute_import, division, print_function
import re
import six
import os
from os.path import split, relpath
from utool import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[str]')


RE_FLAGS = re.MULTILINE | re.DOTALL
RE_KWARGS = {'flags': RE_FLAGS}


REGEX_VARNAME = '[A-Za-z_][A-Za-z0-9_]*'


def extend_regex(regexpr):
    regex_map = {
        r'\<': r'\b(?=\w)',
        r'\>': r'\b(?!\w)',
        ('UNSAFE', r'\x08'): r'\b',
    }
    for key, repl in six.iteritems(regex_map):
        if isinstance(key, tuple):
            search = key[1]
        else:
            search = key
        if regexpr.find(search) != -1:
            if isinstance(key, tuple):
                print('WARNING! Unsafe regex with: %r' % (key,))
            regexpr = regexpr.replace(search, repl)
    return regexpr


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


def whole_word(regex):
    return r'\b%s\b' % regex


def backref_field(key):
    return r'\g<%s>' % (key)


bref_field = backref_field


def regex_replace(regex, repl, text):
    r"""
    thin wrapper around re.sub
    regex_replace

    MULTILINE and DOTALL are on by default in all util_regex functions

    Args:
        regex (str): pattern to find
        repl (str): replace pattern with this
        text (str): text to modify

    Returns:
        str: modified text

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_regex import *  # NOQA
        >>> regex = r'\(.*\):'
        >>> repl = '(*args)'
        >>> text = '''def foo(param1,
        ...                   param2,
        ...                   param3):'''
        >>> result = regex_replace(regex, repl, text)
        >>> print(result)
        def foo(*args)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_regex import *  # NOQA
        >>> import utool as ut
        >>> regex = ut.named_field_regex([('keyword', 'def'), ' ', ('funcname', '.*'), '\(.*\):'])
        >>> repl = ut.named_field_repl([('funcname',), ('keyword',)])
        >>> text = '''def foo(param1,
        ...                   param2,
        ...                   param3):'''
        >>> result = regex_replace(regex, repl, text)
        >>> print(result)
        foodef
    """
    return re.sub(regex, repl, text, **RE_KWARGS)


def named_field_regex(keypat_tups):
    """
    named_field_regex

    Args:
        keypat_tups (list): tuples of (name, pattern) or a string for an unnamed
        pattern

    Returns:
        str: regex

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_regex import *  # NOQA
        >>> keypat_tups = [
        ...    ('name',  r'G\d+'),  # species and 2 numbers
        ...    ('under', r'_'),     # 2 more numbers
        ...    ('id',    r'\d+'),   # 2 more numbers
        ...    ( None,   r'\.'),
        ...    ('ext',   r'\w+'),
        ... ]
        >>> regex = named_field_regex(keypat_tups)
        >>> result = (regex)
        >>> print(result)
        (?P<name>G\d+)(?P<under>_)(?P<id>\d+)\.(?P<ext>\w+)
    """
    # Allow for unnamed patterns
    keypat_tups_ = [(None, tup) if isinstance(tup, six.string_types) else tup
                    for tup in keypat_tups]
    named_fields = [named_field(key, pat) for key, pat in keypat_tups_]
    regex = ''.join(named_fields)
    return regex


def named_field_repl(field_list):
    r"""
    Args:
        field_list (list): list of either a tuples to denote a keyword, or a
            strings for relacement t3ext

    Returns:
        str: repl for regex

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_regex import *  # NOQA
        >>> field_list = [('key',), 'unspecial string']
        >>> repl = named_field_repl(field_list)
        >>> result = repl
        >>> print(result)
        \g<key>unspecial string
    """
    # Allow for unnamed patterns
    bref_field_list = [
        backref_field(key[0]) if isinstance(key, tuple) else key
        for key in field_list
    ]
    repl = ''.join(bref_field_list)
    return repl


def regex_get_match(regex, text, fromstart=False):
    if fromstart:
        match = re.match(regex, text, **RE_KWARGS)
    else:
        match = re.search(regex, text, **RE_KWARGS)
    return match


def regex_matches(regex, text, fromstart=True):
    hasmatch = regex_get_match(regex, text, fromstart=fromstart) is not None
    return hasmatch


def parse_docblock(func_code):
    """
    #TODO: Finish me

    References:
        http://pyparsing.wikispaces.com/share/view/1264103
        http://code.activestate.com/recipes/576704-python-code-minifier/

    Example:
        >>> import utool as ut
        >>> import inspect
        >>> func_code = inspect.getsource(ut.modify_quoted_strs)
        >>> func_code =
    """
    import pyparsing
    doublequote_comment = pyparsing.QuotedString(quoteChar='"""', escChar='\\', multiline=True)
    singlequote_comment = pyparsing.QuotedString(quoteChar='\'\'\'', escChar='\\', multiline=True)
    docblock_parser  = doublequote_comment | singlequote_comment
    docblock_parser.parseString(func_code)


def parse_python_syntax(text):
    """
    step1: split lines

    step2: parse enclosure pairity for each line to find unended lines

    for each unending line, is there a valid merge line?
    (a line that could snytatically finish this runnon statement?
    If no then error. Else try to join the two lines.

    step3: perform context_sensitive_edit

    Example:
        >>> from utool.util_regex import *   # NOQA
        >>> import utool
        >>> from os.path import normpath
        >>> text = utool.read_from(utool.util_regex.__file__)
    """
    import utool as ut
    def find_all(a_str, sub):
        start = 0
        while True:
            start = a_str.find(sub, start)
            if start == -1:
                return
            yield start
            start += len(sub)  # use start += 1 to find overlapping matches
    line_list_ = [line + '\n' for line in text.splitlines()]  # NOQA
    import re  # NOQA
    line_list = [line[0:line.find('#')] for line in line_list_]
    open_tokens  = ['\'\'\'', '"""', '\'', '"', '(', '[']   # ,  '#']
    close_tokens = ['\'\'\'', '"""', '\'', '"', ')', ']']   # , '\n']
    def find_token_pos(line, token):
        return list(find_all(line, token))
    open_tokenxs  = [[find_token_pos(line, token) for line in line_list] for token in open_tokens]
    close_tokenxs = [[find_token_pos(line, token) for line in line_list] for token in close_tokens]
    print(open_tokenxs)
    print(close_tokenxs)
    print(sum(ut.flatten(ut.flatten(open_tokenxs))))
    print(sum(ut.flatten(ut.flatten(close_tokenxs))))


def modify_quoted_strs(text, modify_func=None):
    """

    modify_quoted_strs

    doesnt work with escaped quotes or multilines
    single quotes only. no nesting.

    Args:
        text (?):
        modify_func (None):

    Example:
        >>> from utool.util_regex import *  # NOQA
        >>> text = "'just' 'a' sentance with 'strings' in it "
        >>> text2 = "'this' 'text' wont work 'because \'of \"the\"\' \"nesting\"'"
        >>> text3 = " ''' god \"help\" you ''' if you use 'triple quotes'  "
        >>> def modify_func(quoted_str):
        ...     return quoted_str.upper()
        >>> result = modify_quoted_strs(text, modify_func)
        >>> print(result)
        'JUST' 'A' sentance with 'STRINGS'
    """
    if modify_func is None:
        def idenfunc(quoted_str):
            return quoted_str
        modify_func = idenfunc
    # regex to find a string sequence without any escaped strings in it
    regex = r'(?P<quoted_str>\'[^\']*\')'
    tmp_text = text[:]
    new_text_list = []
    while True:
        result = regex_parse(regex, tmp_text, False)
        if result is not None:
            quoted_str = result['quoted_str']
            len_ = len(quoted_str)
            startpos = tmp_text.find(quoted_str)
            endpos = len_ + startpos
            new_text_list.append(tmp_text[0:startpos])
            new_text_list.append(modify_func(quoted_str))
            tmp_text = tmp_text[endpos:]
        else:
            new_text_list.append(tmp_text)
            break
    new_text = ''.join(new_text_list)
    return new_text


def padded_parse(pattern, text):
    # TODO: move to util_parse
    import parse
    padded_pattern = '{_prefix}' + pattern + '{_suffix}'
    padded_text = ' ' + text + ' '
    parse_result = parse.parse(padded_pattern, padded_text)
    return parse_result


def regex_parse(regex, text, fromstart=True):
    r"""
    regex_parse

    Args:
        regex (str):
        text (str):
        fromstart (bool):

    Returns:
        dict or None:

    Example:
        >>> from utool.util_regex import *  # NOQA
        >>> regex = r'(?P<string>\'[^\']*\')'
        >>> text = " 'just' 'a' sentance with 'strings' in it "
        >>> fromstart = False
        >>> result = regex_parse(regex, text, fromstart)['string']
        >>> print(result)

    """
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
    from utool import util_path
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
    for fpath in util_path.matching_fnames(dpath_list, include_patterns, recursive=recursive):
        sedfile(fpath, regexpr, repl, force)


def sedfile(fpath, regexpr, repl, force=False, verbose=True, veryverbose=False):
    """
    Executes sed on a specific file
    """
    # TODO: move to util_edit
    path, name = split(fpath)
    new_file_lines = []

    if veryverbose:
        print('[sedfile] fpath=%r' % fpath)
        print('[sedfile] regexpr=%r' % regexpr)
        print('[sedfile] repl=%r' % repl)
        print('[sedfile] force=%r' % force)
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
    elif verbose:
        print('Nothing changed')
    return None


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_regex; utool.doctest_funcs(utool.util_regex, allexamples=True)"
        python -c "import utool, utool.util_regex; utool.doctest_funcs(utool.util_regex)"
        python -m utool.util_regex
        python -m utool.util_regex --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
