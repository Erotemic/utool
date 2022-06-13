# -*- coding: utf-8 -*-
r"""
in vim nongreedy .* is .\{-}
in python nongreedy .* is .*?
"""
from __future__ import absolute_import, division, print_function
import re
import six
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)


def convert_text_to_varname(text):
    r"""
    Args:
        text (str): text that might not be a valid variablename

    Returns:
        str: varname

    CommandLine:
        python -m utool.util_regex --test-convert_text_to_varname

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_regex import *  # NOQA
        >>> text = '0) View Application-Files Directory. '
        >>> varname = convert_text_to_varname(text)
        >>> result = ('varname = %s' % (str(varname),))
        >>> print(result)
        _0_View_ApplicationFiles_Directory_
    """
    varname_pattern = '[^a-zA-Z0-9_]'
    varname = text
    # Convert spaces to underscores
    varname = varname.replace(' ', '_')
    # Remove all other non-varname chars
    varname = re.sub(varname_pattern, '', varname)
    # Make sure there are not leading numbers
    if re.match('^[0-9]', varname):
        varname = '_' + varname
    assert re.match(REGEX_VARNAME, varname), 'invalid varname=%r' % (varname,)
    return varname


def regex_or(list_):
    return '(' + '|'.join(list_) + ')'
    #return '|'.join(list_)


def regex_word(w):
    return r'\b%s\b' % (w,)


def extend_regex(regexpr):
    r"""
    Extends the syntax of regular expressions by replacing convineince syntax
      with re friendly syntax. Nameely things that I use in vim like \<\>
    """
    regex_map = {
        #r'\<': r'\b(?=\w)',
        #r'\>': r'\b(?!\w)',
        r'\<': r'\b' + positive_lookahead(r'\w'),
        r'\>': r'\b' + negative_lookahead(r'\w'),
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


def extend_regex3(regex_list, reflags=0):
    extended_regex_list = list(map(extend_regex, regex_list))
    if len(extended_regex_list) == 1:
        IGNORE_CASE_PREFIX = '\\c'
        if extended_regex_list[0].startswith(IGNORE_CASE_PREFIX):
            # hack for vim-like ignore case
            extended_regex_list[0] = extended_regex_list[0][len(IGNORE_CASE_PREFIX):]
            # TODO: reflags_list
            reflags = re.IGNORECASE | reflags
    return extended_regex_list, reflags


def extend_regex2(regexpr, reflags=0):
    """
    also preprocesses flags
    """
    regexpr = extend_regex(regexpr)
    IGNORE_CASE_PREF = '\\c'
    if regexpr.startswith(IGNORE_CASE_PREF):
        # hack for vim-like ignore case
        regexpr = regexpr[len(IGNORE_CASE_PREF):]
        reflags = reflags | re.IGNORECASE
    return regexpr, reflags


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


def named_field(key, regex, vim=False):
    """
    Creates a named regex group that can be referend via a backref.
    If key is None the backref is referenced by number.

    References:
        https://docs.python.org/2/library/re.html#regular-expression-syntax
    """
    if key is None:
        #return regex
        return r'(%s)' % (regex,)
    if vim:
        return r'\(%s\)' % (regex)
    else:
        return r'(?P<%s>%s)' % (key, regex)


def positive_lookahead(regex, vim=False):
    if vim:
        return r'\(' + regex + r'\)\@='
    else:
        return '(?=' + regex + ')'


def positive_lookbehind(regex, vim=False):
    if vim:
        return r'\(' + regex + r'\)\@<='
    else:
        return '(?<=' + regex + ')'


# lookahead_pos
# lookahead_neg

def negative_lookahead(regex, vim=False):
    if vim:
        return r'\(' + regex + r'\)\@!'
    else:
        return '(?!' + regex + ')'


def negative_lookbehind(regex, vim=False):
    r"""
    Args:
        regex (?):

    Returns:
        ?:

    CommandLine:
        python -m utool.util_regex --exec-negative_lookbehind

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_regex import *  # NOQA
        >>> regex = re.escape('\'')
        >>> pattern = negative_lookbehind(regex) + 'foo'
        >>> match1 = re.search(pattern, '\'foo\'')
        >>> match2 = re.search(pattern, '"foo"')
        >>> match3 = re.search(pattern, '\' foo\'')
        >>> match4 = re.search(pattern, '" foo"')
    """
    if vim:
        return r'\(' + regex + r'\)\@<!'
    else:
        return '(?<!' + regex + ')'


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
    r"""
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
        ...     ('name',  r'G\d+'),  # species and 2 numbers
        ...     ('under', r'_'),     # 2 more numbers
        ...     ('id',    r'\d+'),   # 2 more numbers
        ...     ( None,   r'\.'),
        ...     ('ext',   r'\w+'),
        ... ]
        >>> regex = named_field_regex(keypat_tups)
        >>> result = (regex)
        >>> print(result)
        (?P<name>G\d+)(?P<under>_)(?P<id>\d+)(\.)(?P<ext>\w+)
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
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> import inspect
        >>> func_code = inspect.getsource(ut.modify_quoted_strs)
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
        >>> # DISABLE_DOCTEST
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
    r"""

    modify_quoted_strs

    doesnt work with escaped quotes or multilines
    single quotes only. no nesting.

    Args:
        text (?):
        modify_func (None):

    Example:
        >>> # DISABLE_DOCTEST
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
        >>> # DISABLE_DOCTEST
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


def nongreedy_kleene_star(vim=False):
    return r'\{-}' if vim else '*?'


RE_FLAGS = re.MULTILINE | re.DOTALL
RE_KWARGS = {'flags': RE_FLAGS}


REGEX_VARNAME = '[A-Za-z_][A-Za-z0-9_]*'
REGEX_WHITESPACE =  r'\s*'
REGEX_INT = regex_word(r'\d\d*')
REGEX_FLOAT = regex_word(r'\d\d*\.\d\d*')

REGEX_NONGREEDY = '*?'

# FIXME; Include escaped strings
REGEX_STR = r"'[^']*'"
REGEX_ESCSTR = r"'(.)*'"
#REGEX_ESCSTR = r"'(?:\\.|[^'])*'"

REGEX_LATEX_COMMENT = negative_lookbehind(re.escape('\\')) + re.escape('%') + '.*$'
REGEX_C_COMMENT = re.escape('/*') + '.*' + re.escape('*/')

REGEX_RVAL = regex_or([REGEX_VARNAME, REGEX_INT, REGEX_FLOAT, REGEX_STR, REGEX_ESCSTR])


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
