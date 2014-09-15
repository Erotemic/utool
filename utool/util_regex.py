from __future__ import absolute_import, division, print_function
import re
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
