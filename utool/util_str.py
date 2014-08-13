from __future__ import absolute_import, division, print_function
import sys
import six
import textwrap
from six.moves import map, range
from os.path import split
import numpy as np
from .util_inject import inject
from .util_time import get_unix_timedelta
from ._internal.meta_util_six import get_funcname
print, print_, printDBG, rrr, profile = inject(__name__, '[str]')


np.tau = (2 * np.pi)  # tauday.com


def theta_str(theta, taustr=('tau' if '--myway' in sys.argv else '2pi')):
    """ Format theta so it is interpretable in base 10 """
    #coeff = (((tau - theta) % tau) / tau)
    coeff = (theta / np.tau)
    return ('%.2f * ' % coeff) + taustr


def bbox_str(bbox, pad=4):
    if bbox is None:
        return 'None'
    fmtstr = ', '.join(['%' + str(pad) + 'd'] * 4)
    return '(' + fmtstr % bbox + ')'


def verts_str(verts, pad=1):
    if verts is None:
        return 'None'
    fmtstr = ', '.join(['%' + str(pad) + 'd' + ', %' + str(pad) + 'd'] * 1)
    return ', '.join(['(' + fmtstr % vert + ')' for vert in verts])


def tupstr(tuple_):
    """ maps each item in tuple to a string and doesnt include parens """
    return ', '.join(list(map(str, tuple_)))

# --- Strings ----


def remove_chars(instr, illegals_chars):
    outstr = instr
    for ill_char in iter(illegals_chars):
        outstr = outstr.replace(ill_char, '')
    return outstr


def get_indentation(line_):
    """ returns the number of preceding spaces """
    return len(line_) - len(line_.lstrip())


def unindent(string):
    return textwrap.dedent(string)


def indent(string, indent='    '):
    return indent + string.replace('\n', '\n' + indent)


def indentjoin(strlist, indent='\n    ', suffix=''):
    return indent + indent.join([str(str_) + suffix for str_ in strlist])


def truncate_str(str_, maxlen=110):
    if len(str_) < maxlen:
        return str_
    else:
        truncmsg = ' ~~~TRUNCATED~~~ '
        maxlen_ = maxlen - len(truncmsg)
        lowerb  = int(maxlen_ * .8)
        upperb  = maxlen_ - lowerb
        return str_[:lowerb] + truncmsg + str_[-upperb:]


def pack_into(instr, textwidth=160, breakchars=' ', break_words=True, newline_prefix=''):
    textwidth_ = textwidth
    line_list = ['']
    word_list = instr.split(breakchars)
    for word in word_list:
        if len(line_list[-1]) + len(word) > textwidth_:
            line_list.append('')
            textwidth_ = textwidth - len(newline_prefix)
        while break_words and len(word) > textwidth_:
            line_list[-1] += word[:textwidth_]
            line_list.append('')
            word = word[textwidth_:]
        line_list[-1] += word + ' '
    return ('\n' + newline_prefix).join(line_list)


def newlined_list(list_, joinstr=', ', textwidth=160):
    """ Converts a list to a string but inserts a new line after textwidth chars """
    newlines = ['']
    for word in list_:
        if len(newlines[-1]) + len(word) > textwidth:
            newlines.append('')
        newlines[-1] += word + joinstr
    return '\n'.join(newlines)


def joins(string, list_, with_head=True, with_tail=False, tostrip='\n'):
    head = string if with_head else ''
    tail = string if with_tail else ''
    to_return = head + string.join(map(str, list_)) + tail
    to_return = to_return.strip(tostrip)
    return to_return


def indent_list(indent, list_):
    return map(lambda item: indent + str(item), list_)


def filesize_str(fpath):
    _, fname = split(fpath)
    mb_str = file_megabytes_str(fpath)
    return 'filesize(%r)=%s' % (fname, mb_str)


def byte_str2(nBytes):
    if nBytes < 2.0 ** 10:
        return byte_str(nBytes, 'KB')
    if nBytes < 2.0 ** 20:
        return byte_str(nBytes, 'KB')
    if nBytes < 2.0 ** 30:
        return byte_str(nBytes, 'MB')
    else:
        return byte_str(nBytes, 'GB')


def byte_str(nBytes, unit='bytes'):
    if unit.lower().startswith('b'):
        nUnit = nBytes
    elif unit.lower().startswith('k'):
        nUnit =  nBytes / (2.0 ** 10)
    elif unit.lower().startswith('m'):
        nUnit =  nBytes / (2.0 ** 20)
    elif unit.lower().startswith('g'):
        nUnit = nBytes / (2.0 ** 30)
    else:
        raise NotImplementedError('unknown nBytes=%r unit=%r' % (nBytes, unit))
    return '%.2f %s' % (nUnit, unit)


def file_megabytes_str(fpath):
    from . import util_path
    return ('%.2f MB' % util_path.file_megabytes(fpath))


# <Alias repr funcs>
GLOBAL_TYPE_ALIASES = []


def extend_global_aliases(type_aliases):
    global GLOBAL_TYPE_ALIASES
    GLOBAL_TYPE_ALIASES.extend(type_aliases)


def var_aliased_repr(var, type_aliases):
    global GLOBAL_TYPE_ALIASES
    # Replace aliased values
    for alias_type, alias_name in (type_aliases + GLOBAL_TYPE_ALIASES):
        if isinstance(var, alias_type):
            return alias_name + '<' + str(id(var)) + '>'
    return repr(var)


def list_aliased_repr(args, type_aliases=[]):
    return [var_aliased_repr(item, type_aliases)
            for item in args]


def dict_aliased_repr(dict_, type_aliases=[]):
    return ['%s : %s' % (key, var_aliased_repr(val, type_aliases))
            for (key, val) in six.iteritems(dict_)]

# </Alias repr funcs>


def func_str(func, args=[], kwargs={}, type_aliases=[]):
    """ string representation of function definition """
    repr_list = list_aliased_repr(args, type_aliases) + dict_aliased_repr(kwargs)
    argskwargs_str = newlined_list(repr_list, ', ', textwidth=80)
    func_str = '%s(%s)' % (get_funcname(func), argskwargs_str)
    return func_str


def dict_itemstr_list(dict_, strvals=False, sorted_=False):
    iteritems = six.iteritems
    fmtstr = '%r: %r,'
    if strvals:
        fmtstr = '%s: %s,'
    if sorted_:
        iteritems = lambda iter_: iter(sorted(iter_))
    itemstr_list = [fmtstr % (key, val) for (key, val) in iteritems(dict_)]
    return itemstr_list


def list_str(list_):
    return '[%s\n]' % indentjoin(list_, suffix=',')


def dict_str(dict_, strvals=False, sorted_=False):
    """ returns a human-readable and execable string representation of a
    dictionary """
    itemstr_list = dict_itemstr_list(dict_, strvals, sorted_)
    return '{%s\n}' % indentjoin(itemstr_list)


def horiz_string(*args):
    """
    prints a list of objects ensuring that the next item in the list
    is all the way to the right of any previous items.
    >>> # Pretty printing of matrices demo / test
    >>> import utool
    >>> import numpy as np
    >>> # Wouldn't it be nice if we could print this operation easilly?
    >>> B = np.array(((1, 2), (3, 4)))
    >>> C = np.array(((5, 6), (7, 8)))
    >>> A = B.dot(C)
    >>> # Eg 1:
    >>> str_list = ['A = ', str(B), ' * ', str(C)]
    >>> horizstr = (utool.horiz_string(*str_list))
    >>> print(horizstr)
    >>> # Eg 2:
    >>> print(utool.hz_str('A = ', A, ' = ', B, ' * ', C))
    """
    if len(args) == 1 and not isinstance(args[0], str):
        str_list = args[0]
    else:
        str_list = args
    all_lines = []
    hpos = 0
    for sx in range(len(str_list)):
        str_ = str(str_list[sx])
        lines = str_.split('\n')
        line_diff = len(lines) - len(all_lines)
        # Vertical padding
        if line_diff > 0:
            all_lines += [' ' * hpos] * line_diff
        # Add strings
        for lx, line in enumerate(lines):
            all_lines[lx] += line
            hpos = max(hpos, len(all_lines[lx]))
        # Horizontal padding
        for lx in range(len(all_lines)):
            hpos_diff = hpos - len(all_lines[lx])
            if hpos_diff > 0:
                all_lines[lx] += ' ' * hpos_diff
    ret = '\n'.join(all_lines)
    return ret

# Alias
hz_str = horiz_string


def listinfo_str(list_):
    info_list = enumerate([(type(item), item) for item in list_])
    info_str  = indentjoin(map(repr, info_list, '\n  '))
    return info_str


def str2(obj):
    if isinstance(obj, dict):
        return str(obj).replace(', ', '\n')[1:-1]
    if isinstance(obj, type):
        return str(obj).replace('<type \'', '').replace('\'>', '')
    else:
        return str(obj)


def get_unix_timedelta_str(unixtime_diff):
    timedelta = get_unix_timedelta(unixtime_diff)
    sign = '+' if unixtime_diff >= 0 else '-'
    timedelta_str = sign + str(timedelta)
    return timedelta_str


class NpPrintOpts(object):
    def __init__(self, **kwargs):
        self.orig_opts = np.get_printoptions()
        self.new_opts = kwargs
    def __enter__(self):
        np.set_printoptions(**self.new_opts)
    def __exit__(self, type_, value, trace):
        np.set_printoptions(**self.orig_opts)
        if trace is not None:
            print('[util_str] Error in context manager!: ' + str(value))
            return False  # return a falsey value on error


def full_numpy_repr(arr):
    with NpPrintOpts(threshold=np.uint64(-1)):
        arr_repr = repr(arr)
    return arr_repr


def str_between(str_, startstr, endstr):
    startpos = str_.find(startstr) + len(startstr)
    endpos = str_.find(endstr) - 1
    return str_[startpos:endpos]


def padded_str_range(start, end):
    """ Builds a list of (end - start) strings padded with zeros """
    nDigits = np.ceil(np.log10(end))
    fmt = '%0' + str(nDigits) + 'd'
    str_range = (fmt % num for num in range(start, end))
    return list(str_range)


def get_callable_name(func):
    """ Works on must functionlike objects including str, which has no func_name """
    try:
        return get_funcname(func)
    except AttributeError:
        builtin_function_name_dict = {
            len:    'len',
            zip:    'zip',
            range:  'range',
            map:    'map',
            type:   'type',
        }
        if func in builtin_function_name_dict:
            return builtin_function_name_dict[func]
        elif isinstance(func, type):
            return repr(func).replace('<type \'', '').replace('\'>', '')
        else:
            raise NotImplementedError(('cannot get func_name of func=%r'
                                        'type(func)=%r') % (func, type(func)))


def align(text, character='='):
    """ Left justifies text on the left side of character

    >>> character = '='
    >>> text = '''
            a = b
            one = two
            three = fish '''
    >>> print(align(text, '='))

    """
    line_list = text.splitlines()
    new_lines = align_lines(line_list, character)
    new_text = '\n'.join(new_lines)
    return new_text


def align_lines(line_list, character='='):
    """ Left justifies text on the left side of character"""

    tup_list = [line.split(character) for line in line_list]
    maxlen = 0
    for tup in tup_list:
        if len(tup) == 2:
            maxlen = max(maxlen, len(tup[0]))

    new_lines = []
    for tup in tup_list:
        if len(tup) == 2:
            lhs, rhs = tup
            newline = lhs.ljust(maxlen) + character + rhs
            new_lines.append(newline)
        else:
            new_lines.append(character.join(tup))
    return new_lines


def get_freespace_str(dir_='.'):
    from . import util_cplat
    return byte_str2(util_cplat.get_free_diskbytes(dir_))
