"""
python -c "import utool, doctest; print(doctest.testmod(utool.util_str))"
"""
from __future__ import absolute_import, division, print_function
import sys
import six
import textwrap
from six.moves import map, range
import math
from os.path import split
from .util_inject import inject
from .util_time import get_unix_timedelta
from ._internal.meta_util_six import get_funcname
print, print_, printDBG, rrr, profile = inject(__name__, '[str]')


TAU = (2 * math.pi)  # tauday.com


def theta_str(theta, taustr=('tau' if '--myway' in sys.argv else '2pi')):
    """ Format theta so it is interpretable in base 10 """
    #coeff = (((tau - theta) % tau) / tau)
    coeff = (theta / TAU)
    return ('%.2f * ' % coeff) + taustr


def bbox_str(bbox, pad=4):
    """ makes a string from an integer bounding box """
    if bbox is None:
        return 'None'
    fmtstr = ', '.join(['%' + str(pad) + 'd'] * 4)
    return '(' + fmtstr % bbox + ')'


def verts_str(verts, pad=1):
    """ makes a string from a list of integer verticies """
    if verts is None:
        return 'None'
    fmtstr = ', '.join(['%' + str(pad) + 'd' + ', %' + str(pad) + 'd'] * 1)
    return ', '.join(['(' + fmtstr % vert + ')' for vert in verts])


def tupstr(tuple_):
    """ maps each item in tuple to a string and doesnt include parens """
    return ', '.join(list(map(str, tuple_)))

# --- Strings ----


def remove_chars(instr, illegals_chars):
    """
    replaces all illegal characters in instr with ''
    """
    outstr = instr
    for ill_char in iter(illegals_chars):
        outstr = outstr.replace(ill_char, '')
    return outstr


def get_indentation(line_):
    """ returns the number of preceding spaces """
    return len(line_) - len(line_.lstrip())


def unindent(string):
    """
    Unindent a block of text

    Alias for textwrap.dedent
    """
    return textwrap.dedent(string)


def codeblock(block_str):
    return unindent(block_str).strip('\n')


def indent(string, indent='    '):
    """
    Indents a block of text
    """
    indent_ = indent
    return indent_ + string.replace('\n', '\n' + indent_)


def indentjoin(strlist, indent='\n    ', suffix=''):
    r"""
    Convineince

    similar to '\n    '.join(strlist) but indent is also prefixed
    """
    indent_ = indent
    return indent_ + indent_.join([str(str_) + suffix for str_ in strlist])


def truncate_str(str_, maxlen=110, truncmsg=' ~~~TRUNCATED~~~ '):
    """
    Removes the middle part of any string over maxlen characters.
    """
    if maxlen is None or maxlen == -1 or len(str_) < maxlen:
        return str_
    else:
        maxlen_ = maxlen - len(truncmsg)
        lowerb  = int(maxlen_ * .8)
        upperb  = maxlen_ - lowerb
        tup = (str_[:lowerb], truncmsg, str_[-upperb:])
        return ''.join(tup)


def pack_into(instr, textwidth=160, breakchars=' ', break_words=True, newline_prefix=''):
    """
    Inserts newlines into a string enforcing a maximum textwidth.
    Similar to vim's gq command in visual select mode.

    breakchars is a string containing valid characters to insert a newline
    before or after.

    break_words is True if words are allowed to be split over multiple lines.

    all inserted newlines are prefixed with newline_prefix
    """
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


def packstr(instr, textwidth=160, breakchars=' ', break_words=True,
            newline_prefix='', indentation='', nlprefix=None):
    """ alias for pack_into """
    if not isinstance(instr, six.string_types):
        instr = repr(instr)
    if nlprefix is not None:
        newline_prefix = nlprefix
    str_ = pack_into(instr, textwidth, breakchars, break_words, newline_prefix)
    if indentation != '':
        str_ = indent(str_, indentation)
    return str_


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


def seconds_str(num, prefix=None):
    r"""
    Returns:
        str

    Example:
        >>> import utool
        >>> utool.util_str.rrr()
        >>> num_list = sorted([4.2 / (10.0 ** exp_) for exp_ in range(-13, 13, 4)])
        >>> secstr_list = [utool.util_str.seconds_str(num, prefix=None) for num in num_list]
        >>> print(', '.join(secstr_list))
        0.042 ns, 0.42 us, 4.2 ms, 0.042 ks, 0.42 Ms, 4.2 Gs, 42.0 Ts

        #>>> print(',\n'.join(map(str, zip(secstr_list, num_list))))
    """
    exponent_list = [-12, -9, -6, -3, 0, 3, 6, 9, 12]
    small_prefix_list = ['p', 'n', 'u', 'm', '', 'k', 'M', 'G', 'T']
    #large_prefix_list = ['pico', 'nano', 'micro', 'mili', '', 'kilo', 'mega', 'giga', 'tera']
    #large_suffix = 'second'
    small_suffix = 's'
    suffix = small_suffix
    prefix_list = small_prefix_list
    base = 10.0
    secstr = order_of_magnitude_str(num, base, prefix_list, exponent_list,
                                    suffix, prefix=prefix)
    return secstr


def order_of_magnitude_str(num, base=10.0,
                           prefix_list=None,
                           exponent_list=None,
                           suffix='', prefix=None):
    """
    TODO: Rewrite byte_str to use this func
    Returns:
        str
    """
    abs_num = abs(num)
    # Find the right magnidue
    for prefix_, exponent in zip(prefix_list, exponent_list):
        # Let user request the prefix
        requested = False
        if prefix is not None:
            if prefix != prefix_:
                continue
            requested = True
        # Otherwise find the best prefix
        magnitude = base ** exponent
        # Be less than this threshold to use this unit
        thresh_mag = magnitude * base
        if requested or abs_num <= thresh_mag:
            break
    unit_str = _magnitude_str(abs_num, magnitude, prefix_, suffix)
    return unit_str


def _magnitude_str(abs_num, magnitude, prefix_, suffix):
    scaled_num = abs_num / magnitude
    unit = prefix_ + suffix
    unit_str = ('%.2f %s' % (scaled_num, unit))
    return unit_str


def byte_str2(nBytes):
    """
    Automatically chooses relevant unit (KB, MB, or GB) for displaying some
    number of bytes.

    Returns:
        str
    """
    nAbsBytes = abs(nBytes)
    if nAbsBytes < 2.0 ** 10:
        return byte_str(nBytes, 'KB')
    if nAbsBytes < 2.0 ** 20:
        return byte_str(nBytes, 'KB')
    if nAbsBytes < 2.0 ** 30:
        return byte_str(nBytes, 'MB')
    else:
        return byte_str(nBytes, 'GB')


def byte_str(nBytes, unit='bytes'):
    """
    representing the number of bytes with the chosen unit

    Returns:
        str
    """
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
    """
    State function for aliased_repr calls
    """
    global GLOBAL_TYPE_ALIASES
    GLOBAL_TYPE_ALIASES.extend(type_aliases)


def var_aliased_repr(var, type_aliases):
    """
    Replaces unweildy type strings with predefined more human-readable aliases

    Args:
        var: some object

    Returns:
        str: an "intelligently" chosen string representation of var
    """
    global GLOBAL_TYPE_ALIASES
    # Replace aliased values
    for alias_type, alias_name in (type_aliases + GLOBAL_TYPE_ALIASES):
        if isinstance(var, alias_type):
            return alias_name + '<' + str(id(var)) + '>'
    return repr(var)


def list_aliased_repr(list_, type_aliases=[]):
    """
    Replaces unweildy type strings with predefined more human-readable aliases

    Args:
        list_ (list): ``list`` to get repr

    Returns:
        str: string representation of ``list_``
    """
    return [var_aliased_repr(item, type_aliases)
            for item in list_]


def dict_aliased_repr(dict_, type_aliases=[]):
    """
    Replaces unweildy type strings with predefined more human-readable aliases

    Args:
        dict_ (dict): dictionary to get repr

    Returns:
        str: string representation of ``dict_``
    """
    return ['%s : %s' % (key, var_aliased_repr(val, type_aliases))
            for (key, val) in six.iteritems(dict_)]

# </Alias repr funcs>


def func_str(func, args=[], kwargs={}, type_aliases=[]):
    """
    string representation of function definition

    Returns:
        str: a representation of func with args, kwargs, and type_aliases
    """
    repr_list = list_aliased_repr(args, type_aliases) + dict_aliased_repr(kwargs)
    argskwargs_str = newlined_list(repr_list, ', ', textwidth=80)
    func_str = '%s(%s)' % (get_funcname(func), argskwargs_str)
    return func_str


def dict_itemstr_list(dict_, strvals=False, sorted_=False, newlines=True,
                      recursive=True, indent_=''):
    """
    Returns:
        list: a list of human-readable dictionary items
    Example:
        >>> from utool.util_str import dict_str, dict_itemstr_list
        >>> import utool
        >>> CONFIG_DICT = utool.get_default_repo_config()
        #>>> utool.rrrr()
        #>>> utool.util_str.rrr()
        >>> config_str = dict_str(CONFIG_DICT, strvals=True)
        >>> authstr = dict_str(CONFIG_DICT['AUTHORS'], strvals=True)
        >>> mestr = dict_str(CONFIG_DICT['AUTHORS']['joncrall'], strvals=True)
        >>> print(config_str)
        >>> print(authstr)
        >>> print(mestr)

    Dev:
        dict_ = CONFIG_DICT
        strvals = False
        sorted_ = False
        newlines = True
        recursive = True
        indent_ = ''
    """
    iteritems = six.iteritems
    if strvals:
        fmtstr = indent_ + '%s: %s,'
    else:
        fmtstr = indent_ + '%r: %r,'
    if sorted_:
        iteritems = lambda iter_: iter(sorted(iter_))
    if recursive:
        def valfunc(val):
            if isinstance(val, dict):
                # recursive call
                print('reccall')
                return dict_str(val, strvals, sorted_, newlines, recursive, indent_ + '    ')
            print('base')
            # base case
            return val
        itemstr_list = [fmtstr % (key, valfunc(val)) for (key, val) in iteritems(dict_)]
    else:
        itemstr_list = [fmtstr % (key, val) for (key, val) in iteritems(dict_)]
    return itemstr_list


def list_str(list_):
    return '[%s\n]' % indentjoin(list_, suffix=',')


def dict_str(dict_, strvals=False, sorted_=False, newlines=True, recursive=True, indent_=''):
    """
    Returns:
        str: a human-readable and execable string representation of a dictionary
    Example:
        >>> import utool
        >>> CONFIG_DICT = utool.get_default_repo_config()
        >>> config_str = utool.util_str.dict_str(CONFIG_DICT, strvals=True)
        >>> print(config_str)
        >>> authstr = utool.util_str.dict_str(CONFIG_DICT['AUTHORS'], strvals=True)
        >>> print(authstr)
        >>> mestr = utool.util_str.dict_str(CONFIG_DICT['AUTHORS']['joncrall'], strvals=True)
        >>> print(mestr)
    """
    print('dictstr')
    itemstr_list = dict_itemstr_list(dict_, strvals, sorted_, recursive, indent_)
    if newlines:
        return indent_ + '{%s\n}' % indentjoin(itemstr_list)
    else:
        return '{%s}' % ' '.join(itemstr_list)


def horiz_string(*args):
    """
    Horizontally prints objects

    Prints a list of objects ensuring that the next item in the list
    is all the way to the right of any previous items.

    Example:
        >>> # Pretty printing of matrices demo / test
        >>> import utool
        >>> import numpy as np
        >>> # Wouldn't it be nice if we could print this operation easily?
        >>> B = np.array(((1, 2), (3, 4)))
        >>> C = np.array(((5, 6), (7, 8)))
        >>> A = B.dot(C)
        >>> # Eg 1:
        >>> str_list = ['A = ', str(B), ' * ', str(C)]
        >>> horizstr = (utool.horiz_string(*str_list))
        >>> print(horizstr)
        A = [[1 2]  * [[5 6]
             [3 4]]    [7 8]]
        >>> # Eg 2:
        >>> print(utool.hz_str('A = ', A, ' = ', B, ' * ', C))
        A = [[19 22]  = [[1 2]  * [[5 6]
             [43 50]]    [3 4]]    [7 8]]
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


def str_between(str_, startstr, endstr):
    startpos = str_.find(startstr) + len(startstr)
    endpos = str_.find(endstr) - 1
    return str_[startpos:endpos]


def padded_str_range(start, end):
    """ Builds a list of (end - start) strings padded with zeros """
    import numpy as np
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
    r""" Left justifies text on the left side of character

    Example:
        >>> character = '='
        >>> text = 'a = b\none = two\nthree = fish\n'
        >>> print(align(text, '='))
        a     = b
        one   = two
        three = fish
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


# FIXME: HASHLEN is a global var in util_hash
def long_fname_format(fmt_str, fmt_dict, hashable_keys=[], max_len=64, hashlen=16, ABS_MAX_LEN=255):
    """

    Args:
        fmt_str (str): format of fname
        fmt_dict (str): dict to format fname with
        hashable_keys (list): list of dict keys you are willing to have hashed
        max_len (int): tries to fit fname into this length
        ABS_MAX_LEN (int): throws AssertionError if fname over this length

    Example:
        >>> import utool
        >>> fmt_str = 'qaid={qaid}_res_{cfgstr}_quuid={quuid}'
        >>> quuid_str = 'blahblahblahblahblahblah'
        >>> cfgstr = 'big_long_string__________________________________'
        >>> qaid = 5
        >>> fmt_dict = dict(cfgstr=cfgstr, qaid=qaid, quuid=quuid_str)
        >>> hashable_keys = ['cfgstr', 'quuid']
        >>> max_len = 64
        >>> hashlen = 8
        >>> fname0 = utool.long_fname_format(fmt_str, fmt_dict, max_len=None)
        >>> fname1 = utool.long_fname_format(fmt_str, fmt_dict, hashable_keys, max_len=64, hashlen=8)
        >>> fname2 = utool.long_fname_format(fmt_str, fmt_dict, hashable_keys, max_len=42, hashlen=8)
        >>> print(fname0)
        qaid=5_res_big_long_string___________________________________quuid=blahblahblahblahblahblah
        >>> print(fname1)
        qaid=5_res_kjrok785_quuid=blahblahblahblahblahblah
        >>> print(fname2)
        qaid=5_res_du1&i&5l_quuid=euuaxoyi
    """
    from . import util_hash
    fname = fmt_str.format(**fmt_dict)
    if max_len is None:
        return fname
    if len(fname) > max_len:
        # Copy because we will overwrite fmt_dict values with hashed values
        fmt_dict_ = fmt_dict.copy()
        for key in hashable_keys:
            fmt_dict_[key] = util_hash.hashstr(fmt_dict_[key], hashlen=hashlen)
            fname = fmt_str.format(**fmt_dict_)
            if len(fname) <= max_len:
                break
        if len(fname) > max_len:
            diff = len(fname) - max_len
            msg = ('Warning: Too big by %d chars. Exausted all options to make fname fit into size. ' % diff)
            print(msg)
            print('len(fname) = %r' % len(fname))
            print('fname = %r' % fname)
            if ABS_MAX_LEN is not None and len(fname) > ABS_MAX_LEN:
                raise AssertionError(msg)
    return fname


#def parse_commas_wrt_groups(str_):
#    """
#    str_ = 'cdef np.ndarray[np.float64_t, cast=True] x, y, z'
#    """
#    nLParen = 0
#    nLBracket = 0
#    pass
