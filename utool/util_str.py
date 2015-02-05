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
from utool.util_inject import inject
from utool.util_time import get_unix_timedelta
from utool._internal.meta_util_six import get_funcname
print, print_, printDBG, rrr, profile = inject(__name__, '[str]')


TAU = (2 * math.pi)  # References: tauday.com

TRIPLE_DOUBLE_QUOTE = r'"' * 3
TRIPLE_SINGLE_QUOTE = r"'" * 3
SINGLE_QUOTE = r"'"
DOUBLE_QUOTE = r'"'
NEWLINE = '\n'

TAUFMTSTR = '{coeff:,.1f}{taustr}'
if '--myway' not in sys.argv:
    TAUSTR = '*2pi'
else:
    TAUSTR = 'tau'


def theta_str(theta, taustr=TAUSTR, fmtstr='{coeff:,.1f}{taustr}'):
    """
    Format theta so it is interpretable in base 10

    theta_str

    CommandLine:
        python utool\util_str.py --noface --nosrc --test-theta_str:0
        python utool\util_str.py --noface --nosrc --test-theta_str:1

    Args:
        theta (float) angle in radians
        taustr (str): default 2pi

    Returns:
        str : theta_str - the angle in tau units

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> theta = 3.1415
        >>> result = theta_str(theta)
        >>> print(result)
        0.5*2pi

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> theta = 6.9932
        >>> taustr = 'tau'
        >>> result = theta_str(theta, taustr)
        >>> print(result)
        1.1tau
    """
    coeff = theta / TAU
    theta_str = fmtstr.format(coeff=coeff, taustr=taustr)
    return theta_str


def bbox_str(bbox, pad=4, sep=', '):
    """ makes a string from an integer bounding box """
    if bbox is None:
        return 'None'
    fmtstr = sep.join(['%' + str(pad) + 'd'] * 4)
    return '(' + fmtstr % tuple(bbox) + ')'


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


def remove_chars(str_, char_list):
    """
    removes all chars in char_list from str_

    Args:
        str_ (str):
        char_list (list):

    Returns:
        str: outstr

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> str_ = '1, 2, 3, 4'
        >>> char_list = [',']
        >>> result = remove_chars(str_, char_list)
        >>> print(result)
        1234
    """
    outstr = str_[:]
    for char in char_list:
        outstr = outstr.replace(char, '')
    return outstr


def get_indentation(line_):
    """
    returns the number of preceding spaces
    """
    return len(line_) - len(line_.lstrip())


def unindent(string):
    """
    Unindent a block of text

    Alias for textwrap.dedent
    """
    return textwrap.dedent(string)


def codeblock(block_str):
    """
    Convinience function for defining code strings. Esspecially useful for
    templated code.
    """
    return unindent(block_str).strip('\n')


def textblock(multiline_text):
    r"""
    Args:
        block_str (str):

    CommandLine:
        python -m utool.util_str --test-textblock

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> # build test data
        >>> multiline_text = ''' a big string
            that should be layed out flat
            yet still provide nice python
            code that doesnt go too far over
            80 characters.

            Two newlines should be respected though
            '''
        >>> # execute function
        >>> new_text = textblock(multiline_text)
        >>> # verify results
        >>> result = new_text
        >>> print(result)
    """
    import re
    def flatten_textlines(text):
        return re.sub(' *\n *', ' ', text, flags=re.MULTILINE).strip(' ')
    new_text = '\n\n'.join(list(map(flatten_textlines, multiline_text.split('\n\n'))))
    return new_text


def indent(string, indent='    '):
    """
    Indents a block of text
    """
    indent_ = indent
    return indent_ + string.replace('\n', '\n' + indent_)


def indentjoin(strlist, indent='\n    ', suffix=''):
    r"""
    Convineince indentjoin

    similar to '\n    '.join(strlist) but indent is also prefixed

    Args:
        strlist (?):
        indent  (str):
        suffix  (str):

    Returns:
        str: joined list
    """
    indent_ = indent
    strlist = list(strlist)
    if len(strlist) == 0:
        return ''
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


def __OLD_pack_into(instr, textwidth=160, breakchars=' ', break_words=True,
                    newline_prefix='', wordsep=' '):
    """
    BROKEN DO NOT USE
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
        line_list[-1] += word + wordsep
    return ('\n' + newline_prefix).join(line_list)


def pack_into(instr, textwidth=160, breakchars=' ', break_words=True,
              newline_prefix='', wordsep=' ', remove_newlines=True):
    r"""

    TODO: Look into textwrap.wrap

    Inserts newlines into a string enforcing a maximum textwidth.
    Similar to vim's gq command in visual select mode.

    breakchars is a string containing valid characters to insert a newline
    before or after.

    break_words is True if words are allowed to be split over multiple lines.

    all inserted newlines are prefixed with newline_prefix

    #FIXME:

    Example:
        >>> instr = "set_image_uris(ibs<139684018194000>, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [u'66ec193a-1619-b3b6-216d-1784b4833b61.jpg', u'd8903434-942f-e0f5-d6c2-0dcbe3137bf7.jpg', u'b73b72f4-4acb-c445-e72c-05ce02719d3d.jpg', u'0cd05978-3d83-b2ee-2ac9-798dd571c3b3.jpg', u'0a9bc03d-a75e-8d14-0153-e2949502aba7.jpg', u'2deeff06-5546-c752-15dc-2bd0fdb1198a.jpg', u'a9b70278-a936-c1dd-8a3b-bc1e9a998bf0.png', u'42fdad98-369a-2cbc-67b1-983d6d6a3a60.jpg', u'c459d381-fd74-1d99-6215-e42e3f432ea9.jpg', u'33fd9813-3a2b-774b-3fcc-4360d1ae151b.jpg', u'97e8ea74-873f-2092-b372-f928a7be30fa.jpg', u'588bc218-83a5-d400-21aa-d499832632b0.jpg', u'163a890c-36f2-981e-3529-c552b6d668a3.jpg'], ) "
        >>> textwidth = 160
        >>> breakchars = ' '
        >>> break_words = True
        >>> newline_prefix = '    '
        >>> wordsep = ' '
        >>> packstr1 = pack_into(instr, textwidth, breakchars, break_words, newline_prefix, wordsep)
        >>> break_words = False
        >>> packstr2 = pack_into(instr, textwidth, breakchars, break_words, newline_prefix, wordsep)
        >>> print(packstr1)
        >>> print(packstr2)

    CommandLine:
        python -c "import utool" --dump-utool-init


    """
    #FIXME: messy code
    textwidth_ = textwidth
    # Accumulate a list of lines
    line_list = ['']
    # Split text into list of words
    word_list = instr.split(breakchars)
    if remove_newlines:
        word_list = [word.replace('\n', '') for word in word_list]
    for word in word_list:
        available = textwidth_ - len(line_list[-1])
        # Check to see if we need to make a new line
        while len(word) > available:
            if break_words:
                # If we are allowed to break words over multiple lines
                # Fill the rest of the available textwidth with part of the word
                line_list[-1] += word[:available]
                word = word[available:]
            # Append a new line to the list
            # Reset the avaiablable textwidth for new line
            line_list.append('')
            textwidth_ = textwidth - len(newline_prefix)
            available = textwidth_ - len(line_list[-1])
            if not break_words:
                break
        # Append the word and a separator to the current line.
        if len(line_list) > 1:
            # Weird if statement. Probably bug somewhere.
            textwidth_ = textwidth - len(newline_prefix)
        line_list[-1] += word + wordsep
    packed_str = ('\n' + newline_prefix).join(line_list)
    return packed_str


def packstr(instr, textwidth=160, breakchars=' ', break_words=True,
            newline_prefix='', indentation='', nlprefix=None, wordsep=' ',
            remove_newlines=True):
    """ alias for pack_into. has more up to date kwargs """
    if not isinstance(instr, six.string_types):
        instr = repr(instr)
    if nlprefix is not None:
        newline_prefix = nlprefix
    str_ = pack_into(instr, textwidth, breakchars, break_words, newline_prefix,
                     wordsep, remove_newlines)
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
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> utool.util_str.rrr()
        >>> num_list = sorted([4.2 / (10.0 ** exp_) for exp_ in range(-13, 13, 4)])
        >>> secstr_list = [utool.util_str.seconds_str(num, prefix=None) for num in num_list]
        >>> result = (', '.join(secstr_list))
        >>> print(result)
        0.04 ns, 0.42 us, 4.20 ms, 0.04 ks, 0.42 Ms, 4.20 Gs, 42.00 Ts

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
    from utool import util_path
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


def func_str(func, args=[], kwargs={}, type_aliases=[], packed=False,
             packkw=None):
    """
    string representation of function definition

    Returns:
        str: a representation of func with args, kwargs, and type_aliases
    """
    repr_list = list_aliased_repr(args, type_aliases) + dict_aliased_repr(kwargs)
    argskwargs_str = newlined_list(repr_list, ', ', textwidth=80)
    func_str = '%s(%s)' % (get_funcname(func), argskwargs_str)
    if packed:
        packkw_ = dict(textwidth=80, nlprefix='    ', break_words=False)
        if packkw is not None:
            packkw_.update(packkw_)
        func_str = packstr(func_str, **packkw_)
    return func_str


def dict_itemstr_list(dict_, strvals=False, sorted_=False, newlines=True, recursive=True, indent_=''):
    """
    Returns:
        list: a list of human-readable dictionary items

    Example:
        >>> from utool.util_str import dict_str, dict_itemstr_list
        >>> import utool
        >>> REPO_CONFIG = utool.get_default_repo_config()
        >>> GLOBAL_CONFIG = utool.get_default_global_config()
        >>> utool.rrrr()
        >>> utool.util_str.rrr()
        >>> repo_cfgstr   = dict_str(REPO_CONFIG, strvals=True)
        >>> global_cfgstr = dict_str(GLOBAL_CONFIG, strvals=True)
        >>> print(global_cfgstr)
        >>> print(repo_cfgstr)

    Dev:
        dict_ = CONFIG_DICT
        strvals = False
        sorted_ = False
        newlines = True
        recursive = True
        indent_ = ''
    """

    if strvals:
        valfunc = str
    else:
        valfunc = repr

    def recursive_valfunc(val):
        if isinstance(val, dict):
            # recursive call
            return dict_str(val, strvals=strvals, sorted_=sorted_,
                            newlines=newlines, recursive=recursive,
                            indent_=indent_ + '    ')
        else:
            # base case
            return valfunc(val)

    iteritems = six.iteritems
    #def iteritems(x):
    #    try:
    #        return six.iteritems(x)
    #    except AttributeError:
    #        return iter(x.items())
    fmtstr = indent_ + '%r: %s,'
    if sorted_:
        iteritems = lambda iter_: iter(sorted(six.iteritems(iter_)))

    if recursive:
        itemstr_list = [fmtstr % (key, recursive_valfunc(val)) for (key, val) in iteritems(dict_)]
    else:
        itemstr_list = [fmtstr % (key, valfunc(val)) for (key, val) in iteritems(dict_)]
    return itemstr_list


def list_str(list_):
    return '[%s\n]' % indentjoin(list(list_), suffix=',')


def dict_str(dict_, strvals=False, sorted_=False, newlines=True, recursive=True, indent_=''):
    """
    Returns:
        str: a human-readable and execable string representation of a dictionary

    Example:
        >>> from utool.util_str import dict_str, dict_itemstr_list
        >>> import utool
        >>> REPO_CONFIG = utool.get_default_repo_config()
        >>> repo_cfgstr   = dict_str(REPO_CONFIG, strvals=True)
        >>> print(repo_cfgstr)
    """
    itemstr_list = dict_itemstr_list(dict_, strvals, sorted_, newlines, recursive, indent_)
    if newlines:
        return ('{%s\n' + indent_ + '}') % indentjoin(itemstr_list)
    else:
        return '{%s}' % ' '.join(itemstr_list)


def horiz_string(*args, **kwargs):
    """
    Horizontally prints objects

    Prints a list of objects ensuring that the next item in the list
    is all the way to the right of any previous items.

    Example1:
        >>> # ENABLE_DOCTEST
        >>> # Pretty printing of matrices demo / test
        >>> import utool
        >>> import numpy as np
        >>> # Wouldn't it be nice if we could print this operation easily?
        >>> B = np.array(((1, 2), (3, 4)))
        >>> C = np.array(((5, 6), (7, 8)))
        >>> A = B.dot(C)
        >>> # Eg 1:
        >>> result = (utool.hz_str('A = ', A, ' = ', B, ' * ', C))
        >>> print(result)
        A = [[19 22]  = [[1 2]  * [[5 6]
             [43 50]]    [3 4]]    [7 8]]

    Exam2:
        >>> # Eg 2:
        >>> str_list = ['A = ', str(B), ' * ', str(C)]
        >>> horizstr = (utool.horiz_string(*str_list))
        >>> result = (horizstr)
        >>> print(result)
        A = [[1 2]  * [[5 6]
             [3 4]]    [7 8]]
    """
    precision = kwargs.get('precision', None)

    if len(args) == 1 and not isinstance(args[0], str):
        val_list = args[0]
    else:
        val_list = args
    all_lines = []
    hpos = 0
    # for each value in the list or args
    for sx in range(len(val_list)):
        # Ensure value is a string
        val = val_list[sx]
        str_ = None
        if precision is not None:
            # Hack in numpy precision
            try:
                import numpy as np
                if isinstance(val, np.ndarray):
                    str_ = np.array_str(val, precision=precision, suppress_small=True)
            except ImportError:
                pass
        if str_ is None:
            str_ = str(val_list[sx])
        # continue with formating
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
    all_lines = [line.rstrip(' ') for line in all_lines]
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
    """ string representation of time deltas """
    timedelta = get_unix_timedelta(unixtime_diff)
    sign = '+' if unixtime_diff >= 0 else '-'
    timedelta_str = sign + str(timedelta)
    return timedelta_str


def str_between(str_, startstr, endstr):
    """ gets substring between two sentianl strings """
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
    r"""
    Left justifies text on the left side of character

    align

    Args:
        text (str): text to align
        character (str):

    Returns:
        str: new_text

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> character = '='
        >>> text = 'a = b\none = two\nthree = fish\n'
        >>> result = (align(text, '='))
        >>> print(result)
        a     = b
        one   = two
        three = fish
    """
    line_list = text.splitlines()
    new_lines = align_lines(line_list, character)
    new_text = '\n'.join(new_lines)
    return new_text


def align_lines(line_list, character='='):
    """
    Left justifies text on the left side of character

    align_lines

    Args:
        line_list (list of strs):
        character (str):

    Returns:
        list: new_lines

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> line_list = 'a = b\none = two\nthree = fish\n'.split('\n')
        >>> character = '='
        >>> new_lines = align_lines(line_list, character)
        >>> print('\n'.join(new_lines))
        a     = b
        one   = two
        three = fish
    """

    tup_list = [line.split(character) for line in line_list]
    maxlen = 0
    for tup in tup_list:
        if len(tup) >= 2:
            maxlen = max(maxlen, len(tup[0]))

    new_lines = []
    for tup in tup_list:
        if len(tup) >= 2:
            lhs = tup[0]
            rhs = character.join(tup[1:])
            newline = lhs.ljust(maxlen) + character + rhs
            new_lines.append(newline)
        else:
            new_lines.append(character.join(tup))
    return new_lines


def get_freespace_str(dir_='.'):
    """ returns string denoting free disk space in a directory """
    from utool import util_cplat
    return byte_str2(util_cplat.get_free_diskbytes(dir_))


# FIXME: HASHLEN is a global var in util_hash
def long_fname_format(fmt_str, fmt_dict, hashable_keys=[], max_len=64, hashlen=16, ABS_MAX_LEN=255):
    """
    Formats a string and hashes certain parts if the resulting string becomes
    too long. Used for making filenames fit onto disk.

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
    from utool import util_hash
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
            msg = ('Warning: Too big by %d chars. Exausted all options'
                   'to make fname fit into size. ')  % (diff,)
            print(msg)
            print('len(fname) = %r' % len(fname))
            print('fname = %r' % fname)
            if ABS_MAX_LEN is not None and len(fname) > ABS_MAX_LEN:
                raise AssertionError(msg)
    return fname


def replace_nonquoted_text(text, search_list, repl_list):
    """
    replace_nonquoted_text

    WARNING: this function is not safely implemented. It can break of searching
    for single characters or underscores. Depends on utool.modify_quoted_strs
    which is also unsafely implemented

    Args:
        text (?):
        search_list (list):
        repl_list (list):

    Example:
        >>> from utool.util_str import *  # NOQA
        >>> text = '?'
        >>> search_list = '?'
        >>> repl_list = '?'
        >>> result = replace_nonquoted_text(text, search_list, repl_list)
        >>> print(result)
    """
    # Hacky way to preserve quoted text
    # this will not work if search_list uses underscores or single characters
    def preserve_quoted_str(quoted_str):
        return '\'' + '_'.join(list(quoted_str[1:-1])) + '\''
    def unpreserve_quoted_str(quoted_str):
        return '\'' + ''.join(list(quoted_str[1:-1])[::2]) + '\''
    import utool as ut
    text_ = ut.modify_quoted_strs(text, preserve_quoted_str)
    for search, repl in zip(search_list, repl_list):
        text_ = text_.replace(search, repl)
    text_ = ut.modify_quoted_strs(text_, unpreserve_quoted_str)
    return text_


def singular_string(str_, plural_suffix='s', singular_suffix=''):
    """
    tries to use english grammar to make a string singular
    very naive implementation. will break often
    """
    return str_[:-1] if str_.endswith(plural_suffix) else str_


def remove_vowels(str_):
    """ strips all vowels from a string """
    for char_ in 'AEOIUaeiou':
        str_ = str_.replace(char_, '')
    return str_


def clipstr(str_, maxlen):
    """
    tries to shorten string as much as it can until it is just barely readable
    """
    if len(str_) > maxlen:
        str2 = (str_[0] + remove_vowels(str_[1:])).replace('_', '')
        if len(str2) > maxlen:
            return str2[0:maxlen]
        else:
            return str_[0:maxlen]
    else:
        return str_
#def parse_commas_wrt_groups(str_):
#    """
#    str_ = 'cdef np.ndarray[np.float64_t, cast=True] x, y, z'
#    """
#    nLParen = 0
#    nLBracket = 0
#    pass


def msgblock(key, text):
    """ puts text inside a visual ascii block """
    blocked_text = ''.join(
        [' + --- ', key, ' ---\n'] +
        [' | ' + line + '\n' for line in text.split('\n')] +
        [' L --- ', key, ' ---\n']
    )
    return blocked_text


def number_text_lines(text):
    r"""
    Args:
        text (str):

    Returns:
        str: text_with_lineno - string with numbered lines
    """
    numbered_linelist = [
        ''.join((('%2d' % (count + 1)), ' >>> ', line))
        for count, line in enumerate(text.splitlines())
    ]
    text_with_lineno = '\n'.join(numbered_linelist)
    return text_with_lineno


def get_textdiff(text1, text2):
    """
    Uses difflib to return a difference string between two
    similar texts

    References:
        http://www.java2s.com/Code/Python/Utility/IntelligentdiffbetweentextfilesTimPeters.htm
    """
    import difflib
    text1_lines = text1.splitlines()
    text2_lines = text2.splitlines()
    diffline_gen = difflib.ndiff(text1_lines, text2_lines)
    diff_lines = [line for line in diffline_gen
                  if len(line) > 0 and line[0] in '+-?']
    return '\n'.join(diff_lines)


def cond_phrase(list_, cond='or'):
    """
    takes a list of words and joins them using english conjunction rules

    not sure what the right name for this should be
    something to do with conjunctions?

    Args:
        list_ (list):  of strings
        cond (str): a conjunction

    Returns:
        str: the conditional phrase

    References:
        http://en.wikipedia.org/wiki/Conjunction_(grammar)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> list_ = ['a', 'b', 'c']
        >>> result = cond_phrase(list_, 'or')
        >>> print(result)
        a, b, or c

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> list_ = ['a', 'b']
        >>> result = cond_phrase(list_, 'and')
        >>> print(result)
        a and b
    """
    if len(list_) == 0:
        return ''
    elif len(list_) == 1:
        return list_[0]
    elif len(list_) == 2:
        return ' '.join((list_[0], cond, list_[1]))
    else:
        condstr = ''.join((', ' + cond, ' '))
        return ', '.join((', '.join(list_[:-2]), condstr.join(list_[-2:])))

if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_str; utool.doctest_funcs(utool.util_str, allexamples=True)"
        python -c "import utool, utool.util_str; utool.doctest_funcs(utool.util_str)"
        python -m utool.util_str
        python -m utool.util_str --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
