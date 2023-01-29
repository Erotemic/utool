# -*- coding: utf-8 -*-
r"""
Module that handles string formating and manipulation of various data
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import six
import re
import textwrap
from six.moves import map, range, reduce  # NOQA
import itertools  # NOQA
import math
import collections
from os.path import split
from utool import util_type
#from utool import util_time
from utool import util_cplat
from utool._internal import meta_util_six
from utool._internal import meta_util_arg
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)

ENABLE_COLORS = (not util_cplat.WIN32 and
                 not meta_util_arg.get_argflag('--nopygments'))

if util_type.HAVE_NUMPY:
    import numpy as np
if util_type.HAVE_PANDAS:
    import pandas as pd

TAU = (2 * math.pi)  # References: tauday.com

NO_TRUNCATE = '--no-truncate' in sys.argv

TRIPLE_DOUBLE_QUOTE = r'"' * 3
TRIPLE_SINGLE_QUOTE = r"'" * 3
SINGLE_QUOTE = r"'"
DOUBLE_QUOTE = r'"'
BACKSLASH = '\\'
NEWLINE = '\n'

TAUFMTSTR = '{coeff:,.1f}{taustr}'
if '--myway' not in sys.argv:
    TAUSTR = '*2pi'
else:
    TAUSTR = 'tau'


def is_byte_encoded_unicode(str_):
    return r'\x' in repr(str_)


ensure_unicode = meta_util_six.ensure_unicode


def ensure_ascii(str_):
    try:
        return str_.encode('ascii')
    except UnicodeDecodeError:
        print("it was not a ascii-encoded unicode string")
    else:
        print("It may have been an ascii-encoded unicode string")
    return str_


def ensure_unicode_strlist(str_list):
    __STR__ = util_type.__STR__
    flag_list = [not isinstance(str_, __STR__) and is_byte_encoded_unicode(str_)
                 for str_ in str_list]
    new_str_list = [str_.decode('utf-8') if flag else __STR__(str_)
                    for str_, flag in zip(str_list, flag_list)]
    return new_str_list


def replace_between_tags(text, repl_, start_tag, end_tag=None):
    r"""
    Replaces text between sentinal lines in a block of text.

    Args:
        text (str):
        repl_ (str):
        start_tag (str):
        end_tag (str): (default=None)

    Returns:
        str: new_text

    CommandLine:
        python -m utool.util_str --exec-replace_between_tags

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> text = ut.codeblock(
            '''
            class:
                # <FOO>
                bar
                # </FOO>
                baz
            ''')
        >>> repl_ = 'spam'
        >>> start_tag = '# <FOO>'
        >>> end_tag = '# </FOO>'
        >>> new_text = replace_between_tags(text, repl_, start_tag, end_tag)
        >>> result = ('new_text =\n%s' % (str(new_text),))
        >>> print(result)
        new_text =
        class:
            # <FOO>
        spam
            # </FOO>
            baz
    """
    new_lines = []
    editing = False
    lines = text.split('\n')
    for line in lines:
        if not editing:
            new_lines.append(line)
        if line.strip().startswith(start_tag):
            new_lines.append(repl_)
            editing = True
        if end_tag is not None and line.strip().startswith(end_tag):
            editing = False
            new_lines.append(line)
    new_text = '\n'.join(new_lines)
    return new_text


def theta_str(theta, taustr=TAUSTR, fmtstr='{coeff:,.1f}{taustr}'):
    r"""
    Format theta so it is interpretable in base 10

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
    r""" makes a string from an integer bounding box """
    if bbox is None:
        return 'None'
    fmtstr = sep.join(['%' + six.text_type(pad) + 'd'] * 4)
    return '(' + fmtstr % tuple(bbox) + ')'


def verts_str(verts, pad=1):
    r""" makes a string from a list of integer verticies """
    if verts is None:
        return 'None'
    fmtstr = ', '.join(['%' + six.text_type(pad) + 'd' +
                        ', %' + six.text_type(pad) + 'd'] * 1)
    return ', '.join(['(' + fmtstr % vert + ')' for vert in verts])

# --- Strings ----


def scalar_str(val, precision=None, max_precision=None):
    isfloat = (isinstance(val, (float)) or util_type.is_float(val))
    if precision is not None and isfloat:
        return ('%.' + six.text_type(precision) + 'f') % (val,)
    elif max_precision is not None and isfloat:
        str_ = ('%.' + six.text_type(max_precision) + 'f') % (val,)
        str_ = str_.rstrip('0.')
        return str_
    else:
        return six.text_type(val)


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
        1 2 3 4
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


def get_minimum_indentation(text):
    r"""
    returns the number of preceding spaces

    Args:
        text (str): unicode text

    Returns:
        int: indentation

    CommandLine:
        python -m utool.util_str --exec-get_minimum_indentation --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> import utool as ut
        >>> text = '    foo\n   bar'
        >>> result = get_minimum_indentation(text)
        >>> print(result)
        3
    """
    lines = text.split('\n')
    indentations = [get_indentation(line_)
                    for line_ in lines  if len(line_.strip()) > 0]
    if len(indentations) == 0:
        return 0
    return min(indentations)


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
    return textwrap.dedent(block_str).strip('\n')


def flatten_textlines(text):
    new_text = text
    new_text = re.sub(' *\n *', ' ', new_text, flags=re.MULTILINE).strip(' ')
    return new_text


def remove_doublspaces(text):
    new_text = text
    new_text = re.sub('  *', ' ', new_text)
    #, flags=re.MULTILINE)
    return new_text


def remove_doublenewlines(text):
    new_text = text
    new_text = re.sub('\n(\n| )*', '\n', new_text)
    return new_text


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
    new_lines = list(map(flatten_textlines, multiline_text.split('\n\n')))
    new_text = '\n\n'.join(new_lines)
    return new_text


def indent(str_, indent='    '):
    r"""
    Indents a block of text

    Args:
        str_ (str):
        indent (str): (default = '    ') TODO rename to indent_ or rename func

    Returns:
        str:

    CommandLine:
        python -m utool.util_str --test-indent

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> str_ = 'foobar\nbazbiz'
        >>> indent = '    '
        >>> result = indent(str_, indent)
        >>> print(result)
    """
    return indent + indent_rest(str_, indent)


def indent_rest(str_, indent='    '):
    """ TODO fix name Indents every part of the string except the beginning
    SeeAlso: ibeis/templates/generate_notebook.py
    """
    return str_.replace('\n', '\n' + indent)


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
    return indent_ + indent_.join([six.text_type(str_) + suffix
                                   for str_ in strlist])


def truncate_str(str_, maxlen=110, truncmsg=' ~~~TRUNCATED~~~ '):
    """
    Removes the middle part of any string over maxlen characters.
    """
    if NO_TRUNCATE:
        return str_
    if maxlen is None or maxlen == -1 or len(str_) < maxlen:
        return str_
    else:
        maxlen_ = maxlen - len(truncmsg)
        lowerb  = int(maxlen_ * .8)
        upperb  = maxlen_ - lowerb
        tup = (str_[:lowerb], truncmsg, str_[-upperb:])
        return ''.join(tup)


def pack_into(text, textwidth=160, breakchars=' ', break_words=True,
              newline_prefix='', wordsep=' ', remove_newlines=True):
    r"""

    DEPRICATE IN FAVOR OF textwrap.wrap

    TODO: Look into textwrap.wrap

    Inserts newlines into a string enforcing a maximum textwidth.
    Similar to vim's gq command in visual select mode.

    breakchars is a string containing valid characters to insert a newline
    before or after.

    break_words is True if words are allowed to be split over multiple lines.

    all inserted newlines are prefixed with newline_prefix

    #FIXME:

    Example:
        >>> # DISABLE_DOCTEST
        >>> text = "set_image_uris(ibs<139684018194000>, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [u'66ec193a-1619-b3b6-216d-1784b4833b61.jpg', u'd8903434-942f-e0f5-d6c2-0dcbe3137bf7.jpg', u'b73b72f4-4acb-c445-e72c-05ce02719d3d.jpg', u'0cd05978-3d83-b2ee-2ac9-798dd571c3b3.jpg', u'0a9bc03d-a75e-8d14-0153-e2949502aba7.jpg', u'2deeff06-5546-c752-15dc-2bd0fdb1198a.jpg', u'a9b70278-a936-c1dd-8a3b-bc1e9a998bf0.png', u'42fdad98-369a-2cbc-67b1-983d6d6a3a60.jpg', u'c459d381-fd74-1d99-6215-e42e3f432ea9.jpg', u'33fd9813-3a2b-774b-3fcc-4360d1ae151b.jpg', u'97e8ea74-873f-2092-b372-f928a7be30fa.jpg', u'588bc218-83a5-d400-21aa-d499832632b0.jpg', u'163a890c-36f2-981e-3529-c552b6d668a3.jpg'], ) "  # NOQA
        >>> textwidth = 160
        >>> breakchars = ' '
        >>> break_words = True
        >>> newline_prefix = '    '
        >>> wordsep = ' '
        >>> packstr1 = pack_into(text, textwidth, breakchars, break_words, newline_prefix, wordsep)
        >>> break_words = False
        >>> packstr2 = pack_into(text, textwidth, breakchars, break_words, newline_prefix, wordsep)
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
    word_list = text.split(breakchars)
    if remove_newlines:
        word_list = [word.replace('\n', '') for word in word_list]
    for word in word_list:
        available = textwidth_ - len(line_list[-1])
        # Check to see if we need to make a new line
        while len(word) > available:
            if break_words:
                # If we are allowed to break words over multiple lines
                # Fill the rest of the available textwidth with part of the
                # word
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


def packtext(text, width=80):
    r"""
    Args:
        text (str):

    CommandLine:
        python -m utool.util_str --exec-pack_paragraph --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> import utool as ut
        >>> width = 80
        >>> text = lorium_ipsum()
        >>> result = packtext(text)
        >>> print(result)
    """
    import utool as ut
    import textwrap
    new_text = '\n'.join(textwrap.wrap(text, width))
    new_text = ut.remove_doublspaces(new_text).strip()
    return new_text


def joins(string, list_, with_head=True, with_tail=False, tostrip='\n'):
    head = string if with_head else ''
    tail = string if with_tail else ''
    to_return = head + string.join(map(six.text_type, list_)) + tail
    to_return = to_return.strip(tostrip)
    return to_return


def indent_list(indent, list_):
    return list(map(lambda item: indent + six.text_type(item), list_))


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
        >>> from utool.util_str import *  # NOQA
        >>> import utool as ut
        >>> num_list = sorted([4.2 / (10.0 ** exp_)
        >>>                    for exp_ in range(-13, 13, 4)])
        >>> secstr_list = [seconds_str(num, prefix=None) for num in num_list]
        >>> result = (', '.join(secstr_list))
        >>> print(result)
        0.04 ns, 0.42 us, 4.20 ms, 0.04 ks, 0.42 Ms, 4.20 Gs, 42.00 Ts
    """
    exponent_list = [-12, -9, -6, -3, 0, 3, 6, 9, 12]
    small_prefix_list = ['p', 'n', 'u', 'm', '', 'k', 'M', 'G', 'T']
    #large_prefix_list = ['pico', 'nano', 'micro', 'mili', '', 'kilo', 'mega',
    # 'giga', 'tera']
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


def parse_bytes(bytes_str):
    """
    uint8_size = ut.parse_bytes('1B')
    image_size = ut.parse_bytes('3.5MB')
    float32_size = ut.parse_bytes('32bit')
    desc_size = 128 * uint8_size
    kpts_size = 6 * float32_size
    chip_size = ut.parse_bytes('400 KB')
    probchip_size = ut.parse_bytes('50 KB')
    nImgs = 80000 # 80,000
    nAnnots = nImgs * 2
    desc_per_img = 3000
    size_stats = {
        'image': nImgs * image_size,
        'chips': nAnnots * chip_size,
        'probchips': nAnnots * probchip_size,
        'desc': nAnnots * desc_size * desc_per_img,
        'kpts': nAnnots * kpts_size * desc_per_img,
    }
    print(ut.repr3(ut.map_dict_vals(ut.byte_str2, size_stats), align=True))
    print('total = ' + ut.byte_str2(sum(size_stats.values())))
    """
    import utool as ut
    import re
    numstr = ut.named_field('num', r'\d\.?\d*')
    unitstr = ut.named_field('unit', r'[a-zA-Z]+')
    match = re.match(numstr + ' *' + unitstr, bytes_str)
    nUnits = float(match.groupdict()['num'])
    unit = match.groupdict()['unit'].upper()
    nBytes = get_bytes(nUnits, unit)
    return nBytes


def get_bytes(nUnits, unit):
    unitdict = {'TB': 2 ** 40, 'GB': 2 ** 30, 'MB': 2 ** 20, 'KB': 2 ** 10, 'B': 2 ** 0}
    # https://en.wikipedia.org/wiki/Units_of_information#Obsolete_and_unusual_units
    unitdict['BIT'] = 1 / 8
    unitdict['CRUMB'] = 1 / 4
    unitdict['NIBBLE'] = 1 / 2
    unitdict['CHOMP'] = 2
    unit_nBytes = unitdict[unit]
    nBytes = unit_nBytes * nUnits
    return nBytes


def byte_str2(nBytes, precision=2):
    """
    Automatically chooses relevant unit (KB, MB, or GB) for displaying some
    number of bytes.

    Args:
        nBytes (int):

    Returns:
        str:

    CommandLine:
        python -m utool.util_str --exec-byte_str2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> import utool as ut
        >>> nBytes_list = [1, 100, 1024,  1048576, 1073741824, 1099511627776]
        >>> result = ut.list_str(list(map(byte_str2, nBytes_list)), nl=False)
        >>> print(result)
        ['0.00 KB', '0.10 KB', '1.00 KB', '1.00 MB', '1.00 GB', '1.00 TB']
    """
    nAbsBytes = abs(nBytes)
    if nAbsBytes < 2.0 ** 10:
        return byte_str(nBytes, 'KB', precision=precision)
    if nAbsBytes < 2.0 ** 20:
        return byte_str(nBytes, 'KB', precision=precision)
    if nAbsBytes < 2.0 ** 30:
        return byte_str(nBytes, 'MB', precision=precision)
    if nAbsBytes < 2.0 ** 40:
        return byte_str(nBytes, 'GB', precision=precision)
    else:
        return byte_str(nBytes, 'TB', precision=precision)


def byte_str(nBytes, unit='bytes', precision=2):
    """
    representing the number of bytes with the chosen unit

    Returns:
        str
    """
    #return (nBytes * ureg.byte).to(unit.upper())
    if unit.lower().startswith('b'):
        nUnit = nBytes
    elif unit.lower().startswith('k'):
        nUnit =  nBytes / (2.0 ** 10)
    elif unit.lower().startswith('m'):
        nUnit =  nBytes / (2.0 ** 20)
    elif unit.lower().startswith('g'):
        nUnit = nBytes / (2.0 ** 30)
    elif unit.lower().startswith('t'):
        nUnit = nBytes / (2.0 ** 40)
    else:
        raise NotImplementedError('unknown nBytes=%r unit=%r' % (nBytes, unit))
    return repr2(nUnit, precision=precision) + ' ' + unit


def second_str(nsecs, unit=None, precision=None, abbrev=True):
    import utool as ut
    import pint
    ureg = pint.UnitRegistry()
    sec_quant = nsecs * ureg.s
    if unit is not None:
        unit_quant = sec_quant.to(unit)
    else:
        unit_quant = sec_quant.to_compact()
    unit_str = str(unit_quant.units)
    if abbrev:
        lookup = {'millisecond': 'ms', 'second': 's',
                  'nanosecond': 'ns', 'microsecond': 'µs'}
        unit_str = lookup.get(unit_str, unit_str)
    else:
        unit_str = ut.pluralize(unit_str, unit_quant.magnitude)
    quant_str = ut.repr2(unit_quant.magnitude, precision=precision) + ' ' + unit_str
    # quant_str =  str(unit_quant)
    return quant_str


def file_megabytes_str(fpath):
    from utool import util_path
    return ('%.2f MB' % util_path.file_megabytes(fpath))


def func_str(func, args=[], kwargs={}, type_aliases=[], packed=False,
             packkw=None, truncate=False):
    """
    string representation of function definition

    Returns:
        str: a representation of func with args, kwargs, and type_aliases

    Args:
        func (function):
        args (list): argument values (default = [])
        kwargs (dict): kwargs values (default = {})
        type_aliases (list): (default = [])
        packed (bool): (default = False)
        packkw (None): (default = None)

    Returns:
        str: func_str

    CommandLine:
        python -m utool.util_str --exec-func_str

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> func = byte_str
        >>> args = [1024, 'MB']
        >>> kwargs = dict(precision=2)
        >>> type_aliases = []
        >>> packed = False
        >>> packkw = None
        >>> _str = func_str(func, args, kwargs, type_aliases, packed, packkw)
        >>> result = _str
        >>> print(result)
        byte_str(1024, 'MB', precision=2)
    """
    import utool as ut
    # if truncate:
    # truncatekw = {'maxlen': 20}
    # else:
    truncatekw = {}

    argrepr_list = ([] if args is None else
                    ut.get_itemstr_list(args, nl=False, truncate=truncate,
                                        truncatekw=truncatekw))
    kwrepr_list = ([] if kwargs is None else
                   ut.dict_itemstr_list(kwargs, explicit=True, nl=False,
                                        truncate=truncate,
                                        truncatekw=truncatekw))
    repr_list = argrepr_list + kwrepr_list

    argskwargs_str = ', '.join(repr_list)
    _str = '%s(%s)' % (meta_util_six.get_funcname(func), argskwargs_str)
    if packed:
        packkw_ = dict(textwidth=80, nlprefix='    ', break_words=False)
        if packkw is not None:
            packkw_.update(packkw_)
        _str = packstr(_str, **packkw_)
    return _str


def func_defsig(func, with_name=True):
    """
    String of function definition signature

    Args:
        func (function): live python function

    Returns:
        str: defsig

    CommandLine:
        xdoctest -m utool.util_str func_defsig

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> func = func_str
        >>> defsig = func_defsig(func)
        >>> result = str(defsig)
        >>> print(result)
        func_str(func, args=[], kwargs={}, type_aliases=[], packed=False, packkw=None, truncate=False)
    """
    # from utool.util_inspect import six_get_argspect
    import inspect
    try:
        argspec = inspect.signature(func)
        defsig = str(argspec)
    except Exception:
        argspec = inspect.getargspec(func)
        # argspec = six_get_argspect(func)
        (args, varargs, varkw, defaults) = argspec
        defsig = inspect.formatargspec(*argspec)
    if with_name:
        defsig = get_callable_name(func) + defsig
    return defsig


def func_callsig(func, with_name=True):
    """
    String of function call signature

    Args:
        func (function): live python function

    Returns:
        str: callsig
    """
    raise NotImplementedError('dont think anything used this')
    import inspect
    argspec = inspect.getargspec(func)
    (args, varargs, varkw, defaults) = argspec
    callsig = inspect.formatargspec(*argspec[0:3])
    if with_name:
        callsig = get_callable_name(func) + callsig
    return callsig


# def array_repr2(arr, max_line_width=None, precision=None, suppress_small=None,
#                 force_dtype=False, with_dtype=None, **kwargs):
#     """ extended version of np.core.numeric.array_repr

#     ut.editfile(np.core.numeric.__file__)

#     On linux:
#     _typelessdata [numpy.int64, numpy.float64, numpy.complex128, numpy.int64]

#     On BakerStreet
#     _typelessdata [numpy.int32, numpy.float64, numpy.complex128, numpy.int32]

#     # WEIRD
#     np.int64 is np.int64
#     _typelessdata[0] is _typelessdata[-1]
#     _typelessdata[0] == _typelessdata[-1]

#     TODO:
#         replace force_dtype with with_dtype


#     id(_typelessdata[-1])
#     id(_typelessdata[0])


#     from numpy.core.numeric import _typelessdata
#     _typelessdata

#     References:
#         http://stackoverflow.com/questions/28455982/why-are-there-two-np-int64s
#         -in-numpy-core-numeric-typelessdata-why-is-numpy-in/28461928#28461928
#     """
#     data = arr
#     strvals = kwargs.get('strvals', False)
#     # precision = kwargs.get('precision', None)
#     # suppress_small = kwargs.get('supress_small', None)
#     # max_line_width = kwargs.get('max_line_width', None)
#     # with_dtype = kwargs.get('with_dtype', not strvals)
#     newlines = kwargs.pop('nl', kwargs.pop('newlines', 1))

#     # if with_dtype and strvals:
#     #     raise ValueError('cannot format with strvals and dtype')

#     separator = ' '

#     if strvals:
#         prefix = ''
#         suffix = ''
#     else:
#         modname = type(data).__module__
#         # substitute shorthand for numpy module names
#         np_nice = 'np'
#         modname = re.sub('\\bnumpy\\b', np_nice, modname)
#         modname = re.sub('\\bma.core\\b', 'ma', modname)

#         class_name = type(data).__name__
#         if class_name == 'ndarray':
#             class_name = 'array'

#         prefix = class_name + '('

#         if with_dtype:
#             dtype_repr = data.dtype.name
#             # dtype_repr = np.core.arrayprint.dtype_short_repr(data.dtype)
#             suffix = ',{}dtype={}.{})'.format(itemsep, np_nice, dtype_repr)
#         else:
#             suffix = ')'

#     if not strvals and data.size == 0 and data.shape != (0,):
#         # Special case for displaying empty data
#         prefix = modname + '.empty('
#         body = repr(tuple(map(int, data.shape)))
#     else:
#         body = np.array2string(data, precision=precision,
#                                separator=separator,
#                                suppress_small=suppress_small,
#                                prefix=prefix,
#                                legacy='1.13',
#                                max_line_width=max_line_width)
#     if not newlines:
#         # remove newlines if we need to
#         body = re.sub('\n *', '', body)
#     formatted = body
#     #+ suffix
#     return formatted

    # the old way of doing this breaks
    # import numpy as np
    # from numpy.core.numeric import _typelessdata

    # if arr.__class__ is not np.ndarray:
    #     cName = arr.__class__.__name__
    # else:
    #     cName = 'array'

    # prefix = cName + '('

    # if arr.size > 0 or arr.shape == (0,):
    #     separator = ', '
    #     lst = array2string2(
    #         arr, max_line_width, precision, suppress_small, separator, prefix,
    #         **kwargs)
    # else:
    #     # show zero-length shape unless it is (0,)
    #     lst = '[], shape=%s' % (repr(arr.shape),)

    # skipdtype = ((arr.dtype.type in _typelessdata) and arr.size > 0)

    # if with_dtype is None:
    #     with_dtype = not (skipdtype and not (cName == 'array' and force_dtype))

    # if not with_dtype:
    #     return '%s(%s)' % (cName, lst)
    # else:
    #     typename = arr.dtype.name
    #     # Quote typename in the output if it is 'complex'.
    #     if typename and not (typename[0].isalpha() and typename.isalnum()):
    #         typename = '\'%s\'' % typename

    #     lf = ''
    #     if issubclass(arr.dtype.type, np.flexible):
    #         if arr.dtype.names:
    #             typename = '%s' % six.text_type(arr.dtype)
    #         else:
    #             typename = '\'%s\'' % six.text_type(arr.dtype)
    #         lf = '\n' + ' ' * len(prefix)
    #     return cName + '(%s, %sdtype=%s)' % (lst, lf, typename)


# def array2string2(a, max_line_width=None, precision=None, suppress_small=None,
#                   separator=' ', prefix="", style=repr, formatter=None,
#                   threshold=None):
#     """
#     expanded version of np.core.arrayprint.array2string
#     """
#     import numpy as np

#     if a.shape == ():
#         x = a.item()
#         try:
#             import warnings
#             lst = a._format(x)
#             msg = "The `_format` attribute is deprecated in Numpy " \
#                   "2.0 and will be removed in 2.1. Use the " \
#                   "`formatter` kw instead."
#             warnings.warn(msg, DeprecationWarning)
#         except AttributeError:
#             if isinstance(x, tuple):
#                 x = np.core.arrayprint._convert_arrays(x)
#             lst = style(x)
#     elif reduce(np.core.arrayprint.product, a.shape) == 0:
#         # treat as a null array if any of shape elements == 0
#         lst = "[]"
#     else:
#         lst = _array2string2(
#             a, max_line_width, precision, suppress_small, separator, prefix,
#             formatter=formatter, threshold=threshold)
#     return lst


# def _array2string2(a, max_line_width, precision, suppress_small, separator=' ',
#                    prefix="", formatter=None, threshold=None):
#     """
#     expanded version of np.core.arrayprint._array2string
#     TODO: make a numpy pull request with a fixed version

#     """
#     arrayprint = np.core.arrayprint

#     if max_line_width is None:
#         max_line_width = arrayprint._line_width

#     if precision is None:
#         precision = arrayprint._float_output_precision

#     if suppress_small is None:
#         suppress_small = arrayprint._float_output_suppress_small

#     if formatter is None:
#         formatter = arrayprint._formatter

#     if threshold is None:
#         threshold = arrayprint._summaryThreshold

#     if threshold > 0 and a.size > threshold:
#         summary_insert = "..., "
#         data = arrayprint._leading_trailing(a)
#     else:
#         summary_insert = ""
#         data = arrayprint.ravel(a)

#     formatdict = {'bool' : arrayprint._boolFormatter,
#                   'int' : arrayprint.IntegerFormat(data),
#                   'float' : arrayprint.FloatFormat(data, precision, suppress_small),
#                   'longfloat' : arrayprint.LongFloatFormat(precision),
#                   'complexfloat' : arrayprint.ComplexFormat(data, precision,
#                                                             suppress_small),
#                   'longcomplexfloat' : arrayprint.LongComplexFormat(precision),
#                   'datetime' : arrayprint.DatetimeFormat(data),
#                   'timedelta' : arrayprint.TimedeltaFormat(data),
#                   'numpystr' : arrayprint.repr_format,
#                   'str' : str}

#     if formatter is not None:
#         fkeys = [k for k in formatter.keys() if formatter[k] is not None]
#         if 'all' in fkeys:
#             for key in formatdict.keys():
#                 formatdict[key] = formatter['all']
#         if 'int_kind' in fkeys:
#             for key in ['int']:
#                 formatdict[key] = formatter['int_kind']
#         if 'float_kind' in fkeys:
#             for key in ['float', 'longfloat']:
#                 formatdict[key] = formatter['float_kind']
#         if 'complex_kind' in fkeys:
#             for key in ['complexfloat', 'longcomplexfloat']:
#                 formatdict[key] = formatter['complex_kind']
#         if 'str_kind' in fkeys:
#             for key in ['numpystr', 'str']:
#                 formatdict[key] = formatter['str_kind']
#         for key in formatdict.keys():
#             if key in fkeys:
#                 formatdict[key] = formatter[key]

#     try:
#         format_function = a._format
#         msg = "The `_format` attribute is deprecated in Numpy 2.0 and " \
#               "will be removed in 2.1. Use the `formatter` kw instead."
#         import warnings
#         warnings.warn(msg, DeprecationWarning)
#     except AttributeError:
#         # find the right formatting function for the array
#         dtypeobj = a.dtype.type
#         if issubclass(dtypeobj, np.core.arrayprint._nt.bool_):
#             format_function = formatdict['bool']
#         elif issubclass(dtypeobj, np.core.arrayprint._nt.integer):
#             if issubclass(dtypeobj, np.core.arrayprint._nt.timedelta64):
#                 format_function = formatdict['timedelta']
#             else:
#                 format_function = formatdict['int']
#         elif issubclass(dtypeobj, np.core.arrayprint._nt.floating):
#             if issubclass(dtypeobj, np.core.arrayprint._nt.longfloat):
#                 format_function = formatdict['longfloat']
#             else:
#                 format_function = formatdict['float']
#         elif issubclass(dtypeobj, np.core.arrayprint._nt.complexfloating):
#             if issubclass(dtypeobj, np.core.arrayprint._nt.clongfloat):
#                 format_function = formatdict['longcomplexfloat']
#             else:
#                 format_function = formatdict['complexfloat']
#         elif issubclass(dtypeobj, (np.core.arrayprint._nt.unicode_,
#                                    np.core.arrayprint._nt.string_)):
#             format_function = formatdict['numpystr']
#         elif issubclass(dtypeobj, np.core.arrayprint._nt.datetime64):
#             format_function = formatdict['datetime']
#         else:
#             format_function = formatdict['numpystr']

#     # skip over "["
#     next_line_prefix = " "
#     # skip over array(
#     next_line_prefix += " " * len(prefix)

#     lst = np.core.arrayprint._formatArray(a, format_function, len(a.shape), max_line_width,
#                                           next_line_prefix, separator,
#                                           np.core.arrayprint._summaryEdgeItems, summary_insert)[:-1]
#     return lst


def numpy_str(arr, strvals=False, precision=None, pr=None,
              force_dtype=False,
              with_dtype=None, suppress_small=None, max_line_width=None,
              threshold=None, **kwargs):
    """
    suppress_small = False turns off scientific representation
    """
    # strvals = kwargs.get('strvals', False)
    itemsep = kwargs.get('itemsep', ' ')
    # precision = kwargs.get('precision', None)
    # suppress_small = kwargs.get('supress_small', None)
    # max_line_width = kwargs.get('max_line_width', None)
    # with_dtype = kwargs.get('with_dtype', False)
    newlines = kwargs.pop('nl', kwargs.pop('newlines', 1))
    data = arr

    # if with_dtype and strvals:
    #     raise ValueError('cannot format with strvals and dtype')

    separator = ',' + itemsep

    if strvals:
        prefix = ''
        suffix = ''
    else:
        modname = type(data).__module__
        # substitute shorthand for numpy module names
        np_nice = 'np'
        modname = re.sub('\\bnumpy\\b', np_nice, modname)
        modname = re.sub('\\bma.core\\b', 'ma', modname)

        class_name = type(data).__name__
        if class_name == 'ndarray':
            class_name = 'array'

        prefix = modname + '.' + class_name + '('

        if with_dtype:
            dtype_repr = data.dtype.name
            # dtype_repr = np.core.arrayprint.dtype_short_repr(data.dtype)
            suffix = ',{}dtype={}.{})'.format(itemsep, np_nice, dtype_repr)
        else:
            suffix = ')'

    if not strvals and data.size == 0 and data.shape != (0,):
        # Special case for displaying empty data
        prefix = modname + '.empty('
        body = repr(tuple(map(int, data.shape)))
    else:
        body = np.array2string(data, precision=precision,
                               separator=separator,
                               suppress_small=suppress_small,
                               prefix=prefix,
                               max_line_width=max_line_width)
    if not newlines:
        # remove newlines if we need to
        body = re.sub('\n *', '', body)
    formatted = prefix + body + suffix
    return formatted

    # kwargs = kwargs.copy()
    # if 'suppress' in kwargs:
    #     suppress_small = kwargs['suppress']
    # if max_line_width is None and 'linewidth' in kwargs:
    #     max_line_width = kwargs.pop('linewidth')

    # import ubelt as ub
    # return ub.repr2(arr, precision=precision, force_dtype=force_dtype,
    #                 with_dtype=with_dtype, suppress_small=suppress_small,
    #                 max_line_width=max_line_width, threshold=threshold,
    #                 **kwargs)

    # if pr is not None:
    #     precision = pr
    # # TODO: make this a util_str func for numpy reprs
    # if strvals:
    #     valstr = np.array_str(arr, precision=precision,
    #                           suppress_small=suppress_small, **kwargs)
    # else:
    #     #valstr = np.array_repr(arr, precision=precision)
    #     valstr = array_repr2(arr, precision=precision, force_dtype=force_dtype,
    #                          with_dtype=with_dtype,
    #                          suppress_small=suppress_small,
    #                          max_line_width=max_line_width,
    #                          threshold=threshold, **kwargs)
    #     numpy_vals = itertools.chain(util_type.NUMPY_SCALAR_NAMES, ['array'])
    #     for npval in numpy_vals:
    #         valstr = valstr.replace(npval, 'np.' + npval)
    #     if valstr.find('\n') >= 0:
    #         # Align multiline arrays
    #         valstr = valstr.replace('\n', '\n   ')
    # return valstr


def reprfunc(val, precision=None):
    r"""
    Args:
        val (?):
        precision (None): (default = None)

    Returns:
        str:

    CommandLine:
        python -m utool.util_str reprfunc --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> import utool as ut
        >>> #vals = [{'foo': [1/3, 2]}, np.float64, 1/3, 'foo']
        >>> vals = [{'foo': [1, 2]}, np.float64, 1/3, 'foo']
        >>> precision = 2
        >>> result = ut.repr3([reprfunc(val, precision) for val in vals], nobr=True)
        >>> print(result)
        "{u'foo': [1, 2]}",
        'numpy.float64',
        '0.33',
        "'foo'",
    """
    if isinstance(val, six.string_types):
        repr_ = repr(val)
        if repr_.startswith('u\'') or repr_.startswith('u"'):
            # Remove unicode repr from python2 to agree with python3
            # output
            repr_ = repr_[1:]
    elif precision is not None and (isinstance(val, (float)) or util_type.is_float(val)):
        return scalar_str(val, precision)
    elif isinstance(val, type):
        import utool as ut
        repr_ = ut.type_str(val)
    else:
        #import utool as ut
        #print('val = %r' % (val,))
        #ut.repr2(val)
        repr_ = repr(val)
    return repr_


def list_str_summarized(list_, list_name, maxlen=5):
    """
    prints the list members when the list is small and the length when it is
    large
    """
    if len(list_) > maxlen:
        return 'len(%s)=%d' % (list_name, len(list_))
    else:
        return '%s=%r' % (list_name, list_)


def countdown_flag(count_or_bool):
    return _rectify_countdown_or_bool(count_or_bool)


def _rectify_countdown_or_bool(count_or_bool):
    """
    used by recrusive functions to specify which level to turn a bool on in
    counting down yeilds True, True, ..., False
    conting up yeilds False, False, False, ... True

    Args:
        count_or_bool (bool or int): if positive will count down, if negative
            will count up, if bool will remain same

    Returns:
        int or bool: count_or_bool_

    CommandLine:
        python -m utool.util_str --test-_rectify_countdown_or_bool

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import _rectify_countdown_or_bool  # NOQA
        >>> count_or_bool = True
        >>> a1 = (_rectify_countdown_or_bool(2))
        >>> a2 = (_rectify_countdown_or_bool(1))
        >>> a3 = (_rectify_countdown_or_bool(0))
        >>> a4 = (_rectify_countdown_or_bool(-1))
        >>> a5 = (_rectify_countdown_or_bool(-2))
        >>> a6 = (_rectify_countdown_or_bool(True))
        >>> a7 = (_rectify_countdown_or_bool(False))
        >>> result = [a1, a2, a3, a4, a5, a6, a7]
        >>> print(result)
        [1.0, 0.0, 0, 0.0, -1.0, True, False]

        [1.0, True, False, False, -1.0, True, False]
    """
    if count_or_bool is True or count_or_bool is False:
        count_or_bool_ = count_or_bool
    elif isinstance(count_or_bool, int):
        if count_or_bool == 0:
            return 0
        sign_ =  math.copysign(1, count_or_bool)
        count_or_bool_ = int(count_or_bool - sign_)
        #if count_or_bool_ == 0:
        #    return sign_ == 1
    else:
        count_or_bool_ = False
    return count_or_bool_


def trunc_repr(obj, maxlen=50):
    return truncate_str(repr2(obj), maxlen, truncmsg='~//~')


def repr2(obj_, **kwargs):
    """
    Attempt to replace repr more configurable
    pretty version that works the same in both 2 and 3
    """
    kwargs['nl'] = kwargs.pop('nl', kwargs.pop('newlines', False))
    val_str = _make_valstr(**kwargs)
    return val_str(obj_)


def repr2_json(obj_, **kwargs):
    """ hack for json reprs """
    import utool as ut
    kwargs['trailing_sep'] = False
    json_str = ut.repr2(obj_, **kwargs)
    json_str = str(json_str.replace('\'', '"'))
    json_str = json_str.replace('(', '[')
    json_str = json_str.replace(')', ']')
    json_str = json_str.replace('None', 'null')
    return json_str


def repr3(obj_, **kwargs):
    kwargs['nl'] = kwargs.pop('nl', kwargs.pop('newlines', True))
    return repr2(obj_, **kwargs)


def repr4(obj_, **kwargs):
    kwargs['nl'] = kwargs.pop('nl', kwargs.pop('newlines', 1))
    # kwargs['precision'] = kwargs.pop('precision', 2)
    return repr2(obj_, **kwargs)


def dict_str(dict_, **dictkw):
    r"""
    Makes a pretty printable / human-readable string representation of a
    dictionary. In most cases this string could be evaled.

    Args:
        dict_ (dict_): a dictionary

    Args:
        dict_ (dict_):  a dictionary
        **dictkw: stritems, strkeys, strvals, nl, newlines, truncate, nobr,
                  nobraces, align, trailing_sep, explicit, itemsep,
                  truncatekw, sorted_, indent_, key_order, precision,
                  with_comma, key_order_metric, maxlen, recursive, use_numpy,
                  with_dtype, force_dtype, packed

    Kwargs:
        sorted_ (None): returns str sorted by a metric (default = None)
        nl (int): prefered alias for newline. can be a coundown variable
            (default = None)
        key_order (None): overrides default ordering (default = None)
        key_order_metric (str): special sorting of items. Accepted values:
                None, 'strlen', 'val'
        precision (int): (default = 8)
        explicit (int): can be a countdown variable. if True, uses
            dict(a=b) syntax instead of {'a': b}
        nobr (bool): removes outer braces (default = False)

    Ignore:
        python -m utool.util_inspect recursive_parse_kwargs:2 --mod utool.util_str --func dict_str --verbinspect

    CommandLine:
        python -m utool.util_str --test-dict_str:1
        python -m utool.util_str --test-dict_str --truncate=False --no-checkwant
        python -m utool.util_str --test-dict_str --truncate=1 --no-checkwant
        python -m utool.util_str --test-dict_str --truncate=2 --no-checkwant

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import dict_str, dict_itemstr_list
        >>> import utool as ut
        >>> dict_ = {'foo': {'spam': 'barbarbarbarbar' * 3, 'eggs': 'jam'},
        >>>          'baz': 'barbarbarbarbar' * 3}
        >>> truncate = ut.get_argval('--truncate', type_=None, default=1)
        >>> result = dict_str(dict_, strvals=True, truncate=truncate,
        >>>                    truncatekw={'maxlen': 20})
        >>> print(result)
        {
            'baz': barbarbarbarbarbarbarbarbarbarbarbarbarbarbar,
            'foo': {
                'eggs': jam,
                's ~~~TRUNCATED~~~ r,
            },
        }

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> import numpy as np
        >>> a, b, c = 'a', 'b', 'c'
        >>> dict_ = {
        >>>     'float': 2.3333,
        >>>     'slice': slice(1, 2, None),
        >>>     'arr': np.eye(3),
        >>>     'list1': [1, 2, 3, 2.3333, a, b, c],
        >>>     'dict1': {2.3333: 2.3333, a: b, c: [a, b]},
        >>>     't': {c: {c: {c: {c : c}}}},
        >>>     'set1': {c, a, b},
        >>>     'set2': ut.oset([c, a, b]),
        >>>     'list2': [
        >>>         {a: {c, a, b}, 1: slice(1, 2, 3)},
        >>>         [1, 2, {c, a, 2.333}, {a: [b], b: {c}, c: 2.333}]
        >>>     ],
        >>> }
        >>> dictkw = dict(stritems=True, itemsep='', precision=2, nl=1,
        >>>               nobr=True, explicit=True)
        >>> result = ut.dict_str(dict_, **dictkw)
        >>> print(result)
        >>> dictkw = dict(stritems=0, precision=2, nl=True, nobr=False,
        >>>                  explicit=0)
        >>> result = ut.dict_str(dict_, **dictkw)
        >>> print(result)
    """
    import utool as ut

    stritems = dictkw.pop('si', dictkw.pop('stritems', False))
    if stritems:
        dictkw['strkeys'] = True
        dictkw['strvals'] = True

    dictkw['strkeys'] = dictkw.pop('sk', dictkw.pop('strkeys', False))
    dictkw['strvals'] = dictkw.pop('sv', dictkw.pop('strvals', False))

    newlines = dictkw.pop('nl', dictkw.pop('newlines', True))
    truncate = dictkw.pop('truncate', False)
    dictkw['nl'] = _rectify_countdown_or_bool(newlines)
    dictkw['truncate'] = _rectify_countdown_or_bool(truncate)

    nobraces = dictkw.pop('nobr', dictkw.pop('nobraces', False))
    align = dictkw.pop('align', False)

    # Doesn't actually put in trailing comma if on same line
    trailing_sep = dictkw.get('trailing_sep', True)
    explicit = dictkw.get('explicit', False)
    with_comma  = True
    itemsep = dictkw.get('itemsep', ' ')

    if len(dict_) == 0:
        return 'dict()' if explicit else '{}'

    itemstr_list = dict_itemstr_list(dict_, **dictkw)

    do_truncate = truncate is not False and (truncate is True or truncate == 0)
    if do_truncate:
        truncatekw = dictkw.get('truncatekw', {})
        itemstr_list = [truncate_str(item, **truncatekw) for item in itemstr_list]

    if nobraces:
        lbr, rbr = '', ''
    elif explicit:
        lbr, rbr = 'dict(', ')'
    else:
        lbr, rbr = '{', '}'

    if newlines:
        sep = ',\n' if with_comma else '\n'
        if nobraces:
            retstr =  sep.join(itemstr_list)
            if trailing_sep:
                retstr += ','
        else:
            parts = [ut.indent(itemstr, '    ') for itemstr in itemstr_list]
            body_str = sep.join(parts)
            if trailing_sep:
                body_str += ','
            retstr =  (lbr + '\n' + body_str + '\n' + rbr)
            if align:
                retstr = ut.align(retstr, ':')
    else:
        sep = ',' + itemsep if with_comma else itemsep
        # hack away last trailing comma
        sequence_str = sep.join(itemstr_list)
        retstr = lbr +  sequence_str + rbr
    # Is there a way to make truncate for dict_str compatible with list_str?
    return retstr


def dict_itemstr_list(dict_, **dictkw):
    r"""
    Returns:
        list: a list of human-readable dictionary items

    Args:
        explicit : if True uses dict(key=val,...) format instead of {key:val,...}
    """
    import utool as ut

    explicit = dictkw.get('explicit', False)
    dictkw['explicit'] = _rectify_countdown_or_bool(explicit)

    dosort = dictkw.get('sorted_', None)
    if dosort is None:
        dosort = True

    if dosort and not isinstance(dict_, collections.OrderedDict):
        key_order = dictkw.get('key_order', None)
        def iteritems(d):
            if key_order is None:
                # specify order explicilty
                try:
                    return iter(sorted(six.iteritems(d)))
                except TypeError:
                    # catches case where keys are of different types
                    return six.iteritems(d)
            else:
                # Enforce specific key order
                # TODO: depricate and just use ordered dicts
                unordered_keys = list(d.keys())
                other_keys = sorted(list(set(unordered_keys) - set(key_order)))
                keys = key_order + other_keys
                return ((key, d[key]) for key in keys)
    else:
        iteritems = six.iteritems

    _valstr = _make_valstr(**dictkw)

    precision = dictkw.get('precision', None)
    kvsep = dictkw.get('kvsep', ': ')
    if explicit:
        kvsep = '='

    def make_item_str(key, val):
        if explicit or dictkw.get('strkeys', False):
            key_str = six.text_type(key)
        else:
            key_str = repr2(key, precision=precision)

        prefix = key_str + kvsep
        val_str = _valstr(val)

        # FIXME: get indentation right
        if util_type.HAVE_NUMPY and isinstance(val, np.ndarray):
            item_str = hz_str(prefix, val_str)
        else:
            # padded_indent = ' ' * min(len(indent_), len(prefix))
            # val_str = val_str.replace('\n', '\n' + padded_indent)
            item_str = prefix + val_str
        return item_str

    itemstr_list = [make_item_str(key, val)
                    for (key, val) in iteritems(dict_)]

    reverse = False
    key_order_metric = dictkw.get('key_order_metric', None)
    if key_order_metric is not None:
        if key_order_metric.startswith('-'):
            key_order_metric = key_order_metric[1:]
            reverse = True

    if key_order_metric == 'strlen':
        metric_list = [len(itemstr) for itemstr in itemstr_list]
        itemstr_list = ut.sortedby(itemstr_list, metric_list, reverse=reverse)
    elif key_order_metric == 'val':
        metric_list = [val for (key, val) in iteritems(dict_)]
        itemstr_list = ut.sortedby(itemstr_list, metric_list, reverse=reverse)

    maxlen = dictkw.get('maxlen', None)
    if maxlen is not None and len(itemstr_list) > maxlen:
        itemstr_list = itemstr_list[0:maxlen]
    return itemstr_list


def list_str(list_, **listkw):
    r"""
    Makes a pretty list string

    Args:
        list_ (list): input list
        **listkw: nl, newlines, packed, truncate, nobr, nobraces, itemsep,
                  trailing_sep, truncatekw, strvals, recursive,
                  indent_, precision, use_numpy, with_dtype, force_dtype,
                  stritems, strkeys, align, explicit, sorted_, key_order,
                  key_order_metric, maxlen

    Returns:
        str: retstr

    CommandLine:
        python -m utool.util_str --test-list_str
        python -m utool.util_str --exec-list_str --truncate=True
        python -m utool.util_str --exec-list_str --truncate=0


    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> import utool as ut
        >>> list_ = [[(('--verbose-qt', '--verbqt'), 1, False, ''),
        >>>           (('--verbose-qt', '--verbqt'), 1, False, ''),
        >>>           (('--verbose-qt', '--verbqt'), 1, False, ''),
        >>>           (('--verbose-qt', '--verbqt'), 1, False, '')],
        >>>          [(['--nodyn'], 1, False, ''), (['--nodyn'], 1, False, '')]]
        >>> listkw = {'nl':  2}
        >>> result = list_str(list_, **listkw)
        >>> print(result)
        [
            [
                (('--verbose-qt', '--verbqt'), 1, False, ''),
                (('--verbose-qt', '--verbqt'), 1, False, ''),
                (('--verbose-qt', '--verbqt'), 1, False, ''),
                (('--verbose-qt', '--verbqt'), 1, False, ''),
            ],
            [
                (['--nodyn'], 1, False, ''),
                (['--nodyn'], 1, False, ''),
            ],
        ]
    """
    import utool as ut

    newlines = listkw.pop('nl', listkw.pop('newlines', 1))
    packed = listkw.pop('packed', False)
    truncate = listkw.pop('truncate', False)
    listkw['nl'] = _rectify_countdown_or_bool(newlines)
    listkw['truncate'] = _rectify_countdown_or_bool(truncate)
    listkw['packed'] = _rectify_countdown_or_bool(packed)

    nobraces = listkw.pop('nobr', listkw.pop('nobraces', False))

    itemsep = listkw.get('itemsep', ' ')
    # Doesn't actually put in trailing comma if on same line
    trailing_sep = listkw.get('trailing_sep', True)
    with_comma = True

    itemstr_list = get_itemstr_list(list_, **listkw)
    is_tuple = isinstance(list_, tuple)
    is_set = isinstance(list_, (set, frozenset, ut.oset))
    is_onetup = isinstance(list_, (tuple)) and len(list_) <= 1
    if nobraces:
        lbr, rbr = '', ''
    elif is_tuple:
        lbr, rbr  = '(', ')'
    elif is_set:
        lbr, rbr  = '{', '}'
    else:
        lbr, rbr  = '[', ']'

    if len(itemstr_list) == 0:
        newlines = False

    if newlines is not False and (newlines is True or newlines > 0):
        sep = ',\n' if with_comma else '\n'
        if nobraces:
            body_str = sep.join(itemstr_list)
            if trailing_sep:
                body_str += ','
            retstr = body_str
        else:
            if packed:
                # DEPRICATE?
                joinstr = sep + itemsep * len(lbr)
                body_str = joinstr.join([itemstr for itemstr in itemstr_list])
                if trailing_sep:
                    body_str += ','
                braced_body_str = (lbr + '' + body_str + '' + rbr)
            else:
                body_str = sep.join([
                    ut.indent(itemstr) for itemstr in itemstr_list])
                if trailing_sep:
                    body_str += ','
                braced_body_str = (lbr + '\n' + body_str + '\n' + rbr)
            retstr = braced_body_str
    else:
        sep = ',' + itemsep if with_comma else itemsep
        body_str = sep.join(itemstr_list)
        if is_onetup:
            body_str += ','
        retstr  = (lbr + body_str +  rbr)

    # TODO: rectify with dict_truncate
    do_truncate = truncate is not False and (truncate is True or truncate == 0)
    if do_truncate:
        truncatekw = listkw.get('truncatekw', {})
        retstr = truncate_str(retstr, **truncatekw)
    return retstr


def _make_valstr(**kwargs):
    import utool as ut

    stritems = kwargs.get('si', kwargs.get('stritems', False))
    strvals = stritems or kwargs.get('sv', kwargs.get('strvals', False))
    valfunc = six.text_type if strvals else reprfunc

    if not kwargs.get('recursive', True):
        return valfunc

    indent_ = kwargs.pop('indent_', '')
    new_indent = indent_ + '    '

    def recursive_valfunc(val):
        precision = kwargs.get('precision', None)
        if isinstance(val, dict):
            return dict_str(val, indent_=new_indent, **kwargs)
        if isinstance(val, (list, tuple, set, frozenset, ut.oset)):
            return list_str(val, indent_=new_indent, **kwargs)
        if util_type.HAVE_PANDAS and isinstance(val, pd.Index):
            return list_str(val.tolist(), indent_=new_indent, **kwargs)
        if util_type.HAVE_NUMPY and isinstance(val, np.ndarray):
            if kwargs.get('use_numpy', True):
                with_dtype = kwargs.get('with_dtype', False)
                strvals = kwargs.get('strvals', False)
                precision = kwargs.get('precision', None)
                max_line_width = kwargs.get('max_line_width', None)
                linewidth = kwargs.get('linewidth', max_line_width)
                threshold = kwargs.get('threshold', None)

                return numpy_str(val, strvals=strvals, precision=precision,
                                 threshold=threshold, linewidth=linewidth,
                                 with_dtype=with_dtype)
            else:
                return list_str(val, **kwargs)
        if precision is not None and (isinstance(val, (float)) or
                                      util_type.is_float(val)):
            return scalar_str(val, precision)
        elif isinstance(val, slice):
            if kwargs.get('itemsep', ' ') == '':
                return 'slice(%r,%r,%r)' % (val.start, val.stop, val.step)
            else:
                return valfunc(val)
        else:
            # base case
            return valfunc(val)
    return recursive_valfunc


def _peek_isinstance(items, types):
    return len(items) > 0 and isinstance(items[0], types)


def get_itemstr_list(list_, **listkw):
    """
    TODO: have this replace dict_itemstr list or at least most functionality in
    it. have it make two itemstr lists over keys and values and then combine
    them.
    """
    import utool as ut
    _valstr = _make_valstr(**listkw)

    def make_item_str(item):
        item_str = _valstr(item)
        return item_str

    items = list(list_)
    itemstr_list = [make_item_str(item) for item in items]

    dosort = listkw.get('sorted_', None)
    if dosort is None:
        # Force orderings on sets.
        dosort = isinstance(list_, (set, frozenset))
    if dosort:
        # First try to sort items by their normal values
        # If that doesnt work, then sort by their string values
        try:
            # Set ordering is not unique. Sort by strings values instead.
            if _peek_isinstance(items, (set, frozenset)):
                raise Exception
            sortx = ut.argsort2(items)
        except Exception:
            sortx = ut.argsort2(itemstr_list)
        itemstr_list = ut.take(itemstr_list, sortx)
    return itemstr_list


def utf8_len(str_):
    """
    returns num printed characters in utf8

    Returns:
        http://stackoverflow.com/questions/2247205/python-returning-the-wrong-length-of-string-when-using-special-characters
    """
    import unicodedata
    return len(unicodedata.normalize('NFC', ensure_unicode(str_)))


def horiz_string(*args, **kwargs):
    """
    Horizontally concatenates strings reprs preserving indentation

    Concats a list of objects ensuring that the next item in the list
    is all the way to the right of any previous items.

    Args:
        *args: list of strings to concat
        **kwargs: precision, sep

    CommandLine:
        python -m utool.util_str --test-horiz_string

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
    import unicodedata

    precision = kwargs.get('precision', None)
    sep = kwargs.get('sep', '')

    if len(args) == 1 and not isinstance(args[0], six.string_types):
        val_list = args[0]
    else:
        val_list = args

    val_list = [unicodedata.normalize('NFC', ensure_unicode(val))
                for val in val_list]
    all_lines = []
    hpos = 0
    # for each value in the list or args
    for sx in range(len(val_list)):
        # Ensure value is a string
        val = val_list[sx]
        str_ = None
        if precision is not None:
            # Hack in numpy precision
            if util_type.HAVE_NUMPY:
                try:
                    if isinstance(val, np.ndarray):
                        str_ = np.array_str(val, precision=precision,
                                            suppress_small=True)
                except ImportError:
                    pass
        if str_ is None:
            str_ = six.text_type(val_list[sx])
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
            all_lines[lx] += ' ' * hpos_diff + sep
    all_lines = [line.rstrip(' ') for line in all_lines]
    ret = '\n'.join(all_lines)
    return ret

# Alias
hz_str = horiz_string


def str_between(str_, startstr, endstr):
    r"""
    gets substring between two sentianl strings

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> import utool as ut
        >>> str_ = '\n        INSERT INTO vsone(\n'
        >>> startstr = 'INSERT'
        >>> endstr = '('
        >>> result = str_between(str_, startstr, endstr)
        >>> print(result)
    """
    if startstr is None:
        startpos = 0
    else:
        startpos = str_.find(startstr) + len(startstr)
    if endstr is None:
        endpos = None
    else:
        endpos = str_.find(endstr)
        if endpos == -1:
            endpos = None
    newstr = str_[startpos:endpos]
    return newstr


def get_callable_name(func):
    """ Works on must functionlike objects including str, which has no func_name

    Args:
        func (function):

    Returns:
        str:

    CommandLine:
        python -m utool.util_str --exec-get_callable_name

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> func = len
        >>> result = get_callable_name(func)
        >>> print(result)
        len
    """
    try:
        return meta_util_six.get_funcname(func)
    except AttributeError:
        if isinstance(func, type):
            return repr(func).replace('<type \'', '').replace('\'>', '')
        elif hasattr(func, '__name__'):
            return func.__name__
        else:
            raise NotImplementedError(('cannot get func_name of func=%r'
                                        'type(func)=%r') % (func, type(func)))


def align(text, character='=', replchar=None, pos=0):
    r"""
    Left justifies text on the left side of character

    align

    Args:
        text (str): text to align
        character (str): character to align at
        replchar (str): replacement character (default=None)

    Returns:
        str: new_text

    CommandLine:
        python -m utool.util_str --test-align:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> character = '='
        >>> text = 'a = b=\none = two\nthree = fish\n'
        >>> print(text)
        >>> result = (align(text, '='))
        >>> print(result)
        a     = b=
        one   = two
        three = fish
    """
    line_list = text.splitlines()
    new_lines = align_lines(line_list, character, replchar, pos=pos)
    new_text = '\n'.join(new_lines)
    return new_text


def align_lines(line_list, character='=', replchar=None, pos=0):
    r"""
    Left justifies text on the left side of character

    align_lines

    TODO:
        clean up and move to ubelt?

    Args:
        line_list (list of strs):
        character (str):
        pos (int or list or None): does one alignment for all chars beyond this
            column position. If pos is None, then all chars are aligned.

    Returns:
        list: new_lines

    CommandLine:
        python -m utool.util_str --test-align_lines:0
        python -m utool.util_str --test-align_lines:1
        python -m utool.util_str --test-align_lines:2
        python -m utool.util_str --test-align_lines:3

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> line_list = 'a = b\none = two\nthree = fish'.split('\n')
        >>> character = '='
        >>> new_lines = align_lines(line_list, character)
        >>> result = ('\n'.join(new_lines))
        >>> print(result)
        a     = b
        one   = two
        three = fish

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> line_list = 'foofish:\n    a = b\n    one    = two\n    three    = fish'.split('\n')
        >>> character = '='
        >>> new_lines = align_lines(line_list, character)
        >>> result = ('\n'.join(new_lines))
        >>> print(result)
        foofish:
            a        = b
            one      = two
            three    = fish

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> import utool as ut
        >>> character = ':'
        >>> text = ut.codeblock('''
            {'max': '1970/01/01 02:30:13',
             'mean': '1970/01/01 01:10:15',
             'min': '1970/01/01 00:01:41',
             'range': '2:28:32',
             'std': '1:13:57',}''').split('\n')
        >>> new_lines = align_lines(text, ':', ' :')
        >>> result = '\n'.join(new_lines)
        >>> print(result)
        {'max'   : '1970/01/01 02:30:13',
         'mean'  : '1970/01/01 01:10:15',
         'min'   : '1970/01/01 00:01:41',
         'range' : '2:28:32',
         'std'   : '1:13:57',}

    Example3:
        >>> # ENABLE_DOCEST
        >>> from utool.util_str import *  # NOQA
        >>> line_list = 'foofish:\n a = b = c\n one = two = three\nthree=4= fish'.split('\n')
        >>> character = '='
        >>> # align the second occurence of a character
        >>> new_lines = align_lines(line_list, character, pos=None)
        >>> print(('\n'.join(line_list)))
        >>> result = ('\n'.join(new_lines))
        >>> print(result)
        foofish:
         a   = b   = c
         one = two = three
        three=4    = fish

    Ignore:
        # use this as test case
        \begin{tabular}{lrrll}
        \toprule
        {} &  Names &  Annots & Annots size & Training Edges \\
        \midrule
        training &    390 &    1164 &   2.98\pm2.83 &           9360 \\
        testing  &    363 &    1119 &   3.08\pm2.82 &              - \\
        \bottomrule
        \end{tabular}



    """

    # FIXME: continue to fix ansi
    if pos is None:
        # Align all occurences
        num_pos = max([line.count(character) for line in line_list])
        pos = list(range(num_pos))

    # Allow multiple alignments
    if isinstance(pos, list):
        pos_list = pos
        # recursive calls
        new_lines = line_list
        for pos in pos_list:
            new_lines = align_lines(new_lines, character=character,
                                    replchar=replchar, pos=pos)
        return new_lines

    # base case
    if replchar is None:
        replchar = character

    # the pos-th character to align
    lpos = pos
    rpos = lpos + 1

    tup_list = [line.split(character) for line in line_list]

    handle_ansi = True
    if handle_ansi:
        # Remove ansi from length calculation
        # References: http://stackoverflow.com/questions/14693701remove-ansi
        ansi_escape = re.compile(r'\x1b[^m]*m')

    # Find how much padding is needed
    maxlen = 0
    for tup in tup_list:
        if len(tup) >= rpos + 1:
            if handle_ansi:
                tup = [ansi_escape.sub('', x) for x in tup]
            left_lenlist = list(map(len, tup[0:rpos]))
            left_len = sum(left_lenlist) + lpos * len(replchar)
            maxlen = max(maxlen, left_len)

    # Pad each line to align the pos-th occurence of the chosen character
    new_lines = []
    for tup in tup_list:
        if len(tup) >= rpos + 1:
            lhs = character.join(tup[0:rpos])
            rhs = character.join(tup[rpos:])
            # pad the new line with requested justification
            newline = lhs.ljust(maxlen) + replchar + rhs
            new_lines.append(newline)
        else:
            new_lines.append(replchar.join(tup))
    return new_lines


def strip_ansi(text):
    """
    Removes all ansi directives from the string
    Helper to remove ansi from length calculation
    References: http://stackoverflow.com/questions/14693701remove-ansi
    """
    ansi_escape = re.compile(r'\x1b[^m]*m')
    return ansi_escape.sub('', text)


# FIXME: HASHLEN is a global var in util_hash
def long_fname_format(fmt_str, fmt_dict, hashable_keys=[], max_len=64,
                      hashlen=16, ABS_MAX_LEN=255, hack27=False):
    r"""
    DEPRICATE

    Formats a string and hashes certain parts if the resulting string becomes
    too long. Used for making filenames fit onto disk.

    Args:
        fmt_str (str): format of fname
        fmt_dict (str): dict to format fname with
        hashable_keys (list): list of dict keys you are willing to have hashed
        max_len (int): tries to fit fname into this length
        ABS_MAX_LEN (int): throws AssertionError if fname over this length

    CommandLine:
        python -m utool.util_str --exec-long_fname_format

    Example:
        >>> # ENABLE_DOCTET
        >>> import utool as ut
        >>> fmt_str = 'qaid={qaid}_res_{cfgstr}_quuid={quuid}'
        >>> quuid_str = 'blahblahblahblahblahblah'
        >>> cfgstr = 'big_long_string__________________________________'
        >>> qaid = 5
        >>> fmt_dict = dict(cfgstr=cfgstr, qaid=qaid, quuid=quuid_str)
        >>> hashable_keys = ['cfgstr', 'quuid']
        >>> max_len = 64
        >>> hashlen = 8
        >>> fname0 = ut.long_fname_format(fmt_str, fmt_dict, max_len=None)
        >>> fname1 = ut.long_fname_format(fmt_str, fmt_dict, hashable_keys,
        >>>                                  max_len=64, hashlen=8)
        >>> fname2 = ut.long_fname_format(fmt_str, fmt_dict, hashable_keys, max_len=42,
        >>>                         hashlen=8)
        >>> result = fname0 + '\n' + fname1 + '\n' + fname2
        >>> print(result)
        qaid=5_res_big_long_string___________________________________quuid=blahblahblahblahblahblah
        qaid=5_res_racfntgq_quuid=blahblahblahblahblahblah
        qaid=5_res_racfntgq_quuid=yvuaffrp
    """
    from utool import util_hash
    fname = fmt_str.format(**fmt_dict)
    if max_len is None:
        return fname
    if len(fname) > max_len:
        # Copy because we will overwrite fmt_dict values with hashed values
        fmt_dict_ = fmt_dict.copy()
        for key in hashable_keys:
            if hack27:
                fmt_dict_[key] = util_hash.hashstr27(fmt_dict_[key], hashlen=hashlen)
            else:
                fmt_dict_[key] = util_hash.hashstr(fmt_dict_[key], hashlen=hashlen)
            fname = fmt_str.format(**fmt_dict_)
            if len(fname) <= max_len:
                break
        if len(fname) > max_len:
            diff = len(fname) - max_len
            msg = ('[util_str] Warning: Too big by %d chars. Exausted all options'
                   'to make fname fit into size. ')  % (diff,)
            print(msg)
            print('* len(fname) = %r' % len(fname))
            print('* fname = %r' % fname)
            if ABS_MAX_LEN is not None and len(fname) > ABS_MAX_LEN:
                raise AssertionError(msg)
    return fname


def multi_replace(str_, search_list, repl_list):
    r"""
    Performs multiple replace functions foreach item in search_list and
    repl_list.

    Args:
        str_ (str): string to search
        search_list (list): list of search strings
        repl_list (list or str): one or multiple replace strings

    Returns:
        str: str_

    CommandLine:
        python -m utool.util_str --exec-multi_replace

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> str_ = 'foo. bar: baz; spam-eggs --- eggs+spam'
        >>> search_list = ['.', ':', '---']
        >>> repl_list = '@'
        >>> str_ = multi_replace(str_, search_list, repl_list)
        >>> result = ('str_ = %s' % (str(str_),))
        >>> print(result)
        str_ = foo@ bar@ baz; spam-eggs @ eggs+spam
    """
    if isinstance(repl_list, six.string_types):
        repl_list_ = [repl_list] * len(search_list)
    else:
        repl_list_ = repl_list
    newstr = str_
    assert len(search_list) == len(repl_list_), 'bad lens'
    for search, repl in zip(search_list, repl_list_):
        newstr = newstr.replace(search, repl)
    return newstr


def pluralize(wordtext, num=2, plural_suffix='s'):
    r"""
    Heuristically changes a word to its plural form if `num` is not 1

    Args:
        wordtext (str): word in singular form
        num (int): a length of an associated list if applicable (default = 2)
        plural_suffix (str): heurstic plural form (default = 's')

    Returns:
        str: pluralized form. Can handle some genitive cases

    CommandLine:
        python -m utool.util_str pluralize

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> wordtext = 'foo'
        >>> result = pluralize(wordtext)
        >>> print(result)
        foos
    """
    if num == 1:
        return wordtext
    else:
        if wordtext.endswith('\'s'):
            return wordtext[:-2] + 's\''
        else:
            return wordtext + plural_suffix
    return (wordtext + plural_suffix) if num != 1 else wordtext


def quantstr(typestr, num, plural_suffix='s'):
    r"""
    Heuristically generates an english phrase relating to the quantity of
    something.  This is useful for writing user messages.

    Args:
        typestr (str): singular form of the word
        num (int): quanity of the type
        plural_suffix (str): heurstic plural form (default = 's')

    Returns:
        str: quantity phrase

    CommandLine:
        python -m utool.util_str quantity_str

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> items = [1, 2, 3]
        >>> result = 'The list contains ' + (quantstr('item', len(items)))
        >>> items = [1]
        >>> result += '\nThe list contains ' + (quantstr('item', len(items)))
        >>> items = []
        >>> result += '\nThe list contains ' + (quantstr('item', len(items)))
        >>> print(result)
        The list contains 3 items
        The list contains 1 item
        The list contains 0 items
    """
    return six.text_type(num) + ' ' + pluralize(typestr, num, plural_suffix)


def msgblock(key, text, side='|'):
    """ puts text inside a visual ascii block """
    blocked_text = ''.join(
        [' + --- ', key, ' ---\n'] +
        [' ' + side + ' ' + line + '\n' for line in text.split('\n')] +
        [' L ___ ', key, ' ___\n']
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


def get_textdiff(text1, text2, num_context_lines=0, ignore_whitespace=False):
    r"""
    Uses difflib to return a difference string between two similar texts

    Args:
        text1 (str):
        text2 (str):

    Returns:
        str: formatted difference text message

    SeeAlso:
        ut.color_diff_text

    References:
        http://www.java2s.com/Code/Python/Utility/IntelligentdiffbetweentextfilesTimPeters.htm

    CommandLine:
        python -m utool.util_str --test-get_textdiff:1
        python -m utool.util_str --test-get_textdiff:0

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> # build test data
        >>> text1 = 'one\ntwo\nthree'
        >>> text2 = 'one\ntwo\nfive'
        >>> # execute function
        >>> result = get_textdiff(text1, text2)
        >>> # verify results
        >>> print(result)
        - three
        + five

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> # build test data
        >>> text1 = 'one\ntwo\nthree\n3.1\n3.14\n3.1415\npi\n3.4\n3.5\n4'
        >>> text2 = 'one\ntwo\nfive\n3.1\n3.14\n3.1415\npi\n3.4\n4'
        >>> # execute function
        >>> num_context_lines = 1
        >>> result = get_textdiff(text1, text2, num_context_lines)
        >>> # verify results
        >>> print(result)
    """
    import difflib
    text1 = ensure_unicode(text1)
    text2 = ensure_unicode(text2)
    text1_lines = text1.splitlines()
    text2_lines = text2.splitlines()
    if ignore_whitespace:
        text1_lines = [t.rstrip() for t in text1_lines]
        text2_lines = [t.rstrip() for t in text2_lines]
        ndiff_kw = dict(linejunk=difflib.IS_LINE_JUNK,
                        charjunk=difflib.IS_CHARACTER_JUNK)
    else:
        ndiff_kw = {}
    all_diff_lines = list(difflib.ndiff(text1_lines, text2_lines, **ndiff_kw))

    if num_context_lines is None:
        diff_lines = all_diff_lines
    else:
        from utool import util_list
        # boolean for every line if it is marked or not
        ismarked_list = [len(line) > 0 and line[0] in '+-?'
                         for line in all_diff_lines]
        # flag lines that are within num_context_lines away from a diff line
        isvalid_list = ismarked_list[:]
        for i in range(1, num_context_lines + 1):
            isvalid_list[:-i] = util_list.or_lists(isvalid_list[:-i],
                                                   ismarked_list[i:])
            isvalid_list[i:]  = util_list.or_lists(isvalid_list[i:],
                                                   ismarked_list[:-i])
        USE_BREAK_LINE = True
        if USE_BREAK_LINE:
            # insert a visual break when there is a break in context
            diff_lines = []
            prev = False
            visual_break = '\n <... FILTERED CONTEXT ...> \n'
            #print(isvalid_list)
            for line, valid in zip(all_diff_lines, isvalid_list):
                if valid:
                    diff_lines.append(line)
                elif prev:
                    if False:
                        diff_lines.append(visual_break)
                prev = valid
        else:
            diff_lines = util_list.compress(all_diff_lines, isvalid_list)
    return '\n'.join(diff_lines)


difftext = get_textdiff


def conj_phrase(list_, cond='or'):
    """
    Joins a list of words using English conjunction rules

    Args:
        list_ (list):  of strings
        cond (str): a conjunction (or, and, but)

    Returns:
        str: the joined cconjunction phrase

    References:
        http://en.wikipedia.org/wiki/Conjunction_(grammar)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> list_ = ['a', 'b', 'c']
        >>> result = conj_phrase(list_, 'or')
        >>> print(result)
        a, b, or c

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> list_ = ['a', 'b']
        >>> result = conj_phrase(list_, 'and')
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


def doctest_code_line(line_str, varname=None, verbose=True):
    varprefix = varname + ' = ' if varname is not None else ''
    prefix1 = '>>> ' + varprefix
    prefix2 = '\n... ' + (' ' * len(varprefix))
    doctest_line_str = prefix1 + prefix2.join(line_str.split('\n'))
    if verbose:
        print(doctest_line_str)
    return doctest_line_str


def doctest_repr(var, varname=None, precision=2, verbose=True):
    import utool as ut
    varname_ = ut.get_varname_from_stack(var, N=1) if varname is None else varname
    if util_type.HAVE_NUMPY and isinstance(var, np.ndarray):
        line_str = ut.numpy_str(var, precision=precision, suppress_small=True)
    else:
        line_str = repr(var)
    doctest_line_str = doctest_code_line(line_str, varname=varname_, verbose=verbose)
    return doctest_line_str


def format_text_as_docstr(text):
    r"""

    CommandLine:
        python  ~/local/vim/rc/pyvim_funcs.py  --test-format_text_as_docstr

    Example:
        >>> # DISABLE_DOCTEST
        >>> from pyvim_funcs import *  # NOQA
        >>> text = testdata_text()
        >>> formated_text = format_text_as_docstr(text)
        >>> result = ('formated_text = \n%s' % (str(formated_text),))
        >>> print(result)
    """
    import utool as ut
    import re
    min_indent = ut.get_minimum_indentation(text)
    indent_ =  ' ' * min_indent
    formated_text = re.sub('^' + indent_, '' + indent_ + '>>> ', text,
                           flags=re.MULTILINE)
    formated_text = re.sub('^$', '' + indent_ + '>>> #', formated_text,
                           flags=re.MULTILINE)
    return formated_text


def unformat_text_as_docstr(formated_text):
    r"""
    CommandLine:
        python  ~/local/vim/rc/pyvim_funcs.py  --test-unformat_text_as_docstr

    Example:
        >>> # DISABLE_DOCTEST
        >>> from pyvim_funcs import *  # NOQA
        >>> text = testdata_text()
        >>> formated_text = format_text_as_docstr(text)
        >>> unformated_text = unformat_text_as_docstr(formated_text)
        >>> result = ('unformated_text = \n%s' % (str(unformated_text),))
        >>> print(result)
    """
    import utool as ut
    import re
    min_indent = ut.get_minimum_indentation(formated_text)
    indent_ =  ' ' * min_indent
    unformated_text = re.sub('^' + indent_ + '>>> ', '' + indent_,
                             formated_text, flags=re.MULTILINE)
    return unformated_text


def lorium_ipsum():
    """ Standard testing string """
    ipsum_str = '''
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed
    do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad
    minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex
    ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate
    velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat
    cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id
    est laborum.
    '''
    return ipsum_str


def bubbletext(text, font='cybermedium'):
    r"""
    Uses pyfiglet to create bubble text.

    Args:
        font (str): default=cybermedium, other fonts include: cybersmall and
            cyberlarge.

    References:
        http://www.figlet.org/

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> bubble_text = ut.bubbletext('TESTING BUBBLE TEXT', font='cybermedium')
        >>> print(bubble_text)
    """
    import utool as ut
    pyfiglet = ut.tryimport('pyfiglet', 'git+https://github.com/pwaller/pyfiglet')
    if pyfiglet is None:
        return text
    else:
        bubble_text = pyfiglet.figlet_format(text, font=font)
        return bubble_text


def closet_words(query, options, num=1, subset=False):
    import utool as ut
    ranked_list = []
    if subset:
        query_ = query.lower()
        superset = [opt for opt in options if query_ in opt.lower()]
        ranked_list = superset[0:num]
        num -= len(ranked_list)
    dist_list = ut.edit_distance(query, options)
    ranked_list = ranked_list + ut.sortedby(options, dist_list)[0:num]
    ranked_list = ut.unique(ranked_list)
    return ranked_list


def to_title_caps(underscore_case):
    r"""
    Args:
        underscore_case (?):

    Returns:
        str: title_str

    CommandLine:
        python -m utool.util_str --exec-to_title_caps

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> underscore_case = 'the_foo_bar_func'
        >>> title_str = to_title_caps(underscore_case)
        >>> result = ('title_str = %s' % (str(title_str),))
        >>> print(result)
        title_str = The Foo Bar Func
    """
    words = underscore_case.split('_')
    words2 = [
        word[0].upper() + word[1:]
        for count, word in enumerate(words)
    ]
    title_str = ' '.join(words2)
    return title_str


def to_underscore_case(camelcase_str):
    r"""
    References:
        http://stackoverflow.com/questions/1175208/convert-camelcase

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> camelcase_str = 'UnderscoreFuncname'
        >>> camel_case_str = to_underscore_case(camelcase_str)
        >>> result = ('underscore_str = %s' % (str(camel_case_str),))
        >>> print(result)
        underscore_str = underscore_funcname
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camelcase_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def to_camel_case(underscore_case, mixed=False):
    r"""
    Args:
        underscore_case (?):

    Returns:
        str: camel_case_str

    CommandLine:
        python -m utool.util_str --exec-to_camel_case

    References:
        https://en.wikipedia.org/wiki/CamelCase

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> underscore_case = 'underscore_funcname'
        >>> camel_case_str = to_camel_case(underscore_case)
        >>> result = ('camel_case_str = %s' % (str(camel_case_str),))
        >>> print(result)
        camel_case_str = UnderscoreFuncname
    """
    thresh = 0 if mixed else -1
    words = underscore_case.split('_')
    words2 = [
        word[0].upper() + word[1:]
        if count > thresh else
        word
        for count, word in enumerate(words)
    ]
    camel_case_str = ''.join(words2)
    return camel_case_str


def is_url(str_):
    """ heuristic check if str is url formatted """
    return any([
        str_.startswith('http://'),
        str_.startswith('https://'),
        str_.startswith('www.'),
        '.org/' in str_,
        '.com/' in str_,
    ])


def autoformat_pep8(sourcecode, **kwargs):
    r"""
    Args:
        code (str):

    CommandLine:
        python -m utool.util_str --exec-autoformat_pep8

    Kwargs:
        'aggressive': 0,
        'diff': False,
        'exclude': [],
        'experimental': False,
        'files': [u''],
        'global_config': ~/.config/pep8,
        'ignore': set([u'E24']),
        'ignore_local_config': False,
        'in_place': False,
        'indent_size': 4,
        'jobs': 1,
        'line_range': None,
        'list_fixes': False,
        'max_line_length': 79,
        'pep8_passes': -1,
        'recursive': False,
        'select': ,
        'verbose': 0,

    Ignore:
        autopep8 --recursive --in-place --ignore E126,E127,E201,E202,E203,E221,E222,E241,E265,E271,E272,E301,E501,W602,E266,N801,N802,N803,N805,N806,N811,N813 .
    """
    import autopep8
    default_ignore = {
        'E126',  # continuation line hanging-indent
        'E127',  # continuation line over-indented for visual indent
        'E201',  # whitespace after '('
        'E202',  # whitespace before ']'
        'E203',  # whitespace before ', '
        'E221',  # multiple spaces before operator
        'E222',  # multiple spaces after operator
        'E241',  # multiple spaces after ,
        'E265',  # block comment should start with "# "
        'E271',  # multiple spaces after keyword
        'E272',  # multiple spaces before keyword
        'E301',  # expected 1 blank line, found 0
        'E501',  # line length > 79
        'W602',  # Old reraise syntax
        'E266',  # too many leading '#' for block comment
        'N801',  # function name should be lowercase [N806]
        'N802',  # function name should be lowercase [N806]
        'N803',  # argument should be lowercase [N806]
        'N805',  # first argument of a method should be named 'self'
        'N806',  # variable in function should be lowercase [N806]
        'N811',  # constant name imported as non constant
        'N813',  # camel case
    }
    # My defaults
    kwargs['ignore'] = kwargs.get('ignore', default_ignore)
    kwargs['aggressive'] = kwargs.get('aggressive', 1)
    pep8_options = autopep8._get_options(kwargs, False)
    new_source = autopep8.fix_code(sourcecode, pep8_options)
    return new_source


autopep8_format = autoformat_pep8


def filtered_infostr(flags, lbl, reason=None):
    total = len(flags)
    removed = total - sum(flags)
    reasonstr = '' if reason is None else ' based on %s' % (reason,)
    percent = 100 * removed / total
    str_ = ('Removing %d / %d (%.2f%%) %s%s' % (removed, total, percent, lbl, reasonstr))
    return str_


def chr_range(*args, **kw):
    r"""
    Like range but returns characters

    Args:
        start (None): (default = None)
        stop (None): (default = None)
        step (None): (default = None)

    Kwargs:
        base (str): charater to start with (default='a')

    Returns:
        list: list of characters

    CommandLine:
        python -m utool.util_str --exec-chr_range

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> import utool as ut
        >>> args = (5,)
        >>> result = ut.repr2(chr_range(2, base='a'))
        >>> print(chr_range(0, 5))
        >>> print(chr_range(0, 50))
        >>> print(chr_range(0, 5, 2))
        >>> print(result)
        ['a', 'b']
    """
    if len(args) == 1:
        stop, = args
        start, step = 0, 1
    elif len(args) == 2:
        start, stop = args
        step = 1
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise ValueError('incorrect args')

    chr_ = six.unichr

    base = ord(kw.get('base', 'a'))
    if isinstance(start, int):
        start = base + start
    if isinstance(stop, int):
        stop = base + stop

    if isinstance(start, six.string_types):
        start = ord(start)
    if isinstance(stop, six.string_types):
        stop = ord(stop)
    if step is None:
        step = 1
    list_ = list(map(six.text_type, map(chr_, range(start, stop, step))))
    return list_


def color_diff_text(text):
    return highlight_text(text, lexer_name='diff')


def highlight_code(text, lexer_name='python'):
    return highlight_text(text, lexer_name)


def highlight_text(text, lexer_name='python', **kwargs):
    r"""
    SeeAlso:
        color_text
    """
    # Resolve extensions to languages
    lexer_name = {
        'py': 'python',
        'h': 'cpp',
        'cpp': 'cpp',
        'c': 'cpp',
    }.get(lexer_name.replace('.', ''), lexer_name)
    if lexer_name in ['red', 'yellow', 'blue', 'green']:
        # hack for coloring
        return color_text(text, lexer_name)
    import utool as ut
    if ENABLE_COLORS:
        try:
            import pygments
            import pygments.lexers
            import pygments.formatters
            #from pygments import highlight
            #from pygments.lexers import get_lexer_by_name
            #from pygments.formatters import TerminalFormatter
            #if ut.WIN32:
            #    assert False
            #    #formater = pygments.formatters.terminal256.Terminal256Formatter()
            #    import pygments.formatters.terminal256
            #    formater = pygments.formatters.terminal256.Terminal256Formatter()
            #else:
            import pygments.formatters.terminal
            formater = pygments.formatters.terminal.TerminalFormatter(bg='dark')
            lexer = pygments.lexers.get_lexer_by_name(lexer_name, **kwargs)
            return pygments.highlight(text, lexer, formater)
        except Exception:
            if ut.SUPER_STRICT:
                raise
            return text
    return text


def color_text(text, color):
    r"""
    SeeAlso:
        highlight_text
        lexer_shortnames = sorted(ut.flatten(ut.take_column(pygments.lexers.LEXERS.values(), 2)))
    """
    import utool as ut
    if color is None or not ENABLE_COLORS:
        return text
    elif color == 'python':
        return highlight_text(text, color)
    elif color == 'sql':
        return highlight_text(text, 'sql')
    try:
        import pygments
        import pygments.console
        # if color == 'guess':
        #     import linguist  # NOQA
        #     pygments.lexers.guess_lexer(text)
        #     return highlight_text(text, color)
        try:
            ansi_text = pygments.console.colorize(color, text)
        except Exception:
            # Fix colors
            if color == 'turquoise':
                color = 'cyan'
            if color == 'darkgreen':
                color = 'green'
            if color == 'teal':
                color = 'brightcyan'
            if color == 'lightgray':
                color = 'white'
            ansi_text = pygments.console.colorize(color, text)

        if ut.WIN32:
            import colorama
            ansi_reset = (colorama.Style.RESET_ALL)
        else:
            ansi_reset = pygments.console.colorize('reset', '')
        ansi_text = ansi_text + ansi_reset
        return ansi_text
    except ImportError:
        return text


def highlight_regex(str_, pat, reflags=0, color='red'):
    """
    FIXME Use pygments instead
    """
    #import colorama
    # from colorama import Fore, Style
    #color = Fore.MAGENTA
    # color = Fore.RED
    #match = re.search(pat, str_, flags=reflags)
    matches = list(re.finditer(pat, str_, flags=reflags))

    colored = str_

    for match in reversed(matches):
        #pass
        #if match is None:
        #    return str_
        #else:
        start = match.start()
        end = match.end()
        #colorama.init()
        colored_part = color_text(colored[start:end], color)
        colored = colored[:start] + colored_part + colored[end:]
        # colored = (colored[:start] + color + colored[start:end] +
        #            Style.RESET_ALL + colored[end:])
        #colorama.deinit()
    return colored


def highlight_multi_regex(str_, pat_to_color, reflags=0):
    """
    FIXME Use pygments instead. must be mututally exclusive
    """
    #import colorama
    # from colorama import Fore, Style
    #color = Fore.MAGENTA
    # color = Fore.RED
    #match = re.search(pat, str_, flags=reflags)

    colored = str_

    to_replace = []
    for pat, color in pat_to_color.items():
        matches = list(re.finditer(pat, str_, flags=reflags))

        for match in matches:
            start = match.start()
            end = match.end()
            to_replace.append((end, start, color))

    for tup in reversed(sorted(to_replace)):
        end, start, color = tup
        colored_part = color_text(colored[start:end], color)
        colored = colored[:start] + colored_part + colored[end:]

    return colored


def varinfo_str(varval, varname, onlyrepr=False, canshowrepr=True,
                varcolor='yellow', colored=True):
    import utool as ut
    # varval = getattr(cm, varname.replace('cm.', ''))
    varinfo_list = []
    print_summary = not onlyrepr and ut.isiterable(varval)
    show_repr = True
    show_repr = show_repr or (onlyrepr or not print_summary)
    symbol = '*'
    if colored is not False and ut.util_dbg.COLORED_EXCEPTIONS:
        varname = ut.color_text(varname, varcolor)
    if show_repr:
        varval_str = ut.repr2(varval, precision=2)
        if len(varval_str) > 100:
            varval_str = '<omitted>'
        varval_str = ut.truncate_str(varval_str, maxlen=50)
        varinfo_list += ['    * %s = %s' % (varname, varval_str)]
        symbol = '+'
    if print_summary:
        if isinstance(varval, np.ndarray):
            depth = varval.shape
        else:
            depth = ut.depth_profile(varval)
        if not show_repr:
            varinfo_list += [
                # '    %s varinfo(%s):' % (symbol, varname,),
                '    %s %s = <not shown!>' % (symbol, varname,),
            ]
        varinfo_list += [
            '          len = %r' % (len(varval),)]
        if depth != len(varval):
            depth_str = ut.truncate_str(str(depth), maxlen=70)
            varinfo_list += [
                '          depth = %s' % (depth_str,)]
        varinfo_list += [
            '          types = %s' % (ut.list_type_profile(varval),)]
        #varinfo = '\n'.join(ut.align_lines(varinfo_list, '='))
    aligned_varinfo_list = ut.align_lines(varinfo_list, '=')
    varinfo = '\n'.join(aligned_varinfo_list)
    return varinfo


def testdata_text(num=1):
    import utool as ut
    #ut.util_dbg.COLORED_EXCEPTIONS = False
    text = r'''
        % COMMENT
        Image matching relies on finding similar features between query and
        database images, and there are many factors that can cause this to be
        difficult.
        % TALK ABOUT APPEARANCE HERE
        Similar to issues seen in (1) instance and (2) face recognition,
        images of animals taken ``in the wild'' contain many challenges
        such as occlusion, distractors and variations in viewpoint,
        pose, illumination, quality, and camera parameters.  We start
        the discussion of the problem addressed in this thesis by
        considering examples of these challenges.

        \distractorexample

        \paragraph{foobar}
        Occluders are objects in the foreground of an image that impact the
        visibility of the features on the subject animal.
         Both scenery and other animals are the main contributors of occlusion in
        our dataset.
         Occlusion from other animals is especially challenging because not only

        \begin{enumerate} % Affine Adaptation Procedure
           \item Compute the second moment matrix at the warped image patch defined by $\ellmat_i$.

           \item If the keypoint is stable, stop.  If convergence has not been reached in
                some number of iterations stop and discard the keypoint.

           \item
                  Update the affine shape  using the rule $\ellmat_{i + 1} =
                \sqrtm{\momentmat} \ellmat_i$.
                  This ensures the eigenvalues at the previously detected point
                are equal in the new frame.
                  If the keypoint is stable, it should be re-detected close to
                the same location.
                  (The square root of a matrix defined as:
                $\sqrtm{\momentmatNOARG} \equiv \mat{X} \where \mat{X}^T\mat{X}
                = \momentmatNOARG$.
                  If $\momentmatNOARG$ is degenerate than $\mat{X}$ does not
                exist.)
        \end{enumerate}
    '''.strip('\n') + '\n'

    text2 = ut.codeblock(r'''
        \begin{comment}
        python -m ibeis -e rank_cmc -t invar -a viewdiff --test_cfgx_slice=6: --db PZ_Master1 --hargv=expt --prefix "Invariance+View Experiment "  # NOQA
        \end{comment}
        \ImageCommand{figuresX/expt_rank_cmc_PZ_Master1_a_viewdiff_t_invar.png}{\textwidth}{
        Results of the invariance experiment with different viewpoints for plains
        zebras.  Only the results with different viewpoints are shown.  The query and
        database annotations are the same as those in the viewpoint experiment.  Thre
        is less than a $2\percent$ gap between the best results with keypoint
        invariance and the results without any keypoint invariance.  (Note that
        invariance we we discuss here only refers to keypoint shape and not the
        invariance that is implicit in the SIFT descriptor).
        }{PZInvarViewExpt}
    ''')  + '\n\n foobar foobar fooo. hwodefoobardoo\n\n'
    return text if num == 1 else text2


def regex_reconstruct_split(pattern, text, debug=False):
    import re
    #separators = re.findall(pattern, text)
    separators = [match.group() for match in re.finditer(pattern, text)]
    #separators = [match.group() for match in re.finditer(pattern, text, flags=re.MULTILINE)]
    if debug:
        import utool as ut
        ut.colorprint('[recon] separators = ' + ut.repr3(separators), 'green')

    remaining = text
    block_list = []
    for sep in separators:
        head, tail = remaining.split(sep, 1)
        block_list.append(head)
        remaining = tail
    block_list.append(remaining)

    if debug:
        ut.colorprint('[recon] block_list = ' + ut.repr3(block_list), 'red')

    return block_list, separators


def format_multiple_paragraph_sentences(text, debug=False, **kwargs):
    """
    FIXME: funky things happen when multiple newlines in the middle of
    paragraphs

    CommandLine:
        python ~/local/vim/rc/pyvim_funcs.py --test-format_multiple_paragraph_sentences

    CommandLine:
        python -m utool.util_str --exec-format_multiple_paragraph_sentences --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> import os, sys
        >>> #sys.path.append(os.path.expanduser('~/local/vim/rc'))
        >>> text = testdata_text(2)
        >>> formated_text = format_multiple_paragraph_sentences(text, debug=True)
        >>> print('+--- Text ---')
        >>> print(text)
        >>> print('+--- Formated Text ---')
        >>> print(formated_text)
        >>> print('L_____')
    """
    debug = _rectify_countdown_or_bool(debug)
    import utool as ut
    # Hack
    text = re.sub('^ *$', '', text, flags=re.MULTILINE)
    if debug:
        ut.colorprint(msgblock('[fmt] text', text), 'yellow')
    #print(text.replace(' ', '_'))
    #ut.util_dbg.COLORED_EXCEPTIONS = False
    # Patterns that define separations between paragraphs in latex
    pattern_list = [
        '\n\n\n*',     # newlines
        #'\n\n*$',     # newlines
        #'^\n\n*',     # newlines
        #'\n\n*',     # newlines
        '\n? *%.*\n',  # comments

        # paragraph commands
        '\n? *\\\\paragraph{[^}]*}\n',
        # '\n? *\\\\item \\\\textbf{[^}]*}: *\n',
        '\n? *\\\\item \\\\textbf{[^:]*}: *\n',
        '\n? *\\\\section{[^}]*}\n',
        '\n? *\\\\section{[^}]*}\\\\label{[^}]*}\n',
        '\n? *\\\\section{[^}]*}\\~?\\\\label{[^}]*}\n',

        '\n? *\\\\subsection{[^}]*}\\~?\\\\label{[^}]*}\n',
        '\n? *\\\\subsection{[^~]*}\\~?\\\\label{[^}]*}\n',
        '\n? *\\\\subsection{[^}]*}\n',

        '\n? *\\\\subsubsection{[^~]*}\\~?\\\\label{[^}]*}\n',
        '\n? *\\\\subsubsection{[^}]*}\n',

        '\n----*\n',
        '##* .*\n',

        '\\.}\n',
        '\\?}\n',

        '\n? *\\\\newcommand{[^}]*}.*\n',
        # generic multiline commands with text inside (like devcomment)
        '\n? *\\\\[a-zA-Z]+{ *\n',

        '\n? *\\\\begin{[^}]*}\n',
        '\n? *\\\\item *\n',
        '\n? *\\\\noindent *\n',
        '\n? *\\\\ImageCommand[^}]*}[^}]*}{\n',
        '\n? *\\\\end{[^}]*}\n?',
        '\n}{',

        # docstr stuff
        '\n' + ut.TRIPLE_DOUBLE_QUOTE + '\n',
        '\n? *Args: *\n',
        #'\n? [A-Za-z_]*[0-9A-Za-z_]* (.*?) *:',
    ]
    pattern = '|'.join(['(%s)' % (pat,) for pat in pattern_list])
    # break into paragraph blocks
    block_list, separators = regex_reconstruct_split(pattern, text,
                                                     debug=False)

    collapse_pos_list = []
    # Dont format things within certain block types
    _iter = ut.iter_window([''] + separators + [''], 2)
    for count, (block, window) in enumerate(zip(block_list, _iter)):
        if (window[0].strip() == r'\begin{comment}' and
             window[1].strip() == r'\end{comment}'):
            collapse_pos_list.append(count)

    tofmt_block_list = block_list[:]

    collapse_pos_list = sorted(collapse_pos_list)[::-1]
    for pos in collapse_pos_list:
        collapsed_sep = (separators[pos - 1] + tofmt_block_list[pos] +
                         separators[pos])
        separators[pos - 1] = collapsed_sep
        del separators[pos]
        del tofmt_block_list[pos]

    if debug:
        ut.colorprint('[fmt] tofmt_block_list = ' +
                      ut.repr3(tofmt_block_list), 'white')

    # apply formatting
    #if debug:
    #    ut.colorprint('--- FORMAT SENTENCE --- ', 'white')
    formated_block_list = []
    for block in tofmt_block_list:
        fmtblock = format_single_paragraph_sentences(
            block, debug=debug, **kwargs)
        formated_block_list.append(fmtblock)
        #ut.colorprint('---------- ', 'white')
    #if debug:
    #    ut.colorprint('--- / FORMAT SENTENCE --- ', 'white')
    rejoined_list = list(ut.interleave((formated_block_list, separators)))
    if debug:
        ut.colorprint('[fmt] formated_block_list = ' +
                      ut.repr3(formated_block_list), 'turquoise')
        #print(rejoined_list)
    formated_text = ''.join(rejoined_list)
    #ut.colorprint(formated_text.replace(' ', '_'), 'red')
    return formated_text

format_multi_paragraphs = format_multiple_paragraph_sentences


def split_sentences2(text, debug=0):
    import utool as ut
    raw_sep_chars = ['.', '?', '!', ':']
    USE_REGEX_SPLIT = True

    text_ = ut.remove_doublspaces(text)
    # TODO: more intelligent sentence parsing
    text_ = ut.flatten_textlines(text)

    if not USE_REGEX_SPLIT:
        # Old way that just handled periods
        sentence_list = text_.split('. ')
    else:
        # ******* #
        # SPLITS line endings based on regular expressions.
        esc = re.escape
        # Define separation patterns
        regex_sep_chars = list(map(re.escape, raw_sep_chars))
        regex_sep_prefix = [esc('(') + r'\d' + esc(')')]
        regex_sep_list = regex_sep_chars + regex_sep_prefix
        # Combine into a full regex
        sep_pattern = ut.regex_or(regex_sep_list)
        full_pattern = '(' + sep_pattern + r'+\s)'
        full_regex = re.compile(full_pattern)
        # Make the splits
        num_groups = full_regex.groups  # num groups in the regex
        split_list = re.split(full_pattern, text_)
        if len(split_list) > 0:
            num_bins = num_groups + 1
            sentence_list = split_list[0::num_bins]
            sep_list_group1 = split_list[1::num_bins]
            sep_list = sep_list_group1
        if debug:
            print('<SPLIT DBG>')
            print('num_groups = %r' % (num_groups,))
            print('len(split_list) = %r' % (len(split_list)))
            print('len(split_list) / len(sentence_list) = %r' % (
                len(split_list) / len(sentence_list)))
            print('len(sentence_list) = %r' % (len(sentence_list),))
            print('len(sep_list_group1) = %r' % (len(sep_list_group1),))
            #print('len(sep_list_group2) = %r' % (len(sep_list_group2),))
            print('full_pattern = %s' % (full_pattern,))
            #print('split_list = %r' % (split_list,))
            print('sentence_list = %s' % (ut.repr2(sentence_list),))
            print('sep_list = %s' % ((sep_list),))
            print('</SPLIT DBG>')
        # ******* #
        # FIXME: Place the separators either before or after a sentence
        from six.moves import zip_longest
        sentence_list2 = ['']
        _iter = zip_longest(sentence_list, sep_list)
        for count, (sentence, sep) in enumerate(_iter):
            if sep is None:
                sentence_list2[-1] += sentence
                continue
            sepchars = sep.strip()
            if len(sepchars) > 0 and sepchars[0] in raw_sep_chars:
                sentence_list2[-1] += sentence + (sep.strip())
                sentence_list2.append('')
            else:
                # Place before next
                sentence_list2[-1] += sentence
                sentence_list2.append(sep)

        sentence_list2 = [x.strip() for x in sentence_list2 if len(x.strip()) > 0]
        return sentence_list2


def format_single_paragraph_sentences(text, debug=False, myprefix=True,
                                      sentence_break=True, max_width=73,
                                      sepcolon=True):
    r"""
    helps me separatate sentences grouped in paragraphs that I have a
    difficult time reading due to dyslexia

    Args:
        text (str):

    Returns:
        str: wrapped_text

    CommandLine:
        python -m utool.util_str --exec-format_single_paragraph_sentences --show
        python -m utool.util_str --exec-format_single_paragraph_sentences --show --nobreak

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> import utool as ut
        >>> text = '     lorium ipsum doloar dolar dolar dolar erata man foobar is this there yet almost man not quit ate 80 chars yet hold out almost there? dolar erat. sau.ltum. fds.fd... . . fd oob fd. list: (1) abcd, (2) foobar (4) 123456789 123456789 123456789 123456789 123 123 123 123 123456789 123 123 123 123 123456789 123456789 123456789 123456789 123456789 123 123 123 123 123 123456789 123456789 123456789 123456789 123456789 123456789 (3) spam.'
        >>> #text = 'list: (1) abcd, (2) foobar (3) spam.'
        >>> #text = 'foo. when: (1) there is a new individual,'
        >>> #text = 'when: (1) there is a new individual,'
        >>> #text = '? ? . lorium. ipsum? dolar erat. saultum. fds.fd...  fd oob fd. ? '  # causes breakdown
        >>> print('text = %r' % (text,))
        >>> sentence_break = not ut.get_argflag('--nobreak')
        >>> wrapped_text = format_single_paragraph_sentences(text, debug=True, sentence_break=sentence_break)
        >>> result = ('wrapped_text =\n%s' % (str(wrapped_text),))
        >>> print(result)
    """
    import utool as ut
    #ut.util_dbg.COLORED_EXCEPTIONS = False
    import textwrap
    import re
    #ut.rrrr(verbose=False)
    # max_width = 73  # 79  # 80
    debug = _rectify_countdown_or_bool(debug)
    min_indent = ut.get_minimum_indentation(text)
    min_indent = (min_indent // 4) * 4
    if debug:
        print(ut.colorprint(msgblock('preflat', repr(text)), 'darkyellow'))
    text_ = ut.remove_doublspaces(text)
    # TODO: more intelligent sentence parsing
    text_ = ut.flatten_textlines(text)
    if debug:
        print(ut.colorprint(msgblock('postflat', repr(text_)), 'yellow'))

    raw_sep_chars = ['.', '?', '!', ':']
    if not sepcolon:
        raw_sep_chars.remove(':')

    USE_REGEX_SPLIT = True

    def split_sentences(text_):
        # TODO: rectify with split_sentences2
        if not USE_REGEX_SPLIT:
            # Old way that just handled periods
            sentence_list = text_.split('. ')
        else:
            # ******* #
            # SPLITS line endings based on regular expressions.
            esc = re.escape
            # Define separation patterns
            regex_sep_chars = list(map(re.escape, raw_sep_chars))
            regex_sep_prefix = [esc('(') + r'\d' + esc(')')]
            regex_sep_list = regex_sep_chars + regex_sep_prefix
            # Combine into a full regex
            sep_pattern = ut.regex_or(regex_sep_list)
            full_pattern = '(' + sep_pattern + r'+\s)'
            full_regex = re.compile(full_pattern)
            # Make the splits
            num_groups = full_regex.groups  # num groups in the regex
            split_list = re.split(full_pattern, text_)
            if len(split_list) > 0:
                num_bins = num_groups + 1
                sentence_list = split_list[0::num_bins]
                sep_list_group1 = split_list[1::num_bins]
                sep_list = sep_list_group1
            if debug:
                print('<SPLIT DBG>')
                print('num_groups = %r' % (num_groups,))
                print('len(split_list) = %r' % (len(split_list)))
                print('len(split_list) / len(sentence_list) = %r' % (
                    len(split_list) / len(sentence_list)))
                print('len(sentence_list) = %r' % (len(sentence_list),))
                print('len(sep_list_group1) = %r' % (len(sep_list_group1),))
                #print('len(sep_list_group2) = %r' % (len(sep_list_group2),))
                print('full_pattern = %s' % (full_pattern,))
                #print('split_list = %r' % (split_list,))
                print('sentence_list = %s' % (ut.repr2(sentence_list),))
                print('sep_list = %s' % ((sep_list),))
                print('</SPLIT DBG>')
            # ******* #
            return sentence_list, sep_list

    def wrap_sentences(sentence_list, min_indent, max_width):
        # prefix for continuations of a sentence
        if myprefix:
            # helps me read LaTeX
            sentence_prefix = '  '
        else:
            sentence_prefix = ''
        if text_.startswith('>>>'):
            # Hack to do docstrings
            # TODO: make actualy docstring reformater
            sentence_prefix = '...     '

        if max_width is not None:
            width = max_width - min_indent - len(sentence_prefix)

            wrapkw = dict(width=width, break_on_hyphens=False, break_long_words=False)
            #wrapped_lines_list = [textwrap.wrap(sentence_prefix + line, **wrapkw)
            #                      for line in sentence_list]
            wrapped_lines_list = []
            for count, line in enumerate(sentence_list):
                wrapped_lines = textwrap.wrap(line, **wrapkw)
                wrapped_lines = [line_ if count == 0 else sentence_prefix + line_
                                 for count, line_ in enumerate(wrapped_lines)]
                wrapped_lines_list.append(wrapped_lines)

            wrapped_sentences = ['\n'.join(line) for line in wrapped_lines_list]
        else:
            wrapped_sentences = sentence_list[:]
        return wrapped_sentences

    def rewrap_sentences2(sentence_list, sep_list):
        # FIXME: probably where nl error is
        # ******* #
        # put the newline before or after the sep depending on if it is
        # supposed to prefix or suffix the sentence.
        from six.moves import zip_longest
        # FIXME: Place the separators either before or after a sentence
        sentence_list2 = ['']
        _iter = zip_longest(sentence_list, sep_list)
        for count, (sentence, sep) in enumerate(_iter):
            if sep is None:
                sentence_list2[-1] += sentence
                continue
            sepchars = sep.strip()
            if len(sepchars) > 0 and sepchars[0] in raw_sep_chars:
                sentence_list2[-1] += sentence + (sep.strip())
                sentence_list2.append('')
            else:
                # Place before next
                sentence_list2[-1] += sentence
                sentence_list2.append(sep)
        sentence_list2 = [x.strip() for x in sentence_list2 if len(x.strip()) > 0]
        return sentence_list2

    # New way
    #print('last_is_nl = %r' % (last_is_nl,))
    if sentence_break:
        # Break at sentences
        sentence_list, sep_list = split_sentences(text_)
        # FIXME: probably where nl error is
        sentence_list2 = rewrap_sentences2(sentence_list, sep_list)
        wrapped_sentences = wrap_sentences(sentence_list2, min_indent, max_width)
        wrapped_block = '\n'.join(wrapped_sentences)
    else:
        # Break anywhere
        width = max_width - min_indent
        wrapkw = dict(width=width, break_on_hyphens=False,
                      break_long_words=False)
        wrapped_block = '\n'.join(textwrap.wrap(text_, **wrapkw))

        if False:
            # for a one-time purpose
            print('HACKING')
            print('width = {!r}'.format(width))
            # HACK
            words = text_.split(', (')
            lines = []
            line = ''
            for _, w1 in enumerate(words):
                if _ > 0:
                    w1 = '(' + w1 + ', '
                if len(line + w1) > width:
                    line += ''
                    lines.append(line)
                    line = ''
                line += w1
            lines.append(line)
            wrapped_block = '\n'.join(lines)

    # HACK for last nl (seems to only happen if nl follows a seperator)
    last_is_nl = text.endswith('\n') and  not wrapped_block.endswith('\n')
    first_is_nl = len(text) > 1 and text.startswith('\n') and not wrapped_block.startswith('\n')
    # if last_is_nl and wrapped_block.strip().endswith('.'):
    if last_is_nl:
        wrapped_block += '\n'
    if first_is_nl:
        wrapped_block = '\n' + wrapped_block
    # Do the final indentation
    wrapped_text = ut.indent(wrapped_block, ' ' * min_indent)
    return wrapped_text


def find_block_end(row, line_list, sentinal, direction=1):
    """
    Searches up and down until it finds the endpoints of a block Rectify
    with find_paragraph_end in pyvim_funcs
    """
    import re
    row_ = row
    line_ = line_list[row_]
    flag1 = row_ == 0 or row_ == len(line_list) - 1
    flag2 = re.match(sentinal, line_)
    if not (flag1 or flag2):
        while True:
            if (row_ == 0 or row_ == len(line_list) - 1):
                break
            line_ = line_list[row_]
            if re.match(sentinal, line_):
                break
            row_ += direction
    return row_


def insert_block_between_lines(text, row1, row2, line_list, inplace=False):
    lines = [line.encode('utf-8') for line in text.split('\n')]
    if inplace:
        buffer_tail = line_list[row2:]  # Original end of the file
        new_tail = lines + buffer_tail
        del line_list[row1 - 1:]  # delete old data
        line_list.append(new_tail)  # append new data
    else:
        line_list = line_list[:row1 + 1] + lines + line_list[row2:]
    return line_list


def toggle_comment_lines(text, pattern, flag, char='#'):
    # TODO: respect indentation
    lines = text.split('\n')
    if flag:
        lines = [char + line if re.search(pattern, line) else line
                 for line in lines]
    else:
        lines = [line.lstrip(char) if re.search(pattern, line) else line
                 for line in lines]
    new_text = '\n'.join(lines)
    return new_text


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_str; utool.doctest_funcs(utool.util_str)"
        python -m utool.util_str
        python -m utool.util_str --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
