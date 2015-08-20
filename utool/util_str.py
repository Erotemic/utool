# -*- coding: utf-8 -*-
"""
Module that handles string formating and manipulation of varoius data
"""
from __future__ import absolute_import, division, print_function
import sys
import six
import re
import textwrap
from six.moves import map, range, reduce
import itertools
import math
import collections
from os.path import split
from utool import util_type
from utool import util_time  # import get_unix_timedelta
from utool._internal import meta_util_six  # import get_funcname
from utool import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[str]')

if util_type.HAVE_NUMPY:
    import numpy as np
    TAU = (2 * np.pi)  # References: tauday.com
else:
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
    r"""
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


def percent_str(pcnt):
    return 'undef' if pcnt is None else '%06.2f %%' % (pcnt * 100,)


def tupstr(tuple_):
    """ maps each item in tuple to a string and doesnt include parens """
    return ', '.join(list(map(str, tuple_)))

# --- Strings ----


def scalar_str(val, precision=None):
    if precision is not None and (isinstance(val, (float)) or util_type.is_float(val)):
        return ('%.' + str(precision) + 'f') % (val,)
    else:
        return str(val)


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
    """
    returns the number of preceding spaces
    """
    lines = text.split('\n')
    indentations = [get_indentation(line_) for line_ in lines  if len(line_.strip()) > 0]
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
    return unindent(block_str).strip('\n')


def flatten_textlines(text):
    new_text = text
    new_text = re.sub(' *\n *', ' ', new_text, flags=re.MULTILINE).strip(' ')
    return new_text


def remove_doublspaces(text):
    new_text = text
    new_text = re.sub('  *', ' ', new_text)
    #, flags=re.MULTILINE)
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
    new_text = '\n\n'.join(list(map(flatten_textlines, multiline_text.split('\n\n'))))
    return new_text


def indent(str_, indent='    '):
    """
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
    """ TODO fix name """
    return str_.replace('\n', '\n' + indent)


def indentcat(str1, str2, indent='    '):
    return str1  + str2.replace('\n', '\n' + indent)


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
    return list(map(lambda item: indent + str(item), list_))


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
        >>> import utool as ut
        >>> num_list = sorted([4.2 / (10.0 ** exp_) for exp_ in range(-13, 13, 4)])
        >>> secstr_list = [seconds_str(num, prefix=None) for num in num_list]
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


def byte_str2(nBytes, precision=2):
    """
    Automatically chooses relevant unit (KB, MB, or GB) for displaying some
    number of bytes.

    Args:
        nBytes (int):

    Returns:
        str:

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> nBytes_list = [1, 100, 1024,  1048576, 1073741824, 1099511627776]
        >>> result = list(map(byte_str2, nBytes_list))
        >>> print(result)
        ['0.00 KB', '0.10 KB', '1.00 KB', '1.00 MB', '1.00 GB', '1024.00 GB']
    """
    nAbsBytes = abs(nBytes)
    if nAbsBytes < 2.0 ** 10:
        return byte_str(nBytes, 'KB', precision=precision)
    if nAbsBytes < 2.0 ** 20:
        return byte_str(nBytes, 'KB', precision=precision)
    if nAbsBytes < 2.0 ** 30:
        return byte_str(nBytes, 'MB', precision=precision)
    else:
        return byte_str(nBytes, 'GB', precision=precision)


def byte_str(nBytes, unit='bytes', precision=2):
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
    return scalar_str(nUnit, precision) + ' ' + unit
    #fmtstr = ('%.'
    #return ('%.' + str(precision) + 'f %s') % (nUnit, unit)


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
    func_str = '%s(%s)' % (meta_util_six.get_funcname(func), argskwargs_str)
    if packed:
        packkw_ = dict(textwidth=80, nlprefix='    ', break_words=False)
        if packkw is not None:
            packkw_.update(packkw_)
        func_str = packstr(func_str, **packkw_)
    return func_str


def array_repr2(arr, max_line_width=None, precision=None, suppress_small=None, force_dtype=False, **kwargs):
    """ extended version of np.core.numeric.array_repr

    ut.editfile(np.core.numeric.__file__)

    On linux:
    _typelessdata [numpy.int64, numpy.float64, numpy.complex128, numpy.int64]

    On BakerStreet
    _typelessdata [numpy.int32, numpy.float64, numpy.complex128, numpy.int32]

    # WEIRD
    np.int64 is np.int64
    _typelessdata[0] is _typelessdata[-1]
    _typelessdata[0] == _typelessdata[-1]


    id(_typelessdata[-1])
    id(_typelessdata[0])


    from numpy.core.numeric import _typelessdata
    _typelessdata

    Referencs:
        http://stackoverflow.com/questions/28455982/why-are-there-two-np-int64s-in-numpy-core-numeric-typelessdata-why-is-numpy-in/28461928#28461928
    """
    from numpy.core.numeric import _typelessdata

    if arr.__class__ is not np.ndarray:
        cName = arr.__class__.__name__
    else:
        cName = 'array'

    prefix = cName + '('

    if arr.size > 0 or arr.shape == (0,):
        separator = ', '
        lst = array2string2(arr, max_line_width, precision, suppress_small, separator, prefix, **kwargs)
    else:
        # show zero-length shape unless it is (0,)
        lst = '[], shape=%s' % (repr(arr.shape),)

    skipdtype = ((arr.dtype.type in _typelessdata) and arr.size > 0)

    if skipdtype and not (cName == 'array' and force_dtype):
        return '%s(%s)' % (cName, lst)
    else:
        typename = arr.dtype.name
        # Quote typename in the output if it is 'complex'.
        if typename and not (typename[0].isalpha() and typename.isalnum()):
            typename = '\'%s\'' % typename

        lf = ''
        if issubclass(arr.dtype.type, np.flexible):
            if arr.dtype.names:
                typename = '%s' % str(arr.dtype)
            else:
                typename = '\'%s\'' % str(arr.dtype)
            lf = '\n' + ' ' * len(prefix)
        return cName + '(%s, %sdtype=%s)' % (lst, lf, typename)


def array2string2(a, max_line_width=None, precision=None, suppress_small=None,
                  separator=' ', prefix="", style=repr, formatter=None, threshold=None):
    """
    expanded version of np.core.arrayprint.array2string
    """

    if a.shape == ():
        x = a.item()
        try:
            lst = a._format(x)
            msg = "The `_format` attribute is deprecated in Numpy " \
                  "2.0 and will be removed in 2.1. Use the " \
                  "`formatter` kw instead."
            import warnings
            warnings.warn(msg, DeprecationWarning)
        except AttributeError:
            if isinstance(x, tuple):
                x = np.core.arrayprint._convert_arrays(x)
            lst = style(x)
    elif reduce(np.core.arrayprint.product, a.shape) == 0:
        # treat as a null array if any of shape elements == 0
        lst = "[]"
    else:
        lst = _array2string2(a, max_line_width, precision, suppress_small,
                             separator, prefix, formatter=formatter, threshold=threshold)
    return lst


def _array2string2(a, max_line_width, precision, suppress_small, separator=' ',
                   prefix="", formatter=None, threshold=None):
    """
    expanded version of np.core.arrayprint._array2string
    TODO: make a numpy pull request with a fixed version

    """

    if max_line_width is None:
        max_line_width = np.core.arrayprint._line_width

    if precision is None:
        precision = np.core.arrayprint._float_output_precision

    if suppress_small is None:
        suppress_small = np.core.arrayprint._float_output_suppress_small

    if formatter is None:
        formatter = np.core.arrayprint._formatter

    if threshold is None:
        threshold = np.core.arrayprint._summaryThreshold

    if threshold > 0 and a.size > threshold:
        summary_insert = "..., "
        data = np.core.arrayprint._leading_trailing(a)
    else:
        summary_insert = ""
        data = np.core.arrayprint.ravel(a)

    formatdict = {'bool' : np.core.arrayprint._boolFormatter,
                  'int' : np.core.arrayprint.IntegerFormat(data),
                  'float' : np.core.arrayprint.FloatFormat(data, precision, suppress_small),
                  'longfloat' : np.core.arrayprint.LongFloatFormat(precision),
                  'complexfloat' : np.core.arrayprint.ComplexFormat(data, precision, suppress_small),
                  'longcomplexfloat' : np.core.arrayprint.LongComplexFormat(precision),
                  'datetime' : np.core.arrayprint.DatetimeFormat(data),
                  'timedelta' : np.core.arrayprint.TimedeltaFormat(data),
                  'numpystr' : np.core.arrayprint.repr_format,
                  'str' : str}

    if formatter is not None:
        fkeys = [k for k in formatter.keys() if formatter[k] is not None]
        if 'all' in fkeys:
            for key in formatdict.keys():
                formatdict[key] = formatter['all']
        if 'int_kind' in fkeys:
            for key in ['int']:
                formatdict[key] = formatter['int_kind']
        if 'float_kind' in fkeys:
            for key in ['float', 'longfloat']:
                formatdict[key] = formatter['float_kind']
        if 'complex_kind' in fkeys:
            for key in ['complexfloat', 'longcomplexfloat']:
                formatdict[key] = formatter['complex_kind']
        if 'str_kind' in fkeys:
            for key in ['numpystr', 'str']:
                formatdict[key] = formatter['str_kind']
        for key in formatdict.keys():
            if key in fkeys:
                formatdict[key] = formatter[key]

    try:
        format_function = a._format
        msg = "The `_format` attribute is deprecated in Numpy 2.0 and " \
              "will be removed in 2.1. Use the `formatter` kw instead."
        import warnings
        warnings.warn(msg, DeprecationWarning)
    except AttributeError:
        # find the right formatting function for the array
        dtypeobj = a.dtype.type
        if issubclass(dtypeobj, np.core.arrayprint._nt.bool_):
            format_function = formatdict['bool']
        elif issubclass(dtypeobj, np.core.arrayprint._nt.integer):
            if issubclass(dtypeobj, np.core.arrayprint._nt.timedelta64):
                format_function = formatdict['timedelta']
            else:
                format_function = formatdict['int']
        elif issubclass(dtypeobj, np.core.arrayprint._nt.floating):
            if issubclass(dtypeobj, np.core.arrayprint._nt.longfloat):
                format_function = formatdict['longfloat']
            else:
                format_function = formatdict['float']
        elif issubclass(dtypeobj, np.core.arrayprint._nt.complexfloating):
            if issubclass(dtypeobj, np.core.arrayprint._nt.clongfloat):
                format_function = formatdict['longcomplexfloat']
            else:
                format_function = formatdict['complexfloat']
        elif issubclass(dtypeobj, (np.core.arrayprint._nt.unicode_, np.core.arrayprint._nt.string_)):
            format_function = formatdict['numpystr']
        elif issubclass(dtypeobj, np.core.arrayprint._nt.datetime64):
            format_function = formatdict['datetime']
        else:
            format_function = formatdict['numpystr']

    # skip over "["
    next_line_prefix = " "
    # skip over array(
    next_line_prefix += " " * len(prefix)

    lst = np.core.arrayprint._formatArray(a, format_function, len(a.shape), max_line_width,
                                          next_line_prefix, separator,
                                          np.core.arrayprint._summaryEdgeItems, summary_insert)[:-1]
    return lst


def numpy_str2(arr, **kwargs):
    kwargs['force_dtype'] = kwargs.get('force_dtype', False)
    kwargs['suppress_small'] = kwargs.get('suppress_small', True)
    kwargs['precision'] = kwargs.get('precision', 3)
    return numpy_str(arr, **kwargs)


def numpy_str(arr, strvals=False, precision=8, pr=None, force_dtype=True, suppress_small=None, max_line_width=None, threshold=None, **kwargs):
    """
    suppress_small = False turns off scientific representation
    """
    kwargs = kwargs.copy()
    if 'suppress' in kwargs:
        suppress_small = kwargs['suppress']
    if max_line_width is None and 'linewidth' in kwargs:
        max_line_width = kwargs.pop('linewidth')

    if pr is not None:
        precision = pr
    # TODO: make this a util_str func for numpy reprs
    if strvals:
        valstr = np.array_str(arr, precision=precision, suppress_small=suppress_small, **kwargs)
    else:
        #valstr = np.array_repr(arr, precision=precision)
        valstr = array_repr2(arr, precision=precision, force_dtype=force_dtype,
                             suppress_small=suppress_small,
                             max_line_width=max_line_width,
                             threshold=threshold, **kwargs)
        numpy_vals = itertools.chain(util_type.NUMPY_SCALAR_NAMES, ['array'])
        for npval in numpy_vals:
            valstr = valstr.replace(npval, 'np.' + npval)
    if valstr.find('\n') >= 0:
        # Align multiline arrays
        valstr = valstr.replace('\n', '\n   ')
        pass
    return valstr


def numeric_str(num, precision=8, **kwargs):
    """
    Args:
        num (scalar or array):
        precision (int):

    Returns:
        str:

    CommandLine:
        python -m utool.util_str --test-numeric_str

    References:
        http://stackoverflow.com/questions/4541155/check-if-a-number-is-int-or-float

    Notes:
        isinstance(np.array([3], dtype=np.uint8)[0], numbers.Integral)
        isinstance(np.array([3], dtype=np.int32)[0], numbers.Integral)
        isinstance(np.array([3], dtype=np.uint64)[0], numbers.Integral)
        isinstance(np.array([3], dtype=object)[0], numbers.Integral)
        isinstance(np.array([3], dtype=np.float32)[0], numbers.Integral)
        isinstance(np.array([3], dtype=np.float64)[0], numbers.Integral)

    CommandLine:
        python -m utool.util_str --test-numeric_str

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> precision = 2
        >>> result = [numeric_str(num, precision) for num in [1, 2.0, 3.43343,4432]]
        >>> print(result)
        ['1', '2.00', '3.43', '4432']
    """
    import numbers
    if np.isscalar(num):
        if not isinstance(num, numbers.Integral):
            return scalar_str(num, precision)
            #fmtstr = ('%.' + str(precision) + 'f')
            #return fmtstr  % num
        else:
            return '%d' % (num)
        return
    else:
        return numpy_str(num, precision=precision, **kwargs)


def dict_itemstr_list(dict_, strvals=False, sorted_=None, newlines=True,
                      recursive=True, indent_='', precision=8,
                      hack_liststr=False, explicit=False, truncate=False, key_order=None,
                      truncatekw=dict(), key_order_metric=None, use_numpy=True, **dictkw):
    """
    Returns:
        list: a list of human-readable dictionary items

    Args:
        explicit : if True uses dict(key=val,...) format instead of {key:val,...}

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
                            indent_=indent_ + '    ', precision=precision,
                            truncate=truncate, truncatekw=truncatekw,
                            key_order=key_order, use_numpy=use_numpy, **dictkw)
        elif util_type.HAVE_NUMPY and isinstance(val, np.ndarray):
            if use_numpy:
                return numpy_str(val, strvals=strvals, precision=precision)
            else:
                return list_str(val, newlines=newlines, precision=precision)
        if hack_liststr and isinstance(val, list):
            return list_str(val, newlines=newlines, precision=precision)
        else:
            # base case
            return valfunc(val)

    #def iteritems(x):
    #    try:
    #        return six.iteritems(x)
    #    except AttributeError:
    #        return iter(x.items())
    if sorted_ is None:
        sorted_ = not isinstance(dict_, collections.OrderedDict)
    if sorted_:
        def iteritems(d):
            if key_order is None:
                # specify order explicilty
                try:
                    return iter(sorted(six.iteritems(d)))
                except TypeError:
                    # catches case where keys are of different types
                    return six.iteritems(d)
            else:
                unordered_keys = list(d.keys())
                other_keys = sorted(list(set(unordered_keys) - set(key_order)))
                keys = key_order + other_keys
                return ((key, d[key]) for key in keys)

        #iteritems = lambda d: iter(sorted(six.iteritems(d)))
    else:
        iteritems = six.iteritems

    _valstr = recursive_valfunc if recursive else valfunc
    OLD = False
    if OLD:
        if explicit:
            fmtstr = indent_ + '%r: %s,'
        itemstr_list = [fmtstr % (key, _valstr(val)) for (key, val) in iteritems(dict_)]
    else:
        def make_item_str(key, val, indent_):
            if explicit:
                repr_str = key + '='
            else:
                repr_str = repr(key) + ': '
            val_str = _valstr(val)
            #print('2)-----------')
            #print(val_str)
            # valstr is fine at this point
            padded_indent = ' ' * min(len(indent_), len(repr_str))
            val_str = val_str.replace('\n', '\n' + padded_indent)  # ' ' * val_indent)
            #val_str = ut.indent(val_str, ' ' * val_indent)
            item_str = repr_str + val_str + ','
            #print('3)-----------')
            #print(val_str)
            #print('4)===========')
            #print(item_str)
            #print('===========')
            return item_str

        #if isinstance(dict_, dict)
        itemstr_list = [make_item_str(key, val, indent_) for (key, val) in iteritems(dict_)]
        # itemstr_list is fine too. weird

    if key_order_metric == 'strlen':
        #if key_order is None:
        #    # specify order explicilty
        #    try:
        #        return iter(sorted(six.iteritems(d)))
        #    except TypeError:
        #        # catches case where keys are of different types
        #        return six.iteritems(d)
        #else:
        #    unordered_keys = list(d.keys())
        #    other_keys = sorted(list(set(unordered_keys) - set(key_order)))
        #    keys = key_order + other_keys
        import utool as ut
        metric_list = [len(itemstr) for itemstr in itemstr_list]
        itemstr_list = ut.sortedby(itemstr_list, metric_list)
    elif key_order_metric == 'val':
        import utool as ut
        metric_list = [val for (key, val) in iteritems(dict_)]
        itemstr_list = ut.sortedby(itemstr_list, metric_list)
    #import utool as ut
    #ut.embed()
    #itemstr_list = [fmtstr % (key, _valstr(val)) for (key, val) in iteritems(dict_)]
    return itemstr_list


def get_itemstr_list(list_, strvals=False, newlines=True,
                      recursive=True, indent_='', precision=8, label_list=None,
                     **listkws):
    """
    TODO: have this replace dict_itemstr list or at least most functionality in
    it. have it make two itemstr lists over keys and values and then combine
    them.
    """
    if strvals:
        valfunc = str
    else:
        valfunc = repr

    def recursive_valfunc(val, sublabels=None):
        new_indent = indent_ + '    ' if newlines else indent_
        if isinstance(val, dict):
            # recursive call
            #return dict_str(val, strvals=strvals, newlines=newlines,
            #                recursive=recursive, indent_=new_indent,
            #                precision=precision)
            return dict_str(val, strvals=strvals, newlines=newlines,
                            recursive=recursive, indent_=new_indent,
                            precision=precision, sorted_=True,
                            hack_liststr=listkws.get('hack_liststr', False))
        if isinstance(val, (tuple, list)):
            return list_str(val, strvals=strvals, newlines=newlines,
                            recursive=recursive, indent_=new_indent,
                            precision=precision, label_list=sublabels,
                            **listkws)
        elif util_type.HAVE_NUMPY and isinstance(val, np.ndarray):
            # TODO: generally pass down args
            suppress_small = listkws.get('suppress_small', None)
            return numpy_str(val, strvals=strvals, precision=precision, suppress_small=suppress_small)
        elif precision is not None and (isinstance(val, (float)) or util_type.is_float(val)):
            return scalar_str(val, precision)
            #return ('%.' + str(precision) + 'f') % (val,)
        else:
            # base case
            return valfunc(val)

    _valstr = recursive_valfunc if recursive else valfunc

    def make_item_str(item, label=None):
        if isinstance(label, (list, tuple)):
            val_str = _valstr(item, label)
        else:
            val_str = _valstr(item)
        if isinstance(label, six.string_types):
            prefix = label + ' = '
            #item_str = prefix + indent_rest(val_str, ' ' * len(prefix))
            item_str = horiz_string(prefix,  val_str)
        else:
            item_str = val_str + ','
        return item_str

    if label_list is not None:
        assert len(label_list) == len(list_)
        itemstr_list = [make_item_str(item, label) for item, label in zip(list_, label_list)]
    else:
        itemstr_list = [make_item_str(item) for item in list_]
    return itemstr_list


def list_str_summarized(list_, list_name, maxlen=5):
    """
    prints the list members when the list is small and the length when it is
    large
    """
    if len(list_) > maxlen:
        return 'len(%s)=%d' % (list_name, len(list_))
    else:
        return '%s=%r' % (list_name, list_)


def _rectify_countdown_or_bool(count_or_bool):
    """
    used by recrusive functions to specify which level to turn a bool on in
    counting down yeilds True, True, ..., False
    conting up yeilds False, False, False, ... True

    Args:
        count_or_bool (bool or int): if positive will count down, if negative will count up, if bool will remain same

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
    import math
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


def list_str(list_, indent_='', newlines=1, nobraces=False, nl=None, truncate=False, truncatekw={}, label_list=None, **listkw):
    r"""
    Args:
        list_ (list):
        indent_ (str): (default = '')
        newlines (int): (default = 1)
        nobraces (bool): (default = False)
        nl (None): alias for newlines (default = None)

    Returns:
        str: body_str

    CommandLine:
        python -m utool.util_str --test-list_str
        python -m utool.util_str --test-list_str --no-checkwant --truncate=True
        python -m utool.util_str --test-list_str --no-checkwant --truncate=0 --no-checkwant


    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_str import *  # NOQA
        >>> import utool as ut
        >>> list_ = [[(('--verbose-qt', '--verbqt'), 1, False, ''),
        ...     (('--verbose-qt', '--verbqt'), 1, False, ''), (('--verbose-qt',
        ...     '--verbqt'), 1, False, ''), (('--verbose-qt', '--verbqt'), 1,
        ...     False, '')], [(['--nodyn'], 1, False, ''), (['--nodyn'], 1, False,
        ...     '')]]
        >>> indent_ = ''
        >>> newlines = 2
        >>> truncate = ut.get_argval('--truncate', type_=None, default=False)
        >>> nobraces = False
        >>> nl = None
        >>> result = list_str(list_, indent_, newlines, nobraces, nl, truncate=truncate, truncatekw={'maxlen': 10})
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
    #return '[%s\n]' % indentjoin(list(list_), suffix=',')
    #if newlines is True:
    #    newlines_ = newlines
    ## newlines can either be a bool or countdown variable
    #elif isinstance(newlines, int):
    #    newlines_ = newlines - 1
    #else:
    #    newlines_ = False
    if nl is not None:
        newlines = nl
    newlines_ = _rectify_countdown_or_bool(newlines)
    truncate_ = _rectify_countdown_or_bool(truncate)
    #print('--')
    #print(indent_ + 'newlines = %r' % (newlines,))
    #print(indent_ + 'newlines_ = %r' % (newlines_,))

    itemstr_list = get_itemstr_list(list_, indent_=indent_, newlines=newlines_,
                                    truncate=truncate_, truncatekw=truncatekw,
                                    label_list=label_list, **listkw)
    if isinstance(list_, tuple):
        leftbrace, rightbrace  = '(', ')'
    else:
        leftbrace, rightbrace  = '[', ']'

    if newlines is not False and (newlines is True or newlines > 0):
        import utool as ut
        if nobraces or label_list is not None:
            body_str = '\n'.join(itemstr_list)
            retstr = body_str
        else:
            body_str = '\n'.join([ut.indent(itemstr) for itemstr in itemstr_list])
            braced_body_str = (leftbrace + '\n' + body_str + '\n' + rightbrace)
            retstr = braced_body_str
        #return (leftbrace + indentjoin(itemstr_list) + '\n' + indent_ + rightbrace)
    else:
        # hack away last comma
        sequence_str = ' '.join(itemstr_list)
        sequence_str = sequence_str.rstrip(',')
        retstr  = (leftbrace + sequence_str +  rightbrace)

    # TODO: rectify with dict_truncate
    do_truncate = truncate is not False and (truncate is True or truncate == 0)
    if do_truncate:
        retstr = truncate_str(retstr, **truncatekw)
    return retstr


def obj_str(obj_, **kwargs):
    if isinstance(obj_, dict):
        return dict_str(obj_, **kwargs)
    if isinstance(obj_, list):
        return list_str(obj_, **kwargs)
    else:
        return repr(obj_)


def dict_str(dict_, strvals=False, sorted_=None, newlines=True, recursive=True,
             indent_='', precision=8, hack_liststr=False, truncate=False,
             nl=None, explicit=False, truncatekw=dict(), key_order=None,
             key_order_metric=None, nobraces=False, align=False, **dictkw):
    """

    Args:
        dict_ (dict_):  a dictionary
        strvals (bool): (default = False)
        sorted_ (None): returns str sorted by a metric (default = None)
        newlines (bool): can be a coundown variable (default = True)
        recursive (bool): (default = True)
        indent_ (str): (default = '')
        precision (int): (default = 8)
        hack_liststr (bool): turn recursive liststr parsing on (default = False)
        truncate (bool): (default = False)
        nl (None): (default = None)
        explicit (bool): (default = False)
        truncatekw (dict): (default = {})
        key_order (None): overrides default ordering (default = None)
        key_order_metric (None): special sorting of items

    FIXME: ALL LIST DICT STRINGS ARE VERY SPAGEHETTI RIGHT NOW
    Returns:
        str: a human-readable and execable string representation of a dictionary

    CommandLine:
        python -m utool.util_str --test-dict_str
        python -m utool.util_str --test-dict_str --truncate=False --no-checkwant
        python -m utool.util_str --test-dict_str --truncate=1 --no-checkwant
        python -m utool.util_str --test-dict_str --truncate=2 --no-checkwant

    Example:
        >>> from utool.util_str import dict_str, dict_itemstr_list
        >>> import utool
        >>> #REPO_CONFIG = utool.get_default_repo_config()
        >>> dict_ = {'foo': {'spam': 'barbarbarbarbar' * 3, 'eggs': 'jam'}, 'baz': 'barbarbarbarbar' * 3}
        >>> truncate = ut.get_argval('--truncate', type_=None, default=1)
        >>> result  = dict_str(dict_, strvals=True, truncate=truncate, truncatekw={'maxlen': 20})
        >>> print(result)
        {
            'baz': barbarbarbarbarbarbarbarbarbarbarbarbarbarbar,
            'foo': {
                'eggs': jam,
                's ~~~TRUNCATED~~~ ,
            },
        }
    """
    if nl is not None:
        newlines = nl
    if len(dict_) == 0:
        if explicit:
            return 'dict()'
        else:
            return '{}'
    newlines_ = _rectify_countdown_or_bool(newlines)
    truncate_ = _rectify_countdown_or_bool(truncate)
    #print('----')
    #print(indent_ + 'truncate = %r' % (truncate,))
    #print(indent_ + 'truncate_ = %r' % (truncate_,))
    #print('----')

    itemstr_list = dict_itemstr_list(dict_, strvals, sorted_, newlines_,
                                     recursive, indent_, precision,
                                     hack_liststr, explicit,
                                     truncate=truncate_, truncatekw=truncatekw,
                                     key_order=key_order,
                                     key_order_metric=key_order_metric,
                                     **dictkw)

    do_truncate = truncate is not False and (truncate is True or truncate == 0)
    if do_truncate:
        #print('----')
        #print(indent_ + 'truncate = %r' % (truncate,))
        #print(indent_ + 'Truncating')
        #print('----')
        itemstr_list = [truncate_str(item, **truncatekw) for item in itemstr_list]

    leftbrace, rightbrace  = ('dict(', ')') if explicit else ('{', '}')
    if nobraces:
        leftbrace = ''
        rightbrace = ''

    if newlines:
        import utool as ut
        if nobraces:
            retstr =  '\n'.join(itemstr_list)
        else:
            body_str = '\n'.join([ut.indent(itemstr, '    ') for itemstr in itemstr_list])
            retstr =  (leftbrace + '\n' + body_str + '\n' + rightbrace)
            if align:
                retstr = ut.align(retstr, ':')
    else:
        retstr = leftbrace + ' '.join(itemstr_list) + rightbrace
    # Is there a way to make truncate for dict_str compatible with list_str?
    #print('----')
    #print(indent_ + 'truncate = %r' % (truncate,))
    #print(indent_ + 'Truncating')
    #print('----')
    #retstr = truncate_str(retstr, **truncatekw)
    return retstr


def horiz_string(*args, **kwargs):
    """
    Horizontally concatenates strings reprs preserving indentation

    Concats a list of objects ensuring that the next item in the list
    is all the way to the right of any previous items.

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
            if util_type.HAVE_NUMPY:
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
    timedelta = util_time.get_unix_timedelta(unixtime_diff)
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
        return meta_util_six.get_funcname(func)
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

    Args:
        line_list (list of strs):
        character (str):

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

    """
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
            new_lines = align_lines(new_lines, character=character, replchar=replchar, pos=pos)
        return new_lines

    # base case
    if replchar is None:
        replchar = character

    # the pos-th character to align
    lpos = pos
    rpos = lpos + 1

    tup_list = [line.split(character) for line in line_list]

    # Find how much padding is needed
    maxlen = 0
    for tup in tup_list:
        if len(tup) >= rpos + 1:
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
            #if False:
            newline = lhs.ljust(maxlen) + replchar + rhs
            #else:
            #    if lpos < 1:
            #        newline = lhs.rjust(maxlen) + replchar + rhs
            #    else:
            #        newline = lhs.ljust(maxlen) + replchar + rhs
            new_lines.append(newline)
        else:
            new_lines.append(replchar.join(tup))
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


def multi_replace(str_, search_list, repl_list):
    for search, repl in zip(search_list, repl_list):
        str_ = str_.replace(search, repl)
    return str_


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


def get_textdiff(text1, text2, num_context_lines=0):
    r"""
    Uses difflib to return a difference string between two
    similar texts

    References:
        http://www.java2s.com/Code/Python/Utility/IntelligentdiffbetweentextfilesTimPeters.htm

    Args:
        text1 (?):
        text2 (?):

    Returns:
        ?:

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
    text1_lines = text1.splitlines()
    text2_lines = text2.splitlines()
    all_diff_lines = list(difflib.ndiff(text1_lines, text2_lines))
    if num_context_lines is None:
        diff_lines = all_diff_lines
    else:
        from utool import util_list
        # boolean for every line if it is marked or not
        ismarked_list = [len(line) > 0 and line[0] in '+-?' for line in all_diff_lines]
        # flag lines that are within num_context_lines away from a diff line
        isvalid_list = ismarked_list[:]
        for i in range(1, num_context_lines + 1):
            isvalid_list[:-i] = util_list.or_lists(isvalid_list[:-i], ismarked_list[i:])
            isvalid_list[i:]  = util_list.or_lists(isvalid_list[i:], ismarked_list[:-i])
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
                    diff_lines.append(visual_break)
                prev = valid
        else:
            diff_lines = util_list.filter_items(all_diff_lines, isvalid_list)
        #
    return '\n'.join(diff_lines)


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
    if isinstance(var, np.ndarray):
        line_str = ut.numpy_str(var, precision=precision)
    else:
        line_str = repr(var)
    doctest_line_str = doctest_code_line(line_str, varname=varname_, verbose=verbose)
    return doctest_line_str


def lorium_ipsum():
    ipsum_str = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed
    do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad
    minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex
    ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate
    velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat
    cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id
    est laborum.
    """
    return ipsum_str


def bubbletext(text, font='cybermedium'):
    r"""
    Other fonts include: cybersmall, cybermedium, and cyberlarge

    import pyfiglet
    TODO move elsewhere

    References:
        http://www.figlet.org/

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> bubble_text1 = ut.bubbletext('TESTING', font='cyberlarge')
        >>> bubble_text2 = ut.bubbletext('BUBBLE', font='cybermedium')
        >>> bubble_text3 = ut.bubbletext('TEXT', font='cyberlarge')
        >>> print('\n'.join([bubble_text1, bubble_text2, bubble_text3]))
    """
    # TODO: move this function elsewhere
    import utool as ut
    pyfiglet = ut.tryimport('pyfiglet', 'git+https://github.com/pwaller/pyfiglet')
    if pyfiglet is None:
        return text
    else:
        bubble_text = pyfiglet.figlet_format(text, font=font)
        return bubble_text


def edit_distance(query, options):
    """
    pip install python-Levenshtein

    query = 'hello world'
    options = ['goodbye world', 'rofl', 'hello', 'world', 'lowo']
    [7, 9, 6, 6, 7]
    """
    import Levenshtein
    dist_list = [Levenshtein.distance(query, other) for other in options]
    return dist_list


def closet_words(query, options, num=1):
    import utool as ut
    dist_list = edit_distance(query, options)
    ranked_list = ut.sortedby(options, dist_list)
    return ranked_list[0:num]


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
