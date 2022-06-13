# -*- coding: utf-8 -*-
"""
DEPRICATE
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import re

MAX_VALSTR = -1


class AbstractPrintable(object):
    """
    A base class that prints its attributes instead of the memory address
    """

    def __init__(self, child_print_exclude=[]):
        self._printable_exclude = ['_printable_exclude'] + child_print_exclude

    def __str__(self):
        head = printableType(self)
        body = self.get_printable(type_bit=True)
        body = re.sub('\n *\n *\n', '\n\n', body)
        return head + ('\n' + body).replace('\n', '\n    ')

    def printme(self):
        print(str(self))

    def printme3(self):
        print(self.get_printable())

    def printme2(self,
                 type_bit=True,
                 print_exclude_aug=[],
                 val_bit=True,
                 max_valstr=MAX_VALSTR,
                 justlength=True):
        to_print = self.get_printable(type_bit=type_bit,
                                      print_exclude_aug=print_exclude_aug,
                                      val_bit=val_bit,
                                      max_valstr=max_valstr,
                                      justlength=justlength)
        print(to_print)

    def get_printable(self,
                      type_bit=True,
                      print_exclude_aug=[],
                      val_bit=True,
                      max_valstr=MAX_VALSTR,
                      justlength=False):
        from utool.util_str import truncate_str
        body = ''
        attri_list = []
        exclude_key_list = list(self._printable_exclude) + list(print_exclude_aug)
        for (key, val) in six.iteritems(self.__dict__):
            try:
                if key in exclude_key_list:
                    continue
                namestr = str(key)
                typestr = printableType(val, name=key, parent=self)
                if not val_bit:
                    attri_list.append((typestr, namestr, '<ommited>'))
                    continue
                valstr = printableVal(val, type_bit=type_bit, justlength=justlength)
                valstr = truncate_str(valstr, maxlen=max_valstr, truncmsg=' \n ~~~ \n ')
                #if len(valstr) > max_valstr and max_valstr > 0:
                #    pos1 =  max_valstr // 2
                #    pos2 = -max_valstr // 2
                #    valstr = valstr[0:pos1] + ' \n ~~~ \n ' + valstr[pos2: - 1]
                attri_list.append((typestr, namestr, valstr))
            except Exception as ex:
                print('[printable] ERROR %r' % ex)
                print('[printable] ERROR key = %r' % key)
                print('[printable] ERROR val = %r' % val)
                try:
                    print('[printable] ERROR valstr = %r' % valstr)
                except Exception:
                    pass
                raise
        attri_list.sort()
        for (typestr, namestr, valstr) in attri_list:
            entrytail = '\n' if valstr.count('\n') <= 1 else '\n\n'
            typestr2 = typestr + ' ' if type_bit else ''
            body += typestr2 + namestr + ' = ' + valstr + entrytail
        return body

    def format_printable(self, type_bit=False, indstr='  * '):
        _printable_str = self.get_printable(type_bit=type_bit)
        _printable_str = _printable_str.replace('\r', '\n')
        _printable_str = indstr + _printable_str.strip('\n').replace('\n', '\n' + indstr)
        return _printable_str


# - --------------

def printableType(val, name=None, parent=None):
    """
    Tries to make a nice type string for a value.
    Can also pass in a Printable parent object
    """
    import numpy as np
    if parent is not None and hasattr(parent, 'customPrintableType'):
        # Hack for non - trivial preference types
        _typestr = parent.customPrintableType(name)
        if _typestr is not None:
            return _typestr
    if isinstance(val, np.ndarray):
        info = npArrInfo(val)
        _typestr = info.dtypestr
    elif isinstance(val, object):
        _typestr = val.__class__.__name__
    else:
        _typestr = str(type(val))
        _typestr = _typestr.replace('type', '')
        _typestr = re.sub('[\'><]', '', _typestr)
        _typestr = re.sub('  *', ' ', _typestr)
        _typestr = _typestr.strip()
    return _typestr


def printableVal(val, type_bit=True, justlength=False):
    """
    Very old way of doing pretty printing. Need to update and refactor.
    DEPRICATE
    """
    from utool import util_dev
    # Move to util_dev
    # NUMPY ARRAY
    import numpy as np
    if type(val) is np.ndarray:
        info = npArrInfo(val)
        if info.dtypestr.startswith('bool'):
            _valstr = '{ shape:' + info.shapestr + ' bittotal: ' + info.bittotal + '}'
            # + '\n  |_____'
        elif info.dtypestr.startswith('float'):
            _valstr = util_dev.get_stats_str(val)
        else:
            _valstr = '{ shape:' + info.shapestr + ' mM:' + info.minmaxstr + ' }'  # + '\n  |_____'
    # String
    elif isinstance(val, six.text_type):  # NOQA
        _valstr = '\'%s\'' % val
    # List
    elif isinstance(val, list):
        if justlength or len(val) > 30:
            _valstr = 'len=' + str(len(val))
        else:
            _valstr = '[ ' + (', \n  '.join([str(v) for v in val])) + ' ]'
    # ??? isinstance(val, AbstractPrintable):
    elif hasattr(val, 'get_printable') and type(val) != type:
        _valstr = val.get_printable(type_bit=type_bit)
    elif isinstance(val, dict):
        _valstr = '{\n'
        for val_key in val.keys():
            val_val = val[val_key]
            _valstr += '  ' + str(val_key) + ' : ' + str(val_val) + '\n'
        _valstr += '}'
    else:
        _valstr = str(val)
    if _valstr.find('\n') > 0:  # Indent if necessary
        _valstr = _valstr.replace('\n', '\n    ')
        _valstr = '\n    ' + _valstr
    _valstr = re.sub('\n *$', '', _valstr)  # Replace empty lines
    return _valstr


def npArrInfo(arr):
    """
    OLD update and refactor
    """
    from utool.DynamicStruct import DynStruct
    info = DynStruct()
    info.shapestr  = '[' + ' x '.join([str(x) for x in arr.shape]) + ']'
    info.dtypestr  = str(arr.dtype)
    if info.dtypestr == 'bool':
        info.bittotal = 'T=%d, F=%d' % (sum(arr), sum(1 - arr))
    elif info.dtypestr == 'object':
        info.minmaxstr = 'NA'
    elif info.dtypestr[0] == '|':
        info.minmaxstr = 'NA'
    else:
        if arr.size > 0:
            info.minmaxstr = '(%r, %r)' % (arr.min(), arr.max())
        else:
            info.minmaxstr = '(None)'
    return info
