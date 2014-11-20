from __future__ import absolute_import, division, print_function
import six
import re
#from .util_classes import AutoReloader

MAX_VALSTR = -1
#100000

#__BASE_CLASS__ = AutoReloader
__BASE_CLASS__ = object


class AbstractPrintable(__BASE_CLASS__):
    'A base class that prints its attributes instead of the memory address'

    def __init__(self, child_print_exclude=[]):
        self._printable_exclude = ['_printable_exclude'] + child_print_exclude

    def __str__(self):
        from utool.util_dev import printableType
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
        from utool.util_dev import printableVal, printableType
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
