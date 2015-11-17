# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


class ClassNoParam(object):
    """
    # class used in place of None when None might be a valid value
    # probably should just make None not a valid value
    """
    def __init__(self):
        pass
    def __call__(self, default):
        pass

# Used instance of NoParam
NoParam = ClassNoParam()
