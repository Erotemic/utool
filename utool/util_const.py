# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


class _ClassNoParam(object):
    """
    class used in place of None when None might be a valid value
    """
    def __init__(self):
        pass
    def __call__(self, default):
        pass

# Note: if this module is reloaded functions depending on NoParam will break
NoParam = _ClassNoParam()
