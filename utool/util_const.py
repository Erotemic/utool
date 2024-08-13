# -*- coding: utf-8 -*-
# util_const.py
from __future__ import absolute_import, division, print_function, unicode_literals


class _NoParamType(object):
    """
    Class used to define `NoParam`, a setinal that acts like None when None
    might be a valid value. The value of `NoParam` is robust to reloading,
    pickling, and copying.

    Howver, try to never assign this value to a persistant variable.  Use this
    class sparingly.

    CommandLine:
        python -m utool.util_const _NoParamType

    References:
        http://stackoverflow.com/questions/41048643/a-second-none
    """
    def __new__(cls):
        return NoParam
    def __reduce__(self):
        return (_NoParamType, ())
    def __copy__(self):
        return NoParam
    def __deepcopy__(self, memo):
        return NoParam
    def __call__(self, default):
        pass
    def __str__(cls):
        return 'NoParam'
        # return "<type 'NoParamType'>"
    def __repr__(cls):
        return 'NoParam'
        # return "<type 'NoParamType'>"

# Create the only instance of _NoParamType that should ever exist
try:
    # If the module is reloaded (via importlib.reload), globals() will contain
    # NoParam. This skips the code that would instantiate a second object
    NoParam
    # Note: it is possible to hack around this via
    # >>> del util_const.NoParam
    # >>> importlib.reload(util_const)
except NameError:
    # When the module is first loaded, globals() will not contain NoParam. A
    # NameError will be thrown, causing the first instance of NoParam to be
    # instanciated.
    NoParam = object.__new__(_NoParamType)
