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

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> from utool import util_const
        >>> from utool.util_const import _NoParamType, NoParam
        >>> from six.moves import cPickle as pickle
        >>> import copy
        >>> id_ = id(NoParam)
        >>> versions = {
        ... 'ut.util_const.NoParam': ut.util_const.NoParam,
        ... 'NoParam': NoParam,
        ... '_NoParamType()': _NoParamType(),
        ... 'ut.NoParam': ut.NoParam,
        ... 'copy': copy.copy(NoParam),
        ... 'deepcopy': copy.deepcopy(NoParam),
        ... 'pickle': pickle.loads(pickle.dumps(NoParam))
        ... }
        >>> print(ut.align(ut.repr4(ut.map_vals(id, versions)), ':'))
        >>> assert all(id(v) == id_ for v in versions.values())
        >>> import imp
        >>> imp.reload(util_const)
        >>> assert id(util_const.NoParam) == id_
        >>> assert all(id(v) == id_ for v in versions.values())
        >>> #ut.rrrr()
        >>> #import imp
        >>> #imp.reload(ut.util_const)
        >>> #versions['post_reload1'] = ut.util_const.NoParam
        >>> #versions['post_reload2'] = ut.util_const.NoParam
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
    # If the module is reloaded (via imp.reload), globals() will contain
    # NoParam. This skips the code that would instantiate a second object
    NoParam
    # Note: it is possible to hack around this via
    # >>> del util_const.NoParam
    # >>> imp.reload(util_const)
except NameError:
    # When the module is first loaded, globals() will not contain NoParam. A
    # NameError will be thrown, causing the first instance of NoParam to be
    # instanciated.
    NoParam = object.__new__(_NoParamType)


# class Singleton(type):
#     _instances = {}
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]


# import six
# @six.add_metaclass(Singleton)
# class _NoParamType2(object):
#     """
#     http://stackoverflow.com/questions/6760685/creating-a-singleton-in-python

#     CommandLine:
#         python -m utool.util_const _NoParamType2

#     Example:
#         >>> # DISABLE_DOCTEST
#         >>> from utool.util_const import _NoParamType2
#         >>> import utool as ut
#         >>> from six.moves import cPickle as pickle
#         >>> import copy
#         >>> NoParam2 = _NoParamType2()
#         >>> NoParam2Dump = pickle.dumps(NoParam2)
#         >>> versions = {
#         ... 'dup1': ut.util_const._NoParamType2(),
#         ... 'dup2': _NoParamType2(),
#         ... 'copy': copy.copy(NoParam2),
#         ... 'deepcopy': copy.deepcopy(NoParam2),
#         ... 'pickle': pickle.loads(NoParam2Dump)
#         ... }
#         >>> print(ut.align(ut.repr4(ut.map_vals(id, versions)), ':'))

#     """
#     def __reduce__(self):
#         return (_NoParamType2, ())
#     def __copy__(self):
#         return _NoParamType2()
#     def __deepcopy__(self, memo):
#         return _NoParamType2()
#     def __call__(self, default):
#         pass
# NoParam = _NoParamType2()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m utool.util_const
        python -m utool.util_const --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
