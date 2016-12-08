# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


class _NoParamType(object):
    """
    class used in place of None when None might be a valid value.

    The copy methods are overridden to try and enforce that only one instance
    of this class is ever created. However this can not be gaurenteed
    completely.  This is NOT robust to module realoding or pickling.  Try to
    never assign this value to a persistant variable.  Use this class
    sparingly.

    CommandLine:
        python -m utool.util_const _NoParamType

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> from utool.util_const import _NoParamType, NoParam
        >>> from six.moves import cPickle as pickle
        >>> import copy
        >>> NoParamDump = pickle.dumps(NoParam)
        >>> versions = {
        ... 'ut.util_const.NoParam': ut.util_const.NoParam,
        ... 'NoParam': NoParam,
        ... 'ut.NoParam': ut.NoParam,
        ... 'copy': copy.copy(NoParam),
        ... 'deepcopy': copy.deepcopy(NoParam),
        ... 'pickle': pickle.loads(NoParamDump)
        ... }
        >>> print(ut.align(ut.repr4(ut.map_vals(id, versions)), ':'))
    """
    def __init__(self):
        pass
    def __copy__(self):
        return NoParam
    def __deepcopy__(self, memo):
        return NoParam
    def __getstate__(self, state):
        return NoParam
    def __setstate__(self, state):
        return NoParam
    def __hash__(self):
        return 0
    def __call__(self, default):
        pass

# Note: if this module is reloaded functions depending on NoParam will break
NoParam = _NoParamType()


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
