"""
SeeAlso:
    utool._internal.util_importer
"""
from __future__ import absolute_import, division, print_function
from utool import util_inject
from utool import util_arg
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[import]')


lazy_module_attrs =  ['_modname', '_module', '_load_module']


class LazyModule(object):
    """
    Waits to import the module until it is actually used.
    Caveat: there is no access to module attributes used
        in ``lazy_module_attrs``

    CommandLine:
        python -m utool.util_import --test-LazyModule

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_import import *  # NOQA
        >>> import sys
        >>> assert 'this' not in sys.modules,  'this was imported before test start'
        >>> this = LazyModule('this')
        >>> assert 'this' not in sys.modules,  'this should not have been imported yet'
        >>> assert this.i == 25
        >>> assert 'this' in sys.modules,  'this should now be imported'
        >>> print(this)
    """
    def __init__(self, modname):
        r"""
        Args:
            modname (str):  module name
        """
        self._modname = modname
        self._module = None

    def _load_module(self):
        if self._module is None:
            if util_arg.VERBOSE:
                print('lazy loading module module')
            self._module =  __import__(self._modname, globals(), locals(), fromlist=[], level=0)

    def __str__(self):
        return 'LazyModule(%s)' % (self._modname,)

    def __dir__(self):
        self._load_module()
        return dir(self._module)

    def __getattr__(self, item):
        if item in lazy_module_attrs:
            return super(LazyModule, self).__getattr__(item)
        self._load_module()
        return getattr(self._module, item)

    def __setattr__(self, item, value):
        if item in lazy_module_attrs:
            return super(LazyModule, self).__setattr__(item, value)
        self._load_module()
        setattr(self._module, item, value)


#modname = 'theano'
#theano = LazyModule(modname)
if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_import
        python -m utool.util_import --allexamples
        python -m utool.util_import --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
