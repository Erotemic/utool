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


def import_module_from_fpath(module_fpath):
    """ imports module from a file path

    Args:
        module_fpath (str):

    Returns:
        module: module

    CommandLine:
        python -m utool.util_import --test-import_module_from_fpath

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_import import *  # NOQA
        >>> module_fpath = '?'
        >>> module = import_module_from_fpath(module_fpath)
        >>> result = ('module = %s' % (str(module),))
        >>> print(result)


    Ignore:
        module_fpath = '/home/joncrall/code/h5py/h5py'
        module_fpath = '/home/joncrall/code/h5py/build/lib.linux-x86_64-2.7/h5py'
        ut.ls(module_fpath)
        imp.find_module('h5py', '/home/joncrall/code/h5py/')


        # Define two temporary modules that are not in sys.path
        # and have the same name but different values.
        import sys, os, os.path
        def ensuredir(path):
            if not os.path.exists(path):
                os.mkdir(path)
        ensuredir('tmp')
        ensuredir('tmp/tmp1')
        ensuredir('tmp/tmp2')
        ensuredir('tmp/tmp1/testmod')
        ensuredir('tmp/tmp2/testmod')
        with open('tmp/tmp1/testmod/__init__.py', 'w') as file_:
            file_.write('foo = \"spam\"\nfrom . import sibling')
        with open('tmp/tmp1/testmod/sibling.py', 'w') as file_:
            file_.write('bar = \"ham\"')
        with open('tmp/tmp2/testmod/__init__.py', 'w') as file_:
            file_.write('foo = \"eggs\"\nfrom . import sibling')
        with open('tmp/tmp1/testmod/sibling.py', 'w') as file_:
            file_.write('bar = \"jam\"')

        # Neither module should be importable through the normal mechanism
        try:
            import testmod
            assert False, 'should fail'
        except ImportError as ex:
            pass

        # Try temporarilly adding the directory of a module to the path
        sys.path.insert(0, 'tmp/tmp1')
        testmod1 = __import__('testmod', globals(), locals(), 0)
        sys.path.remove('tmp/tmp1')
        print(testmod1.foo)
        print(testmod1.sibling.bar)

        sys.path.insert(0, 'tmp/tmp2')
        testmod2 = __import__('testmod', globals(), locals(), 0)
        sys.path.remove('tmp/tmp2')
        print(testmod2.foo)
        print(testmod2.sibling.bar)

        assert testmod1.foo == "spam"
        assert testmod1.sibling.bar == "ham"

        # Fails, returns spam
        assert testmod2.foo == "eggs"
        assert testmod2.sibling.bar == "jam"

        sys.path.append('/home/username/code/h5py')
        import h5py
        sys.path.append('/home/username/code/h5py/build/lib.linux-x86_64-2.7/')
        import h5py
    """
    from os.path import basename, splitext, isdir, join, exists, dirname, split
    import platform
    if isdir(module_fpath):
        module_fpath = join(module_fpath, '__init__.py')
    assert exists(module_fpath)
    python_version = platform.python_version()
    modname = splitext(basename(module_fpath))[0]
    if modname == '__init__':
        modname = split(dirname(module_fpath))[1]
    if python_version.startswith('2.7'):
        import imp
        module = imp.load_source(modname, module_fpath)
    elif python_version.startswith('3'):
        import importlib.machinery
        loader = importlib.machinery.SourceFileLoader(modname, module_fpath)
        module = loader.load_module()
    else:
        raise AssertionError('invalid python version=%r' % (python_version,))
    return module


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
