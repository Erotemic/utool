# -*- coding: utf-8 -*-
"""
SeeAlso:
    utool._internal.util_importer

    TODO: http://code.activestate.com/recipes/473888-lazy-module-imports/
    https://pypi.python.org/pypi/zope.deferredimport/3.5.2
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from utool import util_inject
# from utool import util_arg
import sys
print, rrr, profile = util_inject.inject2(__name__)


# def dynamic_import(modname, submod):
#     """
#     CommandLine:
#         python -m utool.util_import dynamic_import opengm.opengmcore _opengmcore

#     Tutorial:
#         ### Instead of writing:

#         from {your_submodule} import *
#         ### OR
#         from {your_module}.{your_submodule} import *

#         ### You can use utool's dynamic importer:

#         if True:
#             # To Autogenerate Run
#             '''
#             python -c "import {your_module}" --print-{your_module}-init
#             python -c "import {your_module}" --update-{your_module}-init
#             '''
#             import utool as ut
#             ut.dynamic_import(__name__, '{your_submodule}')
#         else:
#             # <AUTOGEN_INIT>
#             pass
#             # </AUTOGEN_INIT>

#        ### This will do the same thing, but you can "freeze" your module
#        ### and autogenerate what import * would have done. This helps
#        ### with static analysis and overall code readability.

#     Ignore:
#         modname = 'opengm.opengmcore'
#         submod = '_opengmcore'
#     """
#     from utool._internal import util_importer
#     if isinstance(submod, list):
#         import_tuples = submod
#     else:
#         import_tuples = [(submod, None)]
#     import_execstr, initstr = util_importer.dynamic_import(
#         modname, import_tuples, check_not_imported=False,
#         return_initstr=True)
#     return initstr


def import_star(modname, parent=None):
    """
    Args:
        modname (str): module name

    Tutorial:
        Replacement for
        from modname import *

        Usage is like this
        globals().update(ut.import_star('<module>'))
        OR
        ut.import_star('<module>', __name__)

    Ignore:
        >>> from utool.util_import import *  # NOQA
        modname = 'opengm'
        submod = None
        parent = __name__
    """
    import six
    from os.path import dirname
    if parent is None:
        parent_module = None
    else:
        parent_module = sys.modules[parent]
        if isinstance(parent, six.string_types):
            parent_module = sys.modules[parent]
        else:
            raise ValueError('parent must be the module __name__ attribute')

    try:
        module = sys.modules[modname]
    except KeyError:
        try:
            module = __import__(modname, {}, {}, fromlist=[], level=0)
        except ImportError:
            if parent_module is None:
                print('Maybe try specifying parent?')
                raise
            # Inject into the parent if given
            # Temporilly put this module dir in the pythonpath to simulate
            # relative imports
            relative_to = dirname(parent_module.__file__)
            sys.path.append(relative_to)
            try:
                module = __import__(modname, {}, {}, fromlist=[], level=0)
            except Exception:
                raise
            finally:
                sys.path.pop()
    # get public attributes
    module_attrs = [attr for attr in dir(module) if not attr.startswith('_')]
    module_vars = {attr: getattr(module, attr) for attr in module_attrs}
    if parent is not None:
        # Inject into the parent if given
        if isinstance(parent, six.string_types):
            parent_module = sys.modules[parent]
            for attr, var in module_vars.items():
                setattr(parent_module, attr, var)
        else:
            raise ValueError('parent must be the module __name__ attribute')
    # return the module dictionary
    return module_vars


def import_star_execstr(modname, parent=None):
    """
    print(ut.import_star_execstr('opengm.inference'))
    """
    from utool import util_str
    module_vars = import_star(modname, parent=parent)
    fromlist_str = ', '.join(sorted(module_vars.keys()))
    fromimport_prefix = 'from {modname} import ('.format(modname=modname)

    newline_prefix = (' ' * len(fromimport_prefix))
    if fromlist_str:
        rawstr = fromimport_prefix + fromlist_str + ',)'
    else:
        rawstr = ''
    textwidth = 79 - 4
    fromimport_str = util_str.pack_into(rawstr, textwidth=textwidth,
                                        newline_prefix=newline_prefix,
                                        break_words=False)

    # fromimport_str = ut.autopep8_format(fromimport_str, ignore={})
    return fromimport_str


# TODO import_freezestar


def possible_import_patterns(modname):
    r"""
    does not support from x import *
    does not support from x import z, y

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> modname = 'package.submod.submod2.module'
        >>> result = ut.repr3(ut.possible_import_patterns(modname))
        >>> print(result)
        [
            'import\\spackage.submod.submod2.module',
            'from\\spackage\\.submod\\.submod2\\simportmodule',
        ]
    """
    # common regexes
    WS = r'\s'
    import_ = 'import'
    from_ = 'from'
    dot_ = r'\.'
    patterns = [import_ + WS + modname]
    if '.' in modname:
        parts = modname.split('.')
        modpart = dot_.join(parts[0:-1])
        imppart = parts[-1]
        patterns += [from_ + WS + modpart + WS + import_ + imppart]
    NONSTANDARD = False
    if NONSTANDARD:
        if '.' in modname:
            for i in range(1, len(parts) - 1):
                modpart = '.'.join(parts[i:-1])
                imppart = parts[-1]
                patterns += [from_ + WS + modpart + WS + import_ + imppart]
            imppart = parts[-1]
            patterns += [import_ + WS + imppart]
    return patterns


def package_contents(package, with_pkg=False, with_mod=True, ignore_prefix=[],
                     ignore_suffix=[]):
    r"""
    References:
        http://stackoverflow.com/questions/1707709/list-all-the-modules-that-are-part-of-a-python-package

    Args:
        package (?):
        with_pkg (bool): (default = False)
        with_mod (bool): (default = True)

    CommandLine:
        python -m utool.util_import --exec-package_contents

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_import import *  # NOQA
        >>> import utool as ut
        >>> import ibeis
        >>> package = ibeis
        >>> ignore_prefix = ['ibeis.tests', 'ibeis.control.__SQLITE3__',
        >>>                  '_autogen_explicit_controller']
        >>> ignore_suffix = ['_grave']
        >>> with_pkg = False
        >>> with_mod = True
        >>> result = package_contents(package, with_pkg, with_mod,
        >>>                           ignore_prefix, ignore_suffix)
        >>> print(ut.repr2(result))
    """
    import pkgutil
    if not hasattr(package, '__path__'):
        return [package.__name__]
    #    pass
    print('package = %r' % (package,))
    walker = pkgutil.walk_packages(package.__path__,
                                   prefix=package.__name__ + '.',
                                   onerror=lambda x: None)
    module_list = []
    for importer, modname, ispkg in walker:
        if any(modname.startswith(prefix) for prefix in ignore_prefix):
            continue
        if any(modname.endswith(suffix) for suffix in ignore_suffix):
            continue
        if not ispkg and with_mod:
            module_list.append(modname)
        if ispkg and with_pkg:
            module_list.append(modname)
    return module_list


def get_modpath_from_modname(modname, prefer_pkg=False, prefer_main=False):
    """
    Same as get_modpath but doesnt import directly

    SeeAlso:
        get_modpath
    """
    from os.path import dirname, basename, join, exists
    initname = '__init__.py'
    mainname = '__main__.py'
    if modname in sys.modules:
        modpath = sys.modules[modname].__file__.replace('.pyc', '.py')
    else:
        import pkgutil
        loader = pkgutil.find_loader(modname)
        modpath = loader.filename.replace('.pyc', '.py')
        if '.' not in basename(modpath):
            modpath = join(modpath, initname)
    if prefer_pkg:
        if modpath.endswith(initname) or modpath.endswith(mainname):
            modpath = dirname(modpath)
    if prefer_main:
        if modpath.endswith(initname):
            main_modpath = modpath[:-len(initname)] + mainname
            if exists(main_modpath):
                modpath = main_modpath
    return modpath


def check_module_installed(modname):
    """
    Check if a python module is installed without attempting to import it.
    Note, that if ``modname`` indicates a child module, the parent module is
    always loaded.

    Args:
        modname (str):  module name

    Returns:
        bool: found

    References:
        http://stackoverflow.com/questions/14050281/module-exists-without-importing

    CommandLine:
        python -m utool.util_import check_module_installed --show --verbimp --modname=this
        python -m utool.util_import check_module_installed --show --verbimp --modname=ibeis.scripts.iccv

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_import import *  # NOQA
        >>> import utool as ut
        >>> modname = ut.get_argval('--modname', default='this')
        >>> is_installed = check_module_installed(modname)
        >>> is_imported = modname in sys.modules
        >>> print('module(%r).is_installed = %r' % (modname, is_installed))
        >>> print('module(%r).is_imported = %r' % (modname, is_imported))
        >>> assert 'this' not in sys.modules, 'module(this) should not have ever been imported'
    """
    import pkgutil
    if '.' in modname:
        # Prevent explicit import if possible
        parts = modname.split('.')
        base = parts[0]
        submods = parts[1:]
        loader = pkgutil.find_loader(base)
        if loader is not None:
            # TODO: check to see if path to the submod exists
            submods
            return True
    loader = pkgutil.find_loader(modname)
    is_installed = loader is not None
    return is_installed


def import_modname(modname):
    r"""
    Args:
        modname (str):  module name

    Returns:
        module: module

    CommandLine:
        python -m utool.util_import --test-import_modname

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_import import *  # NOQA
        >>> modname_list = [
        >>>     'utool',
        >>>     'utool._internal',
        >>>     'utool._internal.meta_util_six',
        >>>     'utool.util_path',
        >>>     #'utool.util_path.checkpath',
        >>> ]
        >>> modules = [import_modname(modname) for modname in modname_list]
        >>> result = ([m.__name__ for m in modules])
        >>> assert result == modname_list
    """
    # The __import__ statment is weird
    if util_inject.PRINT_INJECT_ORDER:
        if modname not in sys.modules:
            util_inject.noinject(modname, N=2, via='ut.import_modname')
    if '.' in modname:
        fromlist = modname.split('.')[-1]
        fromlist_ = list(map(str, fromlist))  # needs to be ascii for python2.7
        module = __import__(modname, {}, {}, fromlist_, 0)
    else:
        module = __import__(modname, {}, {}, [], 0)
    return module


def tryimport(modname, pipiname=None, ensure=False):
    """
    CommandLine:
        python -m utool.util_import --test-tryimport

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_tests import *  # NOQA
        >>> import utool as ut
        >>> modname = 'pyfiglet'
        >>> pipiname = 'git+https://github.com/pwaller/pyfiglet'
        >>> pyfiglet = ut.tryimport(modname, pipiname)
        >>> assert pyfiglet is None or isinstance(pyfiglet, types.ModuleType), 'unknown error'

    Example2:
        >>> # UNSTABLE_DOCTEST
        >>> # disabled because not everyone has access to being a super user
        >>> from utool.util_tests import *  # NOQA
        >>> import utool as ut
        >>> modname = 'lru'
        >>> pipiname = 'git+https://github.com/amitdev/lru-dict'
        >>> lru = ut.tryimport(modname, pipiname, ensure=True)
        >>> assert isinstance(lru, types.ModuleType), 'did not ensure lru'
    """
    if pipiname is None:
        pipiname = modname
    try:
        if util_inject.PRINT_INJECT_ORDER:
            if modname not in sys.modules:
                util_inject.noinject(modname, N=2, via='ut.tryimport')
        module = __import__(modname)
        return module
    except ImportError as ex:
        import utool as ut
        base_pipcmd = 'pip install %s' % pipiname
        sudo  = not ut.WIN32 and not ut.in_virtual_env()
        if sudo:
            pipcmd = 'sudo ' + base_pipcmd
        else:
            pipcmd = base_pipcmd
        msg = 'unable to find module %s. Please install: %s' % ((modname), (pipcmd))
        print(msg)
        ut.printex(ex, msg, iswarning=True)
        if ensure:
            raise AssertionError('Ensure is dangerous behavior and is is no longer supported.')
            #raise NotImplementedError('not ensuring')
            ut.cmd(base_pipcmd, sudo=sudo)
            module = tryimport(modname, pipiname, ensure=False)
            if module is None:
                raise AssertionError('Cannot ensure modname=%r please install using %r'  % (modname, pipcmd))
            return module
        return None


# lazy_module_attrs =  ['_modname', '_module', '_load_module']


# class LazyModule(object):
#     """
#     Waits to import the module until it is actually used.
#     Caveat: there is no access to module attributes used
#         in ``lazy_module_attrs``

#     CommandLine:
#         python -m utool.util_import --test-LazyModule

#     Example:
#         >>> # DISABLE_DOCTEST
#         >>> from utool.util_import import *  # NOQA
#         >>> import sys
#         >>> assert 'this' not in sys.modules,  'this was imported before test start'
#         >>> this = LazyModule('this')
#         >>> assert 'this' not in sys.modules,  'this should not have been imported yet'
#         >>> assert this.i == 25
#         >>> assert 'this' in sys.modules,  'this should now be imported'
#         >>> print(this)
#     """
#     def __init__(self, modname):
#         r"""
#         Args:
#             modname (str):  module name
#         """
#         self._modname = modname
#         self._module = None

#     def _load_module(self):
#         if self._module is None:
#             if util_arg.VERBOSE:
#                 print('lazy loading module module')
#             self._module =  __import__(self._modname, globals(), locals(), fromlist=[], level=0)

#     def __str__(self):
#         return 'LazyModule(%s)' % (self._modname,)

#     def __dir__(self):
#         self._load_module()
#         return dir(self._module)

#     def __getattr__(self, item):
#         if item in lazy_module_attrs:
#             return super(LazyModule, self).__getattr__(item)
#         self._load_module()
#         return getattr(self._module, item)

#     def __setattr__(self, item, value):
#         if item in lazy_module_attrs:
#             return super(LazyModule, self).__setattr__(item, value)
#         self._load_module()
#         setattr(self._module, item, value)


def import_module_from_fpath(module_fpath):
    r""" imports module from a file path

    Args:
        module_fpath (str):

    Returns:
        module: module

    CommandLine:
        python -m utool.util_import --test-import_module_from_fpath

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_import import *  # NOQA
        >>> import utool
        >>> module_fpath = utool.__file__
        >>> module = import_module_from_fpath(module_fpath)
        >>> result = ('module = %s' % (str(module),))
        >>> print(result)


    Ignore:
        import shutil
        import ubelt as ub
        test_root = ub.ensure_app_cache_dir('test_fpath_import')
        # Clear the directory
        shutil.rmtree(test_root)
        test_root = ub.ensure_app_cache_dir('test_fpath_import')

        # -----
        # Define two temporary modules with the same name that are not in sys.path
        import sys, os, os.path
        from os.path import join

        # Even though they have the same name they have different values
        mod1_fpath = ub.ensuredir((test_root, 'path1', 'testmod'))
        ub.writeto(join(mod1_fpath, '__init__.py'), 'version = 1\nfrom . import sibling\na1 = 1')
        ub.writeto(join(mod1_fpath, 'sibling.py'), 'spam = \"ham\"\nb1 = 2')

        # Even though they have the same name they have different values
        mod2_fpath = ub.ensuredir((test_root, 'path2', 'testmod'))
        ub.writeto(join(mod2_fpath, '__init__.py'), 'version = 2\nfrom . import sibling\na2 = 3')
        ub.writeto(join(mod2_fpath, 'sibling.py'), 'spam = \"jam\"\nb2 = 4')

        # -----
        # Neither module should be importable through the normal mechanism
        try:
            import testmod
            assert False, 'should fail'
        except ImportError as ex:
            pass

        mod1 = ut.import_module_from_fpath(mod1_fpath)
        print('mod1.version = {!r}'.format(mod1.version))
        print('mod1.version = {!r}'.format(mod1.version))
        print(mod1.version == 1, 'mod1 version is 1')
        print('mod1.a1 = {!r}'.format(mod1.a1))

        mod2 = ut.import_module_from_fpath(mod2_fpath)
        print('mod2.version = {!r}'.format(mod2.version))
        print(mod2.version == 2, 'mod2 version is 2')
        print('mod2.a2 = {!r}'.format(mod1.a2))

        # BUT Notice how mod1 is mod2
        print(mod1 is mod2)

        # mod1 has attributes from mod1 and mod2
        print('mod1.a1 = {!r}'.format(mod1.a1))
        print('mod1.a2 = {!r}'.format(mod1.a2))
        print('mod2.a1 = {!r}'.format(mod2.a1))
        print('mod2.a2 = {!r}'.format(mod2.a2))

        # Both are version 2
        print('mod1.version = {!r}'.format(mod1.version))
        print('mod2.version = {!r}'.format(mod2.version))

        # However sibling always remains at version1 (ham)
        print('mod2.sibling.spam = {!r}'.format(mod2.sibling.spam))

        # now importing testmod works because it reads from sys.modules
        import testmod

        # reloading mod1 overwrites attrs again
        mod1 = ut.import_module_from_fpath(mod1_fpath)

        # Removing both from sys.modules
        del sys.modules['testmod']
        del sys.modules['testmod.sibling']
        mod2 = ut.import_module_from_fpath(mod2_fpath)

        print(not hasattr(mod2, 'a1'),
            'mod2 no longer has a1 and it reloads itself correctly')


        # -------

        del sys.modules['testmod']
        del sys.modules['testmod.sibling']
        mod1 = ut.import_module_from_fpath(mod1_fpath)


        # third test
        mod3_fpath = ub.ensuredir((test_root, 'path3', 'testmod'))
        ub.writeto(join(mod3_fpath, '__init__.py'), 'version = 3')

        module_fpath = mod3_fpath
        modname = 'testmod'

        # third test
        mod4_fpath = ub.ensuredir((test_root, 'path3', 'novelmod'))
        ub.writeto(join(mod4_fpath, '__init__.py'), 'version = 4')

    """
    from os.path import basename, splitext, isdir, join, exists, dirname, split
    import platform
    if isdir(module_fpath):
        module_fpath = join(module_fpath, '__init__.py')
    print('module_fpath = {!r}'.format(module_fpath))
    if not exists(module_fpath):
        raise ImportError('module_fpath={!r} does not exist'.format(
            module_fpath))
    python_version = platform.python_version()
    modname = splitext(basename(module_fpath))[0]
    if modname == '__init__':
        modname = split(dirname(module_fpath))[1]
    if util_inject.PRINT_INJECT_ORDER:
        if modname not in sys.argv:
            util_inject.noinject(modname, N=2, via='ut.import_module_from_fpath')
    if python_version.startswith('2.7'):
        import imp
        module = imp.load_source(modname, module_fpath)
    elif python_version.startswith('3'):
        import importlib.machinery
        loader = importlib.machinery.SourceFileLoader(modname, module_fpath)
        module = loader.load_module()
        # module = loader.exec_module(modname)
    else:
        raise AssertionError('invalid python version={!r}'.format(
            python_version))
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
