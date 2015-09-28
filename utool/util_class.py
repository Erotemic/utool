"""
In this module:
    * a metaclass allowing for reloading of single class instances
    * functions to autoinject methods into a class upon instance creation.
    * A wrapper class allowing an object's properties to be used as kwargs
    * a metaclass to forward properties to another class

    ReloadingMetaclass
    KwargsWrapper
"""
from __future__ import absolute_import, division, print_function
import sys
import six
import types
import functools
import collections
from collections import defaultdict
from utool.util_inject import inject
from utool.util_set import oset
from utool import util_arg
from utool._internal.meta_util_six import get_funcname
print, print_, printDBG, rrr, profile = inject(__name__, '[class]', DEBUG=False)


# Registers which classes have which attributes
# FIXME: this might cause memory leaks
# FIXME: this does cause weird reimport behavior
__CLASSTYPE_ATTRIBUTES__ = defaultdict(oset)
__CLASSTYPE_POSTINJECT_FUNCS__ = defaultdict(oset)
__CLASSNAME_CLASSKEY_REGISTER__ = defaultdict(oset)


#_rrr = rrr
#def rrr(verbose=True):
#    """ keep registered functions through reloads ? """
#    global __CLASSTYPE_ATTRIBUTES__
#    global __CLASSTYPE_POSTINJECT_FUNCS__
#    cta = __CLASSTYPE_ATTRIBUTES__.copy()
#    ctpif = __CLASSTYPE_POSTINJECT_FUNCS__.copy()
#    rrr_(verbose=verbose)
#    __CLASSTYPE_ATTRIBUTES__ = cta
#    __CLASSTYPE_POSTINJECT_FUNCS__ = ctpif


QUIET_CLASS = util_arg.get_argflag(('--quiet-class', '--quietclass'))
VERBOSE_CLASS = util_arg.get_argflag(('--verbose-class', '--verbclass')) or (not QUIET_CLASS and util_arg.VERYVERBOSE)


def inject_instance(self, classkey=None, allow_override=False,
                    verbose=VERBOSE_CLASS, strict=True):
    """
    Injects an instance (self) of type (classkey)
    with all functions registered to (classkey)

    call this in the __init__ class function

    Args:
        self: the class instance
        classkey: key for a class, preferably the class type itself, but it
            doesnt have to be

    SeeAlso:
        make_class_method_decorator

    Example:
        >>> # DOCTEST_DISABLE
        >>> utool.make_class_method_decorator(InvertedIndex)(smk_debug.invindex_dbgstr)
        >>> utool.inject_instance(invindex)
    """
    import utool as ut
    if verbose:
        print('[util_class] begin inject_instance')
    try:
        if classkey is None:
            # Probably should depricate this block of code
            # It tries to do too much
            classkey = self.__class__
            if classkey == 'ibeis.gui.models_and_views.IBEISTableView':
                # HACK HACK HACK
                from guitool.__PYQT__ import QtGui
                classkey = QtGui.QAbstractItemView
            if len(__CLASSTYPE_ATTRIBUTES__[classkey]) == 0:
                print('[utool] Warning: no classes of type %r are registered' % (classkey,))
                print('[utool] type(self)=%r, self=%r' % (type(self), self)),
                print('[utool] Checking to see if anybody else was registered...')
                print('[utool] __CLASSTYPE_ATTRIBUTES__ = ' + ut.list_str(__CLASSTYPE_ATTRIBUTES__.keys()))
                for classtype_, _ in six.iteritems(__CLASSTYPE_ATTRIBUTES__):
                    isinstance(self, classtype_)
                    classkey = classtype_
                    print('[utool] Warning: using subclass=%r' % (classtype_,))
                    break
        func_list = __CLASSTYPE_ATTRIBUTES__[classkey]
        if verbose:
            print('[util_class] injecting %d methods\n   with classkey=%r\n   into %r' % (len(func_list), classkey, self,))
        for func in func_list:
            if VERBOSE_CLASS:
                print('[util_class] * injecting %r' % (func,))
            method_name = None
            # Allow user to register tuples for aliases
            if isinstance(func, tuple):
                func, method_name = func
            inject_func_as_method(self, func, method_name=method_name, allow_override=allow_override)
    except Exception as ex:
        ut.printex(ex, 'ISSUE WHEN INJECTING %r' % (classkey,),
                      iswarning=not strict)
        if strict:
            raise


def postinject_instance(self, classkey, verbose=VERBOSE_CLASS):
    if verbose:
        print('[util_class] Running postinject functions on %r' % (self,))
    for func in __CLASSTYPE_POSTINJECT_FUNCS__[classkey]:
        func(self)
    if verbose:
        print('[util_class] Finished injecting instance self=%r' % (self,))


def inject_all_external_modules(self, classname,
                                allow_override='override+warn',
                                strict=True):
    """
    dynamically injects registered module methods into a class instance

    FIXME: naming convention and use this in all places where this clas is used
    """
    #import utool as ut
    injected_modules = get_injected_modules(classname)
    for module in injected_modules:
        #print(module)
        #ut.embed()
        inject_instance(
            self, classkey=module.CLASS_INJECT_KEY,
            allow_override=allow_override, strict=False)

    for module in injected_modules:
        postinject_instance(
            self, classkey=module.CLASS_INJECT_KEY)


def reload_injected_modules(classname):
    injected_modules = get_injected_modules(classname)
    for module in injected_modules:
        module.rrr()


def get_injected_modules(classname):
    r"""
    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_class import __CLASSNAME_CLASSKEY_REGISTER__  # NOQA
    """
    modname_list = __CLASSNAME_CLASSKEY_REGISTER__[classname]

    injected_modules = []
    for modstr in modname_list:
        parts = modstr.split('.')
        pkgname = '.'.join(parts[:-1])
        modname = parts[-1]
        try:
            exec('from %s import %s' % (pkgname, modname, ), globals(), locals())
            module = eval(modname)
            injected_modules.append(module)
        except ImportError as ex:
            ut.printex(ex, 'Cannot load package=%r, module=%r' % (pkgname, modname, ))
    return injected_modules


def autogen_import_list(classname):
    line_list = []
    for modname in __CLASSNAME_CLASSKEY_REGISTER__[classname]:
        parts = modname.split('.')
        frompart = '.'.join(parts[:-1])
        imppart = parts[-1]
        line = 'from %s import %s  # NOQA' % (frompart, imppart)
        line_list.append(line)
    src = '\n'.join(line_list)
    return src


def autogen_explicit_injectable_metaclass(classname):
    r"""
    Args:
        classname (?):

    Returns:
        ?:

    CommandLine:
        python -m utool.util_class --exec-autogen_explicit_injectable_metaclass

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_class import *  # NOQA
        >>> from utool.util_class import  __CLASSTYPE_ATTRIBUTES__  # NOQA
        >>> import ibeis
        >>> import ibeis.control.IBEISControl
        >>> classname = ibeis.control.controller_inject.CONTROLLER_CLASSNAME
        >>> result = autogen_explicit_injectable_metaclass(classname)
        >>> print(result)
    """
    import utool as ut
    vals_list = []

    def make_redirect(func):
        # PRESERVES ALL SIGNATURES WITH EXECS
        src_fmt = r'''
        def {funcname}{defsig}:
            """ {orig_docstr}"""
            return {orig_funcname}{callsig}
        '''
        from utool._internal import meta_util_six
        orig_docstr = meta_util_six.get_funcdoc(func)
        funcname = meta_util_six.get_funcname(func)
        orig_funcname = modname.split('.')[-1] + '.' + funcname
        orig_docstr = '' if orig_docstr is None else orig_docstr
        import textwrap
        # Put wrapped function into a scope
        import inspect
        argspec = inspect.getargspec(func)
        (args, varargs, varkw, defaults) = argspec
        defsig = inspect.formatargspec(*argspec)
        callsig = inspect.formatargspec(*argspec[0:3])
        src_fmtdict = dict(funcname=funcname, orig_funcname=orig_funcname,
                           defsig=defsig, callsig=callsig,
                           orig_docstr=orig_docstr)
        src = textwrap.dedent(src_fmt).format(**src_fmtdict)
        return src

    src_list = []

    for classkey, vals in __CLASSTYPE_ATTRIBUTES__.items():
        modname = classkey[1]
        if classkey[0] == classname:
            vals_list.append(vals)
            for func in vals:
                src = make_redirect(func)
                src = ut.indent(src)
                src = '\n'.join([_.rstrip() for _ in src.split('\n')])
                src_list.append(src)

    source_block = autogen_import_list(classname) + '\n\n\n' + 'class ExplicitInject' + classname + '(object):\n' + ''.join(src_list)
    return source_block


def make_class_method_decorator(classkey, modname=None):
    """
    register a class to be injectable
    classkey is a key that identifies the injected class
    REMEMBER to call inject_instance in __init__

    Args:
        classkey : the class to be injected into
        modname : the global __name__ of the module youa re injecting from

    Returns:
        closure_decorate_class_method (func): decorator for injectable methods

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> class CheeseShop(object):
        ...    def __init__(self):
        ...        import utool as ut
        ...        ut.inject_instance(self)
        >>> cheeseshop_method = ut.make_class_method_decorator(CheeseShop)
        >>> @cheeseshop_method
        >>> def has_cheese(self):
        >>>     return False
        >>> shop = CheeseShop()
        >>> assert shop.has_cheese() is False
    """
    global __APP_MODNAME_REGISTER__
    if util_arg.VERBOSE or VERBOSE_CLASS:
        print('[util_class] register make_class_method_decorator classkey=%r, modname=%r'
              % (classkey, modname))
    if modname == '__main__':
        # skips reinjects into main
        print('WARNING: cannot register class functions as __main__')
        return lambda func: func
    # register that this module was injected into
    if isinstance(classkey, tuple):
        classname, _ = classkey
        __CLASSNAME_CLASSKEY_REGISTER__[classname].append(modname)
    else:
        print('Warning not using classkey for %r %r' % (classkey, modname))
        raise AssertionError('classkey no longer supported. Use class_inject_key instead')
    closure_decorate_class_method = functools.partial(decorate_class_method, classkey=classkey)
    return closure_decorate_class_method


def make_class_postinject_decorator(classkey, modname=None):
    """
    Args:
        classkey : the class to be injected into
        modname : the global __name__ of the module youa re injecting from

    Returns:
        closure_decorate_postinject (func): decorator for injectable methods

    SeeAlso:
        make_class_method_decorator
    """
    if util_arg.VERBOSE or VERBOSE_CLASS:
        print('[util_class] register class_postinject classkey=%r, modname=%r'
              % (classkey, modname))
    if modname == '__main__':
        print('WARNING: cannot register class functions as __main__')
        # skips reinjects into main
        return lambda func: func
    closure_decorate_postinject = functools.partial(decorate_postinject, classkey=classkey)
    return closure_decorate_postinject


def decorate_class_method(func, classkey=None, skipmain=False):
    """
    Will inject all decorated function as methods of classkey

    classkey is some identifying string, tuple, or object

    func can also be a tuple
    """
    #import utool as ut
    global __CLASSTYPE_ATTRIBUTES__
    assert classkey is not None, 'must specify classkey'
    #if not (skipmain and ut.get_caller_modname() == '__main__'):
    __CLASSTYPE_ATTRIBUTES__[classkey].append(func)
    return func


def decorate_postinject(func, classkey=None, skipmain=False):
    """
    Will perform func with argument self after inject_instance is called on classkey

    classkey is some identifying string, tuple, or object
    """
    #import utool as ut
    global __CLASSTYPE_POSTINJECT_FUNCS__
    assert classkey is not None, 'must specify classkey'
    #if not (skipmain and ut.get_caller_modname() == '__main__'):
    __CLASSTYPE_POSTINJECT_FUNCS__[classkey].append(func)
    return func


def inject_func_as_method(self, func, method_name=None, class_=None, allow_override=False, allow_main=False):
    """ Injects a function into an object as a method

    Wraps func as a bound method of self. Then injects func into self
    It is preferable to use make_class_method_decorator and inject_instance

    Args:
       self (object): class instance
       func : some function whos first arugment is a class instance
       method_name (str) : default=func.__name__, if specified renames the method
       class_ (type) : if func is an unbound method of this class


    References:
        http://stackoverflow.com/questions/1015307/python-bind-an-unbound-method
    """
    if method_name is None:
        method_name = get_funcname(func)
    #printDBG('Injecting method_name=%r' % method_name)
    old_method = getattr(self, method_name, None)
    #import utool as ut
    #ut.embed()

    # Bind function to the class instance
    #new_method = types.MethodType(func, self, self.__class__)
    new_method = func.__get__(self, self.__class__)
    #new_method = profile(func.__get__(self, self.__class__))

    if old_method is not None:
        if not allow_main and (
                old_method.im_func.func_globals['__name__'] != '__main__' and
                new_method.im_func.func_globals['__name__'] == '__main__'):
            if True or VERBOSE_CLASS:
                print('[util_class] skipping re-inject of %r from __main__' % method_name)
            return
        if old_method is new_method or old_method.im_func is new_method.im_func:
            print('WARNING: Injecting the same function twice: %r' % new_method)
        elif allow_override is False:
            raise AssertionError('Overrides are not allowed. Already have method_name=%r' % (method_name))
        elif allow_override == 'warn':
            print('WARNING: Overrides are not allowed. Already have method_name=%r. Skipping' % (method_name))
            return
        elif allow_override == 'override+warn':
            #import utool as ut
            #ut.embed()
            print('WARNING: Overrides are allowed, but dangerous. method_name=%r.' % (method_name))
            print('old_method = %r, im_func=%s' % (old_method, str(old_method.im_func)))
            print('new_method = %r, im_func=%s' % (new_method, str(new_method.im_func)))
            print(old_method.im_func.func_globals['__name__'])
            print(new_method.im_func.func_globals['__name__'])
        # TODO: does this actually decrement the refcount enough?
        del old_method
    setattr(self, method_name, new_method)


def makeForwardingMetaclass(forwarding_dest_getter, whitelist, base_class=object):
    """
    makes a metaclass that overrides __getattr__ and __setattr__ to forward some
    specific attribute references to a specified instance variable
    """
    class ForwardingMetaclass(base_class.__class__):
        def __init__(metaself, name, bases, dct):
            # print('ForwardingMetaclass.__init__():
            #  {forwarding_dest_getter: %r; whitelist: %r}' % (forwarding_dest_getter, whitelist))
            super(ForwardingMetaclass, metaself).__init__(name, bases, dct)
            old_getattr = metaself.__getattribute__
            old_setattr = metaself.__setattr__
            def new_getattr(self, item):
                if item in whitelist:
                    #dest = old_getattr(self, forwarding_dest_name)
                    dest = forwarding_dest_getter(self)
                    try:
                        val = dest.__class__.__getattribute__(dest, item)
                    except AttributeError:
                        val = getattr(dest, item)
                else:
                    val = old_getattr(self, item)
                return val
            def new_setattr(self, name, val):
                if name in whitelist:
                    #dest = old_getattr(self, forwarding_dest_name)
                    dest = forwarding_dest_getter(self)
                    dest.__class__.__setattr__(dest, name, val)
                else:
                    old_setattr(self, name, val)
            metaself.__getattribute__ = new_getattr
            metaself.__setattr__ = new_setattr
    return ForwardingMetaclass


def test_reloading_metaclass():
    r"""
    CommandLine:
        python -m utool.util_class --test-test_reloading_metaclass

    References:
        http://stackoverflow.com/questions/8122734/pythons-imp-reload-function-is-not-working

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_class import *  # NOQA
        >>> result = test_reloading_metaclass()
        >>> print(result)
    """
    import utool as ut
    testdir = ut.ensure_app_resource_dir('utool', 'metaclass_tests')
    testfoo_fpath = ut.unixjoin(testdir, 'testfoo.py')
    # os.chdir(testdir)
    #with ut.ChdirContext(testdir, stay=ut.inIPython()):
    with ut.ChdirContext(testdir):
        foo_code1 = ut.codeblock(
            r'''
            # STARTBLOCK
            import utool as ut
            import six


            @six.add_metaclass(ut.ReloadingMetaclass)
            class Foo(object):
                def __init__(self):
                    pass

            spamattr = 'version1'
            # ENDBLOCK
            '''
        )
        foo_code2 = ut.codeblock(
            r'''
            # STARTBLOCK
            import utool as ut
            import six


            @six.add_metaclass(ut.ReloadingMetaclass)
            class Foo(object):
                def __init__(self):
                    pass

                def bar(self):
                    return 'spam'

            eggsattr = 'version2'
            # ENDBLOCK
            '''
        )
        # Write a testclass to disk
        ut.delete(testfoo_fpath)
        ut.write_to(testfoo_fpath, foo_code1, verbose=True)
        testfoo = ut.import_module_from_fpath(testfoo_fpath)
        #import testfoo
        foo = testfoo.Foo()
        print('foo = %r' % (foo,))
        assert not hasattr(foo, 'bar'), 'foo should not have a bar attr'
        ut.delete(testfoo_fpath + 'c')  # remove the pyc file because of the identical creation time
        ut.write_to(testfoo_fpath, foo_code2, verbose=True)
        assert not hasattr(foo, 'bar'), 'foo should still not have a bar attr'
        foo.rrr()
        assert foo.bar() == 'spam'
        ut.delete(testfoo_fpath)
        print('Reloading worked nicely')


class ReloadingMetaclass(type):
    """
    Classes with this metaclass will be able to reload themselves
    on a per-instance basis using the rrr function.

    If the functions _on_reload and _initialize_self exist
    they will be called after and before reload respectively. Any
    inject_instance functions should be handled there.

    SeeAlso:
        test_reloading_metaclass - shows a working example of this doctest

    Example:
        >>> # DIABLE_DOCTEST
        >>> from utool.util_class import *  # NOQA
        >>> import utool as ut
        >>> @six.add_metaclass(ut.ReloadingMetaclass)
        >>> class Foo(object):
        ...     def __init__(self):
        ...         pass
        >>> # You can edit foo on disk and call rrr in ipython
        >>> # if you add a new function to it
        >>> foo = Foo()
        >>> # This will not work as a doctests because
        >>> # Foo's parent module will be __main__ but
        >>> # there will be no easy way to write to it.
        >>> # This does work when you run from ipython
        >>> @six.add_metaclass(ut.ReloadingMetaclass)
        >>> class Foo(object):
        ...     def __init__(self):
        ...         pass
        ...     def bar(self):
        ...         return 'spam'
        >>> foo.rrr()
        >>> result = foo.bar()
        >>> print(result)
        spam
    """
    def __init__(metaself, name, bases, dct):
        super(ReloadingMetaclass, metaself).__init__(name, bases, dct)
        #print('Making rrr for %r' % (name,))
        rrr = private_rrr_factory()
        #def rrr(self, verbose=True):
        #    classname = self.__class__.__name__
        #    try:
        #        modname = self.__class__.__module__
        #        if verbose:
        #            print('reloading ' + classname + ' from ' + modname)
        #        # --HACK--
        #        if hasattr(self, '_on_reload'):
        #            self._on_reload()

        #        NEW = True
        #        if NEW:
        #            # Do for all inheriting classes
        #            def find_base_clases(_class, find_base_clases=None):
        #                class_list = []
        #                for _baseclass in _class.__bases__:
        #                    class_list.extend(find_base_clases(_baseclass, find_base_clases))
        #                if _class is not object:
        #                    class_list.append(_class)
        #                return class_list

        #            _class = self.__class__
        #            class_list = find_base_clases(_class, find_base_clases)
        #            for _class in class_list:
        #                if verbose:
        #                    print('reloading parent ' + _class.__name__ + ' from ' + _class.__module__)
        #                if _class.__module__ != '__main__':
        #                    module_ = sys.modules[_class.__module__]
        #                    if hasattr(module_, 'rrr'):
        #                        module_.rrr()
        #                    else:
        #                        import imp
        #                        imp.reload(module_)
        #                _newclass = getattr(module_, _class.__name__)
        #                reload_class_methods(self, _newclass)
        #        else:
        #            # --------
        #            # Reload the parent module if it is not main
        #            module = sys.modules[modname]
        #            if modname != '__main__':
        #                if hasattr(module, 'rrr'):
        #                    module.rrr()
        #                else:
        #                    import imp
        #                    imp.reload(module)
        #            # --------
        #            # Reload parent classes (if inherited)
        #            # TODO: figure out how to do this
        #            #for _baseclass in self.__class__.__bases__:
        #            #    if hasattr(_baseclass, 'rrr'):
        #            #        print('Reloading parent: %r' % (_baseclass))
        #            #        # make a bound rrr method that belongs to the parent instance
        #            #        base_rrr = _baseclass.rrr.__get__(self, _baseclass)
        #            #        base_rrr(verbose=verbose)
        #            # Get new class definition
        #            class_ = getattr(module, classname)
        #            reload_class_methods(self, class_)
        #        # --HACK--
        #        # TODO: handle injected definitions
        #        if hasattr(self, '_initialize_self'):
        #            self._initialize_self()
        #    except Exception as ex:
        #        import utool as ut
        #        ut.printex(ex, 'Error Reloading Class', keys=[
        #            'modname',
        #            'module',
        #            'class_',
        #            'self', ])
        #        #ut.embed()
        #        #print(ut.dict_str(module.__dict__))
        #        raise
        metaself.rrr = rrr


def private_rrr_factory():
    def rrr(self, verbose=True):
        classname = self.__class__.__name__
        try:
            modname = self.__class__.__module__
            if verbose:
                print('reloading ' + classname + ' from ' + modname)
            # --HACK--
            if hasattr(self, '_on_reload'):
                self._on_reload()

            NEW = True
            if NEW:
                # Do for all inheriting classes
                def find_base_clases(_class, find_base_clases=None):
                    class_list = []
                    for _baseclass in _class.__bases__:
                        class_list.extend(find_base_clases(_baseclass, find_base_clases))
                    if _class is not object:
                        class_list.append(_class)
                    return class_list

                _class = self.__class__
                class_list = find_base_clases(_class, find_base_clases)
                for _class in class_list:
                    if verbose:
                        print('reloading parent ' + _class.__name__ + ' from ' + _class.__module__)
                    if _class.__module__ != '__main__':
                        module_ = sys.modules[_class.__module__]
                        if hasattr(module_, 'rrr'):
                            module_.rrr()
                        else:
                            import imp
                            print('reloading ' + _class.__module__ + ' with imp')
                            try:
                                imp.reload(module_)
                            except (ImportError, AttributeError):
                                print('fallback reloading ' + _class.__module__ + ' with imp')
                                # one last thing to try. probably used ut.import_module_from_fpath
                                # when importing this module
                                imp.load_source(module_.__name__, module_.__file__)
                    _newclass = getattr(module_, _class.__name__)
                    reload_class_methods(self, _newclass)
            else:
                # --------
                # Reload the parent module if it is not main
                module = sys.modules[modname]
                if modname != '__main__':
                    if hasattr(module, 'rrr'):
                        module.rrr()
                    else:
                        import imp
                        imp.reload(module)
                # --------
                # Reload parent classes (if inherited)
                # TODO: figure out how to do this
                #for _baseclass in self.__class__.__bases__:
                #    if hasattr(_baseclass, 'rrr'):
                #        print('Reloading parent: %r' % (_baseclass))
                #        # make a bound rrr method that belongs to the parent instance
                #        base_rrr = _baseclass.rrr.__get__(self, _baseclass)
                #        base_rrr(verbose=verbose)
                # Get new class definition
                class_ = getattr(module, classname)
                reload_class_methods(self, class_)
            # --HACK--
            # TODO: handle injected definitions
            if hasattr(self, '_initialize_self'):
                self._initialize_self()
        except Exception as ex:
            import utool as ut
            ut.printex(ex, 'Error Reloading Class', keys=[
                'modname',
                'module',
                'class_',
                'self', ])
            #print(ut.dict_str(module.__dict__))
            raise
    return rrr


def reloading_meta_metaclass_factory(BASE_TYPE=type):
    """ hack for pyqt """
    class ReloadingMetaclass2(BASE_TYPE):
        def __init__(metaself, name, bases, dct):
            super(ReloadingMetaclass2, metaself).__init__(name, bases, dct)
            #print('Making rrr for %r' % (name,))
            rrr = private_rrr_factory()
            metaself.rrr = rrr
    return ReloadingMetaclass2


def reload_class_methods(self, class_):
    """
    rebinds all class methods

    Args:
        self (object): class instance to reload
        class_ (type): type to reload as

    Example:
        >>> from utool.util_class import *  # NOQA
        >>> self = '?'
        >>> class_ = '?'
        >>> result = reload_class_methods(self, class_)
        >>> print(result)
    """
    print('[util_class] Reloading self=%r as class_=%r' % (self, class_))
    self.__class__ = class_
    for key in dir(class_):
        # Get unbound reloaded method
        func = getattr(class_, key)
        if isinstance(func, types.MethodType):
            # inject it into the old instance
            inject_func_as_method(self, func, class_=class_, allow_override=True)


def get_comparison_methods():
    """ makes methods for >, <, =, etc... """
    method_list = []
    def _register(func):
        method_list.append(func)
        return func

    # Comparison operators for sorting and uniqueness
    @_register
    def __lt__(self, other):
        return self.__hash__() < (other.__hash__())

    @_register
    def __le__(self, other):
        return self.__hash__() <= (other.__hash__())

    @_register
    def __eq__(self, other):
        return self.__hash__() == (other.__hash__())

    @_register
    def __ne__(self, other):
        return self.__hash__() != (other.__hash__())

    @_register
    def __gt__(self, other):
        return self.__hash__() > (other.__hash__())

    @_register
    def __ge__(self, other):
        return self.__hash__() >= (other.__hash__())

    return method_list


class KwargsWrapper(collections.Mapping):
    """
    Allows an arbitrary object attributes to be passed as a **kwargs
    argument
    """
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        return self.obj.__dict__[key]

    def __iter__(self):
        return iter(self.obj.__dict__)

    def __len__(self):
        return len(self.obj.__dict__)


def remove_private_obfuscation(self):
    """
    removes the python obfuscation of class privates so they can be executed as
    they appear in class source. Useful when playing with IPython.
    """
    classname = self.__class__.__name__
    attrlist = [attr for attr in dir(self) if attr.startswith('_' + classname + '__')]
    for attr in attrlist:
        method = getattr(self, attr)
        truename = attr.replace('_' + classname + '__', '__')
        setattr(self, truename, method)


if __name__ == '__main__':
    """
    python -c "import utool; utool.doctest_funcs(utool.util_class, allexamples=True)"
    python -m utool.util_class --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut
    ut.doctest_funcs()
