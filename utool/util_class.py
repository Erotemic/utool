# -*- coding: utf-8 -*-
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
import operator as op
try:
    from collections.abc import Mapping
except Exception:
    from collections import Mapping
from collections import defaultdict
from utool import util_inject
from utool import util_set
from utool import util_arg
from utool._internal.meta_util_six import get_funcname, get_funcglobals
print, rrr, profile = util_inject.inject2(__name__)


# Registers which classes have which attributes
# FIXME: this might cause memory leaks
# FIXME: this does cause weird reimport behavior
__CLASSTYPE_ATTRIBUTES__ = defaultdict(util_set.oset)
__CLASSTYPE_POSTINJECT_FUNCS__ = defaultdict(util_set.oset)
__CLASSNAME_CLASSKEY_REGISTER__ = defaultdict(util_set.oset)


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
VERBOSE_CLASS = (
    util_arg.get_argflag(('--verbose-class', '--verbclass')) or
    (not QUIET_CLASS and util_arg.VERYVERBOSE))


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
        >>> # DISABLE_DOCTEST
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
                try:
                    from guitool_ibeis.__PYQT__ import QtWidgets  # NOQA
                except ImportError:
                    from guitool.__PYQT__ import QtWidgets  # NOQA
                classkey = QtWidgets.QAbstractItemView
            if len(__CLASSTYPE_ATTRIBUTES__[classkey]) == 0:
                print('[utool] Warning: no classes of type %r are registered' % (classkey,))
                print('[utool] type(self)=%r, self=%r' % (type(self), self)),
                print('[utool] Checking to see if anybody else was registered...')
                print('[utool] __CLASSTYPE_ATTRIBUTES__ = ' +
                      ut.repr4(__CLASSTYPE_ATTRIBUTES__.keys()))
                for classtype_, _ in six.iteritems(__CLASSTYPE_ATTRIBUTES__):
                    isinstance(self, classtype_)
                    classkey = classtype_
                    print('[utool] Warning: using subclass=%r' % (classtype_,))
                    break
        func_list = __CLASSTYPE_ATTRIBUTES__[classkey]
        if verbose:
            print('[util_class] injecting %d methods\n   with classkey=%r\n   into %r'
                  % (len(func_list), classkey, self,))
        for func in func_list:
            if VERBOSE_CLASS:
                print('[util_class] * injecting %r' % (func,))
            method_name = None
            # Allow user to register tuples for aliases
            if isinstance(func, tuple):
                func, method_name = func
            inject_func_as_method(self, func, method_name=method_name,
                                  allow_override=allow_override, verbose=verbose)
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


def inject_all_external_modules(self, classname=None,
                                allow_override='override+warn',
                                strict=True):
    """
    dynamically injects registered module methods into a class instance

    FIXME: naming convention and use this in all places where this clas is used
    """
    #import utool as ut
    if classname is None:
        classname = self.__class__.__name__
    #import utool as ut
    #ut.embed()

    NEW = True
    if NEW:
        classkey_list = [key for key in __CLASSTYPE_ATTRIBUTES__
                         if key[0] == classname]
    else:
        injected_modules = get_injected_modules(classname)
        # the variable must be named CLASS_INJECT_KEY
        # and only one class can be specified per module.
        classkey_list = [module.CLASS_INJECT_KEY
                         for module in injected_modules]

    for classkey in classkey_list:
        inject_instance(
            self, classkey=classkey,
            allow_override=allow_override, strict=False)

    for classkey in classkey_list:
        postinject_instance(
            self, classkey=classkey)


def reload_injected_modules(classname):
    injected_modules = get_injected_modules(classname)
    for module in injected_modules:
        if hasattr(module, 'rrr'):
            module.rrr()
        else:
            import imp
            print('rrr not defined in module=%r' % (module,))
            imp.reload(module)


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


def autogen_import_list(classname, conditional_imports=None):
    import utool as ut
    #ut.embed()
    #line_list = []
    line_list = ['import sys  # NOQA']
    for modname in __CLASSNAME_CLASSKEY_REGISTER__[classname]:
        # <super hacky>
        condition = None
        for x in conditional_imports:
            if modname == x[1]:
                condition = x[0]
        # </super hacky>
        parts = modname.split('.')
        frompart = '.'.join(parts[:-1])
        imppart = parts[-1]
        #line = 'from %s import %s  # NOQA' % (frompart, imppart)
        if condition is None:
            line = 'from %s import %s' % (frompart, imppart)
        else:
            line = ut.codeblock(
                '''
                if not ut.get_argflag({condition}) or '{frompart}' in sys.modules:
                    from {frompart} import {imppart}
                ''').format(condition=condition, frompart=frompart,
                            imppart=imppart)
        line_list.append(line)
    src = '\n'.join(line_list)
    return src


def autogen_explicit_injectable_metaclass(classname, regen_command=None,
                                          conditional_imports=None):
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

    if regen_command is None:
        regen_command = 'FIXME None given'

    module_header = ut.codeblock(
        """
        # -*- coding: utf-8 -*-
        """ + ut.TRIPLE_DOUBLE_QUOTE + """
        Static file containing autogenerated functions for {classname}
        Autogenerated on {autogen_time}

        RegenCommand:
            {regen_command}
        """ + ut.TRIPLE_DOUBLE_QUOTE + """

        from __future__ import absolute_import, division, print_function
        import utool as ut

        """).format(
            autogen_time=ut.get_timestamp(),
            regen_command=regen_command,
            classname=classname)

    depends_module_block = autogen_import_list(classname, conditional_imports)
    inject_statement_fmt = ("print, rrr, profile = "
                            "ut.inject2(__name__, '[autogen_explicit_inject_{classname}]')")
    inject_statement = inject_statement_fmt.format(classname=classname)

    source_block_lines = [
        module_header,
        depends_module_block,
        inject_statement,
        '\n',
        'class ExplicitInject' + classname + '(object):',
    ] + src_list
    source_block = '\n'.join(source_block_lines)

    source_block = ut.autoformat_pep8(source_block, aggressive=2)
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
        ...     def __init__(self):
        ...         import utool as ut
        ...         ut.inject_all_external_modules(self)
        >>> cheeseshop_method = ut.make_class_method_decorator(CheeseShop)
        >>> shop1 = CheeseShop()
        >>> assert not hasattr(shop1, 'has_cheese'), 'have not injected yet'
        >>> @cheeseshop_method
        >>> def has_cheese(self):
        >>>     return False
        >>> shop2 = CheeseShop()
        >>> assert shop2.has_cheese() is False, 'external method not injected'
        >>> print('Cheese shop does not have cheese. All is well.')
    """
    global __APP_MODNAME_REGISTER__
    #if util_arg.VERBOSE or VERBOSE_CLASS:
    if VERBOSE_CLASS:
        print('[util_class] register via make_class_method_decorator classkey=%r, modname=%r'
              % (classkey, modname))
    if modname == '__main__':
        # skips reinjects into main
        print('WARNING: cannot register classkey=%r functions as __main__' % (classkey,))
        return lambda func: func
    # register that this module was injected into
    if isinstance(classkey, tuple):
        classname, _ = classkey
        __CLASSNAME_CLASSKEY_REGISTER__[classname].append(modname)
    elif isinstance(classkey, type):
        classname = classkey.__name__
        if modname is not None:
            assert modname == classkey.__module__, (
                'modname=%r does not agree with __module__=%r' % (
                    modname, classkey.__module__))
        modname = classkey.__module__
        # Convert to new classkey format
        classkey = (classname, modname)
        __CLASSNAME_CLASSKEY_REGISTER__[classname].append(modname)
    else:
        print('Warning not using classkey for %r %r' % (classkey, modname))
        raise AssertionError('classkey no longer supported. Use class_inject_key instead')
    closure_decorate_class_method = functools.partial(decorate_class_method,
                                                      classkey=classkey)
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
    closure_decorate_postinject = functools.partial(decorate_postinject,
                                                    classkey=classkey)
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


def get_method_func(method):
    try:
        return method.im_func if six.PY2 else method.__func__
    except AttributeError:
        # check if this is a method-wrapper type
        if isinstance(method, type(all.__call__)):
            # in which case there is no underlying function
            return None
        raise


def inject_func_as_method(self, func, method_name=None, class_=None,
                          allow_override=False, allow_main=False,
                          verbose=True, override=None, force=False):
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
    if override is not None:
        # TODO depcirate allow_override
        allow_override = override
    if method_name is None:
        method_name = get_funcname(func)
    if force:
        allow_override = True
        allow_main = True
    old_method = getattr(self, method_name, None)
    # Bind function to the class instance
    #new_method = types.MethodType(func, self, self.__class__)
    new_method = func.__get__(self, self.__class__)
    #new_method = profile(func.__get__(self, self.__class__))

    if old_method is not None:
        old_im_func = get_method_func(old_method)
        new_im_func = get_method_func(new_method)
        if not allow_main and old_im_func is not None and (
                get_funcglobals(old_im_func)['__name__'] != '__main__' and
                get_funcglobals(new_im_func)['__name__'] == '__main__'):
            if True or VERBOSE_CLASS:
                print('[util_class] skipping re-inject of %r from __main__' % method_name)
            return
        if old_method is new_method or old_im_func is new_im_func:
            #if verbose and util_arg.NOT_QUIET:
            #    print('WARNING: Skipping injecting the same function twice: %r' % new_method)
                #print('WARNING: Injecting the same function twice: %r' % new_method)
            return
        elif allow_override is False:
            raise AssertionError(
                'Overrides are not allowed. Already have method_name=%r' %
                (method_name))
        elif allow_override == 'warn':
            print(
                'WARNING: Overrides are not allowed. Already have method_name=%r. Skipping' %
                (method_name))
            return
        elif allow_override == 'override+warn':
            #import utool as ut
            #ut.embed()
            print('WARNING: Overrides are allowed, but dangerous. method_name=%r.' %
                  (method_name))
            print('old_method = %r, im_func=%s' % (old_method, str(old_im_func)))
            print('new_method = %r, im_func=%s' % (new_method, str(new_im_func)))
            print(get_funcglobals(old_im_func)['__name__'])
            print(get_funcglobals(new_im_func)['__name__'])
        # TODO: does this actually decrement the refcount enough?
        del old_method
    setattr(self, method_name, new_method)


def inject_func_as_property(self, func, method_name=None, class_=None):
    """
    WARNING:
        properties are more safely injected using metaclasses

    References:
        http://stackoverflow.com/questions/13850114/dynamically-adding-methods-with-or-without-metaclass-in-python
    """
    if method_name is None:
        method_name = get_funcname(func)
    #new_method = func.__get__(self, self.__class__)
    new_property = property(func)
    setattr(self.__class__, method_name, new_property)


def inject_func_as_unbound_method(class_, func, method_name=None):
    """ This is actually quite simple """
    if method_name is None:
        method_name = get_funcname(func)
    setattr(class_, method_name, func)


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
    testdir = ut.ensure_app_cache_dir('utool', 'metaclass_tests')
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
        >>> # DISABLE_DOCTEST
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
        metaself.rrr = reload_class


def reloading_meta_metaclass_factory(BASE_TYPE=type):
    """ hack for pyqt """
    class ReloadingMetaclass2(BASE_TYPE):
        def __init__(metaself, name, bases, dct):
            super(ReloadingMetaclass2, metaself).__init__(name, bases, dct)
            #print('Making rrr for %r' % (name,))
            metaself.rrr = reload_class
    return ReloadingMetaclass2


def reload_class(self, verbose=True, reload_module=True):
    """
    special class reloading function
    This function is often injected as rrr of classes
    """
    import utool as ut
    verbose = verbose or VERBOSE_CLASS
    classname = self.__class__.__name__
    try:
        modname = self.__class__.__module__
        if verbose:
            print('[class] reloading ' + classname + ' from ' + modname)
        # --HACK--
        if hasattr(self, '_on_reload'):
            if verbose > 1:
                print('[class] calling _on_reload for ' + classname)
            self._on_reload()
        elif verbose > 1:
            print('[class] ' + classname + ' does not have an _on_reload function')

        # Do for all inheriting classes
        def find_base_clases(_class, find_base_clases=None):
            class_list = []
            for _baseclass in _class.__bases__:
                parents = find_base_clases(_baseclass, find_base_clases)
                class_list.extend(parents)
            if _class is not object:
                class_list.append(_class)
            return class_list

        head_class = self.__class__
        # Determine if parents need reloading
        class_list = find_base_clases(head_class, find_base_clases)
        # HACK
        ignore = {HashComparable2}
        class_list = [_class for _class in class_list
                      if _class not in ignore]
        for _class in class_list:
            if verbose:
                print('[class] reloading parent ' + _class.__name__ +
                      ' from ' + _class.__module__)
            if _class.__module__ == '__main__':
                # Attempt to find the module that is the main module
                # This may be very hacky and potentially break
                main_module_ = sys.modules[_class.__module__]
                main_modname = ut.get_modname_from_modpath(main_module_.__file__)
                module_ = sys.modules[main_modname]
            else:
                module_ = sys.modules[_class.__module__]
            if hasattr(module_, 'rrr'):
                if reload_module:
                    module_.rrr(verbose=verbose)
            else:
                if reload_module:
                    import imp
                    if verbose:
                        print('[class] reloading ' + _class.__module__ + ' with imp')
                    try:
                        imp.reload(module_)
                    except (ImportError, AttributeError):
                        print('[class] fallback reloading ' + _class.__module__ +
                              ' with imp')
                        # one last thing to try. probably used ut.import_module_from_fpath
                        # when importing this module
                        imp.load_source(module_.__name__, module_.__file__)
            # Reset class attributes
            _newclass = getattr(module_, _class.__name__)
            reload_class_methods(self, _newclass, verbose=verbose)

        # --HACK--
        # TODO: handle injected definitions
        if hasattr(self, '_initialize_self'):
            if verbose > 1:
                print('[class] calling _initialize_self for ' + classname)
            self._initialize_self()
        elif verbose > 1:
            print('[class] ' + classname + ' does not have an _initialize_self function')
    except Exception as ex:
        ut.printex(ex, 'Error Reloading Class', keys=[
            'modname', 'module', 'class_', 'class_list', 'self', ])
        raise


def reload_class_methods(self, class_, verbose=True):
    """
    rebinds all class methods

    Args:
        self (object): class instance to reload
        class_ (type): type to reload as

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_class import *  # NOQA
        >>> self = '?'
        >>> class_ = '?'
        >>> result = reload_class_methods(self, class_)
        >>> print(result)
    """
    if verbose:
        print('[util_class] Reloading self=%r as class_=%r' % (self, class_))
    self.__class__ = class_
    for key in dir(class_):
        # Get unbound reloaded method
        func = getattr(class_, key)
        if isinstance(func, types.MethodType):
            # inject it into the old instance
            inject_func_as_method(self, func, class_=class_,
                                  allow_override=True,
                                  verbose=verbose)


def compare_instance(op, self, other):
    hash1 = self.__hash__()
    if other.__hash__ is not None:
        hash2 = hash(other)
    else:
        hash2 = other
    try:
        return op(hash1, hash2)
    except Exception as ex:
        import utool as ut
        ut.printex(ex, 'could not compare hash1 to hash2', keys=['hash1', 'hash2'])
        raise


def get_comparison_methods():
    """ makes methods for >, <, =, etc... """
    method_list = []
    def _register(func):
        method_list.append(func)
        return func

    # Comparison operators for sorting and uniqueness
    @_register
    def __lt__(self, other):
        return compare_instance(op.lt, self, other)

    @_register
    def __le__(self, other):
        return compare_instance(op.le, self, other)

    @_register
    def __eq__(self, other):
        return compare_instance(op.eq, self, other)

    @_register
    def __ne__(self, other):
        return compare_instance(op.ne, self, other)

    @_register
    def __gt__(self, other):
        return compare_instance(op.gt, self, other)

    @_register
    def __ge__(self, other):
        return compare_instance(op.ge, self, other)

    return method_list


def get_comparison_operators():
    import operator
    opdict = {
        'not': operator.not_,
        'is': operator.is_,
        'is not': operator.is_not,
        'contains': operator.contains,
        'in': lambda a, b: operator.contains(b, a),
        'not in': lambda a, b: not operator.contains(b, a),
        '!=': operator.ne,
        '==': operator.eq,
        '>=': operator.ge,
        '<=': operator.le,
        '>': operator.gt,
        '<': operator.lt,
    }
    for k, v in list(opdict.items()):
        op = getattr(v, '__name__')
        if op not in opdict and op != '<lambda>':
            opdict[op] = v
    return opdict


class HashComparableMetaclass(type):
    """
    Defines extra methods for Configs

    FIXME: this breaks in python3 because
    anything that overwrites hash overwrites inherited __eq__

    https://docs.python.org/3.6/reference/datamodel.html#object.__hash__
    """
    def __new__(cls, name, bases, dct):
        """
        Args:
            cls (type): meta
            name (str): classname
            supers (list): bases
            dct (dict): class dictionary
        """
        method_list = get_comparison_methods()
        for func in method_list:
            if get_funcname(func) not in dct:
                funcname = get_funcname(func)
                dct[funcname] = func
            else:
                funcname = get_funcname(func)
                dct['meta_' + funcname] = func
            #ut.inject_func_as_method(metaself, func)
        return type.__new__(cls, name, bases, dct)


@six.add_metaclass(HashComparableMetaclass)
class HashComparable(object):
    pass


class HashComparable2(object):
    def __lt__(self, other):
        return compare_instance(op.lt, self, other)

    def __le__(self, other):
        return compare_instance(op.le, self, other)

    def __eq__(self, other):
        return compare_instance(op.eq, self, other)

    def __ne__(self, other):
        return compare_instance(op.ne, self, other)

    def __gt__(self, other):
        return compare_instance(op.gt, self, other)

    def __ge__(self, other):
        return compare_instance(op.ge, self, other)


def reloadable_class(cls):
    """
    convinience decorator instead of @six.add_metaclass(ReloadingMetaclass)
    """
    return six.add_metaclass(ReloadingMetaclass)(cls)


class KwargsWrapper(Mapping):
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


def get_classname(class_, local=False):
    r"""
    Args:
        class_ (type):
        local (bool): (default = False)

    Returns:
        str: classname

    CommandLine:
        python -m utool.util_class --exec-get_classname --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_class import *  # NOQA
        >>> import utool as ut
        >>> class_ = ReloadingMetaclass
        >>> local = False
        >>> assert get_classname(class_, local) == 'utool.util_class.ReloadingMetaclass'
        >>> assert get_classname(class_, local=True) == 'ReloadingMetaclass'
    """
    if not local:
        classname = class_.__module__ + '.' + class_.__name__
    else:
        classname = class_.__name__
    return classname


if __name__ == '__main__':
    """
    python -c "import utool; utool.doctest_funcs(utool.util_class, allexamples=True)"
    python -m utool.util_class --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut
    ut.doctest_funcs()
