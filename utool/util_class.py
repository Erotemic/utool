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


def inject_instance(self, classtype=None, allow_override=False,
                    verbose=VERBOSE_CLASS, strict=True):
    """
    Injects an instance (self) of type (classtype)
    with all functions registered to (classtype)

    call this in the __init__ class function

    Args:
        self: the class instance
        classtype: key for a class, preferably the class type itself, but it
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
        if classtype is None:
            # Probably should depricate this block of code
            # It tries to do too much
            classtype = self.__class__
            if classtype == 'ibeis.gui.models_and_views.IBEISTableView':
                # HACK HACK HACK
                from guitool.__PYQT__ import QtGui
                classtype = QtGui.QAbstractItemView
            if len(__CLASSTYPE_ATTRIBUTES__[classtype]) == 0:
                print('[utool] Warning: no classes of type %r are registered' % (classtype,))
                print('[utool] type(self)=%r, self=%r' % (type(self), self)),
                print('[utool] Checking to see if anybody else was registered...')
                print('[utool] __CLASSTYPE_ATTRIBUTES__ = ' + ut.list_str(__CLASSTYPE_ATTRIBUTES__.keys()))
                for classtype_, _ in six.iteritems(__CLASSTYPE_ATTRIBUTES__):
                    isinstance(self, classtype_)
                    classtype = classtype_
                    print('[utool] Warning: using subclass=%r' % (classtype_,))
                    break
        func_list = __CLASSTYPE_ATTRIBUTES__[classtype]
        if verbose or util_arg.VERBOSE:
            print('[util_class] injecting %d methods\n   with classtype=%r\n   into %r' % (len(func_list), classtype, self,))
        for func in func_list:
            if VERBOSE_CLASS:
                print('[util_class] * injecting %r' % (func,))
            method_name = None
            # Allow user to register tuples for aliases
            if isinstance(func, tuple):
                func, method_name = func
            inject_func_as_method(self, func, method_name=method_name, allow_override=allow_override)
        if verbose:
            print('[util_class] Running postinject functions on %r' % (self,))
        for func in __CLASSTYPE_POSTINJECT_FUNCS__[classtype]:
            func(self)
        if verbose:
            print('[util_class] Finished injecting instance self=%r' % (self,))
    except Exception as ex:
        ut.printex(ex, 'ISSUE WHEN INJECTING %r' % (classtype,),
                      iswarning=not strict)
        if strict:
            raise


def make_class_method_decorator(classtype, modname=None):
    """
    register a class to be injectable
    classtype is a key that identifies the injected class
    REMEMBER to call inject_instance in __init__

    Args:
        classtype : the class to be injected into
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
    if util_arg.VERBOSE or VERBOSE_CLASS:
        print('[util_class] register make_class_method_decorator classtype=%r, modname=%r'
              % (classtype, modname))
    if modname == '__main__':
        # skips reinjects into main
        print('WARNING: cannot register class functions as __main__')
        return lambda func: func
    closure_decorate_class_method = functools.partial(decorate_class_method, classtype=classtype)
    return closure_decorate_class_method


def make_class_postinject_decorator(classtype, modname=None):
    """
    Args:
        classtype : the class to be injected into
        modname : the global __name__ of the module youa re injecting from

    Returns:
        closure_decorate_postinject (func): decorator for injectable methods

    SeeAlso:
        make_class_method_decorator
    """
    if util_arg.VERBOSE or VERBOSE_CLASS:
        print('[util_class] register class_postinject classtype=%r, modname=%r'
              % (classtype, modname))
    if modname == '__main__':
        print('WARNING: cannot register class functions as __main__')
        # skips reinjects into main
        return lambda func: func
    closure_decorate_postinject = functools.partial(decorate_postinject, classtype=classtype)
    return closure_decorate_postinject


def decorate_class_method(func, classtype=None, skipmain=False):
    """
    Will inject all decorated function as methods of classtype

    classtype is some identifying string, tuple, or object

    func can also be a tuple
    """
    #import utool as ut
    global __CLASSTYPE_ATTRIBUTES__
    assert classtype is not None, 'must specify classtype'
    #if not (skipmain and ut.get_caller_modname() == '__main__'):
    __CLASSTYPE_ATTRIBUTES__[classtype].append(func)
    return func


def decorate_postinject(func, classtype=None, skipmain=False):
    """
    Will perform func with argument self after inject_instance is called on classtype

    classtype is some identifying string, tuple, or object
    """
    #import utool as ut
    global __CLASSTYPE_POSTINJECT_FUNCS__
    assert classtype is not None, 'must specify classtype'
    #if not (skipmain and ut.get_caller_modname() == '__main__'):
    __CLASSTYPE_POSTINJECT_FUNCS__[classtype].append(func)
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


class ReloadingMetaclass(type):
    """
    Classes with this metaclass will be able to reload themselves
    on a per-instance basis using the rrr function.

    If the functions _on_reload and _initialize_self exist
    they will be called after and before reload respectively. Any
    inject_instance functions should be handled there.

    Example:
        >>> # DIABLE_DOCTEST
        >>> from utool.util_class import *  # NOQA
        >>> import utool as ut
        >>> @six.add_metaclass(ut.ReloadingMetaclass)
        >>> class Foo():
        ...     def __init__(self):
        ...        pass
        >>> # You can edit foo on disk and call rrr in ipython
        >>> # if you add a new function to it
        >>> foo = Foo()
        >>> # This will not work as a doctests because
        >>> # Foo's parent module will be __main__ but
        >>> # there will be no easy way to write to it.
        >>> # This does work when you run from ipython
        >>> @six.add_metaclass(ut.ReloadingMetaclass)
        >>> class Foo():
        ...     def __init__(self):
        ...        pass
        ...     def bar(self):
        ...        return "spam"
        >>> foo.rrr()
        >>> result = foo.bar()
        >>> print(result)
        spam
    """
    def __init__(metaself, name, bases, dct):
        super(ReloadingMetaclass, metaself).__init__(name, bases, dct)

        def rrr(self, verbose=True):
            classname = self.__class__.__name__
            try:
                modname = self.__class__.__module__
                if verbose:
                    print('reloading ' + classname + ' from ' + modname)
                if hasattr(self, '_on_reload'):
                    self._on_reload()
                module = sys.modules[modname]
                if modname != '__main__':
                    # Reload the parent module
                    if hasattr(module, 'rrr'):
                        module.rrr()
                    else:
                        import imp
                        imp.reload(module)
                # Get new class definition
                class_ = getattr(module, classname)
                # TODO: handle injected definitions
                reload_class_methods(self, class_)
                if hasattr(self, '_initialize_self'):
                    self._initialize_self()
            except Exception as ex:
                import utool as ut
                ut.printex(ex, 'Error Reloading Class', keys=[
                    'modname',
                    'module',
                    'class_',
                    'self', ])
                #ut.embed()
                #print(ut.dict_str(module.__dict__))
                raise
        metaself.rrr = rrr


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


if __name__ == '__main__':
    """
    python -c "import utool; utool.doctest_funcs(utool.util_class, allexamples=True)"
    python -m utool.util_class --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut
    ut.doctest_funcs()
