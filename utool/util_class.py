from __future__ import absolute_import, division, print_function
import sys
import six
import types
import functools
from collections import defaultdict
from .util_inject import inject
from .util_set import oset
from . import util_arg
from ._internal.meta_util_six import get_funcname
print, print_, printDBG, rrr, profile = inject(__name__, '[class]', DEBUG=False)


# Registers which classes have which attributes
# FIXME: this might cause memory leaks
__CLASSTYPE_ATTRIBUTES__ = defaultdict(oset)
__CLASSTYPE_POSTINJECT_FUNCS__ = defaultdict(oset)

VERBOSE_CLASS = util_arg.get_argflag(('--verbose-class', '--verbclass')) or util_arg.VERYVERBOSE


def inject_instance(self, classtype=None, allow_override=False, verbose=VERBOSE_CLASS):
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
        >>> DOCTEST = False
        >>> utool.make_class_method_decorator(InvertedIndex)(smk_debug.invindex_dbgstr)
        >>> utool.inject_instance(invindex)
    """
    if classtype is None:
        import utool as ut
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
    if verbose:
        print('[util_class] injecting methods into %r' % (self,))
    for func in __CLASSTYPE_ATTRIBUTES__[classtype]:
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


def make_class_method_decorator(classtype):
    """ register a class to be injectable
    classtype is a key which should be a type

    Args:
        classtype : the class to be injected into
            REMEMBER to call inject_instance in __init__

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
    if VERBOSE_CLASS:
        print('[util_class] register make_class_method_decorator=%r' % make_class_method_decorator)
    closure_decorate_class_method = functools.partial(decorate_class_method, classtype=classtype)
    return closure_decorate_class_method


def make_class_postinject_decorator(classtype):
    """
    Args:
        classtype : the class to be injected into

    Returns:
        closure_decorate_postinject (func): decorator for injectable methods

    SeeAlso:
        make_class_method_decorator
    """
    if VERBOSE_CLASS:
        print('[util_class] register class_postinject=%r' % make_class_method_decorator)
    closure_decorate_postinject = functools.partial(decorate_postinject, classtype=classtype)
    return closure_decorate_postinject


def decorate_class_method(func, classtype=None):
    """
    Will inject all decorated function as methods of classtype
    """
    assert classtype is not None, 'must specify classtype'
    global __CLASSTYPE_ATTRIBUTES__
    __CLASSTYPE_ATTRIBUTES__[classtype].append(func)
    return func


def decorate_postinject(func, classtype=None):
    """
    Will perform func with argument self after inject_instance is called on classtype
    """
    assert classtype is not None, 'must specify classtype'
    global __CLASSTYPE_POSTINJECT_FUNCS__
    __CLASSTYPE_POSTINJECT_FUNCS__[classtype].append(func)
    return func


def inject_func_as_method(self, func, method_name=None, class_=None, allow_override=False):
    """ Injects a function into an object as a method

    Wraps func as a bound method of self. Then injects func into self
    It is preferable to use make_class_method_decorator and inject_instance

    Args:
       self (object): class instance
       func : some function whos first arugment is a class instance
       method_name (str) : default=func.__name__, if specified renames the method
       class_ (type) : if func is an unbound method of this class
    """
    if method_name is None:
        method_name = get_funcname(func)
    #printDBG('Injecting method_name=%r' % method_name)
    method = types.MethodType(func, self)
    old_method = getattr(self, method_name, None)
    if old_method:
        if allow_override is False:
            raise AssertionError('Overrides are not allowed. Already have method_name=%r' % (method_name))
        elif allow_override == 'warn':
            print('WARNING: Overrides are not allowed. Already have method_name=%r. Skipping' % (method_name))
            return
        elif allow_override == 'override+warn':
            print('WARNING: Overrides are allowed, but dangerous. Already have method_name=%r.' % (method_name))
        # TODO: does this actually decrement the refcount enough?
        del old_method
    setattr(self, method_name, method)


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
    def __init__(metaself, name, bases, dct):
        super(ReloadingMetaclass, metaself).__init__(name, bases, dct)

        def rrr(self):
            classname = self.__class__.__name__
            try:
                modname = self.__class__.__module__
                print('reloading ' + classname + ' from ' + modname)
                module = sys.modules[modname]
                if modname != '__main__':
                    # Reload the parent module
                    module.rrr()
                # Get new class definition
                class_ = getattr(module, classname)
                # TODO: handle injected definitions
                reload_class_methods(self, class_)
            except Exception as ex:
                import utool
                utool.printex(ex, keys=[
                    'modname',
                    'module',
                    'class_',
                    'self', ])
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


if __name__ == '__main__':
    """
    python -c "import utool; utool.doctest_funcs(utool.util_class, allexamples=True)"
    python utool/util_class.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut
    ut.doctest_funcs()
