from __future__ import absolute_import, division, print_function
import sys
import types
from collections import defaultdict
from .util_inject import inject
from ._internal.meta_util_six import get_funcname
print, print_, printDBG, rrr, profile = inject(__name__, '[class]', DEBUG=False)


# Registers which classes have which attributes
__CLASSTYPE_ATTRIBUTES__ = defaultdict(list)


def classmember(classtype):
    """ classtype is some key, which should be a type """
    def closure_classmember(func):
        global __CLASSTYPE_ATTRIBUTES__
        __CLASSTYPE_ATTRIBUTES__[classtype].append(func)
        return func
    return closure_classmember


def inject_func_as_method(self, func, method_name=None, class_=None):
    """ Injects a function into an object as a method

    Wraps func as a bound method of self. Then injects func into self

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
        del old_method
    setattr(self, method_name, method)


def inject_instance(classtype, self):
    """
    Injects an instance (self) of type (classtype)
    with all functions registered to (classtype)
    """
    for func in __CLASSTYPE_ATTRIBUTES__[classtype]:
        inject_func_as_method(self, func)


#def __instancemember(self, func):
#    if isinstance(func, types.MethodType):
#        return func
#    else:
#        return inject_func_as_method(self, func)


#class ReloadableMetaclass(type):
#    def __new__(meta, name, bases, attrs):
#        #print('meta = %r' (str(meta),))
#        #print('name = %r' (str(name),))
#        #print('bases = %r' (str(bases),))
#        #print('attrs = %r' (str(attrs),))
#        return super(ReloadableMetaclass, meta).__new__(meta, name, bases, attrs)

#    def __init__(self, name, bases, attrs):
#        super(ReloadableMetaclass, self).__init__(name, bases, attrs)
#        # classregistry.register(self, self.interfaces)
#        print('Would register class %r now.' % (self,))

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
                    module.rrr()
                class_ = getattr(module, classname)
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


def get_comparison_methods():
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


def reload_class_methods(self, class_):
    """
    reloads all class methods
    """
    self.__class__ = class_
    for key in dir(class_):
        func = getattr(class_, key)
        if isinstance(func, types.MethodType):
            inject_func_as_method(self, func, class_=class_)
