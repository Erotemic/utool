from __future__ import absolute_import, division, print_function
import types
from collections import defaultdict
from .util_inject import inject
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


def inject_func_as_method(self, func, method_name=None):
    """
    Wraps func as a bound method of self. Then injects func into self
    Input:
        self - class instance
        func - some function whos first arugment is a class instance
        method_name - default=func.func_name, if specified renames the method
    """
    if method_name is None:
        method_name = func.func_name
    printDBG('Injecting method_name=%r' % method_name)
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
    """ makes a metaclass that overrides __getattr__ and __setattr__ to forward
        some specific attribute references to a specified instance variable """
    class ForwardingMetaclass(base_class.__class__):
        def __init__(metaself, name, bases, dct):
            # print('ForwardingMetaclass.__init__(): {forwarding_dest_getter: %r; whitelist: %r}' % (forwarding_dest_getter, whitelist))
            super(ForwardingMetaclass, metaself).__init__(name, bases, dict)
            old_getattr = metaself.__getattribute__
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
            metaself.__getattribute__ = new_getattr
            old_setattr = metaself.__setattr__
            def new_setattr(self, name, val):
                if name in whitelist:
                    #dest = old_getattr(self, forwarding_dest_name)
                    dest = forwarding_dest_getter(self)
                    dest.__class__.__setattr__(dest, name, val)
                else:
                    old_setattr(self, name, val)
            metaself.__setattr__ = new_setattr
    return ForwardingMetaclass
