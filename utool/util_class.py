from __future__ import absolute_import, division, print_function
import types
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[class]', DEBUG=False)


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
