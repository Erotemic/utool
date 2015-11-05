# -*- coding: utf-8 -*-
lazy_module_attrs =  ['_modname', '_module', '_load_module']


class LazyModule(object):
    """
    Waits to import the module until it is actually used

    """
    def __init__(self, modname):
        self._modname = modname
        self._module = None

    def _load_module(self):
        #print('loading module')
        self._module =  __import__(self._modname, globals(), locals(), fromlist=[], level=0)

    def __str__(self):
        return 'LazyModule(%s)' % (self._modname,)

    def __dir__(self):
        self._load_module()
        return dir(self._module)

    def __getattr__(self, item):
        """Maps values to attributes.
        Only called if there *isn't* an attribute with this name
        """
        if item in lazy_module_attrs:
            return super(LazyModule, self).__getattr__(item)
        self._load_module()
        return getattr(self._module, item)

    def __setattr__(self, item, value):
        """Maps attributes to values.
        Only if we are initialised
        """
        if item in lazy_module_attrs:
            return super(LazyModule, self).__setattr__(item, value)
        self._load_module()
        setattr(self._module, item, value)


modname = 'theano'
theano = LazyModule(modname)
