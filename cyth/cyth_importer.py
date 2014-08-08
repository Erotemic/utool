"""
python -c "import doctest, cyth; print(doctest.testmod(cyth.cyth_importer))"
"""
from __future__ import absolute_import, division, print_function
from . import cyth_helpers
import utool


#WITH_CYTH = utool.get_flag('--cyth')
WITH_CYTH = not utool.get_flag('--nocyth')


def pkg_submodule_split(pyth_modname):
    cyth_modname = cyth_helpers.get_cyth_name(pyth_modname)
    # Break module name into package and submodule
    if cyth_modname.find('.') > -1:
        components = cyth_modname.split('.')
        submod = components[-1]
        fromlist = [submod]
        pkgname = '.'.join(components[:-1])
    else:
        pkgname = cyth_modname
        fromlist = []
    return pkgname, fromlist, cyth_modname


def import_cyth(pyth_modname):
    """
    #>>> from cyth import *  # NOQA
    >>> from cyth.cyth_importer import *  # NOQA
    >>> pyth_modname = 'vtool.keypoint'
    >>> import_cyth(pyth_modname)
    """
    try:
        print('[import_cyth] pyth_modname=%r' % (pyth_modname,))
        if not WITH_CYTH:
            print('[import_cyth] NO_CYTH')
            raise ImportError('NO_CYTH')
        pkgname, fromlist, cyth_modname = pkg_submodule_split(pyth_modname)
        cyth_mod = __import__(cyth_modname, globals(), locals(), fromlist=fromlist, level=0)
        mod_dict = cyth_mod.__dict__
        cythonized_funcs = {}
        print('[import_cyth] mod_dict=%s' % (utool.dict_str(mod_dict),))
        for key, val in mod_dict.items():
            valstr = repr(val)
            # FIXME: might change in python3
            if valstr.startswith('<built-in function '):
                cythonized_funcs[key] = val
        print(utool.dict_str(cythonized_funcs))
        return cythonized_funcs
        # TODO: Get list of cythonized funcs and return them
        #from .keypoint_cython import (get_invVR_mats_sqrd_scale_float64,)  # NOQA
        #get_invVR_mats_sqrd_scale_cython = get_invVR_mats_sqrd_scale_float64
    except ImportError as ex:
        raise
        return import_cyth_default(pyth_modname)


def import_cyth_default(pyth_modname):
    # default to python
    #get_invVR_mats_sqrd_scale_cython = get_invVR_mats_sqrd_scale
    return {}
    pass
