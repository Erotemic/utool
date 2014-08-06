"""
python -c "import doctest, cyth; print(doctest.testmod(cyth.cyth_importer))"
"""
from __future__ import absolute_import, division, print_function
from . import cyth_helpers
import utool


WITH_CYTH = utool.get_flag('--cyth')
WITH_CYTH = not utool.get_flag('--nocyth')


def pkg_submodule_split(modname):
    cyth_modname = cyth_helpers.get_cyth_name(modname)
    # Break module name into package and submodule
    if cyth_modname.find('.') > -1:
        components = cyth_modname.split('.')
        fromlist = [components[-1]]
        pkgname = '.'.join(components[:-1])
    else:
        pkgname = cyth_modname
        fromlist = []
    return pkgname, fromlist


def import_cyth(pyth_modname):
    """
    >>> from cyth import *  # NOQA
    >>> pyth_modname = 'vtool.keypoint'
    """
    try:
        if not WITH_CYTH:
            raise ImportError('NO_CYTH')
        pkgname, fromlist = pkg_submodule_split(pyth_modname)
        cyth_mod = __import__(pkgname, globals(), locals(), fromlist=fromlist, level=0)

        # TODO: Get list of cythonized funcs and return them

        #from .keypoint_cython import (get_invVR_mats_sqrd_scale_float64,)  # NOQA
        #get_invVR_mats_sqrd_scale_cython = get_invVR_mats_sqrd_scale_float64
    except ImportError as ex:
        return import_cyth_default(pyth_modname)


def import_cyth_default(pyth_modname):
    # default to python
    #get_invVR_mats_sqrd_scale_cython = get_invVR_mats_sqrd_scale
    pass
