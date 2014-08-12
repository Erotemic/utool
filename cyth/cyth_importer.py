"""
python -c "import doctest, cyth; print(doctest.testmod(cyth.cyth_importer))"
"""
from __future__ import absolute_import, division, print_function
from . import cyth_helpers
from os.path import splitext, basename
import imp
import sys
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


def rectify_modname(pyth_modname_):
    if pyth_modname_ != '__main__':
        pyth_modname = pyth_modname_
        return pyth_modname
    else:
        # http://stackoverflow.com/questions/606561/how-to-get-filename-of-the-main-module-in-python
        def main_is_frozen():
            return (hasattr(sys, "frozen") or  # new py2exe
                    hasattr(sys, "importers")  # old py2exe
                    or imp.is_frozen("__main__"))  # tools/freeze

        def get_main_name():
            if main_is_frozen():
                # print 'Running from path', os.path.dirname(sys.executable)
                return splitext(basename(sys.executable))[0]
            return splitext(basename(sys.argv[0]))[0]

        # find path to where we are running
        pyth_modname = get_main_name()
        return pyth_modname

        # OPTIONAL:
        # add the sibling 'lib' dir to our module search path
        #lib_path = os.path.join(get_main_dir(), os.path.pardir, 'lib')
        #sys.path.insert(0, lib_path)

        # OPTIONAL:
        # use info to find relative data files in 'data' subdir
        #datafile1 = os.path.join(get_main_dir(), 'data', 'file1')


def import_cyth(pyth_modname_):
    """
    #>>> from cyth import *  # NOQA
    >>> from cyth.cyth_importer import *  # NOQA
    >>> pyth_modname = 'vtool.keypoint'
    >>> import_cyth(pyth_modname)
    """
    pyth_modname = rectify_modname(pyth_modname_)
    try:
        print('[import_cyth] pyth_modname=%r' % (pyth_modname,))
        if not WITH_CYTH:
            print('[import_cyth] NO_CYTH')
            raise ImportError('NO_CYTH')
        pkgname, fromlist, cyth_modname = pkg_submodule_split(pyth_modname)
        cyth_mod = __import__(cyth_modname, globals(), locals(), fromlist=fromlist, level=0)
        mod_dict = cyth_mod.__dict__
        cythonized_funcs = {}
        #print('[import_cyth] mod_dict=%s' % (utool.dict_str(mod_dict),))
        for key, val in mod_dict.items():
            valstr = repr(val)
            # FIXME: might change in python3
            if valstr.startswith('<built-in function '):
                cythonized_funcs[key] = val
        #print(utool.dict_str(cythonized_funcs))
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
    from .cyth_decorators import get_registered_funcs
    func_list = get_registered_funcs(pyth_modname)
    from utool.util_six import get_funcname
    dummy_cythonized_funcs = {get_funcname(func) + '_cyth' for func in func_list}
    return dummy_cythonized_funcs
