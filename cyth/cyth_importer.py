"""
python -c "import doctest, cyth; print(doctest.testmod(cyth.cyth_importer))"
"""
from __future__ import absolute_import, division, print_function
from . import cyth_helpers
from os.path import splitext, basename
import imp
from utool.util_six import get_funcdoc, get_funcname, get_funcglobals  # NOQA
import sys
import utool


#WITH_CYTH = utool.get_flag('--cyth')
WITH_CYTH = not utool.get_flag('--nocyth')
CYTH_WRITE = utool.get_flag('--cyth-write')


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


def import_cyth_dict(pyth_modname_):
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
        cythonized_funcs = get_cythonized_funcs(pyth_modname)
        return cythonized_funcs
        # TODO: Get list of cythonized funcs and return them
        #from .keypoint_cython import (get_invVR_mats_sqrd_scale_float64,)  # NOQA
        #get_invVR_mats_sqrd_scale_cython = get_invVR_mats_sqrd_scale_float64
    except ImportError as ex:  # NOQA
        raise
        dummy_cythonized_funcs = import_cyth_default(pyth_modname)
        return dummy_cythonized_funcs


def get_cythonized_funcs(pyth_modname):
    pkgname, fromlist, cyth_modname = pkg_submodule_split(pyth_modname)
    cyth_mod = __import__(cyth_modname, globals(), locals(), fromlist=fromlist, level=0)
    mod_dict = cyth_mod.__dict__
    cythonized_funcs = {}
    #print('[import_cyth] mod_dict=%s' % (utool.dict_str(mod_dict),))
    for key, val in mod_dict.items():
        valstr = repr(val)
        # FIXME: might change in python3
        if valstr.startswith('<built-in function '):
            assert key.startswith("_") and key.endswith("_cyth"), key
            cythonized_funcs[key] = val
            cythonized_funcs[key[1:]] = val
    #print(utool.dict_str(cythonized_funcs))
    return cythonized_funcs


def import_cyth_execstr(pyth_modname):
    dummy_cythonized_funcs = import_cyth_default(pyth_modname)
    pyth_list = []
    for funcname, func in dummy_cythonized_funcs.items():
        pyth_list.append(funcname + ' = ' + get_funcname(func))

    try:
        cyth_list = []
        pkgname, fromlist, cyth_modname = pkg_submodule_split(pyth_modname)
        cythonized_funcs = get_cythonized_funcs(pyth_modname)
        maxlen = 0
        for funcname, func in cythonized_funcs.items():
            maxlen = max(maxlen, len(funcname))
            cyth_list.append((funcname, cyth_modname + '.' + func.__name__))
        cyth_list = sorted(cyth_list)
        cyth_list2 = ['import ' + cyth_modname]
        for lhs, rhs in cyth_list:
            cyth_list2.append(lhs.ljust(maxlen) + ' = ' + rhs)
    except ImportError:
        cyth_list2 = ['raise ImportError("no cyth")']

    cyth_block = utool.indentjoin(cyth_list2).strip('\n')
    pyth_block = utool.indentjoin(pyth_list).strip('\n')
    execstr = utool.unindent(
        '''
        try:
            if not {WITH_CYTH}:
                raise ImportError('no cyth')
        {cyth_block}
        except ImportError:
        {pyth_block}''').format(WITH_CYTH=WITH_CYTH, **locals()).strip('\n')
    #print(execstr)
    if CYTH_WRITE:
        write_explicit(pyth_modname, execstr)
    return execstr


def write_explicit(pyth_modname, execstr):
    import multiprocessing
    is_main_proc = multiprocessing.current_process().name == 'MainProcess'
    if is_main_proc:
        from utool import util_str
        from os.path import exists
        new_else = util_str.indent(execstr)
        #print(new_else)
        # Get path to init file so we can overwrite it
        pyth_fpath = pyth_modname
        module = sys.modules[pyth_modname]  # module currently being imported
        modpath, ext = splitext(module.__file__)
        assert ext == '.pyc' or ext == '.py'
        pyth_fpath = modpath + '.py'
        #import IPython
        #IPython.embed()
        print("attempting to update: %r" % pyth_fpath)
        assert exists(pyth_fpath)
        new_lines = []
        rest_lines = []
        broken = False
        with open(pyth_fpath, 'r') as file_:
            lines = file_.readlines()
            for line in lines:
                if broken:
                    rest_lines.append(line)
                    continue
                new_lines.append(line)
                sentinal = '    # <AUTOGEN_CYTH>'
                if line.startswith(sentinal):
                    new_lines.append(new_else + '\n')
                    broken = True
        if broken:
            print("writing updated file: %r" % pyth_fpath)
            new_text = ''.join(new_lines + rest_lines)
            #print(new_text)
            with open(pyth_fpath, 'w') as file_:
                file_.write(new_text)
        else:
            print("no write hook for file: %r" % pyth_fpath)


def import_cyth_default(pyth_modname):
    from .cyth_decorators import get_registered_funcs
    #import IPython
    #IPython.embed()
    module = sys.modules[pyth_modname]  # module currently being imported
    func_list = []
    func_list = get_registered_funcs(pyth_modname)
    for key, val in module.__dict__.items():
        if hasattr(val, 'func_doc'):
            func = val
            docstr = get_funcdoc(func)
            if docstr is not None and '<CYTH' in docstr:
                func_list.append(func)
                #print(func)
                #print(key)
                #func2 = func
    # default to python
    dummy_cythonized_funcs = {
        utool.util_six.get_funcname(func) + '_cyth': func
        for func in set(func_list)
    }
    return dummy_cythonized_funcs
