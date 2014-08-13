"""
python -c "import doctest, cyth; print(doctest.testmod(cyth.cyth_importer))"
"""
from __future__ import absolute_import, division, print_function
from . import cyth_helpers
from os.path import splitext, basename
import imp
import utool
from utool.util_six import get_funcdoc, get_funcname, get_funcglobals  # NOQA
import sys
from . import cyth_args


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
        if not cyth_args.WITH_CYTH:
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
    pyth_list2 = utool.align_lines(sorted(pyth_list), '=')

    try:
        cyth_list = []
        pkgname, fromlist, cyth_modname = pkg_submodule_split(pyth_modname)
        cythonized_funcs = get_cythonized_funcs(pyth_modname)
        for funcname, func in cythonized_funcs.items():
            cyth_list.append(funcname + ' = ' + cyth_modname + '.' + func.__name__)
        cyth_list2 = ['import ' + cyth_modname] + utool.align_lines(sorted(cyth_list), '=')
    except ImportError:
        cyth_list2 = ['raise ImportError("no cyth")']

    cyth_block = utool.indentjoin(cyth_list2).strip()
    pyth_block = utool.indentjoin(pyth_list2).strip()
    execstr = utool.unindent(
        '''
        try:
            if not cyth.WITH_CYTH:
                raise ImportError('no cyth')
            {cyth_block}
            CYTHONIZED = True
        except ImportError:
            {pyth_block}
            CYTHONIZED = False''').format(**locals()).strip('\n')
    #print(execstr)
    if cyth_args.CYTH_WRITE:
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
        broken = False
        closed = False
        start_sentinal = '    # <AUTOGEN_CYTH>'
        end_sentinal   = '    # </AUTOGEN_CYTH>'
        with open(pyth_fpath, 'r') as file_:
            lines = file_.readlines()
            for line in lines:
                if not closed and not broken:
                    # Append lines until you see start_sentinal
                    new_lines.append(line)
                    if line.startswith(start_sentinal):
                        indent = '    '
                        help_line = '# Regen command: python -c "import %s" --cyth-write\n' % pyth_modname
                        new_lines.append(indent + help_line)
                        new_lines.append(new_else + '\n')
                        broken = True
                elif not closed and broken:
                    # Skip lines between sentinals
                    if line.startswith(end_sentinal):
                        new_lines.append(end_sentinal + '\n')
                        closed = True
                elif closed and broken:
                    # Append lines after sentinals
                    new_lines.append(line)
                else:
                    raise AssertionError('closed before opening cyth tags')
        if broken and closed:
            print("writing updated file: %r" % pyth_fpath)
            new_text = ''.join(new_lines)
            #print(new_text)
            with open(pyth_fpath, 'w') as file_:
                file_.write(new_text)
        else:
            default_cyth_block = utool.unindent('''
            import cyth
            if cyth.DYNAMIC:
                exec(cyth.import_cyth_execstr(__name__))
            else:
                # <AUTOGEN_CYTH>
                # </AUTOGEN_CYTH>
                pass
            ''')
            default_cyth_block  # NOQA
            print("no write hook for file: %r" % pyth_fpath)


def import_cyth_default(pyth_modname):
    #import IPython
    #IPython.embed()
    module = sys.modules[pyth_modname]  # module currently being imported
    func_list = []
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
