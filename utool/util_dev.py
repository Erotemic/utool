from __future__ import absolute_import, division, print_function
import sys
import six
import os
import warnings
try:
    import numpy as np
except ImportError as ex:
    pass
from os.path import splitext, exists, join, split, relpath
from .Printable import printableVal, common_stats, mystats  # NOQA
from . import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[dev]')


def DEPRICATED(func):
    """ deprication decorator """
    warn_msg = 'Depricated call to: %s' % func.__name__

    def __DEP_WRAPPER(*args, **kwargs):
        raise Exception('dep')
        warnings.warn(warn_msg, category=DeprecationWarning)
        #warnings.warn(warn_msg, category=DeprecationWarning)
        return func(*args, **kwargs)
    __DEP_WRAPPER.__name__ = func.__name__
    __DEP_WRAPPER.__doc__ = func.__doc__
    __DEP_WRAPPER.__dict__.update(func.__dict__)
    return __DEP_WRAPPER


# --- Info Strings ---

def print_mystats(_list, lbl=None):
    import utool
    if lbl is not None:
        print('Mystats for %s' % lbl)
    print(utool.dict_str(utool.mystats(_list)))


def stats_str(*args, **kwargs):
    # wrapper for common_stats
    return common_stats(*args, **kwargs)


def myprint(input_=None, prefix='', indent='', lbl=''):
    if len(lbl) > len(prefix):
        prefix = lbl
    if len(prefix) > 0:
        prefix += ' '
    print_(indent + prefix + str(type(input_)) + ' ')
    if isinstance(input_, list):
        print(indent + '[')
        for item in iter(input_):
            myprint(item, indent=indent + '  ')
        print(indent + ']')
    elif isinstance(input_, six.string_types):
        print(input_)
    elif isinstance(input_, dict):
        print(printableVal(input_))
    else:
        print(indent + '{')
        attribute_list = dir(input_)
        for attr in attribute_list:
            if attr.find('__') == 0:
                continue
            val = str(input_.__getattribute__(attr))
            #val = input_[attr]
            # Format methods nicer
            #if val.find('built-in method'):
            #    val = '<built-in method>'
            print(indent + '  ' + attr + ' : ' + val)
        print(indent + '}')


def info(var, lbl):
    if isinstance(var, np.ndarray):
        return npinfo(var, lbl)
    if isinstance(var, list):
        return listinfo(var, lbl)


def npinfo(ndarr, lbl='ndarr'):
    info = ''
    info += (lbl + ': shape=%r ; dtype=%r' % (ndarr.shape, ndarr.dtype))
    return info


def listinfo(list_, lbl='ndarr'):
    if not isinstance(list_, list):
        raise Exception('!!')
    info = ''
    type_set = set([])
    for _ in iter(list_):
        type_set.add(str(type(_)))
    info += (lbl + ': len=%r ; types=%r' % (len(list_), type_set))
    return info


#expected_type = np.float32
#expected_dims = 5
def numpy_list_num_bits(nparr_list, expected_type, expected_dims):
    num_bits = 0
    num_items = 0
    num_elemt = 0
    bit_per_item = {
        np.float32: 32,
        np.uint8: 8
    }[expected_type]
    for nparr in iter(nparr_list):
        arr_len, arr_dims = nparr.shape
        if nparr.dtype.type is not expected_type:
            msg = 'Expected Type: ' + repr(expected_type)
            msg += 'Got Type: ' + repr(nparr.dtype)
            raise Exception(msg)
        if arr_dims != expected_dims:
            msg = 'Expected Dims: ' + repr(expected_dims)
            msg += 'Got Dims: ' + repr(arr_dims)
            raise Exception(msg)
        num_bits += len(nparr) * expected_dims * bit_per_item
        num_elemt += len(nparr) * expected_dims
        num_items += len(nparr)
    return num_bits,  num_items, num_elemt


def runprofile(cmd, globals_=globals(), locals_=locals()):
    # Meliae # from meliae import loader # om = loader.load('filename.json') # s = om.summarize();
    #http://www.huyng.com/posts/python-performance-analysis/
    #Once youve gotten your code setup with the <AT>profile decorator, use kernprof.py to run your script.
    #kernprof.py -l -v fib.py
    import cProfile
    print('[util] Profiling Command: ' + cmd)
    cProfOut_fpath = 'OpenGLContext.profile'
    cProfile.runctx( cmd, globals_, locals_, filename=cProfOut_fpath)
    # RUN SNAKE
    print('[util] Profiled Output: ' + cProfOut_fpath)
    if sys.platform == 'win32':
        rsr_fpath = 'C:/Python27/Scripts/runsnake.exe'
    else:
        rsr_fpath = 'runsnake'
    view_cmd = rsr_fpath + ' "' + cProfOut_fpath + '"'
    os.system(view_cmd)
    return True


def memory_profile(with_gc=False):
    #http://stackoverflow.com/questions/2629680/deciding-between-subprocess-multiprocessing-and-thread-in-python
    from . import util_str
    import guppy
    if with_gc:
        garbage_collect()
    hp = guppy.hpy()
    print('[hpy] Waiting for heap output...')
    heap_output = hp.heap()
    print(heap_output)
    print('[hpy] total heap size: ' + util_str.byte_str2(heap_output.size))
    from . import util_resources
    util_resources.memstats()
    # Graphical Browser
    #hp.pb()


def make_call_graph(func, *args, **kwargs):
    """ profile with pycallgraph
    pycallgraph graphviz -- ./mypythonscript.py
    http://pycallgraph.slowchop.com/en/master/
    """
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    with PyCallGraph(output=GraphvizOutput):
        func(*args, **kwargs)


def make_object_graph(obj, fpath='sample_graph.png'):
    """ memoryprofile with objgraph
    http://mg.pov.lt/objgraph/
    """
    import objgraph
    objgraph.show_refs([obj], filename='ref_graph.png')
    objgraph.show_backrefs([obj], filename='backref_graph.png')
    objgraph.show_most_common_types()


def disable_garbage_collection():
    import gc
    gc.disable()


def enable_garbage_collection():
    import gc
    gc.enable()


def garbage_collect():
    import gc
    gc.collect()


def get_object_size(obj):
    seen = set([])
    def _get_object_size(obj):
        if (obj is None or isinstance(obj, (str, int, bool, float))):
            return sys.getsizeof(obj)

        object_id = id(obj)
        if object_id in seen:
            return 0
        seen.add(object_id)

        totalsize = sys.getsizeof(obj)
        if isinstance(obj, np.ndarray):
            totalsize += obj.nbytes
        elif (isinstance(obj, (tuple, list, set, frozenset))):
            for item in obj:
                totalsize += _get_object_size(item)
        elif isinstance(obj, dict):
            try:
                for key, val in six.iteritems(obj):
                    totalsize += _get_object_size(key)
                    totalsize += _get_object_size(val)
            except RuntimeError:
                print(key)
                raise
        elif isinstance(obj, object) and hasattr(obj, '__dict__'):
            totalsize += _get_object_size(obj.__dict__)
            return totalsize
        return totalsize
    return _get_object_size(obj)


def print_object_size_tree(obj):
    """ Needs work """
    seen = set([])
    def _get_object_size_tree(obj, indent='', lbl='obj'):
        if (obj is None or isinstance(obj, (str, int, bool, float))):
            return [sys.getsizeof(obj)]
        object_id = id(obj)
        if object_id in seen:
            return []
        seen.add(object_id)
        size_list = [(lbl, sys.getsizeof(obj))]
        print(indent + '%s = %s ' % (lbl, str(sys.getsizeof(obj))))
        if isinstance(obj, np.ndarray):
            size_list.append(obj.nbytes)
            print(indent + '%s = %s ' % ('arr', obj.nbytes))
        elif (isinstance(obj, (tuple, list, set, frozenset))):
            for item in obj:
                size_list += _get_object_size_tree(item, indent + '   ', 'item')
        elif isinstance(obj, dict):
            try:
                for key, val in six.iteritems(obj):
                    size_list += _get_object_size_tree(key, indent + '   ', key)
                    size_list += _get_object_size_tree(val, indent + '   ', key)
            except RuntimeError:
                print(key)
                raise
        elif isinstance(obj, object) and hasattr(obj, '__dict__'):
            size_list += _get_object_size_tree(obj.__dict__, indent + '   ', 'dict')
            return size_list
        return size_list
    _get_object_size_tree(obj, '', 'obj')


def get_object_size_str(obj, lbl=''):
    from . import util_str
    nBytes = get_object_size(obj)
    sizestr = lbl + util_str.byte_str2(nBytes)
    return sizestr


def print_object_size(obj, lbl=''):
    print(get_object_size_str(obj, lbl=lbl))


def get_object_base():
    from .DynamicStruct import DynStruct
    from .util_classes import AutoReloader
    if '--min-base' in sys.argv:
        return object
    elif '--noreload-base' not in sys.argv:
        return AutoReloader
    elif '--dyn-base' in sys.argv:
        return DynStruct


def get_cython_exe():
    from . import util_cplat
    if util_cplat.WIN32:
        cython_exe = r'C:\Python27\Scripts\cython.exe'
        if not exists(cython_exe):
            cython_exe = 'cython.py'
    else:
        cython_exe = 'cython'
    return cython_exe


def compile_cython(fpath, clean=True):
    r""" compiles pyx -> pyd/dylib/so


    REAL SETUP.PY OUTPUT
    cythoning vtool/linalg_cython.pyx to vtool\linalg_cython.c
    C:\MinGW\bin\gcc.exe -mdll -O -Wall ^
            -IC:\Python27\Lib\site-packages\numpy\core\include ^
            -IC:\Python27\include -IC:\Python27\PC ^
            -c vtool\linalg_cython.c ^
            -o build\temp.win32-2.7\Release\vtool\linalg_cython.o


    writing build\temp.win32-2.7\Release\vtool\linalg_cython.def


    C:\MinGW\bin\gcc.exe -shared \
            -s \
            build\temp.win32-2.7\Release\vtool\linalg_cython.o \
            build\temp.win32-2.7\Release\vtool\linalg_cython.def \
            -LC:\Python27\libs \
            -LC:\Python27\PCbuild \
            -lpython27 \
            -lmsvcr90 \
            -o build\lib.win32-2.7\vtool\linalg_cython.pyd

    """
    from . import util_cplat
    from . import util_path
    import utool
    import os

    # Get autogenerated filenames
    fpath = util_path.truepath(fpath)
    dpath_, fname_ = split(fpath)
    dpath = relpath(dpath_, os.getcwd())
    fname, ext = splitext(fname_)
    # Prefer pyx over py
    if exists(fname + '.pyx'):
        fpath = fname + '.pyx'
    fname_c  = join(dpath, fname + '.c')
    fname_lib = join(dpath, fname + util_cplat.get_pylib_ext())

    print('[utool.compile_cython] fpath=%r' % (fpath,))
    print(' --- PRECHECKS --- ')
    if clean:
        utool.delete(fname_c)
        utool.delete(fname_lib)

    utool.checkpath(fpath, verbose=True, n=4)
    utool.checkpath(fname_c, verbose=True, info=False, n=4)
    utool.checkpath(fname_lib, verbose=True, info=False, n=4)

    # Cython build arguments
    cython_exe = get_cython_exe()
    if util_cplat.WIN32:
        os.environ['LDFLAGS'] = '-march=i486'
        os.environ['CFLAGS'] = '-march=i486'
        cc_exe = r'C:\MinGW\bin\gcc.exe'
        pyinclude_list = [
            r'C:\Python27\Lib\site-packages\numpy\core\include',
            r'C:\Python27\include',
            r'C:\Python27\PC',
            np.get_include()]
        pylib_list     = [
            r'C:\Python27\libs',
            r'C:\Python27\PCbuild'
            #r'C:\Python27\DLLS',
        ]
        plat_gcc_flags = ' '.join([
            '-mdll',
            '-O',
            '-DNPY_NO_DEPRECATED_API',
            #'-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
            '-Wall',
            '-Wno-unknown-pragmas',
            '-Wno-format',
            '-Wno-unused-function',
            '-m32',
            '-shared',
            #'-fPIC',
            #'-fwrapv',
        ])
        #plat_gcc_flags = ' '.join([
        #    '-shared',
        #    '-m32',
        #    '-mdll',
        #    '-march=i486',
        #    '-O',
        #])
        plat_link_flags = '-lpython27 -lmsvcr90'

#C:\MinGW\bin\gcc.exe -shared -s build\temp.win32-2.7\Release\vtool\linalg_cython.o build\temp.win32-2.7\Release\vtool\linalg_cython.def -LC:\Python27\libs -LC:\Python27\PCbuild -lpython27 -lmsvcr90 -o  build\lib.win32-2.7\linalg_cython.pyd

    else:
        cc_exe = 'gcc'
        cython_exe = 'cython'
        pyinclude_list = [r'/usr/include/python2.7', np.get_include()]
        pylib_list     = []
        plat_gcc_flags = ' '.join([
            '-shared',
            '-pthread',
            '-fPIC',
            '-fwrapv',
            '-O2',
            '-Wall',
            '-fno-strict-aliasing',
        ])
    #C:\MinGW\bin\gcc.exe -mdll -O -Wall -IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include -IC:\Python27\PC -c vtool\linalg_cython.c -o build\temp.win32-2.7\Release\vtool\linalg_cyth

    pyinclude = '' if len(pyinclude_list) == 0 else '-I' + ' -I'.join(pyinclude_list)
    pylib     = '' if len(pylib_list)     == 0 else '-L' + ' -L'.join(pylib_list)
    gcc_flag_list = [
        plat_gcc_flags,
        pyinclude,
        pylib,
        plat_link_flags,
    ]
    gcc_flags = ' '.join(filter(lambda x: len(x) > 0, gcc_flag_list))
    gcc_build_cmd = cc_exe + ' ' + gcc_flags + ' -o ' + fname_lib + ' -c ' + fname_c

    cython_build_cmd = cython_exe + ' ' + fpath

    # HACK
    print('\n --- CYTHON_COMMANDS ---')
    print(utool.pack_into(cython_build_cmd, textwidth=80, newline_prefix='  '))
    print('')
    print(utool.pack_into(gcc_build_cmd, textwidth=80, newline_prefix='  '))
    print(gcc_build_cmd)
    print('\n --- COMMAND_EXECUTION ---')

    def verbose_cmd(cmd):
        print('\n<CMD>')
        print(cmd)
        ret = os.system(cmd)
        print('> ret = %r' % ret)
        print('</CMD>\n')
        #print('-------------------')
        return ret

    ret = verbose_cmd(cython_build_cmd)
    assert utool.checkpath(fname_c, verbose=True, n=2), 'failed cython build'
    ret = verbose_cmd(gcc_build_cmd)
    assert utool.checkpath(fname_lib, verbose=True, n=2), 'failed gcc cython build'
    #try:
    #    #lib_dpath, lib_fname = split(fname_lib)
    #    #cwd = os.getcwd()
    #    #os.chdir(lib_dpath)
    #    ##exec('import ' + splitext(lib_fname)[0])
    #    #os.chdir(cwd)
    #    pass
    #except Exception:
    #    pass
    #    raise

    #out, err, ret = util_cplat.shell(cython_exe + ' ' + fpath)
    #out, err, ret = util_cplat.shell((cython_exe, fpath))
    #if ret == 0:
    #    out, err, ret = util_cplat.shell(cc_exe + ' ' + gcc_flags + ' -o ' + fname_so + ' ' + fname_c)
    return ret


def find_exe(name, path_hints=[], required=True):
    from . import util_cplat
    if util_cplat.WIN32 and not name.endswith('.exe'):
        name += '.exe'

    for path in path_hints:
        exe_fpath = join(path, name)
        if exists(exe_fpath):
            return exe_fpath

    if required:
        raise AssertionError('cannot find ' + name)


def _on_ctrl_c(signal, frame):
    print('Caught ctrl+c')
    sys.exit(0)


def init_catch_ctrl_c():
    import signal
    signal.signal(signal.SIGINT, _on_ctrl_c)


def reset_catch_ctrl_c():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # reset ctrl+c behavior


def is_developer():
    import utool
    return utool.get_computer_name() in ['Hyrule', 'Ooo', 'BakerStreet']
