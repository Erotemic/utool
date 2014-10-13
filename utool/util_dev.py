from __future__ import absolute_import, division, print_function
import sys
import six
import re
import os
import warnings
import weakref
from collections import OrderedDict
try:
    import numpy as np
except ImportError as ex:
    pass
from os.path import splitext, exists, join, split, relpath
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


#try:
#    import numpy as np
#    REUSABLE_ITERABLE_TYPES = (list, tuple, np.ndarray)
#except ImportError as ex:
#    REUSABLE_ITERABLE_TYPES = (list, tuple)


#def ensure_vararg_list(varargs):
#    """
#    It is useful to have a function take a list of objects to act upon.
#    But sometimes you want just one. Varargs lets you pass in as many as you
#    want, and it lets you have just one if needbe.
#    But sometimes the function caller explicitly passes in the list. In that
#    case we parse it out
#    """
#    if len(varargs) == 1:
#        if isinstance(varargs[0], REUSABLE_ITERABLE_TYPES):
#            return varargs[0]
#    return varargs


def timeit_compare(stmt_list, setup='', iterations=100000, verbose=True,
                   strict=False):
    """
    Example:
        >>> import utool
        >>> setup = utool.unindent(
            '''
            import numpy as np
            np.random.seed(0)
            invVR_mats = np.random.rand(1000, 3, 3).astype(np.float64)
            ''')
        >>> stmt1 = 'invVR_mats[:, 0:2, 2].T'
        >>> stmt2 = 'invVR_mats.T[2, 0:2]'
        >>> iterations = 100000
        >>> verbose = True
        >>> stmt_list = [stmt1, stmt2]
        >>> utool.timeit_compare(stmt_list, setup='', iterations=1000, verbose=True)
    """
    import timeit
    import utool
    if verbose:
        print('+----------------')
        print('| TIMEIT COMPARE')
        print('+----------------')
        print('| iterations = %d' % (iterations,))
        print('| Input:')
        #print('|     +------------')
        print('|     | num | stmt')
        for count, stmt in enumerate(stmt_list):
            print('|     | %3d | %r' % (count, stmt))
        sys.stdout.flush()
        #print('+     L________________')

    result_list = [testit(stmt, setup) for stmt in stmt_list]
    time_list   = [timeit.timeit(stmt, setup=setup, number=iterations)
                   for stmt in stmt_list]

    if verbose:
        print('| Output:')
        valid_results = utool.util_list.list_allsame(result_list)
        if not valid_results:
            print('|    * RESULTS ARE NOT VALID!!!')
            print('| Results:')
            for result in result_list:
                for count, result in enumerate(result_list):
                    print('<Result %d>' % count)
                    print(result)
                    print('</Result %d>' % count)
            if strict:
                raise AssertionError('Results are not valid')
        else:
            print('|    * each statement produced the same result')
        #print('|    +-----------------------------------')
        print('|    | num | total time | per loop | stmt')
        for count, tup in enumerate(zip(stmt_list, time_list)):
            stmt, time = tup
            print('|    | %3d | %10s | %8s | %s' %
                  (count, utool.seconds_str(time),
                   utool.seconds_str(time / iterations), stmt))
        #print('|    L___________________________________')
        print('L_________________')


def testit(stmt, setup):
    # Make temporary locals/globals for a sandboxlike run
    _globals = {}
    try:
        exec(setup, _globals)
    except Exception as ex:
        import utool
        print('Setup Error')
        print(setup)
        print('---')
        utool.printex(ex, 'error executing setup', keys=['setup'])
        raise
    try:
        result = eval(stmt, _globals)
    except Exception as ex:
        import utool
        print('Statement Error')
        print(setup)
        print('---')
        print(stmt)
        utool.printex(ex, 'error executing statement', keys=['stmt'])
        raise
    return result


def memory_dump():
    """
    References:
       from http://stackoverflow.com/questions/141351/how-do-i-find-what-is-using-memory-in-a-python-process-in-a-production-system
    """
    import gc
    import cPickle
    dump = open("memory.pickle", 'w')
    for obj in gc.get_objects():
        i = id(obj)
        size = sys.getsizeof(obj, 0)
        #    referrers = [id(o) for o in gc.get_referrers(obj) if hasattr(o, '__class__')]
        referents = [id(o) for o in gc.get_referents(obj) if hasattr(o, '__class__')]
        if hasattr(obj, '__class__'):
            cls = str(obj.__class__)
            cPickle.dump({'id': i, 'class': cls, 'size': size, 'referents': referents}, dump)


class MemoryTracker(object):
    """
    Lightweight ``class`` for tracking memory usage.
    On initialization it logs the current available (free) memory.
    Calling the report method logs the current available memory as well
    as memory usage difference w.r.t the last report.

    Example:
        >>> import utool
        >>> import numpy as np
        >>> memtrack = utool.MemoryTracker('[ENTRY]')
        >>> memtrack.report('[BEFORE_CREATE]')
        >>> arr = np.ones(128 * (2 ** 20), dtype=np.uint8)
        >>> memtrack.report('[AFTER_CREATE]')
        >>> memtrack.track_obj(arr, 'arr')
        >>> memtrack.report_objs()
        >>> del arr
        >>> memtrack.report('[DELETE]')
    """
    def __init__(self, lbl='Memtrack Init'):
        self.prev_nBytes = None
        self.weakref_dict = weakref.WeakValueDictionary()
        self.weakref_dict2 = {}
        self.report(lbl)

    def __call__(self, lbl=''):
        self.report(lbl=lbl)

    def report(self, lbl=''):
        from .util_str import byte_str2
        import gc
        gc.collect()
        nBytes = self.get_available_memory()
        print('[memtrack] +----')
        if self.prev_nBytes is not None:
            diff = self.prev_nBytes - nBytes
            print('[memtrack] | %s MemDiff = %s' % (lbl, byte_str2(diff)))
        else:
            print('[memtrack] | new MemoryTracker(%s)' % (lbl,))
        print('[memtrack] | Available Memory = %s' %  (byte_str2(nBytes),))
        print('[memtrack] L----')
        self.prev_nBytes = nBytes
        self.report_objs()

    def get_available_memory(self):
        from .util_resources import available_memory
        return available_memory()

    def track_obj(self, obj, name):
        oid = id(obj)
        #obj_weakref = weakref.ref(obj)
        self.weakref_dict[oid] = obj
        self.weakref_dict2[oid] = name
        del obj

    def report_objs(self):
        if len(self.weakref_dict) == 0:
            return
        print('[memtrack] +----')
        for oid in self.weakref_dict.iterkeys():
            obj = self.weakref_dict[oid]
            name = self.weakref_dict2[oid]
            report_memsize(weakref.ref(obj), name)
            del obj

        print('[memtrack] L----')


def report_memsize(obj, name=None, verbose=True):
    import gc
    #import types
    import utool
    if name is None:
        name = 'obj'
    referents = gc.get_referents(obj)
    referers  = gc.get_referrers(obj)
    print('+----')
    with utool.Indenter('| '):
        print('Memsize: ')
        print('%s is using: %s' % (name, utool.get_object_size_str(obj)))
        print('%s has %d referents' % (name, len(referents)))
        print('%s has %d referers' % (name, len(referers)))
        if verbose:
            if len(referers) > 0:
                for count, referer in enumerate(referers):
                    print('  <Referer %d>' % count)
                    print('    type(referer) = %r' % type(referer))
                    try:
                        #if isinstance(referer, frames.FrameType)
                        print('    frame(referer).f_code.co_name = %s' % (referer.f_code.co_name))
                    except Exception:
                        pass
                    print('    id(referer) = %r' % id(referer))
                    #print('referer = ' + utool.truncate_str(repr(referer)))
                    print('  </Referer %d>' % count)
        del obj
        del referents
        del referers
    print('L____')


def get_stats(_list, axis=None):
    """
    Args:
        _list (listlike): values to get statistics of
        axis (int): if ``_list`` is ndarray then this specifies the axis

    Returns:
        OrderedDict: stat_dict - dictionary of common numpy statistics
            (min, max, mean, std, nMin, nMax, shape)

    Examples:
        >>> import numpy as np
        >>> import utool
        >>> axis = 0
        >>> _list = np.random.rand(10, 2)
        >>> utool.get_stats(_list, axis=axis)
    """
    # Assure input is in numpy format
    if isinstance(_list, np.ndarray):
        nparr = _list
    elif isinstance(_list, list):
        nparr = np.array(_list)
    else:
        _list = list(_list)
        nparr = np.array(_list)
    # Check to make sure stats are feasible
    if len(_list) == 0:
        stat_dict = {'empty_list': True}
    else:
        # Compute stats
        min_val = nparr.min(axis=axis)
        max_val = nparr.max(axis=axis)
        mean_ = nparr.mean(axis=axis)
        std_  = nparr.std(axis=axis)
        # number of entries with min val
        nMin = np.sum(nparr == min_val, axis=axis)
        # number of entries with min val
        nMax = np.sum(nparr == max_val, axis=axis)
        stat_dict = OrderedDict(
            [('max',   np.float32(max_val)),
             ('min',   np.float32(min_val)),
             ('mean',  np.float32(mean_)),
             ('std',   np.float32(std_)),
             ('nMin',  np.int32(nMin)),
             ('nMax',  np.int32(nMax)),
             ('shape', repr(nparr.shape))])
    return stat_dict


def tuples_to_unique_scalars(tup_list):
    seen = {}
    def addval(tup):
        val = len(seen)
        seen[tup] = val
        return val
    scalar_list = [seen[tup] if tup in seen else addval(tup) for tup in tup_list]
    return scalar_list


# --- Info Strings ---


def get_stats_str(_list, newlines=False):
    """
    Returns the string version of get_stats
    """
    from .util_str import dict_str
    stat_dict = get_stats(_list)
    stat_str  = dict_str(stat_dict, strvals=True, newlines=newlines)
    #stat_strs = ['%r: %s' % (key, val) for key, val in six.iteritems(stat_dict)]
    #if newlines:
    #    indent = '    '
    #    head = '{\n' + indent
    #    sep  = ',\n' + indent
    #    tail = '\n}'
    #else:
    #    head = '{'
    #    sep = ', '
    #    tail = '}'
    #stat_str = head + sep.join(stat_strs) + tail
    return stat_str


def print_stats(_list, lbl=None, newlines=False):
    """
    Prints string representation of stat of _list
    """
    if lbl is not None:
        print('Mystats for %s' % lbl)
    stat_str = get_stats_str(_list, newlines=newlines)
    print(stat_str)


def npArrInfo(arr):
    """
    OLD update and refactor
    """
    from .DynamicStruct import DynStruct
    info = DynStruct()
    info.shapestr  = '[' + ' x '.join([str(x) for x in arr.shape]) + ']'
    info.dtypestr  = str(arr.dtype)
    if info.dtypestr == 'bool':
        info.bittotal = 'T=%d, F=%d' % (sum(arr), sum(1 - arr))
    elif info.dtypestr == 'object':
        info.minmaxstr = 'NA'
    elif info.dtypestr[0] == '|':
        info.minmaxstr = 'NA'
    else:
        if arr.size > 0:
            info.minmaxstr = '(%r, %r)' % (arr.min(), arr.max())
        else:
            info.minmaxstr = '(None)'
    return info


def printableType(val, name=None, parent=None):
    """
    Tries to make a nice type string for a value.
    Can also pass in a Printable parent object
    """
    if parent is not None and hasattr(parent, 'customPrintableType'):
        # Hack for non - trivial preference types
        _typestr = parent.customPrintableType(name)
        if _typestr is not None:
            return _typestr
    if isinstance(val, np.ndarray):
        info = npArrInfo(val)
        _typestr = info.dtypestr
    elif isinstance(val, object):
        _typestr = val.__class__.__name__
    else:
        _typestr = str(type(val))
        _typestr = _typestr.replace('type', '')
        _typestr = re.sub('[\'><]', '', _typestr)
        _typestr = re.sub('  *', ' ', _typestr)
        _typestr = _typestr.strip()
    return _typestr


def printableVal(val, type_bit=True, justlength=False):
    """
    Very old way of doing pretty printing. Need to update and refactor.
    """
    from . import util_dev
    # Move to util_dev
    # NUMPY ARRAY
    if type(val) is np.ndarray:
        info = npArrInfo(val)
        if info.dtypestr.startswith('bool'):
            _valstr = '{ shape:' + info.shapestr + ' bittotal: ' + info.bittotal + '}'  # + '\n  |_____'
        elif info.dtypestr.startswith('float'):
            _valstr = util_dev.get_stats_str(val)
        else:
            _valstr = '{ shape:' + info.shapestr + ' mM:' + info.minmaxstr + ' }'  # + '\n  |_____'
    # String
    elif isinstance(val, (str, unicode)):
        _valstr = '\'%s\'' % val
    # List
    elif isinstance(val, list):
        if justlength or len(val) > 30:
            _valstr = 'len=' + str(len(val))
        else:
            _valstr = '[ ' + (', \n  '.join([str(v) for v in val])) + ' ]'
    elif hasattr(val, 'get_printable') and type(val) != type:  # WTF? isinstance(val, AbstractPrintable):
        _valstr = val.get_printable(type_bit=type_bit)
    elif isinstance(val, dict):
        _valstr = '{\n'
        for val_key in val.keys():
            val_val = val[val_key]
            _valstr += '  ' + str(val_key) + ' : ' + str(val_val) + '\n'
        _valstr += '}'
    else:
        _valstr = str(val)
    if _valstr.find('\n') > 0:  # Indent if necessary
        _valstr = _valstr.replace('\n', '\n    ')
        _valstr = '\n    ' + _valstr
    _valstr = re.sub('\n *$', '', _valstr)  # Replace empty lines
    return _valstr


def myprint(input_=None, prefix='', indent='', lbl=''):
    """
    OLD PRINT FUNCTION USED WITH PRINTABLE VAL
    TODO: Refactor and update
    """
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


def make_call_graph(func, *args, **kwargs):
    """ profile with pycallgraph

    Example:
        pycallgraph graphviz -- ./mypythonscript.py

    References:
        http://pycallgraph.slowchop.com/en/master/
    """
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    with PyCallGraph(output=GraphvizOutput):
        func(*args, **kwargs)


#def runprofile(cmd, globals_=globals(), locals_=locals()):
#    """ DEPRICATE. Tries to run a function and profile it """
#    # Meliae # from meliae import loader # om = loader.load('filename.json') # s = om.summarize();
#    #http://www.huyng.com/posts/python-performance-analysis/
#    #Once youve gotten your code setup with the <AT>profile decorator, use kernprof.py to run your script.
#    #kernprof.py -l -v fib.py
#    import cProfile
#    print('[util] Profiling Command: ' + cmd)
#    cProfOut_fpath = 'OpenGLContext.profile'
#    cProfile.runctx( cmd, globals_, locals_, filename=cProfOut_fpath)
#    # RUN SNAKE
#    print('[util] Profiled Output: ' + cProfOut_fpath)
#    if sys.platform == 'win32':
#        rsr_fpath = 'C:/Python27/Scripts/runsnake.exe'
#    else:
#        rsr_fpath = 'runsnake'
#    view_cmd = rsr_fpath + ' "' + cProfOut_fpath + '"'
#    os.system(view_cmd)
#    return True


def _memory_profile(with_gc=False):
    """
    Helper for memory debugging. Mostly just a namespace where I experiment with
    guppy and heapy.

    References:
        http://stackoverflow.com/questions/2629680/deciding-between-subprocess-multiprocessing-and-thread-in-python

    Reset Numpy Memory::
        %reset out
        %reset array
    """
    import utool
    import guppy
    if with_gc:
        garbage_collect()
    hp = guppy.hpy()
    print('[hpy] Waiting for heap output...')
    heap_output = hp.heap()
    print(heap_output)
    print('[hpy] total heap size: ' + utool.byte_str2(heap_output.size))
    utool.util_resources.memstats()
    # Graphical Browser
    #hp.pb()


def make_object_graph(obj, fpath='sample_graph.png'):
    """ memoryprofile with objgraph

    Examples:
        #import objgraph
        #objgraph.show_most_common_types()
        #objgraph.show_growth()
        #memtrack.report()
        #memtrack.report()
        #objgraph.show_growth()
        #import gc
        #gc.collect()
        #memtrack.report()
        #y = 0
        #objgraph.show_growth()
        #memtrack.report()
        #utool.embed()

    References:
        http://mg.pov.lt/objgraph/
    """
    import objgraph
    objgraph.show_most_common_types()
    #print(objgraph.by_type('ndarray'))
    #objgraph.find_backref_chain(
    #     random.choice(objgraph.by_type('ndarray')),
    #     objgraph.is_proper_module)
    objgraph.show_refs([obj], filename='ref_graph.png')
    objgraph.show_backrefs([obj], filename='backref_graph.png')


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

    def _get_object_size_tree(obj, indent='', lbl='obj', seen=None):
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
                size_list += _get_object_size_tree(item, indent + '   ', 'item', seen)
        elif isinstance(obj, dict):
            try:
                for key, val in six.iteritems(obj):
                    size_list += _get_object_size_tree(key, indent + '   ', key, seen)
                    size_list += _get_object_size_tree(val, indent + '   ', key, seen)
            except RuntimeError:
                print(key)
                raise
        elif isinstance(obj, object) and hasattr(obj, '__dict__'):
            size_list += _get_object_size_tree(obj.__dict__, indent + '   ', 'dict', seen)
            return size_list
        return size_list
    seen = set([])
    _get_object_size_tree(obj, '', 'obj', seen)
    del seen


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
    r""" Compiles a cython pyx into a shared library

    This seems broken
    compiles pyx -> pyd/dylib/so

    Examples:
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
    return utool.get_computer_name().lower() in ['hyrule', 'ooo', 'bakerstreet']
