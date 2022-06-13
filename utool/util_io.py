# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
from six.moves import cPickle as pickle
from utool import util_path
from utool import util_inject
from os.path import splitext, basename, exists
try:
    import lockfile
    HAVE_LOCKFILE = True
except ImportError:
    HAVE_LOCKFILE = False
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

print, rrr, profile = util_inject.inject2(__name__)


__PRINT_IO__ = True
__PRINT_WRITES__ = __PRINT_IO__
__PRINT_READS__  =  __PRINT_IO__
__FORCE_PRINT_READS__ = False
__FORCE_PRINT_WRITES__ = False
__READ_TAIL_N__ = 3
#__FORCE_PRINT_READS__ = True
#__FORCE_PRINT_WRITES__ = True


def load_data(fpath, **kwargs):
    """ More generic interface to load data """
    ext = splitext(fpath)[1]
    if ext in ['.pickle', '.cPkl', '.pkl']:
        return load_cPkl(fpath, **kwargs)
    elif ext in ['.json']:
        return load_json(fpath, **kwargs)
    elif ext in ['.hdf5']:
        return load_hdf5(fpath, **kwargs)
    elif ext in ['.txt']:
        return load_text(fpath, **kwargs)
    elif HAS_NUMPY and ext in ['.npz', '.npy']:
        return load_numpy(fpath, **kwargs)
    else:
        assert False, 'unknown ext=%r for fpath=%r' % (ext, fpath)


def save_data(fpath, data, **kwargs):
    """ More generic interface to write data """
    ext = splitext(fpath)[1]
    if ext in ['.pickle', '.cPkl', '.pkl']:
        return save_cPkl(fpath, data, **kwargs)
    elif ext in ['.json']:
        return save_json(fpath, data, **kwargs)
    elif ext in ['.hdf5']:
        return save_hdf5(fpath, data, **kwargs)
    elif ext in ['.txt']:
        return save_text(fpath, **kwargs)
    elif HAS_NUMPY and ext in ['.npz', '.npy']:
        return save_numpy(fpath, data, **kwargs)
    else:
        assert False, 'unknown ext=%r for fpath=%r' % (ext, fpath)


def _rectify_verb_write(verbose):
    if __FORCE_PRINT_WRITES__:
        return True
    if verbose is None:
        verbose = __PRINT_WRITES__
    return verbose


def _rectify_verb_read(verbose):
    if __FORCE_PRINT_READS__:
        return True
    if verbose is None:
        verbose = __PRINT_READS__
    return verbose


def write_to(fpath, to_write, aslines=False, verbose=None,
             onlyifdiff=False, mode='w', n=None):
    """ Writes text to a file. Automatically encodes text as utf8.

    Args:
        fpath (str): file path
        to_write (str): text to write (must be unicode text)
        aslines (bool): if True to_write is assumed to be a list of lines
        verbose (bool): verbosity flag
        onlyifdiff (bool): only writes if needed!
                checks hash of to_write vs the hash of the contents of fpath
        mode (unicode): (default = u'w')
        n (int):  (default = 2)

    CommandLine:
        python -m utool.util_io --exec-write_to --show

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_io import *  # NOQA
        >>> import utool as ut
        >>> fpath = ut.unixjoin(ut.get_app_resource_dir('utool'), 'testwrite.txt')
        >>> ut.delete(fpath)
        >>> to_write = 'utf-8 symbols Δ, Й, ק, م, ๗, あ, 叶, 葉, and 말.'
        >>> aslines = False
        >>> verbose = True
        >>> onlyifdiff = False
        >>> mode = u'w'
        >>> n = 2
        >>> write_to(fpath, to_write, aslines, verbose, onlyifdiff, mode, n)
        >>> read_ = ut.read_from(fpath)
        >>> print('read_    = ' + read_)
        >>> print('to_write = ' + to_write)
        >>> assert read_ == to_write
    """
    if onlyifdiff:
        import utool as ut
        if ut.hashstr(read_from(fpath)) == ut.hashstr(to_write):
            print('[util_io] * no difference')
            return
    verbose = _rectify_verb_write(verbose)
    if verbose:
        # n = None if verbose > 1 else 2
        # print('[util_io] * Writing to text file: %r ' % util_path.tail(fpath, n=n))
        print('[util_io] * Writing to text file: {}'.format(fpath))

    backup = False and exists(fpath)
    if backup:
        util_path.copy(fpath, fpath + '.backup')

    if not isinstance(fpath, six.string_types):
        # Assuming a file object with a name attribute
        # Should just read from the file
        fpath = fpath.name

    with open(fpath, mode) as file_:
        if aslines:
            file_.writelines(to_write)
        else:
            # Ensure python2 writes in bytes
            if six.PY2:
                if isinstance(to_write, unicode):  # NOQA
                    to_write = to_write.encode('utf8')
            try:
                file_.write(to_write)
            except UnicodeEncodeError as ex:
                start = max(ex.args[2] - 10, 0)
                end = ex.args[3] + 10
                context = to_write[start:end]
                print(repr(context))
                print(context)
                from utool import util_dbg
                util_dbg.printex(ex, keys=[(type, 'to_write')])
                file_.close()
                if backup:
                    # restore
                    util_path.copy(fpath + '.backup', fpath)
                # import utool
                # utool.embed()
                raise


def read_from(fpath, verbose=None, aslines=False, strict=True, n=None, errors='replace'):
    r""" Reads text from a file. Automatically returns utf8.

    Args:
        fpath (str): file path
        aslines (bool): if True returns list of lines
        verbose (bool): verbosity flag

    Returns:
        str: text from fpath (this is unicode)

    Ignore:
        x = b'''/whaleshark_003_fors\xc3\xb8g.wmv" />\r\n'''
        ut.writeto('foo.txt', x)
        y = ut.readfrom('foo.txt')
        y.encode('utf8') == x
    """
    if n is None:
        n = __READ_TAIL_N__
    verbose = _rectify_verb_read(verbose)
    if verbose:
        print('[util_io] * Reading text file: %r ' % util_path.tail(fpath, n=n))
    try:
        if not util_path.checkpath(fpath, verbose=verbose, n=n):
            raise IOError('[io] * FILE DOES NOT EXIST!')
        #with open(fpath, 'r') as file_:
        with open(fpath, 'rb') as file_:
            if aslines:
                #text = file_.readlines()
                if six.PY2:
                    # python2 writes in bytes, so read as bytes then convert to
                    # utf8
                    text = [line.decode('utf8', errors=errors)
                            for line in file_.readlines()]
                else:
                    text = [line.decode('utf8', errors=errors)
                            for line in file_.readlines()]
                    #text = file_.readlines()
            else:
                # text = file_.read()
                if six.PY2:
                    text = file_.read().decode('utf8', errors=errors)
                else:
                    #text = file_.read()
                    text = file_.read().decode('utf8', errors=errors)
        return text
    except IOError as ex:
        from utool import util_dbg
        if verbose or strict:
            util_dbg.printex(ex, ' * Error reading fpath=%r' %
                             util_path.tail(fpath, n=n), '[io]')
        if strict:
            raise


def read_lines_from(fpath, num_lines=None, verbose=None, n=None):
    with open(fpath, 'r') as file_:
        line_list = []
        #for lineno, line in enumerate(file_.readline()):
        for lineno, line in enumerate(file_):
            line_list.append(line)
            if num_lines is not None and lineno > num_lines:
                break
    return line_list

# aliases
readfrom = read_from
writeto = write_to
save_text = write_to
load_text = read_from


def save_json(fpath, data, **kwargs):
    import utool as ut
    json_data = ut.to_json(data, **kwargs)
    ut.save_text(fpath, json_data)


def load_json(fpath):
    import utool as ut
    json_data = ut.load_text(fpath)
    data = ut.from_json(json_data)
    return data


def save_cPkl(fpath, data, verbose=None, n=None):
    """ Saves data to a pickled file with optional verbosity """
    verbose = _rectify_verb_write(verbose)
    if verbose:
        print('[util_io] * save_cPkl(%r, data)' % (util_path.tail(fpath, n=n),))
    with open(fpath, 'wb') as file_:
        # Use protocol 2 to support python2 and 3
        pickle.dump(data, file_, protocol=2)


def load_cPkl(fpath, verbose=None, n=None):
    r"""
    Loads a pickled file with optional verbosity.
    Aims for compatibility between python2 and python3.


    Ignore:
        >>> import utool as ut
        >>> def makedata_simple():
        >>>     data = np.empty((500, 2 ** 20), dtype=np.uint8) + 1
        >>>     return data
        >>> memtrack = ut.MemoryTracker()
        >>> # create a large amount of data
        >>> data = makedata_simple()
        >>> memtrack.report()
        >>> print(ut.get_object_size_str(data))
        >>> fpath = 'tmp.pkl'
        >>> ut.save_cPkl(fpath, data)
        >>> print(ut.get_file_nBytes_str('tmp.pkl'))
        >>> #del data
        >>> memtrack.collect()
        >>> memtrack.report()
        >>> data = ut.load_cPkl(fpath)
        >>> memtrack.report()

    Ignore:
        >>> def makedata_complex():
        >>>     rng = np.random.RandomState(42)
        >>>     item1 = np.empty((100, 2 ** 20), dtype=np.uint8) + 1
        >>>     item2 = [np.empty((10, 2 ** 10), dtype=np.uint8) + 1
        >>>              for a in range(1000)]
        >>>     item3 = {a: np.empty(int(rng.rand() * 10), dtype=np.int16) + 1
        >>>                  for a in range(100)}
        >>>     item4 = {np.int32(a): np.empty((int(rng.rand() * 10), 2), dtype=np.float64) + 1
        >>>                  for a in range(200)}
        >>>     data = {'item1': item1, 'item2': item2,
        >>>             'item3': item3, 'item4': item4}
        >>>     return data
        >>> memtrack = ut.MemoryTracker()
        >>> # create a large amount of data
        >>> data = makedata_complex()
        >>> memtrack.report()
        >>> print(ut.get_object_size_str(data))
        >>> fpath = 'tmp.pkl'
        >>> ut.save_cPkl(fpath, data)
        >>> print(ut.get_file_nBytes_str('tmp.pkl'))
        >>> #del data
        >>> memtrack.collect()
        >>> memtrack.report()
        >>> data2 = ut.load_cPkl(fpath)
        >>> memtrack.report()

    Ignore:
        >>> import utool as ut
        >>> memtrack = ut.MemoryTracker()
        >>> cacher = ut.Cacher('tmp', cache_dir='.', cfgstr='foo')
        >>> data3 = cacher.ensure(makedata_complex)
        >>> memtrack.report()
        >>> data4 = cacher.ensure(makedata_complex)
        >>> memtrack.report()
        >>> import utool as ut
        >>> memtrack = ut.MemoryTracker()
        >>> fpath = '/home/joncrall/Desktop/smkcache/inva_PZ_Master1VUUIDS((5616)vxihbjwtggyovrto)_vpgwpcafbjkkpjdf.cPkl'
        >>> print(ut.get_file_nBytes_str(fpath))
        >>> data = ut.load_cPkl(fpath)
        >>> memtrack.report()


        def makedata_complex():
            data = np.empty((1000, 2 ** 20), dtype=np.uint8)
            data[:] = 0
            return data
    """
    verbose = _rectify_verb_read(verbose)
    if verbose:
        print('[util_io] * load_cPkl(%r)' % (util_path.tail(fpath, n=n),))
    try:
        with open(fpath, 'rb') as file_:
            data = pickle.load(file_)
    except UnicodeDecodeError:
        if six.PY3:
            # try to open python2 pickle
            with open(fpath, 'rb') as file_:
                data = pickle.load(file_, encoding='latin1')
        else:
            raise
    except ValueError as ex:
        if six.PY2:
            if ex.message == 'unsupported pickle protocol: 4':
                raise ValueError(
                    'unsupported Python3 pickle protocol 4 '
                    'in Python2 for fpath=%r' % (fpath,))
            else:
                raise
        else:
            raise
    return data


def _python2_load_cpkl(fpath):
    """
    References:
        https://stackoverflow.com/questions/41720952/unpickle-sklearn-tree-descisiontreeregressor-in-python-2-from-python3
    """
    from lib2to3.fixes.fix_imports import MAPPING
    import sys
    import pickle

    # MAPPING maps Python 2 names to Python 3 names. We want this in reverse.
    REVERSE_MAPPING = {}
    for key, val in MAPPING.items():
        REVERSE_MAPPING[val] = key

    # We can override the Unpickler and loads
    class Python_3_Unpickler(pickle.Unpickler):
        """Class for pickling objects from Python 3"""
        def find_class(self, module, name):
            if module in REVERSE_MAPPING:
                module = REVERSE_MAPPING[module]
            __import__(module)
            mod = sys.modules[module]
            klass = getattr(mod, name)
            return klass

    def load(fpath):
        with open(fpath, 'rb') as file_:
            data = Python_3_Unpickler(file_).load()
        return data


def lock_and_load_cPkl(fpath, verbose=False):
    with lockfile.LockFile(fpath + '.lock'):
        return load_cPkl(fpath, verbose)


def lock_and_save_cPkl(fpath, data, verbose=False):
    with lockfile.LockFile(fpath + '.lock'):
        return save_cPkl(fpath, data, verbose)


def save_hdf5(fpath, data, verbose=None, compression='lzf'):
    r"""
    Restricted save of data using hdf5. Can only save ndarrays and dicts of
    ndarrays.

    Args:
        fpath (str):
        data (ndarray):
        compression (str):
            DEFLATE/GZIP - standard
            LZF  - fast
            SHUFFLE - compression ratio
            FLETCHER32 - error detection
            Scale-offset - integer / float scaling and truncation
            SZIP - fast and patented

    CommandLine:
        python -m utool.util_io --test-save_hdf5

    References:
        http://docs.h5py.org/en/latest/quick.html
        http://docs.h5py.org/en/latest/mpi.html

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_io import *  # NOQA
        >>> import numpy as np
        >>> import utool as ut
        >>> rng = np.random.RandomState(0)
        >>> data = (rng.rand(100000, 128) * 255).astype(np.uint8).copy()
        >>> verbose = True
        >>> fpath = ut.unixjoin(ut.ensure_app_resource_dir('utool'), 'myfile.hdf5')
        >>> compression = 'lzf'
        >>> ut.delete(fpath)
        >>> save_hdf5(fpath, data, verbose, compression)
        >>> data2 = load_hdf5(fpath, verbose)
        >>> assert data is not data2
        >>> assert np.all(data == data2)
        >>> assert ut.delete(fpath)

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_io import *  # NOQA
        >>> import numpy as np
        >>> import utool as ut
        >>> rng = np.random.RandomState(0)
        >>> data = {'name': 'foobar', 'x': [1, 2, 3], 'y': np.array([3, 2, 1])}
        >>> ut.exec_funckw(save_hdf5, globals())
        >>> fpath = ut.unixjoin(ut.ensure_app_resource_dir('utool'), 'myfile2.hdf5')
        >>> ut.delete(fpath)
        >>> save_hdf5(fpath, data, verbose, compression)
        >>> data2 = load_hdf5(fpath, verbose)
        >>> assert data is not data2
        >>> assert all([np.all(data[key] == data2[key]) for key in data.keys()])
        >>> assert ut.delete(fpath)

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> # cPkl / numpy seems to be faster with this initial implementation
        >>> import utool as ut
        >>> data = (rng.rand(1000000, 128) * 255).astype(np.uint8).copy()
        >>> print(ut.get_object_size_str(data))
        >>> del data
        >>> setup = ut.codeblock(
        >>>     '''
                import numpy as np
                import utool as ut
                rng = np.random.RandomState(0)
                fpath = ut.unixjoin(ut.ensure_app_resource_dir('utool'), 'io_test_data')
                data = (rng.rand(1000000, 128) * 255).astype(np.uint8).copy()
                #print(ut.get_object_size_str(data))
                ''')
        >>> # Test load time
        >>> stmt_list1 = ut.codeblock(
        >>>     '''
                ut.save_hdf5(fpath + '.hdf5', data, verbose=False, compression='gzip')
                ut.save_hdf5(fpath + '.hdf5', data, verbose=False, compression='lzf')
                ut.save_cPkl(fpath + '.cPkl', data, verbose=False)
                ut.save_numpy(fpath + '.npy', data, verbose=False)
                ut.save_pytables(fpath + '.tables', data, verbose=False)
                ''').split('\n')
        >>> ut.util_dev.timeit_compare(stmt_list1, setup, int(10))
        >>> # Test save time
        >>> stmt_list2 = ut.codeblock(
        >>>     '''
                ut.load_hdf5(fpath + '.hdf5', verbose=False)
                ut.load_cPkl(fpath + '.cPkl', verbose=False)
                ut.load_numpy(fpath + '.npy', verbose=False)
                ut.load_pytables(fpath + '.tables', verbose=False)
                ''').split('\n')
        >>> ut.util_dev.timeit_compare(stmt_list2, setup, int(10))
        >>> print('finished timeing')
        +----------------
        | TIMEIT COMPARE
        +----------------
        | iterations = 10
        | Input:
        |     | num | stmt
        |     |   0 | u"ut.save_hdf5(fpath + '.hdf5', data, verbose=False, compression='gzip')"
        |     |   1 | u"ut.save_hdf5(fpath + '.hdf5', data, verbose=False, compression='lzf')"
        |     |   2 | u"ut.save_cPkl(fpath + '.cPkl', data, verbose=False)"
        |     |   3 | u"ut.save_numpy(fpath + '.npy', data, verbose=False)"
        |     |   4 | u"ut.save_pytables(fpath + '.tables', data, verbose=False)"
        ...
        | Output:
        |    * PASSED: each statement produced the same result
        |    | num | total time | per loop | stmt
        |    |   0 |    0.03 ks |   3.15 s | ut.save_hdf5(fpath + '.hdf5', data, verbose=False, compression='gzip')
        |    |   1 |    0.01 ks |   1.25 s | ut.save_hdf5(fpath + '.hdf5', data, verbose=False, compression='lzf')
        |    |   2 |     5.30 s |   0.53 s | ut.save_cPkl(fpath + '.cPkl', data, verbose=False)
        |    |   3 |     4.97 s |   0.50 s | ut.save_numpy(fpath + '.npy', data, verbose=False)
        |    |   4 |     9.23 s |   0.92 s | ut.save_pytables(fpath + '.tables', data, verbose=False)
        L_________________
        +----------------
        | TIMEIT COMPARE
        +----------------
        | iterations = 10
        | Input:
        |     | num | stmt
        |     |   0 | u"ut.load_hdf5(fpath + '.hdf5', verbose=False)"
        |     |   1 | u"ut.load_cPkl(fpath + '.cPkl', verbose=False)"
        |     |   2 | u"ut.load_numpy(fpath + '.npy', verbose=False)"
        |     |   3 | u"ut.load_pytables(fpath + '.tables', verbose=False)"
        ...
        | Output:
        |    * PASSED: each statement produced the same result
        |    | num | total time | per loop | stmt
        |    |   0 |     2.39 s |   0.24 s | ut.load_hdf5(fpath + '.hdf5', verbose=False)
        |    |   1 |     0.39 s |   0.04 s | ut.load_cPkl(fpath + '.cPkl', verbose=False)
        |    |   2 |     0.19 s |   0.02 s | ut.load_numpy(fpath + '.npy', verbose=False)
        |    |   3 |     0.33 s |   0.03 s | ut.load_pytables(fpath + '.tables', verbose=False)
        L_________________

    Ignore:
        %timeit save_hdf5(fpath, data, verbose=False, compression='gzip')
        %timeit save_hdf5(fpath, data, verbose=False, compression='lzf')
        %timeit save_cPkl(fpath + '.cPkl', data, verbose=False)
        %timeit save_pytables(fpath + '.tables', data, verbose=False)
        1 loops, best of 3: 258 ms per loop
        10 loops, best of 3: 111 ms per loop
        10 loops, best of 3: 53.1 ms per loop
        10 loops, best of 3: 96.5 ms per loop

        save_hdf5(fpath, data, verbose=False, compression='gzip')
        %timeit load_hdf5(fpath, verbose=False)
        save_hdf5(fpath, data, verbose=False, compression='lzf')
        %timeit load_hdf5(fpath, verbose=False)
        %timeit load_cPkl(fpath + '.cPkl', verbose=False)
        %timeit load_pytables(fpath + '.tables', verbose=False)
        100 loops, best of 3: 19.4 ms per loop
        100 loops, best of 3: 14.4 ms per loop
        100 loops, best of 3: 3.92 ms per loop
        100 loops, best of 3: 6.22 ms per loop

    Notes:
        pip install mpi4py
    """
    import h5py

    verbose = _rectify_verb_write(verbose)
    if verbose:
        print('[util_io] * save_hdf5(%r, data)' % (util_path.tail(fpath),))
    if verbose > 1:
        if isinstance(data, dict):
            print('[util_io] ... shapes=%r' % ([val.shape for val in data.values()],))
        else:
            print('[util_io] ... shape=%r' % (data.shape,))

    chunks = True  # True enables auto-chunking
    fname = basename(fpath)

    # check for parallel hdf5
    #have_mpi = h5py.h5.get_config().mpi
    #if have_mpi:
    #    import mpi4py
    #    h5kw = dict(driver='mpio', comm=mpi4py.MPI.COMM_WORLD)
    #    # cant use compression with mpi
    #    #ValueError: Unable to create dataset (Parallel i/o does not support filters yet)
    #else:
    h5kw = {}

    if isinstance(data, dict):
        array_data = {key: val for key, val in data.items()
                      if isinstance(val, (list, np.ndarray))}
        attr_data = {key: val for key, val in data.items() if key not in array_data}

        #assert all([
        #    isinstance(vals, np.ndarray)
        #    for vals in six.itervalues(data)
        #]), ('can only save dicts as ndarrays')
        # file_ = h5py.File(fpath, 'w', **h5kw)
        with h5py.File(fpath, mode='w', **h5kw) as file_:
            grp = file_.create_group(fname)
            for key, val in six.iteritems(array_data):
                val = np.asarray(val)
                dset = grp.create_dataset(
                    key, val.shape,  val.dtype, chunks=chunks,
                    compression=compression)
                dset[...] = val
            for key, val in six.iteritems(attr_data):
                grp.attrs[key] = val
    else:
        assert isinstance(data, np.ndarray)
        shape = data.shape
        dtype = data.dtype
        #if verbose or (verbose is None and __PRINT_WRITES__):
        #    print('[util_io] * save_hdf5(%r, data)' % (util_path.tail(fpath),))
        # file_ = h5py.File(fpath, 'w', **h5kw)
        with h5py.File(fpath, mode='w', **h5kw) as file_:
            #file_.create_dataset(
            #    fname, shape,  dtype, chunks=chunks, compression=compression,
            #    data=data)
            dset = file_.create_dataset(
                fname, shape,  dtype, chunks=chunks, compression=compression)
            dset[...] = data


def load_hdf5(fpath, verbose=None):
    import h5py
    fname = basename(fpath)
    #file_ = h5py.File(fpath, 'r')
    #file_.values()
    #file_.keys()
    verbose = _rectify_verb_read(verbose)
    if verbose:
        print('[util_io] * load_hdf5(%r)' % (util_path.tail(fpath),))
    with h5py.File(fpath, 'r') as file_:
        value = file_[fname]
        if isinstance(value, h5py.Group):
            grp = value
            data = {}
            for key, dset in six.iteritems(grp):
                shape = dset.shape
                dtype = dset.dtype
                subdata = np.empty(shape, dtype=dtype)
                dset.read_direct(subdata)
                data[key] = subdata
            for key, val in six.iteritems(grp.attrs):
                data[key] = val
        elif isinstance(value, h5py.Dataset):
            dset = value
            shape = dset.shape
            dtype = dset.dtype
            data = np.empty(shape, dtype=dtype)
            dset.read_direct(data)
        else:
            assert False
    return data


def save_pytables(fpath, data, verbose=False):
    """
    sudo pip install numexpr
    sudo pip install tables

    References:
        https://pytables.github.io/cookbook/py2exe_howto.html
        https://gist.github.com/andrewgiessel/7515520
        http://stackoverflow.com/questions/8843062/python-how-to-store-a-numpy-multidimensional-array-in-pytables
        http://pytables.github.io/usersguide/tutorials.html#creating-new-array-objects

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_io import *  # NOQA
        >>> import numpy as np
        >>> import utool as ut
        >>> # build test data
        >>> verbose = True
        >>> fpath = 'myfile.pytables.hdf5'
        >>> np.random.seed(0)
        >>> compression = 'gzip'
        >>> data = (np.random.rand(100000, 128) * 255).astype(np.uint8).copy()
        >>> # execute function
        >>> ut.delete(fpath)
        >>> save_pytables(fpath, data, verbose)
        >>> data2 = load_pytables(fpath, verbose)
        >>> assert data is not data2
        >>> assert np.all(data == data2)
        >>> assert ut.delete(fpath)
    """
    import tables
    #from os.path import basename
    #fname = basename(fpath)
    #shape = data.shape
    #dtype = data.dtype
    #file_ = tables.open_file(fpath)
    verbose = _rectify_verb_write(verbose)
    if verbose:
        print('[util_io] * save_pytables(%r, data)' % (util_path.tail(fpath),))
    with tables.open_file(fpath, 'w') as file_:
        atom = tables.Atom.from_dtype(data.dtype)
        filters = tables.Filters(complib='blosc', complevel=5)
        dset = file_.createCArray(file_.root, 'data', atom, data.shape, filters=filters)
        # save w/o compressive filter
        #dset = file_.createCArray(file_.root, 'all_data', atom, all_data.shape)
        dset[:] = data


def load_pytables(fpath, verbose=False):
    import tables
    #from os.path import basename
    #fname = basename(fpath)
    #file_ = tables.open_file(fpath)
    verbose = _rectify_verb_read(verbose)
    if verbose:
        print('[util_io] * load_pytables(%r, data)' % (util_path.tail(fpath),))
    with tables.open_file(fpath, 'r') as file_:
        data = file_.root.data.read()
    return data


def load_numpy(fpath, mmap_mode=None, verbose=None):
    verbose = _rectify_verb_read(verbose)
    if verbose:
        print('[util_io] * load_numpy(%r)' % util_path.tail(fpath))
    return np.load(fpath, mmap_mode=mmap_mode)


def save_numpy(fpath, data, verbose=None, **kwargs):
    verbose = _rectify_verb_write(verbose)
    if verbose:
        print('[util_io] * save_numpy(%r, data)' % util_path.tail(fpath))
    return np.save(fpath, data)


#def save_capnp(fpath, data, verbose=False):
#    r"""
#    Refernces:
#        http://jparyani.github.io/pycapnp/quickstart.html#dictionaries
#    """
#    import capnp
#    if verbose or __PRINT_WRITES__:
#        print('[util_io] * save_capnp(%r, data)' % (util_path.tail(fpath),))


def try_decode(x):
    # All python encoding formats
    codec_list = [
        'ascii', 'big5', 'big5hkscs', 'cp037', 'cp424', 'cp437', 'cp500',
        'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857',
        'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866',
        'cp869', 'cp874', 'cp875', 'cp932', 'cp949', 'cp950', 'cp1006',
        'cp1026', 'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254',
        'cp1255', 'cp1256', 'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004',
        'euc_jisx0213', 'euc_kr', 'gb2312', 'gbk', 'gb18030', 'hz',
        'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2', 'iso2022_jp_2004',
        'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1', 'iso8859_2',
        'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7',
        'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_13', 'iso8859_14',
        'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_u', 'mac_cyrillic',
        'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman', 'mac_turkish',
        'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213', 'utf_32',
        'utf_32_be', 'utf_32_le', 'utf_16', 'utf_16_be', 'utf_16_le', 'utf_7',
        'utf_8', 'utf_8_sig', ]
    for codec in codec_list:
        try:
            print(('%20s: ' % (codec,)) + x.encode(codec))
        except Exception:
            print(('%20s: ' % (codec,)) + 'FAILED')

    for codec in codec_list:
        try:
            print(('%20s: ' % (codec,)) + x.decode(codec))
        except Exception:
            print(('%20s: ' % (codec,)) + 'FAILED')


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_io
        python -m utool.util_io --allexamples
        python -m utool.util_io --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
