from __future__ import absolute_import, division, print_function
from six.moves import cPickle as pickle
try:
    import lockfile
    HAVE_LOCKFILE = True
except ImportError:
    HAVE_LOCKFILE = False
from utool import util_path
from utool import util_inject
from os.path import splitext
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError as ex:
    HAS_NUMPY = False
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[io]')


__PRINT_IO__ = True
__PRINT_WRITES__ = __PRINT_IO__
__PRINT_READS__  =  __PRINT_IO__
__FORCE_PRINT_READS__ = False
__FORCE_PRINT_WRITES__ = False
#__FORCE_PRINT_READS__ = True
#__FORCE_PRINT_WRITES__ = True


def load_data(fpath, mmap_mode=None):
    """ More generic interface to load data """
    ext = splitext(fpath)[1]
    if ext in ['.pickle', '.cPkl', '.pkl']:
        return load_cPkl(fpath)
    elif ext in ['.hdf5']:
        return load_hdf5(fpath)
    elif ext in ['.txt']:
        return load_text(fpath)
    elif HAS_NUMPY and ext in ['.npz']:
        return load_numpy(fpath, mmap_mode=mmap_mode)
    else:
        assert False, 'unknown ext=%r for fpath=%r' % (ext, fpath)


def save_data(fpath, data):
    """ More generic interface to write data """
    ext = splitext(fpath)[1]
    if ext in ['.pickle', '.cPkl', '.pkl']:
        return save_cPkl(fpath, data)
    elif ext in ['.hdf5']:
        return save_hdf5(fpath, data)
    elif ext in ['.txt']:
        return save_text(fpath)
    elif HAS_NUMPY and ext in ['.npz']:
        # TODO save_numpy
        return np.save(fpath, data, )
    else:
        assert False, 'unknown ext=%r for fpath=%r' % (ext, fpath)


def write_to(fpath, to_write, aslines=False, verbose=None,
             onlyifdiff=False, mode='w', n=2):
    """ Writes text to a file

    Args:
        fpath (str): file path
        to_write (str): text to write
        aslines (bool): if True to_write is assumed to be a list of lines
        verbose (bool): verbosity flag
        onlyifdiff (bool): only writes if needed!
            checks hash of to_write vs the hash of the contents of fpath
    """
    if onlyifdiff:
        import utool as ut
        if ut.hashstr(read_from(fpath)) == ut.hashstr(to_write):
            print('[util_io] * no difference')
            return
    if verbose or (verbose is None and __PRINT_WRITES__) or __FORCE_PRINT_WRITES__:
        print('[util_io] * Writing to text file: %r ' % util_path.tail(fpath, n=2))
    with open(fpath, mode) as file_:
        if aslines:
            file_.writelines(to_write)
        else:
            file_.write(to_write)


def read_from(fpath, verbose=None, aslines=False, strict=True):
    """ Reads text from a file

    Args:
        fpath (str): file path
        aslines (bool): if True returns list of lines
        verbose (bool): verbosity flag

    Returns:
        text from fpath
    """
    if verbose or (verbose is None and __PRINT_READS__) or __FORCE_PRINT_READS__:
        print('[util_io] * Reading text file: %r ' % util_path.tail(fpath))
    try:
        if not util_path.checkpath(fpath, verbose=verbose, n=3):
            raise IOError('[io] * FILE DOES NOT EXIST!')
        with open(fpath, 'r') as file_:
            if aslines:
                text = file_.readlines()
            else:
                text = file_.read()
        return text
    except IOError as ex:
        from utool.util_dbg import printex
        if verbose or strict:
            printex(ex, ' * Error reading fpath=%r' %
                    util_path.tail(fpath), '[io]')
        if strict:
            raise


# aliases
readfrom = read_from
writeto = write_to
save_text = write_to
load_text = read_from


def save_cPkl(fpath, data, verbose=None):
    """ Saves data to a pickled file with optional verbosity """
    if verbose or (verbose is None and __PRINT_WRITES__) or __FORCE_PRINT_WRITES__:
        print('[util_io] * save_cPkl(%r, data)' % (util_path.tail(fpath),))
    with open(fpath, 'wb') as file_:
        pickle.dump(data, file_, pickle.HIGHEST_PROTOCOL)


def load_cPkl(fpath, verbose=None):
    """ Loads a pickled file with optional verbosity """
    if verbose or (verbose is None and __PRINT_READS__) or __FORCE_PRINT_READS__:
        print('[util_io] * load_cPkl(%r)' % (util_path.tail(fpath),))
    with open(fpath, 'rb') as file_:
        data = pickle.load(file_)
    return data


def lock_and_load_cPkl(fpath, verbose=False):
    with lockfile.LockFile(fpath + '.lock'):
        return load_cPkl(fpath, verbose)


def lock_and_save_cPkl(fpath, data, verbose=False):
    with lockfile.LockFile(fpath + '.lock'):
        return save_cPkl(fpath, data, verbose)


def save_hdf5(fpath, data, verbose=False, compression='lzf'):
    r"""
    restricted save of data using hdf5. Can only save ndarrays and dicts of ndarrays

    Args:
        fpath (?):
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

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_io import *  # NOQA
        >>> import numpy as np
        >>> import utool as ut
        >>> # build test data
        >>> rng = np.random.RandomState(0)
        >>> data = (rng.rand(100000, 128) * 255).astype(np.uint8).copy()
        >>> verbose = True
        >>> fpath = 'myfile.hdf5'
        >>> compression = 'lzf'
        >>> # execute function
        >>> ut.delete(fpath)
        >>> save_hdf5(fpath, data, verbose, compression)
        >>> data2 = load_hdf5(fpath, verbose)
        >>> assert data is not data2
        >>> assert np.all(data == data2)
        >>> assert ut.delete(fpath)

    Timeit:
        cPkl seems to be faster with this initial implementation

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
    if verbose or (verbose is None and __PRINT_WRITES__) or __FORCE_PRINT_WRITES__:
        print('[util_io] * save_hdf5(%r, data)' % (util_path.tail(fpath),))
    import h5py
    from os.path import basename
    import numpy as np
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
        import six
        assert all([isinstance(vals, np.ndarray) for vals in six.itervalues(data)]), 'can only save dicts as ndarrays'
        # file_ = h5py.File(fpath, 'w', **h5kw)
        with h5py.File(fpath, mode='w', **h5kw) as file_:
            grp = file_.create_group(fname)
            for key, val in six.iteritems(data):
                dset = grp.create_dataset(
                    key, val.shape,  val.dtype, chunks=chunks, compression=compression)
                dset[...] = val
    else:
        assert isinstance(data, np.ndarray)
        shape = data.shape
        dtype = data.dtype
        if verbose or (verbose is None and __PRINT_WRITES__):
            print('[util_io] * save_hdf5(%r, data)' % (util_path.tail(fpath),))
        # file_ = h5py.File(fpath, 'w', **h5kw)
        with h5py.File(fpath, mode='w', **h5kw) as file_:
            #file_.create_dataset(
            #    fname, shape,  dtype, chunks=chunks, compression=compression,
            #    data=data)
            dset = file_.create_dataset(
                fname, shape,  dtype, chunks=chunks, compression=compression)
            dset[...] = data


def load_hdf5(fpath, verbose=False):
    import h5py
    from os.path import basename
    import numpy as np
    import six
    fname = basename(fpath)
    #file_ = h5py.File(fpath, 'r')
    #file_.values()
    #file_.keys()
    if verbose or (verbose is None and __PRINT_READS__) or __FORCE_PRINT_READS__:
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
                pass
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
    if verbose or (verbose is None and __PRINT_WRITES__) or __FORCE_PRINT_WRITES__:
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
    if verbose or (verbose is None and __PRINT_READS__) or __FORCE_PRINT_READS__:
        print('[util_io] * load_pytables(%r, data)' % (util_path.tail(fpath),))
    with tables.open_file(fpath, 'r') as file_:
        data = file_.root.data.read()
    return data


def load_numpy(fpath, mmap_mode=None, verbose=None):
    if verbose or (verbose is None and __PRINT_READS__) or __FORCE_PRINT_READS__:
        print('[util_io] * load_numpy(%r)' % util_path.tail(fpath))
    return np.load(fpath, mmap_mode=mmap_mode)


def save_numpy(fpath, data, verbose=None):
    if verbose or (verbose is None and __PRINT_WRITES__) or __FORCE_PRINT_WRITES__:
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
