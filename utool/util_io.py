from __future__ import absolute_import, division, print_function
from six.moves import cPickle
import lockfile
from utool import util_path
from utool import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[io]')


__PRINT_IO__ = True
__PRINT_WRITES__ = __PRINT_IO__
__PRINT_READS__  =  __PRINT_IO__


def write_to(fpath, to_write, aslines=False, verbose=False,
             onlyifdiff=False, mode='w'):
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
    if verbose or (verbose is None and __PRINT_WRITES__):
        print('[util_io] * Writing to text file: %r ' % util_path.tail(fpath))
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
    if verbose or (verbose is None and __PRINT_READS__):
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
        from .util_dbg import printex
        if verbose or strict:
            printex(ex, ' * Error reading fpath=%r' %
                    util_path.tail(fpath), '[io]')
        if strict:
            raise


# aliases
readfrom = read_from
writeto = write_to


def save_cPkl(fpath, data, verbose=None):
    if verbose or (verbose is None and __PRINT_WRITES__):
        print('[util_io] * save_cPkl(%r, data)' % (util_path.tail(fpath),))
    with open(fpath, 'wb') as file_:
        cPickle.dump(data, file_, cPickle.HIGHEST_PROTOCOL)


def load_cPkl(fpath, verbose=None):
    if verbose or (verbose is None and __PRINT_READS__):
        print('[util_io] * load_cPkl(%r, data)' % (util_path.tail(fpath),))
    with open(fpath, 'rb') as file_:
        data = cPickle.load(file_)
    return data


def lock_and_load_cPkl(fpath, verbose=False):
    with lockfile.LockFile(fpath + '.lock'):
        return load_cPkl(fpath, verbose)


def lock_and_save_cPkl(fpath, data, verbose=False):
    with lockfile.LockFile(fpath + '.lock'):
        return save_cPkl(fpath, data, verbose)


def save_hdf5(fpath, data, verbose=False, compression='gzip'):
    r"""
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

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_io import *  # NOQA
        >>> import numpy as np
        >>> import utool as ut
        >>> # build test data
        >>> verbose = True
        >>> fpath = 'myfile.hdf5'
        >>> np.random.seed(0)
        >>> compression = 'gzip'
        >>> data = (np.random.rand(100000, 128) * 255).astype(np.uint8).copy()
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

    """
    import h5py
    from os.path import basename
    chunks = True
    fname = basename(fpath)
    shape = data.shape
    dtype = data.dtype
    if verbose or (verbose is None and __PRINT_WRITES__):
        print('[util_io] * save_hdf5(%r, data)' % (util_path.tail(fpath),))
    with h5py.File(fpath, 'w') as file_:
        dset = file_.create_dataset(fname, shape,  dtype, chunks=chunks, compression=compression)
        dset[...] = data


def load_hdf5(fpath, verbose=False):
    import h5py
    from os.path import basename
    import numpy as np
    fname = basename(fpath)
    #file_ = h5py.File(fpath, 'r')
    #file_.values()
    #file_.keys()
    if verbose or (verbose is None and __PRINT_READS__):
        print('[util_io] * load_hdf5(%r, data)' % (util_path.tail(fpath),))
    with h5py.File(fpath, 'r') as file_:
        dset = file_[fname]
        shape = dset.shape
        dtype = dset.dtype
        data = np.empty(shape, dtype=dtype)
        dset.read_direct(data)
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
    if verbose or (verbose is None and __PRINT_WRITES__):
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
    if verbose or (verbose is None and __PRINT_READS__):
        print('[util_io] * load_pytables(%r, data)' % (util_path.tail(fpath),))
    with tables.open_file(fpath, 'r') as file_:
        data = file_.root.data.read()
    return data


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
