from __future__ import absolute_import, division, print_function
import cPickle
from . import util_path
from . import util_dbg
from . import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[io]')


__PRINT_IO__ = True
__PRINT_WRITES__ = False or __PRINT_IO__
__PRINT_READS__  = False or __PRINT_IO__


def write_to(fpath, to_write, verbose=__PRINT_WRITES__):
    """
    writes_to(fpath, to_write)
    writes to_write to fpath
    """
    if __PRINT_WRITES__:
        print('[io] * Writing to text file: %r ' % util_path.tail(fpath))
    with open(fpath, 'w') as file_:
        file_.write(to_write)


def read_from(fpath, verbose=__PRINT_READS__, strict=True):
    if verbose:
        print('[io] * Reading text file: %r ' % util_path.tail(fpath))
    try:
        if not util_path.checkpath(fpath, verbose=verbose):
            raise IOError('[io] * FILE DOES NOT EXIST!')
        with open(fpath, 'r') as file_:
            text = file_.read()
        return text
    except IOError as ex:
        if verbose or strict:
            util_dbg.printex(ex, ' * Error reading fpath=%r' %
                             util_path.tail(fpath), '[io]')
        if strict:
            raise


def save_cPkl(fpath, data):
    # TODO: Split into utool.util_io
    if __PRINT_WRITES__:
        print('[cache] * save_cPkl(%r, data)' % (util_path.tail(fpath),))
    with open(fpath, 'wb') as file_:
        cPickle.dump(data, file_, cPickle.HIGHEST_PROTOCOL)


def load_cPkl(fpath):
    # TODO: Split into utool.util_io
    if __PRINT_READS__:
        print('[cache] * load_cPkl(%r, data)' % (util_path.tail(fpath),))
    with open(fpath, 'rb') as file_:
        data = cPickle.load(file_)
    return data
