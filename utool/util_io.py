from __future__ import absolute_import, division, print_function
from six.moves import cPickle
from utool import util_path
from utool import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[io]')


__PRINT_IO__ = True
__PRINT_WRITES__ = __PRINT_IO__
__PRINT_READS__  =  __PRINT_IO__


def write_to(fpath, to_write, aslines=False, verbose=False,
                onlyifdiff=False):
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
    if verbose or __PRINT_WRITES__:
        print('[util_io] * Writing to text file: %r ' % util_path.tail(fpath))
    with open(fpath, 'w') as file_:
        if aslines:
            file_.writelines(to_write)
        else:
            file_.write(to_write)


def read_from(fpath, verbose=False, aslines=False, strict=True):
    """ Reads text from a file

    Args:
        fpath (str): file path
        aslines (bool): if True returns list of lines
        verbose (bool): verbosity flag

    Returns:
        text from fpath
    """
    if verbose or __PRINT_READS__:
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


def save_cPkl(fpath, data, verbose=False):
    if verbose or __PRINT_WRITES__:
        print('[util_io] * save_cPkl(%r, data)' % (util_path.tail(fpath),))
    with open(fpath, 'wb') as file_:
        cPickle.dump(data, file_, cPickle.HIGHEST_PROTOCOL)


def load_cPkl(fpath, verbose=False):
    if verbose or __PRINT_READS__:
        print('[util_io] * load_cPkl(%r, data)' % (util_path.tail(fpath),))
    with open(fpath, 'rb') as file_:
        data = cPickle.load(file_)
    return data


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
