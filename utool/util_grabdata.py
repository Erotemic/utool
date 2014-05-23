from __future__ import absolute_import, division, print_function
from os.path import dirname, split, join, splitext, exists, realpath
import sys
import zipfile
import urllib
import time
from . import util_path
from . import util_cplat
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[path]')


__QUIET__ = '--quiet' in sys.argv
BadZipfile = zipfile.BadZipfile


def unzip_file(zip_fpath):
    zip_file = zipfile.ZipFile(zip_fpath)
    zip_dir  = dirname(zip_fpath)
    zipped_namelist = zip_file.namelist()
    for member in zipped_namelist:
        (dname, fname) = split(member)
        dpath = join(zip_dir, dname)
        util_path.ensurepath(dpath)
        if not __QUIET__:
            print('[utool] Unzip ' + fname + ' in ' + dpath)
        zip_file.extract(member, path=zip_dir)


def download_url(url, filename):
    # From http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/
    start_time_ptr = [0]
    def reporthook(count, block_size, total_size):
        if count == 0:
            start_time_ptr[0] = time.time()
            return
        duration = time.time() - start_time_ptr[0]
        if duration == 0:
            duration = 1E-9
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write('\r...%d%%, %d MB, %d KB/s, %d seconds passed' %
                         (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()
    print('[utool] Downloading url=%r to filename=%r' % (url, filename))
    urllib.urlretrieve(url, filename=filename, reporthook=reporthook)


def grab_zipped_url(zipped_testdata_url, ensure=True, appname='utool'):
    """
    Input zipped_testdata_url - this must look like:
    http://www.spam.com/eggs/data.zip
    eg:
    https://dl.dropboxusercontent.com/s/of2s82ed4xf86m6/testdata.zip
    """
    zip_fname = split(zipped_testdata_url)[1]
    data_name = splitext(zip_fname)[0]
    # Download zipfile to
    download_dir = util_cplat.get_app_resource_dir(appname)
    # Zipfile should unzip to:
    data_dir = join(download_dir, data_name)
    if ensure:
        util_path.ensurepath(download_dir)
        if not exists(data_dir):
            # Download and unzip testdata
            zip_fpath = realpath(join(download_dir, zip_fname))
            print('[utool] Downloading testdata %s' % zip_fpath)
            util_path.download_url(zipped_testdata_url, zip_fpath)
            util_path.unzip_file(zip_fpath)
            util_path.delete(zip_fpath)  # Cleanup
    util_path.assert_exists(data_dir)
    return data_dir
