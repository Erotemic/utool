from __future__ import absolute_import, division, print_function
from os.path import dirname, split, join, splitext, exists, realpath, basename, commonprefix
import sys
import zipfile
import tarfile
import urllib
import time
from . import util_path
from . import util_cplat
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[grabdata]')


QUIET = '--quiet' in sys.argv
BadZipfile = zipfile.BadZipfile


def unarchive_file(archive_fpath, force_commonprefix=True):
    print('Unarchive: %r' % archive_fpath)
    if tarfile.is_tarfile(archive_fpath):
        return untar_file(archive_fpath, force_commonprefix=force_commonprefix)
    elif zipfile.is_zipfile(archive_fpath):
        return unzip_file(archive_fpath, force_commonprefix=force_commonprefix)
    else:
        raise AssertionError('unknown archive format')


def untar_file(targz_fpath, force_commonprefix=True):
    tar_file = tarfile.open(targz_fpath, 'r:gz')
    output_dir = dirname(targz_fpath)
    archive_namelist = [mem.path for mem in tar_file.getmembers()]
    _extract_archive(targz_fpath, tar_file, archive_namelist, output_dir, force_commonprefix)


def unzip_file(zip_fpath, force_commonprefix=True):
    zip_file = zipfile.ZipFile(zip_fpath)
    output_dir  = dirname(zip_fpath)
    archive_namelist = zip_file.namelist()
    _extract_archive(zip_fpath, zip_file, archive_namelist, output_dir, force_commonprefix)


def _extract_archive(archive_fpath, archive_file, archive_namelist, output_dir, force_commonprefix=True):
    # force extracted components into a subdirectory if force_commonprefix is on
    if force_commonprefix and commonprefix(archive_namelist) == '':
        # use the archivename as the default common prefix
        archive_basename, ext = split_archive_ext(basename(archive_fpath))
        output_dir = join(output_dir, archive_basename)
        util_path.ensurepath(archive_basename)

    for member in archive_namelist:
        (dname, fname) = split(member)
        dpath = join(output_dir, dname)
        util_path.ensurepath(dpath)
        if not QUIET:
            print('[utool] Unarchive ' + fname + ' in ' + dpath)
        archive_file.extract(member, path=output_dir)


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


def fix_dropbox_link(dropbox_url):
    """ Dropbox links should be en-mass downloaed from dl.dropbox """
    return dropbox_url.replace('www.dropbox', 'dl.dropbox')


def split_archive_ext(path):
    special_exts = ['.tar.gz', '.tar.bz2']
    for ext in special_exts:
        if path.endswith(ext):
            name, ext = path[:-len(ext)], path[-len(ext):]
            break
    else:
        name, ext = splitext(path)
    return name, ext


def grab_file_url(file_url, ensure=True, appname='utool', download_dir=None,
                  delay=None):
    file_url = fix_dropbox_link(file_url)
    fname = split(file_url)[1]
    # Download zipfile to
    if download_dir is None:
        download_dir = util_cplat.get_app_resource_dir(appname)
    # Zipfile should unzip to:
    fpath = join(download_dir, fname)
    if ensure:
        util_path.ensurepath(download_dir)
        if not exists(fpath):
            # Download testdata
            print('[utool] Downloading file %s' % fpath)
            if delay is not None:
                print('[utool] delay download by %r seconds' % (delay,))
                time.sleep(delay)
            download_url(file_url, fpath)
    util_path.assert_exists(fpath)
    return fpath


def grab_zipped_url(zipped_url, ensure=True, appname='utool', download_dir=None,
                    force_commonprefix=True, cleanup=True):
    """
    Input zipped_url - this must look like:
    http://www.spam.com/eggs/data.zip
    eg:
    https://dl.dropboxusercontent.com/s/of2s82ed4xf86m6/testdata.zip

    downloads and unzips the url
    """
    zipped_url = fix_dropbox_link(zipped_url)
    zip_fname = split(zipped_url)[1]
    data_name = split_archive_ext(zip_fname)[0]
    # Download zipfile to
    if download_dir is None:
        download_dir = util_cplat.get_app_resource_dir(appname)
    # Zipfile should unzip to:
    data_dir = join(download_dir, data_name)
    if ensure:
        util_path.ensurepath(download_dir)
        if not exists(data_dir):
            # Download and unzip testdata
            zip_fpath = realpath(join(download_dir, zip_fname))
            print('[utool] Downloading archive %s' % zip_fpath)
            download_url(zipped_url, zip_fpath)
            unarchive_file(zip_fpath, force_commonprefix)
            if cleanup:
                util_path.delete(zip_fpath)  # Cleanup
    if cleanup:
        util_path.assert_exists(data_dir)
    return util_path.unixpath(data_dir)
