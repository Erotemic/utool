from __future__ import absolute_import, division, print_function
from os.path import dirname, split, join, splitext, exists, realpath, basename, commonprefix
import sys
import zipfile
import tarfile
import urllib
import functools
import time
from utool import util_path
from utool import util_cplat
from utool import util_arg
from utool import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[grabdata]')


QUIET = util_arg.QUIET
BadZipfile = zipfile.BadZipfile


def archive_files(archive_fpath, fpath_list, small=True, allowZip64=False,
                  overwrite=False):
    """
    Args:
        archive_fpath (str): path to zipfile to create
        fpath_list (list): path of files to add to the zipfile
        small (bool): if True uses compression but the zipfile will take more time to write
        allowZip64 (bool): use if a file is over 2GB
        overwrite (bool):

    References:
        https://docs.python.org/2/library/zipfile.html

    CommandLine:
        python -m utool.util_grabdata --test-archive_files

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_grabdata import *  # NOQA
        >>> # build test data
        >>> archive_fpath = '?'
        >>> fpath_list = '?'
        >>> small = True
        >>> allowZip64 = False
        >>> # execute function
        >>> result = archive_files(archive_fpath, fpath_list, small, allowZip64)
        >>> # verify results
        >>> print(result)

    """
    import utool as ut
    from os.path import relpath, dirname
    if not overwrite and ut.checkpath(archive_fpath, verbose=True):
        raise AssertionError('cannot overrwite archive_fpath=%r' % (archive_fpath,))
    print('Archiving %d files' % len(fpath_list))
    compression = zipfile.ZIP_DEFLATED if small else zipfile.ZIP_STORED
    with zipfile.ZipFile(archive_fpath, 'w', compression, allowZip64) as myzip:
        for fpath in fpath_list:
            arcname = relpath(fpath, dirname(archive_fpath))
            myzip.write(fpath, arcname)


def unarchive_file(archive_fpath, force_commonprefix=True):
    print('Unarchive: %r' % archive_fpath)
    if tarfile.is_tarfile(archive_fpath):
        return untar_file(archive_fpath, force_commonprefix=force_commonprefix)
    elif zipfile.is_zipfile(archive_fpath):
        return unzip_file(archive_fpath, force_commonprefix=force_commonprefix)
    else:
        raise AssertionError('unknown archive format: %r' % (archive_fpath,))


def untar_file(targz_fpath, force_commonprefix=True):
    tar_file = tarfile.open(targz_fpath, 'r:gz')
    output_dir = dirname(targz_fpath)
    archive_namelist = [mem.path for mem in tar_file.getmembers()]
    return _extract_archive(targz_fpath, tar_file, archive_namelist, output_dir, force_commonprefix)


def unzip_file(zip_fpath, force_commonprefix=True):
    zip_file = zipfile.ZipFile(zip_fpath)
    output_dir  = dirname(zip_fpath)
    archive_namelist = zip_file.namelist()
    return _extract_archive(zip_fpath, zip_file, archive_namelist, output_dir, force_commonprefix)


def _extract_archive(archive_fpath, archive_file, archive_namelist, output_dir, force_commonprefix=True):
    # force extracted components into a subdirectory if force_commonprefix is on
    #return_path = output_diG
    # FIXME doesn't work right
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
    return output_dir


def open_url_in_browser(url):
    """
    open_url_in_browser

    Args:
        url (str): web url

    Example:
        >>> from utool.util_grabdata import *  # NOQA
        >>> url = 'http://www.jrsoftware.org/isdl.php'
        >>> open_url_in_browser(url)
    """
    import webbrowser
    print('[utool] Opening url=%r in browser' % (url,))
    return webbrowser.open(url)


def download_url(url, filename=None, spoof=False):
    """ downloads a url to a filename.

    download_url

    Args:
        url (str): url to download
        filename (str): path to download to. Defaults to basename of url
        spoof (bool): if True pretends to by Firefox

    References:
        http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/

    Example:
        >>> from utool.util_grabdata import *  # NOQA
        >>> url = 'http://www.jrsoftware.org/download.php/ispack.exe'
        >>> fpath = download_url(url)
        >>> print(fpath)
        [utool] Downloading url='http://www.jrsoftware.org/download.php/ispack.exe' to filename='ispack.exe'
        ...100%, 1 MB, 606 KB/s, 3 seconds passed
        [utool] Finished downloading filename='ispack.exe'
        ispack.exe
    """
    if filename is None:
        filename = basename(url)
    def reporthook_(num_blocks, block_nBytes, total_nBytes, start_time=0):
        total_seconds = time.time() - start_time + 1E-9
        num_kb_down   = int(num_blocks * block_nBytes) / 1024
        num_mb_down   = num_kb_down / 1024
        percent_down  = int(num_blocks * block_nBytes * 100 / total_nBytes)
        kb_per_second = int(num_kb_down / (total_seconds))
        fmt_msg = '\r...%d%%, %d MB, %d KB/s, %d seconds passed'
        msg = fmt_msg % (percent_down, num_mb_down, kb_per_second, total_seconds)
        sys.stdout.write(msg)
        sys.stdout.flush()
    reporthook = functools.partial(reporthook_, start_time=time.time())
    print('[utool] Downloading url=%r to filename=%r' % (url, filename))
    if spoof:
        # Different agents that can be used for spoofing
        user_agents = [
            'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
            'Opera/9.25 (Windows NT 5.1; U; en)',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
            'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
            'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
            'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9'
        ]
        class SpoofingOpener(urllib.FancyURLopener, object):
            version = user_agents[0]
        spoofing_opener = SpoofingOpener()
        spoofing_opener.retrieve(url, filename=filename, reporthook=reporthook)
    else:
        # no spoofing
        urllib.urlretrieve(url, filename=filename, reporthook=reporthook)
    print('')
    print('[utool] Finished downloading filename=%r' % (filename,))
    return filename


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


TESTIMG_URL_DICT = {
    'grace.jpg' : 'http://i.imgur.com/rgQyu7r.jpg',
    'jeff.png'  : 'http://i.imgur.com/l00rECD.png',
    'ada2.jpg'  : 'http://i.imgur.com/zHOpTCb.jpg',
    'ada.jpg'   : 'http://i.imgur.com/iXNf4Me.jpg',
    'lena.png'  : 'http://i.imgur.com/JGrqMnV.png',
    'carl.jpg'  : 'http://i.imgur.com/flTHWFD.jpg',
    'easy1.png' : 'http://i.imgur.com/Qqd0VNq.png',
    'easy2.png' : 'http://i.imgur.com/BDP8MIu.png',
    'easy3.png' : 'http://i.imgur.com/zBcm5mS.png',
    'hard3.png' : 'http://i.imgur.com/ST91yBf.png',
    'zebra.png' : 'http://i.imgur.com/58hbGcd.png',
    'star.png'  : 'http://i.imgur.com/d2FHuIU.png',
}


def get_valid_test_imgkeys():
    """ returns valid keys for grab_test_imgpath """
    return list(TESTIMG_URL_DICT.keys())


def grab_test_imgpath(key, allow_external=True):
    """
    Gets paths to standard / fun test images.
    Downloads them if they dont exits

    Args:
        key (str): one of the standard test images, e.g. lena.png, carl.jpg, ...
        allow_external (bool): if True you can specify existing fpaths

    Returns:
        str: testimg_fpath - filepath to the downloaded or cached test image.

    SeeAlso:
        ut.get_valid_test_imgkeys

    CommandLine:
        python -m utool.util_grabdata --test-grab_test_imgpath

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_grabdata import *  # NOQA
        >>> import utool as ut
        >>> # build test data
        >>> key = 'carl.jpg'
        >>> # execute function
        >>> testimg_fpath = grab_test_imgpath(key)
        >>> # verify results
        >>> ut.assertpath(testimg_fpath)
    """
    if allow_external and key not in TESTIMG_URL_DICT:
        testimg_fpath = key
        if not util_path.checkpath(testimg_fpath, verbose=True):
            import utool as ut
            raise AssertionError(
                'testimg_fpath=%r not found did you mean %s' % (
                    testimg_fpath,
                    ut.cond_phrase(get_valid_test_imgkeys(), 'or')))
    else:
        testimg_fname = key
        testimg_url = TESTIMG_URL_DICT[key]
        testimg_fpath = grab_file_url(testimg_url, fname=testimg_fname)
    return testimg_fpath


def grab_file_url(file_url, ensure=True, appname='utool', download_dir=None,
                  delay=None, spoof=False, fname=None, verbose=True):
    """
    grab_file_url

    Args:
        file_url (str):
        ensure (bool):
        appname (str):
        download_dir (None):
        delay (None):
        spoof (bool):
        fname (None):

    Returns:
        str: fpath

    CommandLine:
        sh -c "python ~/code/utool/utool/util_grabdata.py --all-examples"

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_grabdata import *  # NOQA
        >>> import utool as ut  # NOQA
        >>> from os.path import basename
        >>> file_url = 'http://i.imgur.com/JGrqMnV.png'
        >>> ensure = True
        >>> appname = 'utool'
        >>> download_dir = None
        >>> delay = None
        >>> spoof = False
        >>> fname ='lena.png'
        >>> lena_fpath = grab_file_url(file_url, ensure, appname, download_dir, delay, spoof, fname)
        >>> result = basename(lena_fpath)
        >>> print(result)
        lena.png

    """
    file_url = fix_dropbox_link(file_url)
    if fname is None:
        fname = basename(file_url)
    # Download zipfile to
    if download_dir is None:
        download_dir = util_cplat.get_app_resource_dir(appname)
    # Zipfile should unzip to:
    fpath = join(download_dir, fname)
    if ensure:
        util_path.ensurepath(download_dir)
        if not exists(fpath):
            # Download testdata
            if verbose:
                print('[utool] Downloading file %s' % fpath)
            if delay is not None:
                print('[utool] delay download by %r seconds' % (delay,))
                time.sleep(delay)
            download_url(file_url, fpath, spoof=spoof)
        else:
            if verbose:
                print('[utool] Already have file %s' % fpath)
    util_path.assert_exists(fpath)
    return fpath


def grab_zipped_url(zipped_url, ensure=True, appname='utool', download_dir=None,
                    force_commonprefix=True, cleanup=False):
    """
    downloads and unzips the url

    Args:
        zipped_url (str): url which must be either a .zip of a .tar.gz file

    Examples:
        >>> from utool.util_grabdata import *  # NOQA
        >>> zipped_url = 'https://dl.dropboxusercontent.com/s/of2s82ed4xf86m6/testdata.zip'
        >>> zipped_url = 'http://www.spam.com/eggs/data.zip'

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
            #true_zipped_fpath = grab_file_url(zipped_url,
            #                                  download_dir=download_dir,
            #                                  appname=appname)
            #data_dir = unarchive_file(true_zipped_fpath, force_commonprefix)
            if not exists(zip_fpath):
                download_url(zipped_url, zip_fpath)
            unarchive_file(zip_fpath, force_commonprefix)
            if cleanup:
                util_path.delete(zip_fpath)  # Cleanup
    if cleanup:
        util_path.assert_exists(data_dir)
    return util_path.unixpath(data_dir)


def geo_locate(default='Unknown', timeout=1):
    try:
        import urllib2
        import json
        req = urllib2.Request('http://freegeoip.net/json/', headers={ 'User-Agent': 'Mozilla/5.0' })
        f = urllib2.urlopen(req, timeout=timeout)
        json_string = f.read()
        f.close()
        location = json.loads(json_string)
        location_city    = location['city']
        location_state   = location['region_name']
        location_country = location['country_name']
        location_zip     = location['zipcode']
        success = True
    except:
        success = False
        location_city    = default
        location_state   = default
        location_zip     = default
        location_country = default
    return success, location_city, location_state, location_country, location_zip


if __name__ == '__main__':
    """
    CommandLine:
        sh -c "python ~/code/utool/utool/util_grabdata.py --all-examples"
        python -c "import utool, utool.util_grabdata; utool.doctest_funcs(utool.util_grabdata, allexamples=True)"
        python -c "import utool, utool.util_grabdata; utool.doctest_funcs(utool.util_grabdata)"
        python -m utool.util_grabdata
        python -m utool.util_grabdata --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
