# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from os.path import dirname, split, join, splitext, exists, realpath, basename, commonprefix
import six
import sys
import zipfile
import tarfile
import gzip
import urllib  # NOQA
import functools
import time
from six.moves import urllib as _urllib
from utool import util_path
from utool import util_cplat
from utool import util_arg
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__, '[grabdata]')


QUIET = util_arg.QUIET
BadZipfile = zipfile.BadZipfile


def archive_files(archive_fpath, fpath_list, small=True, allowZip64=False,
                  overwrite=False, verbose=True, common_prefix=False):
    r"""
    Adds the files in ``fpath_list`` to an zip/tar archive.

    Args:
        archive_fpath (str): path to zipfile to create
        fpath_list (list): path of files to add to the zipfile
        small (bool): if True uses compression but the zipfile will take more
            time to write
        allowZip64 (bool): use if a file is over 2GB
        overwrite (bool):
        verbose (bool):  verbosity flag(default = True)
        common_prefix (bool): (default = False)

    References:
        https://docs.python.org/2/library/zipfile.html

    CommandLine:
        python -m utool.util_grabdata --test-archive_files

    Example:
        >>> # SLOW_DOCTEST
        >>> from utool.util_grabdata import *  # NOQA
        >>> import utool as ut
        >>> archive_fpath = ut.get_app_resource_dir('utool', 'testarchive.zip')
        >>> # remove an existing test archive
        >>> ut.delete(archive_fpath)
        >>> assert not exists(archive_fpath), 'archive should not exist'
        >>> fpath_list = [ut.grab_test_imgpath(key) for key in ut.TESTIMG_URL_DICT]
        >>> small = True
        >>> allowZip64 = False
        >>> overwrite = True
        >>> result = archive_files(archive_fpath, fpath_list, small, allowZip64, overwrite)
        >>> # verify results
        >>> print(result)
        >>> assert exists(archive_fpath), 'archive should exist'

    Ignore:
        # http://superuser.com/questions/281573/best-options-compressing-files-7-zip
        # Create a small 7zip archive
        7z a -t7z -m0=lzma -mx=9 -mfb=64 -md=32m -ms=on archive.7z dir1
        7z a -t7z -m0=lzma -mx=9 -mfb=64 -md=32m -ms=on ibeis-linux-binary.7z ibeis

        # Create a small zip archive
        7za a -mm=Deflate -mfb=258 -mpass=15 -r ibeis-linux-binary.zip ibeis

    """
    import utool as ut
    from os.path import relpath, dirname
    if not overwrite and ut.checkpath(archive_fpath, verbose=True):
        raise AssertionError('cannot overrwite archive_fpath=%r' % (archive_fpath,))
    print('Archiving %d files' % len(fpath_list))
    compression = zipfile.ZIP_DEFLATED if small else zipfile.ZIP_STORED
    if common_prefix:
        # Note: common prefix does not care about file structures
        if isinstance(common_prefix, six.string_types):
            # use given path as base path
            rel_arcpath = common_prefix
        else:
            rel_arcpath = commonprefix(fpath_list)
        rel_arcpath = ut.longest_existing_path(rel_arcpath)
    else:
        rel_arcpath = dirname(archive_fpath)
    with zipfile.ZipFile(archive_fpath, 'w', compression, allowZip64) as myzip:
        for fpath in ut.ProgressIter(fpath_list, lbl='archiving files',
                                     enabled=verbose, adjust=True):
            arcname = relpath(fpath, rel_arcpath)
            myzip.write(fpath, arcname)


def unarchive_file(archive_fpath, force_commonprefix=True):
    print('Unarchive: %r' % archive_fpath)
    if tarfile.is_tarfile(archive_fpath):
        return untar_file(archive_fpath, force_commonprefix=force_commonprefix)
    elif zipfile.is_zipfile(archive_fpath):
        return unzip_file(archive_fpath, force_commonprefix=force_commonprefix)
    elif archive_fpath.endswith('.gz') and not archive_fpath.endswith('.tar.gz'):
        """
        from utool.util_grabdata import *
        archive_fpath = '/home/joncrall/.config/utool/train-images-idx3-ubyte.gz'
        """
        # FIXME: unsure if this is general
        output_fpath = splitext(archive_fpath)[0]
        with gzip.open(archive_fpath, 'rb') as gzfile_:
            contents = gzfile_.read()
            with open(output_fpath, 'wb') as file_:
                file_.write(contents)
        return output_fpath
    #elif archive_fpath.endswith('.gz'):
    #    # This is to handle .gz files (not .tar.gz) like how MNIST is stored
    #    # Example: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    #    return ungz_file(archive_fpath)
    else:
        if archive_fpath.endswith('.zip') or archive_fpath.endswith('.tar.gz'):
            raise AssertionError('archive is corrupted: %r' % (archive_fpath,))
        raise AssertionError('unknown archive format: %r' % (archive_fpath,))


#def ungz_file(gz_fpath):
#    # Jon, this is something I'm not sure how to generalize with your structure
#    # below, so I'm just going to leave it here in a nice little function.
#    # I think the original code will still work correctly with .tar.gz, but
#    # now will also work with just .gz files as a fall-back     -- Jason
#    with gzip.open(gz_fpath, 'rb') as gz_file:
#        extracted_content = gz_file.read()
#    extracted_fpath = gz_fpath.strip('.gz')
#    with open(extracted_fpath, 'w') as extracted_file:
#        extracted_file.write(extracted_content)
#    return extracted_fpath


def untar_file(targz_fpath, force_commonprefix=True):
    tar_file = tarfile.open(targz_fpath, 'r:gz')
    output_dir = dirname(targz_fpath)
    archive_namelist = [mem.path for mem in tar_file.getmembers()]
    output_dir = _extract_archive(targz_fpath, tar_file, archive_namelist,
                                  output_dir, force_commonprefix)
    tar_file.close()
    return output_dir


def unzip_file(zip_fpath, force_commonprefix=True, output_dir=None,
               prefix=None, dryrun=False, overwrite=None):
    zip_file = zipfile.ZipFile(zip_fpath)
    if output_dir is None:
        output_dir  = dirname(zip_fpath)
    archive_namelist = zip_file.namelist()
    output_dir  = _extract_archive(zip_fpath, zip_file, archive_namelist,
                                   output_dir, force_commonprefix,
                                   prefix=prefix, dryrun=dryrun,
                                   overwrite=overwrite)
    zip_file.close()
    return output_dir


def _extract_archive(archive_fpath, archive_file, archive_namelist, output_dir,
                     force_commonprefix=True, prefix=None,
                     dryrun=False, verbose=not QUIET, overwrite=None):
    """
    archive_fpath = zip_fpath
    archive_file = zip_file
    """
    # force extracted components into a subdirectory if force_commonprefix is
    # on return_path = output_diG
    # FIXME doesn't work right
    if prefix is not None:
        output_dir = join(output_dir, prefix)
        util_path.ensurepath(output_dir)

    archive_basename, ext = split_archive_ext(basename(archive_fpath))
    if force_commonprefix and commonprefix(archive_namelist) == '':
        # use the archivename as the default common prefix
        output_dir = join(output_dir, archive_basename)
        util_path.ensurepath(output_dir)

    for member in archive_namelist:
        (dname, fname) = split(member)
        dpath = join(output_dir, dname)
        util_path.ensurepath(dpath)
        if verbose:
            print('[utool] Unarchive ' + fname + ' in ' + dpath)

        if not dryrun:
            if overwrite is False:
                if exists(join(output_dir, member)):
                    continue
            archive_file.extract(member, path=output_dir)
    return output_dir


def open_url_in_browser(url, browsername=None, fallback=False):
    r"""
    Opens a url in the specified or default browser

    Args:
        url (str): web url

    CommandLine:
        python -m utool.util_grabdata --test-open_url_in_browser

    Example:
        >>> # SCRIPT
        >>> from utool.util_grabdata import *  # NOQA
        >>> url = 'http://www.jrsoftware.org/isdl.php'
        >>> open_url_in_browser(url, 'chrome')
    """
    import webbrowser
    print('[utool] Opening url=%r in browser' % (url,))
    if browsername is None:
        browser = webbrowser.open(url)
    else:
        browser = get_prefered_browser(pref_list=[browsername], fallback=fallback)
    return browser.open(url)


def get_prefered_browser(pref_list=[], fallback=True):
    r"""
    Args:
        browser_preferences (list): (default = [])
        fallback (bool): uses default if non of preferences work (default = True)

    CommandLine:
        python -m utool.util_grabdata --test-get_prefered_browser

    Ignore:
        import webbrowser
        webbrowser._tryorder
        pref_list = ['chrome', 'firefox', 'google-chrome']
        pref_list = ['firefox', 'google-chrome']

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_grabdata import *  # NOQA
        >>> browser_preferences = ['firefox', 'chrome', 'safari']
        >>> fallback = True
        >>> browser = get_prefered_browser(browser_preferences, fallback)
        >>> result = ('browser = %s' % (str(browser),))
        >>> print(result)
        >>> ut.quit_if_noshow()
    """
    import webbrowser
    import utool as ut
    pref_list = ut.ensure_iterable(pref_list)
    error_list = []

    # Hack for finding chrome on win32
    if ut.WIN32:
        # http://stackoverflow.com/questions/24873302/webbrowser-chrome-exe-does-not-work
        win32_chrome_fpath = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
        win32_chrome_browsername = win32_chrome_fpath + ' %s'
        win32_map = {
            'chrome': win32_chrome_browsername,
            'google-chrome': win32_chrome_browsername,
        }
        for browsername, win32_browsername in win32_map.items():
            index = ut.listfind(pref_list, browsername)
            if index is not None and True:  # ut.checkpath(win32_browsername):
                pref_list.insert(index + 1, win32_browsername)

    for browsername in pref_list:
        try:
            browser = webbrowser.get(browsername)
            return browser
        except webbrowser.Error as ex:
            error_list.append(ex)
            print(str(browsername) + ' failed. Reason: ' + str(ex))

    if fallback:
        browser = webbrowser
        return browser
    else:
        raise AssertionError('No browser meets preferences=%r. error_list=%r' %
                             (pref_list, error_list,))


def download_url(url, filename=None, spoof=False):
    r""" downloads a url to a filename.

    Args:
        url (str): url to download
        filename (str): path to download to. Defaults to basename of url
        spoof (bool): if True pretends to by Firefox

    References:
        http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/

    TODO:
        Delete any partially downloaded files

    Example:
        >>> from utool.util_grabdata import *  # NOQA
        >>> url = 'http://www.jrsoftware.org/download.php/ispack.exe'
        >>> fpath = download_url(url)
        >>> print(fpath)
        ispack.exe
    """
    # Weird that we seem to need this here for tests
    import urllib  # NOQA
    if filename is None:
        filename = basename(url)
    print('[utool] Downloading url=%r to filename=%r' % (url, filename))
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
    if spoof:
        # Different agents that can be used for spoofing
        user_agents = [
            'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',  # NOQA
            'Opera/9.25 (Windows NT 5.1; U; en)',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',  # NOQA
            'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
            'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',  # NOQA
            'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9'
        ]
        class SpoofingOpener(urllib.FancyURLopener, object):
            version = user_agents[0]
        spoofing_opener = SpoofingOpener()
        spoofing_opener.retrieve(url, filename=filename, reporthook=reporthook)
    else:
        # no spoofing
        if six.PY2:
            urllib.urlretrieve(url, filename=filename, reporthook=reporthook)
        elif six.PY3:
            import urllib.request
            urllib.request.urlretrieve(url, filename=filename, reporthook=reporthook)
        else:
            assert False, 'unknown python'
    print('')
    print('[utool] Finished downloading filename=%r' % (filename,))
    return filename


#if six.PY2:
#    import urllib as _urllib
#elif six.PY3:
#    import urllib.request as _urllib


def url_read(url, verbose=True):
    r"""
    Directly reads data from url
    """
    if url.find('://') == -1:
        url = 'http://' + url
    if verbose:
        print('Reading data from url=%r' % (url,))
    try:
        file_ = _urllib.request.urlopen(url)
        #file_ = _urllib.urlopen(url)
    except IOError:
        raise
    data = file_.read()
    file_.close()
    return data


def experiment_download_multiple_urls(url_list):
    r"""
    References:
        http://stackoverflow.com/questions/1112343/capture-sigint-in-python
        http://stackoverflow.com/questions/16694907/download-large-file-requests
        GracefulInterruptHandler

    Ignore:
        import signal
        import sys
        def signal_handler(signal, frame):
                print('You pressed Ctrl+C!')
                sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        print('Press Ctrl+C')
        signal.pause()

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> url_list = [
        >>>     'https://lev.cs.rpi.edu/public/installers/ibeis-win32-setup-ymd_hm-2015-08-01_16-28.exe',   # NOQA
        >>>     'https://lev.cs.rpi.edu/public/models/vgg.caffe.slice_0_30_None.pickle',
        >>>     'https://lev.cs.rpi.edu/public/models/vgg.caffe.slice_0_30_None.pickle',
        >>>     'https://lev.cs.rpi.edu/public/models/vgg.caffe.slice_0_30_None.pickle',
        >>>     'https://lev.cs.rpi.edu/public/models/vgg.caffe.slice_0_30_None.pickle',
        >>>     'https://lev.cs.rpi.edu/public/models/vgg.caffe.slice_0_30_None.pickle',
        >>>     ]
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/L10/L10_R1/S1_L10_R1_PICT0070.JPG'
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0001.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0002.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0003.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0004.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0005.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0006.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0007.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0008.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0022.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0023.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0024.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0025.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0026.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0027.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0028.JPG',
        >>>     'https://snapshotserengeti.s3.msi.umn.edu/S1/B04/B04_R1/S1_B04_R1_PICT0029.JPG'
        >>> ]
    """
    import requests
    import os
    session = requests.session()

    def session_download_url(url):
        filename = basename(url)
        print('[utool] Downloading url=%r to filename=%r' % (url, filename))
        spoof_header = {'user-agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}
        response = session.get(url, headers=spoof_header, stream=True)
        if response.ok:
            with open(filename, 'wb') as file_:
                _iter = response.iter_content(chunk_size=1024)
                for chunk in ut.ProgressIter(_iter, nTotal=-1, freq=1):
                    if chunk:  # filter out keep-alive new chunks
                        file_.write(chunk)
                        file_.flush()
                        os.fsync(file_.fileno())
        else:
            print('Error downloading file. response=%r' % (response,))
            return False
        return response.ok

    for url in ut.ProgressIter(url_list, 'downlaoding urls'):
        if not session_download_url(url):
            break


def clean_dropbox_link(dropbox_url):
    """ Dropbox links should be en-mass downloaed from dl.dropbox

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_grabdata import *  # NOQA
        >>> dropbox_url = 'www.dropbox.com/s/123456789abcdef/foobar.zip?dl=0'
        >>> cleaned_url = clean_dropbox_link(dropbox_url)
        >>> result = str(cleaned_url)
        >>> print(result)
        dl.dropbox.com/s/123456789abcdef/foobar.zip
    """
    cleaned_url = dropbox_url.replace('www.dropbox', 'dl.dropbox')
    postfix_list = [
        '?dl=0'
    ]
    for postfix in postfix_list:
        if cleaned_url.endswith(postfix):
            cleaned_url = cleaned_url[:-1 * len(postfix)]
    # cleaned_url = cleaned_url.rstrip('?dl=0')
    return cleaned_url


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
    'patsy.jpg' : 'http://i.imgur.com/C1lNRfT.jpg',
}


def get_valid_test_imgkeys():
    r""" returns valid keys for grab_test_imgpath """
    return list(TESTIMG_URL_DICT.keys())


def clear_test_img_cache():
    r"""
    CommandLine:
        python -m utool.util_grabdata --test-clear_test_img_cache

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> from utool.util_grabdata import *  # NOQA
        >>> testimg_fpath = clear_test_img_cache()
        >>> result = str(testimg_fpath)
        >>> print(result)
    """
    import utool as ut
    download_dir = util_cplat.get_app_resource_dir('utool')
    for key in TESTIMG_URL_DICT:
        fpath = join(download_dir, key)
        ut.delete(fpath)


def grab_test_imgpath(key='lena.png', allow_external=True, verbose=True):
    r"""
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
                    ut.conj_phrase(get_valid_test_imgkeys(), 'or')))
    else:
        testimg_fname = key
        testimg_url = TESTIMG_URL_DICT[key]
        testimg_fpath = grab_file_url(testimg_url, fname=testimg_fname, verbose=verbose)
    return testimg_fpath


def grab_selenium_chromedriver():
    r"""
    Automatically download selenium chrome driver if needed

    CommandLine:
        python -m utool.util_grabdata --test-grab_selenium_chromedriver:1

    Example:
        >>> # DISABLE_DOCTEST
        >>> ut.grab_selenium_chromedriver()
        >>> import selenium.webdriver
        >>> driver = selenium.webdriver.Chrome()
        >>> driver.get('http://www.google.com')
        >>> search_field = driver.find_element_by_name('q')
        >>> search_field.send_keys('puppies')
        >>> search_field.send_keys(selenium.webdriver.common.keys.Keys.ENTER)

    Example1:
        >>> # DISABLE_DOCTEST
        >>> import selenium.webdriver
        >>> driver = selenium.webdriver.Firefox()
        >>> driver.get('http://www.google.com')
        >>> search_field = driver.find_element_by_name('q')
        >>> search_field.send_keys('puppies')
        >>> search_field.send_keys(selenium.webdriver.common.keys.Keys.ENTER)
    """
    import utool as ut
    import os
    import stat
    # TODO: use a better download dir (but it must be in the PATh or selenium freaks out)
    chromedriver_dpath = ut.ensuredir(ut.truepath('~/bin'))
    chromedriver_fpath = join(chromedriver_dpath, 'chromedriver')
    if not ut.checkpath(chromedriver_fpath):
        assert chromedriver_dpath in os.environ['PATH'].split(os.pathsep)
        # TODO: make this work for windows as well
        if ut.LINUX and ut.util_cplat.is64bit_python():
            url = 'http://chromedriver.storage.googleapis.com/2.16/chromedriver_linux64.zip'
            ut.grab_zipped_url(url, download_dir=chromedriver_dpath)
        else:
            raise AssertionError('unsupported chrome driver getter script')
        if not ut.WIN32:
            st = os.stat(chromedriver_fpath)
            os.chmod(chromedriver_fpath, st.st_mode | stat.S_IEXEC)
    ut.assert_exists(chromedriver_fpath)
    os.environ['webdriver.chrome.driver'] = chromedriver_fpath
    return chromedriver_fpath


def grab_selenium_driver(driver_name=None):
    """
    pip install selenium -U
    """
    from selenium import webdriver
    if driver_name is None:
        driver_name = 'firefox'
    if driver_name.lower() == 'chrome':
        grab_selenium_chromedriver()
        return webdriver.Chrome()
    elif driver_name.lower() == 'firefox':
        grab_selenium_chromedriver()
        return webdriver.Firefox()
    else:
        raise AssertionError('unknown name = %r' % (driver_name,))


def grab_file_url(file_url, ensure=True, appname='utool', download_dir=None,
                  delay=None, spoof=False, fname=None, verbose=True, redownload=False):
    r"""
    Downloads a file and returns the local path of the file.

    The resulting file is cached, so multiple calls to this function do not
    result in multiple dowloads.

    Args:
        file_url (str): url to the file
        ensure (bool):  if False the file is assumed to be downloaed
            (default = True)
        appname (str): (default = 'utool')
        download_dir custom directory (None): (default = None)
        delay (None): delay time before download (default = None)
        spoof (bool): (default = False)
        fname (str):  custom file name (default = None)
        verbose (bool):  verbosity flag (default = True)
        redownload (bool): if True forces redownload of the file
            (default = False)

    Returns:
        str: fpath

    CommandLine:
        sh -c "python ~/code/utool/utool/util_grabdata.py --all-examples"
        python -m utool.util_grabdata --test-grab_file_url

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
        >>> verbose = True
        >>> redownload = True
        >>> fname ='lena.png'
        >>> lena_fpath = ut.grab_file_url(file_url, ensure, appname, download_dir,
        >>>     delay, spoof, fname, verbose, redownload)
        >>> result = basename(lena_fpath)
        >>> print(result)
        lena.png
    """
    file_url = clean_dropbox_link(file_url)
    if fname is None:
        fname = basename(file_url)
    # Download zipfile to
    if download_dir is None:
        download_dir = util_cplat.get_app_resource_dir(appname)
    # Zipfile should unzip to:
    fpath = join(download_dir, fname)
    if ensure or redownload:
        util_path.ensurepath(download_dir)
        if redownload or not exists(fpath):
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
    if ensure:
        util_path.assert_exists(fpath)
    return fpath


def grab_zipped_url(zipped_url, ensure=True, appname='utool',
                    download_dir=None, force_commonprefix=True, cleanup=False,
                    redownload=False, spoof=False):
    r"""
    downloads and unzips the url

    Args:
        zipped_url (str): url which must be either a .zip of a .tar.gz file
        ensure (bool):  eager evaluation if True(default = True)
        appname (str): (default = 'utool')
        download_dir (str): containing downloading directory
        force_commonprefix (bool): (default = True)
        cleanup (bool): (default = False)
        redownload (bool): (default = False)
        spoof (bool): (default = False)

    CommandLine:
        python -m utool.util_grabdata --exec-grab_zipped_url --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_grabdata import *  # NOQA
        >>> import utool as ut
        >>> zipped_url = '?'
        >>> ensure = True
        >>> appname = 'utool'
        >>> download_dir = None
        >>> force_commonprefix = True
        >>> cleanup = False
        >>> redownload = False
        >>> spoof = False
        >>> result = grab_zipped_url(zipped_url, ensure, appname, download_dir,
        >>>                          force_commonprefix, cleanup, redownload,
        >>>                          spoof)
        >>> print(result)

    Examples:
        >>> from utool.util_grabdata import *  # NOQA
        >>> zipped_url = 'https://lev.cs.rpi.edu/public/data/testdata.zip'
        >>> zipped_url = 'http://www.spam.com/eggs/data.zip'

    """
    zipped_url = clean_dropbox_link(zipped_url)
    zip_fname = split(zipped_url)[1]
    data_name = split_archive_ext(zip_fname)[0]
    # Download zipfile to
    if download_dir is None:
        download_dir = util_cplat.get_app_resource_dir(appname)
    # Zipfile should unzip to:
    data_dir = join(download_dir, data_name)
    if ensure or redownload:
        if redownload:
            util_path.remove_dirs(data_dir)
        util_path.ensurepath(download_dir)
        if not exists(data_dir):
            # Download and unzip testdata
            zip_fpath = realpath(join(download_dir, zip_fname))
            #print('[utool] Downloading archive %s' % zip_fpath)
            if not exists(zip_fpath):
                download_url(zipped_url, zip_fpath, spoof=spoof)
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
        req = urllib2.Request('http://freegeoip.net/json/',
                              headers={'User-Agent': 'Mozilla/5.0' })
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


def s3_dict_encode_to_str(s3_dict):
    default_s3_dict = {
        'bucket'          : None,
        'key'             : None,
        'auth_domain'     : None,
        'auth_access_id'  : None,
        'auth_secret_key' : None,
    }
    default_s3_dict.update(s3_dict)
    assert len(default_s3_dict.keys()) == 5

    for key in default_s3_dict.keys():
        if key.startswith('auth'):
            value = default_s3_dict[key]
            if value is None:
                default_s3_dict[key] = 'EMPTY'

    assert None not in default_s3_dict.values()
    values = (
        default_s3_dict['auth_access_id'],
        default_s3_dict['auth_secret_key'],
        default_s3_dict['auth_domain'],
        default_s3_dict['bucket'],
        default_s3_dict['key'],
    )
    return 's3://%s:%s@%s:%s:%s' % values


def s3_str_decode_to_dict(s3_str):
    default_s3_dict = {
        'bucket'          : None,
        'key'             : None,
        'auth_domain'     : None,
        'auth_access_id'  : None,
        'auth_secret_key' : None,
    }
    assert s3_str.startswith('s3://')

    s3_str = s3_str.strip('s3://')
    left, right = s3_str.split('@')
    left = left.split(':')
    right = right.split(':')
    default_s3_dict['bucket']          = right[1]
    default_s3_dict['key']             = right[2]
    default_s3_dict['auth_domain']     = right[0]
    default_s3_dict['auth_access_id']  = left[0]
    default_s3_dict['auth_secret_key'] = left[1]

    assert len(default_s3_dict.keys()) == 5
    assert None not in default_s3_dict.values()

    for key in default_s3_dict.keys():
        if key.startswith('auth'):
            value = default_s3_dict[key]
            if value == 'EMPTY':
                default_s3_dict[key] = None

    return default_s3_dict


def read_s3_contents(bucket, key, auth_access_id=None, auth_secret_key=None,
                     auth_domain=None):
    import boto
    from boto.s3.connection import S3Connection
    if auth_access_id is not None and auth_secret_key is not None:
        conn = S3Connection(auth_access_id, auth_secret_key)
        bucket = conn.get_bucket(bucket)
    else:
        # Use system defaults, located in /etc/boto.cfg
        # Alternatively, use user defaults, located in ~/.boto
        s3 = boto.connect_s3()
        bucket = s3.get_bucket(bucket)
    key = bucket.get_key(key)
    contents = key.get_contents_as_string()
    return contents


def grab_s3_contents(fpath, bucket, key, auth_access_id=None, auth_secret_key=None,
                     auth_domain=None):
    import boto
    from boto.s3.connection import S3Connection
    if auth_access_id is not None and auth_secret_key is not None:
        conn = S3Connection(auth_access_id, auth_secret_key)
        bucket = conn.get_bucket(bucket)
    else:
        # Use system defaults, located in /etc/boto.cfg
        # Alternatively, use user defaults, located in ~/.boto
        s3 = boto.connect_s3()
        bucket = s3.get_bucket(bucket)
    key = bucket.get_key(key)
    key.get_contents_to_filename(fpath)


def scp_pull(remote_path, local_path='.', remote='localhost', user=None):
    r""" wrapper for scp """
    import utool as ut
    if user is not None:
        remote_uri = user + '@' + remote + ':' + remote_path
    else:
        remote_uri = remote + ':' + remote_path
    scp_exe = 'scp'
    scp_args = (scp_exe, '-r', remote_uri, local_path)
    ut.cmd(scp_args)


def list_remote(remote_uri, verbose=False):
    """
    remote_uri = 'user@xx.xx.xx.xx'
    """
    remote_uri1, remote_dpath = remote_uri.split(':')
    if not remote_dpath:
        remote_dpath = '.'
    import utool as ut
    out = ut.cmd('ssh', remote_uri1, 'ls -l %s' % (remote_dpath,), verbose=verbose)
    import re
    # Find lines that look like ls output
    split_lines = [re.split(r'\s+', t) for t in out[0].split('\n')]
    paths = [' '.join(t2[8:]) for t2 in split_lines if len(t2) > 8]
    return paths


def rsync(src_uri, dst_uri, exclude_dirs=[], port=22, dryrun=False):
    r"""
    Wrapper for rsync

    General function to push or pull a directory from a remote server to a
    local path

    References:
        http://www.tecmint.com/rsync-local-remote-file-synchronization-commands/
        http://serverfault.com/questions/219013/show-progress-in-rsync

    Notes (rsync commandline options):
        rsync [OPTION]... SRC [SRC]... DEST
        -v : verbose
        -r : copies data recursively (but dont preserve timestamps and
                permission while transferring data
        -a : archive mode, allows recursive copying and preserves symlinks,
                permissions, user and group ownerships, and timestamps
        -z : compress file data
        -i, --itemize-changes       output a change-summary for all updates
        -s, --protect-args :        no space-splitting; only wildcard special-chars
        -h : human-readable, output numbers in a human-readable format
        -P                          same as --partial --progress
    """
    from utool import util_cplat
    rsync_exe = 'rsync'
    rsync_options = '-avhzP'
    #rsync_options += ' --port=%d' % (port,)
    rsync_options += ' -e "ssh -p %d"' % (port,)
    if len(exclude_dirs) > 0:
        exclude_tup = ['--exclude ' + dir_ for dir_ in exclude_dirs]
        exclude_opts = ' '.join(exclude_tup)
        rsync_options += ' ' + exclude_opts

    cmdtuple = (rsync_exe, rsync_options, src_uri, dst_uri)
    cmdstr = ' '.join(cmdtuple)
    print('[rsync] src_uri = %r ' % (src_uri,))
    print('[rsync] dst_uri = %r ' % (dst_uri,))
    print('[rsync] cmdstr = %r' % cmdstr)
    print(cmdstr)

    #if not dryrun:
    util_cplat.cmd(cmdstr, dryrun=dryrun)


if __name__ == '__main__':
    """
    CommandLine:
        sh -c "python ~/code/utool/utool/util_grabdata.py --all-examples"
        python -m utool.util_grabdata
        python -m utool.util_grabdata --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
