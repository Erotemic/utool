# -*- coding: utf-8 -*-
"""
python -c "import utool, doctest; print(doctest.testmod(utool.util_path))"

This module becomes nav
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, filter, filterfalse, map, range
import six
from os.path import (join, basename, relpath, normpath, split, isdir, isfile,
                     exists, islink, ismount, dirname, splitext, realpath,
                     splitdrive, commonprefix, expanduser)
import os
import re
import sys
import shutil
import fnmatch
import warnings
import itertools
from utool.util_regex import extend_regex
from utool import util_dbg
from utool import util_progress
from utool._internal import meta_util_path
from utool import util_inject
from utool import util_arg
from utool import util_str
from utool._internal.meta_util_arg import NO_ASSERTS, VERBOSE, VERYVERBOSE, QUIET
print, rrr, profile = util_inject.inject2(__name__)
print_ = util_inject.make_module_write_func(__name__)

try:
    import pathlib
    HAVE_PATHLIB = True
except ImportError:
    HAVE_PATHLIB = False


PRINT_CALLER = util_arg.get_argflag('--print-caller')  # FIXME: name

__IMG_EXTS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.ppm'] + ['.gif', '.bmp']
__LOWER_EXTS = list(ext.lower() for ext in __IMG_EXTS)
__UPPER_EXTS = list(ext.upper() for ext in __IMG_EXTS)
IMG_EXTENSIONS =  set(__LOWER_EXTS + __UPPER_EXTS)


def newcd(path):
    """ DEPRICATE """
    cwd = os.getcwd()
    os.chdir(path)
    return cwd


unixpath = meta_util_path.unixpath
truepath = meta_util_path.truepath
unixjoin = meta_util_path.unixjoin


def ensure_ext(fpath, ext, replace=False):
    r"""
    Args:
        fpath (str):  file name or path
        ext (str or list): valid extensions
        replace (bool): if true all other extensions are removed.  this removes
            all nonstarting characters including and after the first period.
            Otherwise only the trailing extension is kept

    Returns:
        str: fpath -  file name or path

    CommandLine:
        python -m utool.util_path --exec-ensure_ext --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> print(ut.ensure_ext('foo', '.bar'))
        foo.bar
        >>> print(ut.ensure_ext('foo.bar', '.bar'))
        foo.bar
        >>> print(ut.ensure_ext('foo.bar', '.baz'))
        foo.bar.baz
        >>> print(ut.ensure_ext('foo.bar', '.baz', True))
        foo.baz
        >>> print(ut.ensure_ext('foo.bar.baz', '.biz', True))
        foo.biz
        >>> print(ut.ensure_ext('..foo.bar.baz', '.biz', True))
        ..foo.biz
    """
    import utool as ut
    dpath, fname = split(fpath)
    fname_ = fname.lstrip('.')
    n_leading_dots = len(fname) - len(fname_)
    fname_, ext_ = splitext(fname_)
    valid_exts = list(ut.ensure_iterable(ext))
    if ext_ not in valid_exts:
        if replace:
            fname = fname_.split('.')[0] + valid_exts[0]
        else:
            fname = fname_ + ext_ + valid_exts[0]
    fpath = join(dpath, ('.' * n_leading_dots) + fname)
    return fpath


def relpath_unix(path, otherpath):
    return relpath(path, otherpath).replace('\\', '/')


def truepath_relative(path, otherpath=None):
    """ Normalizes and returns absolute path with so specs

    Args:
        path (str):  path to file or directory
        otherpath (None): (default = None)

    Returns:
        str: path_

    CommandLine:
        python -m utool.util_path --exec-truepath_relative --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> path = r'C:/foobar/foobiz'
        >>> otherpath = r'C:/foobar'
        >>> path_ = truepath_relative(path, otherpath)
        >>> result = ('path_ = %s' % (ut.repr2(path_),))
        >>> print(result)
        path_ = 'foobiz'
    """
    if otherpath is None:
        otherpath = os.getcwd()
    otherpath = truepath(otherpath)
    path_ = normpath(relpath(path, otherpath))
    return path_


def tail(fpath, n=2, trailing=True):
    """ Alias for path_ndir_split """
    return path_ndir_split(fpath, n=n, trailing=trailing)


def unexpanduser(path):
    r"""
    Replaces home directory with '~'
    """
    homedir = expanduser('~')
    if path.startswith(homedir):
        path = '~' + path[len(homedir):]
    return path


def path_ndir_split(path_, n, force_unix=True, winroot='C:', trailing=True):
    r"""
    Shows only a little bit of the path. Up to the n bottom-level directories

    TODO: rename to path_tail? ndir_split?

    Returns:
        (str) the trailing n paths of path.

    CommandLine:
        python3 -m utool.util_path --test-path_ndir_split
        python3 -m utool --tf path_ndir_split
        python -m utool --tf path_ndir_split


    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> paths = [r'/usr/bin/local/foo/bar',
        ...          r'C:/',
        ...          #r'lonerel',
        ...          #r'reldir/other',
        ...          r'/ham',
        ...          r'./eggs',
        ...          r'/spam/eggs',
        ...          r'C:\Program Files (x86)/foobar/bin']
        >>> N = 2
        >>> iter_ = ut.iprod(paths, range(1, N + 1))
        >>> force_unix = True
        >>> tuplist = [(n, path_ndir_split(path_, n)) for path_, n in iter_]
        >>> chunklist = list(ut.ichunks(tuplist, N))
        >>> list_ = [['n=%r: %s' % (x, ut.reprfunc(y)) for x, y in chunk]
        >>>          for chunk in chunklist]
        >>> line_list = [', '.join(strs) for strs in list_]
        >>> result = '\n'.join(line_list)
        >>> print(result)
        n=1: '.../bar', n=2: '.../foo/bar'
        n=1: 'C:/', n=2: 'C:/'
        n=1: '.../ham', n=2: '/ham'
        n=1: '.../eggs', n=2: './eggs'
        n=1: '.../eggs', n=2: '.../spam/eggs'
        n=1: '.../bin', n=2: '.../foobar/bin'
    """
    if not isinstance(path_, six.string_types):
        # Probably given a file pointer
        return path_

    if n is None:
        cplat_path = ensure_crossplat_path(path_)
    elif n == 0:
        cplat_path = ''
    else:
        sep = '/' if force_unix else os.sep
        ndirs_list = []
        head = path_
        reached_end = False
        for nx in range(n):
            head, tail = split(head)
            if tail == '':
                if head == '':
                    reached_end = True
                else:
                    root = head if len(ndirs_list) == 0 else head.strip('\\/')
                    ndirs_list.append(root)
                    reached_end = True
                break
            else:
                ndirs_list.append(tail)
        if trailing and not reached_end:
            head, tail = split(head)
            if len(tail) == 0:
                if len(head) == 0:  # or head == '/':
                    reached_end = True
        ndirs = sep.join(ndirs_list[::-1])
        cplat_path = ensure_crossplat_path(ndirs)
        #if trailing and not reached_end:
        if trailing and not reached_end:
            cplat_path = '.../' + cplat_path
    return cplat_path


def remove_file(fpath, verbose=None, ignore_errors=True, dryrun=False,
                quiet=QUIET):
    """ Removes a file """
    if verbose is None:
        verbose = not quiet
    if dryrun:
        if verbose:
            print('[util_path] Dryrem %r' % fpath)
        return
    else:
        try:
            os.remove(fpath)
            if verbose:
                print('[util_path] Removed %r' % fpath)
        except OSError:
            print('[util_path.remove_file] Misrem %r' % fpath)
            #warnings.warn('OSError: %s,\n Could not delete %s' % (str(e), fpath))
            if not ignore_errors:
                raise
            return False
    return True


def remove_dirs(dpath, verbose=None, ignore_errors=True, dryrun=False,
                quiet=QUIET):
    r"""
    Recursively removes a single directory (need to change function name)

    DEPRICATE

    Args:
        dpath (str):  directory path
        dryrun (bool): (default = False)
        ignore_errors (bool): (default = True)
        quiet (bool): (default = False)

    Returns:
        bool: False

    CommandLine:
        python -m utool.util_path --test-remove_dirs

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> dpath = ut.ensure_app_resource_dir('utool', 'testremovedir')
        >>> assert exists(dpath), 'nothing to remove'
        >>> flag = remove_dirs(dpath, verbose=True)
        >>> print('flag = %r' % (flag,))
        >>> assert not exists(dpath), 'did not remove dpath'
        >>> assert flag is True
    """
    if verbose is None:
        verbose = not quiet
    if verbose:
        print('[util_path] Removing directory: %r' % dpath)
    if dryrun:
        return False
    try:
        shutil.rmtree(dpath)
    except OSError as e:
        warnings.warn('OSError: %s,\n Could not delete %s' % (str(e), dpath))
        if not ignore_errors:
            raise
        return False
    return True

#import os


def augpath(path, augsuf='', augext='', augpref='', augdir=None, newext=None,
            newfname=None, ensure=False, prefix=None, suffix=None):
    """
    augments end of path before the extension.

    augpath

    Args:
        path (str):
        augsuf (str): augment filename before extension

    Returns:
        str: newpath

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> path = 'somefile.txt'
        >>> augsuf = '_aug'
        >>> newpath = augpath(path, augsuf)
        >>> result = str(newpath)
        >>> print(result)
        somefile_aug.txt

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> path = 'somefile.txt'
        >>> augsuf = '_aug2'
        >>> newext = '.bak'
        >>> augdir = 'backup'
        >>> newpath = augpath(path, augsuf, newext=newext, augdir=augdir)
        >>> result = str(newpath)
        >>> print(result)
        backup/somefile_aug2.bak
    """
    if prefix is not None:
        augpref = prefix
    if suffix is not None:
        augsuf = suffix
    # Breakup path
    dpath, fname = split(path)
    fname_noext, ext = splitext(fname)
    if newfname is not None:
        fname_noext = newfname
    # Augment ext
    if newext is None:
        newext = ext
    # Augment fname
    new_fname = ''.join((augpref, fname_noext, augsuf, newext, augext))
    # Augment dpath
    if augdir is not None:
        new_dpath = join(dpath, augdir)
        if ensure:
            # create new dir if needebe
            ensuredir(new_dpath)
    else:
        new_dpath = dpath
    # Recombine into new path
    newpath = join(new_dpath, new_fname)
    return newpath


def remove_files_in_dir(dpath, fname_pattern_list='*', recursive=False,
                        verbose=VERBOSE, dryrun=False, ignore_errors=False):
    """ Removes files matching a pattern from a directory """
    if isinstance(fname_pattern_list, six.string_types):
        fname_pattern_list = [fname_pattern_list]
    if verbose > 2:
        print('[util_path] Removing files:')
        print('  * from dpath = %r ' % dpath)
        print('  * with patterns = %r' % fname_pattern_list)
        print('  * recursive = %r' % recursive)
    num_removed, num_matched = (0, 0)
    if not exists(dpath):
        msg = ('!!! dir = %r does not exist!' % dpath)
        if verbose:
            print(msg)
        warnings.warn(msg, category=UserWarning)
    for root, dname_list, fname_list in os.walk(dpath):
        for fname_pattern in fname_pattern_list:
            for fname in fnmatch.filter(fname_list, fname_pattern):
                num_matched += 1
                num_removed += remove_file(join(root, fname),
                                           ignore_errors=ignore_errors,
                                           dryrun=dryrun,
                                           verbose=verbose > 5)
        if not recursive:
            break
    if verbose > 0:
        print('[util_path] ... Removed %d/%d files' % (num_removed, num_matched))
    return True


def delete(path, dryrun=False, recursive=True, verbose=None, print_exists=True,
           ignore_errors=True):
    """ Removes a file, directory, or symlink """
    if verbose is None:
        verbose = VERBOSE
        if not QUIET:
            verbose = 1
    if verbose > 0:
        print('[util_path] Deleting path=%r' % path)
    exists_flag = exists(path)
    link_flag = islink(path)
    if not exists_flag and not link_flag:
        if print_exists and verbose:
            print('..does not exist!')
        flag = False
    else:
        rmargs = dict(verbose=verbose > 1, ignore_errors=ignore_errors,
                      dryrun=dryrun)
        if islink(path):
            os.unlink(path)
            flag = True
        elif isdir(path):
            # First remove everything in the directory
            flag = remove_files_in_dir(path, recursive=recursive, **rmargs)
            # Then remove the directory itself
            flag = flag and remove_dirs(path, **rmargs)
        elif isfile(path):
            flag = remove_file(path, **rmargs)
        else:
            raise ValueError('Unknown type of path=%r' % (path,))
        if verbose > 0:
            print('[util_path] Finished deleting path=%r' % path)
    return flag


def remove_existing_fpaths(fpath_list, verbose=VERBOSE, quiet=QUIET,
                           strict=False, print_caller=PRINT_CALLER,
                           lbl='files'):
    """ checks existance before removing. then tries to remove exisint paths """
    import utool as ut
    if print_caller:
        print(util_dbg.get_caller_name(range(1, 4)) + ' called remove_existing_fpaths')
    fpath_list_ = ut.filter_Nones(fpath_list)
    exists_list = list(map(exists, fpath_list_))
    if verbose:
        n_total = len(fpath_list)
        n_valid = len(fpath_list_)
        n_exist = sum(exists_list)
        print('[util_path.remove_existing_fpaths] request delete of %d %s' % (
            n_total, lbl))
        if n_valid != n_total:
            print(('[util_path.remove_existing_fpaths] '
                   'trying to delete %d/%d non None %s ') %
                  (n_valid, n_total, lbl))
        print(('[util_path.remove_existing_fpaths] '
               ' %d/%d exist and need to be deleted')
              % (n_exist, n_valid))
    existing_fpath_list = ut.compress(fpath_list_, exists_list)
    return remove_fpaths(existing_fpath_list, verbose=verbose, quiet=quiet,
                            strict=strict, print_caller=False, lbl=lbl)


def remove_fpaths(fpaths, verbose=VERBOSE, quiet=QUIET, strict=False,
                  print_caller=PRINT_CALLER, lbl='files'):
    """
    Removes multiple file paths
    """
    import utool as ut
    if print_caller:
        print(util_dbg.get_caller_name(range(1, 4)) + ' called remove_fpaths')
    n_total = len(fpaths)
    _verbose = (not quiet and n_total > 0) or VERYVERBOSE
    if _verbose:
        print('[util_path.remove_fpaths] try removing %d %s' % (n_total, lbl))
    n_removed = 0
    prog = ut.ProgIter(fpaths, label='removing files', enabled=verbose)
    _iter = iter(prog)
    # Try to be fast at first
    try:
        for fpath in _iter:
            os.remove(fpath)
            n_removed += 1
    except OSError as ex:
        # Buf if we fail put a try in the inner loop
        if VERYVERBOSE:
            print('WARNING: Could not remove fpath = %r' % (fpath,))
        if strict:
            util_dbg.printex(ex, 'Could not remove fpath = %r' % (fpath,),
                             iswarning=False)
            raise
        for fpath in _iter:
            try:
                os.remove(fpath)
                n_removed += 1
            except OSError:
                if VERYVERBOSE:
                    print('WARNING: Could not remove fpath = %r' % (fpath,))
    if _verbose:
        print('[util_path.remove_fpaths] ... removed %d / %d %s' % (
            n_removed, n_total, lbl))
    return n_removed


remove_file_list = remove_fpaths  # backwards compatible


def longest_existing_path(_path):
    r"""
    Returns the longest root of _path that exists

    Args:
        _path (str):  path string

    Returns:
        str: _path -  path string

    CommandLine:
        python -m utool.util_path --exec-longest_existing_path

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> target = dirname(ut.__file__)
        >>> _path = join(target, 'nonexist/foobar')
        >>> existing_path = longest_existing_path(_path)
        >>> result = ('existing_path = %s' % (str(existing_path),))
        >>> print(result)
        >>> assert existing_path == target
    """
    existing_path = _path
    while True:
        _path_new = os.path.dirname(existing_path)
        if exists(_path_new):
            existing_path = _path_new
            break
        if _path_new == existing_path:
            print('!!! [utool] This is a very illformated path indeed.')
            existing_path = ''
            break
        existing_path = _path_new
    return existing_path


def get_path_type(path_):
    r"""
    returns if a path is a file, directory, link, or mount
    """
    path_type = ''
    if isfile(path_):
        path_type += 'file'
    if isdir(path_):
        path_type += 'directory'
    if islink(path_):
        path_type += 'link'
    if ismount(path_):
        path_type += 'mount'
    return path_type


def checkpath(path_, verbose=VERYVERBOSE, n=None, info=VERYVERBOSE):
    r""" verbose wrapper around ``os.path.exists``

    Returns:
        true if ``path_`` exists on the filesystem show only the
        top `n` directories

    Args:
        path_ (str): path string
        verbose (bool): verbosity flag(default = False)
        n (int):  (default = None)
        info (bool): (default = False)

    CommandLine:
        python -m utool.util_path --test-checkpath

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> path_ = ut.__file__
        >>> verbose = True
        >>> n = None
        >>> info = False
        >>> result = checkpath(path_, verbose, n, info)
        >>> print(result)
        True

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> path_ = ut.__file__ + 'foobar'
        >>> verbose = True
        >>> result = checkpath(path_, verbose, n=None, info=True)
        >>> print(result)
        False
    """
    assert isinstance(path_, six.string_types), (
        'path_=%r is not a string. type(path_) = %r' % (path_, type(path_)))
    path_ = normpath(path_)
    if sys.platform.startswith('win32'):
        # convert back to windows style path if using unix style
        if path_.startswith('\\'):
            dirs = path_.split('\\')
            if len(dirs) > 1 and len(dirs[0]) == 0 and len(dirs[1]) == 1:
                dirs[1] = dirs[1].upper() + ':'
                path_ = '\\'.join(dirs[1:])
    does_exist = exists(path_)
    if verbose:
        #print_('[utool] checkpath(%r)' % (path_))
        pretty_path = path_ndir_split(path_, n)
        caller_name = util_dbg.get_caller_name(allow_genexpr=False)
        print('[%s] checkpath(%r)' % (caller_name, pretty_path))
        if does_exist:
            path_type = get_path_type(path_)
            #path_type = 'file' if isfile(path_) else 'directory'
            print('[%s] ...(%s) exists' % (caller_name, path_type,))
        else:
            print('[%s] ... does not exist' % (caller_name))
    if not does_exist and info:
        #print('[util_path]  ! Does not exist')
        _longest_path = longest_existing_path(path_)
        _longest_path_type = get_path_type(_longest_path)
        print('[util_path] ... The longest existing path is: %r' % _longest_path)
        print('[util_path] ... and has type %r' % (_longest_path_type,))
    return does_exist


def ensurepath(path_, verbose=None):
    """ DEPRICATE - alias - use ensuredir instead """
    if verbose is None:
        verbose = VERYVERBOSE
    return ensuredir(path_, verbose=verbose)


def ensuredir(path_, verbose=None, info=False, mode=0o1777):
    r"""
    Ensures that directory will exist. creates new dir with sticky bits by
    default

    Args:
        path (str): dpath to ensure. Can also be a tuple to send to join
        info (bool): if True prints extra information
        mode (int): octal mode of directory (default 0o1777)

    Returns:
        str: path - the ensured directory

    """
    if verbose is None:
        verbose = VERYVERBOSE
    if isinstance(path_, (list, tuple)):
        path_ = join(*path_)
    if HAVE_PATHLIB and isinstance(path_, pathlib.Path):
        path_ = str(path_)
    if not checkpath(path_, verbose=verbose, info=info):
        if verbose:
            print('[util_path] mkdir(%r)' % path_)
        try:
            os.makedirs(normpath(path_), mode=mode)
        except OSError as ex:
            util_dbg.printex(
                ex,
                'check that the longest existing path '
                'is not a bad windows symlink.', keys=['path_'])
            raise
    return path_


def touch(fpath, times=None, verbose=True):
    r"""
    Creates file if it doesnt exist

    Args:
        fpath (str): file path
        times (None):
        verbose (bool):

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> fpath = '?'
        >>> times = None
        >>> verbose = True
        >>> result = touch(fpath, times, verbose)
        >>> print(result)

    References:
        http://stackoverflow.com/questions/1158076/implement-touch-using-python
    """
    try:
        if verbose:
            print('[util_path] touching %r' % fpath)
        with open(fpath, 'a'):
            os.utime(fpath, times)
    except Exception as ex:
        import utool
        utool.printex(ex, 'touch %s' % fpath)
        raise
    return fpath


ensurefile = touch


# ---File Copy---

def _copy_worker(src, dst):
    try:
        shutil.copy2(src, dst)
    except OSError:
        return False
    except shutil.Error:
        pass
    return True


def copy_files_to(src_fpath_list, dst_dpath=None, dst_fpath_list=None,
                  overwrite=False, verbose=True, veryverbose=False):
    """
    parallel copier

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *
        >>> import utool as ut
        >>> overwrite = False
        >>> veryverbose = False
        >>> verbose = True
        >>> src_fpath_list = [ut.grab_test_imgpath(key)
        >>>                   for key in ut.get_valid_test_imgkeys()]
        >>> dst_dpath = ut.get_app_resource_dir('utool', 'filecopy_tests')
        >>> copy_files_to(src_fpath_list, dst_dpath, overwrite=overwrite,
        >>>               verbose=verbose)
    """
    from utool import util_list
    from utool import util_parallel

    if verbose:
        print('[util_path] +--- COPYING FILES ---')
        print('[util_path]  * len(src_fpath_list) = %r' % (len(src_fpath_list)))
        print('[util_path]  * dst_dpath = %r' % (dst_dpath,))

    if dst_fpath_list is None:
        ensuredir(dst_dpath, verbose=veryverbose)
        dst_fpath_list = [join(dst_dpath, basename(fpath))
                          for fpath in src_fpath_list]
    else:
        assert dst_dpath is None, 'dst_dpath was specified but overrided'
        assert len(dst_fpath_list) == len(src_fpath_list), 'bad correspondence'

    exists_list = list(map(exists, dst_fpath_list))
    if verbose:
        print('[util_path]  * %d files already exist dst_dpath' % (
            sum(exists_list),))
    if not overwrite:
        notexists_list = util_list.not_list(exists_list)
        dst_fpath_list_ = util_list.compress(dst_fpath_list, notexists_list)
        src_fpath_list_ = util_list.compress(src_fpath_list, notexists_list)
    else:
        dst_fpath_list_ = dst_fpath_list
        src_fpath_list_ = src_fpath_list

    args_list = zip(src_fpath_list_, dst_fpath_list_)
    _gen = util_parallel.generate2(_copy_worker, args_list,
                                   ntasks=len(src_fpath_list_))
    success_list = list(_gen)

    #success_list = copy_list(src_fpath_list_, dst_fpath_list_)
    if verbose:
        print('[util_path]  * Copied %d / %d' % (sum(success_list),
                                                 len(src_fpath_list)))
        print('[util_path] L___ DONE COPYING FILES ___')


def copy(src, dst, overwrite=True, deeplink=True, verbose=True, dryrun=False):
    import utool as ut
    if ut.isiterable(src):
        if not ut.isiterable(dst):
            # list to non list
            ut.copy_files_to(src, dst, overwrite=overwrite, verbose=verbose)
        else:
            # list to list
            ut.copy_files_to(src, dst_fpath_list=dst, overwrite=overwrite,
                             verbose=verbose)
    else:
        return copy_single(src, dst, overwrite=overwrite, deeplink=deeplink,
                           dryrun=dryrun, verbose=verbose)


def copy_single(src, dst, overwrite=True, verbose=True, deeplink=True,
                dryrun=False):
    r"""
    Args:
        src (str): file or directory to copy
        dst (str): directory or new file to copy to

    Copies src file or folder to dst.

    If src is a folder this copy is recursive.
    """
    try:
        if exists(src):
            if not isdir(src) and isdir(dst):
                # copying file to directory
                dst = join(dst, basename(src))
            if exists(dst):
                if overwrite:
                    prefix = 'C+O'
                    if verbose:
                        print('[util_path] [Copying + Overwrite]:')
                else:
                    prefix = 'Skip'
                    if verbose:
                        print('[%s] ->%s' % (prefix, dst))
                    return
            else:
                prefix = 'C'
                if verbose:
                    if dryrun:
                        print('[util_path] [DryRun]: ')
                    else:
                        print('[util_path] [Copying]: ')
            if verbose:
                print('[%s] | %s' % (prefix, src))
                print('[%s] ->%s' % (prefix, dst))
            if not dryrun:
                if not deeplink and islink(src):
                    linkto = os.readlink(src)
                    symlink(linkto, dst)
                elif isdir(src):
                    print('isdir')
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        else:
            prefix = 'Miss'
            if verbose:
                print('[util_path] [Cannot Copy]: ')
                print('[%s] src=%s does not exist!' % (prefix, src))
                print('[%s] dst=%s' % (prefix, dst))
    except Exception as ex:
        from utool import util_dbg
        util_dbg.printex(ex, 'Error copying single', keys=['src', 'dst'])
        raise


def copy_all(src_dir, dest_dir, glob_str_list, recursive=False):
    ensuredir(dest_dir)
    if not isinstance(glob_str_list, list):
        glob_str_list = [glob_str_list]
    for root, dirs, files in os.walk(src_dir):
        for dname_ in dirs:
            for glob_str in glob_str_list:
                if fnmatch.fnmatch(dname_, glob_str):
                    src = normpath(join(src_dir, dname_))
                    dst = normpath(join(dest_dir, dname_))
                    ensuredir(dst)
        for fname_ in files:
            for glob_str in glob_str_list:
                if fnmatch.fnmatch(fname_, glob_str):
                    src = normpath(join(src_dir, fname_))
                    dst = normpath(join(dest_dir, fname_))
                    copy(src, dst)
        if not recursive:
            break


def copy_list(src_list, dst_list, lbl='Copying',
              ioerr_ok=False, sherro_ok=False, oserror_ok=False):
    """ Copies all data and stat info """
    # Feb - 6 - 2014 Copy function
    task_iter = zip(src_list, dst_list)
    def docopy(src, dst):
        try:
            shutil.copy2(src, dst)
        except OSError:
            if ioerr_ok:
                return False
            raise
        except shutil.Error:
            if sherro_ok:
                return False
            raise
        except IOError:
            if ioerr_ok:
                return False
            raise
        return True
    progiter = util_progress.ProgIter(task_iter, adjust=True, lbl=lbl)
    success_list = [docopy(src, dst) for (src, dst) in progiter]
    return success_list


def move(src, dst, verbose=True):
    if verbose:
        print('[path] [Moving]: ')
        print('[path] | {}'.format(src))
        print('[path] ->{}'.format(dst))
    try:
        shutil.move(src, dst)
    except OSError:
        if verbose:
            print('[path] move failed')
        return False
    else:
        return True


def move_list(src_list, dst_list, lbl='Moving', verbose=True):
    import utool as ut
    # Feb - 6 - 2014 Move function
    def trymove(src, dst):
        try:
            shutil.move(src, dst)
        except OSError:
            return False
        return True
    length = ut.length_hint(src_list, default=None)
    task_iter = zip(src_list, dst_list)
    if verbose:
        _iter = ut.ProgIter(task_iter, length=length, lbl=lbl, adjust=True)
    else:
        _iter = task_iter
    success_list = [trymove(src, dst) for (src, dst) in _iter]
    return success_list


def file_bytes(fpath):
    r"""
    Args:
        fpath (str):  file path string

    Returns:
        int: size of file in bytes
    """
    return os.stat(fpath).st_size


def file_megabytes(fpath):
    r"""
    Args:
        fpath (str):  file path string

    Returns:
        float: size of file in megabytes
    """
    return os.stat(fpath).st_size / (2.0 ** 20)


def glob_python_modules(dirname, **kwargs):
    return glob(dirname, '*.py', recursive=True, with_dirs=False)


def glob(dpath, pattern=None, recursive=False, with_files=True, with_dirs=True,
         maxdepth=None, exclude_dirs=[], fullpath=True, **kwargs):
    r"""
    Globs directory for pattern

    DEPRICATED:
        use pathlib.glob instead

    Args:
        dpath (str): directory path or pattern
        pattern (str or list): pattern or list of patterns
            (use only if pattern is not in dpath)
        recursive (bool): (default = False)
        with_files (bool): (default = True)
        with_dirs (bool): (default = True)
        maxdepth (None): (default = None)
        exclude_dirs (list): (default = [])

    Returns:
        list: path_list

    SeeAlso:
        iglob

    CommandLine:
        python -m utool.util_path --test-glob
        python -m utool.util_path --exec-glob:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> from os.path import dirname
        >>> import utool as ut
        >>> dpath = dirname(ut.__file__)
        >>> pattern = '__*.py'
        >>> recursive = True
        >>> with_files = True
        >>> with_dirs = True
        >>> maxdepth = None
        >>> fullpath = False
        >>> exclude_dirs = ['_internal', join(dpath, 'experimental')]
        >>> print('exclude_dirs = ' + ut.repr2(exclude_dirs))
        >>> path_list = glob(dpath, pattern, recursive, with_files, with_dirs,
        >>>                  maxdepth, exclude_dirs, fullpath)
        >>> path_list = sorted(path_list)
        >>> result = ('path_list = %s' % (ut.repr3(path_list),))
        >>> result = result.replace(r'\\', '/')
        >>> # xdoctest: +REQUIRES(POSIX)
        >>> print(result)
        path_list = [
            '__init__.py',
            '__main__.py',
            'tests/__init__.py',
        ]

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> dpath = dirname(ut.__file__) + '/__*.py'
        >>> path_list = glob(dpath)
        >>> result = ('path_list = %s' % (str(path_list),))
        >>> print(result)
    """
    gen = iglob(dpath, pattern, recursive=recursive,
                with_files=with_files, with_dirs=with_dirs, maxdepth=maxdepth,
                fullpath=fullpath, exclude_dirs=exclude_dirs, **kwargs)
    path_list = list(gen)
    return path_list


def iglob(dpath, pattern=None, recursive=False, with_files=True, with_dirs=True,
          maxdepth=None, exclude_dirs=[], fullpath=True, **kwargs):
    r"""
    Iteratively globs directory for pattern

    FIXME:
        This function has a speed issue

    Args:
        dpath (str):  directory path
        pattern (str):
        recursive (bool): (default = False)
        with_files (bool): (default = True)
        with_dirs (bool): (default = True)
        maxdepth (None): (default = None)
        exclude_dirs (list): (default = [])

    Yields:
        path

    References:
        http://stackoverflow.com/questions/19859840/excluding-dirs-in-os-walk
    """
    from utool import util_iter
    if kwargs.get('verbose', False):  # log what i'm going to do
        print('[util_path] glob(dpath=%r)' % truepath(dpath,))

    debug = False
    if pattern is None:
        # separate extract pattern from dpath
        if debug:
            print('[iglob] parsing dpath = %r' % (dpath,))
        dpath_ = dpath
        dpath = longest_existing_path(dpath_)
        pattern = relpath(dpath_, dpath)
    else:
        # hack check for pattern
        GLOB_PATS = ['*', '?']
        for _ in GLOB_PATS:
            assert dpath.find(_) == -1, (
                'warning: pattern _=%r in dpath, but a pattern was specified' %
                (_,))
    if isinstance(pattern, list):
        # overload pattern with list
        pattern_list  = pattern
        subiters = (
            iglob(dpath, pattern=pattern, recursive=recursive,
                  with_files=with_files, with_dirs=with_dirs,
                  maxdepth=maxdepth, exclude_dirs=exclude_dirs,
                  fullpath=fullpath, **kwargs)
            for pattern in pattern_list
        )
        for item in util_iter.iflatten(subiters):
            yield item
        raise StopIteration
    if kwargs.get('verbose', False):
        print('[iglob] pattern = %r' % (pattern,))
        print('[iglob] dpath = %r' % (dpath,))
    n_files = 0
    n_dirs  = 0
    current_depth = 0
    dpath_ = truepath(dpath)
    posx1 = len(dpath_) + len(os.path.sep)
    #exclude_dirs_rel = [relpath(dpath_, dir_) for dir_ in exclude_dirs]
    #exclude_dirs_rel = [relpath(dpath_, dir_) for dir_ in exclude_dirs]
    #print('\n\n\n')
    #import utool as ut
    #print('exclude_dirs = %s' % (ut.repr4(exclude_dirs),))
    for root, dirs, files in os.walk(dpath_, topdown=True):
        # Modifying dirs in-place will prune the subsequent files and
        # directories visitied by os.walk
        # References:
        #     http://stackoverflow.com/questions/19859840/excluding-directories-in-os-walk
        rel_root = relpath(root, dpath_)
        rel_root2 = relpath(root, dirname(dpath_))
        #print('rel_root = %r' % (rel_root,))
        #if len(dirs) > 0:
        #    print('dirs = %s' % (ut.repr4([join(rel_root, d) for d in dirs]),))
        if len(exclude_dirs) > 0:
            dirs[:] = [d for d in dirs if normpath(join(rel_root, d)) not in exclude_dirs]
            # hack
            dirs[:] = [d for d in dirs if normpath(join(rel_root2, d)) not in exclude_dirs]
            # check abs path as well
            dirs[:] = [d for d in dirs if normpath(join(root, d)) not in exclude_dirs]

        # yeild data
        # print it only if you want
        if maxdepth is not None:
            current_depth = root[posx1:].count(os.path.sep)
            if maxdepth <= current_depth:
                continue
        #print('-----------')
        #print(current_depth)
        #print(root)
        #print('==')
        #print(dirs)
        #print('-----------')
        if with_files:
            for fname in fnmatch.filter(files, pattern):
                n_files += 1
                fpath = join(root, fname)
                if fullpath:
                    yield fpath
                else:
                    yield relpath(fpath, dpath_)

        if with_dirs:
            for dname in fnmatch.filter(dirs, pattern):
                dpath = join(root, dname)
                n_dirs += 1
                if fullpath:
                    yield dpath
                else:
                    yield relpath(dpath, dpath_)
        if not recursive:
            break
    if kwargs.get('verbose', False):  # log what i've done
        n_total = n_dirs + n_files
        print('[util_path] iglob Found: %d' % (n_total))


# --- Images ----

def num_images_in_dir(path):
    """
    returns the number of images in a directory
    """
    num_imgs = 0
    for root, dirs, files in os.walk(path):
        for fname in files:
            if fpath_has_imgext(fname):
                num_imgs += 1
    return num_imgs


def fpath_has_imgext(fname):
    """ returns true if a filename matches an image pattern """
    return fpath_has_ext(fname, IMG_EXTENSIONS)


def fpath_has_ext(fname, exts, case_sensitive=False):
    """ returns true if the filename has any of the given extensions """
    fname_ = fname.lower() if not case_sensitive else fname
    if case_sensitive:
        ext_pats = ['*' + ext for ext in exts]
    else:
        ext_pats = ['*' + ext.lower() for ext in exts]
    return any([fnmatch.fnmatch(fname_, pat) for pat in ext_pats])


def dirsplit(path):
    r"""
    Args:
        path (str):

    Returns:
        list: components of the path

    CommandLine:
        python -m utool.util_path --exec-dirsplit

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> paths = []
        >>> paths.append('E:/window file/foo')
        >>> paths.append('/normal/foo')
        >>> paths.append('~/relative/path')
        >>> results = [dirsplit(path) for path in paths]
        >>> import re
        >>> results2 = [re.split('\\/', path) for path in paths]
        >>> print(results2)
        >>> result = ut.repr2(results)
        >>> print(result)
    """
    #return path.split(os.sep)
    parts = []
    remain = path
    part = True
    #while True:
    while part != '' and remain != '':
        remain, part = split(remain)
        parts.append(part)
    parts = [p for p in parts if p != '']
    if remain != '':
        parts.append(remain)
    parts = parts[::-1]
    return parts


def fpaths_to_fnames(fpath_list):
    """
    Args:
        fpath_list (list of strs): list of file-paths
    Returns:
        fname_list (list of strs): list of file-names
    """
    fname_list = [split(fpath)[1] for fpath in fpath_list]
    return fname_list


def fnames_to_fpaths(fname_list, path):
    fpath_list = [join(path, fname) for fname in fname_list]
    return fpath_list


def get_modpath(modname, prefer_pkg=False, prefer_main=False):
    r"""
    Returns path to module

    Args:
        modname (str or module): module name or actual module

    Returns:
        str: module_dir

    CommandLine:
        python -m utool.util_path --test-get_modpath

    Setup:
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> utool_dir = dirname(dirname(ut.__file__))

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> utool_dir = dirname(dirname(ut.__file__))
        >>> modname = 'utool.util_path'
        >>> module_dir = get_modpath(modname)
        >>> result = ut.truepath_relative(module_dir, utool_dir)
        >>> result = ut.ensure_unixslash(result)
        >>> print(result)
        utool/util_path.py

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> utool_dir = dirname(dirname(ut.__file__))
        >>> modname = 'utool._internal'
        >>> module_dir = get_modpath(modname, prefer_pkg=True)
        >>> result = ut.ensure_unixslash(module_dir)
        >>> print(result)
        >>> assert result.endswith('utool/_internal')

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> utool_dir = dirname(dirname(ut.__file__))
        >>> modname = 'utool'
        >>> module_dir = get_modpath(modname)
        >>> result = ut.truepath_relative(module_dir, utool_dir)
        >>> result = ut.ensure_unixslash(result)
        >>> print(result)
        utool/__init__.py
    """
    import importlib
    if isinstance(modname, six.string_types):
        module = importlib.import_module(modname)
    else:
        module = modname  # Hack
    modpath = module.__file__.replace('.pyc', '.py')
    initname = '__init__.py'
    mainname = '__main__.py'
    if prefer_pkg:
        if modpath.endswith(initname) or modpath.endswith(mainname):
            modpath = dirname(modpath)
            # modpath = modpath[:-len(initname)]
    if prefer_main:
        if modpath.endswith(initname):
            main_modpath = modpath[:-len(initname)] + mainname
            if exists(main_modpath):
                modpath = main_modpath
    #modname = modname.replace('.__init__', '').strip()
    #module_dir = get_module_dir(module)
    return modpath


def get_module_dir(module, *args):
    module_dir = truepath(dirname(module.__file__))
    if len(args) > 0:
        module_dir = join(module_dir, *args)
    return module_dir


def ensure_unixslash(path):
    return path.replace('\\', '/')


def ensure_crossplat_path(path, winroot='C:'):
    r"""
    ensure_crossplat_path

    Args:
        path (str):

    Returns:
        str: crossplat_path

    Example(DOCTEST):
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> path = r'C:\somedir'
        >>> cplat_path = ensure_crossplat_path(path)
        >>> result = cplat_path
        >>> print(result)
        C:/somedir
    """
    cplat_path = path.replace('\\', '/')
    if cplat_path == winroot:
        cplat_path += '/'
    return cplat_path


def ensure_native_path(path, winroot='C:'):
    import utool as ut
    if ut.WIN32 and path.startswith('/') or  path.startswith('\\'):
        path = winroot + path


def get_relative_modpath(module_fpath):
    """
    Returns path to module relative to the package root

    Args:
        module_fpath (str): module filepath

    Returns:
        str: modname

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> module_fpath = ut.util_path.__file__
        >>> rel_modpath = ut.get_relative_modpath(module_fpath)
        >>> rel_modpath = rel_modpath.replace('.pyc', '.py')  # allow pyc or py
        >>> result = ensure_crossplat_path(rel_modpath)
        >>> print(result)
        utool/util_path.py
    """
    modsubdir_list = get_module_subdir_list(module_fpath)
    _, ext = splitext(module_fpath)
    rel_modpath = join(*modsubdir_list) + ext
    rel_modpath = ensure_crossplat_path(rel_modpath)
    return rel_modpath


def get_modname_from_modpath(module_fpath):
    """
    returns importable name from file path

    get_modname_from_modpath

    Args:
        module_fpath (str): module filepath

    Returns:
        str: modname

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> module_fpath = ut.util_path.__file__
        >>> modname = ut.get_modname_from_modpath(module_fpath)
        >>> result = modname
        >>> print(result)
        utool.util_path
    """
    modsubdir_list = get_module_subdir_list(module_fpath)
    modname = '.'.join(modsubdir_list)
    modname = modname.replace('.__init__', '').strip()
    modname = modname.replace('.__main__', '').strip()
    return modname


def get_module_subdir_list(module_fpath):
    """
    get_module_subdir_list

    Args:
        module_fpath (str):

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> module_fpath = ut.util_path.__file__
        >>> modsubdir_list = get_module_subdir_list(module_fpath)
        >>> result = modsubdir_list
        >>> print(result)
        ['utool', 'util_path']
    """
    module_fpath = truepath(module_fpath)
    dpath, fname_ext = split(module_fpath)
    fname, ext = splitext(fname_ext)
    full_dpath = dpath
    dpath = full_dpath
    _modsubdir_list = [fname]
    while is_module_dir(dpath):
        dpath, dname = split(dpath)
        _modsubdir_list.append(dname)
    modsubdir_list = _modsubdir_list[::-1]
    return modsubdir_list


def ls(path, pattern='*'):
    """ like unix ls - lists all files and dirs in path"""
    path_iter = glob(path, pattern, recursive=False)
    return sorted(list(path_iter))


def ls_dirs(path, pattern='*'):
    dir_iter = list(glob(path, pattern, recursive=False, with_files=False))
    return sorted(list(dir_iter))


def ls_modulefiles(path, private=True, full=True, noext=False):
    module_file_list = ls(path, '*.py')
    module_file_iter = iter(module_file_list)
    if not private:
        module_file_iter = filterfalse(is_private_module, module_file_iter)
    if not full:
        module_file_iter = map(basename, module_file_iter)
    if noext:
        module_file_iter = (splitext(path)[0] for path in module_file_iter)
    return list(module_file_iter)


def ls_moduledirs(path, private=True, full=True):
    """ lists all dirs which are python modules in path """
    dir_list = ls_dirs(path)
    module_dir_iter = filter(is_module_dir, dir_list)
    if not private:
        module_dir_iter = filterfalse(is_private_module, module_dir_iter)
    if not full:
        module_dir_iter = map(basename, module_dir_iter)
    return list(module_dir_iter)


def basename_noext(path):
    return splitext(basename(path))[0]


def is_private_module(path):
    return basename(path).startswith('__')


def is_python_module(path):
    return path.endswith('.py') or path.endswith('.pyc')


def is_module_dir(path):
    """
    Args:
        path (str)

    Returns:
        flag: True if path contains an __init__ file

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> path = truepath('~/code/utool/utool')
        >>> flag = is_module_dir(path)
        >>> result = (flag)
        >>> print(result)
    """
    return exists(join(path, '__init__.py'))


def list_images(img_dpath_, ignore_list=[], recursive=False, fullpath=False,
                full=None, sort=True):
    r"""
    Returns a list of images in a directory. By default returns relative paths.

    TODO: rename to ls_images
    TODO: Change all instances of fullpath to full

    Args:
        img_dpath_ (str):
        ignore_list (list): (default = [])
        recursive (bool): (default = False)
        fullpath (bool): (default = False)
        full (None): (default = None)
        sort (bool): (default = True)

    Returns:
        list: gname_list

    CommandLine:
        python -m utool.util_path --exec-list_images

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> img_dpath_ = '?'
        >>> ignore_list = []
        >>> recursive = False
        >>> fullpath = False
        >>> full = None
        >>> sort = True
        >>> gname_list = list_images(img_dpath_, ignore_list, recursive,
        >>>                          fullpath, full, sort)
        >>> result = ('gname_list = %s' % (str(gname_list),))
        >>> print(result)
    """
    #if not QUIET:
    #    print(ignore_list)
    if full is not None:
        fullpath = fullpath or full
    img_dpath_ = util_str.ensure_unicode(img_dpath_)
    img_dpath = realpath(img_dpath_)
    ignore_set = set(ignore_list)
    gname_list_ = []
    assertpath(img_dpath)
    # Get all the files in a directory recursively
    true_imgpath = truepath(img_dpath)
    for root, dlist, flist in os.walk(true_imgpath):
        root = util_str.ensure_unicode(root)
        rel_dpath = relpath(root, img_dpath)
        # Ignore directories
        if any([dname in ignore_set for dname in dirsplit(rel_dpath)]):
            continue
        for fname in iter(flist):
            fname = util_str.ensure_unicode(fname)
            gname = join(rel_dpath, fname).replace('\\', '/')
            if gname.startswith('./'):
                gname = gname[2:]
            if fpath_has_imgext(gname):
                # Ignore Files
                if gname in ignore_set:
                    continue
                if fullpath:
                    gpath = join(img_dpath, gname)
                    gname_list_.append(gpath)
                else:
                    gname_list_.append(gname)
        if not recursive:
            break
    if sort:
        gname_list = sorted(gname_list_)
    return gname_list


ls_images = list_images


def assertpath(path_, msg='', **kwargs):
    """ Asserts that a patha exists """
    if NO_ASSERTS:
        return
    if path_ is None:
        raise AssertionError('path=%r is None! %s' % (path_, msg))
    if path_ == '':
        raise AssertionError('path=%r is the empty string! %s' % (path_, msg))
    if not checkpath(path_, **kwargs):
        raise AssertionError('path=%r does not exist! %s' % (path_, msg))


assert_exists = assertpath


def pathsplit_full(path):
    """ splits all directories in path into a list """
    return path.replace('\\', '/').split('/')


def get_standard_exclude_dnames():
    return ['lib.linux-x86_64-2.7', 'dist', 'build', '_page', '_doc',
            'utool.egg-info', '.git']


def get_standard_include_patterns():
    return ['*.py', '*.pyx', '*.pxi', '*.cxx', '*.cpp', '*.hxx', '*.hpp', '*.c', '*.h', '*.vim', '*.cmake']


def matching_fpaths(dpath_list, include_patterns, exclude_dirs=[],
                    greater_exclude_dirs=[], exclude_patterns=[],
                    recursive=True):
    r"""
    walks dpath lists returning all directories that match the requested
    pattern.

    Args:
        dpath_list       (list):
        include_patterns (str):
        exclude_dirs     (None):
        recursive        (bool):

    References:
        # TODO: fix names and behavior of exclude_dirs and greater_exclude_dirs
        http://stackoverflow.com/questions/19859840/excluding-directories-in-os-walk

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> dpath_list = [dirname(dirname(ut.__file__))]
        >>> include_patterns = get_standard_include_patterns()
        >>> exclude_dirs = ['_page']
        >>> greater_exclude_dirs = get_standard_exclude_dnames()
        >>> recursive = True
        >>> fpath_gen = matching_fpaths(dpath_list, include_patterns, exclude_dirs,
        >>>                             greater_exclude_dirs, recursive)
        >>> result = list(fpath_gen)
        >>> print('\n'.join(result))
    """
    if isinstance(dpath_list, six.string_types):
        dpath_list = [dpath_list]
    for dpath in dpath_list:
        for root, dname_list, fname_list in os.walk(dpath):
            # Look at all subdirs
            subdirs = pathsplit_full(relpath(root, dpath))
            # HACK:
            if any([dir_ in greater_exclude_dirs for dir_ in subdirs]):
                continue
            # Look at one subdir
            if basename(root) in exclude_dirs:
                continue
            _match = fnmatch.fnmatch
            for name in fname_list:
                # yeild filepaths that are included
                if any(_match(name, pat) for pat in include_patterns):
                    # ... and not excluded
                    if not any(_match(name, pat) for pat in exclude_patterns):
                        fpath = join(root, name)
                        yield fpath
            if not recursive:
                break


def sed(regexpr, repl, force=False, recursive=False, dpath_list=None,
        fpath_list=None, verbose=None, include_patterns=None,
        exclude_patterns=[]):
    """
    Python implementation of sed. NOT FINISHED

    searches and replaces text in files

    Args:
        regexpr (str): regx patterns to find
        repl (str): text to replace
        force (bool):
        recursive (bool):
        dpath_list (list): directories to search (defaults to cwd)
    """
    #_grep(r, [repl], dpath_list=dpath_list, recursive=recursive)
    if include_patterns is None:
        include_patterns = ['*.py', '*.pyx', '*.pxi', '*.cxx', '*.cpp', '*.hxx', '*.hpp', '*.c', '*.h', '*.html', '*.tex']
    if dpath_list is None:
        dpath_list = [os.getcwd()]
    if verbose is None:
        verbose = ut.NOT_QUIET
    if fpath_list is None:
        greater_exclude_dirs = get_standard_exclude_dnames()
        exclude_dirs = []
        fpath_generator = matching_fpaths(
            dpath_list, include_patterns, exclude_dirs,
            greater_exclude_dirs=greater_exclude_dirs,
            recursive=recursive, exclude_patterns=exclude_patterns)
    else:
        fpath_generator = fpath_list
    if verbose:
        print('sed-ing %r' % (dpath_list,))
        print(' * regular expression : %r' % (regexpr,))
        print(' * replacement        : %r' % (repl,))
        print(' * include_patterns   : %r' % (include_patterns,))
        print(' * recursive: %r' % (recursive,))
        print(' * force: %r' % (force,))
        from utool import util_str
        print(' * fpath_list: %s' % (util_str.repr3(fpath_list),))
    regexpr = extend_regex(regexpr)
    #if '\x08' in regexpr:
    #    print('Remember \\x08 != \\b')
    #    print('subsituting for you for you')
    #    regexpr = regexpr.replace('\x08', '\\b')
    #    print(' * regular expression : %r' % (regexpr,))

    # Walk through each directory recursively
    num_changed = 0
    num_files_checked = 0
    fpaths_changed = []
    for fpath in fpath_generator:
        num_files_checked += 1
        changed_lines = sedfile(fpath, regexpr, repl, force, verbose=verbose)
        if changed_lines is not None:
            fpaths_changed.append(fpath)
            num_changed += len(changed_lines)
    import utool as ut
    print('num_files_checked = %r' % (num_files_checked,))
    print('fpaths_changed = %s' % (ut.repr3(sorted(fpaths_changed)),))
    print('total lines changed = %r' % (num_changed,))


def sedfile(fpath, regexpr, repl, force=False, verbose=True, veryverbose=False):
    """
    Executes sed on a specific file

    Args:
        fpath (str):  file path string
        regexpr (str):
        repl (str):
        force (bool): (default = False)
        verbose (bool):  verbosity flag(default = True)
        veryverbose (bool): (default = False)

    Returns:
        list: changed_lines

    CommandLine:
        python -m utool.util_path --exec-sedfile --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> fpath = ut.get_modpath(ut.util_path)
        >>> regexpr = 'sedfile'
        >>> repl = 'saidfile'
        >>> force = False
        >>> verbose = True
        >>> veryverbose = False
        >>> changed_lines = sedfile(fpath, regexpr, repl, force, verbose, veryverbose)
        >>> result = ('changed_lines = %s' % (ut.repr3(changed_lines),))
        >>> print(result)
    """
    # TODO: move to util_edit
    path, name = split(fpath)
    new_file_lines = []

    if veryverbose:
        print('[sedfile] fpath=%r' % fpath)
        print('[sedfile] regexpr=%r' % regexpr)
        print('[sedfile] repl=%r' % repl)
        print('[sedfile] force=%r' % force)

    import utool as ut
    file_lines = ut.readfrom(fpath, aslines=True, verbose=False)
    # with open(fpath, 'r') as file:
    #     import utool
    #     with utool.embed_on_exception_context:
    #         file_lines = file.readlines()
    # Search each line for the desired regexpr
    new_file_lines = [re.sub(regexpr, repl, line) for line in file_lines]

    changed_lines = [(newline, line)
                     for newline, line in zip(new_file_lines, file_lines)
                     if  newline != line]
    n_changed = len(changed_lines)
    if n_changed > 0:
        try:
            rel_fpath = relpath(fpath, os.getcwd())
        except ValueError:
            # Can happen on windows
            rel_fpath = fpath

        print(' * %s changed %d lines in %r ' %
              (['(dry-run)', '(real-run)'][force], n_changed, rel_fpath))
        print(' * --------------------')
        import utool as ut
        new_file_lines = ut.lmap(ut.ensure_unicode, new_file_lines)
        new_file = ''.join(new_file_lines)
        #print(new_file.replace('\n','\n))
        if verbose:
            if True:
                import utool as ut
                old_file = ut.ensure_unicode(
                    ''.join(ut.lmap(ut.ensure_unicode, file_lines)))
                ut.print_difftext(old_file, new_file)
            else:
                changed_new, changed_old = zip(*changed_lines)
                prefixold = ' * old (%d, %r):  \n | ' % (n_changed, name)
                prefixnew = ' * new (%d, %r):  \n | ' % (n_changed, name)
                print(prefixold + (' | '.join(changed_old)).strip('\n'))
                print(' * ____________________')
                print(prefixnew + (' | '.join(changed_new)).strip('\n'))
                print(' * --------------------')
                print(' * =====================================================')
        # Write back to file
        if force:
            print(' ! WRITING CHANGES')
            ut.writeto(fpath, new_file)
            # with open(fpath, 'w') as file:
            #     file.write(new_file.encode('utf8'))
        else:
            print(' dry run')
        return changed_lines
    #elif verbose:
    #    print('Nothing changed')
    return None


#@profile
def grepfile(fpath, regexpr_list, reflags=0, cache=None):
    """
    grepfile - greps a specific file

    Args:
        fpath (str):
        regexpr_list (list or str): pattern or list of patterns

    Returns:
        tuple (list, list): list of lines and list of line numbers

    CommandLine:
        python -m utool.util_path --exec-grepfile

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> fpath = ut.get_modpath(ut.util_path)
        >>> regexpr_list = ['foundthisline', '__future__']
        >>> cache = None
        >>> reflags = 0
        >>> found_lines, found_lxs = ut.grepfile(fpath, regexpr_list)
        >>> result = ut.repr3({'found_lines': found_lines, 'found_lxs': found_lxs})
        >>> print(result)
        >>> assert 7 in found_lxs
        >>> others = ut.take_complement(found_lxs, [found_lxs.index(7)])
        >>> assert others[0] == others[1]
    """
    found_lines = []
    found_lxs = []
    # Ensure a list
    islist = isinstance(regexpr_list, (list, tuple))
    islist2 = isinstance(reflags, (list, tuple))
    regexpr_list_ = regexpr_list if islist else [regexpr_list]
    reflags_list = reflags if islist2 else [reflags] * len(regexpr_list_)
    re_list = [re.compile(pat, flags=_flags)
               for pat, _flags in  zip(regexpr_list_, reflags_list)]
    #print('regexpr_list_ = %r' % (regexpr_list_,))
    #print('re_list = %r' % (re_list,))

    import numpy as np
    # Open file and search lines or use cache
    if cache is None or fpath not in cache:
        #with open(fpath, 'r') as file_:
        #    lines = list(file_.readlines())
        from utool import util_io
        lines = util_io.read_from(fpath, aslines=True, verbose=False)
        #import utool as ut
        #cumsum = ut.cumsum(map(len, lines))
        cumsum = np.cumsum(list(map(len, lines)))
        text = ''.join(lines)
        if cache is not None:
            cache[fpath] = (cumsum, text, lines)
    else:
        (cumsum, text, lines) = cache[fpath]

    # Search each line for each pattern
    old_method = False
    if old_method:
        for lx, line in enumerate(lines):
            #for regexpr_ in regexpr_list_:
            match_objects = [re_.search(line) for re_ in re_list]
            for match in match_objects:
                if match is not None:
                    found_lines.append(line)
                    found_lxs.append(lx)
    else:
        for re_ in re_list:
            # FIXME: multiline mode doesnt work
            for match_object in re_.finditer(text):
                #print('match_object = %r' % (match_object,))
                lxs = np.where(match_object.start() < cumsum)[0][0:1]
                if len(lxs) == 1:
                    lx = lxs[0]
                    if lx > 0:
                        line_start = cumsum[lx - 1]
                    else:
                        line_start = 0
                    line_end = cumsum[lx]
                    line = text[line_start:line_end]
                    found_lines.append(line)
                    found_lxs.append(lx)
                #[match_object.start > x for x in cumsum]
    return found_lines, found_lxs


def greplines(lines, regexpr_list, reflags=0):
    """
    grepfile - greps a specific file

    TODO: move to util_str, rework to be core of grepfile
    """
    found_lines = []
    found_lxs = []
    # Ensure a list
    islist = isinstance(regexpr_list, (list, tuple))
    islist2 = isinstance(reflags, (list, tuple))
    regexpr_list_ = regexpr_list if islist else [regexpr_list]
    reflags_list = reflags if islist2 else [reflags] * len(regexpr_list_)
    re_list = [re.compile(pat, flags=_flags)
               for pat, _flags in  zip(regexpr_list_, reflags_list)]
    #print('regexpr_list_ = %r' % (regexpr_list_,))
    #print('re_list = %r' % (re_list,))

    import numpy as np
    #import utool as ut
    #cumsum = ut.cumsum(map(len, lines))
    cumsum = np.cumsum(list(map(len, lines)))
    text = ''.join(lines)

    # Search each line for each pattern
    for re_ in re_list:
        # FIXME: multiline mode doesnt work
        for match_object in re_.finditer(text):
            lxs = np.where(match_object.start() < cumsum)[0][0:1]
            if len(lxs) == 1:
                lx = lxs[0]
                if lx > 0:
                    line_start = cumsum[lx - 1]
                else:
                    line_start = 0
                line_end = cumsum[lx]
                line = text[line_start:line_end]
                found_lines.append(line)
                found_lxs.append(lx)
    return found_lines, found_lxs


def testgrep():
    """
    utprof.py -m utool.util_path --exec-testgrep

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> #dpath_list = [ut.truepath('~/code/utool/utool')]
        >>> dpath_list = [ut.truepath(dirname(ut.__file__))]
        >>> include_patterns = ['*.py']
        >>> exclude_dirs = []
        >>> regex_list = ['grepfile']
        >>> verbose = True
        >>> recursive = True
        >>> result = ut.grep(regex_list, recursive, dpath_list, include_patterns,
        >>>                  exclude_dirs)
        >>> (found_fpath_list, found_lines_list, found_lxs_list) = result
        >>> assert 'util_path.py' in list(map(basename, found_fpath_list))
    """
    pass


# FIXME: util_test can't find the function if profile is enabled
#@profile
def grep(regex_list, recursive=True, dpath_list=None, include_patterns=None,
         exclude_dirs=[], greater_exclude_dirs=None, inverse=False,
         exclude_patterns=[], verbose=VERBOSE, fpath_list=None, reflags=0,
         cache=None):
    r"""
    greps for patterns
    Python implementation of grep. NOT FINISHED

    Args:
        regex_list (str or list): one or more patterns to find
        recursive (bool):
        dpath_list (list): directories to search (defaults to cwd)
        include_patterns (list) : defaults to standard file extensions

    Returns:
        (list, list, list): (found_fpaths, found_lines_list, found_lxs_list)

    CommandLine:
        python -m utool.util_path --test-grep
        utprof.py -m utool.util_path --exec-grep
        utprof.py utool/util_path.py --exec-grep

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> #dpath_list = [ut.truepath('~/code/utool/utool')]
        >>> dpath_list = [ut.truepath(dirname(ut.__file__))]
        >>> include_patterns = ['*.py']
        >>> exclude_dirs = []
        >>> regex_list = ['grepfile']
        >>> verbose = True
        >>> recursive = True
        >>> result = ut.grep(regex_list, recursive, dpath_list, include_patterns,
        >>>                  exclude_dirs)
        >>> (found_fpath_list, found_lines_list, found_lxs_list) = result
        >>> assert 'util_path.py' in list(map(basename, found_fpath_list))
    """
    from utool import util_regex
    # from utool import util_str
    from utool import util_list
    if include_patterns is None:
        include_patterns =  ['*']
        # include_patterns = get_standard_include_patterns()
    if greater_exclude_dirs is None:
        greater_exclude_dirs = []
        # greater_exclude_dirs =  get_standard_exclude_dnames()
    # ensure list input
    if isinstance(include_patterns, six.string_types):
        include_patterns = [include_patterns]
    if dpath_list is None:
        dpath_list = [os.getcwd()]
    if verbose:
        recursive_stat_str = ['flat', 'recursive'][recursive]
        print('[util_path] Greping (%s) %r for %r' % (recursive_stat_str,
                                                      dpath_list, regex_list))
        print('[util_path] regex_list = %s' % (regex_list))
    if isinstance(regex_list, six.string_types):
        regex_list = [regex_list]
    found_fpath_list = []
    found_lines_list = []
    found_lxs_list = []
    # Walk through each directory recursively
    if fpath_list is None:
        fpath_generator = matching_fpaths(
            dpath_list=dpath_list, include_patterns=include_patterns,
            exclude_dirs=exclude_dirs,
            greater_exclude_dirs=greater_exclude_dirs,
            exclude_patterns=exclude_patterns, recursive=recursive)
    else:
        fpath_generator = fpath_list
    #     from utool import util_regex
    #     extended_regex_list, reflags = util_regex.extend_regex3(regex_list, reflags)
    #     if verbose:
    #         print('extended_regex_list = %r' % (extended_regex_list,))
    #         print('reflags = %r' % (reflags,))
    _exprs_flags = [util_regex.extend_regex2(expr, reflags)
                    for expr in regex_list]
    extended_regex_list = util_list.take_column(_exprs_flags, 0)
    reflags_list = util_list.take_column(_exprs_flags, 1)
    # HACK
    reflags = reflags_list[0]

    # For each matching filepath
    for fpath in fpath_generator:
        # For each search pattern
        found_lines, found_lxs = grepfile(fpath, extended_regex_list,
                                          reflags_list, cache=cache)
        if inverse:
            if len(found_lines) == 0:
                # Append files that the pattern was not found in
                found_fpath_list.append(fpath)
                found_lines_list.append([])
                found_lxs_list.append([])
        elif len(found_lines) > 0:
            found_fpath_list.append(fpath)  # regular matching
            found_lines_list.append(found_lines)
            found_lxs_list.append(found_lxs)

    grep_result = (found_fpath_list, found_lines_list, found_lxs_list)
    if verbose:
        print('==========')
        print('==========')
        print('[util_path] found matches in %d files' %
              len(found_fpath_list))
        print(make_grep_resultstr(grep_result, extended_regex_list, reflags))
        # print('[util_path] found matches in %d files' % len(found_fpath_list))

        # pat = util_regex.regex_or(extended_regex_list)

        # for fpath, found, lxs in zip(found_fpath_list, found_lines_list,
        #                              found_lxs_list):
        #     if len(found) > 0:
        #         print('----------------------')
        #         print('Found %d line(s) in %r: ' % (len(found), fpath))
        #         name = split(fpath)[1]
        #         max_line = len(lxs)
        #         ndigits = str(len(str(max_line)))
        #         fmt_str = '%s : %' + ndigits + 'd |%s'
        #         for (lx, line) in zip(lxs, found):
        #             # hack
        #             colored_line = util_str.highlight_regex(
        #                 line.rstrip('\n'), pat, reflags=reflags)
        #             print(fmt_str % (name, lx, colored_line))

        #print('[util_path] found matches in %d files' % len(found_fpath_list))

    return grep_result


def make_grep_resultstr(grep_result, extended_regex_list, reflags, colored=True):
    from utool import util_regex
    from utool import util_str
    msg_list = []
    print_ = msg_list.append
    pat = util_regex.regex_or(extended_regex_list)
    found_fpath_list, found_lines_list, found_lxs_list = grep_result
    for fpath, found, lxs in zip(found_fpath_list, found_lines_list,
                                 found_lxs_list):
        if len(found) > 0:
            print_('----------------------')
            print_('Found %d line(s) in %r: ' % (len(found), fpath))
            name = split(fpath)[1]
            max_line = len(lxs)
            ndigits = str(len(str(max_line)))
            fmt_str = '%s : %' + ndigits + 'd |%s'
            for (lx, line) in zip(lxs, found):
                # hack
                colored_line = line.rstrip('\n')
                if colored:
                    colored_line = util_str.highlight_regex(colored_line, pat, reflags=reflags)
                print_(fmt_str % (name, lx, colored_line))
    return '\n'.join(msg_list)


def get_win32_short_path_name(long_name):
    """
    Gets the short path name of a given long path.

    References:
        http://stackoverflow.com/a/23598461/200291
        http://stackoverflow.com/questions/23598289/get-win-short-fname-python

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut  # NOQA
        >>> # build test data
        >>> #long_name = unicode(normpath(ut.get_resource_dir()))
        >>> long_name = unicode(r'C:/Program Files (x86)')
        >>> #long_name = unicode(r'C:/Python27')
        #unicode(normpath(ut.get_resource_dir()))
        >>> # execute function
        >>> result = get_win32_short_path_name(long_name)
        >>> # verify results
        >>> print(result)
        C:/PROGRA~2
    """
    import ctypes
    from ctypes import wintypes
    _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    _GetShortPathNameW.restype = wintypes.DWORD
    output_buf_size = 0
    while True:
        output_buf = ctypes.create_unicode_buffer(output_buf_size)
        needed = _GetShortPathNameW(long_name, output_buf, output_buf_size)
        if output_buf_size >= needed:
            short_name = output_buf.value
            break
        else:
            output_buf_size = needed
    return short_name


def expand_win32_shortname(path1):
    try:
        #try:
        #    import win32file
        #    path2 = win32file.GetLongPathName(path1)
        #except ImportError:
        import ctypes
        #import win32file
        if six.PY2:
            path1 = six.text_type(path1)
        else:
            path1 = str(path1)
        buflen = 260  # max size
        buf = ctypes.create_unicode_buffer(buflen)
        ctypes.windll.kernel32.GetLongPathNameW(path1, buf, buflen)
        # If the path doesnt exist windows doesnt return anything
        path2 = buf.value if len(buf.value) > 0 else path1
    except Exception as ex:
        print(ex)
        util_dbg.printex(ex, 'cannot fix win32 shortcut', keys=['path1', 'path2'])
        path2 = path1
        #raise
    return path2


def platform_path(path):
    r"""
    Returns platform specific path for pyinstaller usage

    Args:
        path (str):

    Returns:
        str: path2

    CommandLine:
        python -m utool.util_path --test-platform_path

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> # FIXME: find examples of the wird paths this fixes (mostly on win32 i think)
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> path = 'some/odd/../weird/path'
        >>> path2 = platform_path(path)
        >>> result = str(path2)
        >>> if ut.WIN32:
        ...     ut.assert_eq(path2, r'some\weird\path')
        ... else:
        ...     ut.assert_eq(path2, r'some/weird/path')

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut    # NOQA
        >>> if ut.WIN32:
        ...     path = 'C:/PROGRA~2'
        ...     path2 = platform_path(path)
        ...     assert path2 == u'..\\..\\..\\..\\Program Files (x86)'
    """
    try:
        if path == '':
            raise ValueError('path cannot be the empty string')
        # get path relative to cwd
        path1 = truepath_relative(path)
        if sys.platform.startswith('win32'):
            path2 = expand_win32_shortname(path1)
        else:
            path2 = path1
    except Exception as ex:
        util_dbg.printex(ex, keys=['path', 'path1', 'path2'])
        raise
    return path2


def existing_subpath(root_path, valid_subpaths, tiebreaker='first',
                     verbose=VERYVERBOSE):
    """
    Returns join(root_path, subpath) where subpath in valid_subpath ane
    exists(subpath)
    """
    # Find the oxford_style groundtruth directory
    for subpath in valid_subpaths:
        path  = join(root_path, subpath)
        if checkpath(path, verbose=verbose):
            if tiebreaker == 'first':
                return path
    raise AssertionError('none of the following subpaths exist: %r' %
                         (valid_subpaths,))


def existing_commonprefix(paths):
    return longest_existing_path(commonprefix(paths))


#def find_executable(exename):
#    import utool as ut
#    search_dpaths = ut.get_install_dirs()
#    pass


def search_in_dirs(fname, search_dpaths=[], shortcircuit=True,
                   return_tried=False, strict=False):
    r"""
    search_in_dirs

    Args:
        fname (str):  file name
        search_dpaths (list):
        shortcircuit (bool):
        return_tried (bool): return tried paths
        strict (bool): (default = False)

    Returns:
        fpath: None

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> fname = r'Inno Setup 5\ISCC.exe'
        >>> search_dpaths = ut.get_install_dirs()
        >>> shortcircuit = True
        >>> fpath = ut.search_in_dirs(fname, search_dpaths, shortcircuit)
        >>> print(fpath)
    """
    fpath_list = []
    tried_list = []
    for dpath in search_dpaths:
        fpath = join(dpath, fname)
        if return_tried:
            tried_list.append(fpath)
        if exists(fpath):
            if shortcircuit:
                if return_tried:
                    return fpath, tried_list
                return fpath
            else:
                fpath_list.append(fpath)
    if strict and len(fpath_list) == 0:
        msg = ('Cannot find: fname=%r\n'  % (fname,))
        if return_tried:
            msg += 'Tried: \n    ' + '\n    '.join(tried_list)
        raise Exception(msg)

    if shortcircuit:
        if return_tried:
            return None, tried_list
        return None
    else:
        if return_tried:
            return fpath_list, tried_list
        return fpath_list


def find_lib_fpath(libname, root_dir, recurse_down=True, verbose=False, debug=False):
    """ Search for the library """

    def get_lib_fname_list(libname):
        """
        input <libname>: library name (e.g. 'hesaff', not 'libhesaff')
        returns <libnames>: list of plausible library file names
        """
        if sys.platform.startswith('win32'):
            libnames = ['lib' + libname + '.dll', libname + '.dll']
        elif sys.platform.startswith('darwin'):
            libnames = ['lib' + libname + '.dylib']
        elif sys.platform.startswith('linux'):
            libnames = ['lib' + libname + '.so']
        else:
            raise Exception('Unknown operating system: %s' % sys.platform)
        return libnames

    def get_lib_dpath_list(root_dir):
        """
        input <root_dir>: deepest directory to look for a library (dll, so, dylib)
        returns <libnames>: list of plausible directories to look.
        """
        'returns possible lib locations'
        get_lib_dpath_list = [root_dir,
                              join(root_dir, 'lib'),
                              join(root_dir, 'build'),
                              join(root_dir, 'build', 'lib')]
        return get_lib_dpath_list

    lib_fname_list = get_lib_fname_list(libname)
    tried_fpaths = []
    while root_dir is not None:
        for lib_fname in lib_fname_list:
            for lib_dpath in get_lib_dpath_list(root_dir):
                lib_fpath = normpath(join(lib_dpath, lib_fname))
                if exists(lib_fpath):
                    if verbose:
                        print('\n[c] Checked: '.join(tried_fpaths))
                    if debug:
                        print('using: %r' % lib_fpath)
                    return lib_fpath
                else:
                    # Remember which candiate library fpaths did not exist
                    tried_fpaths.append(lib_fpath)
            _new_root = dirname(root_dir)
            if _new_root == root_dir:
                root_dir = None
                break
            else:
                root_dir = _new_root
        if not recurse_down:
            break

    msg = ('\n[C!] load_clib(libname=%r root_dir=%r, recurse_down=%r, verbose=%r)' %
           (libname, root_dir, recurse_down, verbose) +
           '\n[c!] Cannot FIND dynamic library')
    print(msg)
    print('\n[c!] Checked: '.join(tried_fpaths))
    raise ImportError(msg)


def ensure_mingw_drive(win32_path):
    r""" replaces windows drives with mingw style drives

    Args:
        win32_path (str):

    CommandLine:
        python -m utool.util_path --test-ensure_mingw_drive

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> win32_path = r'C:/Program Files/Foobar'
        >>> result = ensure_mingw_drive(win32_path)
        >>> print(result)
        /c/Program Files/Foobar
    """
    win32_drive, _path = splitdrive(win32_path)
    mingw_drive = '/' + win32_drive[:-1].lower()
    mingw_path = mingw_drive + _path
    return mingw_path


class ChdirContext(object):
    """
    References http://www.astropython.org/snippet/2009/10/chdir-context-manager
    """
    def __init__(self, dpath=None, stay=False, verbose=None):
        if verbose is None:
            import utool as ut
            verbose = ut.NOT_QUIET
        self.verbose = verbose
        self.stay = stay
        self.dpath = dpath
        self.curdir = os.getcwd()

    def __enter__(self):
        if self.dpath is not None:
            if self.verbose:
                print('[path.push] Change directory to %r' % (self.dpath,))
            os.chdir(self.dpath)
        return self

    def __exit__(self, type_, value, trace):
        if not self.stay:
            if self.verbose:
                print('[path.pop] Change directory to %r' % (self.curdir,))
            os.chdir(self.curdir)
        if trace is not None:
            if self.verbose or VERBOSE:
                print('[util_path] Error in chdir context manager!: ' + str(value))
            return False  # return a falsey value on error


def ancestor_paths(start=None, limit={}):
    """
    All paths above you
    """
    import utool as ut
    limit = ut.ensure_iterable(limit)
    limit = {expanduser(p) for p in limit}.union(set(limit))
    if start is None:
        start = os.getcwd()
    path = start
    prev = None
    while path != prev and prev not in limit:
        yield path
        prev = path
        path = dirname(path)


def search_candidate_paths(candidate_path_list, candidate_name_list=None,
                           priority_paths=None, required_subpaths=[],
                           verbose=None):
    """
    searches for existing paths that meed a requirement

    Args:
        candidate_path_list (list): list of paths to check. If
            candidate_name_list is specified this is the dpath list instead
        candidate_name_list (list): specifies several names to check
            (default = None)
        priority_paths (None): specifies paths to check first.
            Ignore candidate_name_list (default = None)
        required_subpaths (list): specified required directory structure
            (default = [])
        verbose (bool):  verbosity flag(default = True)

    Returns:
        str: return_path

    CommandLine:
        python -m utool.util_path --test-search_candidate_paths

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> candidate_path_list = [ut.truepath('~/RPI/code/utool'),
        >>>                        ut.truepath('~/code/utool')]
        >>> candidate_name_list = None
        >>> required_subpaths = []
        >>> verbose = True
        >>> priority_paths = None
        >>> return_path = search_candidate_paths(candidate_path_list,
        >>>                                      candidate_name_list,
        >>>                                      priority_paths, required_subpaths,
        >>>                                      verbose)
        >>> result = ('return_path = %s' % (str(return_path),))
        >>> print(result)
    """
    import utool as ut
    if verbose is None:
        verbose = 0 if QUIET else 1

    if verbose >= 1:
        print('[search_candidate_paths] Searching for candidate paths')

    if candidate_name_list is not None:
        candidate_path_list_ = [join(dpath, fname) for dpath, fname in
                                itertools.product(candidate_path_list,
                                                  candidate_name_list)]
    else:
        candidate_path_list_ = candidate_path_list

    if priority_paths is not None:
        candidate_path_list_ = priority_paths + candidate_path_list_

    return_path = None
    for path in candidate_path_list_:
        if path is not None and exists(path):
            if verbose >= 2:
                print('[search_candidate_paths] Found candidate directory %r' % (path,))
                print('[search_candidate_paths] ... checking for approprate structure')
            # tomcat directory exists. Make sure it also contains a webapps dir
            subpath_list = [join(path, subpath) for subpath in required_subpaths]
            if all(ut.checkpath(path_, verbose=verbose) for path_ in subpath_list):
                return_path = path
                if verbose >= 2:
                    print('[search_candidate_paths] Found acceptable path')
                return return_path
                break
    if verbose >= 1:
        print('[search_candidate_paths] Failed to find acceptable path')
    return return_path


def sanitize_filename(fname):
    from utool import util_str
    print('fname = %r' % (fname,))
    invalid_sep_chars = ['/', '\\']
    ugly_space_chars = [' ', '\t', '\n', '\r']
    fname = util_str.multi_replace(fname, invalid_sep_chars, '-')
    fname = util_str.multi_replace(fname, ugly_space_chars, '')
    return fname


def win_shortcut(source, link_name):
    """
    Attempt to create windows shortcut
    TODO: TEST / FIXME

    References:
        http://stackoverflow.com/questions/1447575/symlinks-on-windows
    """
    if True:
        import ctypes
        kdll = ctypes.windll.LoadLibrary("kernel32.dll")
        code = 1 if isdir(source) else 0
        kdll.CreateSymbolicLinkA(source, link_name, code)
    else:
        import ctypes
        csl = ctypes.windll.kernel32.CreateSymbolicLinkW
        csl.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        csl.restype = ctypes.c_ubyte
        flags = 1 if isdir(source) else 0
        retval = csl(link_name, source, flags)
        if retval == 0:
            #warn_msg = '[util_path] Unable to create symbolic link on windows.'
            #print(warn_msg)
            #warnings.warn(warn_msg, category=UserWarning)
            if checkpath(link_name):
                return True
            raise ctypes.WinError()


def symlink(real_path, link_path, overwrite=False, on_error='raise',
            verbose=2):
    """
    Attempt to create a symbolic link.

    TODO:
        Can this be fixed on windows?

    Args:
        path (str): path to real file or directory
        link_path (str): path to desired location for symlink
        overwrite (bool): overwrite existing symlinks (default = False)
        on_error (str): strategy for dealing with errors.
            raise or ignore
        verbose (int):  verbosity level (default=2)

    Returns:
        str: link path

    CommandLine:
        python -m utool.util_path symlink

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> dpath = ut.get_app_resource_dir('utool')
        >>> real_path = join(dpath, 'real_file.txt')
        >>> link_path = join(dpath, 'link_file.txt')
        >>> ut.emap(ut.delete, [real_path, link_path], verbose=0)
        >>> ut.writeto(real_path, 'foo')
        >>> result = symlink(real_path, link_path)
        >>> assert ut.readfrom(result) == 'foo'
        >>> ut.emap(ut.delete, [real_path, link_path], verbose=0)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> import ubelt as ub
        >>> import six
        >>> if ub.WIN32 and six.PY2:
        >>>     import pytest
        >>>     pytest.skip('does not work on win32 in 27. Not sure why')
        >>> real_dpath = ut.get_app_resource_dir('utool', 'real_dpath')
        >>> link_dpath = ut.augpath(real_dpath, newfname='link_dpath')
        >>> real_path = join(real_dpath, 'afile.txt')
        >>> link_path = join(link_dpath, 'afile.txt')
        >>> ut.emap(ut.delete, [real_path, link_path], verbose=0)
        >>> ut.ensuredir(real_dpath)
        >>> ut.writeto(real_path, 'foo')
        >>> result = symlink(real_dpath, link_dpath)
        >>> assert ut.readfrom(link_path) == 'foo'
        >>> ut.delete(link_dpath, verbose=2)
        >>> assert ut.checkpath(real_path, verbose=2)
        >>> ut.delete(real_dpath, verbose=2)
        >>> assert not ut.checkpath(real_path, verbose=2)
    """
    if 1:
        # Use ubelt implementation
        import ubelt as ub
        try:
            return ub.symlink(real_path, link_path, overwrite=overwrite,
                              verbose=verbose)
        except Exception:
            if on_error == 'ignore':
                return False
            else:
                raise

    path = normpath(real_path)
    link = normpath(link_path)
    if verbose:
        print('[util_path] Creating symlink: path={} link={}'.format(path, link))
    if os.path.islink(link):
        if verbose:
            print('[util_path] symlink already exists')
        os_readlink = getattr(os, "readlink", None)
        if callable(os_readlink):
            if os_readlink(link) == path:
                if verbose > 1:
                    print('[path] ... and points to the right place')
                return link
        else:
            print('[util_path] Warning, symlinks are not implemented on windows')
        if verbose > 1:
            print('[util_path] ... but it points somewhere else')
        if overwrite:
            delete(link, verbose > 1)
        elif on_error == 'ignore':
            return False
    try:
        os_symlink = getattr(os, "symlink", None)
        if callable(os_symlink):
            os_symlink(path, link)
        else:
            win_shortcut(path, link)
    except Exception as ex:
        import utool as ut
        checkpath(link, verbose=True)
        checkpath(path, verbose=True)
        do_raise = (on_error == 'raise')
        ut.printex(ex, '[util_path] error making symlink',
                   iswarning=not do_raise)
        if do_raise:
            raise
    return link


def remove_broken_links(dpath, verbose=True):
    """
    Removes all broken links in a directory

    Args:
        dpath (str):  directory path

    Returns:
        int: num removed

    References:
        http://stackoverflow.com/questions/20794/find-broken-symlinks-with-python

    CommandLine:
        python -m utool remove_broken_links:0

    Example:
        >>> # DISABLE_DOCTEST
        >>> # SCRIPT
        >>> from utool.util_path import *  # NOQA
        >>> remove_broken_links('.')

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> import ubelt as ub
        >>> if ub.WIN32:
        ...     import pytest
        ...     pytest.skip('does not work on win32')
        >>> dpath = ut.ensure_app_resource_dir('utool', 'path_tests')
        >>> ut.delete(dpath)
        >>> test_dpath = ut.ensuredir(join(dpath, 'testdpath'))
        >>> test_fpath = ut.ensurefile(join(dpath, 'testfpath.txt'))
        >>> flink1 = ut.symlink(test_fpath, join(dpath, 'flink1'))
        >>> dlink1 = ut.symlink(test_fpath, join(dpath, 'dlink1'))
        >>> assert len(ut.ls(dpath)) == 4
        >>> ut.delete(test_fpath)
        >>> assert len(ut.ls(dpath)) == 3
        >>> remove_broken_links(dpath)
        >>> ut.delete(test_dpath)
        >>> remove_broken_links(dpath)
        >>> assert len(ut.ls(dpath)) == 0
    """
    fname_list = [join(dpath, fname) for fname in os.listdir(dpath)]
    broken_links = list(filterfalse(exists, filter(islink, fname_list)))
    num_broken = len(broken_links)
    if verbose:
        if verbose > 1 or num_broken > 0:
            print('[util_path] Removing %d broken links in %r' % (num_broken, dpath,))
    for link in broken_links:
        os.unlink(link)
    return num_broken


def non_existing_path(path_, dpath=None, offset=0, suffix=None,
                            force_fmt=False):
    r"""
    Searches for and finds a path garuenteed to not exist.

    Args:
        path_ (str):  path string. If may include a "%" formatstr.
        dpath (str):  directory path(default = None)
        offset (int): (default = 0)
        suffix (None): (default = None)

    Returns:
        str: path_ - path string

    CommandLine:
        python -m utool.util_path non_existing_path

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> base = ut.ensure_app_resource_dir('utool', 'tmp')
        >>> ut.touch(base + '/tmp.txt')
        >>> ut.touch(base + '/tmp0.txt')
        >>> ut.delete(base + '/tmp1.txt')
        >>> path_ = base + '/tmp.txt'
        >>> newpath = ut.non_existing_path(path_)
        >>> assert basename(newpath) == 'tmp1.txt'

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_path import *  # NOQA
        >>> import utool as ut
        >>> base = ut.ensure_app_resource_dir('utool', 'tmp')
        >>> ut.ensurepath(base + '/dir_old')
        >>> ut.ensurepath(base + '/dir_old0')
        >>> ut.ensurepath(base + '/dir_old1')
        >>> ut.delete(base + '/dir_old2')
        >>> path_ = base + '/dir'
        >>> suffix = '_old'
        >>> newpath = ut.non_existing_path(path_, suffix=suffix)
        >>> ut.assert_eq(basename(newpath), 'dir_old2')
    """
    import utool as ut
    from os.path import basename, dirname

    if dpath is None:
        dpath = dirname(path_)
    base_fmtstr = basename(path_)
    if suffix is not None:
        base_fmtstr = ut.augpath(base_fmtstr, suffix)

    if '%' not in base_fmtstr:
        if not force_fmt:
            # If we have don't have to format,
            # then try to use the first choice
            first_choice = join(dpath, base_fmtstr)
            if not exists(first_choice):
                return first_choice
        # otherwise we ensure we can format and we continue
        base_fmtstr = ut.augpath(base_fmtstr, '%d')

    dname_list = ut.glob(dpath, pattern='*', recursive=False, with_files=True,
                         with_dirs=True)
    conflict_set = set(basename(dname) for dname in dname_list)

    newname = ut.get_nonconflicting_string(base_fmtstr, conflict_set,
                                           offset=offset)
    newpath = join(dpath, newname)
    return newpath


get_nonconflicting_path = non_existing_path


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_path; utool.doctest_funcs(utool.util_path)"
        python -m utool.util_path
        python -m utool.util_path --allexamples
        python -m utool.util_path --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
