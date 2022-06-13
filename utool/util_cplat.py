# -*- coding: utf-8 -*-
"""
cross platform utilities
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pipes
import six
import sys
import platform
import subprocess
import shlex
from os.path import exists, normpath, basename, dirname, join, expanduser
from utool import util_inject
from utool._internal import meta_util_cplat
from utool._internal.meta_util_path import unixpath, truepath
from six.moves import zip
print, rrr, profile = util_inject.inject2(__name__)
print_ = util_inject.make_module_write_func(__name__)

try:
    import pathlib
    HAVE_PATHLIB = True
except ImportError:
    HAVE_PATHLIB = False

COMPUTER_NAME = platform.node()

OS_TYPE = meta_util_cplat.OS_TYPE
WIN32  = meta_util_cplat.WIN32
LINUX  = meta_util_cplat.LINUX
DARWIN = meta_util_cplat.DARWIN
UNIX = not WIN32


# class Win32Err(object):
#     """
#     References:
#         https://msdn.microsoft.com/en-us/library/cc231199.aspx
#         https://docs.python.org/3/library/os.html#os.EX_OK
#     """
#     ERROR_SUCCESS = 0x0
#     ERROR_INVALID_FUNCTION = 0x1
#     ERROR_FATAL_APP_EXIT = 0x2C9

# Define standarized exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

LIB_EXT_LIST = ['.so', '.dll', '.dylib', '.pyd']

LIB_DICT = {
    'win32': '.dll',
    'linux': '.so',
    'darwin': '.dylib',
}


PYLIB_DICT = {
    'win32': '.pyd',
    'linux': '.so',
    'darwin': '.dylib',
}


def get_plat_specifier():
    """
    Standard platform specifier used by distutils
    """
    import setuptools  # NOQA
    import distutils
    plat_name = distutils.util.get_platform()
    plat_specifier = ".%s-%s" % (plat_name, sys.version[0:3])
    if hasattr(sys, 'gettotalrefcount'):
        plat_specifier += '-pydebug'
    return plat_specifier


def in_pyinstaller_package():
    """
    References:
        http://stackoverflow.com/questions/22472124/what-is-sys-meipass-in-python
        http://stackoverflow.com/questions/7674790/bundling-data-files-with-pyinstaller-onefile
    """
    return hasattr(sys, '_MEIPASS')


def get_system_python_library():
    """
    FIXME; hacky way of finding python library. Not cross platform yet.
    """
    import os
    import utool as ut
    from os.path import basename, realpath
    pyname = basename(realpath(sys.executable))
    ld_library_path = os.environ['LD_LIBRARY_PATH']
    libdirs = [x for x in ld_library_path.split(os.pathsep) if x] + ['/usr/lib']
    libfiles = ut.flatten([ut.glob(d, '*' + ut.get_lib_ext(), recursive=True) for d in libdirs])
    python_libs = [realpath(f) for f in libfiles if 'lib' + pyname in basename(f)]
    python_libs = ut.unique_ordered(python_libs)
    assert len(python_libs) == 1, str(python_libs)
    return python_libs[0]


def get_free_diskbytes(dir_):
    r"""
    Args:
        dir_ (str):

    Returns:
        int: bytes_ folder/drive free space (in bytes)

    References::
        http://stackoverflow.com/questions/51658/cross-platform-space-remaining-on-volume-using-python
        http://linux.die.net/man/2/statvfs

    CommandLine:
        python -m utool.util_cplat --exec-get_free_diskbytes
        python -m utool.util_cplat --exec-get_free_diskbytes --dir /media/raid
        python -m utool.util_cplat --exec-get_free_diskbytes --dir E:

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> import utool as ut
        >>> dir_ = ut.get_argval('--dir', type_=str, default=ut.truepath('~'))
        >>> bytes_ = get_free_diskbytes(dir_)
        >>> result = ('bytes_ = %s' % (str(bytes_),))
        >>> print(result)
        >>> print('Unused space in %r = %r' % (dir_, ut.byte_str2(bytes_)))
        >>> print('Total space in %r = %r' % (dir_, ut.byte_str2(get_total_diskbytes(dir_))))
    """
    if WIN32:
        import ctypes
        free_bytes = ctypes.c_ulonglong(0)
        outvar = ctypes.pointer(free_bytes)
        dir_ptr = ctypes.c_wchar_p(dir_)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(dir_ptr, None, None, outvar)
        bytes_ = free_bytes.value
        return bytes_
    else:
        st = os.statvfs(dir_)
        # blocks avaiable * block size
        bytes_ = st.f_bavail * st.f_frsize
        #bytes_ = st.f_bfree * st.f_frsize  # includes root only space
        return bytes_


def get_total_diskbytes(dir_):
    if WIN32:
        import ctypes
        total_bytes = ctypes.c_ulonglong(0)
        outvar = ctypes.pointer(total_bytes)
        dir_ptr = ctypes.c_wchar_p(dir_)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(dir_ptr, None, outvar, None)
        bytes_ = total_bytes.value
        return bytes_
    else:
        #raise NotImplementedError('')
        st = os.statvfs(dir_)
        # blocks total * block size
        bytes_ = st.f_blocks * st.f_frsize
        return bytes_


def chmod_add_executable(fpath, group=True, user=True):
    """
    References:
        http://stackoverflow.com/questions/15607903/python-module-os-chmodfile-664-does-not-change-the-permission-to-rw-rw-r-bu
        http://www.tutorialspoint.com/python/os_chmod.htm
        https://en.wikipedia.org/wiki/Chmod
    """
    import stat
    orig_mode = os.stat(fpath).st_mode
    new_mode = orig_mode
    if group:
        new_mode |= stat.S_IXGRP
    if user:
        # new_mode |= stat.S_IXUSR | stat.S_IEXEC
        new_mode |= stat.S_IXGRP | stat.S_IEXEC
    os.chmod(fpath, new_mode)


def chmod(fpath, option):
    import stat
    orig_mode = os.stat(fpath).st_mode
    new_mode = orig_mode
    if option == '+x':
        # Hack
        new_mode |= stat.S_IEXEC
    if option == 'g+x':
        new_mode |= stat.S_IXGRP

    os.chmod(fpath, new_mode)


def is_file_writable(fpath):
    return os.access(fpath, os.W_OK)


def is_file_executable(fpath):
    return os.access(fpath, os.X_OK)


def get_file_info(fpath, with_fpath=False, nice=True):
    from utool import util_time
    import os
    import time
    from collections import OrderedDict
    statbuf = os.stat(fpath)

    from pwd import getpwuid
    owner = getpwuid(os.stat(fpath).st_uid).pw_name

    info = OrderedDict([
        ('filesize', get_file_nBytes_str(fpath)),
        ('last_modified', statbuf.st_mtime),
        ('last_accessed', statbuf.st_atime),
        ('created', statbuf.st_ctime),
        ('owner', owner)
    ])
    if nice:
        for k in ['last_modified', 'last_accessed', 'created']:
            info[k] = util_time.unixtime_to_datetimestr(info[k], isutc=False)
            info[k] += (' ' + time.tzname[0])

    if with_fpath:
        info['fpath'] = fpath
    return info
    #print "Modification time:",statbuf.st_mtime


def get_file_nBytes(fpath):
    return os.path.getsize(fpath)


def get_file_nBytes_str(fpath):
    from utool import util_str
    return util_str.byte_str2(os.path.getsize(fpath))


def get_disk_space(start_path='.'):
    """
    References:
        http://stackoverflow.com/questions/1392413/calculating-a-directory-size-using-python
    """
    total_size = 0
    for root, dname_list, fname_list in os.walk(start_path):
        for fname in fname_list:
            fpath = os.path.join(root, fname)
            try:
                total_size += os.path.getsize(fpath)
            except OSError:
                pass
    return total_size


def get_dir_diskspaces(dir_):
    from utool import util_path
    path_list = util_path.ls(dir_)
    nBytes_list = [get_disk_space(path) for path in path_list]
    spacetup_list = sorted(list(zip(nBytes_list, path_list)))
    return spacetup_list


def print_dir_diskspace(dir_):
    import utool
    spacetup_list = sorted(get_dir_diskspaces(dir_))
    nBytes_list = [tup[0] for tup in spacetup_list]
    path_list   = [tup[1] for tup in spacetup_list]
    space_list = map(utool.byte_str2, nBytes_list)
    n = max(map(len, space_list))
    fmtstr = ('%' + str(n) + 's')
    space_list2 = [fmtstr % space for space in space_list]
    tupstr_list = ['%s %s' % (space2, path) for space2, path in zip(space_list2, path_list)]
    print('\n'.join(tupstr_list))


def get_lib_ext():
    return LIB_DICT[OS_TYPE]


def get_pylib_ext():
    return PYLIB_DICT[OS_TYPE]


def python_executable(check=True, short=False):
    r"""
    Args:
        short (bool): (default = False)

    Returns:
        str:

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> short = False
        >>> result = python_executable(short)
        >>> print(result)
    """
    if not check:
        python_exe = 'python'
    else:
        from os.path import isdir
        python_exe_long = unixpath(sys.executable)
        python_exe = python_exe_long
        if short:
            python_exe_short = basename(python_exe_long)
            found = search_env_paths(python_exe_short, key_list=['PATH'],
                                     verbose=False)
            found = [f for f in found if not isdir(f)]
            if len(found) > 0:
                if found[0] == python_exe_long:
                    # Safe to use the short name in this env
                    python_exe = python_exe_short
    return python_exe


def ls_libs(dpath):
    from utool import util_list
    from utool import util_path
    lib_patterns = get_dynamic_lib_globstrs()
    libpaths_list = [util_path.ls(dpath, pat) for pat in lib_patterns]
    libpath_list = util_list.flatten(libpaths_list)
    return libpath_list


def get_dynlib_dependencies(lib_path):
    """
    Executes tools for inspecting dynamic library dependencies depending on the
    current platform.
    """
    if LINUX:
        ldd_fpath = '/usr/bin/ldd'
        depend_out, depend_err, ret = cmd(ldd_fpath, lib_path, verbose=False)
    elif DARWIN:
        otool_fpath = '/opt/local/bin/otool'
        depend_out, depend_err, ret = cmd(otool_fpath, '-L', lib_path, verbose=False)
    elif WIN32:
        depend_out, depend_err, ret = cmd('objdump', '-p', lib_path, verbose=False)
        #fnmatch.filter(depend_out.split('\n'), '*DLL*')
        relevant_lines = [line for line in depend_out.splitlines() if 'DLL Name:' in line]
        depend_out = '\n'.join(relevant_lines)
    assert ret == 0, 'bad dependency check'
    return depend_out
    # objdump -p C:\Python27\Lib\site-packages\PIL\_imaging.pyd | grep dll
    # dumpbin /dependents C:\Python27\Lib\site-packages\PIL\_imaging.pyd
    # depends /c /a:1 /f:1 C:\Python27\Lib\site-packages\PIL\_imaging.pyd


def get_dynlib_exports(lib_path):
    """
    Executes tools for inspecting dynamic library dependencies depending on the
    current platform. Returns the names of callable functions.

    Args:
        lib_path (str):

    Returns:
        str: depend_out

    CommandLine:
        python -m utool.util_cplat --test-get_dynlib_exports

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> lib_path = '/home/joncrall/venv/local/lib/python2.7/site-packages/pyflann/lib/libflann.so'
        >>> depend_out = get_dynlib_exports(lib_path)
        >>> result = ('depend_out = %s' % (str(depend_out),))
        >>> print(result)
    """
    if LINUX:
        '''
        nm_fpath = '/usr/bin/nm'
        exportssout, err, ret = cmd(nm_fpath, '-D', lib_path, '|', 'c++filt', verbose=False)
        lines = exportssout.split('\n')
        #lines = [line[19:] for line in line]
        others = []
        info = []
        for line in lines:
            if line == '':
                continue
            line = ut.remove_doublspaces(line)
            words = line.split(' ')
            if len(words) > 2:
                # address, type_, rest
                rest = ' '.join(words[2:])
                info.append((rest, words[0], words[1]))
            else:
                others.append(line)

        # remove duplicate address spaces
        info = ut.unique_ordered(info)
        # remove stdlib
        info = [line for line in info if 'std::' not in line[0]]
        info = [line for line in info if not line[0].startswith('typeinfo')]
        info = [line for line in info if not line[0].startswith('vtable')]
        info = [line for line in info if 'flann' in line[0]]
        info = [line for line in info if 'flann_' in line[0]]

        info2 = []
        for rest, loc, type_ in info:
            parts = rest.split(' ')
            rettype = parts[0]
            rest2 = ' '.join(parts[1:])
            if not rest2.startswith('__'):
                info2.append((rettype, rest2, type_))
                #info2.append((rettype, rest2, type_, loc))

        len([line for line in info if 'flann' in line[0]])

        len([(line.split(' ')[0], line.split(' ')[1], ' '.join(line.split(' ')[2:])) for line in lines])
        len([line for line in lines if line.startswith('flann::')])
        len([line for line in lines if 'flann_' in line])
        len([line for line in lines if not line.endswith(')') and 'flann_' in line])
        # HACK: FIND A CORRECT PARSING
        return info2
        '''
    elif DARWIN:
        otool_fpath = '/opt/local/bin/otool'
        exportssout, err, ret = cmd(otool_fpath, '-L', lib_path, verbose=False)
        #TODO
    elif WIN32:
        exportssout, err, ret = cmd('objdump', '-p', lib_path, verbose=False)
        #TODO
        #fnmatch.filter(depend_out.split('\n'), '*DLL*')
        #relevant_lines = [line for line in depend_out.splitlines() if 'DLL Name:' in line]
        #depend_out = '\n'.join(relevant_lines)
    assert ret == 0, 'bad dependency check'
    return exportssout


def get_dynamic_lib_globstrs():
    return ['*' + libext for libext in LIB_EXT_LIST]


def get_computer_name():
    """ Returns machine name """
    return COMPUTER_NAME


def get_user_name():
    """ Returns user homefolder name """
    return basename(truepath('~'))


def get_install_dirs():
    if WIN32:
        return [r'C:\Program Files', r'C:\Program Files (x86)']
    else:
        return ['/usr/bin', '/usr/local/bin', '~/bin']


def getroot():
    root = {
        'win32': 'C:\\',  # HACK
        'linux': '/',
        'darwin': '/',
    }[OS_TYPE]
    return root


def startfile(fpath, detatch=True, quote=False, verbose=False, quiet=True):
    """ Uses default program defined by the system to open a file.

    References:
        http://stackoverflow.com/questions/2692873/quote-posix-shell-special-characters-in-python-output

    """
    print('[cplat] startfile(%r)' % fpath)
    fpath = normpath(fpath)
    # print('[cplat] fpath=%s' % fpath)
    if not exists(fpath):
        raise Exception('Cannot start nonexistant file: %r' % fpath)
    #if quote:
    #    fpath = '"%s"' % (fpath,)
    if not WIN32:
        fpath = pipes.quote(fpath)
    if LINUX:
        #out, err, ret = cmd(['xdg-open', fpath], detatch=True)
        outtup = cmd(('xdg-open', fpath), detatch=detatch, verbose=verbose, quiet=quiet)
        #outtup = cmd('xdg-open', fpath, detatch=detatch)
    elif DARWIN:
        outtup = cmd(('open', fpath), detatch=detatch, verbose=verbose, quiet=quiet)
    elif WIN32:
        os.startfile(fpath)
    else:
        raise RuntimeError('Unknown Platform')
    if outtup is not None:
        out, err, ret = outtup
        if not ret:
            raise Exception(out + ' -- ' + err)
    pass


def geteditor():
    return 'gvim'  # HACK (use util_profile)


def editfile(fpath):
    """ Runs gvim. Can also accept a module / class / function """
    if not isinstance(fpath, six.string_types):
        from six import types
        print('Rectify to module fpath = %r' % (fpath,))
        if isinstance(fpath, types.ModuleType):
            fpath = fpath.__file__
        else:
            fpath =  sys.modules[fpath.__module__].__file__
        fpath_py = fpath.replace('.pyc', '.py')
        if exists(fpath_py):
            fpath = fpath_py

    print('[cplat] startfile(%r)' % fpath)
    if not exists(fpath):
        raise Exception('Cannot start nonexistant file: %r' % fpath)
    if LINUX:
        out, err, ret = cmd(geteditor(), fpath, detatch=True)
        if not ret:
            raise Exception(out + ' -- ' + err)
    elif DARWIN:
        out, err, ret = cmd(geteditor(), fpath, detatch=True)
        if not ret:
            raise Exception(out + ' -- ' + err)
    else:
        out, err, ret = cmd(geteditor(), fpath, detatch=True)
        if not ret:
            raise Exception(out + ' -- ' + err)
        #os.startfile(fpath)
    pass


def view_file_in_directory(fpaths):
    import utool as ut
    fpaths = ut.ensure_iterable(fpaths)
    fnames = [basename(f) for f in fpaths]
    dpaths = [dirname(f) for f in fpaths]
    dpath_to_fnames = ut.group_items(fnames, dpaths)
    for dpath, fnames in dpath_to_fnames.items():
        ut.view_directory(dpath, fnames[0], verbose=False)


def view_directory(dname=None, fname=None, verbose=True):
    """
    View a directory in the operating system file browser. Currently supports
    windows explorer, mac open, and linux nautlius.

    Args:
        dname (str): directory name
        fname (str): a filename to select in the directory (nautlius only)
        verbose (bool):

    CommandLine:
        python -m utool.util_cplat --test-view_directory

    Example:
        >>> # DISABLE_DOCTEST
        >>> # DOCTEST_DISABLE
        >>> from utool.util_cplat import *  # NOQA
        >>> import utool as ut
        >>> dname = ut.truepath('~')
        >>> verbose = True
        >>> view_directory(dname, verbose)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> import utool as ut
        >>> base = ut.ensure_app_cache_dir('utool', 'test_vd')
        >>> dirs = [
        >>>     '',
        >>>     'dir1',
        >>>     'has space',
        >>>     'space at end ',
        >>>     ' space at start ',
        >>>     '"quotes and spaces"',
        >>>     "'single quotes and spaces'",
        >>>     'Frogram Piles (y2K)',
        >>> ]
        >>> dirs_ = [ut.ensuredir(join(base, d)) for d in dirs]
        >>> for dname in dirs_:
        >>>     ut.view_directory(dname, verbose=False)
        >>> fpath = join(base, 'afile.txt')
        >>> ut.touch(fpath)
        >>> ut.view_directory(base, fpath, verbose=False)
    """
    from utool.util_arg import STRICT
    from utool.util_path import checkpath
    # from utool.util_str import SINGLE_QUOTE, DOUBLE_QUOTE

    if HAVE_PATHLIB and isinstance(dname, pathlib.Path):
        dname = str(dname)

    if verbose:
        print('[cplat] view_directory(%r) ' % dname)
    dname = os.getcwd() if dname is None else dname
    open_prog = {
        'win32': 'explorer.exe',
        'linux': 'nautilus',
        'darwin': 'open'
    }[OS_TYPE]
    dname = normpath(dname)
    if STRICT:
        assert checkpath(dname, verbose=verbose), 'directory doesnt exit'
    if fname is not None and OS_TYPE == 'linux':
        arg = join(dname, fname)
    else:
        arg = dname
    # if ' ' in dname and not dname.startswith((SINGLE_QUOTE, DOUBLE_QUOTE)):
    #     # Ensure quotations
    #     dname = '"%s"' % dname
    # if not WIN32:
    #     arg = dname
    #     # arg = subprocess.list2cmdline([dname])
    #     # arg = pipes.quote(dname)
    # else:
    #     arg = dname
    # spawn and detatch process
    args = (open_prog, arg)
    print(subprocess.list2cmdline(args))
    subprocess.Popen(args)
    # print('[cplat] exit view directory')

# Alias
vd = view_directory

get_resource_dir = meta_util_cplat.get_resource_dir

get_app_resource_dir = meta_util_cplat.get_app_resource_dir


def platform_cache_dir():
    """
    Returns a directory which should be writable for any application
    This should be used for temporary deletable data.
    """
    if WIN32:  # nocover
        dpath_ = '~/AppData/Local'
    elif LINUX:  # nocover
        dpath_ = '~/.cache'
    elif DARWIN:  # nocover
        dpath_  = '~/Library/Caches'
    else:  # nocover
        raise NotImplementedError('Unknown Platform  %r' % (sys.platform,))
    dpath = normpath(expanduser(dpath_))
    return dpath


def get_app_cache_dir(appname, *args):
    r"""
    Returns a writable directory for an application.
    This should be used for temporary deletable data.

    Args:
        appname (str): the name of the application
        *args: any other subdirectories may be specified

    Returns:
        str: dpath: writable cache directory
    """
    dpath = join(platform_cache_dir(), appname, *args)
    return dpath


def ensure_app_cache_dir(appname, *args):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> import utool as ut
        >>> dpath = ut.ensure_app_cache_dir('utool')
        >>> assert exists(dpath)
    """
    import utool as ut
    dpath = get_app_cache_dir(appname, *args)
    ut.ensuredir(dpath)
    return dpath


def ensure_app_resource_dir(*args, **kwargs):
    import utool as ut
    app_resource_dir = get_app_resource_dir(*args, **kwargs)
    ut.ensuredir(app_resource_dir)
    return app_resource_dir


def shell(*args, **kwargs):
    """
    Dangerous. Take out of production code
    """
    kwargs['shell'] = True
    return cmd(*args, **kwargs)


def __parse_cmd_kwargs(kwargs):
    verbose = kwargs.get('verbose', True)
    detatch = kwargs.get('detatch', False)
    #shell   = kwargs.get('shell', False)
    # seems like linux needs the shell to work well
    # maybe thats because I'm a windows admin
    # FIXME: Turn shell off by default and fix __parse_cmd_args
    shell   = kwargs.get('shell', LINUX or DARWIN)
    # TODO: gksudo
    sudo    = kwargs.get('sudo', False)
    # pads stdout of cmd before and after
    # TODO: rename separate to something else
    silence    = kwargs.get('silence', False)
    quiet    = kwargs.get('quiet', False)
    pad_stdout    = kwargs.get('pad_stdout', not (silence or quiet))
    return verbose, detatch, shell, sudo, pad_stdout


def __parse_cmd_args(args, sudo, shell):
    """
    When shell is True, Popen will only accept strings. No tuples
    Shell really should not be true.

    Returns:
        args suitable for subprocess.Popen

        I'm not quite sure what those are yet. Plain old string seem to work
        well? But I remember needing shlex at some point.

    CommandLine:
        python -m utool.util_cplat --test-__parse_cmd_args

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> # build test data
        >>> args = 'echo "hello world"'
        >>> sudo = False
        >>> shell = False
        >>> # execute function
        >>> args = __parse_cmd_args(args, sudo, shell)
        >>> # verify results
        >>> result = str(args)
        >>> print(result)
    """
    # Case where tuple is passed in as only argument
    if isinstance(args, tuple) and len(args) == 1 and isinstance(args[0], tuple):
        args = args[0]

    if shell:
        # When shell is True, ensure args is a string
        if isinstance(args, six.string_types):
            pass
        elif  isinstance(args, (list, tuple)) and len(args) > 1:
            args = ' '.join(args)
        elif isinstance(args, (list, tuple)) and len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                args = ' '.join(args)
            elif isinstance(args[0], six.string_types):
                args = args[0]
    else:
        # When shell is False, ensure args is a tuple
        if isinstance(args, six.string_types):
            args = shlex.split(args, posix=not WIN32)
        elif isinstance(args, (list, tuple)):
            if len(args) > 1:
                args = tuple(args)
            elif len(args) == 1:
                if isinstance(args[0], (tuple, list)):
                    args = tuple(args[0])
                elif isinstance(args[0], six.string_types):
                    args = shlex.split(args[0], posix=not WIN32)
    if sudo is True:
        if not WIN32:
            if shell:
                args = 'sudo ' + args
            else:
                args = tuple(['sudo']) + tuple(args)
            #if isinstance(args, six.string_types):
            #    args = shlex.split(args)
            #args = ['sudo'] + args
            ## using sudo means we need to use a single string I believe
            #args = ' '.join(args)
        else:
            # TODO: strip out sudos
            pass
    # HACK FOR WINDOWS AGAIN
    # makes  this command work:
    # python -c "import utool as ut; ut.cmd('build\\hesaffexe.exe ' + ut.grab_test_imgpath('star.png'))"
    # and this should still work
    # python -c "import utool as ut; ut.cmd('build\\hesaffexe.exe', ut.grab_test_imgpath('star.png'))"
    if WIN32:
        if len(args) == 1 and isinstance(args[0], six.string_types):
            args = shlex.split(args[0], posix=not WIN32)
    return args


def run_realtime_process(exe, shell=False):
    proc = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell)
    while(True):
        # WARNING: this can cause deadlocks apparently if the OS pipe buffers
        # fill up.
        retcode = proc.poll()  # returns None while subprocess is running
        line = proc.stdout.readline()
        yield line
        if retcode is not None:
            return
            # raise StopIteration('process finished')


def _run_process(proc):
    while True:
        # returns None while subprocess is running
        retcode = proc.poll()
        line = proc.stdout.readline()
        yield line
        if retcode is not None:
            # The program has a return code, so its done executing.
            # Grab any remaining data in stdout
            for line in proc.stdout.readlines():
                yield line
            return
            # raise StopIteration('process finished')


def quote_single_command(cmdstr):
    if ' ' in cmdstr:
        return '\'' + cmdstr + '\''
    return cmdstr


def cmd(*args, **kwargs):
    r""" A really roundabout way to issue a system call

    # FIXME: This function needs some work
    # It should work without a hitch on windows or unix.
    # It should be able to spit out stdout in realtime.
    # Should be able to configure detatchment, shell, and sudo.

    FIXME:
        on a mac ut.cmd('/Users/joncrall/Library/Application Support/ibeis/tomcat/bin/shutdown.sh') will fail due to spaces


    Kwargs:
        quiet (bool) :
        silence (bool) :
        verbose (bool) :
        detatch (bool) :
        shell (bool) :
        sudo (bool) :
        pad_stdout (bool) :
        dryrun (bool) :

    Returns:
        tuple: (None, None, None)

    CommandLine:
        python -m utool.util_cplat --test-cmd
        python -m utool.util_cplat --test-cmd:0
        python -m utool.util_cplat --test-cmd:1
        python -m utool.util_cplat --test-cmd:2
        python -m utool.util_cplat --test-cmd:1 --test-sudo
        python -m utool.util_cplat --test-cmd:2 --test-sudo

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> (out, err, ret) = ut.cmd('echo', 'hello world')
        >>> result = ut.repr4(list(zip(('out', 'err', 'ret'), (out, err, ret))), nobraces=True)
        >>> print(result)
        ('out', 'hello world\n'),
        ('err', None),
        ('ret', 0),

    Example1:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> target = ut.codeblock(
        ...      r'''
                 ('out', 'hello world\n'),
                 ('err', None),
                 ('ret', 0),
                 ''')
        >>> varydict = {
        ...    'shell': [True, False],
        ...    'detatch': [False],
        ...    'sudo': [True, False] if ut.get_argflag('--test-sudo') else [False],
        ...    'args': ['echo hello world', ('echo', 'hello world')],
        ... }
        >>> for count, kw in enumerate(ut.all_dict_combinations(varydict), start=1):
        >>>     print('+ --- TEST CMD %d ---' % (count,))
        >>>     print('testing cmd with params ' + ut.repr4(kw))
        >>>     args = kw.pop('args')
        >>>     restup = ut.cmd(args, pad_stdout=False, **kw)
        >>>     tupfields = ('out', 'err', 'ret')
        >>>     output = ut.repr4(list(zip(tupfields, restup)), nobraces=True)
        >>>     ut.assert_eq(output, target)
        >>>     print('L ___ TEST CMD %d ___\n' % (count,))

    Example2:
        >>> # DISABLE_DOCTEST
        >>> # UNSTABLE_DOCTEST
        >>> # ping is not as universal of a command as I thought
        >>> from utool.util_cplat import *  # NOQA
        >>> import utool as ut
        >>> varydict = {
        ...    'shell': [True, False],
        ...    'detatch': [True],
        ...    'args': ['ping localhost', ('ping', 'localhost')],
        ... }
        >>> proc_list = []
        >>> for count, kw in enumerate(ut.all_dict_combinations(varydict), start=1):
        >>>     print('+ --- TEST CMD %d ---' % (count,))
        >>>     print('testing cmd with params ' + ut.repr4(kw))
        >>>     args = kw.pop('args')
        >>>     restup = ut.cmd(args, pad_stdout=False, **kw)
        >>>     out, err, proc = restup
        >>>     proc_list.append(proc)
        >>>     print(proc)
        >>>     print(proc)
        >>>     print(proc.poll())
        >>>     print('L ___ TEST CMD %d ___\n' % (count,))
    """
    try:
        # Parse the keyword arguments
        verbose, detatch, shell, sudo, pad_stdout = __parse_cmd_kwargs(kwargs)
        quiet = kwargs.pop('quiet', False)
        silence = kwargs.pop('silence', False)
        if pad_stdout:
            sys.stdout.flush()
            print('\n+--------')
        args = __parse_cmd_args(args, sudo, shell)
        # Print what you are about to do
        if not quiet:
            print('[ut.cmd] RUNNING: %r' % (args,))
        # Open a subprocess with a pipe
        if kwargs.get('dryrun', False):
            print('[ut.cmd] Exiting because dryrun=True')
            return None, None, None
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, shell=shell,
                                universal_newlines=True
                                # universal_newlines=False
                                )
        hack_use_stdout = True
        if detatch:
            if not quiet:
                print('[ut.cmd] PROCESS DETATCHING. No stdoutput can be reported...')
            # There is no immediate confirmation as to whether or not the script
            # finished. It might still be running for all you know
            return None, None, proc
        else:
            if verbose and not detatch:
                if not quiet:
                    print('[ut.cmd] RUNNING WITH VERBOSE OUTPUT')
                logged_out = []
                for line in _run_process(proc):
                    #line_ = line if six.PY2 else line.decode('utf-8')
                    line_ = line if six.PY2 else line
                    if len(line_) > 0:
                        if not silence:
                            if hack_use_stdout:
                                sys.stdout.write(line_)
                                sys.stdout.flush()
                            else:
                                # TODO make this play nicely with loggers
                                print_(line_)
                        logged_out.append(line)
                try:
                    from utool import util_str  # NOQA
                    # logged_out = util_str.ensure_unicode_strlist(logged_out)
                    out = '\n'.join(logged_out)
                except UnicodeDecodeError:
                    from utool import util_str  # NOQA
                    logged_out = util_str.ensure_unicode_strlist(logged_out)
                    out = '\n'.join(logged_out)
                    # print('logged_out = %r' % (logged_out,))
                    # raise
                (out_, err) = proc.communicate()
                #print('[ut.cmd] out: %s' % (out,))
                if not quiet:
                    try:
                        print('[ut.cmd] stdout: %s' % (out_,))
                        print('[ut.cmd] stderr: %s' % (err,))
                    except UnicodeDecodeError:
                        from utool import util_str  # NOQA
                        print('[ut.cmd] stdout: %s' % (util_str.ensure_unicode(out_),))
                        print('[ut.cmd] stderr: %s' % (util_str.ensure_unicode(err),))

            else:
                # Surpress output
                #print('[ut.cmd] RUNNING WITH SUPRESSED OUTPUT')
                (out, err) = proc.communicate()
            # Make sure process if finished
            ret = proc.wait()
            if not quiet:
                print('[ut.cmd] PROCESS FINISHED')
            if pad_stdout:
                print('L________\n')
            return out, err, ret
    except Exception as ex:
        import utool as ut
        #if isinstance(args, tuple):
        #    print(ut.truepath(args[0]))
        #elif isinstance(args, six.string_types):
        #    print(ut.unixpath(args))
        ut.printex(ex, 'Exception running ut.cmd',
                   keys=['verbose', 'detatch', 'shell', 'sudo', 'pad_stdout'],
                   tb=True)


def cmd2(command, shell=False, detatch=False, verbose=False, verbout=None):
    """
    Trying to clean up cmd

    Args:
        command (str): string command
        shell (bool): if True, process is run in shell
        detatch (bool): if True, process is run in background
        verbose (int): verbosity mode
        verbout (bool): if True, `command` writes to stdout in realtime.
            defaults to True iff verbose > 0

    Returns:
        dict: info - information about command status
    """
    import shlex
    if isinstance(command, (list, tuple)):
        raise ValueError('command tuple not supported yet')
    args = shlex.split(command, posix=not WIN32)
    if verbose is True:
        verbose = 2
    if verbout is None:
        verbout = verbose >= 1
    if verbose >= 2:
        print('+=== START CMD2 ===')
        print('Command:')
        print(command)
        if verbout:
            print('----')
            print('Stdout:')
    proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, shell=shell,
                            universal_newlines=True)
    if detatch:
        info = {'proc': proc}
    else:
        write_fn = sys.stdout.write
        flush_fn = sys.stdout.flush
        logged_out = []
        for line in _run_process(proc):
            #line_ = line if six.PY2 else line.decode('utf-8')
            line_ = line if six.PY2 else line
            if len(line_) > 0:
                if verbout:
                    write_fn(line_)
                    flush_fn()
                logged_out.append(line)
        try:
            from utool import util_str  # NOQA
            # out = '\n'.join(logged_out)
            out = ''.join(logged_out)
        except UnicodeDecodeError:
            from utool import util_str  # NOQA
            logged_out = util_str.ensure_unicode_strlist(logged_out)
            # out = '\n'.join(logged_out)
            out = ''.join(logged_out)
            # print('logged_out = %r' % (logged_out,))
            # raise
        (out_, err) = proc.communicate()
        ret = proc.wait()
        info = {
            'out': out,
            'err': err,
            'ret': ret,
        }
    if verbose >= 2:
        print('L___ END CMD2 ___')
    return info


def get_flops():
    """ # DOESNT WORK """
    from sys import stdout
    from re import compile

    filename = "linpack.out"
    fpnum = r'\d+\.\d+E[+-]\d\d'
    fpnum_1 = fpnum + r' +'
    pattern = compile(r'^ *' + fpnum_1 + fpnum_1 + fpnum_1 + r'(' + fpnum + r') +' + fpnum_1 + fpnum + r' *\n$')
    speeds = [0.0, 1.0e75, 0.0]

    file = open(filename)
    count = 0
    while file :
        line = file.readline()
        if not line :
            break
        if pattern.match(line) :
            count = count + 1
            x = float(pattern.sub(r'\1', line))
            if x < 1.0 :
                print(count)
            speeds[0] = speeds[0] + x
            speeds[1] = min(speeds[1], x)
            speeds[2] = max(speeds[2], x)
    file.close()
    if count != 0 :
        speeds[0] = speeds[0] / count

    stdout.write("%6.1f MFlops (%d from %.1f to %.1f)\n" % (speeds[0], count, speeds[1], speeds[2]))


def set_process_title(title):
    try:
        import setproctitle
        setproctitle.setproctitle(title)
    except ImportError as ex:
        import utool
        utool.printex(ex, iswarning=True)


def is64bit_python():
    """
    Returns:
        True if running 64 bit python and False if running on 32 bit
    """
    #http://stackoverflow.com/questions/1405913/how-do-i-determine-if-my-python-shell-is-executing-in-32bit-or-64bit-mode-on-os
    is64bit = sys.maxsize > 2 ** 32
    #import platform
    #platform.architecture()
    #import ctypes
    #(ctypes.sizeof(ctypes.c_voidp))
    return is64bit


def get_python_dynlib():
    """
    python -c "import utool; print(utool.get_python_dynlib())"

    get_python_dynlib

    Returns:
        ?: dynlib

    Example:
        >>> # DISABLE_DOCTEST
        >>> # DOCTEST_DISABLE
        >>> from utool.util_cplat import *  # NOQA
        >>> dynlib = get_python_dynlib()
        >>> print(dynlib)
        /usr/lib/x86_64-linux-gnu/libpython2.7.so
    """
    import sysconfig
    cfgvars = sysconfig.get_config_vars()
    dynlib = os.path.join(cfgvars['LIBDIR'], cfgvars['MULTIARCH'], cfgvars['LDLIBRARY'])
    if not exists(dynlib):
        dynlib = os.path.join(cfgvars['LIBDIR'], cfgvars['LDLIBRARY'])
    assert exists(dynlib)
    return dynlib


def get_path_dirs():
    """
    returns a list of directories in the PATH system variable

    Returns:
        list: pathdirs

    CommandLine:
        python -m utool.util_cplat --exec-get_path_dirs

    Example:
        >>> # DISABLE_DOCTEST
        >>> # SCRIPT
        >>> from utool.util_cplat import *  # NOQA
        >>> import utool as ut
        >>> pathdirs = get_path_dirs()
        >>> result = ('pathdirs = %s' % (ut.repr4(pathdirs),))
        >>> print(result)

    """
    pathdirs = os.environ['PATH'].split(os.pathsep)
    return pathdirs


def print_path(sort=True):
    pathdirs = get_path_dirs()
    if sort:
        pathdirs = sorted(pathdirs)
    print('\n'.join(pathdirs))


def search_env_paths(fname, key_list=None, verbose=None):
    r"""
    Searches your PATH to see if fname exists

    Args:
        fname (str): file name to search for (can be glob pattern)

    CommandLine:
        python -m utool search_env_paths --fname msvcr*.dll
        python -m utool search_env_paths --fname '*flann*'

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> import utool as ut
        >>> fname = 'opencv2/highgui/libopencv_highgui.so'
        >>> fname = ut.get_argval('--fname', default='*')
        >>> print('fname = %r' % (fname,))
        >>> key_list = None # ['PATH']
        >>> found = search_env_paths(fname, key_list)
        >>> print(ut.repr4(found, nl=True, strvals=True))

    Ignore:
        OpenCV_DIR:PATH={share_opencv}
        OpenCV_CONFIG_PATH:FILEPATH={share_opencv}

    """
    import utool as ut
    # from os.path import join
    if key_list is None:
        key_list = [key for key in os.environ if key.find('PATH') > -1]
        print('key_list = %r' % (key_list,))

    found = ut.ddict(list)

    for key in key_list:
        dpath_list = os.environ[key].split(os.pathsep)
        for dpath in dpath_list:
            #if verbose:
            #    print('dpath = %r' % (dpath,))
            # testname = join(dpath, fname)
            matches = ut.glob(dpath, fname)
            found[key].extend(matches)
            #import fnmatch
            #import utool
            #utool.embed()
            #if ut.checkpath(testname, verbose=False):
            #    if verbose:
            #        print('Found in key=%r' % (key,))
            #        ut.checkpath(testname, verbose=True, info=True)
            #    found += [testname]
    return dict(found)


def __debug_win_msvcr():
    import utool as ut
    fname = 'msvcr*.dll'
    key_list = ['PATH']
    found = ut.search_env_paths(fname, key_list)
    fpaths = ut.unique(ut.flatten(found.values()))
    fpaths = ut.lmap(ut.ensure_unixslash, fpaths)
    from os.path import basename
    dllnames = [basename(x) for x in fpaths]
    grouped = dict(ut.group_items(fpaths, dllnames))
    print(ut.repr4(grouped, nl=4))

    keytoid = {
    }

    for key, vals in grouped.items():
        infos = ut.lmap(ut.get_file_nBytes, vals)
        #infos = ut.lmap(ut.get_file_uuid, vals)
        #uuids = [ut.get_file_uuid(val) for val in vals]
        keytoid[key] = list(zip(infos, vals))
    ut.print_dict(keytoid, nl=2)


def change_term_title(title):
    r"""
    only works on unix systems only tested on Ubuntu GNOME changes text on
    terminal title for identifying debugging tasks.

    The title will remain until python exists

    Args:
        title (str):

    References:
        http://stackoverflow.com/questions/5343265/setting-title-for-tabs-in-terminator-console-application-in-ubuntu/8850484#8850484

    CommandLine:
        python -m utool change_term_title
        echo -en "\033]0;newtitle\a"

         printf "\e]2;newtitle\a";

        echo -en "\033]0;DocTest /home/joncrall/code/ibeis/ibeis.algo.graph.core.py --test-AnnotInference._make_state_delta\a"

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> title = 'change title test'
        >>> result = change_term_title(title)
        >>> print(result)
    """
    if True:
        # Disabled
        return
    if not WIN32:
        #print("CHANGE TERM TITLE to %r" % (title,))
        if title:
            #os.environ['PS1'] = os.environ['PS1'] + '''"\e]2;\"''' + title + '''\"\a"'''
            cmd_str = r'''echo -en "\033]0;''' + title + '''\a"'''
            os.system(cmd_str)


def send_keyboard_input(text=None, key_list=None):
    """
    Args:
        text (None):
        key_list (list):

    References:
        http://stackoverflow.com/questions/14788036/python-win32api-sendmesage
        http://www.pinvoke.net/default.aspx/user32.sendinput

    CommandLine:
        python -m utool.util_cplat --test-send_keyboard_input

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> text = '%paste'
        >>> result = send_keyboard_input('%paste')
        >>> print(result)
    """
    #key_mapping = {
    #    'enter':
    #}
    if WIN32:
        #raise NotImplementedError()
        #import win32api
        #import win32gui
        #import win32con
        #hwnd = win32gui.GetForegroundWindow()
        #print('entering text into %r' % (win32gui.GetWindowText(hwnd ),))
        #win32con.VK_RETURN

        #def callback(hwnd, hwnds):
        #   #if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
        #       #hwnds[win32gui.GetClassName(hwnd)] = hwnd
        #   #return True
        #hwnds = {}
        #win32gui.EnumChildWindows(hwnd, callback, hwnds)

        #for ord_char in map(ord, text):
        #   #win32api.SendMessage(hwnd, win32con.WM_CHAR, ord_char, 0)
        from utool._internal import win32_send_keys
        pause = float(.05)
        text = 'paste'
        keys = text
        kw = dict(with_spaces=False, with_tabs=True, with_newlines=False)
        win32_send_keys.SendKeys(keys, pause=pause, turn_off_numlock=True, **kw)
        #win32_send_keys
        #import time
        #keys_ = win32_send_keys.parse_keys(keys, **kw)
        #for k in keys_:
        #    k.Run()
        #    time.sleep(pause)

    else:
        if key_list is None:
            char_map = {
                '%': 'shift+5'
            }
            key_list = [char_map.get(char, char) for char in text]
        xdotool_args = ['xdotool', 'key'] + key_list
        #, 'shift+5', 'p', 'a', 's', 't', 'e', 'enter']
        cmd = ' '.join(xdotool_args)
        print('Running: cmd=%r' % (cmd,))
        print('+---')
        print(cmd)
        print('L___')
        os.system(cmd)


def ipython_paste(*args, **kwargs):
    """ pastes for me FIXME: make something like this work on unix and windows"""
    if WIN32:
        pass
    else:
        winhandle = 'joncrall@Hyrule'
        args1 = ['wmctrl', '-a', winhandle]
        args2 = ['xdotool', 'key', 'shift+5', 'p', 'a', 's', 't', 'e', 'enter']
        os.system(' '.join(args1))
        os.system(' '.join(args2))

    #fallback_execute(args1)
    #fallback_execute(args2)


def spawn_delayed_ipython_paste():
    import utool as ut
    # Gonna be pasting
    def delayed_ipython_paste(delay):
        import time
        import utool as ut
        #import os
        print('waiting')
        time.sleep(delay)
        ut.send_keyboard_input(text='%paste')
        ut.send_keyboard_input(key_list=['KP_Enter'])
        #os.system(' '.join(['xdotool', 'key', 'shift+5', 'p', 'a', 's', 't', 'e', 'KP_Enter']))
    ut.spawn_background_thread(delayed_ipython_paste, .1)


def print_system_users():
    r"""

    prints users on the system

    On unix looks for /bin/bash users in /etc/passwd

    CommandLine:
        python -m utool.util_cplat --test-print_system_users

    Example:
        >>> # DISABLE_DOCTEST
        >>> # SCRIPT
        >>> from utool.util_cplat import *  # NOQA
        >>> result = print_system_users()
        >>> print(result)
    """
    import utool as ut
    text = ut.read_from('/etc/passwd')
    userinfo_text_list = text.splitlines()
    userinfo_list = [uitext.split(':') for uitext in userinfo_text_list]
    #print(ut.repr4(sorted(userinfo_list)))
    bash_users = [tup for tup in userinfo_list if tup[-1] == '/bin/bash']
    print(ut.repr4(sorted(bash_users)))


def check_installed_debian(pkgname):
    """
    References:
        http://www.cyberciti.biz/faq/find-out-if-package-is-installed-in-linux/
    """
    import utool as ut
    #pkgname = 'espeak'
    #pkgname = 'sudo'
    #ut.cmd('hash ' + pkgname + ' 2>/dev/null')
    tup = ut.cmd('hash ' + pkgname + ' 2>/dev/null', quiet=True, pad_stdout=False)
    out, err, ret = tup
    is_installed = (ret == 0)
    return is_installed


def assert_installed_debian(pkgname):
    import utool as ut
    if not ut.check_installed_debian(pkgname):
        raise AssertionError('espeak must be installed. run sudo apt-get install -y ' + pkgname)


def unload_module(modname):
    """
    WARNING POTENTIALLY DANGEROUS AND MAY NOT WORK

    References:
        http://stackoverflow.com/questions/437589/how-do-i-unload-reload-a-python-module

    CommandLine:
        python -m utool.util_cplat --test-unload_module

    Example:
        >>> # DISABLE_DOCTEST
        >>> import sys, gc  # NOQA
        >>> import pyhesaff
        >>> import utool as ut
        >>> modname = 'pyhesaff'
        >>> print('%s refcount=%r' % (modname, sys.getrefcount(pyhesaff),))
        >>> #referrer_list = gc.get_referrers(sys.modules[modname])
        >>> #print('referrer_list = %s' % (ut.repr4(referrer_list),))
        >>> ut.unload_module(modname)
        >>> assert pyhesaff is None

    """
    import sys
    import gc
    if modname in sys.modules:
        referrer_list = gc.get_referrers(sys.modules[modname])
        #module = sys.modules[modname]
        for referer in referrer_list:
            if referer is not sys.modules:
                referer[modname] = None
            #del referer[modname]
        #sys.modules[modname] = module
        #del module
        refcount = sys.getrefcount(sys.modules[modname])
        print('%s refcount=%r' % (modname, refcount))
        del sys.modules[modname]


#def get_ipython_config_file():
#    """
#    or to create an empty default profile, populated with default config files:
#    ipython profile create
#    """

#from subprocess import check_output
#http://stackoverflow.com/questions/8015163/how-to-check-screen-is-running
#def screen_present(name):
#        var = check_output(["screen -ls; true"],shell=True)
#        if "."+name+"\t(" in var:
#                print name+" is running"
#        else:
#                print name+" is not running"


def pip_install(package):
    """
    References:
        http://stackoverflow.com/questions/15974100/ipython-install-new-modules
    """
    import pip
    pip.main(['install', package])


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_cplat; utool.doctest_funcs(utool.util_cplat, allexamples=True)"
        python -c "import utool, utool.util_cplat; utool.doctest_funcs(utool.util_cplat)"
        python -m utool.util_cplat
        python -m utool.util_cplat --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
