"""
cross platform utilities
"""
from __future__ import absolute_import, division, print_function
import os
import six
import sys
import platform
import subprocess
import shlex
from os.path import exists, normpath, basename
from .util_inject import inject
from utool._internal import meta_util_cplat
from utool._internal.meta_util_path import unixpath, truepath
print, print_, printDBG, rrr, profile = inject(__name__, '[cplat]')

COMPUTER_NAME = platform.node()

OS_TYPE = meta_util_cplat.OS_TYPE
WIN32  = meta_util_cplat.WIN32
LINUX  = meta_util_cplat.LINUX
DARWIN = meta_util_cplat.DARWIN

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


def get_free_diskbytes(dir_):
    """
    Returns:
        folder/drive free space (in bytes)

    References::
        http://stackoverflow.com/questions/51658/cross-platform-space-remaining-on-volume-using-python
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
        bytes_ = st.f_bavail * st.f_frsize
        return bytes_


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


def python_executable():
    return unixpath(sys.executable)


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
        'win32': 'C:\\',
        'linux': '/',
        'darwin': '/',
    }[OS_TYPE]
    return root


def startfile(fpath):
    """ Uses default program defined by the system to open a file. """
    print('[cplat] startfile(%r)' % fpath)
    if not exists(fpath):
        raise Exception('Cannot start nonexistant file: %r' % fpath)
    if LINUX:
        out, err, ret = cmd(['xdg-open', fpath], detatch=True)
        if not ret:
            raise Exception(out + ' -- ' + err)
    elif DARWIN:
        out, err, ret = cmd(['open', fpath], detatch=True)
        if not ret:
            raise Exception(out + ' -- ' + err)
    else:
        os.startfile(fpath)
    pass


def geteditor(fpath):
    return 'gvim'


def editfile(fpath):
    """ Runs gvim """
    print('[cplat] startfile(%r)' % fpath)
    if not exists(fpath):
        raise Exception('Cannot start nonexistant file: %r' % fpath)
    if LINUX:
        out, err, ret = cmd([geteditor(), fpath], detatch=True)
        if not ret:
            raise Exception(out + ' -- ' + err)
    elif DARWIN:
        out, err, ret = cmd([geteditor(), fpath], detatch=True)
        if not ret:
            raise Exception(out + ' -- ' + err)
    else:
        out, err, ret = cmd([geteditor(), fpath], detatch=True)
        if not ret:
            raise Exception(out + ' -- ' + err)
        #os.startfile(fpath)
    pass


def view_directory(dname=None, verbose=True):
    """
    view directory

    Args:
        dname (str): directory name
        verbose (bool):

    Example:
        >>> # DOCTEST_DISABLE
        >>> from utool.util_cplat import *  # NOQA
        >>> dname = None
        >>> verbose = True
        >>> view_directory(dname, verbose)
    """
    from utool.util_arg import STRICT
    from utool.util_path import checkpath

    if verbose:
        print('[cplat] view_directory(%r) ' % dname)
    dname = os.getcwd() if dname is None else dname
    open_prog = {'win32': 'explorer.exe',
                 'linux': 'nautilus',
                 'darwin': 'open'}[OS_TYPE]
    dname = normpath(dname)
    if STRICT:
        assert checkpath(dname, verbose=verbose)
    if dname.find(' ') != -1 and not dname.startswith(('"', '\'')):
        dname = '"%s"' % dname
    os.system(open_prog + ' ' + dname)

# Alias
vd = view_directory

get_resource_dir = meta_util_cplat.get_resource_dir

get_app_resource_dir = meta_util_cplat.get_app_resource_dir


def ensure_app_resource_dir(*args):
    import utool as ut
    app_resource_dir = get_app_resource_dir(*args)
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
    shell   = kwargs.get('shell', LINUX)
    sudo    = kwargs.get('sudo', False)
    separate    = kwargs.get('separate', True)
    return verbose, detatch, shell, sudo, separate


def __parse_cmd_args(args, sudo, shell):
    """
    Returns:
        args suitable for subprocess.Popen

        I'm not quite sure what those are yet. Plain old string seem to work
        well? But I remember needing shlex at some point.
    """
    #from .util_arg import VERBOSE
    #args = ' '.join(args)
    #if VERBOSE:
    #    print('[cplat] Joined args:')
    #    print(' '.join(args))
    print(type(args))
    print(args)
    #print(shlex)
    if shell:
        # Popen only accepts strings is shell is True, which
        # it really shouldn't be.
        if  isinstance(args, (list, tuple)) and len(args) > 1:
            # Input is ['cmd', 'arg1', 'arg2']
            args = ' '.join(args)
        elif isinstance(args, (list, tuple)) and len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                # input got nexted
                args = ' '.join(args)
            elif isinstance(args[0], six.string_types):
                # input is just nested string
                args = args[0]
        elif isinstance(args, six.string_types):
            pass
    if sudo is True:
        if not WIN32:
            if isinstance(args, six.string_types):
                args = shlex.split(args)
            args = ['sudo'] + args
            # using sudo means we need to use a single string I believe
            args = ' '.join(args)
        else:
            # TODO: strip out sudos
            pass

    #if isinstance(args, (list, tuple)):
    #    if len(args) == 1:
    #        print('HERE1')
    #        if isinstance(args[0], list):
    #            print('HERE5')
    #            args = args[0]
    #        elif isinstance(args[0], str):
    #            print('HERE2')
    #            if WIN32:
    #                args = shlex.split(args[0])
    #    #else:
    #        #if LINUX:
    #        #    args = ' '.join(args)
    #if isinstance(args, six.string_types):
    #    print('HERE3')
    #    #if os.name == 'posix':
    #    #    args = shlex.split(args)
    #    #else:
    #        #args = [args]
    return args


def run_realtime_process(exe, shell=False):
    proc = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell)
    while(True):
        retcode = proc.poll()  # returns None while subprocess is running
        line = proc.stdout.readline()
        yield line
        if retcode is not None:
            raise StopIteration('process finished')


def _run_process(proc):
    while True:
        # returns None while subprocess is running
        retcode = proc.poll()
        line = proc.stdout.readline()
        yield line
        if retcode is not None:
            raise StopIteration('process finished')


def cmd(*args, **kwargs):
    """ A really roundabout way to issue a system call

    # FIXME: This function needs some work
    # It should work without a hitch on windows or unix.
    # It should be able to spit out stdout in realtime.
    # Should be able to configure detatchment, shell, and sudo.

    """
    try:
        sys.stdout.flush()
        # Parse the keyword arguments
        verbose, detatch, shell, sudo, separate = __parse_cmd_kwargs(kwargs)
        if separate:
            print('\n+--------------')
        args = __parse_cmd_args(args, sudo, shell)
        # Print what you are about to do
        print('[ut.cmd] RUNNING: %r' % (args,))
        # Open a subprocess with a pipe
        proc = subprocess.Popen(args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                shell=shell)
        if detatch:
            print('[ut.cmd] PROCESS DETATCHING')
            return None, None, 1
        if verbose and not detatch:
            print('[ut.cmd] RUNNING WITH VERBOSE OUTPUT')
            logged_out = []
            for line in _run_process(proc):
                if six.PY2:
                    sys.stdout.write(line)
                elif six.PY3:
                    sys.stdout.write(line.decode('utf-8'))
                sys.stdout.flush()
                logged_out.append(line)
            out = '\n'.join(logged_out)
            (out_, err) = proc.communicate()
            #print('[ut.cmd] out: %s' % (out,))
            print('[ut.cmd] stdout: %s' % (out_,))
            print('[ut.cmd] stderr: %s' % (err,))
        else:
            # Surpress output
            #print('[ut.cmd] RUNNING WITH SUPRESSED OUTPUT')
            (out, err) = proc.communicate()
        # Make sure process if finished
        ret = proc.wait()
        print('[ut.cmd] PROCESS FINISHED')
        if separate:
            print('L--------------\n')
        return out, err, ret
    except Exception as ex:
        import utool as ut
        if isinstance(args, tuple):
            print(ut.truepath(args[0]))
        elif isinstance(args, six.string_types):
            print(ut.unixpath(args))
        ut.printex(ex, 'Exception running ut.cmd',
                   keys=['verbose', 'detatch', 'shell', 'sudo', 'separate'],
                   tb=True)


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
    """ returns a list of directories in the PATH system variable """
    pathdirs = os.environ['PATH'].split(os.pathsep)
    return pathdirs


#from subprocess import check_output
#http://stackoverflow.com/questions/8015163/how-to-check-screen-is-running
#def screen_present(name):
#        var = check_output(["screen -ls; true"],shell=True)
#        if "."+name+"\t(" in var:
#                print name+" is running"
#        else:
#                print name+" is not running"

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
