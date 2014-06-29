"""
cross platform utilities
"""
from __future__ import absolute_import, division, print_function
import os
import sys
import platform
import subprocess
import shlex
from os.path import exists, normpath
from .util_inject import inject
from ._internal import meta_util_cplat
from ._internal.meta_util_path import unixpath
print, print_, printDBG, rrr, profile = inject(__name__, '[cplat]')

COMPUTER_NAME = platform.node()

WIN32  = meta_util_cplat.WIN32
LINUX  = meta_util_cplat.LINUX
DARWIN = meta_util_cplat.DARWIN

LIB_EXT_LIST = ['.so', '.dll', '.dylib', '.pyd']


def python_executable():
    return unixpath(sys.executable)


def ls_libs(dpath):
    from . import util_list
    from . import util_path
    lib_patterns = get_dynamic_lib_globstrs()
    libpaths_list = [util_path.ls(dpath, pat) for pat in lib_patterns]
    libpath_list = util_list.flatten(libpaths_list)
    return libpath_list


def get_dynlib_dependencies(lib_path):
    if LINUX:
        depend_out, depend_err, ret = cmd('ldd', lib_path, verbose=False)
    elif DARWIN:
        depend_out, depend_err, ret = cmd('otool', '-L', lib_path, verbose=False)
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
    return COMPUTER_NAME


def getroot():
    root = {
        'WIN32': 'C:\\',
        'LINUX': '/',
        'DARWIN': '/',
    }[sys.platform]
    return root


def startfile(fpath):
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


def view_directory(dname=None):
    """ view directory """
    print('[cplat] view_directory(%r) ' % dname)
    dname = os.getcwd() if dname is None else dname
    open_prog = {'win32': 'explorer.exe',
                 'linux2': 'nautilus',
                 'darwin': 'open'}[sys.platform]
    dname = normpath(dname)
    if dname.find(' ') != -1 and not dname.startswith(('"', '\'')):
        dname = '"%s"' % dname
    os.system(open_prog + ' ' + dname)

# Alias
vd = view_directory


get_resource_dir = meta_util_cplat.get_resource_dir

get_app_resource_dir = meta_util_cplat.get_app_resource_dir


def shell(*args, **kwargs):
    """
    Dangerous. Take out of production code
    """
    kwargs['shell'] = True
    return cmd(*args, **kwargs)


def __parse_cmd_kwargs(kwargs):
    verbose = kwargs.get('verbose', True)
    detatch = kwargs.get('detatch', False)
    shell   = kwargs.get('shell', False)
    sudo    = kwargs.get('sudo', False)
    return verbose, detatch, shell, sudo


def __parse_cmd_args(args, sudo):
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]
    if isinstance(args, (str, unicode)):
        if os.name == 'posix':
            args = shlex.split(args)
        else:
            args = [args]
    if sudo is True and not WIN32:
        args = ['sudo'] + args
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
    """ A really roundabout way to issue a system call """
    sys.stdout.flush()
    # Parse the keyword arguments
    verbose, detatch, shell, sudo = __parse_cmd_kwargs(kwargs)
    args = __parse_cmd_args(args, sudo)
    # Print what you are about to do
    print('[cplat] RUNNING: %r' % (args,))
    # Open a subprocess with a pipe
    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            shell=shell)
    if detatch:
        print('[cplat] PROCESS DETATCHING')
        return None, None, 1
    if verbose and not detatch:
        print('[cplat] RUNNING WITH VERBOSE OUTPUT')
        logged_out = []
        for line in _run_process(proc):
            sys.stdout.write(line)
            sys.stdout.flush()
            logged_out.append(line)
        out = '\n'.join(logged_out)
        (out_, err) = proc.communicate()
        #print('[cplat] out: %s' % (out,))
        print('[cplat] stdout: %s' % (out_,))
        print('[cplat] stderr: %s' % (err,))
    else:
        # Surpress output
        #print('[cplat] RUNNING WITH SUPRESSED OUTPUT')
        (out, err) = proc.communicate()
    # Make sure process if finished
    ret = proc.wait()
    print('[cplat] PROCESS FINISHED')
    return out, err, ret


def get_flops():
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
