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


def startfile(fpath, detatch=True, quote=False):
    """ Uses default program defined by the system to open a file.

    References:
        http://stackoverflow.com/questions/2692873/quote-posix-shell-special-characters-in-python-output

    """
    print('[cplat] startfile(%r)' % fpath)
    if not exists(fpath):
        raise Exception('Cannot start nonexistant file: %r' % fpath)
    #if quote:
    #    fpath = '"%s"' % (fpath,)
    import pipes
    fpath = pipes.quote(fpath)
    if LINUX:
        #out, err, ret = cmd(['xdg-open', fpath], detatch=True)
        outtup = cmd(('xdg-open', fpath), detatch=detatch)
        #outtup = cmd('xdg-open', fpath, detatch=detatch)
    elif DARWIN:
        outtup = cmd(('open', fpath), detatch=detatch)
    else:
        os.startfile(fpath)
    if outtup is not None:
        out, err, ret = outtup
        if not ret:
            raise Exception(out + ' -- ' + err)
    pass


def geteditor():
    return 'gvim'


def editfile(fpath):
    """ Runs gvim """
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
    # FIXME: Turn shell off by default and fix __parse_cmd_args
    shell   = kwargs.get('shell', LINUX or DARWIN)
    sudo    = kwargs.get('sudo', False)
    # pads stdout of cmd before and after
    # TODO: rename separate to something else
    pad_stdout    = kwargs.get('pad_stdout', True)
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
        # When shell is False, ensure args is a string
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
    r""" A really roundabout way to issue a system call

    # FIXME: This function needs some work
    # It should work without a hitch on windows or unix.
    # It should be able to spit out stdout in realtime.
    # Should be able to configure detatchment, shell, and sudo.

    Returns:
        tuple: (None, None, None)

    CommandLine:
        python -m utool.util_cplat --test-cmd:0
        python -m utool.util_cplat --test-cmd:1
        python -m utool.util_cplat --test-cmd:1 --test-sudo
        python -m utool.util_cplat --test-cmd:2 --test-sudo

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> (out, err, ret) = ut.cmd('echo', 'hello world')
        >>> result = ut.list_str(list(zip(('out', 'err', 'ret'), (out, err, ret))), nobraces=True)
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
        >>>     print('testing cmd with params ' + ut.dict_str(kw))
        >>>     args = kw.pop('args')
        >>>     restup = ut.cmd(args, pad_stdout=False, **kw)
        >>>     tupfields = ('out', 'err', 'ret')
        >>>     output = ut.list_str(list(zip(tupfields, restup)), nobraces=True)
        >>>     ut.assert_eq(output, target)
        >>>     print('L ___ TEST CMD %d ___\n' % (count,))

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> import utool as ut
        >>> target = ut.codeblock(
        ...     r'''
                ('out', 'hello world\n'),
                ('err', None),
                ('ret', 0),
                ''')
        >>> varydict = {
        ...    'shell': [True, False],
        ...    'detatch': [True],
        ...    'args': ['ping localhost', ('ping', 'localhost')],
        ... }
        >>> proc_list = []
        >>> for count, kw in enumerate(ut.all_dict_combinations(varydict), start=1):
        >>>     print('+ --- TEST CMD %d ---' % (count,))
        >>>     print('testing cmd with params ' + ut.dict_str(kw))
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
        if pad_stdout:
            sys.stdout.flush()
            print('\n+--------')
        args = __parse_cmd_args(args, sudo, shell)
        # Print what you are about to do
        print('[ut.cmd] RUNNING: %r' % (args,))
        # Open a subprocess with a pipe
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, shell=shell,
                                universal_newlines=True)
        if detatch:
            print('[ut.cmd] PROCESS DETATCHING. No stdoutput can be reported...')
            # There is no immediate confirmation as to whether or not the script
            # finished. It might still be running for all you know
            return None, None, proc
        else:
            if verbose and not detatch:
                print('[ut.cmd] RUNNING WITH VERBOSE OUTPUT')
                logged_out = []
                for line in _run_process(proc):
                    line_ = line if six.PY2 else line.decode('utf-8')
                    if len(line_) > 0:
                        sys.stdout.write(line_)
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


def print_path(sort=True):
    pathdirs = get_path_dirs()
    if sort:
        pathdirs = sorted(pathdirs)
    print('\n'.join(pathdirs))


def search_env_paths(fname):
    r"""
    Args:
        fname (?):

    CommandLine:
        python -m utool.util_cplat --test-search_path

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> # build test data
        >>> fname = 'opencv2/highgui/libopencv_highgui.so'
        >>> # execute function
        >>> result = search_env_paths(fname)
        >>> # verify results
        >>> print(result)

    Dev:
        OpenCV_DIR:PATH={share_opencv}
        OpenCV_CONFIG_PATH:FILEPATH={share_opencv}

    """
    key_list = [key for key in os.environ if key.find('PATH') > -1]

    import utool as ut
    from os.path import join
    for key in key_list:
        dpath_list = os.environ[key].split(os.pathsep)
        for dpath in dpath_list:
            testname = join(dpath, fname)
            if ut.checkpath(testname, verbose=False):
                print('Found in key=%r' % (key,))
                ut.checkpath(testname, verbose=True, info=True)


def change_term_title(title):
    """
    only works on unix systems only tested on Ubuntu GNOME changes text on
    terminal title for identifying debugging tasks.

    The title will remain until python exists

    Args:
        title (str):

    CommandLine:
        python -m utool.util_cplat --test-change_term_title

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> # build test data
        >>> title = 'change title test'
        >>> # execute function
        >>> result = change_term_title(title)
        >>> # verify results
        >>> print(result)
    """
    if not WIN32:
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
        >>> # build test data
        >>> text = '%paste'
        >>> # execute function
        >>> result = send_keyboard_input('%paste')
        >>> # verify results
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
            #if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                #hwnds[win32gui.GetClassName(hwnd)] = hwnd
            #return True
        #hwnds = {}
        #win32gui.EnumChildWindows(hwnd, callback, hwnds)

        #for ord_char in map(ord, text):
            #win32api.SendMessage(hwnd, win32con.WM_CHAR, ord_char, 0)
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
        >>> from utool.util_cplat import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> result = print_system_users()
        >>> # verify results
        >>> print(result)
    """
    import utool as ut
    text = ut.read_from('/etc/passwd')
    userinfo_text_list = text.splitlines()
    userinfo_list = [uitext.split(':') for uitext in userinfo_text_list]
    #print(ut.list_str(sorted(userinfo_list)))
    bash_users = [tup for tup in userinfo_list if tup[-1] == '/bin/bash']
    print(ut.list_str(sorted(bash_users)))


def unload_module(modname):
    """
    WARNING POTENTIALLY DANGEROUS AND MAY NOT WORK

    References:
        http://stackoverflow.com/questions/437589/how-do-i-unload-reload-a-python-module

    CommandLine:
        python -m utool.util_cplat --test-unload_module

    Example:
        >>> import sys, gc  # NOQA
        >>> import pyhesaff
        >>> import utool as ut
        >>> modname = 'pyhesaff'
        >>> print('%s refcount=%r' % (modname, sys.getrefcount(pyhesaff),))
        >>> #referrer_list = gc.get_referrers(sys.modules[modname])
        >>> #print('referrer_list = %s' % (ut.list_str(referrer_list),))
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
