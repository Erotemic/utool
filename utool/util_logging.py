"""
If logging is on, utool will overwrite the print function with a logging function

This is a special module which will not get injected into (should it be internal?)
"""
from __future__ import absolute_import, division, print_function
from six.moves import builtins
from os.path import exists, join, realpath
import logging
import logging.config
import multiprocessing
import os
import sys


def __inside_doctest(original_stdout=sys.stdout):
    return original_stdout != sys.stdout

__IN_MAIN_PROCESS__ = multiprocessing.current_process().name == 'MainProcess'

__UTOOL_ROOT_LOGGER__ = None

PRINT_ALL_CALLERS = '--print-all-callers' in sys.argv
VERBOSE = '--verbose' in sys.argv
VERYVERBOSE = '--veryverbose' in sys.argv

# Remeber original python values
__PYTHON_STDOUT__ = sys.stdout
__PYTHON_PRINT__  = builtins.print
__PYTHON_WRITE__  = __PYTHON_STDOUT__.write
__PYTHON_FLUSH__  = __PYTHON_STDOUT__.flush

# Initialize utool values
__UTOOL_STDOUT__    = __PYTHON_STDOUT__
__UTOOL_PRINT__     = __PYTHON_PRINT__
__UTOOL_PRINTDBG__  = __PYTHON_PRINT__
__UTOOL_WRITE__     = __PYTHON_WRITE__
__UTOOL_FLUSH__     = __PYTHON_FLUSH__
__UTOOL_WRITE_BUFFER__ = []

logdir_cacheid = 'log_dpath'


def get_logging_dir(appname='default'):
    """
    Returns:
        log_dir_realpath (str): real path to logging directory
    """
    from ._internal.meta_util_cache import global_cache_read
    from ._internal.meta_util_cplat import get_resource_dir
    from .util_cache import get_default_appname  # Hacky
    if appname is None or  appname == 'default':
        appname = get_default_appname()
    log_dir = global_cache_read(logdir_cacheid, appname=appname, default=join(get_resource_dir(), appname, 'logs'))
    log_dir_realpath = realpath(log_dir)
    return log_dir_realpath


def get_log_fpath(num='next', appname=None):
    """
    Returns:
        log_fpath (str): path to log file
    """
    log_dir = get_logging_dir(appname=appname)
    if not exists(log_dir):
        os.makedirs(log_dir)
    if appname is not None:
        log_fname = appname + '_logs_%04d.out'
    else:
        log_fname = 'utool_logs_%04d.out'
    if isinstance(num, str):
        if num == 'next':
            count = 0
            log_fpath = join(log_dir, log_fname % count)
            while exists(log_fpath):
                log_fpath = join(log_dir, log_fname % count)
                count += 1
    else:
        log_fpath = join(log_dir, log_fname % num)
    return log_fpath


def add_logging_handler(handler, format_='file'):
    """
    mostly for util_logging internals
    """
    global __UTOOL_ROOT_LOGGER__
    if __UTOOL_ROOT_LOGGER__ is None:
        builtins.print('[WARNING] logger not started, cannot add handler')
        return
    # create formatter and add it to the handlers
    #logformat = '%Y-%m-%d %H:%M:%S'
    #logformat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    timeformat = '%H:%M:%S'
    if format_ == 'file':
        logformat = '[%(asctime)s]%(message)s'
    elif format_ == 'stdout':
        logformat = '%(message)s'
    else:
        raise AssertionError('unknown logging format_: %r' % format_)
    # Create formatter for handlers
    formatter = logging.Formatter(logformat, timeformat)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    __UTOOL_ROOT_LOGGER__.addHandler(handler)


def start_logging(log_fpath=None, mode='a', appname='default'):
    """
    Overwrites utool print functions to use a logger
    """
    global __UTOOL_ROOT_LOGGER__
    global __UTOOL_PRINT__
    global __UTOOL_PRINTDBG__
    global __UTOOL_WRITE__
    global __UTOOL_FLUSH__
    if VERYVERBOSE:
        print('[utool] start_logging()')
    # FIXME: The test for doctest may not work
    if __UTOOL_ROOT_LOGGER__ is None and __IN_MAIN_PROCESS__ and not __inside_doctest():
        if VERYVERBOSE:
            print('[utool] start_logging()... rootcheck OK')
        #logging.config.dictConfig(LOGGING)
        if log_fpath is None:
            log_fpath = get_log_fpath(num='next', appname=appname)
        # Print what is about to happen
        if VERBOSE:
            startmsg = ('logging to log_fpath=%r' % log_fpath)
            __UTOOL_PRINT__(startmsg)
        # Create root logger
        __UTOOL_ROOT_LOGGER__ = logging.getLogger('root')
        __UTOOL_ROOT_LOGGER__.setLevel('DEBUG')
        # create file handler which logs even debug messages
        #fh = logging.handlers.WatchedFileHandler(log_fpath)
        logfile_handler = logging.FileHandler(log_fpath, mode=mode)
        stdout_handler = logging.StreamHandler(__UTOOL_STDOUT__)
        add_logging_handler(logfile_handler, format_='file')
        add_logging_handler(stdout_handler, format_='stdout')
        __UTOOL_ROOT_LOGGER__.propagate = False
        __UTOOL_ROOT_LOGGER__.setLevel(logging.DEBUG)
        # Overwrite utool functions with the logging functions

        def utool_flush(*args):
            global __UTOOL_WRITE_BUFFER__
            msg = ''.join(__UTOOL_WRITE_BUFFER__)
            __UTOOL_WRITE_BUFFER__ = []
            return __UTOOL_ROOT_LOGGER__.info(msg)
            #__PYTHON_FLUSH__()

        def utool_write(*args):
            global __UTOOL_WRITE_BUFFER__
            msg = ', '.join(map(str, args))
            __UTOOL_WRITE_BUFFER__.append(msg)
            if msg.endswith('\n'):
                # Flush on newline
                __UTOOL_WRITE_BUFFER__[-1] = __UTOOL_WRITE_BUFFER__[-1][:-1]
                utool_flush()

        if PRINT_ALL_CALLERS:
            def utool_print(*args):
                import utool
                __UTOOL_ROOT_LOGGER__.info('\n\n----------')
                __UTOOL_ROOT_LOGGER__.info(utool.get_caller_name(range(0, 20)))
                return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
        else:
            def utool_print(*args):
                return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
        def utool_printdbg(*args):
            return  __UTOOL_ROOT_LOGGER__.debug(', '.join(map(str, args)))
        # overwrite the utool printers
        __UTOOL_WRITE__    = utool_write
        __UTOOL_FLUSH__    = utool_flush
        __UTOOL_PRINT__    = utool_print
        __UTOOL_PRINTDBG__ = utool_printdbg
        # Test out our shiney new logger
        if VERBOSE:
            __UTOOL_PRINT__('<__LOG_START__>')
            __UTOOL_PRINT__(startmsg)


def stop_logging():
    """
    Restores utool print functions to python defaults
    """
    global __UTOOL_ROOT_LOGGER__
    global __UTOOL_PRINT__
    global __UTOOL_PRINTDBG__
    global __UTOOL_WRITE__
    global __UTOOL_FLUSH__
    if __UTOOL_ROOT_LOGGER__ is not None:
        if VERBOSE:
            __UTOOL_PRINT__('<__LOG_STOP__>')
        # Remove handlers
        for h in __UTOOL_ROOT_LOGGER__.handlers:
            __UTOOL_ROOT_LOGGER__.removeHandler(h)
        # Reset objects
        __UTOOL_ROOT_LOGGER__ = None
        __UTOOL_PRINT__    = __PYTHON_PRINT__
        __UTOOL_PRINTDBG__ = __PYTHON_PRINT__
        __UTOOL_WRITE__    = __PYTHON_WRITE__
        __UTOOL_FLUSH__    = __PYTHON_FLUSH__
