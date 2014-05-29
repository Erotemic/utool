"""
If logging is on, utool will overwrite the print function with a logging function
Yay __future__ print_function. Lets try to get to python3 quickly!
Unrelated, but important reminder: Don't Panic

This is a special module which will not get injected into (should it be internal?)
"""
from __future__ import absolute_import, division, print_function
import __builtin__
from os.path import exists, join, realpath
import logging
import logging.config
import multiprocessing
import os
import sys

__IN_MAIN_PROCESS__ = multiprocessing.current_process().name == 'MainProcess'

__UTOOL_ROOT_LOGGER__ = None

# Remeber original python values
__PYTHON_STDOUT__ = sys.stdout
__PYTHON_PRINT__  = __builtin__.print
__PYTHON_WRITE__  = __PYTHON_STDOUT__.write
__PYTHON_FLUSH__  = __PYTHON_STDOUT__.flush

# Initialize utool values
__UTOOL_STDOUT__    = __PYTHON_STDOUT__
__UTOOL_PRINT__     = __PYTHON_PRINT__
__UTOOL_PRINTDBG__  = __PYTHON_PRINT__
__UTOOL_WRITE__     = __PYTHON_WRITE__
__UTOOL_FLUSH__     = __PYTHON_FLUSH__


def get_logging_dir():
    log_dir = 'logs'
    return realpath(log_dir)


def get_log_fpath(num='next'):
    log_dir = get_logging_dir()
    if not exists(log_dir):
        os.makedirs(log_dir)
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
    global __UTOOL_ROOT_LOGGER__
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


def start_logging(log_fpath=None, mode='a'):
    """
    Overwrites utool print functions to use a logger
    """
    global __UTOOL_ROOT_LOGGER__
    global __UTOOL_PRINT__
    global __UTOOL_PRINTDBG__
    global __UTOOL_WRITE__
    if __UTOOL_ROOT_LOGGER__ is None and __IN_MAIN_PROCESS__:
        #logging.config.dictConfig(LOGGING)
        if log_fpath is None:
            log_fpath = get_log_fpath(num='next')
        # Print what is about to happen
        msg = ('logging to log_fpath=%r' % log_fpath)
        __UTOOL_PRINT__(msg)
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
        __UTOOL_WRITE__    = lambda msg: __UTOOL_ROOT_LOGGER__.info(msg)
        __UTOOL_PRINT__    = lambda msg: __UTOOL_ROOT_LOGGER__.info(msg)
        __UTOOL_PRINTDBG__ = lambda msg: __UTOOL_ROOT_LOGGER__.debug(msg)
        # Test out our shiney new logger
        __UTOOL_PRINT__('<__LOG_START__>')
        __UTOOL_PRINT__(msg)


def stop_logging():
    """
    Restores utool print functions to python defaults
    """
    global __UTOOL_ROOT_LOGGER__
    global __UTOOL_PRINT__
    global __UTOOL_PRINTDBG__
    global __UTOOL_WRITE__
    if __UTOOL_ROOT_LOGGER__ is not None:
        __UTOOL_PRINT__('<__LOG_STOP__>')
        # Remove handlers
        for h in __UTOOL_ROOT_LOGGER__.handlers:
            __UTOOL_ROOT_LOGGER__.removeHandler(h)
        # Reset objects
        __UTOOL_ROOT_LOGGER__ = None
        __UTOOL_PRINT__    = __PYTHON_PRINT__
        __UTOOL_PRINTDBG__ = __PYTHON_PRINT__
        __UTOOL_WRITE__    = __PYTHON_WRITE__
