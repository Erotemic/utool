"""
If logging is on, utool will overwrite the print function with a logging function

This is a special module which will not get injected into (should it be internal?)

References:
    # maybe we can do something like this Queue to try fixing error when
    # when using injected print statments with Qt signals and slots
    http://stackoverflow.com/questions/21071448/redirecting-stdout-and-stderr-to-a-pyqt4-qtextedit-from-a-secondary-thread
"""
from __future__ import absolute_import, division, print_function
from six.moves import builtins
from os.path import exists, join, realpath
import logging
import logging.config
import multiprocessing
import os
import sys
from utool._internal import meta_util_arg

VERBOSE            = meta_util_arg.VERBOSE
VERYVERBOSE        = meta_util_arg.VERYVERBOSE
PRINT_ALL_CALLERS  = meta_util_arg.PRINT_ALL_CALLERS
LOGGING_VERBOSE    = meta_util_arg.LOGGING_VERBOSE
PRINT_INJECT_ORDER = meta_util_arg.PRINT_INJECT_ORDER


def __inside_doctest(original_stdout=sys.stdout):
    return original_stdout != sys.stdout


__IN_MAIN_PROCESS__ = multiprocessing.current_process().name == 'MainProcess'

__UTOOL_ROOT_LOGGER__ = None
__CURRENT_LOG_FPATH__ = None

# Remeber original python values
__PYTHON_STDOUT__ = sys.stdout
__PYTHON_PRINT__  = builtins.print
__PYTHON_WRITE__  = __PYTHON_STDOUT__.write
__PYTHON_FLUSH__  = __PYTHON_STDOUT__.flush

# Initialize utool values
__UTOOL_STDOUT__    = __PYTHON_STDOUT__
__UTOOL_PRINT__     = __PYTHON_PRINT__
__UTOOL_PRINTDBG__  = __PYTHON_PRINT__

# TODO: Allow write and flush to have a logging equivalent
__UTOOL_WRITE__     = __PYTHON_WRITE__
__UTOOL_FLUSH__     = __PYTHON_FLUSH__
__UTOOL_WRITE_BUFFER__ = []

logdir_cacheid = 'log_dpath'


def get_logging_dir(appname='default'):
    """
    The default log dir is in the system resource directory
    But the utool global cache allows for the user to override
    where the logs for a specific app should be stored.

    Returns:
        log_dir_realpath (str): real path to logging directory
    """
    from utool._internal.meta_util_cache import global_cache_read
    from utool._internal.meta_util_cplat import get_resource_dir
    from utool.util_cache import get_default_appname  # Hacky
    if appname is None or  appname == 'default':
        appname = get_default_appname()
    resource_dpath = get_resource_dir()
    default = join(resource_dpath, appname, 'logs')
    # Check global cache for a custom logging dir otherwise
    # use the default.
    log_dir = global_cache_read(logdir_cacheid, appname=appname,
                                default=default)
    log_dir_realpath = realpath(log_dir)
    return log_dir_realpath


def get_current_log_fpath():
    global __CURRENT_LOG_FPATH__
    return __CURRENT_LOG_FPATH__


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
    r"""
    Overwrites utool print functions to use a logger

    Example0:
        >>> # DISABLE_DOCTEST
        >>> import sys
        >>> sys.argv.append('--verb-logging')
        >>> import utool as ut
        >>> ut.start_logging()
        >>> ut.util_logging.__UTOOL_PRINT__('hello world')
        >>> ut.util_logging.__UTOOL_WRITE__('writing1')
        >>> ut.util_logging.__UTOOL_WRITE__('writing2\n')
        >>> ut.util_logging.__UTOOL_WRITE__('writing3')
        >>> ut.util_logging.__UTOOL_FLUSH__()
        >>> handler = ut.util_logging.__UTOOL_ROOT_LOGGER__.handlers[0]
        >>> current_log_fpath = handler.stream.name
        >>> current_log_text = ut.read_from(current_log_fpath)
        >>> assert current_log_text.find('hello world') > 0
        >>> assert current_log_text.find('writing1writing2') > 0
        >>> assert current_log_text.find('writing3') > 0

    Example1:
        >>> # DISABLE_DOCTEST
        >>> # Ensure that progress is logged
        >>> import sys
        >>> sys.argv.append('--verb-logging')
        >>> import utool as ut
        >>> ut.start_logging()
        >>> [x for x in  ut.ProgressIter(range(0, 1000), freq=4)]
        >>> handler = ut.util_logging.__UTOOL_ROOT_LOGGER__.handlers[0]
        >>> current_log_fpath = handler.stream.name
        >>> current_log_text = ut.read_from(current_log_fpath)
        >>> assert current_log_text.find('rate') > 0, 'progress was not logged'
        >>> print(current_log_text)
    """
    global __UTOOL_ROOT_LOGGER__
    global __UTOOL_PRINT__
    global __UTOOL_PRINTDBG__
    global __UTOOL_WRITE__
    global __UTOOL_FLUSH__
    global __CURRENT_LOG_FPATH__
    if LOGGING_VERBOSE:
        print('[utool] start_logging()')
    # FIXME: The test for doctest may not work
    if __UTOOL_ROOT_LOGGER__ is None and __IN_MAIN_PROCESS__ and not __inside_doctest():
        if LOGGING_VERBOSE:
            print('[utool] start_logging()... rootcheck OK')
        #logging.config.dictConfig(LOGGING)
        if log_fpath is None:
            log_fpath = get_log_fpath(num='next', appname=appname)
        __CURRENT_LOG_FPATH__ = log_fpath
        # Print what is about to happen
        if VERBOSE or LOGGING_VERBOSE:
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
            """ flushes whatever is in the current utool write buffer """
            global __UTOOL_WRITE_BUFFER__
            if len(__UTOOL_WRITE_BUFFER__) > 0:
                msg = ''.join(__UTOOL_WRITE_BUFFER__)
                __UTOOL_WRITE_BUFFER__ = []
                return __UTOOL_ROOT_LOGGER__.info(msg)
            #__PYTHON_FLUSH__()

        def utool_write(*args):
            """ writes to current utool logs and to sys.stdout.write """
            global __UTOOL_WRITE_BUFFER__
            msg = ', '.join(map(str, args))
            __UTOOL_WRITE_BUFFER__.append(msg)
            if msg.endswith('\n'):
                # Flush on newline
                __UTOOL_WRITE_BUFFER__[-1] = __UTOOL_WRITE_BUFFER__[-1][:-1]
                utool_flush()

        if PRINT_ALL_CALLERS:
            def utool_print(*args):
                """ debugging utool print function """
                import utool as ut
                utool_flush()
                __UTOOL_ROOT_LOGGER__.info('\n\n----------')
                __UTOOL_ROOT_LOGGER__.info(ut.get_caller_name(range(0, 20)))
                return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
        else:
            def utool_print(*args):
                """ standard utool print function """
                return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
        def utool_printdbg(*args):
            """ standard utool print debug function """
            return  __UTOOL_ROOT_LOGGER__.debug(', '.join(map(str, args)))
        # overwrite the utool printers
        __UTOOL_WRITE__    = utool_write
        __UTOOL_FLUSH__    = utool_flush
        __UTOOL_PRINT__    = utool_print
        __UTOOL_PRINTDBG__ = utool_printdbg
        # Test out our shiney new logger
        if VERBOSE or LOGGING_VERBOSE:
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
        if VERBOSE or LOGGING_VERBOSE:
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

# HAVE TO HACK THIS IN FOR UTOOL SPECIAL CASE ONLY
# OTHER MODULE CAN USE NOINJECT
if PRINT_INJECT_ORDER:
    from utool._internal import meta_util_dbg
    callername = meta_util_dbg.get_caller_name(N=1, strict=False)
    fmtdict = dict(callername=callername, modname='utool.util_logging')
    msg = '[util_inject] {modname} is imported by {callername}'.format(**fmtdict)
    builtins.print(msg)


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_logging
        python -m utool.util_logging --allexamples
        python -m utool.util_logging --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
