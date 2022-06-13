# -*- coding: utf-8 -*-
"""
If logging is on, utool will overwrite the print function with a logging function

This is a special module which will not get injected into (should it be internal?)

References:
    # maybe we can do something like this Queue to try fixing error when
    # when using injected print statments with Qt signals and slots
    http://stackoverflow.com/questions/21071448/redirecting-stdout-and-stderr-to-a-pyqt4-qtextedit-from-a-secondary-thread
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
from six.moves import builtins, map, zip, range  # NOQA
from os.path import exists, join, realpath
import logging
import logging.config
import multiprocessing
import os
import sys
from utool._internal import meta_util_arg, meta_util_six

VERBOSE            = meta_util_arg.VERBOSE
VERYVERBOSE        = meta_util_arg.VERYVERBOSE
PRINT_ALL_CALLERS  = meta_util_arg.PRINT_ALL_CALLERS
LOGGING_VERBOSE    = meta_util_arg.LOGGING_VERBOSE  # --verb-logging
PRINT_INJECT_ORDER = meta_util_arg.PRINT_INJECT_ORDER


def __inside_doctest(original_stdout=sys.stdout):
    return original_stdout != sys.stdout


__IN_MAIN_PROCESS__ = multiprocessing.current_process().name == 'MainProcess'

__UTOOL_ROOT_LOGGER__ = None
__CURRENT_LOG_FPATH__ = None

# Remeber original python values
# __PYTHON_STDOUT__ = sys.stdout
# __PYTHON_PRINT__  = builtins.print
# __PYTHON_WRITE__  = __PYTHON_STDOUT__.write
# __PYTHON_FLUSH__  = __PYTHON_STDOUT__.flush

# Initialize utool values
__UTOOL_STDOUT__    = None
__UTOOL_PRINT__     = None

# TODO: Allow write and flush to have a logging equivalent
__UTOOL_WRITE__     = None
__UTOOL_FLUSH__     = None
__UTOOL_WRITE_BUFFER__ = []


def _utool_stdout():
    if __UTOOL_STDOUT__ is not None:
        return __UTOOL_STDOUT__
    else:
        return sys.stdout


def _utool_write():
    if __UTOOL_WRITE__ is not None:
        return __UTOOL_WRITE__
    else:
        return sys.stdout.write


def _utool_flush():
    if __UTOOL_FLUSH__ is not None:
        return __UTOOL_FLUSH__
    else:
        return sys.stdout.flush


def _utool_print():
    if __UTOOL_PRINT__ is not None:
        return __UTOOL_PRINT__
    else:
        return builtins.print


__STR__ = six.text_type

logdir_cacheid = 'log_dpath'


def testlogprog():
    r"""
    Test to ensure that all progress lines are outputed to the file logger
    while only a few progress lines are outputed to stdout.
    (if backspace is specified)

    CommandLine:
        python -m utool.util_logging testlogprog --show --verb-logging
        python -m utool.util_logging testlogprog --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_logging import *  # NOQA
        >>> import utool as ut
        >>> result = testlogprog()
        >>> print(result)
    """
    import utool as ut
    print('Starting test log function')

    def test_body(count, logmode, backspace):
        ut.colorprint('\n---- count = %r -----' % (count,), 'yellow')
        ut.colorprint('backspace = %r' % (backspace,), 'yellow')
        ut.colorprint('logmode = %r' % (logmode,), 'yellow')
        if logmode:
            ut.delete('test.log')
            ut.start_logging('test.log')
        print('Start main loop')
        import time
        for count in ut.ProgressIter(range(20), freq=3, backspace=backspace):
            time.sleep(.01)
        print('Done with main loop work')
        print('Exiting main body')
        if logmode:
            ut.stop_logging()
            #print('-----DONE LOGGING----')
            testlog_text = ut.readfrom('test.log')
            print(ut.indent(testlog_text.replace('\r', '\n'), '        '))

    def test_body2(count, logmode, backspace):
        ut.colorprint('\n---- count = %r -----' % (count,), 'yellow')
        ut.colorprint('backspace = %r' % (backspace,), 'yellow')
        ut.colorprint('logmode = %r' % (logmode,), 'yellow')
        if logmode:
            ut.delete('test.log')
            ut.start_logging('test.log')
        print('Start main loop')
        import time
        for count in ut.ProgressIter(range(2), freq=1, backspace=backspace):
            for count in ut.ProgressIter(range(50), freq=1, backspace=backspace):
                time.sleep(.01)
        print('Done with main loop work')
        print('Exiting main body')
        if logmode:
            ut.stop_logging()
            #print('-----DONE LOGGING----')
            #testlog_text = ut.readfrom('test.log')
            #print(ut.indent(testlog_text.replace('\r', '\n'), '        '))

    #test_body(0, False, True)
    #test_body(1, False, False)
    #test_body(2, True, True)
    #test_body(3, True, False)

    test_body2(4, True, True)
    test_body2(5, False, True)


def ensure_logging():
    flag = is_logging()
    if not flag:
        start_logging()
    return flag


def is_logging():
    global __UTOOL_ROOT_LOGGER__
    flag = __UTOOL_ROOT_LOGGER__ is not None
    return flag


def debug_logging_iostreams():
    print(' --- <DEBUG IOSTREAMS> --')
    print('__STR__ = %r' % (__STR__,))
    print('__IN_MAIN_PROCESS__ = %r' % (__IN_MAIN_PROCESS__,))
    print('__UTOOL_ROOT_LOGGER__ = %r' % (__UTOOL_ROOT_LOGGER__,))
    print('__CURRENT_LOG_FPATH__ = %r' % (__CURRENT_LOG_FPATH__,))
    # print('__PYTHON_STDOUT__ = %r' % (__PYTHON_STDOUT__,))
    # print('__PYTHON_PRINT__ = %r' % (__PYTHON_PRINT__,))
    # print('__PYTHON_WRITE__ = %r' % (__PYTHON_WRITE__,))
    # print('__PYTHON_FLUSH__ = %r' % (__PYTHON_FLUSH__,))
    print('__UTOOL_STDOUT__ = %r' % (__UTOOL_STDOUT__,))
    print('__UTOOL_PRINT__ = %r' % (__UTOOL_PRINT__,))
    print('__UTOOL_FLUSH__ = %r' % (__UTOOL_FLUSH__,))
    print('__UTOOL_WRITE__ = %r' % (__UTOOL_WRITE__,))
    print(' --- </DEBUG IOSTREAMS> --')


def get_logging_dir(appname='default'):
    """
    The default log dir is in the system resource directory
    But the utool global cache allows for the user to override
    where the logs for a specific app should be stored.

    Returns:
        log_dir_realpath (str): real path to logging directory
    """
    from utool._internal import meta_util_cache
    from utool._internal import meta_util_cplat
    from utool import util_cache
    if appname is None or  appname == 'default':
        appname = util_cache.get_default_appname()
    resource_dpath = meta_util_cplat.get_resource_dir()
    default = join(resource_dpath, appname, 'logs')
    # Check global cache for a custom logging dir otherwise
    # use the default.
    log_dir = meta_util_cache.global_cache_read(logdir_cacheid,
                                                appname=appname,
                                                default=default)
    log_dir_realpath = realpath(log_dir)
    return log_dir_realpath


def get_shelves_dir(appname='default'):
    """
    The default shelf dir is in the system resource directory
    But the utool global cache allows for the user to override
    where the shelf for a specific app should be stored.

    Returns:
        log_dir_realpath (str): real path to shelves directory
    """
    from utool._internal import meta_util_cache
    from utool._internal import meta_util_cplat
    from utool import util_cache
    if appname is None or  appname == 'default':
        appname = util_cache.get_default_appname()
    resource_dpath = meta_util_cplat.get_resource_dir()
    default = join(resource_dpath, appname, 'shelves')
    # Check global cache for a custom logging dir otherwise
    # use the default.
    log_dir = meta_util_cache.global_cache_read(logdir_cacheid,
                                                appname=appname,
                                                default=default)
    log_dir_realpath = realpath(log_dir)
    return log_dir_realpath


def get_current_log_fpath():
    global __CURRENT_LOG_FPATH__
    return __CURRENT_LOG_FPATH__


def get_current_log_text():
    fpath = get_current_log_fpath()
    if fpath is None:
        text = None
    else:
        with open(fpath, 'r') as file_:
            text = file_.read()
    return text


def get_log_fpath(num='next', appname=None, log_dir=None):
    """
    Returns:
        log_fpath (str): path to log file
    """
    if log_dir is None:
        log_dir = get_logging_dir(appname=appname)
    if not exists(log_dir):
        os.makedirs(log_dir)
    if appname is not None:
        log_fname = appname + '_logs_%04d.out'
    else:
        log_fname = 'utool_logs_%04d.out'
    if isinstance(num, six.string_types):
        if num == 'next':
            count = 0
            log_fpath = join(log_dir, log_fname % count)
            while exists(log_fpath):
                log_fpath = join(log_dir, log_fname % count)
                count += 1
    else:
        log_fpath = join(log_dir, log_fname % num)
    return log_fpath


def get_utool_logger():
    return __UTOOL_ROOT_LOGGER__


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


class CustomStreamHandler(logging.Handler):
    """
    Modified from logging.py
    """

    def __init__(self, stream=None):
        """
        Initialize the handler.

        If stream is not specified, sys.stderr is used.
        """
        self.terminator = "\n"
        logging.Handler.__init__(self)
        if stream is None:
            stream = sys.stderr
        self.stream = stream

    def flush(self):
        """
        Flushes the stream.
        """
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            fs = "%s%s"
            if six.PY3 or not logging._unicode:  # if no unicode support...
                stream.write(fs % (msg, self.terminator))
            else:
                try:
                    if (isinstance(msg, six.text_type) and getattr(stream, 'encoding', None)):
                        ufs = u'%s%s'
                        try:
                            stream.write(ufs % (msg, self.terminator))
                        except UnicodeEncodeError:
                            #Printing to terminals sometimes fails. For example,
                            #with an encoding of 'cp1251', the above write will
                            #work if written to a stream opened or wrapped by
                            #the codecs module, but fail when writing to a
                            #terminal even when the codepage is set to cp1251.
                            #An extra encoding step seems to be needed.
                            stream.write((ufs % (msg, self.terminator)).encode(stream.encoding))
                    else:
                        stream.write(fs % (msg, self.terminator))
                except UnicodeError:
                    stream.write(fs % (msg.encode("UTF-8"), self.terminator.encode("UTF-8")))
            #self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def start_logging(log_fpath=None, mode='a', appname='default', log_dir=None):
    r"""
    Overwrites utool print functions to use a logger

    CommandLine:
        python -m utool.util_logging --test-start_logging:0
        python -m utool.util_logging --test-start_logging:1

    Example0:
        >>> # DISABLE_DOCTEST
        >>> import sys
        >>> sys.argv.append('--verb-logging')
        >>> import utool as ut
        >>> ut.start_logging()
        >>> ut.util_logging._utool_print()('hello world')
        >>> ut.util_logging._utool_write()('writing1')
        >>> ut.util_logging._utool_write()('writing2\n')
        >>> ut.util_logging._utool_write()('writing3')
        >>> ut.util_logging._utool_flush()()
        >>> handler = ut.util_logging.__UTOOL_ROOT_LOGGER__.handlers[0]
        >>> current_log_fpath = handler.stream.name
        >>> current_log_text = ut.read_from(current_log_fpath)
        >>> print('current_log_text =\n%s' % (current_log_text,))
        >>> assert current_log_text.find('hello world') > 0, 'cant hello world'
        >>> assert current_log_text.find('writing1writing2') > 0, 'cant find writing1writing2'
        >>> assert current_log_text.find('writing3') > 0, 'cant find writing3'

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
            log_fpath = get_log_fpath(num='next', appname=appname, log_dir=log_dir)
        __CURRENT_LOG_FPATH__ = log_fpath
        # Print what is about to happen
        if VERBOSE or LOGGING_VERBOSE:
            startmsg = ('logging to log_fpath=%r' % log_fpath)
            _utool_print()(startmsg)
        # Create root logger
        __UTOOL_ROOT_LOGGER__ = logging.getLogger('root')
        __UTOOL_ROOT_LOGGER__.setLevel('DEBUG')
        # create file handler which logs even debug messages
        #fh = logging.handlers.WatchedFileHandler(log_fpath)
        logfile_handler = logging.FileHandler(log_fpath, mode=mode)
        #stdout_handler = logging.StreamHandler(__UTOOL_STDOUT__)
        stdout_handler = CustomStreamHandler(__UTOOL_STDOUT__)
        stdout_handler.terminator = ''
        # http://stackoverflow.com/questions/7168790/suppress-newline-in-python-logging-module
        #stdout_handler.terminator = ''
        add_logging_handler(logfile_handler, format_='file')
        add_logging_handler(stdout_handler, format_='stdout')
        __UTOOL_ROOT_LOGGER__.propagate = False
        __UTOOL_ROOT_LOGGER__.setLevel(logging.DEBUG)
        # Overwrite utool functions with the logging functions

        def utool_flush(*args):
            """ flushes whatever is in the current utool write buffer """
            # Flushes only the stdout handler
            stdout_handler.flush()
            #__UTOOL_ROOT_LOGGER__.flush()
            #global __UTOOL_WRITE_BUFFER__
            #if len(__UTOOL_WRITE_BUFFER__) > 0:
            #    msg = ''.join(__UTOOL_WRITE_BUFFER__)
            #    #sys.stdout.write('FLUSHING %r\n' % (len(__UTOOL_WRITE_BUFFER__)))
            #    __UTOOL_WRITE_BUFFER__ = []
            #    return __UTOOL_ROOT_LOGGER__.info(msg)
            #__PYTHON_FLUSH__()

        def utool_write(*args):
            """ writes to current utool logs and to sys.stdout.write """
            #global __UTOOL_WRITE_BUFFER__
            #sys.stdout.write('WRITEING\n')
            msg = ', '.join(map(six.text_type, args))
            #__UTOOL_WRITE_BUFFER__.append(msg)
            __UTOOL_ROOT_LOGGER__.info(msg)
            #if msg.endswith('\n'):
            #    # Flush on newline, and remove newline
            #    __UTOOL_WRITE_BUFFER__[-1] = __UTOOL_WRITE_BUFFER__[-1][:-1]
            #    utool_flush()
            #elif len(__UTOOL_WRITE_BUFFER__) > 32:
            #    # Flush if buffer is too large
            #    utool_flush()

        if not PRINT_ALL_CALLERS:
            def utool_print(*args):
                """ standard utool print function """
                #sys.stdout.write('PRINT\n')
                endline = '\n'
                try:
                    msg = ', '.join(map(six.text_type, args))
                    return  __UTOOL_ROOT_LOGGER__.info(msg + endline)
                except UnicodeDecodeError:
                    new_msg = ', '.join(map(meta_util_six.ensure_unicode, args))
                    #print(new_msg)
                    return  __UTOOL_ROOT_LOGGER__.info(new_msg + endline)
        else:
            def utool_print(*args):
                """ debugging utool print function """
                import utool as ut
                utool_flush()
                endline = '\n'
                __UTOOL_ROOT_LOGGER__.info('\n\n----------')
                __UTOOL_ROOT_LOGGER__.info(ut.get_caller_name(range(0, 20)))
                return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(six.text_type, args)) + endline)

        def utool_printdbg(*args):
            """ DRPRICATE standard utool print debug function """
            return  __UTOOL_ROOT_LOGGER__.debug(', '.join(map(six.text_type, args)))
        # overwrite the utool printers
        __UTOOL_WRITE__    = utool_write
        __UTOOL_FLUSH__    = utool_flush
        __UTOOL_PRINT__    = utool_print
        # Test out our shiney new logger
        if VERBOSE or LOGGING_VERBOSE:
            __UTOOL_PRINT__('<__LOG_START__>')
            __UTOOL_PRINT__(startmsg)
    else:
        if LOGGING_VERBOSE:
            print('[utool] start_logging()... FAILED TO START')
            print('DEBUG INFO')
            print('__inside_doctest() = %r' % (__inside_doctest(),))
            print('__IN_MAIN_PROCESS__ = %r' % (__IN_MAIN_PROCESS__,))
            print('__UTOOL_ROOT_LOGGER__ = %r' % (__UTOOL_ROOT_LOGGER__,))


def stop_logging():
    """
    Restores utool print functions to python defaults
    """
    global __UTOOL_ROOT_LOGGER__
    global __UTOOL_PRINT__
    global __UTOOL_WRITE__
    global __UTOOL_FLUSH__
    if __UTOOL_ROOT_LOGGER__ is not None:
        # Flush remaining buffer
        if VERBOSE or LOGGING_VERBOSE:
            _utool_print()()('<__LOG_STOP__>')
        _utool_flush()()
        # Remove handlers
        for h in __UTOOL_ROOT_LOGGER__.handlers[:]:
            __UTOOL_ROOT_LOGGER__.removeHandler(h)
        # Reset objects
        __UTOOL_ROOT_LOGGER__ = None
        __UTOOL_PRINT__    = None
        __UTOOL_WRITE__    = None
        __UTOOL_FLUSH__    = None

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
