# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from loguru import logger
try:
    import numpy as np
except ImportError:
    pass
import six
import sys
from contextvars import ContextVar
from utool._internal import meta_util_arg
from utool import util_str

QUIET        = meta_util_arg.QUIET
VERBOSE      = meta_util_arg.VERBOSE
NO_INDENT    = meta_util_arg.NO_INDENT
SILENT       = meta_util_arg.SILENT


def print_dict(dict_, dict_name=None, **kwargs):
    import utool as ut
    if dict_name is None:
        dict_name = ut.get_varname_from_stack(dict_, N=1)
    dict_repr = util_str.repr4(dict_, **kwargs)
    logger.info(dict_name + ' = ' + dict_repr)

printdict = print_dict
dictprint = print_dict


def print_list(list_, **kwargs):
    import utool as ut
    list_name = ut.get_varname_from_stack(list_, N=1)
    logger.info(list_name + ' = ' + util_str.repr4(list_, **kwargs))


def horiz_print(*args):
    toprint = util_str.horiz_string(args)
    logger.info(toprint)


#def set_indenting_enabled(flag):
#    global NO_INDENT
#    prev_flag = NO_INDENT
#    NO_INDENT = not flag
#    return prev_flag


_LOG_INDENT = ContextVar('utool_log_indent', default='')


def _test_indent_print():
    """Exercise nested indentation without configuring application logging."""
    messages = []

    def capture(message):
        prefix = message.record['extra'].get('utool_indent', '')
        messages.append(prefix + message.record['message'])

    sink_id = logger.add(capture, format='{message}')
    try:
        logger.info('Checking indent. Should have none')
        with Indenter('[INDENT] '):
            logger.info('Checking indent. Should be indented')
        logger.info('Should no longer be indented')
    finally:
        logger.remove(sink_id)

    assert '[INDENT] ' not in messages[0]
    assert messages[1].startswith('[INDENT] ')
    assert '[INDENT] ' not in messages[2]


class Indenter(object):
    """Attach a contextual prefix to Loguru records emitted in this scope.

    IBEIS includes ``utool_indent`` in its sink formatter, so existing
    ``ut.Indenter`` and ``ut.indent_func`` call sites retain their diagnostic
    grouping without monkey-patching module globals.
    """

    def __init__(self, lbl='    ', enabled=True):
        self.enabled = enabled
        self.lbl = lbl
        self._token = None
        self._context = None

    def start(self):
        if NO_INDENT or not self.enabled:
            return logger
        prefix = _LOG_INDENT.get() + self.lbl
        self._token = _LOG_INDENT.set(prefix)
        self._context = logger.contextualize(utool_indent=prefix)
        self._context.__enter__()
        return logger

    def stop(self):
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None
        if self._token is not None:
            _LOG_INDENT.reset(self._token)
            self._token = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type_, value, trace):
        self.stop()
        if trace is not None:
            if VERBOSE:
                logger.error('[util_print] Error in print context manager!: {}', value)
            return False


def printshape(arr_name, locals_):
    arr = locals_[arr_name]
    if isinstance(arr, np.ndarray):
        logger.info(arr_name + '.shape = ' + str(arr.shape))
    else:
        logger.info('len(%s) = %r' % (arr_name, len(arr)))


#class NpPrintOpts(object):
#    def __init__(self, **kwargs):
#        self.orig_opts = np.get_printoptions()
#        self.new_opts = kwargs
#    def __enter__(self):
#        np.set_printoptions(**self.new_opts)
#    def __exit__(self, exc_type, exc_value, exc_traceback):
#        np.set_printoptions(**self.orig_opts)
#        if exc_traceback is not None:
#            print('[util_print] ERROR IN TRACEBACK: ' + str(exc_value))
#            return False


#def full_numpy_repr(arr):
#    with NpPrintOpts(threshold=np.uint64(-1)):
#        arr_repr = repr(arr)
#    return arr_repr


def printVERBOSE(msg, verbarg):
    if VERBOSE or verbarg in sys.argv:
        logger.info(msg)


def printNOTQUIET(msg):
    if not QUIET:
        logger.info(msg)


def printWARN(msg):
    try:
        import colorama
        from colorama import Fore, Style
        colorama.init()
        logger.info(Fore.RED + msg + Style.RESET_ALL)
        colorama.deinit()
    except ImportError:
        logger.info(msg)


def print_filesize(fpath):
    logger.info(util_str.filesize_str(fpath))


def printif(func, condition=VERBOSE and not QUIET):
    """ execute printfunc only if condition=QUIET"""
    if condition:
        logger.info(func())


def print_python_code(text):
    r"""
    SeeAlso:
        print_code
    """
    print_code(text, 'python')


def print_code(text, lexer_name='python'):
    r"""
    Args:
        text (str):

    CommandLine:
        python -m utool.util_print --test-print_python_code

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_print import *  # NOQA
        >>> import utool as ut
        >>> # build test data
        >>> text = ut.read_from(ut.__file__.replace('.pyc', '.py'))
        >>> # execute function
        >>> print_python_code(text)
    """
    logger.info(util_str.highlight_code(text, lexer_name))


def print_difftext(text, other=None):
    """
    Args:
        text (str):

    CommandLine:
        #python -m utool.util_print --test-print_difftext
        #autopep8 ingest_data.py --diff | python -m utool.util_print --test-print_difftext

    """
    if other is not None:
        # hack
        text = util_str.difftext(text, other)
    colortext = util_str.color_diff_text(text)
    try:
        logger.info(colortext)
    except UnicodeEncodeError as ex:  # NOQA
        import unicodedata
        colortext = unicodedata.normalize('NFKD', colortext).encode('ascii', 'ignore')
        logger.info(colortext)


def colorprint(text, color=None):
    r""" provides some color to terminal output

    Args:
        text (str):
        color (str):

    Ignore:
        assert color in ['', 'yellow', 'blink', 'lightgray', 'underline',
        'darkyellow', 'blue', 'darkblue', 'faint', 'fuchsia', 'black', 'white',
        'red', 'brown', 'turquoise', 'bold', 'darkred', 'darkgreen', 'reset',
        'standout', 'darkteal', 'darkgray', 'overline', 'purple', 'green', 'teal',
        'fuscia']

    CommandLine:
        python -c "import pygments.console; print(list(pygments.console.codes.keys()))"
        python -m utool.util_print --exec-colorprint
        python -m utool.util_print --exec-colorprint:1

        import pygments
        print(ut.repr4(list(pygments.formatters.get_all_formatters())))
        print(list(pygments.styles.get_all_styles()))

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_print import *  # NOQA
        >>> import pygments.console
        >>> msg_list = list(pygments.console.codes.keys())
        >>> color_list = list(pygments.console.codes.keys())
        >>> [colorprint(text, color) for text, color in zip(msg_list, color_list)]

    Example1:
        >>> # DISABLE_DOCTEST (Windows test)
        >>> from utool.util_print import *  # NOQA
        >>> import pygments.console
        >>> print('line1')
        >>> colorprint('line2', 'red')
        >>> colorprint('line3', 'blue')
        >>> colorprint('line4', 'fuchsia')
        >>> colorprint('line5', 'reset')
        >>> colorprint('line5', 'fuchsia')
        >>> print('line6')
    """
    logger.info(util_str.color_text(text, color))


cprint = colorprint


def print_locals(*args, **kwargs):
    """
    Prints local variables in function.

    If no arguments all locals are printed.

    Variables can be specified directly (variable values passed in) as varargs
    or indirectly (variable names passed in) in kwargs by using keys and a list
    of strings.
    """
    from utool import util_str
    from utool import util_dbg
    from utool import util_dict
    locals_ = util_dbg.get_parent_frame().f_locals
    keys = kwargs.get('keys', None if len(args) == 0 else [])
    to_print = {}
    for arg in args:
        varname = util_dbg.get_varname_from_locals(arg, locals_)
        to_print[varname] = arg
    if keys is not None:
        to_print.update(util_dict.dict_take(locals_, keys))
    if not to_print:
        to_print = locals_
    locals_str = util_str.repr4(to_print)
    logger.info(locals_str)


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_print
        python -m utool.util_print --allexamples
        python -m utool.util_print --allexamples --noface --nosrc """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
