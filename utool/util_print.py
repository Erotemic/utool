# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
try:
    import numpy as np
except ImportError:
    pass
import six
import functools
import sys
from six.moves import builtins
from utool._internal import meta_util_arg
from utool import util_str
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__, '[print]')

QUIET        = meta_util_arg.QUIET
VERBOSE      = meta_util_arg.VERBOSE
NO_INDENT    = meta_util_arg.NO_INDENT
SILENT       = meta_util_arg.SILENT


def print_dict(dict_, dict_name=None, **kwargs):
    import utool as ut
    if dict_name is None:
        dict_name = ut.get_varname_from_stack(dict_, N=1)
    dict_repr = util_str.dict_str(dict_, **kwargs)
    print(dict_name + ' = ' + dict_repr)

printdict = print_dict
dictprint = print_dict


def print_list(list_, **kwargs):
    import utool as ut
    list_name = ut.get_varname_from_stack(list_, N=1)
    print(list_name + ' = ' + util_str.list_str(list_, **kwargs))


def horiz_print(*args):
    toprint = util_str.horiz_string(args)
    print(toprint)


#def set_indenting_enabled(flag):
#    global NO_INDENT
#    prev_flag = NO_INDENT
#    NO_INDENT = not flag
#    return prev_flag


class Indenter(object):
    """
    Monkey patches modules injected with print to change the way print behaves.

    Works with utool.inject to allow for prefixing of all within-context
    print statements in a semi-dynamic manner. There seem to be some bugs
    but it works pretty well.
    """
    # THIS IS MUCH BETTER
    #@profile
    def __init__(self, lbl='    ', enabled=True):
        self.enabled = enabled
        if not NO_INDENT or not self.enabled:
            #self.modules = modules
            self.modules = util_inject.get_injected_modules()
            self.old_print_dict = {}
            #self.old_prints_ = {}
            self.old_printDBG_dict = {}
            self.lbl = lbl
            self.INDENT_PRINT_ = False

    @profile
    def start(self):
        # Chain functions together rather than overwriting stdout
        if NO_INDENT or not self.enabled:
            return builtins.print
        def indent_msg(*args):
            mgs = ', '.join(map(str, args))
            return self.lbl + mgs.replace('\n', '\n' + self.lbl)

        def push_module_functions(dict_, funcname):
            for mod in self.modules:
                try:
                    dict_[mod] = getattr(mod, funcname)
                except KeyError as ex:
                    print('[utool] KeyError: ' + str(ex))
                    print('[utool] WARNING: module=%r was loaded between indent sessions' % mod)
                except AttributeError as ex:
                    print('[utool] AttributeError: ' + str(ex))
                    print('[utool] WARNING: module=%r does not have injected utool prints' % mod)

        push_module_functions(self.old_print_dict, 'print')
        for mod in self.old_print_dict.keys():
            @functools.wraps(self.old_print_dict[mod])
            def indent_print(*args):
                self.old_print_dict[mod](indent_msg(', '.join(map(str, args))))
            setattr(mod, 'print', indent_print)
        return indent_print

        #push_module_functions(self.old_printDBG_dict, 'printDBG')
        #for mod in self.old_printDBG_dict.keys():
        #    @functools.wraps(self.old_printDBG_dict[mod])
        #    def indent_printDBG(msg):
        #        self.old_printDBG_dict[mod](indent_msg(msg))
        #    setattr(mod, 'printDBG', indent_printDBG)

    @profile
    def stop(self):
        if NO_INDENT or not self.enabled:
            return
        def pop_module_functions(dict_, funcname):
            for mod in six.iterkeys(dict_):
                setattr(mod, funcname, dict_[mod])
        pop_module_functions(self.old_print_dict, 'print')
        #pop_module_functions(self.old_printDBG_dict, 'printDBG')
        #for mod in six.iterkeys(self.old_print_dict):
        #    setattr(mod, 'print', self.old_print_dict[mod])
        #for mod in six.iterkeys(self.old_printDBG_dict):
        #    setattr(mod, 'printDBG', self.old_printDBG_dict[mod])
        return builtins.print

    @profile
    def __enter__(self):
        self.start()
        return self

    @profile
    def __exit__(self, type_, value, trace):
        self.stop()
        if trace is not None:
            if VERBOSE:
                print('[util_print] Error in print context manager!: ' + str(value))
            return False  # return a falsey value on error


def printshape(arr_name, locals_):
    arr = locals_[arr_name]
    if isinstance(arr, np.ndarray):
        print(arr_name + '.shape = ' + str(arr.shape))
    else:
        print('len(%s) = %r' % (arr_name, len(arr)))


class PrintStartEndContext(object):
    """
    prints on open and close of some section of interest
    """

    def __init__(self, msg='', verbose=True):
        self.verbose = verbose
        self.msg = msg

    def __enter__(self):
        if self.verbose:
            print('+--- START ' + self.msg)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.verbose:
            print('L___ END ' + self.msg)


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
        print(msg)


def printNOTQUIET(msg):
    if not QUIET:
        print(msg)


def printWARN(msg):
    try:
        import colorama
        from colorama import Fore, Style
        colorama.init()
        print(Fore.RED + msg + Style.RESET_ALL)
        colorama.deinit()
    except ImportError:
        print(msg)


def print_filesize(fpath):
    print(util_str.filesize_str(fpath))


def printif(func, condition=VERBOSE and not QUIET):
    """ execute printfunc only if condition=QUIET"""
    if condition:
        print(func())


def print_python_code(text):
    r"""
    Args:
        text (?):

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

    Ignore:
        import pygments
        print(ut.list_str(list(pygments.formatters.get_all_formatters())))
        print(list(pygments.styles.get_all_styles()))

    """
    print_code(text, 'python')
    #import pygments
    #import utool as ut
    ##with ut.embed_on_exception_context:
    #try:
    #    if ut.WIN32:
    #        #formater = pygments.formatters.terminal256.Terminal256Formatter()
    #        formater = pygments.formatters.terminal256.Terminal256Formatter()
    #    else:
    #        import pygments.formatters.terminal
    #        formater = pygments.formatters.terminal.TerminalFormatter(bg='dark')
    #    #, colorscheme='darkbg')
    #    lexer = pygments.lexers.get_lexer_by_name('python')
    #    print(pygments.highlight(text, lexer, formater))
    #except Exception:
    #    raise
    #    print(text)


def print_code(text, lexer_name='python'):
    r"""
    Args:
        text (?):

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

    Ignore:
        import pygments
        print(ut.list_str(list(pygments.formatters.get_all_formatters())))
        print(list(pygments.styles.get_all_styles()))

    """
    import utool as ut
    #with ut.embed_on_exception_context:
    try:
        import pygments
        if ut.WIN32:
            assert False
            #formater = pygments.formatters.terminal256.Terminal256Formatter()
            import pygments.formatters.terminal256
            formater = pygments.formatters.terminal256.Terminal256Formatter()
        else:
            import pygments.formatters.terminal
            formater = pygments.formatters.terminal.TerminalFormatter(bg='dark')
        #, colorscheme='darkbg')
        lexer = pygments.lexers.get_lexer_by_name(lexer_name)
        print(pygments.highlight(text, lexer, formater))
    except Exception:
        #raise
        print(text)


def get_colored_diff(text):
    import utool as ut
    try:
        import pygments
        if ut.WIN32:
            formater = pygments.formatters.terminal256.Terminal256Formatter()
        else:
            import pygments.formatters.terminal
            formater = pygments.formatters.terminal.TerminalFormatter(bg='dark')
        lexer = pygments.lexers.get_lexer_by_name('diff')
        return (pygments.highlight(text, lexer, formater))
    except Exception:
        #raise
        return text


def print_difftext(text):
    """
    Args:
        text (str):

    CommandLine:
        #python -m utool.util_print --test-print_difftext
        #autopep8 ingest_data.py --diff | python -m utool.util_print --test-print_difftext

    """
    print(get_colored_diff(text))
    #import pygments
    #import utool as ut
    #try:
    #    if ut.WIN32:
    #        formater = pygments.formatters.terminal256.Terminal256Formatter()
    #    else:
    #        import pygments.formatters.terminal
    #        formater = pygments.formatters.terminal.TerminalFormatter(bg='dark')
    #    lexer = pygments.lexers.get_lexer_by_name('diff')
    #    print(pygments.highlight(text, lexer, formater))
    #except Exception:
    #    raise
    #    print(text)


def colorprint(msg, color):
    """ provides some color to terminal output

    assert color in ['', 'yellow', 'blink', 'lightgray', 'underline',
    'darkyellow', 'blue', 'darkblue', 'faint', 'fuchsia', 'black', 'white',
    'red', 'brown', 'turquoise', 'bold', 'darkred', 'darkgreen', 'reset',
    'standout', 'darkteal', 'darkgray', 'overline', 'purple', 'green', 'teal',
    'fuscia']


    python -c "import pygments.console; print(list(pygments.console.codes.keys()))"

    Args:
        msg (?):
        color (?):

    CommandLine:
        python -m utool.util_print --exec-colorprint

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_print import *  # NOQA
        >>> import pygments.console
        >>> msg_list = list(pygments.console.codes.keys())
        >>> color_list = list(pygments.console.codes.keys())
        >>> [colorprint(msg, color) for msg, color in zip(msg_list, color_list)]
    """
    try:
        import pygments
        import pygments.console
        print(pygments.console.colorize(color, msg) + pygments.console.colorize('reset', ''))
    except ImportError:
        print(msg)


def print_locals():
    from utool import util_str
    from utool import util_dbg
    locals_ = util_dbg.get_caller_locals()
    print(util_str.dict_str(locals_))


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_print
        python -m utool.util_print --allexamples
        python -m utool.util_print --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
