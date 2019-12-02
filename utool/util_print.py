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
print, rrr, profile = util_inject.inject2(__name__)

QUIET        = meta_util_arg.QUIET
VERBOSE      = meta_util_arg.VERBOSE
NO_INDENT    = meta_util_arg.NO_INDENT
SILENT       = meta_util_arg.SILENT


def print_dict(dict_, dict_name=None, **kwargs):
    import utool as ut
    if dict_name is None:
        dict_name = ut.get_varname_from_stack(dict_, N=1)
    dict_repr = util_str.repr4(dict_, **kwargs)
    print(dict_name + ' = ' + dict_repr)

printdict = print_dict
dictprint = print_dict


def print_list(list_, **kwargs):
    import utool as ut
    list_name = ut.get_varname_from_stack(list_, N=1)
    print(list_name + ' = ' + util_str.repr4(list_, **kwargs))


def horiz_print(*args):
    toprint = util_str.horiz_string(args)
    print(toprint)


#def set_indenting_enabled(flag):
#    global NO_INDENT
#    prev_flag = NO_INDENT
#    NO_INDENT = not flag
#    return prev_flag


def _test_indent_print():
    # Indent test code doesnt work in doctest blocks.
    import utool as ut
    flag = ut.ensure_logging()
    print('Checking indent. Should have none')
    with ut.Indenter('[INDENT] '):
        print('Checking indent. Should be indented')
    print('Should no longer be indented')
    text = ut.get_current_log_text()
    # The last line might sometimes be empty or not.
    # Not sure.
    # New hack: had to put in stride. Seems like logs get written
    # with two line breaks now
    last_lines = text.split('\n')[-8::2]
    if last_lines[-1] != '':
        assert False, 'DEV ERROR. REMOVE FIRST LINE INSTEAD OF LAST'
        last_lines = last_lines[:-1]

    #print('last_lines = %r' % (ut.repr3(last_lines)))
    try:
        assert last_lines[0].find('[INDENT] ') == -1, last_lines[0]
        assert last_lines[1].find('[INDENT] ') >= 0, 'did not indent %r' % (last_lines[1],)
        assert last_lines[2].find('[INDENT] ') == -1, last_lines[2]
    except AssertionError:
        print('Error. Last 3 lines')
        print(ut.repr3(last_lines))
        raise
    if not flag:
        ut.stop_logging()


class Indenter(object):
    r"""
    Monkey patches modules injected with print to change the way print behaves.

    Works with utool.inject to allow for prefixing of all within-context
    print statements in a semi-dynamic manner. There seem to be some bugs
    but it works pretty well.

    CommandLine:
        python -m utool.util_print --exec-Indenter

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_print import *  # NOQA
        >>> import utool as ut
        >>> ut.util_print._test_indent_print()
    """
    # THIS IS MUCH BETTER
    def __init__(self, lbl='    ', enabled=True):
        self.enabled = enabled
        if not NO_INDENT or not self.enabled:
            #self.modules = modules
            self.modules = util_inject.get_injected_modules()
            self.old_print_dict = {}
            #self.old_prints_ = {}
            #self.old_printDBG_dict = {}
            self.lbl = lbl
            #self.INDENT_PRINT_ = False

    def start(self):
        # Chain functions together rather than overwriting stdout
        if NO_INDENT or not self.enabled:
            return builtins.print
        def indent_msg(*args):
            mgs = ', '.join(map(six.text_type, args))
            return self.lbl + mgs.replace('\n', '\n' + self.lbl)

        def push_module_functions(dict_, funcname):
            for mod in self.modules:
                try:
                    dict_[mod] = getattr(mod, funcname)
                except KeyError as ex:
                    print('[utool] KeyError: ' + six.text_type(ex))
                    print('[utool] WARNING: module=%r was loaded between indent sessions' % mod)
                except AttributeError as ex:
                    print('[utool] AttributeError: ' + six.text_type(ex))
                    print('[utool] WARNING: module=%r does not have injected utool prints' % mod)

        push_module_functions(self.old_print_dict, 'print')
        for mod in self.old_print_dict.keys():
            # Define the new print function
            @functools.wraps(self.old_print_dict[mod])
            def indent_print(*args):
                self.old_print_dict[mod](indent_msg(', '.join(map(six.text_type, args))))
            setattr(mod, 'print', indent_print)
        return indent_print

        #push_module_functions(self.old_printDBG_dict, 'printDBG')
        #for mod in self.old_printDBG_dict.keys():
        #    @functools.wraps(self.old_printDBG_dict[mod])
        #    def indent_printDBG(msg):
        #        self.old_printDBG_dict[mod](indent_msg(msg))
        #    setattr(mod, 'printDBG', indent_printDBG)

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

    def __enter__(self):
        self.start()
        return self

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
    print(util_str.highlight_code(text, lexer_name))


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
        print(colortext)
    except UnicodeEncodeError as ex:  # NOQA
        import unicodedata
        colortext = unicodedata.normalize('NFKD', colortext).encode('ascii', 'ignore')
        print(colortext)


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
    print(util_str.color_text(text, color))


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
    print(locals_str)


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
