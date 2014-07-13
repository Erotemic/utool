from __future__ import absolute_import, division, print_function
import numpy as np
import functools
import sys
from .util_str import horiz_string, filesize_str
from .util_inject import inject, get_injected_modules
print, print_, printDBG, rrr, profile = inject(__name__, '[print]')

QUIET = '--quiet' in sys.argv
VERBOSE = '--verbose' in sys.argv
SILENT = '--silent' in sys.argv
NO_INDENT = '--noindent' in sys.argv or '--no-indent' in sys.argv or SILENT


def horiz_print(*args):
    toprint = horiz_string(args)
    print(toprint)


#def set_indenting_enabled(flag):
#    global NO_INDENT
#    prev_flag = NO_INDENT
#    NO_INDENT = not flag
#    return prev_flag


class Indenter(object):
    # THIS IS MUCH BETTER
    @profile
    def __init__(self, lbl='    '):
        if not NO_INDENT:
            #self.modules = modules
            self.modules = get_injected_modules()
            self.old_print_dict = {}
            #self.old_prints_ = {}
            self.old_printDBG_dict = {}
            self.lbl = lbl
            self.INDENT_PRINT_ = False

    @profile
    def start(self):
        # Chain functions together rather than overwriting stdout
        if NO_INDENT:
            return
        def indent_msg(*args):
            mgs = ', '.join(map(str, args))
            return self.lbl + mgs.replace('\n', '\n' + self.lbl)

        def push_module_functions(dict_, func_name):
            for mod in self.modules:
                try:
                    dict_[mod] = getattr(mod, func_name)
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

        #push_module_functions(self.old_printDBG_dict, 'printDBG')
        #for mod in self.old_printDBG_dict.keys():
            #@functools.wraps(self.old_printDBG_dict[mod])
            #def indent_printDBG(msg):
            #    self.old_printDBG_dict[mod](indent_msg(msg))
            #setattr(mod, 'printDBG', indent_printDBG)

    @profile
    def stop(self):
        if NO_INDENT:
            return
        def pop_module_functions(dict_, func_name):
            for mod in dict_.iterkeys():
                setattr(mod, func_name, dict_[mod])
        pop_module_functions(self.old_print_dict, 'print')
        #pop_module_functions(self.old_printDBG_dict, 'printDBG')
        #for mod in self.old_print_dict.iterkeys():
        #    setattr(mod, 'print', self.old_print_dict[mod])
        #for mod in self.old_printDBG_dict.iterkeys():
        #    setattr(mod, 'printDBG', self.old_printDBG_dict[mod])

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type_, value, trace):
        self.stop()
        if trace is not None:
            print('[util_print] Error in context manager!: ' + str(value))
            return False  # return a falsey value on error


def printshape(arr_name, locals_):
    arr = locals_[arr_name]
    if type(arr) is np.ndarray:
        print(arr_name + '.shape = ' + str(arr.shape))
    else:
        print('len(%s) = %r' % (arr_name, len(arr)))


class NpPrintOpts(object):
    def __init__(self, **kwargs):
        self.orig_opts = np.get_printoptions()
        self.new_opts = kwargs
    def __enter__(self):
        np.set_printoptions(**self.new_opts)
    def __exit__(self, type_, value, trace):
        np.set_printoptions(**self.orig_opts)
        if trace is not None:
            print('[util_print] ERROR IN TRACEBACK: ' + str(value))
            # PYTHON 2.7 DEPRICATED:
            raise type_, value, trace


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
    print(filesize_str(fpath))


def printif(func, condition=VERBOSE and not QUIET):
    """ execute printfunc only if condition=QUIET"""
    if condition:
        print(func())
