""" Helpers for tests """
from __future__ import absolute_import, division, print_function
import __builtin__
import sys
from . import util_print
from . import util_dbg
from . import util_arg
from . import util_time
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[tests]')


HAPPY_FACE = r'''
               .-""""""-.
             .'          '.
            /   O      O   \
           :                :
           |                |
           : ',          ,' :
            \  '-......-'  /
             '.          .'
               '-......-'
                   '''


SAD_FACE = r'''
               .-""""""-.
             .'          '.
            /   O      O   \
           :           `    :
           |                |
           :    .------.    :
            \  '        '  /
             '.          .'
               '-......-'
                  '''


def run_test(func, *args, **kwargs):
    """
    Runs the test function with success / failure printing
    Input:
        Anything that needs to be passed to <func>
    """
    upper_func_name = func.func_name.upper()
    with util_print.Indenter('[' + upper_func_name + ']'):
        try:
            printTEST('[TEST.BEGIN] %s ' % (func.func_name,))
            with util_time.Timer(upper_func_name) as timer:
                test_locals = func(*args, **kwargs)
                # Write timings
            printTEST('[TEST.FINISH] %s -- SUCCESS' % (func.func_name,))
            print(HAPPY_FACE)
            with open('test_times.txt', 'a') as file_:
                msg = '%.4fs in %s\n' % (timer.ellapsed, upper_func_name)
                file_.write(msg)
            return test_locals
        except Exception as ex:
            # Get locals in the wrapped function
            util_dbg.printex(ex)
            exc_type, exc_value, tb = sys.exc_info()
            printTEST('[TEST.FINISH] %s -- FAILED: %s %s' % (func.func_name, type(ex), ex))
            print(SAD_FACE)
            if util_arg.STRICT:
                # Remove this function from stack strace
                exc_type, exc_value, exc_traceback = sys.exc_info()
                # PYTHON 2.7 DEPRICATED:
                raise exc_type, exc_value, exc_traceback.tb_next
                # PYTHON 3.3 NEW METHODS
                #ex = exc_type(exc_value)
                #ex.__traceback__ = exc_traceback.tb_next
                #raise ex


def printTEST(msg, wait=False):
    __builtin__.print('\n=============================')
    __builtin__.print('**' + msg)
    #if INTERACTIVE and wait:
    # raw_input('press enter to continue')
