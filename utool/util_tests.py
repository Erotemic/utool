""" Helpers for tests """
from __future__ import absolute_import, division, print_function
from six.moves import builtins
import sys
from . import util_print
from . import util_dbg
from . import util_arg
from . import util_time
from .util_inject import inject
from utool._internal.meta_util_six import get_funcname
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
    upper_funcname = get_funcname(func).upper()
    with util_print.Indenter('[' + upper_funcname + ']'):
        try:
            import utool
            if utool.VERBOSE:
                printTEST('[TEST.BEGIN] %s ' % (sys.executable))
                printTEST('[TEST.BEGIN] %s ' % (get_funcname(func),))
            with util_time.Timer(upper_funcname) as timer:
                test_locals = func(*args, **kwargs)
                # Write timings
            printTEST('[TEST.FINISH] %s -- SUCCESS' % (get_funcname(func),))
            print(HAPPY_FACE)
            with open('test_times.txt', 'a') as file_:
                msg = '%.4fs in %s\n' % (timer.ellapsed, upper_funcname)
                file_.write(msg)
            return test_locals
        except Exception as ex:
            # Get locals in the wrapped function
            util_dbg.printex(ex)
            exc_type, exc_value, tb = sys.exc_info()
            printTEST('[TEST.FINISH] %s -- FAILED: %s %s' % (get_funcname(func), type(ex), ex))
            print(SAD_FACE)
            if util_arg.STRICT:
                # Remove this function from stack strace
                exc_type, exc_value, exc_traceback = sys.exc_info()
                exc_traceback = exc_traceback.tb_next
                # Python 2*3=6
                import six
                six.reraise(exc_type, exc_value, exc_traceback)
                # PYTHON 2.7 DEPRICATED:
                #raise exc_type, exc_value, exc_traceback.tb_next
                # PYTHON 3.3 NEW METHODS
                #ex = exc_type(exc_value)
                #ex.__traceback__ = exc_traceback.tb_next
                #raise ex
            return False


def printTEST(msg, wait=False):
    builtins.print('\n=============================')
    builtins.print('**' + msg)
    #if INTERACTIVE and wait:
    # raw_input('press enter to continue')
