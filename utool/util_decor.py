from __future__ import absolute_import, division, print_function
import __builtin__
import sys
from functools import wraps
from .util_iter import isiterable
from .util_print import Indenter
from .util_dbg import printex
import numpy as np
from .util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(__name__, '[decor]')


# do not ignore traceback when profiling
PROFILING = hasattr(__builtin__, 'profile')
FULL_TRACEBACK = '--noignore-exctb' in sys.argv or \
                 PROFILING or \
                 '--fulltb' in sys.argv
TRACE = '--trace' in sys.argv
UNIQUE_NUMPY = True
NOINDENT_DECOR = False


#def composed(*decs):
#    """ combines multiple decorators """
#    def deco(f):
#        for dec in reversed(decs):
#            f = dec(f)
#        return f
#    return deco


def ignores_exc_tb(func):
    """
    ignore_exc_tb decorates a function and remove both itself
    and the function from any exception traceback that occurs.

    This is useful to decorate other trivial decorators
    which are polluting your stacktrace.

    if IGNORE_EXC_TB is False then this decorator does nothing
    (and it should do nothing in production code!)
    """
    if FULL_TRACEBACK:
        return func
    else:
        @wraps(func)
        def wrp_no_exectb(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                # Code to remove this decorator from traceback
                exc_type, exc_value, exc_traceback0 = sys.exc_info()
                try:
                    exc_traceback1 = exc_traceback0.tb_next
                    exc_traceback2 = exc_traceback1.tb_next
                except Exception:
                    raise exc_type, exc_value, exc_traceback0
                # Remove two levels to remove this one as well
                # https://github.com/jcrocholl/pep8/issues/34  # NOQA
                # http://legacy.python.org/dev/peps/pep-3109/
                # PYTHON 2.7 DEPRICATED:
                raise exc_type, exc_value, exc_traceback2
                # PYTHON 3.3 NEW METHODS
                #ex = exc_type(exc_value)
                #ex.__traceback__ = exc_traceback.tb_next.tb_next
                #raise ex
        return wrp_no_exectb


def on_exception_report_input(func):
    @wraps(func)
    def wrp_exception_report_input(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            msg = ('ERROR: func_name=%r,\n * args=%r,\n * kwargs=%r' % (func.func_name, args, kwargs))
            printex(ex, msg)
            raise
    return wrp_exception_report_input


def _indent_decor(lbl):
    def closure_indent(func):
        printDBG('Indenting lbl=%r, func=%r' % (lbl, func))
        @ignores_exc_tb
        @wraps(func)
        def wrp_indent(*args, **kwargs):
            with Indenter(lbl):
                if TRACE:
                    print('    ...trace')
                return func(*args, **kwargs)
        return wrp_indent
    return closure_indent


def indent_func(input_):
    """
    Takes either no arguments or an alias label
    """
    if isinstance(input_, (str, unicode)):
        # A label was specified
        lbl = input_
        return _indent_decor(lbl)
    elif isinstance(input_, (bool, tuple)):
        # Allow individually turning of of this decorator
        func = input_
        return func
    else:
        # Use the function name as the label
        func = input_
        lbl = '[' + func.func_name + ']'
        return _indent_decor(lbl)(func)

#----------


def accepts_scalar_input(func):
    """
    accepts_scalar_input is a decorator which expects to be used on class methods.
    It lets the user pass either a vector or a scalar to a function, as long as
    the function treats everything like a vector. Input and output is sanatized
    to the user expected format on return.
    """
    @ignores_exc_tb
    @wraps(func)
    def wrp_si(self, input_, *args, **kwargs):
        if isiterable(input_):
            # If input is already iterable do default behavior
            return func(self, input_, *args, **kwargs)
        else:
            # If input is scalar, wrap input, execute, and unpack result
            ret = func(self, (input_,), *args, **kwargs)
            if ret is not None:
                return ret[0]
    return wrp_si


def __assert_param_consistency(args, argx_list):
    if len(argx_list) == 0:
        return True
    argx_flags = [isiterable(args[argx]) for argx in argx_list]
    try:
        assert all([argx_flags[0] == flag for flag in argx_flags]), (
            'invalid mixing of iterable and scalar inputs')
    except AssertionError:
        print('!!! ASSERTION ERROR !!!')
        for argx in argx_list:
            print('args[%d] = %r' % (argx, args[argx]))
        raise


def accepts_scalar_input2(argx_list=range(0, 1)):
    """
    accepts_scalar_input is a decorator which expects to be used on class methods.
    It lets the user pass either a vector or a scalar to a function, as long as
    the function treats everything like a vector. Input and output is sanatized
    to the user expected format on return.
    """
    def closure_si2(func):
        @ignores_exc_tb
        @wraps(func)
        def wrp_si2(self, *args, **kwargs):
            __assert_param_consistency(args, argx_list)
            if all([isiterable(args[ix]) for ix in argx_list]):
                # If input is already iterable do default behavior
                return func(self, *args, **kwargs)
            else:
                # If input is scalar, wrap input, execute, and unpack result
                args_wrapped = [(arg,) if ix in argx_list else arg
                                for arg in args]
                ret = func(self, *args_wrapped, **kwargs)
                if ret is not None:
                    return ret[0]
        return wrp_si2
    return closure_si2


def accepts_scalar_input_vector_output(func):
    """
    accepts_scalar_input is a decorator which expects to be used on class
    methods.  It lets the user pass either a vector or a scalar to a function,
    as long as the function treats everything like a vector. Input and output is
    sanatized to the user expected format on return.
    """
    @ignores_exc_tb
    @wraps(func)
    def wrp_sivo(self, input_, *args, **kwargs):
        if isiterable(input_):
            # If input is already iterable do default behavior
            return func(self, input_, *args, **kwargs)
        else:
            # If input is scalar, wrap input, execute, and unpack result
            result = func(self, (input_,), *args, **kwargs)
            # The output length could be 0 on a scalar input
            if len(result) != 0:
                return result[0]
            else:
                return result
    return wrp_sivo

# TODO: Rename to listget_1to1 1toM etc...
getter_1to1 = accepts_scalar_input
getter_1toM = accepts_scalar_input_vector_output
#----------


#def accepts_scalar_input_vector_output(func):
#    @wraps(func)
#    def wrp_sivo(self, input_, *args, **kwargs):
#        is_scalar = not isiterable(input_)
#        if is_scalar:
#            iter_input = (input_,)
#        else:
#            iter_input = input_
#        result = func(self, iter_input, *args, **kwargs)
#        if is_scalar:
#            if len(result) != 0:
#                result = result[0]
#        return result
#    return wrp_sivo


def accepts_numpy(func):
    """ Allows the first input to be a numpy array and get result in numpy form """
    #@ignores_exc_tb
    @wraps(func)
    def wrp_accepts_numpy(self, input_, *args, **kwargs):
        if not isinstance(input_, np.ndarray):
            # If the input is not numpy, just call the function
            return func(self, input_, *args, **kwargs)
        else:
            # If the input is a numpy array, and return the output with the same
            # shape as the input
            if UNIQUE_NUMPY:
                # Remove redundant input (because we are passing it to SQL)
                input_list, inverse_unique = np.unique(input_, return_inverse=True)
            else:
                input_list = input_.flatten()
            input_list = input_list.tolist()
            output_list = func(self, input_list, *args, **kwargs)
            if UNIQUE_NUMPY:
                # Reconstruct redundant queries (the user will never know!)
                output_arr = np.array(output_list)[inverse_unique]
                output_shape = tuple(list(input_.shape) + list(output_arr.shape[1:]))
                return np.array(output_arr).reshape(output_shape)
            else:
                return np.array(output_list).reshape(input_.shape)
    return wrp_accepts_numpy


def memorize(func):
    """
    Memoization decorator for functions taking one or more arguments.
    # http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/
    """
    class _memorizer(dict):
        def __init__(self, func):
            self.func = func
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.func(*key)
            return ret
    return _memorizer(func)


def interested(func):
    @indent_func
    #@ignores_exc_tb
    @wraps(func)
    def wrp_interested(*args, **kwargs):
        sys.stdout.write('#\n')
        sys.stdout.write('#\n')
        sys.stdout.write('<!INTERESTED>: ' + func.func_name + '\n')
        print('INTERESTING... ' + (' ' * 30) + ' <----')
        return func(*args, **kwargs)
    return wrp_interested
