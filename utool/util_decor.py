from __future__ import absolute_import, division, print_function
import __builtin__
import sys
from functools import wraps
from .util_iter import isiterable
from .util_print import Indenter
import numpy as np
from .util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(__name__, '[decor]')


# do not ignore traceback when profiling
IGNORE_EXC_TB = '--noignore-exctb' not in sys.argv or hasattr(__builtin__, 'profile')
TRACE = '--trace' in sys.argv


#def composed(*decs):
#    """ combines multiple decorators """
#    def deco(f):
#        for dec in reversed(decs):
#            f = dec(f)
#        return f
#    return deco


DISABLE_WRAPPERS = '--disable-wrappers' in sys.argv


def ignores_exc_tb(func):
    """ decorator that removes other decorators from traceback """
    if IGNORE_EXC_TB:
        @wraps(func)
        def wrapper_ignore_exctb(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                # Code to remove this decorator from traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                # Remove two levels to remove this one as well
                # https://github.com/jcrocholl/pep8/issues/34  # NOQA
                # http://legacy.python.org/dev/peps/pep-3109/
                # PYTHON 2.7 DEPRICATED:
                raise exc_type, exc_value, exc_traceback.tb_next.tb_next
                # PYTHON 3.3 NEW METHODS
                #ex = exc_type(exc_value)
                #ex.__traceback__ = exc_traceback.tb_next.tb_next
                #raise ex
        return wrapper_ignore_exctb
    else:
        return func


def identity_decor(func):
    return func


def indent_decor(lbl):
    def indent_decor_outer_wrapper(func):
        #@ignores_exc_tb
        @wraps(func)
        def indent_decor_inner_wrapper(*args, **kwargs):
            with Indenter(lbl):
                if TRACE:
                    print('    ...trace')
                return func(*args, **kwargs)
        return indent_decor_inner_wrapper
    return indent_decor_outer_wrapper


def indent_func(input_):
    """
    Takes either no arguments or an alias label
    """
    if isinstance(input_, (str, unicode)):
        lbl = input_
        return indent_decor(lbl)
    elif isinstance(input_, (bool, tuple)):
        return identity_decor
    if not isinstance(input_, (str, unicode)):
        func = input_
        # No arguments were passed
        @wraps(func)
        @indent_decor('[' + func.func_name + ']')
        #@ignores_exc_tb
        def wrapper_indent_func(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper_indent_func


def accepts_scalar_input(func):
    """
    accepts_scalar_input is a decorator which expects to be used on class methods.
    It lets the user pass either a vector or a scalar to a function, as long as
    the function treats everything like a vector. Input and output is sanatized
    to the user expected format on return.
    """
    @wraps(func)
    #@ignores_exc_tb
    def wrapper_scalar_input(self, input_, *args, **kwargs):
        if isiterable(input_):
            # If input is already iterable do default behavior
            return func(self, input_, *args, **kwargs)
        else:
            # If input is scalar, wrap input, execute, and unpack result
            ret = func(self, (input_,), *args, **kwargs)
            if ret is not None:
                return ret[0]
    return wrapper_scalar_input


#def accepts_scalar_input_vector_output(func):
#    @wraps(func)
#    def wrapper_vec_output(self, input_, *args, **kwargs):
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
#    return wrapper_vec_output


def accepts_scalar_input_vector_output(func):
    """
    accepts_scalar_input is a decorator which expects to be used on class
    methods.  It lets the user pass either a vector or a scalar to a function,
    as long as the function treats everything like a vector. Input and output is
    sanatized to the user expected format on return.
    """
    #@ignores_exc_tb
    @wraps(func)
    def wrapper_vec_output(self, input_, *args, **kwargs):
        if isiterable(input_):
            # If input is already iterable do default behavior
            return func(self, input_, *args, **kwargs)
        else:
            # If input is scalar, wrap input, execute, and unpack result
            result = func(self, (input_,), *args, **kwargs)
            # The length of the output has the potential to be 0 on a scalar input
            if len(result) != 0:
                return result[0]
            else:
                return result
    return wrapper_vec_output


UNIQUE_NUMPY = True


def accepts_numpy(func):
    """ Allows the first input to be a numpy objet and get result in numpy form """
    @wraps(func)
    def numpy_wrapper(self, input_, *args, **kwargs):
        if isinstance(input_, np.ndarray):
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
                output_ = np.array(output_arr).reshape(output_shape)
            else:
                output_ = np.array(output_list).reshape(input_.shape)
        else:
            output_ = func(self, input_)
        return output_
    return numpy_wrapper


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
    def interested_wrapper(*args, **kwargs):
        sys.stdout.write('#\n')
        sys.stdout.write('#\n')
        sys.stdout.write('<!INTERESTED>: ' + func.func_name + '\n')
        print('INTERESTING... ' + (' ' * 30) + ' <----')
        return func(*args, **kwargs)
    return interested_wrapper
