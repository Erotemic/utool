from __future__ import absolute_import, division, print_function
# import decorator  # NOQA
from six.moves import builtins
import inspect
import textwrap
import six
import sys
from functools import wraps, update_wrapper
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
from . import util_print
from . import util_time
from . import util_iter
from . import util_dbg
from . import util_arg
from .util_inject import inject
from utool._internal.meta_util_six import get_funcname
(print, print_, printDBG, rrr, profile) = inject(__name__, '[decor]')

#SIG_PRESERVE = not util_arg.SAFE
SIG_PRESERVE = not util_arg.get_argflag('--nosigpreserve')

# do not ignore traceback when profiling
PROFILING = hasattr(builtins, 'profile')
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
    if not util_arg.IGNORE_TRACEBACK:
        return func
    else:
        @wraps(func)
        def wrp_no_exectb(*args, **kwargs):
            try:
                #import utool
                #if utool.DEBUG:
                #    print('[IN IGNORETB] args=%r' % (args,))
                #    print('[IN IGNORETB] kwargs=%r' % (kwargs,))
                return func(*args, **kwargs)
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                # Code to remove this decorator from traceback
                # Remove two levels to remove this one as well
                exc_type, exc_value, exc_traceback = sys.exc_info()
                try:
                    exc_traceback = exc_traceback.tb_next
                    exc_traceback = exc_traceback.tb_next
                except Exception:
                    pass
                # Python 2*3=6
                six.reraise(exc_type, exc_value, exc_traceback)
                # PYTHON 2.7 DEPRICATED:
                #raise exc_type, exc_value, exc_traceback
                # PYTHON 3.3 NEW METHODS
                #ex = exc_type(exc_value)
                #ex.__traceback__ = exc_traceback
                #raise ex
                # https://github.com/jcrocholl/pep8/issues/34  # NOQA
                # http://legacy.python.org/dev/peps/pep-3109/
        wrp_no_exectb = preserve_sig(wrp_no_exectb, func)
        return wrp_no_exectb


#@decorator.decorator
def on_exception_report_input(func):
    """
    If an error is thrown in the scope of this function's stack frame then the
    decorated function name and the arguments passed to it will be printed to
    the utool print function.
    """
    #@ignores_exc_tb
    @wraps(func)
    def wrp_exception_report_input(*args, **kwargs):
        try:
            #import utool
            #if utool.DEBUG:
            #    print('[IN EXCPRPT] args=%r' % (args,))
            #    print('[IN EXCPRPT] kwargs=%r' % (kwargs,))
            return func(*args, **kwargs)
        except Exception as ex:
            msg = ('ERROR: funcname=%r,\n * args=%r,\n * kwargs=%r\n' % (get_funcname(func), args, kwargs))
            msg += ' * len(args) = %r\n' % len(args)
            msg += ' * len(kwlargs) = %r\n' % len(kwargs)
            util_dbg.printex(ex, msg, separate=True)
            raise
    wrp_exception_report_input = preserve_sig(wrp_exception_report_input, func)
    return wrp_exception_report_input


def _indent_decor(lbl):
    """
    does the actual work of indent_func
    """
    #@decorator.decorator
    def closure_indent(func):
        #printDBG('Indenting lbl=%r, func=%r' % (lbl, func))
        if util_arg.TRACE:
            @ignores_exc_tb
            @wraps(func)
            def wrp_indent(*args, **kwargs):
                with util_print.Indenter(lbl):
                    print('    ...trace[in]')
                    ret = func(*args, **kwargs)
                    print('    ...trace[out]')
                    return ret
        else:
            @ignores_exc_tb
            @wraps(func)
            def wrp_indent(*args, **kwargs):
                with util_print.Indenter(lbl):
                    ret = func(*args, **kwargs)
                    return ret
        wrp_indent_ = ignores_exc_tb(wrp_indent)
        wrp_indent_ = preserve_sig(wrp_indent, func)
        return wrp_indent_
    return closure_indent


def indent_func(input_):
    """
    Takes either no arguments or an alias label
    """
    if isinstance(input_, six.string_types):
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
        from ._internal.meta_util_six import get_funcname
        lbl = '[' + get_funcname(func) + ']'
        return _indent_decor(lbl)(func)

#----------

#try:
#    import pandas as pd
#    HAS_PANDAS = True
#except Exception as ex:
#    HAS_PANDAS = False


#@decorator.decorator
def accepts_scalar_input(func):
    """
    accepts_scalar_input is a decorator which expects to be used on class methods.
    It lets the user pass either a vector or a scalar to a function, as long as
    the function treats everything like a vector. Input and output is sanatized
    to the user expected format on return.
    """
    #@on_exception_report_input
    @ignores_exc_tb
    @wraps(func)
    def wrp_si(self, input_, *args, **kwargs):
        #if HAS_PANDAS:
        #    if isinstance(input_, (pd.DataFrame, pd.Series)):
        #        input_ = input_.values
        if util_iter.isiterable(input_):
            # If input is already iterable do default behavior
            return func(self, input_, *args, **kwargs)
        else:
            # If input is scalar, wrap input, execute, and unpack result
            ret = func(self, (input_,), *args, **kwargs)
            if ret is not None:
                return ret[0]
    return wrp_si


def __assert_param_consistency(args, argx_list):
    """
    debugging function for accepts_scalar_input2
    """
    if util_arg.NO_ASSERTS:
        return
    if len(argx_list) == 0:
        return True
    argx_flags = [util_iter.isiterable(args[argx]) for argx in argx_list]
    try:
        assert all([argx_flags[0] == flag for flag in argx_flags]), (
            'invalid mixing of iterable and scalar inputs')
    except AssertionError:
        print('!!! ASSERTION ERROR !!!')
        for argx in argx_list:
            print('args[%d] = %r' % (argx, args[argx]))
        raise


def accepts_scalar_input2(argx_list=[0]):
    """
    FIXME: change to better name. Complete implementation.

    Args:
        argx_list (list): indexes of args that could be passed in as scalars to
            code that operates on lists. Ensures that decorated function gets
            the argument as an iterable.

    accepts_scalar_input is a decorator which expects to be used on class methods.
    It lets the user pass either a vector or a scalar to a function, as long as
    the function treats everything like a vector. Input and output is sanatized
    to the user expected format on return.
    """
    if not isinstance(argx_list, (list, tuple)):
        raise AssertionError('accepts_scalar_input2 must be called with argument positions')

    #@decorator.decorator
    def closure_si2(func):
        @ignores_exc_tb
        @wraps(func)
        def wrp_si2(self, *args, **kwargs):
            __assert_param_consistency(args, argx_list)
            if all([util_iter.isiterable(args[ix]) for ix in argx_list]):
                # If input is already iterable do default behavior
                return func(self, *args, **kwargs)
            else:
                # If input is scalar, wrap input, execute, and unpack result
                args_wrapped = [(arg,) if ix in argx_list else arg
                                for ix, arg in enumerate(args)]
                ret = func(self, *args_wrapped, **kwargs)
                if ret is not None:
                    return ret[0]
        return wrp_si2
    return closure_si2


#@decorator.decorator
def accepts_scalar_input_vector_output(func):
    """
    DEPRICATE IN FAVOR OF accepts_scalar_input2

    accepts_scalar_input is a decorator which expects to be used on class
    methods.  It lets the user pass either a vector or a scalar to a function,
    as long as the function treats everything like a vector. Input and output is
    sanatized to the user expected format on return.
    """
    @ignores_exc_tb
    @wraps(func)
    def wrp_sivo(self, input_, *args, **kwargs):
        #import utool
        #if utool.DEBUG:
        #    print('[IN SIVO] args=%r' % (args,))
        #    print('[IN SIVO] kwargs=%r' % (kwargs,))
        if util_iter.isiterable(input_):
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


#@decorator.decorator
def accepts_numpy(func):
    """ Allows the first input to be a numpy array and get result in numpy form """
    #@ignores_exc_tb
    @wraps(func)
    def wrp_accepts_numpy(self, input_, *args, **kwargs):
        if not (HAS_NUMPY and isinstance(input_, np.ndarray)):
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


#@decorator.decorator
def interested(func):
    @indent_func
    #@ignores_exc_tb
    @wraps(func)
    def wrp_interested(*args, **kwargs):
        sys.stdout.write('#\n')
        sys.stdout.write('#\n')
        sys.stdout.write('<!INTERESTED>: ' + get_funcname(func) + '\n')
        print('INTERESTING... ' + (' ' * 30) + ' <----')
        return func(*args, **kwargs)
    return wrp_interested


#@decorator.decorator
def show_return_value(func):
    from .util_str import func_str
    @wraps(func)
    def wrp_show_return_value(*args, **kwargs):
        ret = func(*args, **kwargs)
        #print('%s(*%r, **%r) returns %r' % (get_funcname(func), args, kwargs, rv))
        print(func_str(func, args, kwargs)  + ' -> ret=%r' % (ret,))
        return ret
    return wrp_show_return_value


#@decorator.decorator
def time_func(func):
    @wraps(func)
    def wrp_time(*args, **kwargs):
        with util_time.Timer(get_funcname(func)):
            return func(*args, **kwargs)
    return wrp_time


#class copy_argspec(object):
#    """
#    copy_argspec is a signature modifying decorator.
#    Specifically, it copies the signature from `source_func` to the wrapper, and
#    the wrapper will call the original function (which should be using *args,
#    **kwds).  The argspec, docstring, and default values are copied from
#    src_func, and __module__ and __dict__ from tgt_func.
#    .. References
#    http://stackoverflow.com/questions/18625510/how-can-i-programmatically-change-the-argspec-of-a-function-not-in-a-python-de
#    """
#    def __init__(self, src_func):
#        self.argspec = inspect.getargspec(src_func)
#        self.src_doc = src_func.__doc__
#        self.src_defaults = src_func.func_defaults
#    def __call__(self, tgt_func):
#        try:
#            tgt_argspec = inspect.getargspec(tgt_func)
#            need_self = False
#            if len(tgt_argspec) > 0 and len(tgt_argspec[0]) > 0 and tgt_argspec[0][0] == 'self':
#                need_self = True
#            name = tgt_func.__name__
#            argspec = self.argspec
#            if len(argspec) > 0 and len(argspec[0]) > 0 and argspec[0][0] == 'self':
#                need_self = False
#            if need_self:
#                newargspec = (['self'] + argspec[0],) + argspec[1:]
#            else:
#                newargspec = argspec
#            signature = inspect.formatargspec(formatvalue=lambda val: "",
#                                              *newargspec)[1:-1]
#            new_func = (
#                'def _wrapper_({signature}):\n'
#                '    return {tgt_func}({signature})'
#            ).format(signature=signature, tgt_func='tgt_func')
#            evaldict = {'tgt_func' : tgt_func}
#            exec new_func in evaldict
#            wrapped = evaldict['_wrapper_']
#            wrapped.__name__ = name
#            wrapped.__doc__ = self.src_doc
#            wrapped.func_defaults = self.src_defaults
#            wrapped.__module__ = tgt_func.__module__
#            wrapped.__dict__ = tgt_func.__dict__
#            return wrapped
#        except Exception as ex:
#            util_dbg.printex(ex, 'error wrapping: %r' % (tgt_func,))
#            raise


#@decorator.decorator
def lazyfunc(func):
    closuremem_ = [{}]
    def wrapper(*args, **kwargs):
        mem = closuremem_[0]
        key = (repr(args), repr(kwargs))
        try:
            return mem[key]
        except KeyError:
            mem[key] = func(*args, **kwargs)
        return mem[key]
    return wrapper


def preserve_sig(wrapper, orig_func, force=False):
    """
    Decorates a wrapper function.

    It seems impossible to presever signatures in python 2 without eval

    Args:
        wrapper: the function wrapping orig_func to change the signature of
        orig_func: the original function to take the signature from

    References:
        http://emptysqua.re/blog/copying-a-python-functions-signature/
    """
    if force or SIG_PRESERVE:
        src_fmt = r'''
        def _wrp_preserve{defsig}:
            try:
                return wrapper{callsig}
            except Exception as ex:
                import utool
                msg = ('Failure in signature preserving wrapper:\n')
                msg += ("defsig={defsig}\n")
                msg += ("callsig={callsig}\n")
                utool.print(ex, msg)
                raise
        '''
        # Put wrapped function into a scope
        globals_ =  {'wrapper': wrapper}
        locals_ = {}
        # Extract argspec from orig function
        argspec = inspect.getargspec(orig_func)
        # argspec is :ArgSpec(args=['bar', 'baz'], varargs=None, keywords=None, defaults=(True,))
        (args, varargs, varkw, defaults) = argspec
        # Get the function definition signature
        defsig = inspect.formatargspec(*argspec)
        # Get function call signature (no defaults)
        callsig = inspect.formatargspec(*argspec[0:3])
        # Define an exec function
        src = textwrap.dedent(src_fmt).format(defsig=defsig, callsig=callsig)
        # Define the new function on the fly
        # (I wish there was a non exec / eval way to do this)
        #print(src)
        exec(src, globals_, locals_)
        # Use functools.update_wapper to complete preservation
        _wrp_preserve = update_wrapper(locals_['_wrp_preserve'], orig_func)
        # Keep debug info
        _wrp_preserve._utinfo = {}
        _wrp_preserve._utinfo['src'] = src
        # Set an internal sig variable that we may use
        #_wrp_preserve.__sig__ = defsig
        _wrp_preserve._dbgsrc = src
        return _wrp_preserve
    else:
        # signature preservation is turned off. just preserve the name.
        # Does not use any exec or eval statments.
        return update_wrapper(wrapper, orig_func)
