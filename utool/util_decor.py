# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import builtins
import inspect
import textwrap
import six
import sys
import functools
import os
from utool import util_print
from utool import util_time
from utool import util_iter
from utool import util_dbg
from utool import util_arg
from utool import util_type
from utool import util_inject
from utool._internal import meta_util_six
(print, rrr, profile) = util_inject.inject2(__name__, '[decor]')

if util_type.HAVE_NUMPY:
    import numpy as np

# Commandline to toggle certain convinience decorators
SIG_PRESERVE = util_arg.get_argflag('--sigpreserve')
#SIG_PRESERVE = not util_arg.SAFE or util_arg.get_argflag('--sigpreserve')
ONEX_REPORT_INPUT = '--onex-report-input' in sys.argv
#IGNORE_TRACEBACK = '--smalltb' in sys.argv or '--ignoretb' in sys.argv
# FIXME: dupliated in _internal/py2_syntax_funcs
IGNORE_TRACEBACK = not ('--nosmalltb' in sys.argv or '--noignoretb' in sys.argv)
#if util_arg.STRICT:
#    IGNORE_TRACEBACK = False

# do not ignore traceback when profiling
PROFILING = hasattr(builtins, 'profile')
UNIQUE_NUMPY = True
NOINDENT_DECOR = False

#os.environ.get('UTOOL_AUTOGEN_SPHINX_RUNNING', 'OFF')

#def composed(*decs):
#    """ combines multiple decorators """
#    def deco(f):
#        for dec in reversed(decs):
#            f = dec(f)
#        return f
#    return deco


def test_ignore_exec_traceback():
    r"""
    CommandLine:
        python -m utool.util_decor --test-test_ignore_exec_traceback

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_decor import *  # NOQA
        >>> result = test_ignore_exec_traceback()
        >>> print(result)
    """
    import utool as ut
    @ut.indent_func
    def foobar():
        print('foobar')
        raise AssertionError('This error is exepcted')

    try:
        print('printing foobar')
        foobar()
    except AssertionError as ex:
        #import sys
        #exc_type, exc_value, exc_traceback = sys.exc_info()
        #print(exc_traceback)
        # TODO: ensure decorators are not printed in stack trace
        ut.printex(ex, 'There is no error. This is a test', tb=True)


if six.PY2:
    # Use version that has special python2 only syntax.
    # can not include it here for that reason
    from utool._internal import py2_syntax_funcs
    ignores_exc_tb = py2_syntax_funcs.ignores_exc_tb
else:
    def ignores_exc_tb(*args, **kwargs):
        """
        PYTHON 3 VERSION

        ignore_exc_tb decorates a function and remove both itself
        and the function from any exception traceback that occurs.

        This is useful to decorate other trivial decorators
        which are polluting your stacktrace.

        if IGNORE_TRACEBACK is False then this decorator does nothing
        (and it should do nothing in production code!)

        References:
            https://github.com/jcrocholl/pep8/issues/34  # NOQA
            http://legacy.python.org/dev/peps/pep-3109/
        """
        outer_wrapper = kwargs.get('outer_wrapper', True)
        def ignores_exc_tb_closure(func):
            # HACK JUST TURN THIS OFF
            return func

            if not IGNORE_TRACEBACK:
                # if the global enforces that we should not ignore anytracebacks
                # then just return the original function without any modifcation
                return func
            #@wraps(func)
            def wrp_noexectb(*args, **kwargs):
                try:
                    #import utool
                    #if utool.DEBUG:
                    #    print('[IN IGNORETB] args=%r' % (args,))
                    #    print('[IN IGNORETB] kwargs=%r' % (kwargs,))
                    return func(*args, **kwargs)
                except Exception:
                    # PYTHON 3.3 NEW METHODS
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    # Code to remove this decorator from traceback
                    # Remove two levels to remove this one as well
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    try:
                        exc_traceback = exc_traceback.tb_next
                        exc_traceback = exc_traceback.tb_next
                    except Exception:
                        pass
                    ex = exc_type(exc_value)
                    ex.__traceback__ = exc_traceback
                    raise ex
            if outer_wrapper:
                wrp_noexectb = preserve_sig(wrp_noexectb, func)
            return wrp_noexectb
        if len(args) == 1:
            # called with one arg means its a function call
            func = args[0]
            return ignores_exc_tb_closure(func)
        else:
            # called with no args means kwargs as specified
            return ignores_exc_tb_closure


# NEW PYTHON 2.7/3 VERSION
#def ignores_exc_tb(*args, **kwargs):
#    """
#    ignore_exc_tb decorates a function and remove both itself
#    and the function from any exception traceback that occurs.

#    This is useful to decorate other trivial decorators
#    which are polluting your stacktrace.

#    if IGNORE_TRACEBACK is False then this decorator does nothing
#    (and it should do nothing in production code!)

#    References:
#        https://github.com/jcrocholl/pep8/issues/34  # NOQA
#        http://legacy.python.org/dev/peps/pep-3109/
#    """
#    outer_wrapper = kwargs.get('outer_wrapper', True)
#    def ignores_exc_tb_closure(func):
#        if not IGNORE_TRACEBACK:
#            # if the global enforces that we should not ignore anytracebacks
#            # then just return the original function without any modifcation
#            return func
#        if six.PY2:
#            #python2_func = """
#            common_wrp_noexcept_tb = """
#            def wrp_noexectb(*args, **kwargs):
#                try:
#                    return func(*args, **kwargs)
#                except Exception:
#                    exc_type, exc_value, exc_traceback = sys.exc_info()
#                    # Code to remove this decorator from traceback
#                    # Remove two levels to remove this one as well
#                    exc_type, exc_value, exc_traceback = sys.exc_info()
#                    try:
#                        exc_traceback = exc_traceback.tb_next
#                        exc_traceback = exc_traceback.tb_next
#                    except Exception:
#                        pass
#            """
#            if six.PY2:
#                python2_reraise = """
#                    raise exc_type, exc_value, exc_traceback
#                """
#                six_reraise = python2_reraise
#            elif six.PY3:
#                python3_reraise = """
#                    ex = exc_type(exc_value)
#                    ex.__traceback__ = exc_traceback
#                    raise ex
#                """
#                six_reraise = python3_reraise
#            wrp_noexcept_tb_codeblock = common_wrp_noexcept_tb + six_reraise
#            globals_ = globals()
#            locals_ = locals()
#            six.exec_(wrp_noexcept_tb_codeblock, globals_, locals_)
#            wrp_noexectb = locals_['wrp_noexectb']
#        if outer_wrapper:
#            wrp_noexectb = preserve_sig(wrp_noexectb, func)
#        return wrp_noexectb
#    if len(args) == 1:
#        # called with one arg means its a function call
#        func = args[0]
#        return ignores_exc_tb_closure(func)
#    else:
#        # called with no args means kwargs as specified
#        return ignores_exc_tb_closure


def on_exception_report_input(func_=None, force=False, keys=None):
    """
    If an error is thrown in the scope of this function's stack frame then the
    decorated function name and the arguments passed to it will be printed to
    the utool print function.
    """
    def _closure_onexceptreport(func):
        if not ONEX_REPORT_INPUT and not force:
            return func
        @ignores_exc_tb(outer_wrapper=False)
        #@wraps(func)
        def wrp_onexceptreport(*args, **kwargs):
            try:
                #import utool
                #if utool.DEBUG:
                #    print('[IN EXCPRPT] args=%r' % (args,))
                #    print('[IN EXCPRPT] kwargs=%r' % (kwargs,))
                return func(*args, **kwargs)
            except Exception as ex:
                from utool import util_str
                print('ERROR occured! Reporting input to function')
                if keys is not None:
                    from utool import util_inspect
                    from utool import util_list
                    from utool import util_dict
                    argspec = util_inspect.get_func_argspec(func)
                    in_kwargs_flags = [key in kwargs for key in keys]
                    kwarg_keys = util_list.compress(keys, in_kwargs_flags)
                    kwarg_vals = [kwargs.get(key) for key in kwarg_keys]
                    flags = util_list.not_list(in_kwargs_flags)
                    arg_keys = util_list.compress(keys, flags)
                    arg_idxs = [argspec.args.index(key) for key in arg_keys]
                    num_nodefault = len(argspec.args) - len(argspec.defaults)
                    default_vals = (([None] * (num_nodefault)) +
                                    list(argspec.defaults))
                    args_ = list(args) + default_vals[len(args) + 1:]
                    arg_vals = util_list.take(args_, arg_idxs)
                    requested_dict = dict(util_list.flatten(
                        [zip(kwarg_keys, kwarg_vals), zip(arg_keys, arg_vals)]))
                    print('input dict = ' + util_str.repr4(
                        util_dict.dict_subset(requested_dict, keys)))
                    # (print out specific keys only)
                    pass
                arg_strs = ', '.join([repr(util_str.truncate_str(str(arg)))
                                      for arg in args])
                kwarg_strs = ', '.join([
                    util_str.truncate_str('%s=%r' % (key, val))
                    for key, val in six.iteritems(kwargs)])
                msg = ('\nERROR: funcname=%r,\n * args=%s,\n * kwargs=%r\n' % (
                    meta_util_six.get_funcname(func), arg_strs, kwarg_strs))
                msg += ' * len(args) = %r\n' % len(args)
                msg += ' * len(kwargs) = %r\n' % len(kwargs)
                util_dbg.printex(ex, msg, pad_stdout=True)
                raise
        wrp_onexceptreport = preserve_sig(wrp_onexceptreport, func)
        return wrp_onexceptreport
    if func_ is None:
        return _closure_onexceptreport
    else:
        return _closure_onexceptreport(func_)


def debug_function_exceptions(func):
    def _wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            import utool as ut
            ut.printex(ex)
            import inspect  # NOQA
            trace = inspect.trace()
            locals_ = trace[-1][0].f_locals
            print('-- <TRACE LOCALS> --')
            for level, t in enumerate(trace[1:]):
                frame = t[0]
                locals_ = frame.f_locals
                local_repr_dict = {key: ut.trunc_repr(val)
                                   for key, val in locals_.items()}
                print('LOCALS LEVEL %d' % (level,))
                print(ut.repr3(local_repr_dict, strvals=True, nl=1))
            print('-- </TRACE LOCALS> --')
            #import utool
            #utool.embed()
            raise
    return _wrapper


#class DebugContext(object):
#    def __enter__():
#        pass
#    def __exit__(self, exc_type, exc_value, exc_traceback):
#        pass


def _indent_decor(lbl):
    """
    does the actual work of indent_func
    """
    def closure_indent(func):
        if util_arg.TRACE:
            @ignores_exc_tb(outer_wrapper=False)
            #@wraps(func)
            def wrp_indent(*args, **kwargs):
                with util_print.Indenter(lbl):
                    print('    ...trace[in]')
                    ret = func(*args, **kwargs)
                    print('    ...trace[out]')
                    return ret
        else:
            @ignores_exc_tb(outer_wrapper=False)
            #@wraps(func)
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
        lbl = '[' + meta_util_six.get_funcname(func) + ']'
        return _indent_decor(lbl)(func)


def tracefunc_xml(func):
    """
    Causes output of function to be printed in an XML style block
    """
    funcname = meta_util_six.get_funcname(func)
    def wrp_tracefunc2(*args, **kwargs):
        verbose = kwargs.get('verbose', True)
        if verbose:
            print('<%s>' % (funcname,))
        with util_print.Indenter('    '):
            ret = func(*args, **kwargs)
        if verbose:
            print('</%s>' % (funcname,))
        return ret
    wrp_tracefunc2_ = ignores_exc_tb(wrp_tracefunc2)
    wrp_tracefunc2_ = preserve_sig(wrp_tracefunc2_, func)
    return wrp_tracefunc2_

#----------


def accepts_scalar_input(func):
    """
    DEPRICATE in favor of accepts_scalar_input2
    only accepts one input as vector

    accepts_scalar_input is a decorator which expects to be used on class
    methods.  It lets the user pass either a vector or a scalar to a function,
    as long as the function treats everything like a vector. Input and output
    is sanitized to the user expected format on return.

    Args:
        func (func):

    Returns:
        func: wrp_asi

    CommandLine:
        python -m utool.util_decor --test-accepts_scalar_input

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_decor import *  # NOQA
        >>> @accepts_scalar_input
        ... def foobar(self, list_):
        ...     return [x + 1 for x in list_]
        >>> self = None  # dummy self because this decorator is for classes
        >>> assert 2 == foobar(self, 1)
        >>> assert [2, 3] == foobar(self, [1, 2])
    """
    #@on_exception_report_input
    @ignores_exc_tb(outer_wrapper=False)
    #@wraps(func)
    def wrp_asi(self, input_, *args, **kwargs):
        #if HAVE_PANDAS:
        #    if isinstance(input_, (pd.DataFrame, pd.Series)):
        #        input_ = input_.values
        if util_iter.isiterable(input_):
            # If input is already iterable do default behavior
            return func(self, input_, *args, **kwargs)
        else:
            # If input is scalar, wrap input, execute, and unpack result
            #ret = func(self, (input_,), *args, **kwargs)
            ret = func(self, [input_], *args, **kwargs)
            if ret is not None:
                return ret[0]
    wrp_asi = preserve_sig(wrp_asi, func)
    return wrp_asi


def accepts_scalar_input2(argx_list=[0], outer_wrapper=True):
    r"""
    FIXME: change to better name. Complete implementation.

    used in IBEIS setters

    accepts_scalar_input2 is a decorator which expects to be used on class
    methods.  It lets the user pass either a vector or a scalar to a function,
    as long as the function treats everything like a vector. Input and output
    is sanitized to the user expected format on return.

    Args:
        argx_list (list): indexes of args that could be passed in as scalars to
            code that operates on lists. Ensures that decorated function gets
            the argument as an iterable.
    """
    assert isinstance(argx_list, (list, tuple)), (
        'accepts_scalar_input2 must be called with argument positions')

    def closure_asi2(func):
        #@on_exception_report_input
        @ignores_exc_tb(outer_wrapper=False)
        def wrp_asi2(self, *args, **kwargs):
            # Hack in case wrapping a function with varargs
            argx_list_ = [argx for argx in argx_list if argx < len(args)]
            __assert_param_consistency(args, argx_list_)
            if all([util_iter.isiterable(args[ix]) for ix in argx_list_]):
                # If input is already iterable do default behavior
                return func(self, *args, **kwargs)
            else:
                # If input is scalar, wrap input, execute, and unpack result
                args_wrapped = [(arg,) if ix in argx_list_ else arg
                                for ix, arg in enumerate(args)]
                ret = func(self, *args_wrapped, **kwargs)
                if ret is not None:
                    return ret[0]
        if outer_wrapper:
            wrp_asi2 = on_exception_report_input(preserve_sig(wrp_asi2, func))
        return wrp_asi2
    return closure_asi2


def __assert_param_consistency(args, argx_list_):
    """
    debugging function for accepts_scalar_input2
    checks to make sure all the iterable inputs are of the same length
    """
    if util_arg.NO_ASSERTS:
        return
    if len(argx_list_) == 0:
        return True
    argx_flags = [util_iter.isiterable(args[argx]) for argx in argx_list_]
    try:
        assert all([argx_flags[0] == flag for flag in argx_flags]), (
            'invalid mixing of iterable and scalar inputs')
    except AssertionError as ex:
        print('!!! ASSERTION ERROR IN UTIL_DECOR !!!')
        for argx in argx_list_:
            print('[util_decor] args[%d] = %r' % (argx, args[argx]))
        raise ex


def accepts_scalar_input_vector_output(func):
    """
    DEPRICATE IN FAVOR OF accepts_scalar_input2

    accepts_scalar_input_vector_output

    Notes:
        Input:                           Excpeted Output 1to1   Expected Output 1toM
            scalar         : 1           x                      [X]
            n element list : [1, 2, 3]   [x, y, z]              [[X], [Y], [Z]]
            1 element list : [1]         [x]                    [[X]]
            0 element list : []          []                     []
        There seems to be no real issue here, I be the thing that tripped me up
        was when using sql and getting multiple columns that returned the
        values inside of the N-tuple whereas when you get one column you get
        one element inside of a 1-tuple, no that still makes sense.  There was
        something where when you couln't unpack it becuase it was already
        empty...
    """
    @ignores_exc_tb(outer_wrapper=False)
    #@wraps(func)
    def wrp_asivo(self, input_, *args, **kwargs):
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
            if len(result) == 0:
                return []
            else:
                assert len(result) == 1, 'error in asivo'
                return result[0]
    return wrp_asivo

# TODO: Rename to listget_1to1 1toM etc...
getter_1to1 = accepts_scalar_input
getter_1toM = accepts_scalar_input_vector_output
#----------


def accepts_numpy(func):
    """ Allows the first input to be a numpy array and get result in numpy form """
    #@ignores_exc_tb
    #@wraps(func)
    def wrp_accepts_numpy(self, input_, *args, **kwargs):
        if not (util_type.HAVE_NUMPY and isinstance(input_, np.ndarray)):
            # If the input is not numpy, just call the function
            return func(self, input_, *args, **kwargs)
        else:
            # TODO: use a variant of util_list.unflat_unique_rowid_map
            # If the input is a numpy array, and return the output with the same
            # shape as the input
            if UNIQUE_NUMPY:
                # Remove redundant input (because we are passing it to SQL)
                input_list, inverse_unique = np.unique(input_, return_inverse=True)
            else:
                input_list = input_.flatten()
            # Call the function in list format
            # TODO: is this necessary?
            input_list = input_list.tolist()
            output_list = func(self, input_list, *args, **kwargs)
            # Put the output back into numpy
            if UNIQUE_NUMPY:
                # Reconstruct redundant queries
                output_arr = np.array(output_list)[inverse_unique]
                output_shape = tuple(list(input_.shape) + list(output_arr.shape[1:]))
                return np.array(output_arr).reshape(output_shape)
            else:
                return np.array(output_list).reshape(input_.shape)
    wrp_accepts_numpy = preserve_sig(wrp_accepts_numpy, func)
    return wrp_accepts_numpy


def memoize_nonzero(func):
    """
    Memoization decorator for functions taking a nonzero number of arguments.

    References:
        http://code.activestate.com/recipes/578231-fastest-memoization-decorator
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


def memoize_single(func):
    """ Memoization decorator for a function taking a single argument

    References:
        http://code.activestate.com/recipes/578231-fastest-memoization-decorator
    """
    class memodict_single(dict):
        def __missing__(self, key):
            ret = self[key] = func(key)
            return ret
    return memodict_single().__getitem__


def memoize_zero(func):
    """ Memoization decorator for a function taking no arguments """
    wrp_memoize_single = memoize_single(func)
    def wrp_memoize_zero():
        return wrp_memoize_single(None)
    return wrp_memoize_zero


def memoize(func):
    """
    simple memoization decorator

    References:
        https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize

    Args:
        func (function):  live python function

    Returns:
        func:

    CommandLine:
        python -m utool.util_decor memoize

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_decor import *  # NOQA
        >>> import utool as ut
        >>> closure = {'a': 'b', 'c': 'd'}
        >>> incr = [0]
        >>> def foo(key):
        >>>     value = closure[key]
        >>>     incr[0] += 1
        >>>     return value
        >>> foo_memo = memoize(foo)
        >>> assert foo('a') == 'b' and foo('c') == 'd'
        >>> assert incr[0] == 2
        >>> print('Call memoized version')
        >>> assert foo_memo('a') == 'b' and foo_memo('c') == 'd'
        >>> assert incr[0] == 4
        >>> assert foo_memo('a') == 'b' and foo_memo('c') == 'd'
        >>> print('Counter should no longer increase')
        >>> assert incr[0] == 4
        >>> print('Closure changes result without memoization')
        >>> closure = {'a': 0, 'c': 1}
        >>> assert foo('a') == 0 and foo('c') == 1
        >>> assert incr[0] == 6
        >>> assert foo_memo('a') == 'b' and foo_memo('c') == 'd'
    """
    cache = func._util_decor_memoize_cache = {}
    # @functools.wraps(func)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    memoizer = preserve_sig(memoizer, func)
    memoizer.cache = cache
    return memoizer


def interested(func):
    @indent_func
    #@ignores_exc_tb
    #@wraps(func)
    def wrp_interested(*args, **kwargs):
        sys.stdout.write('#\n')
        sys.stdout.write('#\n')
        sys.stdout.write(
            '<!INTERESTED>: ' + meta_util_six.get_funcname(func) + '\n')
        print('INTERESTING... ' + (' ' * 30) + ' <----')
        return func(*args, **kwargs)
    return wrp_interested


def tracefunc(func):
    lbl = '[trace.' + meta_util_six.get_funcname(func) + ']'
    def wrp_tracefunc(*args, **kwargs):
        print(lbl + ' +--- ENTER ---')
        with util_print.Indenter(lbl + ' |'):
            ret = func(*args, **kwargs)
        print(lbl + ' L___ EXIT ____')
        return ret
    return wrp_tracefunc


def show_return_value(func):
    from utool.util_str import func_str
    #@wraps(func)
    def wrp_show_return_value(*args, **kwargs):
        ret = func(*args, **kwargs)
        #print('%s(*%r, **%r) returns %r' % (meta_util_six.get_funcname(func), args, kwargs, rv))
        print(func_str(func, args, kwargs)  + ' -> ret=%r' % (ret,))
        return ret
    return wrp_show_return_value


def time_func(func):
    #@wraps(func)
    def wrp_time(*args, **kwargs):
        with util_time.Timer(meta_util_six.get_funcname(func)):
            return func(*args, **kwargs)
    wrp_time = preserve_sig(wrp_time, func)
    return wrp_time


#def rename_func(newname):
#    import utool as ut
#    return ut.partial(ut.set_funcname, newname=newname)


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


def lazyfunc(func):
    """
    Returns a memcached version of a function
    """
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


def apply_docstr(docstr_func):
    """
    Changes docstr of one functio to that of another
    """
    def docstr_applier(func):
        #docstr = meta_util_six.get_funcdoc(docstr_func)
        #meta_util_six.set_funcdoc(func, docstr)
        if isinstance(docstr_func, six.string_types):
            olddoc = meta_util_six.get_funcdoc(func)
            if olddoc is None:
                olddoc = ''
            newdoc = olddoc + docstr_func
            meta_util_six.set_funcdoc(func, newdoc)
            return func
        else:
            preserved_func = preserve_sig(func, docstr_func)
            return preserved_func
    return docstr_applier


def preserve_sig(wrapper, orig_func, force=False):
    """
    Decorates a wrapper function.

    It seems impossible to presever signatures in python 2 without eval
    (Maybe another option is to write to a temporary module?)

    Args:
        wrapper: the function wrapping orig_func to change the signature of
        orig_func: the original function to take the signature from

    References:
        http://emptysqua.re/blog/copying-a-python-functions-signature/
        https://code.google.com/p/micheles/source/browse/decorator/src/decorator.py

    TODO:
        checkout funcsigs
        https://funcsigs.readthedocs.org/en/latest/

    CommandLine:
        python -m utool.util_decor --test-preserve_sig

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> #ut.rrrr(False)
        >>> def myfunction(self, listinput_, arg1, *args, **kwargs):
        >>>     " just a test function "
        >>>     return [x + 1 for x in listinput_]
        >>> #orig_func = ut.take
        >>> orig_func = myfunction
        >>> wrapper = ut.accepts_scalar_input2([0])(orig_func)
        >>> #_wrp_preserve1 = ut.preserve_sig(wrapper, orig_func, True)
        >>> _wrp_preserve2 = ut.preserve_sig(wrapper, orig_func, False)
        >>> #print('_wrp_preserve2 = %r' % (_wrp_preserve1,))
        >>> print('_wrp_preserve2 = %r' % (_wrp_preserve2,))
    """
    #if True:
    #    import functools
    #    return functools.wraps(orig_func)(wrapper)
    from utool._internal import meta_util_six
    from utool import util_str
    from utool import util_inspect

    if wrapper is orig_func:
        # nothing to do
        return orig_func
    orig_docstr = meta_util_six.get_funcdoc(orig_func)
    orig_docstr = '' if orig_docstr is None else orig_docstr
    orig_argspec = util_inspect.get_func_argspec(orig_func)
    wrap_name = meta_util_six.get_funccode(wrapper).co_name
    orig_name = meta_util_six.get_funcname(orig_func)

    # At the very least preserve info in a dictionary
    _utinfo = {}
    _utinfo['orig_func'] = orig_func
    _utinfo['wrap_name'] = wrap_name
    _utinfo['orig_name'] = orig_name
    _utinfo['orig_argspec'] = orig_argspec

    if hasattr(wrapper, '_utinfo'):
        parent_wrapper_utinfo = wrapper._utinfo
        _utinfo['parent_wrapper_utinfo'] = parent_wrapper_utinfo
    if hasattr(orig_func, '_utinfo'):
        parent_orig_utinfo = orig_func._utinfo
        _utinfo['parent_orig_utinfo'] = parent_orig_utinfo

    # environment variable is set if you are building documentation
    # preserve sig if building docs
    building_docs = os.environ.get('UTOOL_AUTOGEN_SPHINX_RUNNING', 'OFF') == 'ON'

    if force or SIG_PRESERVE or building_docs:
        # PRESERVES ALL SIGNATURES WITH EXECS
        src_fmt = r'''
        def _wrp_preserve{defsig}:
            """ {orig_docstr} """
            try:
                return wrapper{callsig}
            except Exception as ex:
                import utool as ut
                msg = ('Failure in signature preserving wrapper:\n')
                ut.printex(ex, msg)
                raise
        '''
        raise NotImplementedError('broke in 311, but probably unused')
        # Put wrapped function into a scope
        globals_ =  {'wrapper': wrapper}
        locals_ = {}
        # argspec is :ArgSpec(args=['bar', 'baz'], varargs=None, keywords=None,
        # defaults=(True,))
        # get orig functions argspec
        # get functions signature
        # Get function call signature (no defaults)
        # Define an exec function
        argspec = inspect.getargspec(orig_func)
        (args, varargs, varkw, defaults) = argspec
        defsig = inspect.formatargspec(*argspec)
        callsig = inspect.formatargspec(*argspec[0:3])
        # TODO:
        src_fmtdict = dict(defsig=defsig, callsig=callsig, orig_docstr=orig_docstr)
        src = textwrap.dedent(src_fmt).format(**src_fmtdict)
        # Define the new function on the fly
        # (I wish there was a non exec / eval way to do this)
        #print(src)
        code = compile(src, '<string>', 'exec')
        six.exec_(code, globals_, locals_)
        #six.exec_(src, globals_, locals_)
        # Use functools.update_wapper to complete preservation
        _wrp_preserve = functools.update_wrapper(locals_['_wrp_preserve'], orig_func)
        # Keep debug info
        _utinfo['src'] = src
        # Set an internal sig variable that we may use
        #_wrp_preserve.__sig__ = defsig
    else:
        # PRESERVES SOME SIGNATURES NO EXEC
        # signature preservation is turned off. just preserve the name.
        # Does not use any exec or eval statments.
        _wrp_preserve = functools.update_wrapper(wrapper, orig_func)
        # Just do something to preserve signature

    DEBUG_WRAPPED_DOCSTRING = False
    if DEBUG_WRAPPED_DOCSTRING:
        new_docstr_fmtstr = util_str.codeblock(
            '''
            Wrapped function {wrap_name}({orig_name})

            orig_argspec = {orig_argspec}

            orig_docstr = {orig_docstr}
            '''
        )
    else:
        new_docstr_fmtstr = util_str.codeblock(
            '''
            {orig_docstr}
            '''
        )
    new_docstr = new_docstr_fmtstr.format(
        wrap_name=wrap_name, orig_name=orig_name, orig_docstr=orig_docstr,
        orig_argspec=orig_argspec)
    meta_util_six.set_funcdoc(_wrp_preserve, new_docstr)
    _wrp_preserve._utinfo = _utinfo
    return _wrp_preserve


def dummy_args_decor(*args, **kwargs):
    def dummy_args_closure(func):
        return func
    return dummy_args_closure


class classproperty(property):
    """
    Decorates a method turning it into a classattribute

    References:
        https://stackoverflow.com/questions/1697501/python-staticmethod-with-property
    """
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_decor; utool.doctest_funcs(utool.util_decor)"
        python -m utool.util_decor
        python -m utool.util_decor --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
