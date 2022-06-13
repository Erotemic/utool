# -*- coding: utf-8 -*-
"""
TODO: rectify name difference between parent and caller
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import range, zip, map, filter  # NOQA
import fnmatch
import operator
import inspect
import traceback
import time
import sys
import six
import shelve
import textwrap
import keyword
import re
import types
import functools
try:
    import numpy as np
except ImportError:
    pass
from os.path import splitext, split, basename, dirname
from utool import util_inject
from utool import util_arg
from utool import util_list
from utool import util_print
from utool import util_str
from utool import util_type
from utool._internal import meta_util_six
print, rrr, profile = util_inject.inject2(__name__)

RAISE_ALL = util_arg.get_argflag('--raise-all', help='Causes ut.printex to always reraise errors')
FORCE_TB = util_arg.get_argflag('--force-tb', help='Causes ut.printex to always print traceback')
# COLORED_EXCEPTIONS = util_arg.get_argflag(('--colorex', '--cex'))
COLORED_EXCEPTIONS = not util_arg.get_argflag(('--no-colorex', '--no-cex'))


def print_traceback(with_colors=True):
    """
    prints current stack
    """
    #traceback.print_tb()
    import traceback
    stack = traceback.extract_stack()
    stack_lines = traceback.format_list(stack)
    tbtext = ''.join(stack_lines)
    if with_colors:
        try:
            from pygments import highlight
            from pygments.lexers import get_lexer_by_name
            from pygments.formatters import TerminalFormatter
            lexer = get_lexer_by_name('pytb', stripall=True)
            formatter = TerminalFormatter(bg='dark')
            formatted_text = highlight(tbtext, lexer, formatter)
            print(formatted_text)
        except Exception:
            print(tbtext)
    else:
        print(tbtext)


def ipython_execstr():
    return textwrap.dedent(r'''
    import sys
    embedded = False
    if '-w' in sys.argv or '--wait' in sys.argv or '--wshow' in sys.argv:
        print('waiting')
        in_ = raw_input('press enter')
    if '--cmd' in sys.argv or locals().get('in_', '') == 'cmd':
        print('[utool.dbg] Requested IPython shell with --cmd argument.')
        try:
            __IPYTHON__
            print('[ipython_execstr] Already in IPython!')
        except NameError:
            try:
                import IPython
                print('[utool.dbg] Presenting in new ipython shell.')
                embedded = True
                IPython.embed()
            except Exception as ex:
                print('[ipython_execstr]: Error: ' + six.text_type(type(ex)) + six.text_type(ex))
                raise
    ''')


def execstr_parent_locals():
    parent_locals = get_parent_frame().f_locals
    return execstr_dict(parent_locals, 'parent_locals')


def execstr_attr_list(obj_name, attr_list=None):
    execstr_list = [obj_name + '.' + attr for attr in attr_list]
    return execstr_list


varname_regex = re.compile('[_A-Za-z][_a-zA-Z0-9]*$')


def is_valid_varname(varname):
    """ Checks syntax and validity of a variable name """
    if not isinstance(varname, six.string_types):
        return False
    match_obj = re.match(varname_regex, varname)
    valid_syntax = match_obj is not None
    valid_name = not keyword.iskeyword(varname)
    isvalid = valid_syntax and valid_name
    return isvalid


def execstr_dict(dict_, local_name=None, exclude_list=None, explicit=False):
    """
    returns execable python code that declares variables using keys and values

    execstr_dict

    Args:
        dict_ (dict):
        local_name (str): optional: local name of dictionary. Specifying this
            is much safer
        exclude_list (list):

    Returns:
        str: execstr --- the executable string that will put keys from dict
            into local vars

    CommandLine:
        python -m utool.util_dbg --test-execstr_dict

    Example:
        >>> # DISABLE_DOCTEST
        >>> # UNSTABLE_DOCTEST
        >>> from utool.util_dbg import *  # NOQA
        >>> my_dictionary = {'a': True, 'b': False}
        >>> execstr = execstr_dict(my_dictionary)
        >>> exec(execstr)
        >>> assert 'a' in vars() and 'b' in vars(), 'execstr failed'
        >>> assert b is False and a is True, 'execstr failed'
        >>> result = execstr
        >>> print(result)
        a = my_dictionary['a']
        b = my_dictionary['b']

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dbg import *  # NOQA
        >>> import utool as ut
        >>> my_dictionary = {'a': True, 'b': False}
        >>> execstr = execstr_dict(my_dictionary)
        >>> locals_ = locals()
        >>> exec(execstr, locals_)
        >>> a, b = ut.dict_take(locals_, ['a', 'b'])
        >>> assert 'a' in locals_ and 'b' in locals_, 'execstr failed'
        >>> assert b is False and a is True, 'execstr failed'
        >>> result = execstr
        >>> print(result)
        a = my_dictionary['a']
        b = my_dictionary['b']

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dbg import *  # NOQA
        >>> import utool as ut
        >>> my_dictionary = {'a': True, 'b': False}
        >>> execstr = execstr_dict(my_dictionary, explicit=True)
        >>> result = execstr
        >>> print(result)
        a = True
        b = False
    """
    import utool as ut
    if explicit:
        expr_list = []
        for (key, val) in sorted(dict_.items()):
            assert isinstance(key, six.string_types), 'keys must be strings'
            expr_list.append('%s = %s' % (key, ut.repr2(val),))
        execstr = '\n'.join(expr_list)
        return execstr
    else:
        if local_name is None:
            # Magic way of getting the local name of dict_
            local_name = get_varname_from_locals(dict_, get_parent_frame().f_locals)
        try:
            if exclude_list is None:
                exclude_list = []
            assert isinstance(exclude_list, list)
            exclude_list.append(local_name)
            expr_list = []
            assert isinstance(dict_, dict), 'incorrect type type(dict_)=%r, dict_=%r' % (type(dict), dict_)
            for (key, val) in sorted(dict_.items()):
                assert isinstance(key, six.string_types), 'keys must be strings'
                if not is_valid_varname(key):
                    continue
                if not any((fnmatch.fnmatch(key, pat) for pat in exclude_list)):
                    expr = '%s = %s[%s]' % (key, local_name, ut.repr2(key))
                    expr_list.append(expr)
            execstr = '\n'.join(expr_list)
            return execstr
        except Exception as ex:
            locals_ = locals()
            ut.printex(ex, key_list=['locals_'])
            raise


def execstr_func(func):
    print(' ! Getting executable source for: ' + meta_util_six.get_funcname(func))
    _src = inspect.getsource(func)
    execstr = textwrap.dedent(_src[_src.find(':') + 1:])
    # Remove return statments
    while True:
        # Find first 'return'
        stmtx = execstr.find('return')
        if stmtx == -1:
            break  # Fail condition
        # The characters which might make a return not have its own line
        stmt_endx = len(execstr) - 1
        for stmt_break in '\n;':
            print(execstr)
            print('')
            print(stmtx)
            stmt_endx_new = execstr[stmtx:].find(stmt_break)
            if -1 < stmt_endx_new < stmt_endx:
                stmt_endx = stmt_endx_new
        # now have variables stmt_x, stmt_endx
        before = execstr[:stmtx]
        after  = execstr[stmt_endx:]
        execstr = before + after
    return execstr


def save_testdata(*args, **kwargs):
    """
    caches testdata
    """
    uid = kwargs.get('uid', '')
    shelf_fname = 'test_data_%s.shelf' % uid
    shelf = shelve.open(shelf_fname)
    locals_ = get_parent_frame().f_locals
    print('save_testdata(%r)' % (args,))
    for key in args:
        shelf[key] = locals_[key]
    shelf.close()


def load_testdata(*args, **kwargs):
    """
    tries to load previously cached testdata
    """
    uid = kwargs.get('uid', '')
    shelf_fname = 'test_data_%s.shelf' % uid
    shelf = shelve.open(shelf_fname)
    ret = [shelf[key] for key in args]
    shelf.close()
    if len(ret) == 1:
        ret = ret[0]
    print('load_testdata(%r)' % (args,))
    return ret


def import_testdata():
    shelf = shelve.open('test_data.shelf')
    print('importing\n * ' + '\n * '.join(shelf.keys()))
    shelf_exec = execstr_dict(shelf, 'shelf')
    exec(shelf_exec)
    shelf.close()
    return import_testdata.func_code.co_code


def breakpoint(*tags):
    import utool as ut
    if ut.get_argflag('--break'):
        ut.embed(N=1)
        return True
    return False


def fix_embed_globals():
    """
    HACK adds current locals() to globals().
    Can be dangerous.
    """
    frame = get_stack_frame(N=1)
    frame.f_globals.update(frame.f_locals)
    frame.f_globals['_did_embed_fix'] = True
    """
    def fix_embed_globals(N=0):
        import inspect
        # Get the parent frame
        frame_cur = inspect.currentframe()
        for _ix in range(N + 1):
            # always skip the frame of this function
            frame_next = frame_cur.f_back
            if frame_next is None:
                break
            frame_cur = frame_next
        # Add locals to parent globals
        frame = frame_cur
        frame.f_globals.update(frame.f_locals)
        frame.f_globals['_did_embed_fix'] = True
    """


def _wip_embed(parent_locals=None, parent_globals=None, exec_lines=None,
               remove_pyqt_hook=True, N=0):
    """
    Starts interactive session. Similar to keyboard command in matlab.
    Wrapper around IPython.embed

    Notes:
        #https://github.com/ipython/ipython/wiki/Cookbook%3a-Updating-code-for-use-with-IPython-0.11-and-later

        import IPython
        x = 3
        IPython.embed()
        c = IPython.Config()
        c.InteractiveShellApp.exec_lines = [
            '%pylab qt4',
            "print 'System Ready!'",
        ]

        def foo():
            return x + 3

        a = 3
        def bar():
            return a + 3
        bar()
        #NameError: global name 'a' is not defined


        from IPython.terminal.ipapp import TerminalIPythonApp
        x = 3
        app = TerminalIPythonApp.instance()
        app.initialize(argv=[]) # argv=[] instructs IPython to ignore sys.argv
        app.start()


    Args:
        parent_locals (None):
        parent_globals (None):
        exec_lines (None):
        remove_pyqt_hook (bool):
        N (int):

    CommandLine:
        python -m utool.util_dbg --test-embed

    References:
       http://stackoverflow.com/questions/27911570/can-you-specify-a-command-to-run-after-you-embed-into-ipython/27914204#27914204
       http://stackoverflow.com/questions/15167200/how-do-i-embed-an-ipython-interpreter-into-an-application-running-in-an-ipython

    Notes:
        Use cases I want to achieve

        1) Simply stop execution and embed in an IPython terminal session
        2) Like case 1, but execute a specific set of command (eg '%gui qt')
           AFTER IPython has started
        3) Embed and pause GUI execution (this is just case 1)
        3) Embed and let GUI execution continue while embeded. (basically just need case 2)

    TODO:
        try:
            get_ipython
        except NameError:
            banner=exit_msg=''
        else:
            banner = '*** Nested interpreter ***'
            exit_msg = '*** Back in main IPython ***'

        # First import the embed function
        from IPython.frontend.terminal.embed import InteractiveShellEmbed
        # Now create the IPython shell instance. Put ipshell() anywhere in your code
        # where you want it to open.
        ipshell = InteractiveShellEmbed(banner1=banner, exit_msg=exit_msg)
        #Then use ipshell() whenever you want to be dropped into an IPython shell. This
        #will allow you to embed (and even nest) IPython interpreters in your code and
        #inspect objects or the state of the program.

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dbg import *  # NOQA
        >>> # build test data
        >>> parent_locals = None
        >>> parent_globals = None
        >>> exec_lines = None
        >>> remove_pyqt_hook = True
        >>> N = 0
        >>> # execute function
        >>> result = embed(parent_locals, parent_globals, exec_lines, remove_pyqt_hook, N)
        >>> # verify results
        >>> print(result)
    """
    import utool as ut
    from functools import partial
    import IPython

    if parent_globals is None:
        parent_globals = get_parent_frame(N=N).f_globals
    if parent_locals is None:
        parent_locals = get_parent_frame(N=N).f_locals

    stackdepth = N  # NOQA
    getframe = partial(ut.get_parent_frame, N=N)  # NOQA

    exec(execstr_dict(parent_globals, 'parent_globals'))
    exec(execstr_dict(parent_locals,  'parent_locals'))
    print('')
    print('================')
    print(ut.bubbletext('EMBEDDING'))
    print('================')
    print('[util] embedding')
    try:
        if remove_pyqt_hook:
            try:
                try:
                    import guitool_ibeis as gt
                except ImportError:
                    import guitool as gt
                gt.remove_pyqt_input_hook()
            except (ImportError, ValueError, AttributeError) as ex:
                #print(ex)
                printex(ex, iswarning=True)
                pass
            # make qt not loop forever (I had qflag loop forever with this off)
    except ImportError as ex:
        print(ex)
    user_ns = globals()
    user_ns = globals().copy()
    user_ns.update(locals())
    if parent_globals is not None:
        user_ns.update(parent_globals)
    if parent_locals is not None:
        user_ns.update(parent_locals)
    orig_argv = sys.argv  # NOQA
    print('About to start_ipython')
    config = IPython.Config()
    exec_lines_ = [
        '%pylab qt4',
        'print("Entered IPYTHON via utool")',
        'print("Entry Point: %r" % (ut.get_parent_frame(N=11).f_code.co_name,))',
        #'print("Entry Point: %r" % (ut.get_parent_frame(N=10).f_code.co_name,))',
        #'print("Entry Point: %r" % (ut.get_parent_frame(N=9).f_code.co_name,))',
        #'print("Entry Point: %r" % (ut.get_parent_frame(N=8).f_code.co_name,))',
        #'print("Entry Point: %r" % (ut.get_parent_frame(N=7).f_code.co_name,))',
        #'print("Entry Point: %r" % (ut.get_parent_frame(N=6).f_code.co_name,))',
        #'print("Entry Point: %r" % (ut.get_parent_frame(N=5).f_code.co_name,))',
        #execstr_dict(parent_locals)
    ] + ut.ensure_str_list(exec_lines if exec_lines is not None else [])
    config.InteractiveShellApp.exec_lines = exec_lines_
    print('Exec Lines: ')
    print(ut.indentjoin(exec_lines_, '\n    >>> '))
    IPython.start_ipython(config=config, argv=[], user_ns=user_ns)
    # Exit python immediately if specifed
    if user_ns.get('qqq', False) or vars.get('qqq', False) or user_ns.get('EXIT_NOW', False):
        print('[utool.embed] EXIT_NOW or qqq specified')
        sys.exit(1)


def embed(parent_locals=None, parent_globals=None, exec_lines=None,
          remove_pyqt_hook=True, N=0):
    """
    Starts interactive session. Similar to keyboard command in matlab.
    Wrapper around IPython.embed

    """
    import utool as ut
    from functools import partial
    import IPython

    if parent_globals is None:
        parent_globals = get_parent_frame(N=N).f_globals
    if parent_locals is None:
        parent_locals = get_parent_frame(N=N).f_locals

    stackdepth = N  # NOQA
    getframe = partial(ut.get_parent_frame, N=N)  # NOQA

    # exec(execstr_dict(parent_globals, 'parent_globals'))
    # exec(execstr_dict(parent_locals,  'parent_locals'))
    print('')
    print('================')
    print(ut.bubbletext('EMBEDDING'))
    print('================')
    print('[util] embedding')
    try:
        if remove_pyqt_hook:
            try:
                try:
                    import guitool_ibeis as gt
                except ImportError:
                    import guitool as gt
                gt.remove_pyqt_input_hook()
            except (ImportError, ValueError, AttributeError) as ex:
                #print(ex)
                printex(ex, iswarning=True)
                pass
            # make qt not loop forever (I had qflag loop forever with this off)
    except ImportError as ex:
        print(ex)

    #from IPython.config.loader import Config
    # cfg = Config()
    #config_dict = {}
    #if exec_lines is not None:
    #    config_dict['exec_lines'] = exec_lines
    #IPython.embed(**config_dict)
    print('[util]  Get stack location with: ')
    print('[util] ut.get_parent_frame(N=8).f_code.co_name')
    print('[util] set EXIT_NOW or qqq to True(ish) to hard exit on unembed')
    #print('set iup to True to draw plottool stuff')
    print('[util] call %pylab qt4 to get plottool stuff working')
    once = True
    # Allow user to set iup and redo the loop
    while once or vars().get('iup', False):
        if not once:
            # SUPER HACKY WAY OF GETTING FIGURES ON THE SCREEN BETWEEN UPDATES
            #vars()['iup'] = False
            # ALL YOU NEED TO DO IS %pylab qt4
            print('re-emebeding')
            for _ in range(100):
                time.sleep(.01)

        once = False
        #vars().get('iup', False):
        print('[util] calling IPython.embed()')
        """
        Notes:
            /usr/local/lib/python2.7/dist-packages/IPython/terminal/embed.py
            IPython.terminal.embed.InteractiveShellEmbed

            # instance comes from  IPython.config.configurable.SingletonConfigurable.instance
        """
        #c = IPython.Config()
        #c.InteractiveShellApp.exec_lines = [
        #    '%pylab qt4',
        #    '%gui qt4',
        #    "print 'System Ready!'",
        #]
        #IPython.embed(config=c)
        parent_ns = parent_globals.copy()
        parent_ns.update(parent_locals)
        locals().update(parent_ns)
        try:
            IPython.embed()
        except RuntimeError as ex:
            ut.printex(ex, 'Failed to open ipython')
        #config = IPython.terminal.ipapp.load_default_config()
        #config.InteractiveShellEmbed = config.TerminalInteractiveShell
        #module = sys.modules[parent_globals['__name__']]
        #config['module'] = module
        #config['module'] = module
        #embed2(stack_depth=N + 2 + 1)
        #IPython.embed(config=config)
        #IPython.embed(config=config)
        #IPython.embed(module=module)
        # Exit python immediately if specifed
        if vars().get('EXIT_NOW', False) or vars().get('qqq', False):
            print('[utool.embed] EXIT_NOW specified')
            sys.exit(1)


def embed2(**kwargs):
    """
    Modified from IPython.terminal.embed.embed so I can mess with stack_depth
    """
    config = kwargs.get('config')
    header = kwargs.pop('header', u'')
    stack_depth = kwargs.pop('stack_depth', 2)
    compile_flags = kwargs.pop('compile_flags', None)
    import IPython
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.terminal.embed import InteractiveShellEmbed
    if config is None:
        config = IPython.terminal.ipapp.load_default_config()
        config.InteractiveShellEmbed = config.TerminalInteractiveShell
        kwargs['config'] = config
    #save ps1/ps2 if defined
    ps1 = None
    ps2 = None
    try:
        ps1 = sys.ps1
        ps2 = sys.ps2
    except AttributeError:
        pass
    #save previous instance
    saved_shell_instance = InteractiveShell._instance
    if saved_shell_instance is not None:
        cls = type(saved_shell_instance)
        cls.clear_instance()
    shell = InteractiveShellEmbed.instance(**kwargs)
    shell(header=header, stack_depth=stack_depth, compile_flags=compile_flags)
    InteractiveShellEmbed.clear_instance()
    #restore previous instance
    if saved_shell_instance is not None:
        cls = type(saved_shell_instance)
        cls.clear_instance()
        for subclass in cls._walk_mro():
            subclass._instance = saved_shell_instance
    if ps1 is not None:
        sys.ps1 = ps1
        sys.ps2 = ps2


def quitflag(num=None, embed_=False, parent_locals=None, parent_globals=None):
    if num is None or util_arg.get_argflag('--quit' + six.text_type(num)):
        if parent_locals is None:
            parent_locals = get_parent_frame().f_locals
        if parent_globals is None:
            parent_globals = get_parent_frame().f_globals
        exec(execstr_dict(parent_locals, 'parent_locals'))
        exec(execstr_dict(parent_globals, 'parent_globals'))
        if embed_:
            print('Triggered --quit' + six.text_type(num))
            embed(parent_locals=parent_locals,
                  parent_globals=parent_globals)
        print('Triggered --quit' + six.text_type(num))
        sys.exit(1)


def qflag(num=None, embed_=True):
    frame = get_parent_frame()
    return quitflag(num, embed_=embed_,
                    parent_locals=frame.f_locals,
                    parent_globals=frame.f_globals)


def quit(num=None, embed_=False):
    frame = get_parent_frame()
    return quitflag(num, embed_=embed_,
                    parent_locals=frame.f_globals,
                    parent_globals=frame.f_locals)


def in_jupyter_notebook():
    """
    http://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        cfg = get_ipython().config
        #print('cfg = %s' % (ut.repr4(cfg),))
        #x = cfg['IPKernelApp']['parent_appname']
        # might not work if using jupyter-console
        if cfg['IPKernelApp']['connection_file'].count('jupyter'):
            return True
        else:
            return False
    except (AttributeError, NameError):
        return False


def inIPython():
    """
    Tests if running in IPython
    """
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def haveIPython():
    """
    Tests if IPython is available
    """
    try:
        import IPython  # NOQA
        return True
    except NameError:
        return False


def print_frame(frame):
    frame = frame if 'frame' in vars() else inspect.currentframe()
    attr_list = ['f_code.co_name', 'f_code.co_filename', 'f_back', 'f_lineno',
                 'f_code.co_names']
    obj_name = 'frame'
    execstr_print_list = ['print("%r=%%r" %% (%s,))' % (_execstr, _execstr)
                          for _execstr in execstr_attr_list(obj_name, attr_list)]
    execstr = '\n'.join(execstr_print_list)
    exec(execstr)
    local_varnames = util_str.pack_into('; '.join(frame.f_locals.keys()))
    print('Local varnames: ' + local_varnames)
    print('--- End Frame ---')


def search_stack_for_localvar(varname):
    """
    Finds a local varable somewhere in the stack and returns the value

    Args:
        varname (str): variable name

    Returns:
        None if varname is not found else its value
    """
    curr_frame = inspect.currentframe()
    print(' * Searching parent frames for: ' + six.text_type(varname))
    frame_no = 0
    while curr_frame.f_back is not None:
        if varname in curr_frame.f_locals.keys():
            print(' * Found in frame: ' + six.text_type(frame_no))
            return curr_frame.f_locals[varname]
        frame_no += 1
        curr_frame = curr_frame.f_back
    print('... Found nothing in all ' + six.text_type(frame_no) + ' frames.')
    return None


def search_stack_for_var(varname, verbose=util_arg.NOT_QUIET):
    """
    Finds a varable (local or global) somewhere in the stack and returns the value

    Args:
        varname (str): variable name

    Returns:
        None if varname is not found else its value
    """
    curr_frame = inspect.currentframe()
    if verbose:
        print(' * Searching parent frames for: ' + six.text_type(varname))
    frame_no = 0
    while curr_frame.f_back is not None:
        if varname in curr_frame.f_locals.keys():
            if verbose:
                print(' * Found local in frame: ' + six.text_type(frame_no))
            return curr_frame.f_locals[varname]
        if varname in curr_frame.f_globals.keys():
            if verbose:
                print(' * Found global in frame: ' + six.text_type(frame_no))
            return curr_frame.f_globals[varname]
        frame_no += 1
        curr_frame = curr_frame.f_back
    if verbose:
        print('... Found nothing in all ' + six.text_type(frame_no) + ' frames.')
    return None

# Alias
get_var_from_stack = search_stack_for_var
get_localvar_from_stack = search_stack_for_localvar


def get_stack_frame(N=0, strict=True):
    """
    Args:
        N (int): N=0 means the frame you called this function in.
                 N=1 is the parent frame.
        strict (bool): (default = True)
    """
    frame_cur = inspect.currentframe()
    for _ix in range(N + 1):
        # always skip the frame of this function
        frame_next = frame_cur.f_back
        if frame_next is None:
            if strict:
                raise AssertionError('Frame level %r is root' % _ix)
            else:
                break
        frame_cur = frame_next
    return frame_cur


def get_parent_frame(N=0):
    parent_frame = get_stack_frame(N=N + 2)
    return parent_frame


def quasiquote(string):
    """ mimics lisp quasi quote functionality """
    return string.format(**get_parent_frame().f_locals)


fmtlocals = quasiquote  # non-lispy alias for quasiquote


def get_caller_prefix(N=0, aserror=False):
    prefix_fmt = '[!%s]' if aserror else '[%s]'
    prefix = prefix_fmt % (get_caller_name(N=N + 1),)
    return prefix


def get_caller_lineno(N=0, strict=True):
    parent_frame = get_stack_frame(N=N + 2)
    lineno =  parent_frame.f_lineno
    return lineno


def get_caller_name(N=0, allow_genexpr=True):
    """
    get the name of the function that called you

    Args:
        N (int): (defaults to 0) number of levels up in the stack
        allow_genexpr (bool): (default = True)

    Returns:
        str: a function name

    CommandLine:
        python -m utool.util_dbg get_caller_name
        python -m utool get_caller_name
        python ~/code/utool/utool/__main__.py get_caller_name
        python ~/code/utool/utool/__init__.py get_caller_name
        python ~/code/utool/utool/util_dbg.py get_caller_name

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dbg import *  # NOQA
        >>> import utool as ut
        >>> N = list(range(0, 13))
        >>> allow_genexpr = True
        >>> caller_name = get_caller_name(N, allow_genexpr)
        >>> print(caller_name)
    """
    if isinstance(N, (list, tuple, range)):
        name_list = []
        for N_ in N:
            try:
                name_list.append(get_caller_name(N_))
            except AssertionError:
                name_list.append('X')
        return '[' + ']['.join(name_list) + ']'
    parent_frame = get_stack_frame(N=N + 2)
    caller_name = parent_frame.f_code.co_name
    co_filename = parent_frame.f_code.co_filename

    if not allow_genexpr:
        count = 0
        while True:
            count += 1
            if caller_name == '<genexpr>':
                parent_frame = get_stack_frame(N=N + 1 + count)
                caller_name = parent_frame.f_code.co_name
            else:
                break
    #try:
    #    if 'func' in  parent_frame.f_locals:
    #        caller_name += '(' + meta_util_six.get_funcname(parent_frame.f_locals['func']) + ')'
    #except Exception:
    #    pass
    if caller_name == '<module>':
        # Make the caller name the filename
        caller_name = splitext(split(co_filename)[1])[0]
    if caller_name in {'__init__', '__main__'}:
        # Make the caller name the filename
        caller_name = basename(dirname(co_filename)) + '.' + caller_name
    return caller_name


def get_caller_modname(N=0, allowmain=True):
    parent_frame = get_stack_frame(N=N + 2)
    assert allowmain is True
    caller_modname = parent_frame.f_globals['__name__']
    return caller_modname


def get_current_stack_depth():
    import traceback
    stack = traceback.extract_stack()
    return len(stack)


def module_functions(module):
    module_members = inspect.getmembers(module)
    function_list = []
    for key, val in module_members:
        if inspect.isfunction(val) and inspect.getmodule(val) == module:
            function_list.append((key, val))
    return function_list


def public_attributes(input):
    public_attr_list = []
    all_attr_list = dir(input)
    for attr in all_attr_list:
        if attr.find('__') == 0:
            continue
        public_attr_list.append(attr)
    return public_attr_list


def explore_stack():
    stack = inspect.stack()
    tup = stack[0]
    for ix, tup in reversed(list(enumerate(stack))):
        frame = tup[0]
        print('--- Frame %2d: ---' % (ix))
        print_frame(frame)
        print('\n')
        #next_frame = curr_frame.f_back


def explore_module(module_, seen=None, maxdepth=2, nonmodules=False):
    def __childiter(module):
        for aname in iter(dir(module)):
            if aname.find('_') == 0:
                continue
            try:
                yield module.__dict__[aname], aname
            except KeyError as ex:
                print(repr(ex))
                pass

    def __explore_module(module, indent, seen, depth, maxdepth, nonmodules):
        valid_children = []
        ret = u''
        modname = six.text_type(module.__name__)
        #modname = repr(module)
        for child, aname in __childiter(module):
            try:
                childtype = type(child)
                if not isinstance(childtype, types.ModuleType):
                    if nonmodules:
                        fullstr = indent + '    ' + six.text_type(aname) + ' = ' + repr(child)
                        truncstr = util_str.truncate_str(fullstr) + '\n'
                        ret +=  truncstr
                    continue
                childname = six.text_type(child.__name__)
                if seen is not None:
                    if childname in seen:
                        continue
                    elif maxdepth is None:
                        seen.add(childname)
                if childname.find('_') == 0:
                    continue
                valid_children.append(child)
            except Exception as ex:
                print(repr(ex))
                pass
        # Print
        ret += indent + modname + '\n'
        # Recurse
        if maxdepth is not None and depth >= maxdepth:
            return ret
        ret += ''.join([__explore_module(child,
                                         indent + '    ',
                                         seen, depth + 1,
                                         maxdepth,
                                         nonmodules)
                       for child in iter(valid_children)])
        return ret
    #ret +=
    #print('#module = ' + str(module_))
    ret = __explore_module(module_, '     ', seen, 0, maxdepth, nonmodules)
    #print(ret)
    sys.stdout.flush()
    return ret


def debug_npstack(stacktup):
    print('Debugging numpy [hv]stack:')
    print('len(stacktup) = %r' % len(stacktup))
    for count, item in enumerate(stacktup):
        if isinstance(item, np.ndarray):
            print(' * item[%d].shape = %r' % (count, item.shape))
        elif isinstance(item, list) or isinstance(item, tuple):
            print(' * len(item[%d]) = %d' % (count, len(item)))
            print(' * DEBUG LIST')
            with util_print.Indenter(' * '):
                debug_list(item)
        else:
            print(' *  type(item[%d]) = %r' % (count, type(item)))


def debug_list(list_):
    dbgmessage = []
    append = dbgmessage.append
    append('debug_list')
    dim2 = None
    if all([util_type.is_listlike(item) for item in list_]):
        append(' * list items are all listlike')
        all_lens = [len(item) for item in list_]
        if util_list.allsame(all_lens):
            dim2 = all_lens[0]
            append(' * uniform lens=%d' % dim2)
        else:
            append(' * nonuniform lens = %r' % np.unique(all_lens).tolist())
    else:
        all_types = [type(item) for item in list_]
        if util_list.allsame(all_types):
            append(' * uniform types=%r' % all_types[0])
        else:
            append(' * nonuniform types: %r' % np.unique(all_types).tolist())
    print('\n'.join(dbgmessage))
    return dim2


def debug_hstack(stacktup):
    try:
        return np.hstack(stacktup)
    except ValueError as ex:
        print('ValueError in debug_hstack: %s' % (ex,))
        debug_npstack(stacktup)
        raise


def debug_vstack(stacktup):
    try:
        return np.vstack(stacktup)
    except ValueError as ex:
        print('ValueError in debug_vstack: %s' % (ex,))
        debug_npstack(stacktup)
        raise


def debug_exception(func):
    @functools.wraps(func)
    def ex_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            import utool
            msg = ('[tools] ERROR: %s(%r, %r)' % (meta_util_six.get_funcname(func), args, kwargs))
            #print(msg)
            utool.printex(ex, msg)
            #print('[tools] ERROR: %r' % ex)
            raise
    return ex_wrapper


TB = util_arg.get_flag('--tb')


def printex(ex, msg='[!?] Caught exception', prefix=None, key_list=[],
            locals_=None, iswarning=False, tb=TB, pad_stdout=True, N=0,
            use_stdout=False, reraise=False, msg_=None, keys=None,
            colored=None):
    """
    Prints (and/or logs) an exception with relevant info

    Args:
        ex (Exception): exception to print
        msg (str): a message to display to the user
        keys (None): a list of strings denoting variables or expressions of interest
        iswarning (bool): prints as a warning rather than an error if True (defaults to False)
        tb (bool): if True prints the traceback in the error message
        pad_stdout (bool): separate the error message from the rest of stdout with newlines
        prefix (None):
        locals_ (None):
        N (int):
        use_stdout (bool):
        reraise (bool):
        msg_ (None):
        key_list (list): DEPRICATED use keys

    Returns:
        None
    """
    import utool as ut
    if isinstance(ex, MemoryError):
        ut.print_resource_usage()
    if keys is not None:
        # shorthand for key_list
        key_list = keys
    # Get error prefix and local info
    if prefix is None:
        prefix = get_caller_prefix(aserror=True, N=N)
    if locals_ is None:
        locals_ = get_parent_frame(N=N).f_locals
    # build exception message
    if msg is True:
        key_list = get_parent_frame().f_locals
        msg = msg_
    exstr = formatex(ex, msg, prefix, key_list, locals_, iswarning, tb=tb, colored=colored)
    # get requested print function
    if use_stdout:
        def print_func(*args):
            msg = ', '.join(list(map(six.text_type, args)))
            sys.stdout.write(msg + '\n')
            sys.stdout.flush()
    else:
        print_func = ut.partial(ut.colorprint, color='yellow' if iswarning else 'red')
        # print_func = print
    if pad_stdout:
        print_func('\n+------\n')
    # print the execption
    print_func(exstr)
    if pad_stdout:
        print_func('\nL______\n')
    # If you dont know where an error is coming from raise-all
    if (reraise and not iswarning) or RAISE_ALL:
        sys.stdout.flush()
        sys.stderr.flush()
        raise ex
    if ut.get_argflag('--exit-on-error'):
        print('WARNING: dont use this flag. Some errors are meant to be caught')
        ut.print_traceback()
        print('REQUESTED EXIT ON ERROR')
        sys.exit(1)


def formatex(ex, msg='[!?] Caught exception',
             prefix=None, key_list=[], locals_=None, iswarning=False, tb=False,
             N=0, keys=None, colored=None):
    r"""
    Formats an exception with relevant info

    Args:
        ex (Exception): exception to print
        msg (unicode):  a message to display to the user (default = u'[!?] Caught exception')
        keys (None): a list of strings denoting variables or expressions of interest (default = [])
        iswarning (bool): prints as a warning rather than an error if True (default = False)
        tb (bool): if True prints the traceback in the error message (default = False)
        prefix (None): (default = None)
        locals_ (None): (default = None)
        N (int): (default = 0)
        colored (None): (default = None)
        key_list (list): DEPRICATED use keys

    Returns:
        str: formated exception

    CommandLine:
        python -m utool.util_dbg --exec-formatex

    Example:
        >>> # ENABLE_DOCTET
        >>> from utool.util_dbg import *  # NOQA
        >>> import utool as ut
        >>> msg = 'Testing Exception'
        >>> prefix = '[test]'
        >>> key_list = ['N', 'foo', 'tb']
        >>> locals_ = None
        >>> iswarning = True
        >>> keys = None
        >>> colored = None
        >>> def failfunc():
        >>>     tb = True
        >>>     N = 0
        >>>     try:
        >>>         raise Exception('test exception. This is not an error')
        >>>     except Exception as ex:
        >>>         result = formatex(ex, msg, prefix, key_list, locals_,
        >>>                           iswarning, tb, N, keys, colored)
        >>>         return result
        >>> result = failfunc().replace('\n\n', '')
        >>> print(result)

        <!!! WARNING !!!>
        Traceback (most recent call last):
          File "<string>", line 15, in failfunc
        Exception: test exception. This is not an error[test] Testing Exception
        <class 'Exception'>: test exception. This is not an error
        [test] N = 0
        [test] foo = NameError (this likely due to a misformatted printex and is not related to the exception)
        [test] tb = True
        </!!! WARNING !!!>

    """
    # Get error prefix and local info
    if prefix is None:
        prefix = get_caller_prefix(aserror=True, N=N)
    if locals_ is None:
        locals_ = get_parent_frame(N=N).f_locals
    if keys is not None:
        # shorthand for key_list
        key_list = keys
    # build exception message
    errstr_list = []  # list of exception strings
    ex_tag = 'WARNING' if iswarning else 'EXCEPTION'
    errstr_list.append('<!!! %s !!!>' % ex_tag)
    if tb or FORCE_TB:
        tbtext = traceback.format_exc()
        if colored or COLORED_EXCEPTIONS:
            from utool import util_str
            tbtext = util_str.highlight_text(tbtext, lexer_name='pytb', stripall=True)
        errstr_list.append(tbtext)
    errstr_list.append(prefix + ' ' + six.text_type(msg) + '\n%r: %s' % (type(ex), six.text_type(ex)))
    #errstr_list.append(prefix + ' ' + six.text_type(msg) + '\ntype(ex)=%r' % (type(ex),))
    parse_locals_keylist(locals_, key_list, errstr_list, prefix)
    errstr_list.append('</!!! %s !!!>' % ex_tag)
    return '\n'.join(errstr_list)


def get_varname_from_stack(var, N=0, **kwargs):
    return get_varname_from_locals(var, get_parent_frame(N=N).f_locals, **kwargs)


def get_varname_from_locals(val, locals_, default='varname-not-found',
                            strict=False, cmpfunc_=operator.is_):
    """ Finds the string name which has where locals_[name] is val

    Check the varname is in the parent namespace
    This will only work with objects not primatives

    Args:
        val (): some value
        locals_ (dict): local dictionary to search
        default (str):
        strict (bool):

    Returns:
        str: the varname which is Val (if it exists)

    """
    if val is None or isinstance(val, (int, float, bool)):
        # Cannot work on primative types
        return default
    try:
        for count, val_ in enumerate(six.itervalues(locals_)):
            if cmpfunc_(val, val_):
                index_ = count
        varname = six.text_type(list(locals_.keys())[index_])
    except NameError:
        varname = default
        if strict:
            raise
    return varname


def get_varval_from_locals(key, locals_, strict=False):
    """
    Returns a variable value from locals.
    Different from locals()['varname'] because
    get_varval_from_locals('varname.attribute', locals())
    is allowed
    """
    assert isinstance(key, six.string_types), 'must have parsed key into a string already'
    if key not in locals_:
        dotpos = key.find('.')
        if dotpos > -1:
            key_ = key[:dotpos]
            attrstr_ = key[dotpos:]
            try:
                baseval = locals_[key_]  # NOQA
                val = eval('baseval' + attrstr_)
            except Exception as ex:
                if strict:
                    raise
                val = ex
        else:
            raise AssertionError('%s = NameError' % (key))
    else:
        val = locals_[key]
    return val


def get_varstr(val, pad_stdout=True, locals_=None):
    # TODO: combine with printex functionality
    if locals_ is None:
        locals_ = get_parent_frame().f_locals
    name = get_varname_from_locals(val, locals_)
    varstr_list = []
    if pad_stdout:
        varstr_list.append('\n\n+==========')
    varstr_list.append(repr(type(val)) + ' ' + name + ' = ')
    varstr_list.append(six.text_type(val))
    if pad_stdout:
        varstr_list.append('L==========')
    varstr = '\n'.join(varstr_list)
    return varstr


def super_print(val, locals_=None):
    if locals_ is None:
        locals_ = get_parent_frame().f_locals
    print(get_varstr(val, locals_=locals_))


def print_keys(key_list, locals_=None):
    if locals_ is None:
        locals_ = get_parent_frame().f_locals
    strlist_ = parse_locals_keylist(locals_, key_list)
    print('\n'.join(strlist_))


def parse_locals_keylist(locals_, key_list, strlist_=None, prefix=''):
    """ For each key in keylist, puts its value in locals into a stringlist

    Args:
        locals_ (?):
        key_list (list):
        strlist_ (list): (default = None)
        prefix (unicode): (default = u'')

    Returns:
        list: strlist_

    CommandLine:
        python -m utool.util_dbg --exec-parse_locals_keylist

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dbg import *  # NOQA
        >>> import utool as ut
        >>> locals_ = {'foo': [1, 2, 3], 'bar': 'spam', 'eggs': 4, 'num': 5}
        >>> key_list = [(len, 'foo'), 'bar.lower.__name__', 'eggs', 'num', 'other']
        >>> strlist_ = None
        >>> prefix = u''
        >>> strlist_ = parse_locals_keylist(locals_, key_list, strlist_, prefix)
        >>> result = ('strlist_ = %s' % (ut.repr2(strlist_, nl=True),))
        >>> print(result)
        strlist_ = [
            ' len(foo) = 3',
            " bar.lower.__name__ = 'lower'",
            ' eggs = 4',
            ' num = 5',
            ' other = NameError (this likely due to a misformatted printex and is not related to the exception)',
        ]
    """
    from utool import util_str
    if strlist_ is None:
        strlist_ = []

    for key in key_list:
        try:
            if key is None:
                strlist_.append('')
            elif isinstance(key, tuple):
                # Given a tuple of information
                tup = key
                func, key_ = tup
                val = get_varval_from_locals(key_, locals_)
                funcvalstr = six.text_type(func(val))
                callname = util_str.get_callable_name(func)
                strlist_.append('%s %s(%s) = %s' % (prefix, callname, key_, funcvalstr))
            elif isinstance(key, six.string_types):
                # Try to infer print from variable name
                val = get_varval_from_locals(key, locals_)
                #valstr = util_str.truncate_str(repr(val), maxlen=200)
                valstr = util_str.truncate_str(util_str.repr2(val), maxlen=200)
                strlist_.append('%s %s = %s' % (prefix, key, valstr))
            else:
                # Try to infer print from variable value
                val = key
                typestr = repr(type(val))
                namestr = get_varname_from_locals(val, locals_)
                #valstr = util_str.truncate_str(repr(val), maxlen=200)
                valstr = util_str.truncate_str(util_str.repr2(val), maxlen=200)
                strlist_.append('%s %s %s = %s' % (prefix, typestr, namestr, valstr))
        except AssertionError as ex:
            strlist_.append(prefix + ' ' + six.text_type(ex) + ' (this likely due to a misformatted printex and is not related to the exception)')
    return strlist_


def get_reprs(*args, **kwargs):
    if 'locals_' in kwargs:
        locals_ = kwargs['locals_']
    else:
        locals_ = locals()
        locals_.update(get_parent_frame().f_locals)

    msg_list = []
    var_list = list(args) + kwargs.get('var_list', [])
    for key in var_list:
        var = locals_[key]
        msg = util_str.horiz_string(six.text_type(key) + ' = ', repr(var))
        msg_list.append(msg)

    reprs = '\n' + util_str.indent('\n##\n'.join(msg_list)) + '\n'
    return reprs


def printvar2(varstr, attr='', typepad=0):
    locals_ = get_parent_frame().f_locals
    printvar(locals_, varstr, attr, typepad=typepad)


def printvar(locals_, varname, attr='.shape', typepad=0):
    #npprintopts = np.get_printoptions()
    #np.set_printoptions(threshold=5)
    dotpos = varname.find('.')
    # Locate var
    if dotpos == -1:
        var = locals_[varname]
    else:
        varname_ = varname[:dotpos]
        dotname_ = varname[dotpos:]
        var_ = locals_[varname_]  # NOQA
        var = eval('var_' + dotname_)
    # Print in format
    typestr = six.text_type(util_type.get_type(var)).ljust(typepad)

    if isinstance(var, np.ndarray):
        varstr = eval('str(var' + attr + ')')
        print('[var] %s %s = %s' % (typestr, varname + attr, varstr))
    elif isinstance(var, list):
        if attr == '.shape':
            func = 'len'
        else:
            func = ''
        varstr = eval('str(' + func + '(var))')
        print('[var] %s len(%s) = %s' % (typestr, varname, varstr))
    else:
        print('[var] %s %s = %r' % (typestr, varname, var))
    #np.set_printoptions(**npprintopts)


#def my_numpy_printops(linewidth=200, threshold=500, precision=8, edgeitems=5):
#    np.set_printoptions(linewidth=linewidth,
#                        precision=precision,
#                        edgeitems=edgeitems,
#                        threshold=threshold)


class EmbedOnException(object):
    """
    Context manager which embeds in ipython if an exception is thrown
    """
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __call__(self):
        return self

    def __exit__(self, type_, value, trace):
        if trace is not None:
            print('!!! EMBED ON EXCEPTION !!!')
            print('[util_dbg] %r in context manager!: %s ' % (type_, str(value)))
            import utool as ut
            import traceback
            traceback.print_exception(type_, value, trace)
            # Grab the context of the frame where the failure occurred
            trace_globals = trace.tb_frame.f_globals
            trace_locals = trace.tb_frame.f_locals
            trace_ns = trace_globals.copy()
            trace_ns.update(trace_locals)
            # Hack to bring back self
            if 'self' in trace_ns:
                self = trace_ns['self']
            # execstr_trace_g = ut.execstr_dict(trace_globals, 'trace_globals')
            # execstr_trace_l = ut.execstr_dict(trace_locals, 'trace_locals')
            # execstr_trace = execstr_trace_g + '\n' + execstr_trace_l
            # exec(execstr_trace)
            locals().update(trace_ns)
            ut.embed()

# maybe this will be easier to type?
embed_on_exception_context = EmbedOnException()
eoxc = embed_on_exception_context

if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_dbg; utool.doctest_funcs(utool.util_dbg, allexamples=True)"
        python -c "import utool, utool.util_dbg; utool.doctest_funcs(utool.util_dbg)"
        python -m utool.util_dbg
        python -m utool.util_dbg --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
