from __future__ import absolute_import, division, print_function
#import six
from six.moves import range
import fnmatch
import operator
import inspect
import traceback
import time
try:
    import numpy as np
except ImportError:
    pass
import sys
import six
import shelve
import textwrap
import keyword
import re
import types
import functools
from os.path import splitext, split, basename, dirname
from utool import util_inject
from utool import util_arg
from utool import util_list
from utool import util_print
from utool import util_str
from utool import util_type
from utool._internal import meta_util_six
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[dbg]')

RAISE_ALL = util_arg.get_argflag('--raise-all', help='Causes ut.printex to always reraise errors')
FORCE_TB = util_arg.get_argflag('--force-tb', help='Causes ut.printex to always print traceback')

# --- Exec Strings ---
IPYTHON_EMBED_STR = r'''
try:
    import IPython
    print('Presenting in new ipython shell.')
    embedded = True
    IPython.embed()
except Exception as ex:
    warnings.warn(repr(ex)+'\n!!!!!!!!')
    embedded = False
'''


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


def execstr_embed():
    return IPYTHON_EMBED_STR


def ipython_execstr2():
    return textwrap.dedent(r'''
    import sys
    embedded = False
    try:
        __IPYTHON__
        in_ipython = True
    except NameError:
        in_ipython = False
    try:
        import IPython
        have_ipython = True
    except NameError:
        have_ipython = False
    if in_ipython:
        print('Presenting in current ipython shell.')
    elif '--cmd' in sys.argv:
        print('[utool.dbg] Requested IPython shell with --cmd argument.')
        if have_ipython:
            print('[utool.dbg] Found IPython')
            try:
                import IPython
                print('[utool.dbg] Presenting in new ipython shell.')
                embedded = True
                IPython.embed()
            except Exception as ex:
                print(repr(ex)+'\n!!!!!!!!')
                embedded = False
        else:
            print('[utool.dbg] IPython is not installed')
    ''')


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
                print('[ipython_execstr]: Error: ' + str(type(ex)) + str(ex))
                raise
    ''')


def execstr_parent_locals():
    parent_locals = get_parent_locals()
    return execstr_dict(parent_locals, 'parent_locals')


def execstr_attr_list(obj_name, attr_list=None):
    #if attr_list is None:
        #exec(execstr_parent_locals())
        #exec('attr_list = dir('+obj_name+')')
    execstr_list = [obj_name + '.' + attr for attr in attr_list]
    return execstr_list


varname_regex = re.compile('[_A-Za-z][_a-zA-Z0-9]*$')


def is_valid_varname(varname):
    """ Checks syntax and validity of a variable name """
    if not isinstance(varname, str):
        return False
    match_obj = re.match(varname_regex, varname)
    valid_syntax = match_obj is not None
    valid_name = not keyword.iskeyword(varname)
    isvalid = valid_syntax and valid_name
    return isvalid


def execstr_dict(dict_, local_name=None, exclude_list=None):
    """ returns execable python code that declares variables using keys and values

    execstr_dict

    Args:
        dict_ (dict):
        local_name (str): optional: local name of dictionary. Specifying this is much safer
        exclude_list (list):

    Returns:
        str: execstr --- the executable string that will put keys from dict
            into local vars

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dbg import *  # NOQA
        >>> my_dictionary = {'a': True, 'b':False}
        >>> execstr = execstr_dict(my_dictionary)
        >>> exec(execstr)
        >>> assert 'a' in vars() and 'b' in vars(), 'execstr failed'
        >>> assert b is False and a is True, 'execstr failed'
        >>> result = execstr
        >>> print(result)
        a = my_dictionary['a']
        b = my_dictionary['b']
    """
    if local_name is None:
        # Magic way of getting the local name of dict_
        local_name = get_varname_from_locals(dict_, get_parent_locals())
    try:
        #if exclude_list is None:
        #    execstr = '\n'.join((key + ' = ' + local_name + '[' + repr(key) + ']'
        #                        for (key, val) in dict_.items()))
        #else:
        if exclude_list is None:
            exclude_list = []
        assert isinstance(exclude_list, list)
        exclude_list.append(local_name)
        expr_list = []
        assert isinstance(dict_, dict), 'incorrect type type(dict_)=%r, dict_=%r' % (type(dict), dict_)
        for (key, val) in dict_.items():
            assert isinstance(key, str), 'keys must be strings'
            if not is_valid_varname(key):
                continue
            if not any((fnmatch.fnmatch(key, pat) for pat in exclude_list)):
                expr = '%s = %s[%r]' % (key, local_name, key)
                expr_list.append(expr)
        execstr = '\n'.join(expr_list)
        #print(execstr)
        return execstr
    except Exception as ex:
        import utool
        locals_ = locals()
        print(utool.printex(ex, key_list=['locals_']))
        raise


def execstr_func(func):
    print(' ! Getting executable source for: ' + meta_util_six.get_funcname(func))
    _src = inspect.getsource(func)
    execstr = textwrap.dedent(_src[_src.find(':') + 1:])
    # Remove return statments
    while True:
        stmtx = execstr.find('return')  # Find first 'return'
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


def execstr_src(func):
    return execstr_func(func)


def save_testdata(*args, **kwargs):
    """
    caches testdata
    """
    uid = kwargs.get('uid', '')
    shelf_fname = 'test_data_%s.shelf' % uid
    shelf = shelve.open(shelf_fname)
    locals_ = get_parent_locals()
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


def embed(parent_locals=None, parent_globals=None, exec_lines=None,
          remove_pyqt_hook=True, N=0):
    """
    Starts interactive session. Similar to keyboard command in matlab.
    Wrapper around IPython.embed

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
    if parent_globals is None:
        parent_globals = get_parent_globals(N=N)
        #parent_globals1 = get_parent_globals(N=0)
        #exec(execstr_dict(parent_globals1, 'parent_globals1'))
    if parent_locals is None:
        parent_locals = get_parent_locals(N=N)

    stackdepth = N  # NOQA
    import utool as ut
    from functools import partial
    getframe = partial(ut.get_caller_stack_frame, N=N)  # NOQA

    exec(execstr_dict(parent_globals, 'parent_globals'))
    exec(execstr_dict(parent_locals,  'parent_locals'))
    print('')
    print('================')
    print(ut.bubbletext('EMBEDING'))
    print('================')
    print('[util] embedding')
    import IPython
    try:
        if remove_pyqt_hook:
            try:
                import guitool
                guitool.remove_pyqt_input_hook()
            except (ImportError, ValueError) as ex:
                #print(ex)
                printex(ex, iswarning=True)
                pass
            # make qt not loop forever (I had qflag loop forever with this off)
    except ImportError as ex:
        print(ex)
    NEW_METHOD = False
    if NEW_METHOD:
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
            'print("Entry Point: %r" % (ut.get_caller_stack_frame(N=11).f_code.co_name,))',
            #'print("Entry Point: %r" % (ut.get_caller_stack_frame(N=10).f_code.co_name,))',
            #'print("Entry Point: %r" % (ut.get_caller_stack_frame(N=9).f_code.co_name,))',
            #'print("Entry Point: %r" % (ut.get_caller_stack_frame(N=8).f_code.co_name,))',
            #'print("Entry Point: %r" % (ut.get_caller_stack_frame(N=7).f_code.co_name,))',
            #'print("Entry Point: %r" % (ut.get_caller_stack_frame(N=6).f_code.co_name,))',
            #'print("Entry Point: %r" % (ut.get_caller_stack_frame(N=5).f_code.co_name,))',
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
    else:
        #from IPython.config.loader import Config
        # cfg = Config()
        #config_dict = {}
        #if exec_lines is not None:
        #    config_dict['exec_lines'] = exec_lines
        #IPython.embed(**config_dict)
        print('Get stack location with: ')
        print('ut.get_caller_stack_frame(N=8).f_code.co_name')
        print('set EXIT_NOW or qqq to True to hard exit on unembed')
        #print('set iup to True to draw plottool stuff')
        print('call %pylab qt4 to get plottool stuff working')
        once = True
        # Allow user to set iup and redo the loop
        while once or vars().get('iup', False):
            if not once:
                # SUPER HACKY WAY OF GETTING FIGURES ON THE SCREEN BETWEEN UPDATES
                #vars()['iup'] = False
                # ALL YOU NEED TO DO IS %pylab qt4
                print('re-emebeding')
                #import plottool as pt
                #pt.update()
                #(pt.present())
                for _ in range(100):
                    time.sleep(.01)

            once = False
            #vars().get('iup', False):
            IPython.embed()
            # Exit python immediately if specifed
            if vars().get('EXIT_NOW', False) or vars().get('qqq', False):
                print('[utool.embed] EXIT_NOW specified')
                sys.exit(1)


def quitflag(num=None, embed_=False, parent_locals=None, parent_globals=None):
    if num is None or util_arg.get_argflag('--quit' + str(num)):
        if parent_locals is None:
            parent_locals = get_parent_locals()
        if parent_globals is None:
            parent_globals = get_parent_globals()
        exec(execstr_dict(parent_locals, 'parent_locals'))
        exec(execstr_dict(parent_globals, 'parent_globals'))
        if embed_:
            print('Triggered --quit' + str(num))
            embed(parent_locals=parent_locals,
                  parent_globals=parent_globals)
        print('Triggered --quit' + str(num))
        sys.exit(1)


def qflag(num=None, embed_=True):
    return quitflag(num, embed_=embed_,
                    parent_locals=get_parent_locals(),
                    parent_globals=get_parent_globals())


def quit(num=None, embed_=False):
    return quitflag(num, embed_=embed_,
                    parent_locals=get_parent_locals(),
                    parent_globals=get_parent_globals())


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
    print(' * Searching parent frames for: ' + str(varname))
    frame_no = 0
    while curr_frame.f_back is not None:
        if varname in curr_frame.f_locals.keys():
            print(' * Found in frame: ' + str(frame_no))
            return curr_frame.f_locals[varname]
        frame_no += 1
        curr_frame = curr_frame.f_back
    print('... Found nothing in all ' + str(frame_no) + ' frames.')
    return None


def search_stack_for_var(varname):
    """
    Finds a varable (local or global) somewhere in the stack and returns the value

    Args:
        varname (str): variable name

    Returns:
        None if varname is not found else its value
    """
    curr_frame = inspect.currentframe()
    print(' * Searching parent frames for: ' + str(varname))
    frame_no = 0
    while curr_frame.f_back is not None:
        if varname in curr_frame.f_locals.keys():
            print(' * Found local in frame: ' + str(frame_no))
            return curr_frame.f_locals[varname]
        if varname in curr_frame.f_globals.keys():
            print(' * Found global in frame: ' + str(frame_no))
            return curr_frame.f_globals[varname]
        frame_no += 1
        curr_frame = curr_frame.f_back
    print('... Found nothing in all ' + str(frame_no) + ' frames.')
    return None

# Alias
get_var_from_stack = search_stack_for_var
get_localvar_from_stack = search_stack_for_localvar


def get_stack_frame(N=0, strict=True):
    frame_level0 = inspect.currentframe()
    frame_cur = frame_level0
    for _ix in range(N + 1):
        frame_next = frame_cur.f_back
        if frame_next is None:
            if strict:
                raise AssertionError('Frame level %r is root' % _ix)
            else:
                break
        frame_cur = frame_next
    return frame_cur


def get_caller_stack_frame(N=0):
    return get_stack_frame(N=N + 2)


def get_parent_frame(N=0):
    parent_frame = get_stack_frame(N=N + 2)
    return parent_frame


def get_parent_locals(N=0):
    """
    returns the locals of the function that called you

    Args:
        N (int): (defaults to 0) number of levels up in the stack

    Returns:
        dict : locals_
    """
    parent_frame = get_parent_frame(N=N + 1)
    locals_ = parent_frame.f_locals
    return locals_


def get_parent_globals(N=0):
    parent_frame = get_parent_frame(N=N + 1)
    globals_ = parent_frame.f_globals
    return globals_


def get_caller_locals(N=0):
    """
    returns the locals of the function that called you
    alias for get_parent_locals

    Args:
        N (int): (defaults to 0) number of levels up in the stack

    Returns:
        dict : locals_
    """

    locals_ = get_parent_locals(N=N + 1)
    return locals_


def quasiquote(string):
    return string.format(**get_caller_locals())


fmtlocals = quasiquote  # non-lispy alias for quasiquote


def get_caller_prefix(N=0, aserror=False):
    prefix_fmt = '[!%s]' if aserror else '[%s]'
    prefix = prefix_fmt % (get_caller_name(N=N + 1),)
    return prefix


def get_caller_lineno(N=0, strict=True):
    parent_frame = get_parent_frame(N=N + 1)
    lineno =  parent_frame.f_lineno
    return lineno


def get_caller_name(N=0):
    """
    get the name of the function that called you

    Args:
        N (int): (defaults to 0) number of levels up in the stack

    Returns:
        str: a function name
    """
    if isinstance(N, (list, tuple, range)):
        name_list = []
        for N_ in N:
            try:
                name_list.append(get_caller_name(N_))
            except AssertionError:
                name_list.append('X')
        return '[' + ']['.join(name_list) + ']'
    parent_frame = get_parent_frame(N=N + 1)
    caller_name = parent_frame.f_code.co_name
    #try:
    #    if 'func' in  parent_frame.f_locals:
    #        caller_name += '(' + meta_util_six.get_funcname(parent_frame.f_locals['func']) + ')'
    #except Exception:
    #    pass
    if caller_name == '<module>':
        co_filename = parent_frame.f_code.co_filename
        caller_name = splitext(split(co_filename)[1])[0]
    if caller_name == '__init__':
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
        modname = str(module.__name__)
        #modname = repr(module)
        for child, aname in __childiter(module):
            try:
                childtype = type(child)
                if not isinstance(childtype, types.ModuleType):
                    if nonmodules:
                        #print_(depth)
                        fullstr = indent + '    ' + str(aname) + ' = ' + repr(child)
                        truncstr = util_str.truncate_str(fullstr) + '\n'
                        ret +=  truncstr
                    continue
                childname = str(child.__name__)
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
        # print_(depth)
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
        if util_list.list_allsame(all_lens):
            dim2 = all_lens[0]
            append(' * uniform lens=%d' % dim2)
        else:
            append(' * nonuniform lens = %r' % np.unique(all_lens).tolist())
    else:
        all_types = [type(item) for item in list_]
        if util_list.list_allsame(all_types):
            append(' * uniform types=%r' % all_types[0])
        else:
            append(' * nonuniform types: %r' % np.unique(all_types).tolist())
    print('\n'.join(dbgmessage))
    return dim2


def debug_hstack(stacktup):
    try:
        return np.hstack(stacktup)
    except ValueError as ex:
        print('ValueError in debug_hstack: ' + str(ex))
        debug_npstack(stacktup)
        raise


def debug_vstack(stacktup):
    try:
        return np.vstack(stacktup)
    except ValueError as ex:
        print('ValueError in debug_vstack: ' + str(ex))
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
            use_stdout=False, reraise=False, msg_=None, keys=None):
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
    if isinstance(ex, MemoryError):
        import utool as ut
        ut.print_resource_usage()
    #ut.embed()
    if keys is not None:
        # shorthand for key_list
        key_list = keys
    # Get error prefix and local info
    if prefix is None:
        prefix = get_caller_prefix(aserror=True, N=N)
    if locals_ is None:
        locals_ = get_caller_locals(N=N)
    # build exception message
    if msg is True:
        key_list = get_caller_locals()
        msg = msg_
    exstr = formatex(ex, msg, prefix, key_list, locals_, iswarning, tb=tb)
    # get requested print function
    if use_stdout:
        def print_func(*args):
            msg = ', '.join(map(str, args))
            sys.stdout.write(msg + '\n')
            sys.stdout.flush()
    else:
        print_func = print
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


def formatex(ex, msg='[!?] Caught exception',
             prefix=None, key_list=[], locals_=None, iswarning=False, tb=False,
             N=0, keys=None):
    """ Formats an exception with relevant info """
    # Get error prefix and local info
    if prefix is None:
        prefix = get_caller_prefix(aserror=True, N=N)
    if locals_ is None:
        locals_ = get_caller_locals(N=N)
    if keys is not None:
        # shorthand for key_list
        key_list = keys
    # build exception message
    errstr_list = []  # list of exception strings
    ex_tag = 'WARNING' if iswarning else 'EXCEPTION'
    errstr_list.append('<!!! %s !!!>' % ex_tag)
    if tb or FORCE_TB:
        errstr_list.append(traceback.format_exc())
    errstr_list.append(prefix + ' ' + str(msg) + '\n%r: %s' % (type(ex), str(ex)))
    parse_locals_keylist(locals_, key_list, errstr_list, prefix)
    errstr_list.append('</!!! %s !!!>' % ex_tag)
    return '\n'.join(errstr_list)


def get_varname_from_stack(var, N=0, **kwargs):
    return get_varname_from_locals(var, get_caller_locals(N=N), **kwargs)


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
        varname = str(locals_.keys()[index_])
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
    assert isinstance(key, str), 'must have parsed key into a string already'
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
            raise AssertionError('!!! %s not populated!' % (key))
    else:
        val = locals_[key]
    return val


def get_varstr(val, pad_stdout=True, locals_=None):
    # TODO: combine with printex functionality
    if locals_ is None:
        locals_ = get_parent_locals()
    name = get_varname_from_locals(val, locals_)
    varstr_list = []
    if pad_stdout:
        varstr_list.append('\n\n+==========')
    varstr_list.append(repr(type(val)) + ' ' + name + ' = ')
    varstr_list.append(str(val))
    if pad_stdout:
        varstr_list.append('L==========')
    varstr = '\n'.join(varstr_list)
    return varstr


def super_print(val, locals_=None):
    if locals_ is None:
        locals_ = get_parent_locals()
    print(get_varstr(val, locals_=locals_))


def print_keys(key_list, locals_=None):
    if locals_ is None:
        locals_ = get_parent_locals()
    strlist_ = parse_locals_keylist(locals_, key_list)
    print('\n'.join(strlist_))


def parse_locals_keylist(locals_, key_list, strlist_=None, prefix=''):
    """ For each key in keylist, puts its value in locals into a stringlist """
    #from utool.util_str import get_callable_name
    from utool.util_str import get_callable_name
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
                funcvalstr = str(func(val))
                strlist_.append('%s %s(%s) = %s' % (prefix, get_callable_name(func), key_, funcvalstr))
            elif isinstance(key, six.string_types):
                # Try to infer print from variable name
                val = get_varval_from_locals(key, locals_)
                valstr = util_str.truncate_str(repr(val), maxlen=200)
                strlist_.append('%s %s = %s' % (prefix, key, valstr))
            else:
                # Try to infer print from variable value
                val = key
                typestr = repr(type(val))
                namestr = get_varname_from_locals(val, locals_)
                valstr = util_str.truncate_str(repr(val), maxlen=200)
                strlist_.append('%s %s %s = %s' % (prefix, typestr, namestr, valstr))
        except AssertionError as ex:
            strlist_.append(str(ex))
    return strlist_


def get_reprs(*args, **kwargs):
    if 'locals_' in kwargs:
        locals_ = kwargs['locals_']
    else:
        locals_ = locals()
        locals_.update(get_caller_locals())

    msg_list = []
    var_list = list(args) + kwargs.get('var_list', [])
    for key in var_list:
        var = locals_[key]
        msg = util_str.horiz_string(str(key) + ' = ', repr(var))
        msg_list.append(msg)

    reprs = '\n' + util_str.indent('\n##\n'.join(msg_list)) + '\n'
    return reprs


def printvar2(varstr, attr='', typepad=0):
    locals_ = get_parent_locals()
    printvar(locals_, varstr, attr, typepad=typepad)


def printvar(locals_, varname, attr='.shape', typepad=0):
    npprintopts = np.get_printoptions()
    np.set_printoptions(threshold=5)
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
    typestr = str(util_type.get_type(var)).ljust(typepad)

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
    np.set_printoptions(**npprintopts)


def my_numpy_printops(linewidth=200, threshold=500, precision=8, edgeitems=5):
    np.set_printoptions(linewidth=linewidth,
                        precision=precision,
                        edgeitems=edgeitems,
                        threshold=threshold)


def dict_dbgstr(dict_name, locals_=None):
    if locals_ is None:
        locals_ = get_parent_locals()
    lenstr = len_dbgstr(dict_name, locals_)
    keystr = keys_dbgstr(dict_name, locals_)
    return keystr + ' ' + lenstr
    #printvar(locals_, dict_name)


def keys_dbgstr(dict_name, locals_=None):
    if locals_ is None:
        locals_ = get_parent_locals()
    dict_ = locals_[dict_name]
    key_str = dict_name + '.keys() = ' + repr(dict_.keys())
    return key_str
    #dict_ = locals_[dict_name]


def print_varlen(name_, locals_=None):
    if locals_ is None:
        locals_ = get_parent_locals()
    prefix = get_caller_prefix()
    print(prefix + ' ' + len_dbgstr(name_, locals_))


def len_dbgstr(lenable_name, locals_=None):
    try:
        if locals_ is None:
            locals_ = get_parent_locals()
        lenable_ = locals_[lenable_name]
    except Exception:
        exec(execstr_dict(locals_, 'locals_'))
        try:
            lenable_ = eval(lenable_name)
        except Exception as ex:
            print('locals.keys = %r' % (locals_.keys(),))
            printex(ex, '[!util_dbg]')
            raise Exception('Cannot lendbg: %r' % lenable_name)
    len_str = 'len(%s) = %d' % (lenable_name, len(lenable_))
    return len_str


def list_dbgstr(list_name, trunc=2):
    locals_ = get_parent_locals()
    list_   = locals_[list_name]
    if trunc is None:
        pos = len(list_)
    else:
        pos     = min(trunc, len(list_) - 1)
    list_str = list_name + ' = ' + repr(list_[0:pos],)
    return list_str


def all_rrr():
    raise NotImplementedError('!!! STOP !!!')
    util_inject.inject_all()
    for mod in util_inject.get_injected_modules():
        try:
            if hasattr(mod, 'rrr'):
                mod.rrr()
        except Exception as ex:
            print(ex)
            print('mod = %r ' % mod)


class EmbedOnException(object):
    """
    Context manager which embeds in ipython if an exception is thrown
    """
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type_, value, trace):
        if trace is not None:
            print('!!!!!!!!!!!!!!!!!!!')
            print('[util_dbg] %r in context manager!: %s ' % (type_, str(value)))
            import utool
            import traceback
            #traceback.print_stack(type_, value, trace)
            traceback.print_exception(type_, value, trace)
            #parent_locals = utool.get_parent_locals()
            #execstr_parent = utool.execstr_dict(parent_locals, 'parent_locals')
            #exec(execstr_parent)
            trace_locals = trace.tb_frame.f_locals
            execstr_trace = utool.execstr_dict(trace_locals, 'trace_locals')
            exec(execstr_trace)
            utool.embed()

# maybe this will be easier to type?
embed_on_exception_context = EmbedOnException()

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
