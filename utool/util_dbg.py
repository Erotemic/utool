from __future__ import absolute_import, division, print_function
#import six
from six.moves import range
import fnmatch
import inspect
import traceback
import numpy as np
import sys
import shelve
import textwrap
import keyword
import re
import types
import functools
from os.path import splitext, split
from . import util_inject
from .util_arg import get_flag, SUPER_STRICT
from .util_inject import inject
from .util_list import list_allsame
from .util_print import Indenter
from .util_str import pack_into, truncate_str, horiz_string, indent
from .util_type import is_listlike, get_type
from utool._internal.meta_util_six import get_funcname
print, print_, printDBG, rrr, profile = inject(__name__, '[dbg]')

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
    if '-w' in sys.argv or '--wait' in sys.argv:
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


def execstr_dict(dict_, local_name, exclude_list=None):
    """ returns execable python code that declares variables using keys and values """
    #if local_name is None:
    #    local_name = dict_
    #    exec(execstr_parent_locals())
    #    exec('dict_ = local_name')
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
    print(' ! Getting executable source for: ' + get_funcname(func))
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
    uid = kwargs.get('uid', '')
    shelf_fname = 'test_data_%s.shelf' % uid
    shelf = shelve.open(shelf_fname)
    locals_ = get_parent_locals()
    print('save_testdata(%r)' % (args,))
    for key in args:
        shelf[key] = locals_[key]
    shelf.close()


def load_testdata(*args, **kwargs):
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
          remove_pyqt_hook=True):
    if parent_globals is None:
        parent_globals = get_parent_globals(N=0)
        # not sure why N=1 works over N=0 here only for globals
        parent_globals1 = get_parent_globals(N=1)
        exec(execstr_dict(parent_globals1, 'parent_globals1'))
    if parent_locals is None:
        parent_locals = get_parent_locals(N=0)

    exec(execstr_dict(parent_globals, 'parent_globals'))
    exec(execstr_dict(parent_locals,  'parent_locals'))
    print('')
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
    config_dict = {}
    #if exec_lines is not None:
    #    config_dict['exec_lines'] = exec_lines
    IPython.embed(**config_dict)


def quitflag(num=None, embed_=False, parent_locals=None, parent_globals=None):
    if num is None or get_flag('--quit' + str(num)):
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
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def haveIPython():
    try:
        import IPython  # NOQA
        return True
    except NameError:
        return False


def print_frame(frame):
    frame = frame if 'frame' in vars() else inspect.currentframe()
    attr_list = ['f_code.co_name', 'f_back', 'f_lineno',
                 'f_code.co_names', 'f_code.co_filename']
    obj_name = 'frame'
    execstr_print_list = ['print("%r=%%r" %% (%s,))' % (_execstr, _execstr)
                          for _execstr in execstr_attr_list(obj_name, attr_list)]
    execstr = '\n'.join(execstr_print_list)
    exec(execstr)
    local_varnames = pack_into('; '.join(frame.f_locals.keys()))
    print(local_varnames)
    print('--- End Frame ---')


def search_stack_for_localvar(varname):
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


def get_stack_frame(N=0):
    frame_level0 = inspect.currentframe()
    frame_cur = frame_level0
    for _ix in range(N + 1):
        frame_next = frame_cur.f_back
        if frame_next is None:
            raise AssertionError('Frame level %r is root' % _ix)
        frame_cur = frame_next
    return frame_cur


def get_parent_frame(N=0):
    parent_frame = get_stack_frame(N=N + 2)
    return parent_frame


def get_parent_locals(N=0):
    """ returns the locals of the function that called you """
    parent_frame = get_parent_frame(N=N + 1)
    return parent_frame.f_locals


def get_parent_globals(N=0):
    parent_frame = get_stack_frame(N=N + 1)
    return parent_frame.f_globals


def get_caller_locals(N=0):
    """ returns the locals of the function that called you """
    locals_ = get_parent_locals(N=N + 1)
    return locals_


def quasiquote(string):
    return string.format(**get_caller_locals())


fmtlocals = quasiquote  # non-lispy alias for quasiquote


def get_caller_prefix(N=0, aserror=False):
    prefix_fmt = '[!%s]' if aserror else '[%s]'
    prefix = prefix_fmt % (get_caller_name(N=N + 1),)
    return prefix


def get_caller_name(N=0):
    """ returns the name of the function that called you """
    if isinstance(N, (list, tuple)):
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
    #        caller_name += '(' + get_funcname(parent_frame.f_locals['func']) + ')'
    #except Exception:
    #    pass
    if caller_name == '<module>':
        co_filename = parent_frame.f_code.co_filename
        caller_name = splitext(split(co_filename)[1])[0]
    return caller_name


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
                        truncstr = truncate_str(fullstr) + '\n'
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
            with Indenter(' * '):
                debug_list(item)
        else:
            print(' *  type(item[%d]) = %r' % (count, type(item)))


def debug_list(list_):
    dbgmessage = []
    append = dbgmessage.append
    append('debug_list')
    dim2 = None
    if all([is_listlike(item) for item in list_]):
        append(' * list items are all listlike')
        all_lens = [len(item) for item in list_]
        if list_allsame(all_lens):
            dim2 = all_lens[0]
            append(' * uniform lens=%d' % dim2)
        else:
            append(' * nonuniform lens = %r' % np.unique(all_lens).tolist())
    else:
        all_types = [type(item) for item in list_]
        if list_allsame(all_types):
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
            print('[tools] ERROR: %s(%r, %r)' % (get_funcname(func), args, kwargs))
            print('[tools] ERROR: %r' % ex)
            raise
    return ex_wrapper


def printex(ex, msg='[!?] Caught exception', prefix=None, key_list=[],
            locals_=None, iswarning=False, tb=False, separate=False, N=0,
            use_stdout=False, reraise=False, msg_=None, keys=None):
    """ Prints an exception with relevant info """
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
    if separate:
        print_func('\n\n\n')
    # print the execption
    print_func(exstr)
    if separate:
        print_func('\n\n\n')
    # If you dont know where an error is coming from be super-strict
    if SUPER_STRICT or (reraise and not iswarning):
        sys.stdout.flush()
        sys.stderr.flush()
        raise ex


def formatex(ex, msg='[!?] Caught exception',
             prefix=None, key_list=[], locals_=None, iswarning=False, tb=False, N=0):
    """ Formats an exception with relevant info """
    # Get error prefix and local info
    if prefix is None:
        prefix = get_caller_prefix(aserror=True, N=N)
    if locals_ is None:
        locals_ = get_caller_locals(N=N)
    # build exception message
    errstr_list = []  # list of exception strings
    ex_tag = 'WARNING' if iswarning else 'EXCEPTION'
    errstr_list.append('<!!! %s !!!>' % ex_tag)
    if tb:
        errstr_list.append(traceback.format_exc())
    errstr_list.append(prefix + ' ' + str(msg) + '\n%r: %s' % (type(ex), str(ex)))
    parse_locals_keylist(locals_, key_list, errstr_list, prefix)
    errstr_list.append('</!!! %s !!!>' % ex_tag)
    return '\n'.join(errstr_list)


def parse_locals_keylist(locals_, key_list, strlist_, prefix):
    """ For each key in keylist, puts its value in locals into a stringlist """
    from .util_str import get_callable_name

    def get_key_value(key):
        assert isinstance(key, str), 'must have parsed key into a string already'
        if key not in locals_:
            dotpos = key.find('.')
            if dotpos > -1:
                key_ = key[:dotpos]
                attrstr_ = key[dotpos:]
                baseval = locals_[key_]  # NOQA
                val = eval('baseval' + attrstr_)
            else:
                raise AssertionError('%s !!! %s not populated!' % (prefix, key))
        else:
            val = locals_[key]
        return val

    for key in key_list:
        try:
            if key is None:
                strlist_.append('')
            elif isinstance(key, tuple):
                tup = key
                func, key_ = tup
                #assert key_ in locals_, 'key=%r not in locals' % (key_,)
                #val = locals_[key_]
                val = get_key_value(key_)
                funcvalstr = str(func(val))
                strlist_.append('%s %s(%s) = %s' % (prefix, get_callable_name(func), key_, funcvalstr))
            #elif key in locals_:
            #    #val = locals_[key]
            #    val = get_key_value(key_)
            #    valstr = truncate_str(repr(val), maxlen=200)
            #    strlist_.append('%s %s = %s' % (prefix, key, valstr))
            else:
                val = get_key_value(key)
                valstr = truncate_str(repr(val), maxlen=200)
                strlist_.append('%s %s = %s' % (prefix, key, valstr))
        except AssertionError as ex:
            strlist_.append(str(ex))


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
        msg = horiz_string(str(key) + ' = ', repr(var))
        msg_list.append(msg)

    reprs = '\n' + indent('\n##\n'.join(msg_list)) + '\n'
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
    typestr = str(get_type(var)).ljust(typepad)

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
