"""
Injects code into live modules or into text source files
"""
from __future__ import absolute_import, division, print_function
from six.moves import builtins
#import builtins
import sys
from functools import wraps
from utool._internal import meta_util_six
from utool._internal import meta_util_arg
from utool import util_logging
try:
    import traceback
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import TerminalFormatter
    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False


__AGGROFLUSH__ = '--aggroflush' in sys.argv
__LOGGING__    = '--logging'    in sys.argv
__DEBUG_ALL__  = '--debug-all'  in sys.argv
__DEBUG_PROF__ = '--debug-prof' in sys.argv or '--debug-profile' in sys.argv
DEBUG_PRINT = '--debug-print' in sys.argv
QUIET = '--quiet' in sys.argv
SILENT = '--silent' in sys.argv
VERYVERBOSE = meta_util_arg.VERYVERBOSE
PRINT_INJECT_ORDER = meta_util_arg.PRINT_INJECT_ORDER


if __LOGGING__:
    util_logging.start_logging()

# Read all flags with --debug in them
ARGV_DEBUG_FLAGS = []
for argv in sys.argv:
    if argv.startswith('--debug-'):
        ARGV_DEBUG_FLAGS.append(argv.replace('--debug-', '').replace('-', '_'))


#print('ARGV_DEBUG_FLAGS: %r' % (ARGV_DEBUG_FLAGS,))

#__STDOUT__ = sys.stdout
#__PRINT_FUNC__     = builtins.print
#__PRINT_DBG_FUNC__ = builtins.print
#__WRITE_FUNC__ = __STDOUT__.write
#__FLUSH_FUNC__ = __STDOUT__.flush
__RELOAD_OK__  = '--noreloadable' not in sys.argv


__INJECTED_MODULES__ = set([])

# Do not inject into these modules
__INJECT_BLACKLIST__ = frozenset(['tri', 'gc', 'sys', 'string', 'types', '_dia', 'responce', 'six', __name__])


def _inject_funcs(module, *func_list):
    for func in func_list:
        if (module is not None and
                hasattr(module, '__name__') and
                module.__name__ not in __INJECT_BLACKLIST__ and
                not module.__name__.startswith('six') and
                not module.__name__.startswith('sys')):
            #print('setting: %s.%s = %r' % (module.__name__, meta_util_six.get_funcname(func), func))
            setattr(module, meta_util_six.get_funcname(func), func)


def _add_injected_module(module):
    global __INJECTED_MODULES__
    __INJECTED_MODULES__.add(module)


def get_injected_modules():
    return list(__INJECTED_MODULES__)


def _get_module(module_name=None, module=None, register=True):
    """ finds module in sys.modules based on module name unless the module has
    already been found and is passed in """
    if module is None and module_name is not None:
        try:
            module = sys.modules[module_name]
        except KeyError as ex:
            print(ex)
            raise KeyError(('module_name=%r must be loaded before ' +
                            'receiving injections') % module_name)
    elif module is not None and module_name is None:
        pass
    else:
        raise ValueError('module_name or module must be exclusively specified')
    if register is True:
        _add_injected_module(module)
    return module


def inject_colored_exceptions():
    """
    Causes exceptions to be colored if not already

    Hooks into sys.excepthook
    """
    #COLORED_INJECTS = '--nocolorex' not in sys.argv
    #COLORED_INJECTS = '--colorex' in sys.argv
    def myexcepthook(type, value, tb):
        #https://stackoverflow.com/questions/14775916/coloring-exceptions-from-python-on-a-terminal
        tbtext = ''.join(traceback.format_exception(type, value, tb))
        lexer = get_lexer_by_name('pytb', stripall=True)
        formatter = TerminalFormatter(bg='dark')
        formatted_text = highlight(tbtext, lexer, formatter)
        sys.stderr.write(formatted_text)
    # Ignore colored exceptions on win32
    if HAS_PYGMENTS and not sys.platform.startswith('win32'):
        sys.excepthook = myexcepthook


def inject_print_functions(module_name=None, module_prefix='[???]', DEBUG=False, module=None):
    """
    makes print functions to be injected into the module
    """
    module = _get_module(module_name, module)
    if SILENT:
        def print(*args):
            """ silent builtins.print """
            pass
        def printDBG(*args):
            """ silent debug print """
            pass
        def print_(*args):
            """ silent stdout.write """
            pass
    else:
        if DEBUG_PRINT:
            # Turns on printing where a message came from
            def print(*args):
                """ debugging logging builtins.print """
                from utool._internal.meta_util_dbg import get_caller_name
                calltag = ''.join(('[caller:', get_caller_name(), ']' ))
                util_logging.__UTOOL_PRINT__(calltag, *args)
        else:
            def print(*args):
                """ logging builtins.print """
                util_logging.__UTOOL_PRINT__(*args)

        if __AGGROFLUSH__:
            def print_(*args):
                """ aggressive logging stdout.write """
                util_logging.__UTOOL_WRITE__(*args)
                util_logging.__UTOOL_FLUSH__()
        else:
            def print_(*args):
                """ logging stdout.write """
                util_logging.__UTOOL_WRITE__(*args)

        # turn on module debugging with command line flags
        dotpos = module.__name__.rfind('.')
        if dotpos == -1:
            module_name = module.__name__
        else:
            module_name = module.__name__[dotpos + 1:]
        def _replchars(str_):
            return str_.replace('_', '-').replace(']', '').replace('[', '')
        flag1 = '--debug-%s' % _replchars(module_name)
        flag2 = '--debug-%s' % _replchars(module_prefix)
        DEBUG_FLAG = any([flag in sys.argv for flag in [flag1, flag2]])
        for curflag in ARGV_DEBUG_FLAGS:
            if curflag in module_prefix:
                DEBUG_FLAG = True
        if __DEBUG_ALL__ or DEBUG or DEBUG_FLAG:
            print('INJECT_PRINT: %r == %r' % (module_name, module_prefix))
            def printDBG(*args):
                """ debug logging print """
                msg = ', '.join(map(str, args))
                util_logging.__UTOOL_PRINTDBG__(module_prefix + ' DEBUG ' + msg)
        else:
            def printDBG(*args):
                """ silent debug logging print """
                pass
    _inject_funcs(module, print, print_, printDBG)
    print_funcs = (print, print_, printDBG)
    return print_funcs


def inject_reload_function(module_name=None, module_prefix='[???]', module=None):
    """ Injects dynamic module reloading """
    module = _get_module(module_name, module, register=False)
    if module_name is None:
        module_name = str(module.__name__)
    def rrr(verbose=True):
        """ Dynamic module reloading """
        if not __RELOAD_OK__:
            raise Exception('Reloading has been forced off')
        try:
            import imp
            if verbose and not QUIET:
                builtins.print('RELOAD: ' + str(module_prefix) + ' __name__=' + module_name)
            imp.reload(module)
        except Exception as ex:
            print(ex)
            print('%s Failed to reload' % module_prefix)
            raise
    # this doesn't seem to set anything on import *
    #rrr.__dict__['module_name'] = module_name
    #rrr.__dict__['module_prefix'] = module_prefix
    #print(id(rrr))
    #print('module_name = %r' % module_name)
    #print('module_prefix = %r' % module_prefix)
    #print('rrr.__dict__ = %r' % (rrr.__dict__,))
    _inject_funcs(module, rrr)
    return rrr


def DUMMYPROF_FUNC(func):
    """ dummy profiling func. does nothing """
    return func


def TIMERPROF_FUNC(func):
    @wraps(func)
    def prof_wrapper(*args, **kwargs):
        import utool
        with utool.Timer(meta_util_six.get_funcname(func)):
            return func(*args, **kwargs)
        #return ret
    return prof_wrapper

try:
    #KERNPROF_FUNC = TIMERPROF_FUNC
    # TODO: Fix this in case using memprof instead
    #raise AttributeError('')
    KERNPROF_FUNC = getattr(builtins, 'profile')
    PROFILING = True
except AttributeError:
    PROFILING = False
    KERNPROF_FUNC = DUMMYPROF_FUNC
    #KERNPROF_FUNC = TIMERPROF_FUNC


#def inject_profile_function(module_name=None, module_prefix='[???]', module=None):
#    module = _get_module(module_name, module)
#    try:
#        kernprof_func = getattr(builtins, 'profile')
#        #def profile(func):
#        #    #print('decorate: %r' % meta_util_six.get_funcname(func))
#        #    # hack to filter profiled functions
#        #    if meta_util_six.get_funcname(func)).startswith('get_affine'):
#        #        return kernprof_func(func)
#        #    return func
#        profile = kernprof_func
#        if __DEBUG_PROF__:
#            print('[util_inject] PROFILE ON: %r' % module)
#    except AttributeError:
#        # Create dummy kernprof_func
#        def profile(func):
#            #print('decorate: %r' % meta_util_six.get_funcname(func)))
#            return func
#        if __DEBUG_PROF__:
#            print('[util_inject] PROFILE OFF: %r' % module)
#    _inject_funcs(module, profile)
#    return profile

#PROF_MOD_PAT_LIST = None  # ['spatial']
# TODO: Add this to command line

#PROF_MOD_PAT_LIST = ['spatial', 'linalg', 'keypoint']

# Look in command line for functions to profile
PROF_FUNC_PAT_LIST = meta_util_arg.get_argval('--prof-func', type_=str, default=None)
if PROF_FUNC_PAT_LIST is not None:
    PROF_FUNC_PAT_LIST = PROF_FUNC_PAT_LIST.split(',')
    print('[util_inject] PROF_FUNC_PAT_LIST: %r' % (PROF_FUNC_PAT_LIST,))

# Look in command line for modules to profile
PROF_MOD_PAT_LIST = meta_util_arg.get_argval('--prof-mod', type_=str, default=None)
if PROF_MOD_PAT_LIST is not None:
    PROF_MOD_PAT_LIST = PROF_MOD_PAT_LIST.split(',')
    print('[util_inject] PROF_MOD_PAT_LIST: %r' % (PROF_MOD_PAT_LIST,))


def memprof(func):
    """ requires memory_profiler
    pip install memory_profiler

    References:
        https://pypi.python.org/pypi/memory_profiler

    """
    import memory_profiler
    return memory_profiler.profile(func)


def _matches_list(name, pat_list):
    return any([name.find(pat) != -1 for pat in pat_list])


def _profile_func_flag(funcname):
    """ checks if func has been requested to be profiled """
    if PROF_FUNC_PAT_LIST is None:
        return True
    return _matches_list(funcname, PROF_FUNC_PAT_LIST)


def _profile_module_flag(module_name):
    """ checks if module has been requested to be profiled """
    if PROF_MOD_PAT_LIST is None:
        return True
    return _matches_list(module_name, PROF_MOD_PAT_LIST)


def inject_profile_function(module_name=None, module_prefix='[???]', module=None):
    # FIXME: not injecting right
    module = _get_module(module_name, module)
    if not _profile_module_flag(str(module)):
        return DUMMYPROF_FUNC
    #if module_name is None:
    #    return DUMMYPROF_FUNC
    #profile_module_flag = PROF_MODULE_PAT is None or module_name.startswith(PROF_MODULE_PAT)
    #if not profile_module_flag:
    #    return DUMMYPROF_FUNC

    def profile_withfuncname_filter(func):
        # Test to see if this function is specified
        if _profile_func_flag(meta_util_six.get_funcname(func)):
            return KERNPROF_FUNC(func)
        return func
    #profile = KERNPROF_FUNC
    #try:
    #    profile = getattr(builtins, 'profile')
    #    if __DEBUG_PROF__:
    #        print('[util_inject] PROFILE ON: %r' % module)
    #    return profile
    #except AttributeError:
    #    def profile(func):
    #        return func
    #    if __DEBUG_PROF__:
    #        print('[util_inject] PROFILE OFF: %r' % module)
    #_inject_funcs(module, profile)
    return profile_withfuncname_filter


def noinject(module_name=None, module_prefix='[???]', DEBUG=False, module=None, N=0):
    """
    Use in modules that do not have inject in them

    Does not inject anything into the module. Just lets utool know that a module
    is being imported so the import order can be debuged
    """
    if PRINT_INJECT_ORDER:
        from utool._internal import meta_util_dbg
        callername = meta_util_dbg.get_caller_name(N=2 + N, strict=False)
        lineno = meta_util_dbg.get_caller_lineno(N=2 + N, strict=False)
        fmtdict = dict(N=N, lineno=lineno, callername=callername, modname=module_name)
        msg = '[util_inject] N={N} {modname} is imported by {callername} at lineno={lineno}'.format(**fmtdict)
        builtins.print(msg)


def inject(module_name=None, module_prefix='[???]', DEBUG=False, module=None):
    """
    Injects your module with utool magic

    Utool magic is not actually magic. It just turns your ``print`` statments into
    logging statments, allows for your module to be used with the utool.Indent
    context manager and the and utool.indent_func decorator. ``printDBG`` will soon
    be deprecated as will ``print_``. The function rrr is a developer convinience for
    reloading the module dynamically durring runtime. The profile decorator is
    a no-op if not using kernprof.py, otherwise it is kernprof.py's profile
    decorator.

    Args:
        module_name (str): the __name__ varaible in your module
        module_prefix (str): a user defined module prefix
        DEBUG (bool):
        module (None): the actual module (optional)

    Returns:
        tuple : (print, print_, printDBG, rrr, profile_)

    Example:
        >>> from utool.util_inject import *  # NOQA
        >>> from __future__ import absolute_import, division, print_function
        >>> from util.util_inject import inject
        >>> print, print_, printDBG, rrr, profile = inject(__name__, '[mod]')
    """
    noinject(module_name, module_prefix, DEBUG, module, N=1)
    module = _get_module(module_name, module)
    rrr         = inject_reload_function(None, module_prefix, module)
    profile_    = inject_profile_function(None, module_prefix, module)
    print_funcs = inject_print_functions(None, module_prefix, DEBUG, module)
    (print, print_, printDBG) = print_funcs
    return (print, print_, printDBG, rrr, profile_)


def inject_all(DEBUG=False):
    """
    UNFINISHED. DO NOT USE

    Injects the ``print``, ``print_``, ``printDBG``, rrr, and profile functions into all
    loaded modules
    """
    raise NotImplementedError('!!!')
    for key, module in sys.modules.items():
        if module is None or not hasattr(module, '__name__'):
            continue
        try:
            module_prefix = '[%s]' % key
            inject(module_name=key, module_prefix=module_prefix, DEBUG=DEBUG)
        except Exception as ex:
            print('<!!!>')
            print('[util_inject] Cannot Inject: %s: %s' % (type(ex), ex))
            print('[util_inject] key=%r' % key)
            print('[util_inject] module=%r' % module)
            print('</!!!>')
            raise


def split_python_text_into_lines(text):
    """
    # TODO: make it so this function returns text so one statment is on one line
    # that means no splitting up things like function definitions into multiple
    # lines
    """
    #import jedi
    #script = jedi.Script(text, line=1, column=None, path='')
    def parentesis_are_balanced(line):
        """
        helper

        References:
            http://stackoverflow.com/questions/18007995/recursive-method-for-parentheses-balancing-python
        """
        def balanced(str_, i=0, cnt=0, left='(', right=')'):
            if i == len(str_):
                return cnt == 0
            if cnt < 0:
                return False
            if str_[i] == left:
                return  balanced(str_, i + 1, cnt + 1)
            elif str_[i] == right:
                return  balanced(str_, i + 1, cnt - 1)
            return balanced(str_, i + 1, cnt)
        return balanced(line)

    lines = text.split('\n')
    new_lines = []
    current_line = ''
    for line in lines:
        current_line += line
        if parentesis_are_balanced(current_line):
            new_lines.append(current_line)
            current_line = ''
    return lines


def inject_python_code(fpath, patch_code, tag=None, inject_location='after_imports'):
    """ puts code into files on disk """
    import utool as ut
    assert tag is not None, 'TAG MUST BE SPECIFIED IN INJECTED CODETEXT'
    text = ut.read_from(fpath)
    comment_start_tag = '# <util_inject:%s>' % tag
    comment_end_tag  = '# </util_inject:%s>' % tag

    tagstart_txtpos = text.find(comment_start_tag)
    tagend_txtpos = text.find(comment_end_tag)

    text_lines = ut.split_python_text_into_lines(text)

    # split the file into two parts and inject code between them
    if tagstart_txtpos != -1 or tagend_txtpos != -1:
        assert tagstart_txtpos != -1, 'both tags must not be found'
        assert tagend_txtpos != -1, 'both tags must not be found'

        for pos, line in enumerate(text_lines):
            if line.startswith(comment_start_tag):
                tagstart_pos = pos
            if line.startswith(comment_end_tag):
                tagend_pos = pos
        part1 = text_lines[0:tagstart_pos]
        part2 = text_lines[tagend_pos + 1:]
    else:
        if inject_location == 'after_imports':
            first_nonimport_pos = 0
            for line in text_lines:
                list_ = ['import ', 'from ', '#', ' ']
                isvalid = len(line) == 0 or any([line.startswith(str_) for str_ in list_])
                if not isvalid:
                    break
                first_nonimport_pos += 1
            part1 = text_lines[0:first_nonimport_pos]
            part2 = text_lines[first_nonimport_pos:]
        else:
            raise AssertionError('Unknown inject location')

    newtext = (
        '\n'.join(part1 + [comment_start_tag]) +
        '\n' + patch_code + '\n' +
        '\n'.join( [comment_end_tag] + part2)
    )
    text_backup_fname = fpath + '.' + ut.get_timestamp() + '.bak'
    ut.write_to(text_backup_fname, text)
    ut.write_to(fpath, newtext)
    #print(newtext)


if '--inject-color' in sys.argv:
    inject_colored_exceptions()

# Inject this module with itself!
print, print_, printDBG, rrr, profile = inject(__name__, '[inject]')
