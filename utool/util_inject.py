from __future__ import absolute_import, division, print_function
from six.moves import builtins
#import builtins
import sys
from functools import wraps
from . import util_logging
from ._internal.meta_util_six import get_funcname
from ._internal.meta_util_arg import get_argval


__AGGROFLUSH__ = '--aggroflush' in sys.argv
__LOGGING__    = '--logging'    in sys.argv
__DEBUG_ALL__  = '--debug-all'  in sys.argv
__DEBUG_PROF__ = '--debug-prof' in sys.argv or '--debug-profile' in sys.argv
QUIET = '--quiet' in sys.argv
SILENT = '--silent' in sys.argv


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
            #print('setting: %s.%s = %r' % (module.__name__, get_funcname(func), func))
            setattr(module, get_funcname(func), func)


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
    def myexcepthook(type, value, tb):
        #https://stackoverflow.com/questions/14775916/coloring-exceptions-from-python-on-a-terminal
        import traceback
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import TerminalFormatter
        tbtext = ''.join(traceback.format_exception(type, value, tb))
        lexer = get_lexer_by_name('pytb', stripall=True)
        formatter = TerminalFormatter(bg='dark')
        formatted_text = highlight(tbtext, lexer, formatter)
        sys.stderr.write(formatted_text)
    # Ignore colored exceptions on win32
    if not sys.platform.startswith('win32'):
        sys.excepthook = myexcepthook


def inject_print_functions(module_name=None, module_prefix='[???]', DEBUG=False, module=None):
    """
    makes print functions to be injected into the module
    """
    module = _get_module(module_name, module)
    if SILENT:
        def print(*args):
            """ silent print """
            pass
        def printDBG(*args):
            pass
        def print_(*args):
            pass
    else:
        def print(*args):
            """ logging print """
            util_logging.__UTOOL_PRINT__(*args)

        if __AGGROFLUSH__:
            def print_(*args):
                util_logging.__UTOOL_WRITE__(*args)
                util_logging.__UTOOL_FLUSH__()
        else:
            def print_(*args):
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
                msg = ', '.join(map(str, args))
                util_logging.__UTOOL_PRINTDBG__(module_prefix + ' DEBUG ' + msg)
        else:
            def printDBG(*args):
                pass
    _inject_funcs(module, print, print_, printDBG)
    return print, print_, printDBG


def inject_reload_function(module_name=None, module_prefix='[???]', module=None):
    """ Injects dynamic module reloading """
    module = _get_module(module_name, module, register=False)
    if module_name is None:
        module_name = str(module.__name__)
    def rrr():
        """ Dynamic module reloading """
        if not __RELOAD_OK__:
            raise Exception('Reloading has been forced off')
        try:
            import imp
            if not QUIET:
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
        with utool.Timer(get_funcname(func)):
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
#        #    #print('decorate: %r' % get_funcname(func))
#        #    # hack to filter profiled functions
#        #    if get_funcname(func)).startswith('get_affine'):
#        #        return kernprof_func(func)
#        #    return func
#        profile = kernprof_func
#        if __DEBUG_PROF__:
#            print('[util_inject] PROFILE ON: %r' % module)
#    except AttributeError:
#        # Create dummy kernprof_func
#        def profile(func):
#            #print('decorate: %r' % get_funcname(func)))
#            return func
#        if __DEBUG_PROF__:
#            print('[util_inject] PROFILE OFF: %r' % module)
#    _inject_funcs(module, profile)
#    return profile

#PROF_MOD_PAT_LIST = None  # ['spatial']
# TODO: Add this to command line

#PROF_MOD_PAT_LIST = ['spatial', 'linalg', 'keypoint']

# Look in command line for functions to profile
PROF_FUNC_PAT_LIST = get_argval('--prof-func', type_=str, default=None)
if PROF_FUNC_PAT_LIST is not None:
    PROF_FUNC_PAT_LIST = PROF_FUNC_PAT_LIST.split(',')
    print('[util_inject] PROF_FUNC_PAT_LIST: %r' % (PROF_FUNC_PAT_LIST,))

# Look in command line for modules to profile
PROF_MOD_PAT_LIST = get_argval('--prof-mod', type_=str, default=None)
if PROF_MOD_PAT_LIST is not None:
    PROF_MOD_PAT_LIST = PROF_MOD_PAT_LIST.split(',')
    print('[util_inject] PROF_MOD_PAT_LIST: %r' % (PROF_MOD_PAT_LIST,))


def memprof(func):
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
        if _profile_func_flag(get_funcname(func)):
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


PRINT_INJECT_ORDER = '--veryverbose' in sys.argv or '--print-inject-order' in sys.argv


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
    if PRINT_INJECT_ORDER:
        from ._internal import meta_util_dbg
        callername = meta_util_dbg.get_caller_name(N=2)
        builtins.print('[util_inject] {callername} is importing {modname}'.format(callername=callername, modname=module_name))
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


# Inject this module with itself!
print, print_, printDBG, rrr, profile = inject(__name__, '[inject]')
