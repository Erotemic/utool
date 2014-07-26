from __future__ import absolute_import, division, print_function
import __builtin__
import sys
from functools import wraps
from . import util_logging


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
    if argv.startswith('--debug'):
        ARGV_DEBUG_FLAGS.append(argv.replace('--debug', '').strip('-'))


#print('ARGV_DEBUG_FLAGS: %r' % (ARGV_DEBUG_FLAGS,))

#__STDOUT__ = sys.stdout
#__PRINT_FUNC__     = __builtin__.print
#__PRINT_DBG_FUNC__ = __builtin__.print
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
            #print('setting: %s.%s = %r' % (module.__name__, func.func_name, func))
            setattr(module, func.func_name, func)


def _add_injected_module(module):
    global __INJECTED_MODULES__
    __INJECTED_MODULES__.add(module)


def get_injected_modules():
    return list(__INJECTED_MODULES__)


def _get_module(module_name=None, module=None):
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
    _add_injected_module(module)
    return module


def inject_colored_exceptions():
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
    module = _get_module(module_name, module)
    if SILENT:
        def print(*args):
            pass
        def printDBG(*args):
            pass
        def print_(*args):
            pass
    else:
        def print(*args):
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
    module = _get_module(module_name, module)
    if module_name is None:
        module_name = str(module.__name__)
    def rrr():
        """ Dynamic module reloading """
        if not __RELOAD_OK__:
            raise Exception('Reloading has been forced off')
        try:
            import imp
            if not QUIET:
                __builtin__.print('RELOAD: ' + str(module_prefix) + ' __name__=' + module_name)
            imp.reload(module)
        except Exception as ex:
            print(ex)
            print('%s Failed to reload' % module_prefix)
            raise
    _inject_funcs(module, rrr)
    return rrr


def DUMMYPROF_FUNC(func):
    return func

def TIMERPROF_FUNC(func):
    @wraps(func)
    def prof_wrapper(*args, **kwargs):
        import utool
        with utool.Timer(func.func_name):
            return func(*args, **kwargs)
        #return ret
    return prof_wrapper

try:
    #KERNPROF_FUNC = TIMERPROF_FUNC
    KERNPROF_FUNC = getattr(__builtin__, 'profile')
    PROFILING = True
except AttributeError:
    PROFILING = False
    KERNPROF_FUNC = DUMMYPROF_FUNC
    #KERNPROF_FUNC = TIMERPROF_FUNC


#def inject_profile_function(module_name=None, module_prefix='[???]', module=None):
#    module = _get_module(module_name, module)
#    try:
#        kernprof_func = getattr(__builtin__, 'profile')
#        #def profile(func):
#        #    #print('decorate: %r' % func.func_name)
#        #    # hack to filter profiled functions
#        #    if func.func_name.startswith('get_affine'):
#        #        return kernprof_func(func)
#        #    return func
#        profile = kernprof_func
#        #filtered_profile.func_name = 'profile'
#        if __DEBUG_PROF__:
#            print('[util_inject] PROFILE ON: %r' % module)
#    except AttributeError:
#        # Create dummy kernprof_func
#        def profile(func):
#            #print('decorate: %r' % func.func_name)
#            return func
#        if __DEBUG_PROF__:
#            print('[util_inject] PROFILE OFF: %r' % module)
#    _inject_funcs(module, profile)
#    return profile

PROF_FUNC_PAT_LIST = None
PROF_MOD_PAT_LIST = None  # ['spatial']
# TODO: Add this to command line

from ._internal.meta_util_arg import get_arg
#PROF_MOD_PAT_LIST = ['spatial', 'linalg', 'keypoint']
PROF_MOD_PAT_LIST = get_arg('--prof-mod', type_=str, default=None)
if PROF_MOD_PAT_LIST is not None:
    PROF_MOD_PAT_LIST = PROF_MOD_PAT_LIST.split(';')
    #print('PROF_MOD_PAT_LIST: %r' % (PROF_MOD_PAT_LIST,))
    #sys.exit(1)
#else:
#    print('PROF_MOD_PAT_LIST: %r' % (PROF_MOD_PAT_LIST,))
#    sys.exit(1)
del get_arg


def _matches_list(name, pat_list):
    return any([name.find(pat) != -1 for pat in pat_list])


def _profile_func_flag(func_name):
    if PROF_FUNC_PAT_LIST is None:
        return True
    return _matches_list(func_name, PROF_FUNC_PAT_LIST)


def _profile_module_flag(module_name):
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
        if _profile_func_flag(func.func_name):
            return KERNPROF_FUNC(func)
        return func
    #profile = KERNPROF_FUNC
    #try:
    #    profile = getattr(__builtin__, 'profile')
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


def inject(module_name=None, module_prefix='[???]', DEBUG=False, module=None):
    """
    Usage:
        from __future__ import absolute_import, division, print_function
        from util.util_inject import inject
        print, print_, printDBG, rrr, profile = inject(__name__, '[mod]')
    """
    module = _get_module(module_name, module)
    rrr         = inject_reload_function(None, module_prefix, module)
    profile_    = inject_profile_function(None, module_prefix, module)
    print_funcs = inject_print_functions(None, module_prefix, DEBUG, module)
    print, print_, printDBG = print_funcs
    return print, print_, printDBG, rrr, profile_


def inject_all(DEBUG=False):
    """
    Injects the print, print_, printDBG, rrr, and profile functions into all
    loaded modules
    """
    raise NotImplemented('!!!')
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


print, print_, printDBG, rrr, profile = inject(__name__, '[inject]')
