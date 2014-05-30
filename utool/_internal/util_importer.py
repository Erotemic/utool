from __future__ import absolute_import, division, print_function
#import __builtin__
import sys
import multiprocessing
import textwrap
#import types


def make_reload_subs_string(IMPORTS):
    header = '''
    def reload_subs():
        """Reloads ''' + __name__ + ''' and submodules """
        rrr()'''
    body_fmt = '''
        getattr(%s, 'rrr', lambda: None)()'''
    body = ''.join([body_fmt % (name) for name in IMPORTS])
    footer = '''
        rrr()
    rrrr = reload_subs
    '''
    reload_subs_func_str = textwrap.dedent(header + body + footer)
    return reload_subs_func_str


def __excecute_import(module, module_name, IMPORT_TUPLES):
    """ Module Imports """
    # level: -1 is a the Python2 import strategy
    # level:  0 is a the Python3 absolute import
    IMPORTS      = [name for name, fromlist in IMPORT_TUPLES]
    level = 0
    for name in IMPORTS:
        if level == -1:
            tmp = __import__(name, globals(), locals(), fromlist=[], level=level)
        elif level == 0:
            tmp = __import__(module_name, globals(), locals(), fromlist=[name], level=level)
    return IMPORTS


def __execute_fromimport(module, IMPORT_TUPLES):
    FROM_IMPORTS = [(name, fromlist) for name, fromlist in IMPORT_TUPLES
                    if fromlist is not None and len(fromlist) > 0]
    for name, fromlist in FROM_IMPORTS:
        tmp = __import__(name, globals(), locals(), fromlist=fromlist, level=-1)
        for member in fromlist:
            setattr(module, member, getattr(tmp, member))
    return FROM_IMPORTS


def __execute_fromimport_star(module, module_name, IMPORT_TUPLES):
    """ Effectively import * statements """
    FROM_IMPORTS = []
    # Explicitly ignore these special functions
    ignoreset = set(['print', 'print_', 'printDBG', 'rrr', 'profile',
                     'print_function', 'absoulte_import', 'division'])
    for name, fromlist in IMPORT_TUPLES:
        #absname = module_name + '.' + name
        other_module = sys.modules[module_name + '.' + name]
        varset = set(vars(module))
        fromset = set(fromlist) if fromlist is not None else set()
        def valid_member(member):
            """
            Guess if the member is valid based on its name
            """
            is_private = member.startswith('_')
            is_conflit = member in varset
            is_module  = member in sys.modules  # Isn't fool proof (next step is)
            is_forced  = member in fromset
            is_ignore  = member in ignoreset
            return (is_forced or not (is_ignore or is_private or is_conflit or is_module))
        fromlist_ = [member for member in dir(other_module) if valid_member(member)]
        valid_fromlist_ = []
        for member in fromlist_:
            member_val = getattr(other_module, member)
            try:
                # Disallow fromimport modules
                forced = member in fromset
                if not forced and getattr(member_val, '__name__') in sys.modules:
                    #print('not importing: %r' % member)
                    continue
            except AttributeError:
                pass
            #if isinstance(member_val, types.FunctionType):
            #    if member_val.func_globals['__name__'] != absname:
            #        print('not importing: %r' % member)
            #        continue
            #print('importing: %r' % member)
            valid_fromlist_.append(member)
            setattr(module, member, member_val)
        FROM_IMPORTS.append((name, valid_fromlist_))
    return FROM_IMPORTS


def dynamic_import(module_name, IMPORT_TUPLES, developing=True):
    """
    Dynamically import listed util libraries and their members.
    Create reload_subs function.

    Using __import__ like this is typically not considered good style However,
    it is better than import * and this will generate the good file text that
    can be used when the module is "frozen"
    """
    #__builtin__.print('[DYNAMIC IMPORT] Running Dynamic Imports')
    __PRINT_IMPORTS__ = (('--dump-%s-init' % module_name) in sys.argv or
                         ('--print-%s-init' % module_name) in sys.argv)
    module = sys.modules[module_name]

    IMPORTS = __excecute_import(module, module_name, IMPORT_TUPLES)

    if developing:
        # If developing do explicit import stars
        FROM_IMPORTS = __execute_fromimport_star(module, module_name, IMPORT_TUPLES)
    else:
        FROM_IMPORTS = __execute_fromimport(module, IMPORT_TUPLES)

    # Injection and Reload String Defs
    utool_inject_str = 'print, print_, printDBG, rrr, profile = util_inject.inject(__name__, \'[%s]\')' % module_name
    reload_subs_func_str = make_reload_subs_string(IMPORTS)
    import_execstr = utool_inject_str + reload_subs_func_str

    current_process = multiprocessing.current_process().name
    is_main_proc = current_process == 'MainProcess'
    # If requested: print what the __init__ module should look like
    if __PRINT_IMPORTS__ and is_main_proc:
        from utool import util_str
        print('')
        pack_into = util_str.pack_into
        import_str = '\n'.join(['from . import %s' % (name,) for name in IMPORTS])
        def _fromimport_str(name, fromlist):
            from_module_str = 'from .%s import (' % name
            newline_prefix = (' ' * len(from_module_str))
            rawstr = from_module_str + ', '.join(fromlist) + ',)'
            packstr = pack_into(rawstr, textwidth=80, newline_prefix=newline_prefix)
            return packstr
        from_str   = '\n'.join([_fromimport_str(name, fromlist) for (name, fromlist) in FROM_IMPORTS])
        print(util_str.indent(import_str))
        print(util_str.indent(from_str))
        print(util_str.indent(import_execstr))
        print('')
    return import_execstr
