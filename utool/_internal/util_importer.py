""" NEEDS CLEANUP SO IT EITHER DOES THE IMPORTS OR GENERATES THE FILE """
from __future__ import absolute_import, division, print_function
import sys
import multiprocessing
import textwrap
#import types

#----------
# EXECUTORS
#----------

def __excecute_imports(module, module_name, IMPORTS):
    """ Module Imports """
    # level: -1 is a the Python2 import strategy
    # level:  0 is a the Python3 absolute import
    level = 0
    for name in IMPORTS:
        if level == -1:
            tmp = __import__(name, globals(), locals(), fromlist=[], level=level)
        elif level == 0:
            tmp = __import__(module_name, globals(), locals(), fromlist=[name], level=level)


def __execute_fromimport(module, module_name, IMPORT_TUPLES):
    """ Module From Imports """
    FROM_IMPORTS = __get_from_imports(IMPORT_TUPLES)
    for name, fromlist in FROM_IMPORTS:
        tmp = __import__(module_name + '.' + name, globals(), locals(), fromlist=fromlist, level=0)
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

#----------
# PARSERS
#----------

def __get_imports(IMPORT_TUPLES):
    """ Returns import names
    IMPORT_TUPLES are specified as
    (name, fromlist, ispackage)
    """
    IMPORTS = [tup[0] for tup in IMPORT_TUPLES]
    return IMPORTS


def __get_from_imports(IMPORT_TUPLES):
    """ Returns import names and fromlist
    IMPORT_TUPLES are specified as
    (name, fromlist, ispackage)
    """
    FROM_IMPORTS = [(tup[0], tup[1]) for tup in IMPORT_TUPLES
                    if tup[1] is not None and len(tup[1]) > 0]
    return FROM_IMPORTS

#----------
# STRING MAKERS
#----------

def _initstr(module_name, IMPORTS, FROM_IMPORTS, inject_execstr):
    """ Calls the other string makers """
    header         = _make_module_header()
    import_str     = _make_imports_str(IMPORTS)
    fromimport_str = _make_fromimport_str(FROM_IMPORTS)
    initstr = '\n'.join([str_ for str_ in [
        header,
        import_str,
        fromimport_str,
        inject_execstr,
    ] if len(str_) > 0])
    return initstr

def _make_module_header():
    return '\n'.join([
        '# flake8: noqa',
        'from __future__ import absolute_import, division, print_function'])

def _make_imports_str(IMPORTS):
    return '\n'.join(['from . import %s' % (name,) for name in IMPORTS])

def _make_fromimport_str(FROM_IMPORTS):
    from utool import util_str
    def _pack_fromimport(tup):
        name, fromlist = tup[0], tup[1]
        from_module_str = 'from .%s import (' % name
        newline_prefix = (' ' * len(from_module_str))
        rawstr = from_module_str + ', '.join(fromlist) + ',)'
        packstr = util_str.pack_into(rawstr, textwidth=80,
                                     newline_prefix=newline_prefix)
        return packstr
    from_str = '\n'.join(map(_pack_fromimport, FROM_IMPORTS))
    return from_str

def _inject_execstr(module_name, IMPORT_TUPLES):
    # Injection and Reload String Defs
    if module_name == 'utool':
        injecter = 'util_inject'
        injecter_import = ''
    else:
        injecter_import = 'import utool'
        injecter = 'utool'
    injectstr_fmt = textwrap.dedent('''
    {injecter_import}
    print, print_, printDBG, rrr, profile = {injecter}.inject(
        __name__, '[{module_name}]')

    def reload_subs():
        """ Reloads {module_name} and submodules """
        rrr()
        {body}
        rrr()
    rrrr = reload_subs''')
    rrrdir_fmt = '    getattr(%s, \'reload_subs\', lambda: None)()'
    rrrfile_fmt = '    getattr(%s, \'rrr\', lambda: None)()'

    def _reload_command(tup):
        if len(tup) > 2 and tup[2] is True:
            return rrrdir_fmt % tup[0]
        else:
            return rrrfile_fmt % tup[0]
    body = '\n'.join(map(_reload_command, IMPORT_TUPLES)).strip()
    format_dict = {
        'module_name': module_name,
        'body': body,
        'injecter': injecter,
        'injecter_import': injecter_import,
    }
    inject_execstr = injectstr_fmt.format(**format_dict).strip()
    return inject_execstr

#----------
# PUBLIC FUNCTIONS
#----------

def dynamic_import(module_name, IMPORT_TUPLES, developing=True, dump=False):
    """
    Dynamically import listed util libraries and their members.
    Create reload_subs function.

    Using __import__ like this is typically not considered good style However,
    it is better than import * and this will generate the good file text that
    can be used when the module is "frozen"
    """
    #print('[DYNAMIC IMPORT] Running Dynamic Imports: %r ' % module_name)
    module = sys.modules[module_name]

    IMPORTS = __get_imports(IMPORT_TUPLES)
    __excecute_imports(module, module_name, IMPORTS)
    # If developing do explicit import stars
    if developing:
        FROM_IMPORTS = __execute_fromimport_star(module, module_name, IMPORT_TUPLES)
    else:
        FROM_IMPORTS = __execute_fromimport(module, module_name, IMPORT_TUPLES)

    inject_execstr = _inject_execstr(module_name, IMPORT_TUPLES)

    # If requested: print what the __init__ module should look like
    dump_requested = (('--dump-%s-init' % module_name) in sys.argv or
                      ('--print-%s-init' % module_name) in sys.argv) or dump
    if dump_requested:
        is_main_proc = multiprocessing.current_process().name == 'MainProcess'
        if is_main_proc:
            from utool import util_str
            initstr = _initstr(module_name, IMPORTS, FROM_IMPORTS, inject_execstr)
            print(util_str.indent(initstr))
    return inject_execstr


def make_initstr(module_name, IMPORT_TUPLES):
    """
    Just creates the string representation. Does no importing.
    """
    IMPORTS      = __get_imports(IMPORT_TUPLES)
    FROM_IMPORTS = __get_from_imports(IMPORT_TUPLES)
    inject_execstr = _inject_execstr(module_name, IMPORT_TUPLES)
    return _initstr(module_name, IMPORTS, FROM_IMPORTS, inject_execstr)


def make_import_tuples(module_path):
    """ Infer the IMPORT_TUPLES from a module_path """
    from utool import util_path
    kwargs = dict(private=False, full=False)
    module_list = util_path.ls_modulefiles(module_path, noext=True, **kwargs)
    package_list = util_path.ls_moduledirs(module_path, **kwargs)
    IMPORT_TUPLES = ([(modname, None, False) for modname in module_list] +
                     [(modname, None, True)  for modname in package_list])
    return IMPORT_TUPLES
