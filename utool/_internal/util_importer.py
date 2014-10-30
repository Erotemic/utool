""" NEEDS CLEANUP SO IT EITHER DOES THE IMPORTS OR GENERATES THE FILE """
from __future__ import absolute_import, division, print_function
import sys
import multiprocessing
import textwrap
#import types

DEBUG_IMPORTS = False
#----------
# EXECUTORS
#----------

def __excecute_imports(module, module_name, IMPORTS):
    """ Module Imports """
    # level: -1 is a the Python2 import strategy
    # level:  0 is a the Python3 absolute import
    if DEBUG_IMPORTS:
        print('[UTIL_IMPORT] EXECUTING IMPORT')
    level = 0
    for name in IMPORTS:
        if level == -1:
            tmp = __import__(name, globals(), locals(), fromlist=[], level=level)
        elif level == 0:
            tmp = __import__(module_name, globals(), locals(), fromlist=[name], level=level)


def __execute_fromimport(module, module_name, IMPORT_TUPLES):
    """ Module From Imports """
    if DEBUG_IMPORTS:
        print('[UTIL_IMPORT] EXECUTING FROM STAR')
    FROM_IMPORTS = __get_from_imports(IMPORT_TUPLES)
    for name, fromlist in FROM_IMPORTS:
        tmp = __import__(module_name + '.' + name, globals(), locals(), fromlist=fromlist, level=0)
        for attrname in fromlist:
            setattr(module, attrname, getattr(tmp, attrname))
    return FROM_IMPORTS


def __execute_fromimport_star(module, module_name, IMPORT_TUPLES):
    """ Effectively import * statements """
    if DEBUG_IMPORTS:
        print('[UTIL_IMPORT] EXECUTE FROMIMPORT STAR.')
    FROM_IMPORTS = []
    # Explicitly ignore these special functions (usually stdlib functions)
    ignoreset = set(['print', 'print_', 'printDBG', 'rrr', 'profile',
                     'print_function', 'absoulte_import', 'division', 'zip',
                     'map', 'range', 'list', 'zip_longest', 'filter', 'filterfalse',
                     'dirname', 'realpath', 'join', 'exists', 'normpath',
                     'splitext', 'expanduser', 'relpath', 'isabs',
                     'commonprefix', 'basename' ])
                     #'isdir', 'isfile', '
    for name, fromlist in IMPORT_TUPLES:
        #absname = module_name + '.' + name
        other_module = sys.modules[module_name + '.' + name]
        varset = set(vars(module))
        fromset = set(fromlist) if fromlist is not None else set()
        def valid_attrname(attrname):
            """
            Guess if the attrname is valid based on its name
            """
            is_private = attrname.startswith('_')
            is_conflit = attrname in varset
            is_module  = attrname in sys.modules  # Isn't fool proof (next step is)
            is_forced  = attrname in fromset
            is_ignore  = attrname in ignoreset
            return (is_forced or not (is_ignore or is_private or is_conflit or is_module))
        fromlist_ = [attrname for attrname in dir(other_module) if valid_attrname(attrname)]
        valid_fromlist_ = []
        for attrname in fromlist_:
            attrval = getattr(other_module, attrname)
            try:
                # Disallow fromimport modules
                forced = attrname in fromset
                if not forced and getattr(attrval, '__name__') in sys.modules:
                    if DEBUG_IMPORTS:
                        print('[UTIL_IMPORT] not importing: %r' % attrname)
                    continue
            except AttributeError:
                pass
            #if isinstance(attrval, types.FunctionType):
            #    if attrval.func_globals['__name__'] != absname:
            #        print('not importing: %r' % attrname)
            #        continue
            #print('importing: %r' % attrname)
            if DEBUG_IMPORTS:
                print('[UTIL_IMPORT] importing: %r' % attrname)
            valid_fromlist_.append(attrname)
            setattr(module, attrname, attrval)
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

def _initstr(module_name, IMPORTS, FROM_IMPORTS, inject_execstr, withheader=True):
    """ Calls the other string makers """
    header         = _make_module_header() if withheader else ''
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
        if len(fromlist) > 0:
            rawstr = from_module_str + ', '.join(fromlist) + ',)'
        else:
            rawstr = ''

        # not sure why this isn't 76? >= maybe?
        packstr = util_str.pack_into(rawstr, textwidth=75,
                                     newline_prefix=newline_prefix,
                                    break_words=False)
        return packstr
    from_str = '\n'.join(map(_pack_fromimport, FROM_IMPORTS))
    return from_str

def _inject_execstr(module_name, IMPORT_TUPLES):
    """ Injection and Reload String Defs """
    if module_name == 'utool':
        # Special case import of the util_inject module
        injecter = 'util_inject'
        injecter_import = ''
    else:
        # Normal case implicit import of util_inject
        injecter_import = 'import utool'
        injecter = 'utool'
    injectstr_fmt = textwrap.dedent('''
    {injecter_import}
    print, print_, printDBG, rrr, profile = {injecter}.inject(
        __name__, '[{module_name}]')


    def reassign_submodule_attributes():
        """
        why reloading all the modules doesnt do this I don't know
        """
        import sys
        if '--quiet' not in sys.argv:
            print('dev reimport')
        # Self import
        import {module_name}
        # Implicit reassignment.
        for submodname, fromimports in IMPORT_TUPLES:
            submod = getattr({module_name}, submodname)
            for attr in dir(submod):
                if attr.startswith('_'):
                    continue
                setattr({module_name}, attr, getattr(submod, attr))


    def reload_subs():
        """ Reloads {module_name} and submodules """
        rrr()
        {body}
        rrr()
        try:
            # hackish way of propogating up the new reloaded submodule attributes
            reassign_submodule_attributes()
        except Exception:
            pass
    rrrr = reload_subs''')
    rrrdir_fmt  = '    getattr(%s, \'reload_subs\', lambda: None)()'
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
    Dynamically import listed util libraries and their attributes.
    Create reload_subs function.

    Using __import__ like this is typically not considered good style However,
    it is better than import * and this will generate the good file text that
    can be used when the module is "frozen"
    """
    if DEBUG_IMPORTS:
        print('[UTIL_IMPORT] Running Dynamic Imports: %r ' % module_name)
    # Get the module that will be imported into
    module = sys.modules[module_name]
    # List of modules to be imported
    IMPORTS = __get_imports(IMPORT_TUPLES)
    # Import the modules
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
    overwrite_requested = ('--update-%s-init' % module_name) in sys.argv

    if dump_requested:
        is_main_proc = multiprocessing.current_process().name == 'MainProcess'
        if is_main_proc:
            from utool import util_str
            initstr = _initstr(module_name, IMPORTS, FROM_IMPORTS, inject_execstr)
            print(util_str.indent(initstr))
    # Overwrite the __init__.py file with new explicit imports
    if overwrite_requested:
        is_main_proc = multiprocessing.current_process().name == 'MainProcess'
        if is_main_proc:
            from utool import util_str
            from os.path import join, exists
            initstr = _initstr(module_name, IMPORTS, FROM_IMPORTS, inject_execstr, withheader=False)
            new_else = util_str.indent(initstr)
            #print(new_else)
            # Get path to init file so we can overwrite it
            init_fpath = join(module.__path__[0], '__init__.py')
            print("attempting to update: %r" % init_fpath)
            assert exists(init_fpath)
            new_lines = []
            broken = False
            with open(init_fpath, 'r') as file_:
                lines = file_.readlines()
                for line in lines:
                    new_lines.append(line)
                    if line.strip().startswith('# <AUTOGEN_INIT>'):
                        new_lines.append('\n' + new_else + '\n    # </AUTOGEN_INIT>')
                        broken = True
                        break
            if broken:
                print("writing updated file: %r" % init_fpath)
                new_text = ''.join(new_lines)
                with open(init_fpath, 'w') as file_:
                    file_.write(new_text)
            else:
                print("no write hook for file: %r" % init_fpath)

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
