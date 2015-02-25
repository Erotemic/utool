""" NEEDS CLEANUP SO IT EITHER DOES THE IMPORTS OR GENERATES THE FILE """
from __future__ import absolute_import, division, print_function
import sys
import multiprocessing
import textwrap
#import types

#DEBUG_IMPORTS = '--debug-imports' in sys.argv
#----------
# EXECUTORS
#----------

def __excecute_imports(module, modname, IMPORTS, verbose=False):
    """ Module Imports """
    # level: -1 is a the Python2 import strategy
    # level:  0 is a the Python3 absolute import
    if verbose:
        print('[UTIL_IMPORT] EXECUTING %d IMPORT TUPLES' % (len(IMPORTS),))
    level = 0
    for name in IMPORTS:
        if level == -1:
            tmp = __import__(name, globals(), locals(), fromlist=[], level=level)
        elif level == 0:
            tmp = __import__(modname, globals(), locals(), fromlist=[name], level=level)


def __execute_fromimport(module, modname, IMPORT_TUPLES, verbose=False):
    """ Module From Imports """
    if verbose:
        print('[UTIL_IMPORT] EXECUTING %d FROM IMPORT TUPLES' % (len(IMPORT_TUPLES),))
    FROM_IMPORTS = __get_from_imports(IMPORT_TUPLES)
    for name, fromlist in FROM_IMPORTS:
        full_modname = '.'.join((modname, name))
        tmp = __import__(full_modname, globals(), locals(), fromlist=fromlist, level=0)
        for attrname in fromlist:
            setattr(module, attrname, getattr(tmp, attrname))
    return FROM_IMPORTS


def __execute_fromimport_star(module, modname, IMPORT_TUPLES, ignore_list=[],
                              ignore_startswith=[], ignore_endswith=[],
                              verbose=False, veryverbose=False):
    """ Effectively import * statements

    The dynamic_import must happen before any * imports otherwise it wont catch
    anything.
    """
    if verbose:
        print('[UTIL_IMPORT] EXECUTE %d FROMIMPORT STAR TUPLES.' % (len(IMPORT_TUPLES),))
    FROM_IMPORTS = []
    # Explicitly ignore these special functions (usually stdlib functions)
    ignoreset = set(['print', 'print_', 'printDBG', 'rrr', 'profile',
                     'print_function', 'absoulte_import', 'division', 'zip',
                     'map', 'range', 'list', 'zip_longest', 'filter', 'filterfalse',
                     'dirname', 'realpath', 'join', 'exists', 'normpath',
                     'splitext', 'expanduser', 'relpath', 'isabs',
                     'commonprefix', 'basename', 'input', 'reduce',
                     #'OrderedDict',
                     #'product',
                     ] + ignore_list)
                     #'isdir', 'isfile', '
    for name, fromlist in IMPORT_TUPLES:
        #absname = modname + '.' + name
        child_module = sys.modules[modname + '.' + name]
        varset = set(vars(module))
        fromset = set(fromlist) if fromlist is not None else set()
        def valid_attrname(attrname):
            """
            Guess if the attrname is valid based on its name
            """
            is_forced  = attrname in fromset
            is_private = attrname.startswith('_')
            is_conflit = attrname in varset
            is_module  = attrname in sys.modules  # Isn't fool proof (next step is)
            is_ignore1 = attrname in ignoreset
            is_ignore2 = any([attrname.startswith(prefix) for prefix in ignore_startswith])
            is_ignore3 = any([attrname.endswith(suffix) for suffix in ignore_endswith])
            is_ignore  = any((is_ignore1, is_ignore2, is_ignore3))
            is_valid = not any((is_ignore, is_private, is_conflit, is_module))
            return (is_forced or is_valid)
        allattrs = dir(child_module)
        fromlist_ = [attrname for attrname in allattrs if valid_attrname(attrname)]
        #if verbose:
        #    print('[UTIL_IMPORT]     name=%r, len(allattrs)=%d' % (name, len(allattrs)))
        #if verbose:
        #    print('[UTIL_IMPORT]     name=%r, len(fromlist_)=%d' % (name, len(fromlist_)))
        valid_fromlist_ = []
        for attrname in fromlist_:
            attrval = getattr(child_module, attrname)
            try:
                # Disallow fromimport modules
                forced = attrname in fromset
                if not forced and getattr(attrval, '__name__') in sys.modules:
                    if veryverbose:
                        print('[UTIL_IMPORT] not importing: %r' % attrname)
                    continue
            except AttributeError:
                pass
            if veryverbose:
                print('[UTIL_IMPORT] importing: %r' % attrname)
            valid_fromlist_.append(attrname)
            setattr(module, attrname, attrval)
        if verbose:
            print('[UTIL_IMPORT]     name=%r, len(valid_fromlist_)=%d' % (name, len(valid_fromlist_)))
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

def _initstr(modname, IMPORTS, FROM_IMPORTS, inject_execstr, withheader=True):
    """ Calls the other string makers """
    header         = _make_module_header() if withheader else ''
    import_str     = _make_imports_str(IMPORTS, modname)
    fromimport_str = _make_fromimport_str(FROM_IMPORTS, modname)
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

def _make_imports_str(IMPORTS, rootmodname='.'):
    imports_fmtstr = 'from {rootmodname} import %s'.format(rootmodname=rootmodname)
    return '\n'.join([imports_fmtstr % (name,) for name in IMPORTS])

def _make_fromimport_str(FROM_IMPORTS, rootmodname='.'):
    from utool import util_str
    if rootmodname == '.':
        # dot is already taken care of in fmtstr
        rootmodname = ''
    def _pack_fromimport(tup):
        name, fromlist = tup[0], tup[1]
        from_module_str = 'from {rootmodname}.{name} import ('.format(rootmodname=rootmodname, name=name)
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

def _inject_execstr(modname, IMPORT_TUPLES):
    """ Injection and Reload String Defs """
    if modname == 'utool':
        # Special case import of the util_inject module
        injecter = 'util_inject'
        injecter_import = ''
    else:
        # Normal case implicit import of util_inject
        injecter_import = 'import utool'
        injecter = 'utool'
    injectstr_fmt = textwrap.dedent(r'''
    # STARTBLOCK
    {injecter_import}
    print, print_, printDBG, rrr, profile = {injecter}.inject(
        __name__, '[{modname}]')


    def reassign_submodule_attributes(verbose=True):
        """
        why reloading all the modules doesnt do this I don't know
        """
        import sys
        if verbose and '--quiet' not in sys.argv:
            print('dev reimport')
        # Self import
        import {modname}
        # Implicit reassignment.
        seen_ = set([])
        for tup in IMPORT_TUPLES:
            if len(tup) > 2 and tup[2]:
                continue  # dont import package names
            submodname, fromimports = tup[0:2]
            submod = getattr({modname}, submodname)
            for attr in dir(submod):
                if attr.startswith('_'):
                    continue
                if attr in seen_:
                    # This just holds off bad behavior
                    # but it does mimic normal util_import behavior
                    # which is good
                    continue
                seen_.add(attr)
                setattr({modname}, attr, getattr(submod, attr))


    def reload_subs(verbose=True):
        """ Reloads {modname} and submodules """
        rrr(verbose=verbose)
        def fbrrr(*args, **kwargs):
            """ fallback reload """
            pass
        {reload_body}
        rrr(verbose=verbose)
        try:
            # hackish way of propogating up the new reloaded submodule attributes
            reassign_submodule_attributes(verbose=verbose)
        except Exception as ex:
            print(ex)
    rrrr = reload_subs
    # ENDBLOCK
    ''')
    injectstr_fmt = injectstr_fmt.replace('# STARTBLOCK', '')
    injectstr_fmt = injectstr_fmt.replace('# ENDBLOCK', '')
    rrrdir_fmt  = '    getattr(%s, \'reload_subs\', fbrrr)(verbose=verbose)'
    rrrfile_fmt = '    getattr(%s, \'rrr\', fbrrr)(verbose=verbose)'

    def _reload_command(tup):
        if len(tup) > 2 and tup[2] is True:
            return rrrdir_fmt % tup[0]
        else:
            return rrrfile_fmt % tup[0]
    reload_body = '\n'.join(map(_reload_command, IMPORT_TUPLES)).strip()
    format_dict = {
        'modname': modname,
        'reload_body': reload_body,
        'injecter': injecter,
        'injecter_import': injecter_import,
    }
    inject_execstr = injectstr_fmt.format(**format_dict).strip()
    return inject_execstr

#----------
# PUBLIC FUNCTIONS
#----------

def dynamic_import(modname, IMPORT_TUPLES, developing=True, ignore_froms=[],
                   dump=False, ignore_startswith=[], ignore_endswith=[],
                   ignore_list=[],
                   verbose=False):
    """
    MAIN ENTRY POINT

    Dynamically import listed util libraries and their attributes.
    Create reload_subs function.

    Using __import__ like this is typically not considered good style However,
    it is better than import * and this will generate the good file text that
    can be used when the module is "frozen"
    """
    if verbose:
        print('[UTIL_IMPORT] Running Dynamic Imports for modname=%r ' % modname)
    # Get the module that will be imported into
    module = sys.modules[modname]
    # List of modules to be imported
    IMPORTS = __get_imports(IMPORT_TUPLES)
    # Import the modules
    __excecute_imports(module, modname, IMPORTS, verbose=verbose)
    # If developing do explicit import stars
    if developing:
        FROM_IMPORTS = __execute_fromimport_star(module, modname, IMPORT_TUPLES,
                                                 ignore_list=ignore_list,
                                                 ignore_startswith=ignore_startswith,
                                                 ignore_endswith=ignore_endswith,
                                                 verbose=verbose)
    else:
        FROM_IMPORTS = __execute_fromimport(module, modname, IMPORT_TUPLES, verbose=verbose)

    inject_execstr = _inject_execstr(modname, IMPORT_TUPLES)

    # If requested: print what the __init__ module should look like
    dump_requested = (('--dump-%s-init' % modname) in sys.argv or
                      ('--print-%s-init' % modname) in sys.argv) or dump
    overwrite_requested = ('--update-%s-init' % modname) in sys.argv
    if verbose:
        print('[UTIL_IMPORT] Finished Dynamic Imports for modname=%r ' % modname)

    if dump_requested:
        is_main_proc = multiprocessing.current_process().name == 'MainProcess'
        if is_main_proc:
            from utool import util_str
            initstr = _initstr(modname, IMPORTS, FROM_IMPORTS, inject_execstr)
            print(util_str.indent(initstr))
    # Overwrite the __init__.py file with new explicit imports
    if overwrite_requested:
        is_main_proc = multiprocessing.current_process().name == 'MainProcess'
        if is_main_proc:
            from utool import util_str
            from os.path import join, exists
            initstr = _initstr(modname, IMPORTS, FROM_IMPORTS, inject_execstr, withheader=False)
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


def make_initstr(modname, IMPORT_TUPLES, verbose=False):
    """
    Just creates the string representation. Does no importing.
    """
    IMPORTS      = __get_imports(IMPORT_TUPLES)
    FROM_IMPORTS = __get_from_imports(IMPORT_TUPLES)
    inject_execstr = _inject_execstr(modname, IMPORT_TUPLES)
    return _initstr(modname, IMPORTS, FROM_IMPORTS, inject_execstr)


def make_import_tuples(module_path, exclude_modnames=[]):
    """ Infer the IMPORT_TUPLES from a module_path """
    from utool import util_path
    kwargs = dict(private=False, full=False)
    module_list  = util_path.ls_modulefiles(module_path, noext=True, **kwargs)
    package_list = util_path.ls_moduledirs(module_path, **kwargs)
    exclude_set = set(exclude_modnames)
    module_import_tuples = [(modname, None) for modname in module_list
                            if modname not in exclude_set]
    package_import_tuples = [(modname, None, True)  for modname in package_list
                            if modname not in exclude_set]
    IMPORT_TUPLES = (module_import_tuples + package_import_tuples)
    return IMPORT_TUPLES
