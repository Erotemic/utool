# -*- coding: utf-8 -*-
"""
NEEDS CLEANUP SO IT EITHER DOES THE IMPORTS OR GENERATES THE FILE

python -c "import utool"

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import multiprocessing
import textwrap
#import types

#DEBUG_IMPORTS = '--debug-imports' in sys.argv
#----------
# EXECUTORS
#----------

def __excecute_imports(module, modname, imports, verbose=False):
    """ Module Imports """
    # level: -1 is a the Python2 import strategy
    # level:  0 is a the Python3 absolute import
    if verbose:
        print('[UTIL_IMPORT] EXECUTING %d IMPORT TUPLES' % (len(imports),))
    level = 0
    for name in imports:
        if level == -1:
            tmp = __import__(name, globals(), locals(), fromlist=[], level=level)
        elif level == 0:
            # FIXME: should support unicode. Maybe just a python2 thing
            tmp = __import__(modname, globals(), locals(), fromlist=[str(name)], level=level)


def __execute_fromimport(module, modname, import_tuples, verbose=False):
    """ Module From Imports """
    if verbose:
        print('[UTIL_IMPORT] EXECUTING %d FROM IMPORT TUPLES' % (len(import_tuples),))
    from_imports = __get_from_imports(import_tuples)
    for name, fromlist in from_imports:
        full_modname = '.'.join((modname, name))
        tmp = __import__(full_modname, globals(), locals(), fromlist=fromlist, level=0)
        for attrname in fromlist:
            setattr(module, attrname, getattr(tmp, attrname))
    return from_imports


def __execute_fromimport_star(module, modname, import_tuples, ignore_list=[],
                              ignore_startswith=[], ignore_endswith=[],
                              check_not_imported=True, verbose=False,
                              veryverbose=False):
    r"""
    Effectively import * statements

    The dynamic_import must happen before any * imports otherwise it wont catch
    anything.

    Ignore:
        ignore_startswith = []
        ignore_endswith = []
        check_not_imported = False
        verbose = True
        veryverbose = True
    """
    if verbose:
        print('[UTIL_IMPORT] EXECUTE %d FROMIMPORT STAR TUPLES.' % (len(import_tuples),))
    from_imports = []
    # Explicitly ignore these special functions (usually stdlib functions)
    ignoreset = set(['print', 'print_', 'printDBG', 'rrr', 'profile',
                     'print_function', 'absolute_import', 'division', 'zip',
                     'map', 'range', 'list', 'zip_longest', 'filter', 'filterfalse',
                     'dirname', 'realpath', 'join', 'exists', 'normpath',
                     'splitext', 'expanduser', 'relpath', 'isabs',
                     'commonprefix', 'basename', 'input', 'reduce',
                     #'OrderedDict',
                     #'product',
                     ] + ignore_list)
                     #'isdir', 'isfile', '

    #def is_defined_by_module2(item, module):
    #    belongs = False
    #    if hasattr(item, '__module__'):
    #        belongs = item.__module__ == module.__name__
    #    elif hasattr(item, 'func_globals'):
    #        belongs = item.func_globals['__name__'] == module.__name__
    #    return belongs

    for name, fromlist in import_tuples:
        #absname = modname + '.' + name
        child_module = sys.modules[modname + '.' + name]
        # Check if the variable already belongs to the module
        varset = set(vars(module)) if check_not_imported else set()
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
            #is_valid = is_valid and is_defined_by_module2(getattr(child_module, attrname), child_module)
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
                print('[UTIL_IMPORT] %s is importing: %r' % (modname, attrname))
            valid_fromlist_.append(attrname)
            setattr(module, attrname, attrval)
        if verbose:
            print('[UTIL_IMPORT]     name=%r, len(valid_fromlist_)=%d' % (name, len(valid_fromlist_)))
        from_imports.append((name, valid_fromlist_))
    return from_imports

#----------
# PARSERS
#----------

def __get_from_imports(import_tuples):
    """ Returns import names and fromlist
    import_tuples are specified as
    (name, fromlist, ispackage)
    """
    from_imports = [(tup[0], tup[1]) for tup in import_tuples
                    if tup[1] is not None and len(tup[1]) > 0]
    return from_imports

#----------
# STRING MAKERS
#----------

def _initstr(modname, imports, from_imports, inject_execstr, withheader=True):
    """ Calls the other string makers """
    header         = _make_module_header() if withheader else ''
    import_str     = _make_imports_str(imports, modname)
    fromimport_str = _make_fromimport_str(from_imports, modname)
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
        'from __future__ import absolute_import, division, print_function, unicode_literals'])

def _make_imports_str(imports, rootmodname='.'):
    imports_fmtstr = 'from {rootmodname} import %s'.format(rootmodname=rootmodname)
    return '\n'.join([imports_fmtstr % (name,) for name in imports])

def _make_fromimport_str(from_imports, rootmodname='.'):
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
    from_str = '\n'.join(map(_pack_fromimport, from_imports))
    return from_str

def _inject_execstr(modname, import_tuples):
    """ Injection and Reload String Defs """
    if modname == 'utool':
        # Special case import of the util_inject module
        injecter = 'util_inject'
        injecter_import = ''
    else:
        # Normal case implicit import of util_inject
        injecter_import = 'import utool'
        injecter = 'utool'
    injectstr_fmt = textwrap.dedent(
        r'''
        # STARTBLOCK
        {injecter_import}
        print, rrr, profile = {injecter}.inject2(__name__, '[{modname}]')


        def reassign_submodule_attributes(verbose=1):
            """
            Updates attributes in the __init__ modules with updated attributes
            in the submodules.
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


        def reload_subs(verbose=1):
            """ Reloads {modname} and submodules """
            if verbose:
                print('Reloading {modname} submodules')
            rrr(verbose > 1)
            def wrap_fbrrr(mod):
                def fbrrr(*args, **kwargs):
                    """ fallback reload """
                    if verbose > 0:
                        print('Auto-reload (using rrr) not setup for mod=%r' % (mod,))
                return fbrrr
            def get_rrr(mod):
                if hasattr(mod, 'rrr'):
                    return mod.rrr
                else:
                    return wrap_fbrrr(mod)
            def get_reload_subs(mod):
                return getattr(mod, 'reload_subs', wrap_fbrrr(mod))
            {reload_body}
            rrr(verbose > 1)
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
    rrrdir_fmt  = '    get_reload_subs({modname})(verbose=verbose)'
    rrrfile_fmt = '    get_rrr({modname})(verbose > 1)'

    def _reload_command(tup):
        if len(tup) > 2 and tup[2] is True:
            return rrrdir_fmt.format(modname=tup[0])
        else:
            return rrrfile_fmt.format(modname=tup[0])
    reload_body = '\n'.join(map(_reload_command, import_tuples)).strip()
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

def dynamic_import(modname, import_tuples, developing=True, ignore_froms=[],
                   dump=False, ignore_startswith=[], ignore_endswith=[],
                   ignore_list=[], check_not_imported=True, return_initstr=False,
                   verbose=False):
    """
    MAIN ENTRY POINT

    Dynamically import listed util libraries and their attributes.
    Create reload_subs function.

    Using __import__ like this is typically not considered good style However,
    it is better than import * and this will generate the good file text that
    can be used when the module is 'frozen"

    Returns:
        str: init_inject_str - by default all imports are executed in this
            function and only the remainig code needed to be executed is
            returned to define the reload logic.

        str, str: init_inject_str, init_str - if return_initstr is True then
            also returns init_str defining the from imports.

    Ignore:
        ignore_startswith = []
        ignore_endswith = []
        check_not_imported = True
        verbose = True
    """
    if verbose:
        print('[UTIL_IMPORT] Running Dynamic Imports for modname=%r ' % modname)
    # Get the module that will be imported into
    try:
        module = sys.modules[modname]
    except:
        module = __import__(modname)
    # List of modules to be imported
    imports = [tup[0] for tup in import_tuples]
    # Import the modules
    __excecute_imports(module, modname, imports, verbose=verbose)
    # If developing do explicit import stars
    if developing:
        from_imports = __execute_fromimport_star(module, modname, import_tuples,
                                                 ignore_list=ignore_list,
                                                 ignore_startswith=ignore_startswith,
                                                 ignore_endswith=ignore_endswith,
                                                 check_not_imported=check_not_imported,
                                                 verbose=verbose)
    else:
        from_imports = __execute_fromimport(module, modname, import_tuples, verbose=verbose)

    inject_execstr = _inject_execstr(modname, import_tuples)

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
            initstr = _initstr(modname, imports, from_imports, inject_execstr)
            print(util_str.indent(initstr))
    # Overwrite the __init__.py file with new explicit imports
    if overwrite_requested:
        """
        SeeAlso:
            util_inject.inject_python_code
            util_str.replace_between_tags
        """
        is_main_proc = multiprocessing.current_process().name == 'MainProcess'
        if is_main_proc:
            from utool import util_str
            from os.path import join, exists
            initstr = _initstr(modname, imports, from_imports, inject_execstr, withheader=False)
            new_else = util_str.indent(initstr)
            #print(new_else)
            # Get path to init file so we can overwrite it
            init_fpath = join(module.__path__[0], '__init__.py')
            print('attempting to update: %r' % init_fpath)
            assert exists(init_fpath)
            new_lines = []
            editing = False
            updated = False
            #start_tag = '# <AUTOGEN_INIT>'
            #end_tag = '# </AUTOGEN_INIT>'
            with open(init_fpath, 'r') as file_:
                #text = file_.read()
                lines = file_.readlines()
                for line in lines:
                    if not editing:
                        new_lines.append(line)
                    if line.strip().startswith('# <AUTOGEN_INIT>'):
                        new_lines.append('\n' + new_else + '\n    # </AUTOGEN_INIT>\n')
                        editing = True
                        updated = True
                    if line.strip().startswith('# </AUTOGEN_INIT>'):
                        editing = False
            # TODO:
            #new_text = util_str.replace_between_tags(text, new_else, start_tag, end_tag)
            if updated:
                print('writing updated file: %r' % init_fpath)
                new_text = ''.join(new_lines)
                with open(init_fpath, 'w') as file_:
                    file_.write(new_text)
            else:
                print('no write hook for file: %r' % init_fpath)
    if return_initstr:
        initstr = _initstr(modname, imports, from_imports, '', withheader=False)
        return inject_execstr, initstr
    else:
        return inject_execstr


def make_initstr(modname, import_tuples, verbose=False):
    """
    Just creates the string representation. Does no importing.
    """
    imports = [tup[0] for tup in import_tuples]
    from_imports = __get_from_imports(import_tuples)
    inject_execstr = _inject_execstr(modname, import_tuples)
    return _initstr(modname, imports, from_imports, inject_execstr)


def make_import_tuples(module_path, exclude_modnames=[]):
    """ Infer the import_tuples from a module_path """
    from utool import util_path
    kwargs = dict(private=False, full=False)
    module_list  = util_path.ls_modulefiles(module_path, noext=True, **kwargs)
    package_list = util_path.ls_moduledirs(module_path, **kwargs)
    exclude_set = set(exclude_modnames)
    module_import_tuples = [(modname, None) for modname in module_list
                            if modname not in exclude_set]
    package_import_tuples = [(modname, None, True)  for modname in package_list
                            if modname not in exclude_set]
    import_tuples = (module_import_tuples + package_import_tuples)
    return import_tuples
