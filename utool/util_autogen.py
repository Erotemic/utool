# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import re
import os
import six
from collections import deque  # NOQA
from os.path import exists, dirname, join, expanduser, normpath
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)


class PythonStatement(object):
    """ Thin wrapper around a string representing executable python code """
    def __init__(self, stmt):
        self.stmt = stmt
    def __repr__(self):
        return self.stmt
    def __str__(self):
        return self.stmt


def dump_autogen_code(fpath, autogen_text, codetype='python', fullprint=None,
                      show_diff=None, dowrite=None):
    """
    Helper that write a file if -w is given on command line, otherwise
    it just prints it out. It has the opption of comparing a diff to the file.
    """
    import utool as ut
    if dowrite is None:
        dowrite = ut.get_argflag(('-w', '--write'))
    if show_diff is None:
        show_diff = ut.get_argflag('--diff')
    num_context_lines = ut.get_argval('--diff', type_=int, default=None)
    show_diff = show_diff or num_context_lines is not None

    num_context_lines = ut.get_argval('--diff', type_=int, default=None)

    if fullprint is None:
        fullprint = True

    if fullprint is False:
        fullprint = ut.get_argflag('--print')

    print('[autogen] Autogenerated %s...\n+---\n' % (fpath,))
    if not dowrite:
        if fullprint:
            ut.print_code(autogen_text, lexer_name=codetype)
            print('\nL___')
        else:
            print('specify --print to write to stdout')
            pass
        print('specify -w to write, or --diff to compare')
        print('...would write to: %s' % fpath)
    if show_diff:
        if ut.checkpath(fpath, verbose=True):
            prev_text = ut.read_from(fpath)
            textdiff = ut.get_textdiff(prev_text, autogen_text,
                                       num_context_lines=num_context_lines)
            try:
                ut.print_difftext(textdiff)
            except UnicodeDecodeError:
                import unicodedata
                textdiff = unicodedata.normalize('NFKD', textdiff).encode('ascii', 'ignore')
                ut.print_difftext(textdiff)

        if dowrite:
            print('WARNING: Not writing. Remove --diff from command line')
    elif dowrite:
        ut.write_to(fpath, autogen_text)


def makeinit(mod_dpath, exclude_modnames=[], use_star=False):
    r"""
    Args:
        mod_dpath (str):
        exclude_modnames (list): (Defaults to [])
        use_star (bool): (Defaults to False)

    Returns:
        str: init_codeblock

    CommandLine:
        python -m utool.util_autogen makeinit --modname=ibeis.algo

    Example:
        >>> # DISABLE_DOCTEST
        >>> # SCRIPT
        >>> from utool.util_autogen import *  # NOQA
        >>> import utool as ut
        >>> modname = ut.get_argval('--modname', str, default=None)
        >>> mod_dpath = (os.getcwd() if modname is None else
        >>>              ut.get_modpath(modname, prefer_pkg=True))
        >>> mod_dpath = ut.unixpath(mod_dpath)
        >>> mod_fpath = join(mod_dpath, '__init__.py')
        >>> exclude_modnames = ut.get_argval(('--exclude', '-x'), list, default=[])
        >>> use_star = ut.get_argflag('--star')
        >>> init_codeblock = makeinit(mod_dpath, exclude_modnames, use_star)
        >>> ut.dump_autogen_code(mod_fpath, init_codeblock)
    """
    from utool._internal import util_importer
    import utool as ut
    module_name = ut.get_modname_from_modpath(mod_dpath)
    IMPORT_TUPLES = util_importer.make_import_tuples(mod_dpath, exclude_modnames=exclude_modnames)
    initstr = util_importer.make_initstr(module_name, IMPORT_TUPLES)
    regen_command = 'cd %s\n' % (mod_dpath)
    regen_command += '    makeinit.py'
    regen_command += ' --modname={modname}'.format(modname=module_name)
    if use_star:
        regen_command += ' --star'
    if len(exclude_modnames ) > 0:
        regen_command += ' -x ' + ' '.join(exclude_modnames)

    regen_block = (ut.codeblock('''
    """
    Regen Command:
        {regen_command}
    """
    ''').format(regen_command=regen_command))

    importstar_codeblock = ut.codeblock(
        '''
        """
        python -c "import {module_name}" --dump-{module_name}-init
        python -c "import {module_name}" --update-{module_name}-init
        """
        __DYNAMIC__ = True
        if __DYNAMIC__:
            # TODO: import all utool external prereqs. Then the imports will not import
            # anything that has already in a toplevel namespace
            # COMMENTED OUT FOR FROZEN __INIT__
            # Dynamically import listed util libraries and their members.
            from utool._internal import util_importer
            # FIXME: this might actually work with rrrr, but things arent being
            # reimported because they are already in the modules list
            import_execstr = util_importer.dynamic_import(__name__, IMPORT_TUPLES)
            exec(import_execstr)
            DOELSE = False
        else:
            # Do the nonexec import (can force it to happen no matter what if alwyas set
            # to True)
            DOELSE = True

        if DOELSE:
            # <AUTOGEN_INIT>
            pass
            # </AUTOGEN_INIT>
        '''.format(module_name=module_name)
    )

    ts_line = '# Autogenerated on {ts}'.format(ts=ut.get_timestamp('printable'))

    init_codeblock_list = ['# -*- coding: utf-8 -*-', ts_line]
    init_codeblock_list.append(initstr)
    init_codeblock_list.append('\nIMPORT_TUPLES = ' + ut.repr4(IMPORT_TUPLES))
    if use_star:
        init_codeblock_list.append(importstar_codeblock)
    init_codeblock_list.append(regen_block)

    init_codeblock = '\n'.join(init_codeblock_list)
    return init_codeblock


def write_modscript_alias(fpath, modname, args='', pyscript='python'):
    """
    convinience function because $@ is annoying to paste into the terminal
    """
    import utool as ut
    from os.path import splitext
    allargs_dict = {
        '.sh': ' $@',
        '.bat': ' %1', }
    _, script_ext = splitext(fpath)
    if script_ext not in ['.sh', '.bat']:
        script_ext = '.bat' if ut.WIN32 else 'sh'
    allargs = (args + allargs_dict[script_ext]).strip(' ')
    if not modname.endswith('.py'):
        fmtstr = '{pyscript} -m {modname} {allargs}'
    else:
        fmtstr = '{pyscript} {modname} {allargs}'

    cmdstr = fmtstr.format(pyscript=pyscript, modname=modname, allargs=allargs)
    ut.write_to(fpath, cmdstr)
    os.system('chmod +x ' + fpath)


def autofix_codeblock(codeblock, max_line_len=80,
                      aggressive=False,
                      very_aggressive=False,
                      experimental=False):
    r"""
    Uses autopep8 to format a block of code

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> codeblock = ut.codeblock(
            '''
            def func( with , some = 'Problems' ):


             syntax ='Ok'
             but = 'Its very messy'
             if None:
                    # syntax might not be perfect due to being cut off
                    ommiting_this_line_still_works=   True
            ''')
        >>> fixed_codeblock = ut.autofix_codeblock(codeblock)
        >>> print(fixed_codeblock)
    """
    # FIXME idk how to remove the blank line following the function with
    # autopep8. It seems to not be supported by them, but it looks bad.
    import autopep8
    arglist = ['--max-line-length', '80']
    if aggressive:
        arglist.extend(['-a'])
    if very_aggressive:
        arglist.extend(['-a', '-a'])
    if experimental:
        arglist.extend(['--experimental'])
    arglist.extend([''])
    autopep8_options = autopep8.parse_args(arglist)
    fixed_codeblock = autopep8.fix_code(codeblock, options=autopep8_options)
    return fixed_codeblock


def load_func_from_module(modname, funcname, verbose=True, moddir=None, modpath=None):
    r"""
    Args:
        modname (str):  module name
        funcname (str):  function name
        verbose (bool):  verbosity flag(Defaults to True)
        moddir (None): (Defaults to None)

    CommandLine:
        python -m utool.util_autogen load_func_from_module

    Example:
        >>> # DISABLE_DOCTEST
        >>> # UNSTABLE_DOCTEST
        >>> from utool.util_autogen import *  # NOQA
        >>> import utool as ut
        >>> #funcname = 'multi_plot'
        >>> modname = 'utool.util_path'
        >>> funcname = 'checkpath'
        >>> verbose = True
        >>> moddir = None
        >>> func, module, error_str = load_func_from_module(modname, funcname, verbose, moddir)
        >>> source = ut.get_func_sourcecode(func, strip_docstr=True, strip_comments=True)
        >>> keyname = ut.named_field('keyname', ut.REGEX_VARNAME)
        >>> default = ut.named_field('default', '[\'\"A-Za-z_][A-Za-z0-9_\'\"]*')
        >>> pattern = re.escape('kwargs.get(\'') + keyname + re.escape('\',')
        >>> kwarg_keys = [match.groupdict()['keyname'] for match in re.finditer(pattern, source)]
    """
    import utool as ut
    from os.path import join
    import importlib
    func = None
    module = None
    error_str = None
    print('modname = %r' % (modname,))
    print('funcname = %r' % (funcname,))
    print('moddir = %r' % (moddir,))

    try:
        from xdoctest import static_analysis as static
        from xdoctest import utils
        print('modpath = {!r}'.format(modpath))
    except ImportError:
        modpath = None
    else:
        if modpath is None:
            modpath = static.modname_to_modpath(modname)
            if modpath is None:
                modname = modname.split('.')[-1]
                if moddir is not None:
                    modpath = join(moddir, modname + '.py')
                    if not exists(modpath):
                        modpath = None
                if modpath is None:
                    raise Exception('Cannot find modname={} in moddir={}'.format(modname, moddir))
        print('modpath = {!r}'.format(modpath))
        module = utils.import_module_from_path(modpath)
        print('module = {!r}'.format(module))
        try:
            func = eval('module.{}'.format(funcname))
        except AttributeError:
            importlib.reload(module)
            func = eval('module.{}'.format(funcname))
        print('func = {!r}'.format(func))
        return func, module, error_str

    if not isinstance(modname, six.string_types):
        error_str = 'modname=%r is not a string. bad input' % (modname,)
    else:
        if False:
            # TODO: static analysis
            import jedi  # NOQA
            modpath = ut.util_import.get_modpath_from_modname(modname)
            script = jedi.Script(path=modpath)
            mod = script._get_module()
            # monkeypatch
            func.script = script
            func = None
            for name in mod.names_dict[funcname]:
                if name.parent.type == 'funcdef':
                    func = name.parent
                    break
            return func, mod, error_str
            # ut.get_modpath_from_modname(modname)
        if module is None:
            try:
                module = __import__(modname)
            except ImportError:
                if moddir is not None:
                    #parts =
                    # There can be a weird double import error thing happening here
                    # Rectify the dots in the filename
                    module = ut.import_module_from_fpath(join(moddir, modname.split('.')[-1] + '.py'))
                else:
                    raise
            #import inspect
            # try:
            #     imp.reload(module)
            # except Exception as ex:
            #     pass
        # if False:
        #     # Try removing pyc if it exists
        #     if module.__file__.endswith('.pyc'):
        #         ut.delete(module.__file__, verbose=False)
        #         try:
        #             module = __import__(modname)
        #         except ImportError:
        #             if moddir is not None:
        #                 module = ut.import_module_from_fpath(join(moddir, modname.split('.')[-1] + '.py'))
        #             else:
        #                 raise
        try:
            importlib.reload(module)
        except Exception:
            pass
        try:
            # FIXME: PYTHON 3
            execstr = ut.codeblock(
                '''
                try:
                    import {modname}
                    module = {modname}
                    #print('Trying to reload module=%r' % (module,))
                    importlib.reload(module)
                except Exception:
                    # If it fails maybe the module is not in the path
                    if moddir is not None:
                        try:
                            import imp
                            import os
                            orig_dir = os.getcwd()
                            os.chdir(moddir)
                            modname_str = '{modname}'
                            modinfo = imp.find_module(modname_str, [moddir])
                            module = imp.load_module(modname_str, *modinfo)
                            #print('loaded module=%r' % (module,))
                        except Exception as ex:
                            ut.printex(ex, 'failed to imp.load_module')
                            pass
                        finally:
                            os.chdir(orig_dir)
                import importlib
                import utool as ut
                importlib.reload(ut.util_autogen)
                importlib.reload(ut.util_inspect)
                try:
                    func = module.{funcname}
                except AttributeError:
                    docstr = 'Could not find attribute funcname={funcname} in modname={modname} This might be a reloading issue'
                    importlib.reload(module)
                '''
            ).format(**locals())
            exec_locals = locals()
            exec_globals = globals()
            exec(execstr, exec_globals, exec_locals)
            func = exec_locals.get('func', None)
            module = exec_locals.get('module', None)
        except Exception as ex2:
            docstr = 'error ' + str(ex2)
            if verbose:
                import utool as ut
                #ut.printex(ex1, 'ex1')
                ut.printex(ex2, 'ex2', tb=True)
            testcmd = 'python -c "import utool; print(utool.auto_docstr(\'%s\', \'%s\'))"' % (modname, funcname)
            error_str = ut.formatex(ex2, 'ex2', tb=True, keys=['modname', 'funcname', 'testcmd'])
            error_str += '---' + execstr
    return func, module, error_str


def auto_docstr(modname, funcname, verbose=True, moddir=None, modpath=None, **kwargs):
    r"""
    called from vim. Uses strings of filename and modnames to build docstr

    Args:
        modname (str): name of a python module
        funcname (str): name of a function in the module

    Returns:
        str: docstr

    CommandLine:
        python -m utool.util_autogen auto_docstr
        python -m utool --tf auto_docstr

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> from utool.util_autogen import *  # NOQA
        >>> ut.util_autogen.rrr(verbose=False)
        >>> #docstr = ut.auto_docstr('ibeis.algo.hots.smk.smk_index', 'compute_negentropy_names')
        >>> modname = ut.get_argval('--modname', default='utool.util_autogen')
        >>> funcname = ut.get_argval('--funcname', default='auto_docstr')
        >>> moddir = ut.get_argval('--moddir', type_=str, default=None)
        >>> docstr = ut.util_autogen.auto_docstr(modname, funcname)
        >>> print(docstr)
    """
    #import utool as ut
    func, module, error_str = load_func_from_module(
        modname, funcname, verbose=verbose, moddir=moddir, modpath=modpath)
    if error_str is None:
        try:
            docstr = make_default_docstr(func, **kwargs)
        except Exception as ex:
            import utool as ut
            error_str = ut.formatex(ex, 'Caught Error in parsing docstr', tb=True)
            #ut.printex(ex)
            error_str += (
                '\n\nReplicateCommand:\n    '
                'python -m utool --tf auto_docstr '
                '--modname={modname} --funcname={funcname} --moddir={moddir}').format(
                    modname=modname, funcname=funcname, moddir=moddir)
            error_str += '\n kwargs='  + ut.repr4(kwargs)
            return error_str
    else:
        docstr = error_str
    return docstr


def print_auto_docstr(modname, funcname):
    """
    python -c "import utool; utool.print_auto_docstr('ibeis.algo.hots.smk.smk_index', 'compute_negentropy_names')"
    python -c "import utool;
    utool.print_auto_docstr('ibeis.algo.hots.smk.smk_index', 'compute_negentropy_names')"
    """
    docstr = auto_docstr(modname, funcname)
    print(docstr)


# <INVIDIAL DOCSTR COMPONENTS>

def make_args_docstr(argname_list, argtype_list, argdesc_list, ismethod,
                     va_name=None, kw_name=None, kw_keys=[]):
    r"""
    Builds the argument docstring

    Args:
        argname_list (list): names
        argtype_list (list): types
        argdesc_list (list): descriptions
        ismethod (bool): if generating docs for a method
        va_name (Optional[str]): varargs name
        kw_name (Optional[str]): kwargs name
        kw_keys (Optional[list]): accepted kwarg keys

    Returns:
        str: arg_docstr

    CommandLine:
        python -m utool.util_autogen make_args_docstr

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_autogen import *  # NOQA
        >>> argname_list = ['argname_list', 'argtype_list', 'argdesc_list']
        >>> argtype_list = ['list', 'list', 'list']
        >>> argdesc_list = ['names', 'types', 'descriptions']
        >>> va_name = 'args'
        >>> kw_name = 'kwargs'
        >>> kw_keys = ['']
        >>> ismethod = False
        >>> arg_docstr = make_args_docstr(argname_list, argtype_list,
        >>>                               argdesc_list, ismethod, va_name,
        >>>                               kw_name, kw_keys)
        >>> result = str(arg_docstr)
        >>> print(result)
        argname_list (list): names
        argtype_list (list): types
        argdesc_list (list): descriptions
        *args:
        **kwargs:

    """
    import utool as ut
    if ismethod:
        # Remove self from the list
        argname_list = argname_list[1:]
        argtype_list = argtype_list[1:]
        argdesc_list = argdesc_list[1:]

    argdoc_list = [arg + ' (%s): %s' % (_type, desc)
                   for arg, _type, desc in zip(argname_list, argtype_list, argdesc_list)]

    # Add in varargs and kwargs
    # References:
    # http://www.sphinx-doc.org/en/stable/ext/example_google.html#example-google
    if va_name is not None:
        argdoc_list.append('*' + va_name + ':')
    if kw_name is not None:
        import textwrap
        prefix = '**' + kw_name + ': '
        wrapped_lines = textwrap.wrap(', '.join(kw_keys), width=70 - len(prefix))
        sep = '\n' + (' ' * len(prefix))
        kw_keystr = sep.join(wrapped_lines)
        argdoc_list.append((prefix + kw_keystr).strip())

    # align?
    align_args = False
    if align_args:
        argdoc_aligned_list = ut.align_lines(argdoc_list, character='(')
        arg_docstr = '\n'.join(argdoc_aligned_list)
    else:
        arg_docstr = '\n'.join(argdoc_list)
    return arg_docstr


def make_returns_or_yeilds_docstr(return_type, return_name, return_desc):
    return_doctr = return_type + ': '
    if return_name is not None:
        return_doctr += return_name
        if len(return_desc) > 0:
            return_doctr += ' - '
    return_doctr += return_desc
    return return_doctr


def make_example_docstr(funcname=None, modname=None, argname_list=None,
                        defaults=None, return_type=None, return_name=None,
                        ismethod=False):
    """
    Creates skeleton code to build an example doctest

    Args:
        funcname (str):  function name
        modname (str):  module name
        argname_list (str):  list of argument names
        defaults (None):
        return_type (None):
        return_name (str):  return variable name
        ismethod (bool):

    Returns:
        str: examplecode

    CommandLine:
        python -m utool.util_autogen --test-make_example_docstr

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_autogen import *  # NOQA
        >>> funcname = 'make_example_docstr'
        >>> modname = 'utool.util_autogen'
        >>> argname_list = ['qaids', 'qreq_']
        >>> defaults = None
        >>> return_type = tuple
        >>> return_name = 'foo'
        >>> ismethod = False
        >>> examplecode = make_example_docstr(funcname, modname, argname_list, defaults, return_type, return_name, ismethod)
        >>> result = str(examplecode)
        >>> print(result)
        # DISABLE_DOCTEST
        from utool.util_autogen import *  # NOQA
        import utool as ut
        import ibeis
        species = ibeis.const.TEST_SPECIES.ZEB_PLAIN
        qaids = ibs.get_valid_aids(species=species)
        qreq_ = ibeis.testdata_qreq_()
        foo = make_example_docstr(qaids, qreq_)
        result = ('foo = %s' % (ut.repr2(foo),))
        print(result)
    """
    import utool as ut

    examplecode_lines = []
    top_import_fmstr = 'from {modname} import *  # NOQA'
    top_import = top_import_fmstr.format(modname=modname)
    import_lines = [top_import]
    if modname.startswith('utool'):
        import_lines += ['import utool as ut']
    # is_show_func = not modname.startswith('utool') and not modname.startswith('mtgmonte')
    is_show_func = modname.startswith('plottool')

    # TODO: Externally register these
    default_argval_map = {
        'ibs'       : 'ibeis.opendb(defaultdb=\'testdb1\')',
        'testres'   : 'ibeis.testdata_expts(\'PZ_MTEST\')',
        'qreq_'     : 'ibeis.testdata_qreq_()',
        'cm_list'   : 'qreq_.execute()',
        'cm'        : 'qreq_.execute()[0]',
        'aid_list'  : 'ibs.get_valid_aids()',
        'nid_list'  : 'ibs._get_all_known_nids()',
        'qaids'     : 'ibs.get_valid_aids(species=species)',
        'daids'     : 'ibs.get_valid_aids(species=species)',
        'species'   : 'ibeis.const.TEST_SPECIES.ZEB_PLAIN',
        'kpts'      : 'vt.dummy.get_dummy_kpts()',
        'dodraw'    : 'ut.show_was_requested()',
        'img_fpath' : 'ut.grab_test_imgpath(\'carl.jpg\')',
        'gfpath'    : 'ut.grab_test_imgpath(\'carl.jpg\')',
        'img'       : 'vt.imread(img_fpath)',
        'img_in'    : 'vt.imread(img_fpath)',
        'bbox'      : '(10, 10, 50, 50)',
        'theta'     : '0.0',
        'rng'       : 'np.random.RandomState(0)',
    }
    import_depends_map = {
        'ibeis':    'import ibeis',
        'vt':       'import vtool as vt',
        #'img':      'import vtool as vt',  # TODO: remove. fix dependency
        #'species':  'import ibeis',
    }
    var_depends_map = {
        'species':   ['ibeis'],
        'ibs':       ['ibeis'],
        'testres': ['ibeis'],
        'kpts':      ['vt'],
        #'qreq_':     ['ibs', 'species', 'daids', 'qaids'],
        'qreq_':     ['ibeis'],
        'qaids':     ['ibs'],
        'daids':     ['ibs'],
        'qaids':     ['species'],
        'daids':     ['species'],
        'img':       ['img_fpath', 'vt'],
    }

    def find_arg_defaultrepr(argname, val):
        import types
        if val == '?':
            if argname in default_argval_map:
                val = ut.PythonStatement(default_argval_map[argname])
                if argname in import_depends_map:
                    import_lines.append(import_depends_map[argname])
        elif isinstance(val, types.ModuleType):
            return val.__name__
        return repr(val)

    # augment argname list with dependencies
    dependant_argnames = []  # deque()
    def append_dependant_argnames(argnames, dependant_argnames):
        """ use hints to add known dependencies for certain argument inputs """
        for argname in argnames:
            # Check if argname just implies an import
            if argname in import_depends_map:
                import_lines.append(import_depends_map[argname])
            # Check if argname was already added as dependency
            if (argname not in dependant_argnames and argname not in
                 argname_list and argname not in import_depends_map):
                dependant_argnames.append(argname)
            # Check if argname has dependants
            if argname in var_depends_map:
                argdeps = var_depends_map[argname]
                # RECURSIVE CALL
                append_dependant_argnames(argdeps, dependant_argnames)
    append_dependant_argnames(argname_list, dependant_argnames)

    # Define argnames and dependencies in example code
    # argnames prefixed with dependeancies
    argname_list_ = list(dependant_argnames) + argname_list

    # Default example values
    defaults_ = [] if defaults is None else defaults
    num_unknown = (len(argname_list_) - len(defaults_))
    default_vals = ['?'] * num_unknown + list(defaults_)
    arg_val_iter = zip(argname_list_, default_vals)
    inferred_defaults = [find_arg_defaultrepr(argname, val)
                         for argname, val in arg_val_iter]
    argdef_lines = ['%s = %s' % (argname, inferrepr)
                    for argname, inferrepr in
                    zip(argname_list_, inferred_defaults)]
    import_lines = ut.unique_ordered(import_lines)

    if any([inferrepr == repr('?') for inferrepr in inferred_defaults]):
        examplecode_lines.append('# DISABLE_DOCTEST')
    else:
        # Enable the test if it can be run immediately
        examplecode_lines.append('# DISABLE_DOCTEST')

    examplecode_lines.extend(import_lines)
    examplecode_lines.extend(argdef_lines)
    # Default example result assignment
    result_assign = ''
    result_print = None
    if 'return_name' in vars():
        if return_type is not None:
            if return_name is None:
                return_name = 'result'
            result_assign = return_name + ' = '
            result_print = 'print(result)'  # + return_name + ')'
    # Default example call
    if ismethod:
        selfname = argname_list[0]
        methodargs = ', '.join(argname_list[1:])
        tup = (selfname, '.', funcname, '(', methodargs, ')')
        example_call = ''.join(tup)
    else:
        funcargs = ', '.join(argname_list)
        tup = (funcname, '(', funcargs, ')')
        example_call = ''.join(tup)
    # Append call line
    examplecode_lines.append(result_assign + example_call)
    if result_print is not None:
        if return_name != 'result':
            #examplecode_lines.append('result = str(' + return_name + ')')
            result_line_fmt = 'result = (\'{return_name} = %s\' % (ut.repr2({return_name}),))'
            result_line = result_line_fmt.format(return_name=return_name)
            examplecode_lines.append(result_line)
        examplecode_lines.append(result_print)

    # TODO: infer this
    if is_show_func:
        examplecode_lines += [
            '# xdoctest: +REQUIRES(--show)',
        ]

    examplecode = '\n'.join(examplecode_lines)
    return examplecode


def make_cmdline_docstr(funcname, modname):
    #cmdline_fmtstr = 'python -m {modname} --test-{funcname}'  # --enableall'
    #cmdline_fmtstr = 'python -m {modname} --exec-{funcname}'  # --enableall'
    if False and  '.' in modname and '.' not in funcname:
        pkg = modname.split('.')[0]
        # TODO check if __main__ exists with the necessary utool stuffs
        # TODO check if --show should be given
        cmdline_fmtstr = 'python -m {pkg} --tf {funcname}'  # --enableall'
        return cmdline_fmtstr.format(**locals())
    else:
        # TODO: infer this
        is_show_func = 'draw' in funcname or 'show' in funcname
        if is_show_func:
            cmdline_fmtstr = 'python -m {modname} {funcname} --show'
        else:
            cmdline_fmtstr = 'python -m {modname} {funcname}'  # --enableall'
        return cmdline_fmtstr.format(**locals())

# </INVIDIAL DOCSTR COMPONENTS>


def make_docstr_block(header, block):
    import utool as ut
    indented_block = '\n' + ut.indent(block)
    docstr_block = ''.join([header, ':', indented_block])
    return docstr_block


def make_default_docstr(func, with_args=True, with_ret=True,
                        with_commandline=True, with_example=True,
                        with_header=False, with_debug=False):
    r"""
    Tries to make a sensible default docstr so the user
    can fill things in without typing too much

    # TODO: Interleave old documentation with new documentation

    Args:
        func (function): live python function
        with_args (bool):
        with_ret (bool): (Defaults to True)
        with_commandline (bool): (Defaults to True)
        with_example (bool): (Defaults to True)
        with_header (bool): (Defaults to False)
        with_debug (bool): (Defaults to False)

    Returns:
        tuple: (argname, val)

    Ignore:
        pass

    CommandLine:
        python -m utool.util_autogen --exec-make_default_docstr --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_autogen import *  # NOQA
        >>> import utool as ut
        >>> func = ut.make_default_docstr
        >>> #func = ut.make_args_docstr
        >>> #func = PythonStatement
        >>> func = auto_docstr
        >>> default_docstr = make_default_docstr(func)
        >>> result = str(default_docstr)
        >>> print(result)

    """
    import utool as ut
    #from utool import util_inspect
    funcinfo = ut.util_inspect.infer_function_info(func)

    argname_list   = funcinfo.argname_list
    argtype_list   = funcinfo.argtype_list
    argdesc_list   = funcinfo.argdesc_list
    return_header  = funcinfo.return_header
    return_type    = funcinfo.return_type
    return_name    = funcinfo.return_name
    return_desc    = funcinfo.return_desc
    funcname       = funcinfo.funcname
    modname        = funcinfo.modname
    defaults       = funcinfo.defaults
    num_indent     = funcinfo.num_indent
    needs_surround = funcinfo.needs_surround
    funcname       = funcinfo.funcname
    ismethod       = funcinfo.ismethod
    va_name        = funcinfo.va_name
    kw_name        = funcinfo.kw_name
    kw_keys        = funcinfo.kw_keys

    docstr_parts = []
    # Header part
    if with_header:
        header_block = funcname
        docstr_parts.append(header_block)

    # Args part
    if with_args and len(argname_list) > 0:
        argheader = 'Args'
        arg_docstr = make_args_docstr(argname_list, argtype_list, argdesc_list,
                                      ismethod, va_name, kw_name, kw_keys)
        argsblock = make_docstr_block(argheader, arg_docstr)

        docstr_parts.append(argsblock)

    # if False:
    #     with_kw = with_args
    #     if with_kw and len(kwarg_keys) > 0:
    #         #ut.embed()
    #         import textwrap
    #         kwargs_docstr = ', '.join(kwarg_keys)
    #         kwargs_docstr = '\n'.join(textwrap.wrap(kwargs_docstr))
    #         kwargsblock = make_docstr_block('Kwargs', kwargs_docstr)
    #         docstr_parts.append(kwargsblock)

    # Return / Yeild part
    if with_ret and return_header is not None:
        if return_header is not None:
            return_doctr = make_returns_or_yeilds_docstr(return_type, return_name, return_desc)
            returnblock = make_docstr_block(return_header, return_doctr)
            docstr_parts.append(returnblock)

    # Example part
    # try to generate a simple and unit testable example
    if with_commandline:
        cmdlineheader = 'CommandLine'
        cmdlinecode = make_cmdline_docstr(funcname, modname)
        cmdlineblock = make_docstr_block(cmdlineheader, cmdlinecode)
        docstr_parts.append(cmdlineblock)

    if with_example:
        exampleheader = 'Example'
        examplecode = make_example_docstr(funcname, modname, argname_list,
                                          defaults, return_type, return_name,
                                          ismethod)
        examplecode_ = ut.indent(examplecode, '>>> ')
        exampleblock = make_docstr_block(exampleheader, examplecode_)
        docstr_parts.append(exampleblock)

    # DEBUG part (in case something goes wrong)
    if with_debug:
        debugheader = 'Debug'
        debugblock = ut.codeblock(
            '''
            num_indent = {num_indent}
            '''
        ).format(num_indent=num_indent)
        debugblock = make_docstr_block(debugheader, debugblock)
        docstr_parts.append(debugblock)

    # Enclosure / Indentation Parts
    if needs_surround:
        docstr_parts = ['r"""'] + ['\n\n'.join(docstr_parts)] + ['"""']
        default_docstr = '\n'.join(docstr_parts)
    else:
        default_docstr = '\n\n'.join(docstr_parts)

    docstr_indent = ' ' * (num_indent + 4)
    default_docstr = ut.indent(default_docstr, docstr_indent)
    return default_docstr


def remove_codeblock_syntax_sentinals(code_text):
    r"""
    Removes template comments and vim sentinals

    Args:
        code_text (str):

    Returns:
        str: code_text_
    """
    flags = re.MULTILINE | re.DOTALL
    code_text_ = code_text
    code_text_ = re.sub(r'^ *# *REM [^\n]*$\n?', '', code_text_, flags=flags)
    code_text_ = re.sub(r'^ *# STARTBLOCK *$\n', '', code_text_, flags=flags)
    code_text_ = re.sub(r'^ *# ENDBLOCK *$\n?', '', code_text_, flags=flags)
    code_text_ = code_text_.rstrip()
    return code_text_


def make_default_module_maintest(modname, modpath=None, test_code=None):
    """
    make_default_module_maintest

    DEPRICATE

    TODO: use path relative to home dir if the file is a script

    Args:
        modname (str):  module name

    Returns:
        str: text source code

    CommandLine:
        python -m utool.util_autogen --test-make_default_module_maintest

    References:
        http://legacy.python.org/dev/peps/pep-0338/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_autogen import *  # NOQA
        >>> modname = 'utool.util_autogen'
        >>> text = make_default_module_maintest(modname)
        >>> result = str(text)
        >>> print(result)
    """
    import utool as ut
    # Need to use python -m to run a module
    # otherwise their could be odd platform specific errors.
    #python -c "import utool, {modname};
    # ut.doctest_funcs({modname}, allexamples=True)"
    #in_pythonpath, module_type, path = find_modname_in_pythonpath(modname)
    # only use the -m if it is part of a package directory

    if modpath is not None:
        moddir = dirname(modpath)
        pkginit_fpath = join(moddir, '__init__.py')
        use_modrun = exists(pkginit_fpath)
    else:
        use_modrun = True

    if use_modrun:
        pyargs = '-m ' + modname
    else:
        if ut.WIN32:
            modpath = normpath(modpath).replace(expanduser('~'), '%HOME%')
            pyargs = '-B ' + ut.ensure_unixslash(modpath)
        else:
            modpath = normpath(modpath).replace(expanduser('~'), '~')
            pyargs = modpath

    cmdline = ut.codeblock(
        '''
        python {pyargs}
        ''')

    if not use_modrun:
        if ut.WIN32:
            augpath = 'set PYTHONPATH=%PYTHONPATH%' + os.pathsep + moddir
        else:
            augpath = 'export PYTHONPATH=$PYTHONPATH' + os.pathsep + moddir
        cmdline = augpath + '\n' + cmdline

    cmdline = ut.indent(cmdline, ' ' * 8).lstrip(' ').format(pyargs=pyargs)

    if test_code is None:
        test_code = ut.codeblock(
            r'''
            import multiprocessing
            multiprocessing.freeze_support()  # for win32
            import utool as ut  # NOQA
            ut.doctest_funcs()
            ''')

    test_code = ut.indent(test_code, ' ' * 4).lstrip(' ')

    text = ut.codeblock(
        r'''
        # STARTBLOCK
        if __name__ == '__main__':
            r"""
            CommandLine:
                {cmdline}
            """
            {test_code}
        # ENDBLOCK
        '''
    ).format(cmdline=cmdline, test_code=test_code)
    print('test_code = {!r}'.format(test_code))
    text = remove_codeblock_syntax_sentinals(text)
    return text


def is_modname_in_pythonpath(modname):
    in_pythonpath, module_type, path = find_modname_in_pythonpath(modname)
    print(module_type)
    return in_pythonpath


def find_modname_in_pythonpath(modname):
    import sys
    from os.path import exists, join
    rel_modpath = modname.replace('.', '/')
    in_pythonpath = False
    module_type = None
    path_list = sys.path
    path_list = os.environ['PATH'].split(os.pathsep)
    for path in path_list:
        full_modpath = join(path, rel_modpath)
        if exists(full_modpath + '.py'):
            in_pythonpath = True
            module_type = 'module'
            break
        if exists(join(full_modpath, '__init__.py')):
            in_pythonpath = True
            module_type = 'package'
            break
    return in_pythonpath, module_type, path


if __name__ == '__main__':
    """
    CommandLine:
        python ibeis/control/template_generator.py --tbls annotations --Tflags getters native
        python -c "import utool, utool.util_autogen; utool.doctest_funcs(utool.util_autogen, allexamples=True)"
        python -m utool.util_autogen
        python -m utool.util_autogen --allexamples
        python -m utool.util_autogen --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
