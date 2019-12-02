# -*- coding: utf-8 -*-
# Utools for setup.py files
from __future__ import absolute_import, division, print_function
import sys
import textwrap
from os.path import exists, join, dirname, split, splitext
import os
from utool import util_cplat
from utool import util_path
from utool import util_io
from utool import util_str
#from utool import util_dev
from utool.util_dbg import printex
from utool._internal import meta_util_arg

VERBOSE = meta_util_arg.VERBOSE


class SetupManager(object):
    """
    Helps with writing setup.py
    """
    def __init__(self):
        self.cmdclass = {}

    def _register_command(self, name, func):
        import setuptools
        class _WrapCommand(setuptools.Command):
            """
            https://dankeder.com/posts/adding-custom-commands-to-setup-py/
            """
            description = name
            user_options = []
            def initialize_options(self):
                pass
            def finalize_options(self):
                pass
            def run(cmd):
                func()
        self.cmdclass[name] = _WrapCommand
        #self.cmdclass[name] = func
        #TmpCommand

    def register_command(self, name):
        import utool as ut
        if ut.is_funclike(name):
            func = name
            name = ut.get_funcname(func)
            self._register_command(name, func)
            return func
        else:
            def _wrap(func):
                self._register_command(name, func)
            return _wrap

    def get_cmdclass(self):
        cmdclass = self.cmdclass.copy()
        cmdclass.update(**get_cmdclass())
        return cmdclass


class SETUP_PATTERNS():
    clutter_pybuild = [
        '*.pyc',
        '*.pyo',
    ]
    clutter_cyth = [
        '_*_cyth.o',
        '_*_cyth_bench.py',
        'run_cyth_benchmarks.sh',
        '_*_cyth.c',
        '_*_cyth.pyd',
        '_*_cyth.pxd',
        '_*_cyth.html',
        '_*_cyth.pyx',
        '_*_cyth.so',
        '_*_cyth.dylib'
    ]
    chmod_test   = ['test_*.py']


def read_license(license_file):
    try:
        with open(license_file, 'r') as file_:
            firstline = file_.readline()
            license_type = firstline.replace('License', '').strip()
            return license_type
    except IOError:
        print('Warning no LICENCE file')
        return '???'


def build_pyo(project_dirs):
    pyexe = util_cplat.python_executable()
    for dir_ in project_dirs:
        #command_args = [pyexe, '-O', '-m', 'compileall', dir_ + '\*.py']
        command_args = [pyexe, '-O', '-m', 'compileall', dir_]
        #command = ' '.join(command_args)
        #os.system(command)
        util_cplat.shell(command_args)


def setup_chmod(setup_fpath, setup_dir, chmod_patterns):
    """ Gives files matching pattern the same chmod flags as setup.py """
    #st_mode = os.stat(setup_fpath).st_mode
    st_mode = 33277
    for pattern in chmod_patterns:
        for fpath in util_path.glob(setup_dir, pattern, recursive=True):
            print('[setup] chmod fpath=%r' % fpath)
            os.chmod(fpath, st_mode)


def assert_in_setup_repo(setup_fpath, name=''):
    """ pass in __file__ from setup.py """
    setup_dir, setup_fname = split(setup_fpath)
    cwd = os.getcwd()
    #repo_dname = split(setup_dir)[1]
    #print('cwd       = %r' % (cwd))
    #print('repo_dname = %r' % repo_dname)
    #print('setup_dir = %r' % (setup_dir))
    #print('setup_fname = %r' % (setup_fname))
    try:
        assert setup_fname == 'setup.py', 'name is not setup.py'
        #assert name == '' or repo_dname == name,
        ('name=%r' % name)
        assert cwd == setup_dir, 'cwd is not setup_dir'
        assert exists(setup_dir), 'setup dir does not exist'
        assert exists(join(setup_dir, 'setup.py')), 'setup.py does not exist'
    except AssertionError as ex:
        printex(ex, 'ERROR!: setup.py must be run from repository root')
        raise


def clean(setup_dir, clutter_patterns, clutter_dirs):
    print('[setup] clean()')

    clutter_patterns_ = [pat for pat in clutter_patterns if not pat.endswith('/')]
    _clutter_dirs = [pat[:-1] for pat in clutter_patterns if pat.endswith('/')]
    clutter_dirs_ = _clutter_dirs + clutter_dirs

    util_path.remove_files_in_dir(setup_dir,
                                  clutter_patterns_,
                                  recursive=True,
                                  verbose=VERBOSE)

    for dir_ in clutter_dirs_:
        util_path.delete(dir_, verbose=VERBOSE, print_exists=False)
        #util_path.remove_files_in_dir(dir_)

    #for fpath in cython_files:
    #    fname, ext = splitext(fpath)
    #    for libext in util_cplat.LIB_EXT_LIST:
    #        util_path.remove_file(fname + libext)
    #    util_path.remove_file(fname + '.c')


#def build_cython(cython_files):
#    """ doesn't work """
#    for fpath in cython_files:
#        util_dev.compile_cython(fpath)


# def translate_cyth():
#     import cyth
#     cyth.translate_all()


def get_numpy_include_dir():
    try:
        import numpy as np
        return np.get_include()
    except ImportError:
        return ''


def find_ext_modules(disable_warnings=True):
    from setuptools import Extension
    import utool
    from os.path import relpath
    cwd = os.getcwd()

    # CYTH      = 'cyth' in sys.argv
    BEXT      = 'bext' in sys.argv
    BUILD     = 'build' in sys.argv
    BUILD_EXT = 'build_ext' in sys.argv

    # if any([BEXT, CYTH]):
    #     translate_cyth()  # translate cyth before finding ext modules

    if not any([BEXT, BUILD, BUILD_EXT]):
        # dont find modules if they are not being built
        return []

    #pyx_list = utool.glob(cwd, '*_cython.pyx', recursive=True)
    pyx_list = utool.glob(cwd, '*.pyx', recursive=True)

    if disable_warnings:
        extra_compile_args = ['-Wno-format', '-Wno-unused-function']
    else:
        extra_compile_args = []

    ext_modules = []
    for pyx_abspath in pyx_list:
        pyx_relpath = relpath(pyx_abspath, cwd)
        pyx_modname, _ = splitext(pyx_relpath.replace('\\', '.').replace('/', '.'))
        print('[find_ext] Found Module:')
        print('   * pyx_modname = %r' % (pyx_modname,))
        print('   * pyx_relpath = %r' % (pyx_relpath,))
        extmod = Extension(pyx_modname, [pyx_relpath],
                           include_dirs=[get_numpy_include_dir()],
                           extra_compile_args=extra_compile_args)
        ext_modules.append(extmod)
    return ext_modules


def find_packages(recursive=True, maxdepth=None):
    """
    Finds all directories with an __init__.py file in them
    """
    import utool
    if utool.VERBOSE:
        print('[util_setup] find_packages(recursive=%r, maxdepth=%r)' % (recursive, maxdepth))
    from os.path import relpath
    cwd = os.getcwd()
    init_files = utool.glob(cwd, '__init__.py', recursive=recursive, maxdepth=maxdepth)
    package_paths = list(map(dirname, init_files))
    package_relpaths = [relpath(path, cwd) for path in package_paths]

    packages = []
    for path in package_relpaths:
        base = utool.dirsplit(path)[0]
        if exists(join(base, '__init__.py')):
            package = path.replace('/', '.').replace('\\', '.')
            packages.append(package)
    return packages


def get_cmdclass():
    """ DEPRICATE """
    try:
        from Cython.Distutils import build_ext
        cmdclass = {'build_ext': build_ext}
        return cmdclass
    except Exception as ex:
        print(ex)
        print('WARNING: Cython is not installed. This is only a problem if you are building C extensions')
        return {}


def parse_author():
    """ TODO: this function should parse setup.py or a module for
    the author variable
    """
    return 'Jon Crall'   # FIXME


def autogen_sphinx_apidoc():
    r"""
    autogen_sphinx_docs.py

    Ignore:
        C:\Python27\Scripts\autogen_sphinx_docs.py
        autogen_sphinx_docs.py

        pip uninstall sphinx
        pip install sphinx
        pip install sphinxcontrib-napoleon
        pip install sphinx --upgrade
        pip install sphinxcontrib-napoleon --upgrade

        cd C:\Python27\Scripts
        ls C:\Python27\Scripts

        python -c "import sphinx; print(sphinx.__version__)"

    CommandLine:
        python -m utool.util_setup --exec-autogen_sphinx_apidoc

    Example:
        >>> # DISABLE_DOCTEST
        >>> # SCRIPT
        >>> from utool.util_setup import *  # NOQA
        >>> autogen_sphinx_apidoc()
    """
    # TODO: assert sphinx-apidoc exe is found
    # TODO: make find_exe work?
    import utool as ut

    def build_sphinx_apidoc_cmdstr():
        print('')
        print('if this fails try: sudo pip install sphinx')
        print('')
        apidoc = 'sphinx-apidoc'
        if ut.WIN32:
            winprefix = 'C:/Python27/Scripts/'
            sphinx_apidoc_exe = winprefix + apidoc + '.exe'
        else:
            sphinx_apidoc_exe = apidoc
        apidoc_argfmt_list = [
            sphinx_apidoc_exe,
            '--force',
            '--full',
            '--maxdepth="{maxdepth}"',
            '--doc-author="{author}"',
            '--doc-version="{doc_version}"',
            '--doc-release="{doc_release}"',
            '--output-dir="_doc"',
            #'--separate',  # Put documentation for each module on its own page
            '--private',  # Include "_private" modules
            '{pkgdir}',
        ]
        outputdir = '_doc'
        author = ut.parse_author()
        packages = ut.find_packages(maxdepth=1)
        assert len(packages) != 0, 'directory must contain at least one package'
        if len(packages) > 1:
            assert len(packages) == 1,\
                ('FIXME I dont know what to do with more than one root package: %r'
                 % (packages,))
        pkgdir = packages[0]
        version = ut.parse_package_for_version(pkgdir)
        modpath = dirname(ut.truepath(pkgdir))

        apidoc_fmtdict = {
            'author': author,
            'maxdepth': '8',
            'pkgdir': pkgdir,
            'doc_version': version,
            'doc_release': version,
            'outputdir': outputdir,
        }
        ut.assert_exists('setup.py')
        ut.ensuredir('_doc')
        apidoc_fmtstr = ' '.join(apidoc_argfmt_list)
        apidoc_cmdstr = apidoc_fmtstr.format(**apidoc_fmtdict)
        print('[util_setup] autogenerate sphinx docs for %r' % (pkgdir,))
        if ut.VERBOSE:
            print(ut.repr4(apidoc_fmtdict))
        return apidoc_cmdstr, modpath, outputdir

    def build_conf_replstr():
        #
        # Make custom edits to conf.py
        # FIXME:
        #ext_search_text = ut.unindent(
        #    r'''
        #    extensions = [
        #    [^\]]*
        #    ]
        #    ''')
        ext_search_text = r'extensions = \[[^/]*\]'
        # TODO: http://sphinx-doc.org/ext/math.html#module-sphinx.ext.pngmath
        #'sphinx.ext.mathjax',
        exclude_modules = []
        ext_repl_text = ut.codeblock(
            '''
            MOCK_MODULES = {exclude_modules}
            if len(MOCK_MODULES) > 0:
                import mock
                for mod_name in MOCK_MODULES:
                    sys.modules[mod_name] = mock.Mock()

            extensions = [
                'sphinx.ext.autodoc',
                'sphinx.ext.viewcode',
                # For LaTeX
                'sphinx.ext.pngmath',
                # For Google Sytle Docstrs
                # https://pypi.python.org/pypi/sphinxcontrib-napoleon
                'sphinxcontrib.napoleon',
                #'sphinx.ext.napoleon',
            ]
            '''
        ).format(exclude_modules=str(exclude_modules))
        #theme_search = 'html_theme = \'default\''
        theme_search = 'html_theme = \'[a-zA-Z_1-3]*\''
        theme_repl = ut.codeblock(
            '''
            import sphinx_rtd_theme
            html_theme = "sphinx_rtd_theme"
            html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
            ''')
        head_text = ut.codeblock(
            '''
            from sphinx.ext.autodoc import between
            import sphinx_rtd_theme
            import sys
            import os

            # Dont parse IBEIS args
            os.environ['IBIES_PARSE_ARGS'] = 'OFF'
            os.environ['UTOOL_AUTOGEN_SPHINX_RUNNING'] = 'ON'

            sys.path.append('{modpath}')
            sys.path.append(sys.path.insert(0, os.path.abspath("../")))

            autosummary_generate = True

            modindex_common_prefix = ['_']
            '''
        ).format(modpath=ut.truepath(modpath))
        tail_text = ut.codeblock(
            '''
            def setup(app):
                # Register a sphinx.ext.autodoc.between listener to ignore everything
                # between lines that contain the word IGNORE
                app.connect('autodoc-process-docstring', between('^.*IGNORE.*$', exclude=True))
                return app
            '''
        )
        return (ext_search_text, ext_repl_text, theme_search, theme_repl, head_text, tail_text)

    apidoc_cmdstr, modpath, outputdir = build_sphinx_apidoc_cmdstr()
    ext_search_text, ext_repl_text, theme_search, theme_repl, head_text, tail_text = build_conf_replstr()

    dry = ut.get_argflag('--dry')

    if not dry:
        # Execute sphinx-apidoc
        ut.cmd(apidoc_cmdstr, shell=True)
        # sphinx-apidoc outputs conf.py to <outputdir>, add custom commands
        #
        # Change dir to <outputdir>
        print('chdir' + outputdir)
        os.chdir(outputdir)
        conf_fname = 'conf.py'
        conf_text = ut.read_from(conf_fname)
        conf_text = conf_text.replace('import sys', 'import sys  # NOQA')
        conf_text = conf_text.replace('import os', 'import os  # NOQA')
        conf_text = ut.regex_replace(theme_search, theme_repl, conf_text)
        conf_text = ut.regex_replace(ext_search_text, ext_repl_text, conf_text)
        conf_text = head_text + '\n' + conf_text + tail_text
        ut.write_to(conf_fname, conf_text)
        # Make the documentation
        #if ut.LINUX:
        #    ut.cmd('make html', shell=True)
        #if ut.WIN32:
        #raw_input('waiting')
        if not ut.get_argflag('--nomake'):
            ut.cmd('make', 'html', shell=True)
    else:
        print(apidoc_cmdstr)
        print('cd ' + outputdir)
        print('manual edits of conf.py')
        print('make html')


def NOOP():
    pass


def presetup_commands(setup_fpath, kwargs):
    if VERBOSE:
        print('[setup] presetup_commands()')
    name = kwargs.get('name', '')
    # Parse args
    project_dirs     = kwargs.pop('project_dirs', None)
    chmod_patterns   = kwargs.pop('chmod_patterns', [])
    clutter_dirs     = kwargs.pop('clutter_dirs', None)
    clutter_patterns = kwargs.pop('clutter_patterns', [])
    build_command    = kwargs.pop('build_command', NOOP)
    # Augment patterns with builtin patterns
    chmod_patterns   += SETUP_PATTERNS.chmod_test if kwargs.pop('chmod_tests', True) else []
    clutter_patterns += SETUP_PATTERNS.clutter_pybuild if kwargs.pop('clean_pybuild', True) else []
    clutter_patterns += SETUP_PATTERNS.clutter_cyth if kwargs.pop('clean_pybuild', True) else []
    setup_fpath = util_path.truepath(setup_fpath)
    #
    setup_dir = dirname(setup_fpath)
    build_dir = join(setup_dir, 'build')
    os.chdir(setup_dir)  # change into setup directory
    assert_in_setup_repo(setup_fpath, name)

    # Augment clutter dirs
    if clutter_dirs is None:
        clutter_dirs = []
    clutter_dirs += [
        'build',
        'dist',
        name + '.egg-info',
        '__pycache__'
    ]

    if project_dirs is None:
        project_dirs = util_path.ls_moduledirs(setup_dir)
# Execute pre-setup commands based on argv
    #BEXT = 'bext' in sys.argv
    #BUILD_EXT = 'build_ext' in sys.argv
    #CYTH = 'cyth' in sys.argv

    #if CYTH:
    #    translate_cyth()

    for arg in iter(sys.argv[:]):
        #print(arg)
        # Clean clutter files
        if arg in ['clean']:
            clean(setup_dir, clutter_patterns, clutter_dirs)
            #sys.exit(0)
        if arg in ['build'] or (not exists(build_dir) and
                                'clean' not in sys.argv):
            if VERBOSE:
                print('[setup] Executing build command')
            try:
                build_command()
            except Exception as ex:
                printex(ex, 'Error calling buildcommand from cwd=%r\n' % os.getcwd())
                raise
        if arg in ['versions']:
            print('+ ---')
            print('Testing preqres package versions')
            install_requires = kwargs.get('install_requires', [])
            import pip
            print('Checking install_requires = [\n%s\n]' % '\n'.join(install_requires))
            pip.main(['show'] + [depline.split(' ')[0] for depline in install_requires])
            print('L ___ Done Version Testing')
        if arg in ['docs']:
            # FIXME
            autogen_sphinx_apidoc()
        # Build optimized files
        if arg in ['o', 'pyo']:
            build_pyo(project_dirs)
        # Cythonize files
        #if arg in ['cython']:
        #    build_cython(cython_files)
        # Chmod files
        if arg in ['chmod']:
            setup_chmod(setup_fpath, setup_dir, chmod_patterns)

    try:
        sys.argv.remove('docs')
        sys.argv.remove('cyth')
        #sys.argv.remove('--cyth-write')
    except ValueError:
        pass

    try:
        # SUPER HACK
        # aliases bext to build_ext --inplace
        sys.argv.remove('bext')
        sys.argv.append('build_ext')
        sys.argv.append('--inplace')
    except ValueError:
        pass


presetup = presetup_commands


# def parse_package_version(fpath='__init__.py', varname='__version__'):
#     """ Statically parse the version number from __init__.py """
#     # from os.path import dirname, join
#     import ast
#     # repo_dpath = dirname(__file__)
#     # file_fpath = join(repo_dpath, 'ibeis', '__init__.py')
#     # file_fpath = join(repo_dpath, 'ibeis', 'control', 'DB_SCHEMA_CURRENT.py')
#     with open(fpath) as file_:
#         sourcecode = file_.read()
#     pt = ast.parse(sourcecode)

#     class VersionVisitor(ast.NodeVisitor):
#         def visit_Assign(self, node):
#             for target in node.targets:
#                 if target.id == varname:
#                     self.version = node.value.s
#     visitor = VersionVisitor()
#     visitor.visit(pt)
#     return visitor.version


def parse_package_for_version(name):
    """
    Searches for a variable named __version__ in name's __init__.py file and
    returns the value.  This function parses the source text. It does not load
    the module.
    """
    from utool import util_regex
    init_fpath = join(name, '__init__.py')
    version_errmsg = textwrap.dedent(
        '''
        You must include a __version__ variable
        in %s\'s __init__.py file.
        Try something like:
        __version__ = '1.0.0.dev1' ''' % (name,))
    if not exists(init_fpath):
        raise AssertionError(version_errmsg)
    val_regex = util_regex.named_field('version', '[0-9a-zA-Z.]+')
    regexstr = '__version__ *= *[\'"]' + val_regex
    def parse_version(line):
        # Helper
        line = line.replace(' ', '').replace('\t', '')
        match_dict = util_regex.regex_parse(regexstr, line)
        if match_dict is not None:
            return match_dict['version']
    # Find the version  in the text of the source
    #version = 'UNKNOWN_VERSION'
    with open(init_fpath, 'r') as file_:
        for line in file_.readlines():
            if line.startswith('__version__'):
                version = parse_version(line)
                if version is not None:
                    return version
    raise AssertionError(version_errmsg)


def __infer_setup_kwargs(module, kwargs):
    """ Implicitly build kwargs based on standard info """
    # Get project name from the module
    #if 'name' not in kwargs:
    #    kwargs['name'] = module.__name__
    #else:
    #    raise AssertionError('must specify module name!')
    name = kwargs['name']
    # Our projects depend on utool
    #if kwargs['name'] != 'utool':
    #    install_requires = kwargs.get('install_requires', [])
    #    if 'utool' not in install_requires:
    #        install_requires.append('utool')
    #    kwargs['install_requires'] = install_requires

    packages = kwargs.get('packages', [])
    if name not in packages:
        packages.append(name)
        kwargs['packages'] = packages

    if 'version' not in kwargs:
        version = parse_package_for_version(name)
        kwargs['version'] = version

    # Parse version
    #if 'version' not in kwargs:
    #    if module is None:
    #        version_errmsg = 'You must include a version (preferably one that matches the __version__ variable in your modules init file'
    #        raise AssertionError(version_errmsg)
    #    else:
    # Parse license
    if 'license' not in kwargs:
        try:
            kwargs['license'] = read_license('LICENSE')
        except IOError:
            pass
    # Parse readme
    if 'long_description' not in kwargs:
        kwargs['long_description'] = parse_readme()


def parse_readme(readmefile='README.md'):
    # DEPRICATE
    return util_io.read_from(readmefile, verbose=False, strict=False)


def setuptools_setup(setup_fpath=None, module=None, **kwargs):
    # TODO: Learn this better
    # https://docs.python.org/3.1/distutils/apiref.html
    # https://pythonhosted.org/an_example_pypi_project/setuptools.html
    # https://docs.python.org/2/distutils/setupscript.html https://docs.python.org/2/distutils/setupscript.html
    # Useful documentation: http://bashelton.com/2009/04/setuptools-tutorial/#setup.py-package_dir
    """
    Arguments which can be passed to setuptools::

        ============       =====            ===========
        Install-Data       Value            Description
        ------------       -----            -----------
        *packages          strlist          a list of packages modules to be distributed
        py_modules         strlist          a list of singlefile modules to be distributed
        scripts            strlist          a list of standalone scripts to build and install
        *install_requires  list             e.g: ['distribute == 0.7.3', 'numpy', 'matplotlib']
        data_files         strlist          a list of data files to install
        zip_safe           bool             install efficiently installed as a zipped module?
        namespace_packages list             packages without meaningful __init__.py's
        package_dir        dict             keys are packagenames ('' is the root)
        package_data       dict             keys are foldernames, values are a list of globstrs
        *entry_pionts      dict             installs a script {'console_scripts': ['script_name_to_install = entry_module:entry_function']}

        ============       =====            ===========
        Meta-Data          Value            Description
        ------------       -----            -----------
        name               short string     ('name of the package')
        version            short string     ('version of this release')
        author             short string     ('package authors name')
        author_email       email address    ('email address of the package author')
        maintainer         short string     ('package maintainers name')
        maintainer_email   email address    ('email address of the package maintainer')
        url                URL              ('home page for the package')
        description        short string     ('short, summary description of the package')
        long_description   long string      ('longer description of the package')
        download_url       URL              ('location where the package may be downloaded')
        classifiers        list of strings  ('a list of classifiers')
        platforms          list of strings  ('a list of platforms')
        license            short string     ('license for the package')
    """
    from utool.util_inject import inject_colored_exceptions
    inject_colored_exceptions()  # Fluffly, but nice
    if VERBOSE:
        print(util_str.repr4(kwargs))
    __infer_setup_kwargs(module, kwargs)
    presetup_commands(setup_fpath, kwargs)
    if VERBOSE:
        print(util_str.repr4(kwargs))
    return kwargs


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_setup
        python -m utool.util_setup --allexamples
        python -m utool.util_setup --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
