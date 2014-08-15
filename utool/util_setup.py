# Utools for setup.py files
from __future__ import absolute_import, division, print_function
import sys
import textwrap
from os.path import exists, join, dirname, split, splitext
import os
from . import util_cplat
from . import util_path
from . import util_dev
from . import util_io
from . import util_str
from .util_dbg import printex

VERBOSE = '--verbose' in sys.argv


class SETUP_PATTERNS():
    clutter = ['*.pyc', '*.pyo', '*_cython.o', '*_cython.c', '*_cython.pyd',
               '*_cython.so', '*_cython.dylib']
    chmod   = ['test_*.py']


def read_license(license_file):
    with open(license_file, 'r') as file_:
        firstline = file_.readline()
        license_type = firstline.replace('License', '').strip()
        return license_type


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


def clean(setup_dir, clutter_patterns, clutter_dirs, cython_files):
    print('[setup] clean()')

    clutter_patterns_ = [pat for pat in clutter_patterns if not pat.endswith('/')]
    clutter_dirs_ = [pat[:-1] for pat in clutter_patterns if pat.endswith('/')] + clutter_dirs

    util_path.remove_files_in_dir(setup_dir,
                                  clutter_patterns_,
                                  recursive=True,
                                  verbose=VERBOSE)

    for dir_ in clutter_dirs_:
        util_path.delete(dir_, verbose=VERBOSE, print_exists=False)
        #util_path.remove_files_in_dir(dir_)

    for fpath in cython_files:
        fname, ext = splitext(fpath)
        for libext in util_cplat.LIB_EXT_LIST:
            util_path.remove_file(fname + libext)
        util_path.remove_file(fname + '.c')


def build_cython(cython_files):
    """ doesn't work """
    for fpath in cython_files:
        util_dev.compile_cython(fpath)


def translate_cyth():
    import cyth
    cyth.translate_all()


def find_ext_modules(disable_warnings=True):
    from setuptools import Extension
    import utool
    from os.path import relpath
    import numpy as np
    cwd = os.getcwd()

    BEXT = 'bext' in sys.argv
    BUILD_EXT = 'build_ext' in sys.argv
    BUILD = 'build' in sys.argv
    CYTH = 'cyth' in sys.argv
    if not any([BEXT, BUILD, BUILD_EXT, CYTH]):
        return []

    translate_cyth()  # translate cyth before finding ext modules

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
                           include_dirs=[np.get_include()],
                           extra_compile_args=extra_compile_args)
        ext_modules.append(extmod)
    return ext_modules


def find_packages():
    import utool
    from os.path import relpath
    cwd = os.getcwd()
    init_files = utool.glob(cwd, '__init__.py', recursive=True)
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
    from Cython.Distutils import build_ext
    cmdclass = {'build_ext': build_ext}
    return cmdclass


def NOOP():
    pass


def presetup(setup_fpath, kwargs):
    if VERBOSE:
        print('[setup] presetup()')
    name = kwargs.get('name', '')
    project_dirs     = kwargs.pop('project_dirs', None)
    chmod_patterns   = kwargs.pop('chmod_patterns', SETUP_PATTERNS.chmod)
    clutter_dirs     = kwargs.pop('clutter_dirs', None)
    clutter_patterns = kwargs.pop('clutter_patterns', SETUP_PATTERNS.clutter)
    cython_files     = kwargs.pop('cython_files', [])  # todo remove
    build_command    = kwargs.pop('build_command', NOOP)
    setup_fpath = util_path.truepath(setup_fpath)
    setup_dir = dirname(setup_fpath)
    build_dir = join(setup_dir, 'build')
    os.chdir(setup_dir)  # change into setup directory
    assert_in_setup_repo(setup_fpath, name)

    if clutter_dirs is None:
        clutter_dirs = ['build', 'dist', name + '.egg-info']

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
            clean(setup_dir, clutter_patterns, clutter_dirs, cython_files)
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


presetup_commands = presetup  # TODO:


def parse_package_for_version(name):
    from .util_regex import named_field, regex_parse
    init_fpath = join(name, '__init__.py')
    val_regex = named_field('version', '[0-9a-zA-Z.]+')
    version_errmsg = textwrap.dedent(
        '''
        You must include a __version__ variable
        in %s\'s __init__.py file.
        Try something like:
        __version__ = '1.0.0.dev1' ''')
    if not exists(init_fpath):
        raise AssertionError(version_errmsg)
    def parse_version(line):
        # Helper
        line = line.replace(' ', '').replace('\t', '')
        match_dict = regex_parse('__version__ *= *[\'"]' + val_regex, line)
        if match_dict is not None:
            return match_dict['version']
    # Find the version  in the text of the source
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
    return util_io.read_from(readmefile, verbose=False, strict=False)


def setuptools_setup(setup_fpath=None, module=None, **kwargs):
    # TODO: Learn this better
    # https://docs.python.org/3.1/distutils/apiref.html
    # https://pythonhosted.org/an_example_pypi_project/setuptools.html
    # https://docs.python.org/2/distutils/setupscript.html https://docs.python.org/2/distutils/setupscript.html
    # Useful documentation: http://bashelton.com/2009/04/setuptools-tutorial/#setup.py-package_dir
    """
    Arguments which can be passed to setuptools

    -------------------------------------------------------------------------------------
    Install-Data       Value            Description
    -------------------------------------------------------------------------------------
   *packages           strlist          a list of packages modules to be distributed
    py_modules         strlist          a list of singlefile modules to be distributed
    scripts            strlist          a list of standalone scripts to build and install
   *install_requires   list             e.g: ['distribute == 0.7.3', 'numpy', 'matplotlib']
    data_files         strlist          a list of data files to install
    zip_safe           bool             install efficiently installed as a zipped module?
    namespace_packages list             packages without meaningful __init__.py's
    package_dir        dict             keys are packagenames ('' is the root)
    package_data       dict             keys are foldernames, values are a list of globstrs
   *entry_pionts       dict             installs a script {'console_scripts': ['script_name_to_install = entry_module:entry_function']}


    -------------------------------------------------------------------------------------
    Meta-Data          Value            Description
    -------------------------------------------------------------------------------------
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
    from .util_inject import inject_colored_exceptions
    inject_colored_exceptions()  # Fluffly, but nice
    if VERBOSE:
        print(util_str.dict_str(kwargs))
    __infer_setup_kwargs(module, kwargs)
    presetup(setup_fpath, kwargs)
    if VERBOSE:
        print(util_str.dict_str(kwargs))
    return kwargs
