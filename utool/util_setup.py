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
from .util_arg import VERBOSE


class SETUP_PATTERNS():
    clutter = ['*.pyc', '*.pyo']
    chmod   = ['test_*.py']


def read_license(license_file):
    with open(license_file, 'r') as file_:
        firstline = file_.readline()
        license_type = firstline.replace('License', '').strip()
        return license_type


def build_pyo(project_dirs):
    for dir_ in project_dirs:
        util_cplat.shell('python -O -m compileall ' + dir_ + '/*.py')


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
        assert setup_fname == 'setup.py'
        #assert name == '' or repo_dname == name,
        ('name=%r' % name)
        assert cwd == setup_dir
        assert exists(setup_dir)
        assert exists(join(setup_dir, 'setup.py'))
    except AssertionError as ex:
        printex(ex, 'ERROR!: setup.py must be run from repository root')
        raise


def clean(setup_dir, clutter_patterns, clutter_dirs, cython_files):
    print('[setup] clean()')
    util_path.remove_files_in_dir(setup_dir,
                                  clutter_patterns,
                                  recursive=True,
                                  verbose=VERBOSE)

    for dir_ in clutter_dirs:
        util_path.delete(dir_)
        #util_path.remove_files_in_dir(dir_)

    for fpath in cython_files:
        fname, ext = splitext(fpath)
        for libext in util_cplat.LIB_EXT_LIST:
            util_path.remove_file(fname + libext)
        util_path.remove_file(fname + '.c')


def build_cython(cython_files):
    for fpath in cython_files:
        util_dev.compile_cython(fpath)


def NOOP():
    pass


def presetup(setup_fpath, kwargs):
    if VERBOSE:
        print('[setup] presetup()')
    name     = kwargs.pop('name', '')
    project_dirs     = kwargs.pop('project_dirs', None)
    chmod_patterns   = kwargs.pop('chmod_patterns', SETUP_PATTERNS.chmod)
    clutter_dirs     = kwargs.pop('clutter_dirs', None)
    clutter_patterns = kwargs.pop('clutter_patterns', SETUP_PATTERNS.clutter)
    cython_files     = kwargs.pop('cython_files', [])
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
    for arg in iter(sys.argv[1:]):
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
        if arg in ['cython']:
            build_cython(cython_files)
        # Chmod files
        if arg in ['chmod']:
            setup_chmod(setup_fpath, setup_dir, chmod_patterns)


def __infer_setup_kwargs(module, kwargs):
    """ Implicitly build kwargs based on standard info """
    # Get project name from the module
    if 'name'  not in kwargs:
        kwargs['name'] = module.__name__
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

    # Parse version
    if 'version' not in kwargs:
        if module is None:
            version_errmsg = 'You must include a version (preferably one that matches the __version__ variable in your modules init file'
            raise AssertionError(version_errmsg)
        else:
            version_errmsg = textwrap.dedent(
                '''
                You must include a __version__ variable
                in %s\'s __init__.py file.
                Try something like:
                __version__ = '1.0.0.dev1' ''')
            assert hasattr(module, '__version__'), (version_errmsg % str(module))
        kwargs['version'] = module.__version__
    # Parse license
    if 'license' not in kwargs:
        try:
            kwargs['license'] = read_license('LICENSE')
        except IOError:
            pass
    # Parse readme
    if 'long_description' not in kwargs:
        kwargs['long_description'] = util_io.read_from('README.md',
                                                       verbose=False,
                                                       strict=False)
    if 'name' not in kwargs:
        kwargs['name'] = name


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
    from setuptools import setup
    from .util_inject import inject_colored_exceptions
    inject_colored_exceptions()  # Fluffly, but nice
    __infer_setup_kwargs(module, kwargs)
    presetup(setup_fpath, kwargs)
    if VERBOSE:
        print(util_str.dict_str(kwargs))
    setup(**kwargs)
