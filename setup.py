#!/usr/bin/env python
"""
pip install git+https://github.com/Erotemic/utool.git@next

Pypi:
     # Presetup
     pip install twine

     # First tag the source-code
     VERSION=$(python -c "import setup; print(setup.version)")
     echo $VERSION
     git tag $VERSION -m "tarball tag $VERSION"
     git push --tags origin master

     # NEW API TO UPLOAD TO PYPI
     # https://packaging.python.org/tutorials/distributing-packages/

     # Build wheel or source distribution
     python setup.py bdist_wheel --universal

     # Use twine to upload. This will prompt for username and password
     twine upload --username erotemic --skip-existing dist/*

     # Check the url to make sure everything worked
     https://pypi.org/project/utool/

     # ---------- OLD ----------------
     # Check the url to make sure everything worke
     https://pypi.python.org/pypi?:action=display&name=utool
"""
# -*- coding: utf-8 -*-
# Utool is released under the Apache License Version 2.0
# no warenty liability blah blah blah blah legal blah
# just use the software, don't be a jerk.
from __future__ import absolute_import, division, print_function
from setuptools import setup
import sys


def parse_version():
    """ Statically parse the version number from __init__.py """
    from os.path import dirname, join
    import ast
    init_fpath = join(dirname(__file__), 'utool', '__init__.py')
    with open(init_fpath) as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                try:
                    if target.id == '__version__':
                        self.version = node.value.s
                except AttributeError:
                    pass
    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version


version = parse_version()


def utool_setup():
    INSTALL_REQUIRES = [
        'six >= 1.8.0',
        # 'psutil >= 2.1.3',
        'parse >= 1.6.6',
        'requests',
        #'numpy >= 1.8.0',  # TODO REMOVE DEPENDENCY
        'numpy',  # 1.10 has hard time in comparison
        'pyparsing',
        'pint',
        'delorean',
        #'decorator',
    ]
    import platform

    if platform.python_version().startswith('2.7'):
        INSTALL_REQUIRES += [
            'lockfile >= 0.10.2',
            'futures',
        ]

    INSTALL_OPTIONAL = [
        'autopep8',
        'astor',
        'pyperclip >= 1.5.7',
        'pyfiglet >= 0.7.2',
        'boto'
        #pip install pygments-markdown-lexer
    ]

    #REQUIRES_LINKS = [
    #]

    #OPTIONAL_DEPENDS_LINKS = [
    #    #'git+https://github.com/amitdev/lru-dict',  # TODO REMOVE DEPENDENCY
    #    #'git+https://github.com/pwaller/pyfiglet',

    #]

    INSTALL_OPTIONAL_DEV = [  # NOQA
        'lru-dict >= 1.1.1',  # import as lru
        'guppy',
        'sphinx',
        'setproctitle',
        'sphinxcontrib-napoleon',
        'objgraph',
        'h5py',
        'delorean',
    ]

    # format optional dependencies
    INSTALL_EXTRA = {item.split(' ')[0]: item for item in INSTALL_OPTIONAL}

    # TODO: remove optional depends
    #INSTALL_OPTIONAL += INSTALL_OPTIONAL_DEV
    #INSTALL_REQUIRES += INSTALL_OPTIONAL

    try:
        # HACK: Please remove someday
        from utool import util_setup
        import utool
        from os.path import dirname
        for arg in iter(sys.argv[:]):
            # Clean clutter files
            if arg in ['clean']:
                clutter_dirs = ['cyth']
                CLUTTER_PATTERNS = [
                    '\'',
                    'cyth',
                    '*.dump.txt',
                    '*.sqlite3',
                    '*.prof',
                    '*.prof.txt',
                    '*.lprof',
                    '*.ln.pkg',
                    'failed.txt',
                    'failed_doctests.txt',
                    'failed_shelltests.txt',
                    'test_pyflann_index.flann',
                    'test_pyflann_ptsdata.npz',
                    '_timeings.txt',
                    'timeings.txt',
                    'Tgen.sh',
                    'raw_profile.*.prof',
                    'clean_profile.*.prof',
                    'raw_profile.txt',
                    'clean_profile.txt',
                    'profile_output.*',
                ]
                utool.clean(dirname(__file__), CLUTTER_PATTERNS, clutter_dirs)
        ext_modules = util_setup.find_ext_modules()
        cmdclass = util_setup.get_cmdclass()
    except Exception as ex:
        print(ex)
        ext_modules = {}
        cmdclass = {}

    # run setuptools setup function
    setup(
        name='utool',
        packages=[
            'utool',
            'utool._internal',
            'utool.tests',
            'utool.util_scripts',
        ],
        #packages=util_setup.find_packages(),
        version=version,
        description='Useful utilities',
        url='https://github.com/Erotemic/utool',
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        author='Jon Crall',
        author_email='erotemic@gmail.com',
        keywords='',
        install_requires=INSTALL_REQUIRES,
        extras_require=INSTALL_EXTRA,
        package_data={},
        scripts=[
            # 'utool/util_scripts/makesetup.py',
            'utool/util_scripts/makeinit.py',
            #'utool/util_scripts/utprof.sh',
            #'utool/util_scripts/utprof.py',
            #'utool/util_scripts/utprof_cleaner.py',
            # 'utool/util_scripts/utoolwc.py',
            # 'utool/util_scripts/grabzippedurl.py',
            # 'utool/util_scripts/autogen_sphinx_docs.py',
            # 'utool/util_scripts/permit_gitrepo.py',
            # 'utool/util_scripts/viewdir.py',
            # 'utool/util_scripts/pipinfo.py',
        ],
        classifiers=[
            # List of classifiers available at:
            # https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Utilities',
            # This should be interpreted as Apache License v2.0
            'License :: OSI Approved :: Apache Software License',
            # Supported Python versions
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
        ],
    )


if __name__ == '__main__':
    utool_setup()
