#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from setuptools import setup
import sys


def utool_setup():
    INSTALL_REQUIRES = [
        'six >= 1.8.0',
        'psutil >= 2.1.3',
        'parse >= 1.6.6',
        #'decorator',
    ]

    INSTALL_OPTIONAL = [
        'numpy >= 1.8.0',  # TODO REMOVE DEPENDENCY
        'astor',
        'sphinx',
        'sphinxcontrib-napoleon',
        #'git+https://github.com/amitdev/lru-dict',  # TODO REMOVE DEPENDENCY
    ]

    INSTALL_OPTIONAL_EXTRA = [  # NOQA
        'guppy',
        'objgraph',
        'git+https://github.com/pwaller/pyfiglet',
    ]

    INSTALL_REQUIRES += INSTALL_OPTIONAL

    try:
        # HACK: Please remove someday
        from utool import util_setup
        import utool
        from os.path import dirname
        for arg in iter(sys.argv[:]):
            # Clean clutter files
            if arg in ['clean']:
                clutter_dirs = ['cyth']
                clutter_patterns = ['cyth']
                utool.clean(dirname(__file__), clutter_patterns, clutter_dirs)
        ext_modules = util_setup.find_ext_modules()
        cmdclass = util_setup.get_cmdclass()
    except Exception as ex:
        print(ex)
        ext_modules = {}
        cmdclass = {}

    setup(
        name='utool',
        packages=[
            'utool',
            'utool._internal',
            'utool.tests',
            'utool.util_scripts',
        ],
        #packages=util_setup.find_packages(),
        version='1.0.0.dev1',
        description='Univerally useful utility tools for you!',
        url='https://github.com/Erotemic/utool',
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        author='Jon Crall',
        author_email='erotemic@gmail.com',
        keywords='',
        install_requires=INSTALL_REQUIRES,
        package_data={},
        scripts=[
            'utool/util_scripts/makesetup.py',
            'utool/util_scripts/makeinit.py',
            'utool/util_scripts/profiler.sh',
            'utool/util_scripts/profiler.py',
            'utool/util_scripts/profiler_cleaner.py',
            'utool/util_scripts/utoolwc.py',
            'utool/util_scripts/grabzippedurl.py',
            'utool/util_scripts/autogen_sphinx_docs.py',
            'utool/util_scripts/permit_gitrepo.py',
            'utool/util_scripts/viewdir.py',
        ],
        classifiers=[],
    )


if __name__ == '__main__':
    utool_setup()
