#!/usr/bin/env python2.7
"""
pip install git+https://github.com/Erotemic/utool.git@next
"""
# -*- coding: utf-8 -*-
# Utool is released under the Apache License Version 2.0
# no warenty liability blah blah blah blah legal blah
# just use the software, don't be a jerk.
from __future__ import absolute_import, division, print_function
from setuptools import setup
import sys


version = '1.6.1.dev1'


def pypi_publish():
    """
     References:
        https://packaging.python.org/en/latest/distributing.html#uploading-your-project-to-pypi
        http://peterdowns.com/posts/first-time-with-pypi.html
     CommandLine:
         UTVERSION=$(python -c "import setup; print(setup.version)")
         echo $UTVERSION
         git tag $UTVERSION -m "tarball tag $UTVERSION"
         git push --tags origin master
         git push --tags Erotemic next
         git push --tags Erotemic master

         # Register on Pypi test
         python setup.py register -r pypitest
         python setup.py sdist upload -r pypitest

         # Register on Pypi live
         python setup.py register -r pypi
         python setup.py sdist upload -r pypi
     Notes:
         this file needs to be in my home directory apparently
         ~/.pypirc

    """
    pass


def get_tarball_download_url(version):
    download_url = 'https://github.com/erotemic/utool/tarball/' + version
    return download_url


def utool_setup():
    INSTALL_REQUIRES = [
        'six >= 1.8.0',
        'psutil >= 2.1.3',
        'parse >= 1.6.6',
        #'numpy >= 1.8.0',  # TODO REMOVE DEPENDENCY
        'numpy',  # 1.10 has hard time in comparison
        'pyparsing',
        #'decorator',
    ]
    import platform

    if platform.python_version().startswith('2.7'):
        INSTALL_REQUIRES += [
            'lockfile >= 0.10.2',
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
        download_url=get_tarball_download_url(version),
        description='Univerally useful utility tools for you!',
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
            'utool/util_scripts/makesetup.py',
            'utool/util_scripts/makeinit.py',
            #'utool/util_scripts/utprof.sh',
            #'utool/util_scripts/utprof.py',
            #'utool/util_scripts/utprof_cleaner.py',
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
