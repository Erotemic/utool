#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from setuptools import setup
from Cython.Distutils import build_ext


INSTALL_REQUIRES = [
    'six',
    'numpy >= 1.8.0',
    'psutil',
]

# HACK: Please remove someday
import utool
ext_modules = utool.find_ext_modules()


if __name__ == '__main__':
    setup(
        name='utool',
        version='1.0.0.dev1',
        description='Univerally useful utility tools for you!',
        url='https://github.com/Erotemic/utool',
        ext_modules=ext_modules,
        cmdclass={'build_ext': build_ext},
        packages=[
            'utool',
            'utool._internal',
            'utool.tests',
            'utool.util_scripts',
            'cyth',
        ],
        author='Jon Crall',
        author_email='erotemic@gmail.com',
        keywords='',
        install_requires=INSTALL_REQUIRES,
        package_data={},
        scripts=['utool/util_scripts/makeinit.py',
                 'utool/util_scripts/profiler.sh',
                 'utool/util_scripts/profiler.py',
                 ],
        classifiers=[],
    )
