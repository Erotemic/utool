#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from setuptools import setup


INSTALL_REQUIRES = [
    'numpy >= 1.8.0',
    'psutil',
]

if __name__ == '__main__':
    setup(
        name='utool',
        version='1.0.0.dev1',
        description='Univerally useful utility tools for you!',
        url='https://github.com/Erotemic/utool',
        packages=['utool'],
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
