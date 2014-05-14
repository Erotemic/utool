#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from utool.util_setup import setuptools_setup
import utool

if __name__ == '__main__':
    setuptools_setup(
        setup_fpath=__file__,
        module=utool,
        description='Useful utility tools for you!',
        url='https://github.com/Erotemic/utool',
        packages=['utool'],
        author='Jon Crall',
        author_email='erotemic@gmail.com',
        keywords='',
        package_data={},
        classifiers=[],
    )
