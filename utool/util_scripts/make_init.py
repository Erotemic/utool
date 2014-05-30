#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool
import os
from os.path import splitext, basename, exists, join
from utool._internal import util_importer


if __name__ == '__main__':
    module_path = os.getcwd()
    module_name = basename(module_path)

    module_list = [splitext(fname)[0]
                   for fname in utool.ls(module_path)
                   if fname.endswith('.py') and not fname.startswith('__')]

    package_list = [fname for fname in utool.ls(module_path)
                    if exists(join(module_path, fname, '__init__.py')) and not fname.startswith('__')]

    IMPORT_TUPLES = [(modname, None) for modname in module_list + package_list]
    util_importer.dynamic_import(module_name, IMPORT_TUPLES, False, True, dry=True)
