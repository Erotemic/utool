from __future__ import absolute_import, division, print_function
from os.path import exists
import os


def ensuredir(dpath):
    if not exists(dpath):
        os.makedirs(dpath)
