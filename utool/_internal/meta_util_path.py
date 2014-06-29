from __future__ import absolute_import, division, print_function
from os.path import expanduser, normpath, realpath, exists
import os


def ensuredir(dpath):
    if not exists(dpath):
        os.makedirs(dpath)


def truepath(path):
    return normpath(realpath(expanduser(path)))


def unixpath(path):
    return truepath(path).replace('\\', '/')
