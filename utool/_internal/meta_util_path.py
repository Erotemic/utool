from __future__ import absolute_import, division, print_function
from os.path import expanduser, normpath, realpath, exists
import os


def ensuredir(dpath):
    if not exists(dpath):
        os.makedirs(dpath)


def truepath(path):
    """ Normalizes and returns absolute path with so specs """
    return normpath(realpath(expanduser(path)))


def unixpath(path):
    """ Corrects fundamental problems with windows paths.~ """
    return truepath(path).replace('\\', '/')
