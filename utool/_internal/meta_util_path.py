from __future__ import absolute_import, division, print_function
from six.moves import map
from os.path import expanduser, normpath, realpath, exists, isabs
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


def unixjoin(*args):
    isabs_list = list(map(isabs, args))
    if any(isabs_list):
        poslist = [count for count, flag in enumerate(isabs_list) if flag]
        pos = poslist[-1]
        return '/'.join(args[pos:])
    else:
        return '/'.join(args)
