# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import six
from os.path import normpath, expanduser, join

OS_TYPE = sys.platform
if six.PY2 and OS_TYPE == 'linux2':
    OS_TYPE = 'linux'  # python2 fix

WIN32  = sys.platform.startswith('win32')
LINUX  = sys.platform.startswith('linux')
DARWIN = sys.platform.startswith('darwin')


def get_resource_dir():
    """
    Returns a directory which should be writable for any application
    """
    #resource_prefix = '~'
    if WIN32:
        dpath_ = '~/AppData/Roaming'
    elif LINUX:
        dpath_ = '~/.config'
    elif DARWIN:
        dpath_  = '~/Library/Application Support'
    else:
        raise AssertionError('unknown os')
    dpath = normpath(expanduser(dpath_))
    return dpath


def get_app_resource_dir(*args, **kwargs):
    """ Returns a writable directory for an application
    Input: appname - the name of the application
           *args, - any other subdirectories may be specified
    """
    if len(args) == 0:
        raise AssertionError('Missing appname. The first argument the '
                             'application name')
    return join(get_resource_dir(), *args)
