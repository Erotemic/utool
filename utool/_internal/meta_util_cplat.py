from __future__ import absolute_import, division, print_function
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
    """ Returns a directory which should be writable for any application """
    if WIN32:
        return normpath(expanduser('~/AppData/Roaming'))
    if LINUX:
        return normpath(expanduser('~/.config'))
    if DARWIN:
        return normpath(expanduser('~/Library/Application Support'))


def get_app_resource_dir(*args):
    """ Returns a writable directory for an application
    Input: appname - the name of the application
           *args, - any other subdirectories may be specified
    """
    if len(args) == 0:
        raise AssertionError('Missing appname. The first argument the application name')
    return join(get_resource_dir(), *args)
