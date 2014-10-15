"""
util_config
"""
from __future__ import absolute_import, division, print_function


#import atexit
#import sys
#import inspect
#from six.moves import range
#from os.path import join, normpath
#import functools
#from itertools import chain
#from . import util_arg
#from . import util_hash
from . import util_inject
#from . import util_path
#from . import util_io
#from . import util_str
#from . import util_cplat
#from ._internal.meta_util_six import get_funcname
#from ._internal.meta_util_constants import (global_cache_fname,
#                                            global_cache_dname,
#                                            default_appname)
(print, print_, printDBG, rrr, profile) = util_inject.inject(__name__, '[config]')


def read_repo_config():
    pass


def get_default_repo_config():
    """
    import utool
    """
    CONFIG_DICT = {
        # Core Info
        'project_name': 'default_behavior',
        'authors': ['default_author'],
        'licence': None,

        # Utool Configs
        'enable_cyth': False,
        'docstr_style': 'Google',

        # Supplementary Info
        'AUTHORS': {
            #'username': {
            #    'name': None,
            #    'email': None,
            #}
            'default_author': {
                'name': 'Arthur, King of the Britans',
                'email': 'kingarthur@camelot.com',
                'quest': 'To seek the holy grail.',
                'airspeed velocity of an unladen swallow': 'What to you mean? An African or European swallow?',
            },

            'joncrall': {
                'name': 'Jon Crall',
                'email': 'erotemic@gmail.com',
            }
        }
    }
    return CONFIG_DICT


def write_default_repo_config():
    import utool
    CONFIG_DICT = utool.get_default_repo_config()
    config_str = utool.dict_str(CONFIG_DICT, strvals=True, newlines=True)
    print(config_str)
