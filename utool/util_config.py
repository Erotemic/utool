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
    REPO_CONFIG = {
        'project_name': None,
        'authors': [],
        'licence': None,
        'enable_cyth': None,
        'docstr_style': None,
    }

    return REPO_CONFIG


def get_default_global_config():
    GLOBAL_CONFIG = {
        # Supplementary Info
        'AUTHORS': {
            'username': {'name': None,
                         'email': None, },

            'kingarthur': {'name': 'Arthur, King of the Britans',
                           'email': 'kingarthur@camelot.com',
                           'permited repos': ['kotr'],
                           'aliases': [],
                           'machines': [],
                           'quest': 'To seek the holy grail.',
                           'sa5ws5%lp&30sjnk': 'What to you mean? An African or European swallow?', },

            'joncrall': {'name': 'Jon Crall',
                         'email': 'erotemic@gmail.com',
                         'permited repos': ['ibeis', 'utool', 'hesaff',
                                            'guitool', 'plottool', 'vtool'],
                         'aliases': [],
                         'machines': [], },

        },
    }
    return GLOBAL_CONFIG


def write_default_repo_config():
    import utool
    CONFIG_DICT = utool.get_default_repo_config()
    config_str = utool.dict_str(CONFIG_DICT, strvals=True, newlines=True,
                                recursive=True)
    print(config_str)
