from __future__ import absolute_import, division, print_function
import shelve
from .meta_util_cplat import get_app_resource_dir
from .meta_util_path import ensuredir
from .meta_util_constants import global_cache_fname, global_cache_dname, default_appname
from os.path import join


def global_cache_read(key, appname=None, **kwargs):
    if appname is None:
        appname = default_appname
    global_cache_dir = get_app_resource_dir(appname, global_cache_dname)
    ensuredir(global_cache_dir)
    shelf_fpath = join(global_cache_dir, global_cache_fname)
    try:
        shelf = shelve.open(shelf_fpath)
        if 'default' in kwargs:
            return shelf.get(key, kwargs['default'])
        else:
            return shelf[key]
        shelf.close()
    except Exception as ex:
        print(ex)
        print('Error reading: shelf_fpath=%r' % shelf_fpath)
        return kwargs['default']
        #raise
