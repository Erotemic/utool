from __future__ import absolute_import, division, print_function
import shelve
#import atexit
import sys
from six.moves import range
from os.path import join, normpath
import functools
from itertools import chain
from . import util_arg
from . import util_hash
from . import util_inject
from . import util_path
from . import util_io
from . import util_str
from . import util_cplat
from ._internal.meta_util_six import get_funcname
from ._internal.meta_util_constants import (global_cache_fname,
                                            global_cache_dname,
                                            default_appname)
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[cache]')


# TODO: Remove globalness

VERBOSE = '--verbose' in sys.argv
__SHELF__ = None  # GLOBAL CACHE
__APPNAME__ = default_appname  # the global application name


def text_dict_write(fpath, key, val):
    """
    Very naive, but readable way of storing a dictionary on disk
    FIXME: This broke on RoseMary's big dataset. Not sure why. It gave bad
    syntax. And the SyntaxError did not seem to be excepted.
    """
    try:
        dict_text = util_io.read_from(fpath)
    except IOError:
        dict_text = '{}'
    try:
        dict_ = eval(dict_text)
    except SyntaxError:
        print('Bad Syntax:')
        print(dict_text)
        dict_ = {}
        if util_arg.SUPER_STRICT:
            raise
    dict_[key] = val
    dict_text2 = util_str.dict_str(dict_, strvals=False)
    if VERBOSE:
        print('[cache] ' + str(dict_text2))
    util_io.write_to(fpath, dict_text2)


def _args2_fpath(dpath, fname, cfgstr, ext, write_hashtbl=False):
    """
    Ensures that the filename is not too long (looking at you windows)
    Windows MAX_PATH=260 characters
    Absolute length is limited to 32,000 characters
    Each filename component is limited to 255 characters

    if write_hashtbl is True, hashed values expaneded and written to a text file
    """
    if len(ext) > 0 and ext[0] != '.':
        raise Exception('Fatal Error: Please be explicit and use a dot in ext')
    fname_cfgstr = fname + cfgstr
    if len(fname_cfgstr) > 128:
        hashed_cfgstr = util_hash.hashstr(cfgstr, 8)
        if write_hashtbl:
            text_dict_write(join(dpath, 'hashtbl.txt'), hashed_cfgstr, cfgstr)
        fname_cfgstr = fname + '_' + hashed_cfgstr
    fpath = join(dpath, fname_cfgstr + ext)
    fpath = normpath(fpath)
    return fpath


def save_cache(dpath, fname, cfgstr, data):
    fpath = _args2_fpath(dpath, fname, cfgstr, '.cPkl', write_hashtbl=False)
    util_io.save_cPkl(fpath, data)


def load_cache(dpath, fname, cfgstr):
    fpath = _args2_fpath(dpath, fname, cfgstr, '.cPkl')
    return util_io.load_cPkl(fpath)


class Cacher(object):
    def __init__(self, fname, cfgstr=None, cache_dir='default', appname='utool',
                 verbose=VERBOSE):
        if cache_dir == 'default':
            cache_dir = util_cplat.get_app_resource_dir(appname)
        self.dpath = cache_dir
        self.fname = fname
        self.cfgstr = cfgstr
        self.verbose = verbose

    def load(self, cfgstr=None):
        cfgstr = self.cfgstr if cfgstr is None else cfgstr
        assert cfgstr is not None, 'must specify cfgstr in constructor or call'
        assert self.fname is not None, 'no fname'
        assert self.dpath is not None, 'no dpath'
        data = load_cache(self.dpath, self.fname, cfgstr)
        if self.verbose:
            print('... ' + self.fname + ' Cacher hit')
        return data

    def tryload(self, cfgstr=None):
        try:
            if self.verbose:
                assert cfgstr is not None or self.cfgstr is not None, 'must specify cfgstr in constructor or call'
                print('[cache] tryload fname=' + self.fname)
                print('[cache] cfgstr= ' + self.cfgstr if cfgstr is None else cfgstr)
            return self.load(cfgstr)
        except IOError:
            if self.verbose:
                print('... ' + self.fname + ' Cacher miss')

    def save(self, data, cfgstr=None):
        cfgstr = self.cfgstr if cfgstr is None else cfgstr
        assert cfgstr is not None, 'must specify cfgstr in constructor or call'
        assert self.fname is not None, 'no fname'
        assert self.dpath is not None, 'no dpath'
        if self.verbose:
            print('... ' + self.fname + ' Cacher save')
        save_cache(self.dpath, self.fname, cfgstr, data)


def get_argname(func, x):
    # FINISHME
    return ('arg%d' % x)


def get_cfgstr_from_args(func, args, kwargs, key_argx, key_kwds):
    fmt_str = '%s(%s)'
    hashstr = util_hash.hashstr
    if key_argx is None:
        key_argx = range(len(args))
    if key_kwds is None:
        key_kwds = kwargs.keys()
    args_hash_iter = (fmt_str % (get_argname(func, x), hashstr(repr(args[x])))
                      for x in key_argx)
    kwds_hash_iter = (fmt_str % (key, hashstr(repr(kwargs[key])))
                      for key in key_kwds)
    cfgstr = '_'.join(chain(args_hash_iter, kwds_hash_iter))
    return cfgstr


def cached_func(fname=None, cache_dir='default', appname='utool', key_argx=None,
                key_kwds=None, use_cache=None):
    """
    Wraps a function with a Cacher object
    """
    def cached_closure(func):
        fname_ = get_funcname(func) if fname is None else fname
        cacher = Cacher(fname_, cache_dir=cache_dir, appname=appname)
        if use_cache is None:
            use_cache_ = not util_arg.get_flag('--nocache-' + fname)
        @functools.wraps(func)
        def cached_wraper(*args, **kwargs):
            # Implicitly adds use_cache to kwargs
            cfgstr = get_cfgstr_from_args(func, args, kwargs, key_argx, key_kwds)
            assert cfgstr is not None, 'cfgstr cannot be None'
            if kwargs.get('use_cache', use_cache_):
                # Make cfgstr from specified input
                data = cacher.tryload(cfgstr)
                if data is not None:
                    return data
            # Cached missed compute function
            data = func(*args, **kwargs)
            # Cache save
            cacher.save(data, cfgstr)
            return data
        # Give function a handle to the cacher object
        cached_wraper.cacher = cacher
        return cached_wraper
    return cached_closure


# --- Global Cache ---

def view_global_cache_dir(appname='default'):
    import utool
    dir_ = utool.get_global_cache_dir(appname=appname)
    utool.view_directory(dir_)


def get_global_cache_dir(appname='default', ensure=False):
    """ Returns (usually) writable directory for an application cache """
    if appname is None or  appname == 'default':
        appname = __APPNAME__
    global_cache_dir = util_cplat.get_app_resource_dir(appname, global_cache_dname)
    if ensure:
        util_path.ensuredir(global_cache_dir)
    return global_cache_dir


def get_global_shelf_fpath(appname='default', ensure=False):
    """ Returns the filepath to the global shelf """
    global_cache_dir = get_global_cache_dir(appname, ensure=ensure)
    shelf_fpath = join(global_cache_dir, global_cache_fname)
    return shelf_fpath


#def get_global_shelf(appname='default'):
#    """ Returns the global shelf object """
#    global __SHELF__
#    if __SHELF__ is None:
#        try:
#            shelf_fpath = get_global_shelf_fpath(appname, ensure=True)
#            print(shelf_fpath)
#            __SHELF__ = shelve.open(shelf_fpath)
#        except Exception as ex:
#            from . import util_dbg
#            util_dbg.printex(ex, 'Failed opening shelf_fpath',
#                             key_list=['shelf_fpath'])
#            raise
#        #shelf_file = open(shelf_fpath, 'w')
#    return __SHELF__


#@atexit.register
#def close_global_shelf(appname='default'):
#    # FIXME: If the program closes with ctrl+c this isnt called and
#    # the global cache is not written
#    global __SHELF__
#    if __SHELF__ is not None:
#        __SHELF__.close()
#    __SHELF__ = None


class GlobalShelfContext(object):
    def __init__(self, appname):
        self.appname = appname

    def __enter__(self):
        #self.shelf = get_global_shelf(self.appname)
        try:
            shelf_fpath = get_global_shelf_fpath(self.appname, ensure=True)
            if VERBOSE:
                print('[cache] open: ' + shelf_fpath)
            self.shelf = shelve.open(shelf_fpath)
        except Exception as ex:
            from . import util_dbg
            util_dbg.printex(ex, 'Failed opening shelf_fpath',
                             key_list=['shelf_fpath'])
            raise
        return self.shelf

    def __exit__(self, type_, value, trace):
        self.shelf.close()
        if trace is not None:
            print('[cache] Error under GlobalShelfContext!: ' + str(value))
            return False  # return a falsey value on error
        #close_global_shelf(self.appname)


def global_cache_read(key, appname='default', **kwargs):
    with GlobalShelfContext(appname) as shelf:
        if 'default' in kwargs:
            return shelf.get(key, kwargs['default'])
        else:
            return shelf[key]


def global_cache_dump(appname='default'):
    shelf_fpath = get_global_shelf_fpath(appname)
    print('shelf_fpath = %r' % shelf_fpath)
    with GlobalShelfContext(appname) as shelf:
        print(util_str.dict_str(shelf))


def global_cache_write(key, val, appname='default'):
    """ Writes cache files to a safe place in each operating system """
    with GlobalShelfContext(appname) as shelf:
        shelf[key] = val


def delete_global_cache(appname='default'):
    """ Reads cache files to a safe place in each operating system """
    #close_global_shelf(appname)
    shelf_fpath = get_global_shelf_fpath(appname)
    util_path.remove_file(shelf_fpath, verbose=True, dryrun=False)
