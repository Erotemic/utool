from __future__ import absolute_import, division, print_function
import shelve
#import atexit
import inspect
from six.moves import range, zip
from os.path import join, normpath, basename, exists
import functools
from itertools import chain
from zipfile import error as BadZipFile  # Screwy naming convention.
from utool import util_arg
from utool import util_hash
from utool import util_inject
from utool import util_path
from utool import util_io
from utool import util_str
from utool import util_cplat
from utool._internal.meta_util_six import get_funcname
from utool._internal.meta_util_constants import (global_cache_fname,
                                                 global_cache_dname,
                                                 default_appname)
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[cache]')


# TODO: Remove globalness

VERBOSE = util_arg.VERBOSE
QUIET = util_arg.QUIET
__SHELF__ = None  # GLOBAL CACHE
__APPNAME__ = default_appname  # the global application name


def get_default_appname():
    global __APPNAME__
    return __APPNAME__


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
    """
    save_cache

    Args:
        dpath  (str)
        fname  (str):
        cfgstr (str):
        data   (pickleable):

    Returns:
        str: fpath

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cache import *  # NOQA
        >>> import utool as ut
        >>> dpath = ut.get_app_resource_dir('utool')
        >>> fname = 'foo'
        >>> cfgstr = 'bar'
        >>> data = object()
        >>> fpath = save_cache(dpath, fname, cfgstr, data)
        >>> result = str(fpath)
        >>> print(result)
    """
    fpath = _args2_fpath(dpath, fname, cfgstr, '.cPkl', write_hashtbl=False)
    util_io.save_cPkl(fpath, data)
    return fpath


def load_cache(dpath, fname, cfgstr):
    """
    load_cache

    Args:
        dpath  (?):
        fname  (?):
        cfgstr (?):

    Returns:
        ?: data

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cache import *  # NOQA
        >>> dpath = '?'
        >>> fname = '?'
        >>> cfgstr = '?'
        >>> data = load_cache(dpath, fname, cfgstr)
        >>> result = str(data)
        >>> print(result)
    """
    fpath = _args2_fpath(dpath, fname, cfgstr, '.cPkl')
    return util_io.load_cPkl(fpath)


class Cacher(object):
    """
    old non inhertable version of cachable
    """
    def __init__(self, fname, cfgstr=None, cache_dir='default', appname='utool',
                 verbose=VERBOSE):
        if cache_dir == 'default':
            cache_dir = util_cplat.get_app_resource_dir(appname)
        util_path.ensuredir(cache_dir)
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


def get_kwdefaults(func):
    argspec = inspect.getargspec(func)
    if argspec.keywords is None or argspec.defaults is None:
        return {}
    kwdefaults = dict(zip(argspec.keywords, argspec.defaults))
    return kwdefaults


def get_argnames(func):
    argspec = inspect.getargspec(func)
    argnames = argspec.args
    return argnames


def get_cfgstr_from_args(func, args, kwargs, key_argx, key_kwds, kwdefaults, argnames):
    """
    Dev:
        argx = ['fdsf', '432443432432', 43423432, 'fdsfsd', 3.2, True]
        memlist = list(map(bytes_or_repr, argx))

    Ignore:
        argx = key_argx[0]
        argval = args[argx]
        val = argval
        %timeit repr(argval)
        %timeit any_repr(argval)
        %timeit utool.hashstr(any_repr(argval))
        %timeit memoryview(argval)
    """
    #try:
    #fmt_str = '%s(%s)'
    hashstr = util_hash.hashstr
    if key_argx is None:
        key_argx = list(range(len(args)))
    if key_kwds is None:
        key_kwds = list(kwargs.keys())

    # TODO: Use bytes or memoryview instead of repr
    # Helper funcs

    def any_repr(val):
        """ hopefully json will be able to compute a string representation """
        import json
        import numpy as np

        class NumPyArangeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()  # or map(int, obj)
                return json.JSONEncoder.default(self, obj)

        data = (json.dumps(val, cls=NumPyArangeEncoder))
        return data

    def bytes_or_repr(val):
        try:
            memview = memoryview(val)
            return memview.tobytes()
        except Exception:
            return any_repr(val)
            #if isinstance(val, dict):
            #    dict_keys = list(val.keys())
            #    dict_values = list(val.values())
            #    dict_keys_repr = bytes_or_repr(dict_keys)
            #    dict_values_repr = bytes_or_repr(dict_values)
            #    return dict_keys_repr + dict_values_repr
            #elif isinstance(val, list):
            #    pass
            #else:
            #    return repr(val)

    def hashrepr(val):
        return hashstr(bytes_or_repr(val))

    def kwdval(key):
        return kwargs.get(key, kwdefaults.get(key, None))

    arg_hashfmtstr = [argnames[argx] + '(%s)' for argx in key_argx]
    kwn_hashfmtstr = [kwdefaults.get(key, '???') + '(%s)' for key in key_kwds]
    cfgstr_fmt = '_' + '_'.join(chain(arg_hashfmtstr, kwn_hashfmtstr))
    print('cfgstr_fmt = %r' % cfgstr_fmt)
    print('hashing args')
    #import utool
    #with utool.Timer('time') as t:
    arg_hashiter = [hashrepr(args[argx]) for argx in key_argx]
    #try:
    #    if t.ellapsed > 4:
    #        utool.embed()
    #except Exception as ex:
    #    utool.embed()
    print('hashing kwargs')
    kwn_hashiter = [hashrepr(kwdval(key)) for key in key_kwds]
    print('formating args and kwargs')
    cfgstr = cfgstr_fmt % tuple(chain(arg_hashiter, kwn_hashiter))
    print('made cfgstr = %r' % cfgstr)

    ## Iter funcs
    #def argfmter(argx):
    #    try:
    #        return fmt_str % (argnames[argx], hashrepr(args[argx]))
    #    except Exception as ex:
    #        import utool
    #        msg = utool.unindent('''
    #        [argfmter] argx=%r
    #        [argfmter] len(args)=%r
    #        [argfmter] map(type, args)) = %r
    #        [argfmter] %s
    #        ''') % (argx, len(args), map(type, args), utool.func_str(func),)
    #        utool.printex(ex, msg, separate=True)
    #        raise
    #def kwdfmter(key):
    #    return fmt_str % (key, hashrepr(kwdval(key)))
    #iterfun = list
    #iterfun = iter
    #args_hash_iter = [argfmter(argx) for argx in key_argx]
    #kwds_hash_iter = [kwdfmter(key) for key in key_kwds]
    #cfgstr = '_' + '_'.join(chain(args_hash_iter, kwds_hash_iter))
    #except Exception as ex:
    #    import utool
    #    try:
    #        dbg_args_hash_iter = [argfmter(argx) for argx in key_argx if argx < len(args)]
    #        dbg_kwds_hash_iter = [kwdfmter(key) for key in key_kwds]
    #        dbg_cfgstr =  '_' + '_'.join(chain(dbg_args_hash_iter, dbg_kwds_hash_iter))
    #    except Exception:
    #        pass
    #    utool.printex(ex, keys=['key_argsx', 'key_kwds', 'kwdefaults',
    #                            dbg_cfgstr], separate=True)
    #    raise
    return cfgstr


def cached_func(fname=None, cache_dir='default', appname='utool', key_argx=None,
                key_kwds=None, use_cache=None):
    """
    Wraps a function with a Cacher object

    uses a hash of arguments as input

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> def costly_func(a, b, c='d', *args, **kwargs):
        ...    return ([a] * b, c, args, kwargs)
        >>> ans0 = costly_func(41, 3)
        >>> ans1 = costly_func(42, 3)
        >>> closure_ = utool.cached_func('costly_func', appname='utool_test', key_argx=[0, 1])
        >>> efficient_func = closure_(costly_func)
        >>> ans2 = efficient_func(42, 3)
        >>> ans3 = efficient_func(42, 3)
        >>> ans4 = efficient_func(41, 3)
        >>> ans5 = efficient_func(41, 3)
        >>> assert ans1 == ans2
        >>> assert ans2 == ans3
        >>> assert ans5 == ans4
        >>> assert ans5 == ans0
        >>> assert ans1 != ans0
    """
    def cached_closure(func):
        fname_ = get_funcname(func) if fname is None else fname
        kwdefaults = get_kwdefaults(func)
        argnames   = get_argnames(func)
        cacher = Cacher(fname_, cache_dir=cache_dir, appname=appname)
        if use_cache is None:
            use_cache_ = not util_arg.get_argflag('--nocache-' + fname)
        #_dbgdict = dict(fname_=fname_, key_kwds=key_kwds, appname=appname,
        #                key_argx=key_argx, use_cache_=use_cache_)
        @functools.wraps(func)
        def cached_wraper(*args, **kwargs):
            try:
                if True:
                    print('[utool] computing cached function fname_=%s' % (fname_,))
                # Implicitly adds use_cache to kwargs
                cfgstr = get_cfgstr_from_args(func, args, kwargs, key_argx,
                                              key_kwds, kwdefaults, argnames)
                assert cfgstr is not None, 'cfgstr=%r cannot be None' % (cfgstr,)
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
            except Exception as ex:
                import utool
                _dbgdict2 = dict(key_argx=key_argx, lenargs=len(args), lenkw=len(kwargs),)
                msg = '\n'.join([
                    '+--- UTOOL --- ERROR IN CACHED FUNCTION',
                    #'dbgdict = ' + utool.dict_str(_dbgdict),
                    'dbgdict2 = ' + utool.dict_str(_dbgdict2),
                ])
                utool.printex(ex, msg)
                raise
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
        appname = get_default_appname()
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
#            from utool import util_dbg
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


#class ShelfContext(shelve.Shelf):
#    """
#    References:
#        http://stackoverflow.com/questions/7489732/easiest-way-to-add-a-function-to-existing-class
#    """
#    def __enter__(self):
#        return self

#    def __exit__(self, exc_type, exc_value, exc_trace):
#        self.close()


class GlobalShelfContext(object):
    """ older class. might need update """
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
            from utool import util_dbg
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

#import abc  # abstract base class
#import six


#@six.add_metaclass(abc.ABCMeta)
class Cachable(object):
    """
    Abstract base class.

    This class which enables easy caching of object dictionarys

    must implement get_cfgstr()

    """
    ext = '.cPkl'  # TODO: Capt'n Proto backend to replace pickle backend

    #@abc.abstractmethod
    #def get_cfgstr(self):
    #    raise NotImplementedError('abstract method')

    #@abc.abstractmethod
    #def get_prefix(self):
    #    raise NotImplementedError('abstract method')

    def get_fname(self, cfgstr=None):
        # convinience
        return basename(self.get_fpath('', cfgstr=cfgstr))

    def get_fpath(self, cachedir, cfgstr=None):
        _fname = self.get_prefix()
        _cfgstr = self.get_cfgstr() if cfgstr is None else cfgstr
        fpath = _args2_fpath(cachedir, _fname, _cfgstr, self.ext, write_hashtbl=False)
        return fpath

    def delete(self, cachedir, cfgstr=None, verbose=True or VERBOSE or util_arg.VERBOSE):
        """
        saves query result to directory
        """
        import os
        fpath = self.get_fpath(cachedir, cfgstr=cfgstr)
        if verbose:
            print('[Cachable] cache delete: %r' % (basename(fpath),))
        os.remove(fpath)

    def save(self, cachedir, cfgstr=None, verbose=VERBOSE, quiet=QUIET):
        """
        saves query result to directory
        """
        fpath = self.get_fpath(cachedir, cfgstr=cfgstr)
        if verbose:
            print('[Cachable] cache save: %r' % (basename(fpath),))
        util_io.save_cPkl(fpath, self.__dict__)
        return fpath
        #save_cache(cachedir, '', cfgstr, self.__dict__)
        #with open(fpath, 'wb') as file_:
        #    cPickle.dump(self.__dict__, file_)

    def _unsafe_load(self, fpath):
        loaded_dict = util_io.load_cPkl(fpath)
        self.__dict__.update(loaded_dict)
        #with open(fpath, 'rb') as file_:
        #    loaded_dict = cPickle.load(file_)
        #    self.__dict__.update(loaded_dict)

    def load(self, cachedir, cfgstr=None, verbose=VERBOSE, quiet=QUIET):
        """
        Loads the result from the given database
        """
        fpath = self.get_fpath(cachedir, cfgstr=cfgstr)
        if verbose:
            print('[Cachable] cache tryload: %r' % (basename(fpath),))
        try:
            self._unsafe_load(fpath)
            if verbose:
                print('... self cache hit: %r' % (basename(fpath),))
        except IOError as ex:
            import utool as ut
            if not exists(fpath):
                msg = '... self cache miss: %r' % (basename(fpath),)
                if verbose:
                    print(msg)
                raise
            msg = '[!Cachable] Cachable(%s) is corrupt' % (self.get_cfgstr())
            ut.printex(ex, msg, iswarning=True)
            raise
        except BadZipFile as ex:
            import utool as ut
            msg = '[!Cachable] Cachable(%s) has bad zipfile' % (self.get_cfgstr())
            ut.printex(ex, msg, iswarning=True)
            raise
            #if exists(fpath):
            #    #print('[Cachable] Removing corrupted file: %r' % fpath)
            #    #os.remove(fpath)
            #    raise hsexcept.HotsNeedsRecomputeError(msg)
            #else:
            #    raise Exception(msg)
        except Exception as ex:
            import utool as ut
            ut.printex(ex, 'unknown exception while loading query result')
            raise


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_cache; utool.doctest_funcs(utool.util_cache, allexamples=True)"
        python -c "import utool, utool.util_cache; utool.doctest_funcs(utool.util_cache)"
        python -m utool.util_cache
        python -m utool.util_cache --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
