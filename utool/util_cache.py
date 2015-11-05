# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import shelve
import six
import json
#import lru
#git+https://github.com/amitdev/lru-dict
#import atexit
#import inspect
import contextlib
import collections
from six.moves import range, zip  # NOQA
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
from utool import util_inspect
from utool import util_list
from utool import util_class  # NOQA
from utool._internal import meta_util_constants
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[cache]')


# TODO: Remove globalness

VERBOSE = util_arg.VERBOSE
QUIET = util_arg.QUIET
VERBOSE_CACHE = util_arg.NOT_QUIET
USE_CACHE = not util_arg.get_argflag('--nocache')
__SHELF__ = None  # GLOBAL CACHE
__APPNAME__ = meta_util_constants.default_appname  # the global application name


class CacheMissException(Exception):
    pass


#class YACacher(object):
class ShelfCacher(object):
    """ yet another cacher """
    def __init__(self, fpath, enabled=True):
        self.fpath = fpath
        self.shelf = None if not enabled else shelve.open(fpath)

    def load(self, cachekey):
        if self.shelf is None or cachekey not in self.shelf:
            raise CacheMissException(
                'Cache miss cachekey=%r self.fpath=%r' % (cachekey, self.fpath))
        else:
            return self.shelf[cachekey]

    def save(self, cachekey, data):
        if self.shelf is not None:
            self.shelf[cachekey] = data

    def close(self):
        if self.shelf is not None:
            self.shelf.close()


def get_default_appname():
    global __APPNAME__
    return __APPNAME__


def text_dict_read(fpath):
    try:
        with open(fpath, 'r') as file_:
            dict_text = file_.read()
    except IOError:
        dict_text = '{}'
    try:
        dict_ = eval(dict_text, {}, {})
    except SyntaxError as ex:
        import utool as ut
        print(dict_text)
        ut.printex(ex, 'Bad Syntax', keys=['dict_text'])
        dict_ = {}
        if util_arg.SUPER_STRICT:
            raise
    return dict_


#def text_dict_write(fpath, key, val):
def text_dict_write(fpath, dict_):
    """
    Very naive, but readable way of storing a dictionary on disk
    FIXME: This broke on RoseMary's big dataset. Not sure why. It gave bad
    syntax. And the SyntaxError did not seem to be excepted.
    """
    #dict_ = text_dict_read(fpath)
    #dict_[key] = val
    dict_text2 = util_str.dict_str(dict_, strvals=False)
    if VERBOSE:
        print('[cache] ' + str(dict_text2))
    util_io.write_to(fpath, dict_text2)


def consensed_cfgstr(prefix, cfgstr, max_len=128, cfgstr_hashlen=16):
    if len(prefix) + len(cfgstr) > max_len:
        hashed_cfgstr = util_hash.hashstr(cfgstr, cfgstr_hashlen)
        #if write_hashtbl:
        #    # DONT WRITE TO HASHTABLE THE FUNCTION IS BROKEN
        #    text_dict_write(join(dpath, 'hashtbl.txt'), hashed_cfgstr, cfgstr)
        # Hack for prettier names
        if not prefix.endswith('_'):
            fname_cfgstr = prefix + '_' + hashed_cfgstr
        else:
            fname_cfgstr = prefix + hashed_cfgstr
    else:
        fname_cfgstr = prefix + cfgstr
    return fname_cfgstr


def _args2_fpath(dpath, fname, cfgstr, ext, write_hashtbl=False):
    r"""
    Internal util_cache helper function

    Ensures that the filename is not too long (looking at you windows)
    Windows MAX_PATH=260 characters
    Absolute length is limited to 32,000 characters
    Each filename component is limited to 255 characters

    if write_hashtbl is True, hashed values expaneded and written to a text file

    Args:
        dpath (str):
        fname (str):
        cfgstr (str):
        ext (str):
        write_hashtbl (bool):

    Returns:
        str: fpath

    CommandLine:
        python -m utool.util_cache --test-_args2_fpath

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_cache import *  # NOQA
        >>> from utool.util_cache import _args2_fpath
        >>> import utool as ut
        >>> dpath = 'F:\\data\\work\\PZ_MTEST\\_ibsdb\\_ibeis_cache'
        >>> fname = 'normalizer_'
        >>> cfgstr = u'PZ_MTEST_DSUUIDS((9)67j%dr%&bl%4oh4+)_QSUUIDS((9)67j%dr%&bl%4oh4+)zebra_plains_vsone_NN(single,K1+1,last,cks1024)_FILT(ratio<0.625;1.0,fg;1.0)_SV(0.01;2;1.57minIn=4,nRR=50,nsum,)_AGG(nsum)_FLANN(4_kdtrees)_FEATWEIGHT(ON,uselabel,rf)_FEAT(hesaff+sift_)_CHIP(sz450)'
        >>> ext = '.cPkl'
        >>> write_hashtbl = False
        >>> fpath = _args2_fpath(dpath, fname, cfgstr, ext, write_hashtbl)
        >>> result = str(ut.ensure_unixslash(fpath))
        >>> target = 'F:/data/work/PZ_MTEST/_ibsdb/_ibeis_cache/normalizer_5cv1%3s&@1dtgjlj.cPkl'
        >>> ut.assert_eq(result, target)

    """
    if len(ext) > 0 and ext[0] != '.':
        raise Exception('Fatal Error: Please be explicit and use a dot in ext')
    max_len = 128
    cfgstr_hashlen = 16  # TODO: make bigger before production
    prefix = fname
    fname_cfgstr = consensed_cfgstr(prefix, cfgstr, max_len=max_len,
                                    cfgstr_hashlen=cfgstr_hashlen)
    fpath = join(dpath, fname_cfgstr + ext)
    fpath = normpath(fpath)
    return fpath


def save_cache(dpath, fname, cfgstr, data, verbose=None):
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
    util_io.save_cPkl(fpath, data, verbose)
    return fpath


def load_cache(dpath, fname, cfgstr, verbose=None):
    """
    load_cache

    Args:
        dpath  (str):
        fname  (str):
        cfgstr (str):

    Returns:
        pickleable: data

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
    if not USE_CACHE:
        if verbose:
            print('[util_io] ... cache disabled: dpath=%s cfgstr=%r' % (basename(dpath), cfgstr,))
        raise IOError(3, 'Cache Loading Is Disabled')
    if verbose is None:
        verbose = VERBOSE_CACHE
    fpath = _args2_fpath(dpath, fname, cfgstr, '.cPkl')
    if not exists(fpath):
        if verbose:
            print('[util_io] ... cache does not exist: dpath=%s cfgstr=%r' % (
                basename(dpath), cfgstr,))
        raise IOError(2, 'No such file or directory: %r' % (fpath,))
    else:
        if verbose:
            print('[util_io] ... cache exists: dpath=%s cfgstr=%r' % (basename(dpath), cfgstr,))
    try:
        data = util_io.load_cPkl(fpath, verbose)
    except IOError:
        print('CORRUPTED? fpath = %s' % (fpath,))
        if verbose:
            print('[util_io] ... cache miss dpath=%s cfgstr=%r' % (basename(dpath), cfgstr,))
        raise
    except Exception:
        print('CORRUPTED? fpath = %s' % (fpath,))
        raise
    else:
        if verbose:
            print('[util_io] ... cache hit')
    return data


def tryload_cache(dpath, fname, cfgstr, verbose=None):
    """
    returns None if cache cannot be loaded
    """
    try:
        return load_cache(dpath, fname, cfgstr, verbose)
    except IOError:
        return None


@profile
def tryload_cache_list(dpath, fname, cfgstr_list, verbose=False):
    """
    loads a list of similar cached datas. Returns flags that needs to be computed
    """
    data_list = [tryload_cache(dpath, fname, cfgstr, verbose) for cfgstr in cfgstr_list]
    ismiss_list = [data is None for data in data_list]
    return data_list, ismiss_list


@profile
def tryload_cache_list_with_compute(dpath, fname, cfgstr_list, compute_fn, *args):
    """
    tries to load data, but computes it if it can't give a compute function
    """
    # Load precomputed values
    data_list, ismiss_list = tryload_cache_list(dpath, fname, cfgstr_list, verbose=False)
    num_total = len(cfgstr_list)
    if any(ismiss_list):
        # Compute missing values
        newdata_list = compute_fn(ismiss_list, *args)
        newcfgstr_list = util_list.list_compress(cfgstr_list, ismiss_list)
        index_list = util_list.list_where(ismiss_list)
        print('[cache] %d/%d cache hits for %s in %s' % (num_total -
                                                         len(index_list),
                                                         num_total, fname,
                                                         util_path.tail(dpath)))
        # Cache write
        for newcfgstr, newdata in zip(newcfgstr_list, newdata_list):
            save_cache(dpath, fname, newcfgstr, newdata, verbose=False)
        # Populate missing result
        for index, newdata in zip(index_list, newdata_list):
            data_list[index] = newdata
    else:
        print('[cache] %d/%d cache hits for %s in %s' % (num_total, num_total,
                                                         fname,
                                                         util_path.tail(dpath)))
    return data_list


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
                assert cfgstr is not None or self.cfgstr is not None, (
                    'must specify cfgstr in constructor or call')
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


if HAVE_NUMPY:
    class UtoolJSONEncoder(json.JSONEncoder):
        numpy_type_tuple = tuple([np.ndarray] + list(set(np.typeDict.values())))
        def default(self, obj):
            if isinstance(obj, self.numpy_type_tuple):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
else:
    UtoolJSONEncoder = json.JSONEncoder


def to_json(val):
    """
    References:
        http://stackoverflow.com/questions/11561932/why-does-json-dumpslistnp-arange5-fail-while-json-dumpsnp-arange5-tolis
    """
    json_str = (json.dumps(val, cls=UtoolJSONEncoder))
    return json_str


def from_json(json_str):
    val = json.loads(json_str)
    return val


def get_func_result_cachekey(func_, args_=tuple(), kwargs_={}):
    """
    TODO: recursive partial definitions
    kwargs = {}
    args = ([],)
    """
    from functools import partial
    import utool as ut
    # Rectify partials and whatnot
    true_args = args_
    true_kwargs = kwargs_
    true_func = func_
    if isinstance(func_, partial):
        true_func = func_.func
        if func_.args is not None:
            true_args = tuple(list(func_.args) + list(args_))
        if func_.keywords is not None:
            true_kwargs.update(func_.keywords)

    if ut.is_method(true_func):
        method = true_func
        true_func = method.im_func
        self = method.im_self
        true_args = tuple([self] + list(true_args))

    # Build up cachekey
    funcname = ut.get_funcname(true_func)
    kwdefaults = ut.get_kwdefaults(true_func, parse_source=False)
    #kwdefaults = ut.get_kwdefaults(true_func, parse_source=True)
    argnames   = ut.get_argnames(true_func)
    key_argx = None
    key_kwds = None
    func = true_func  # NOQA
    args = true_args  # NOQA
    kwargs = true_kwargs  # NOQA
    args_key = ut.get_cfgstr_from_args(true_func, true_args, true_kwargs,
                                       key_argx, key_kwds, kwdefaults,
                                       argnames)
    cachekey = funcname + '(' + args_key + ')'
    return cachekey


def cachestr_repr(val):
    """
    Representation of an object as a cache string.
    """
    try:
        memview = memoryview(val)
        return memview.tobytes()
    except Exception:
        try:
            return to_json(val)
        except Exception:
            # SUPER HACK
            if repr(val.__class__) == "<class 'ibeis.control.IBEISControl.IBEISController'>":
                return val.get_dbname()


def get_cfgstr_from_args(func, args, kwargs, key_argx, key_kwds, kwdefaults,
                         argnames, use_hash=None):
    """
    Dev:
        argx = ['fdsf', '432443432432', 43423432, 'fdsfsd', 3.2, True]
        memlist = list(map(cachestr_repr, argx))

    Ignore:
        argx = key_argx[0]
        argval = args[argx]
        val = argval
        %timeit repr(argval)
        %timeit to_json(argval)
        %timeit utool.hashstr(to_json(argval))
        %timeit memoryview(argval)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cache import *  # NOQA
        >>> use_hash = None
    """
    #try:
    #fmt_str = '%s(%s)'
    import utool as ut
    hashstr_ = util_hash.hashstr27
    if key_argx is None:
        key_argx = list(range(len(args)))
    if key_kwds is None:
        key_kwds = ut.unique_keep_order(list(kwdefaults.keys()) + list(kwargs.keys()))

    #def kwdval(key):
    #    return kwargs.get(key, kwdefaults.get(key, None))
    given_kwargs = ut.merge_dicts(kwdefaults, kwargs)

    arg_hashfmtstr = [argnames[argx] + '=(%s)' for argx in key_argx]
    #kw_hashfmtstr = [kwdefaults.get(key, '???') + '(%s)' for key in key_kwds]
    kw_hashfmtstr = [key + '=(%s)' for key in key_kwds]
    cfgstr_fmt = '_'.join(chain(arg_hashfmtstr, kw_hashfmtstr))
    #print('cfgstr_fmt = %r' % cfgstr_fmt)
    argrepr_iter = (cachestr_repr(args[argx]) for argx in key_argx)
    kwdrepr_iter = (cachestr_repr(given_kwargs[key]) for key in key_kwds)
    if use_hash is None:
        #print('conditional hashing args')
        argcfg_list = [hashstr_(argrepr) if len(argrepr) > 16 else argrepr
                       for argrepr in argrepr_iter]
        kwdcfg_list =  [hashstr_(kwdrepr) if len(kwdrepr) > 16 else kwdrepr
                        for kwdrepr in kwdrepr_iter]
    elif use_hash is True:
        #print('hashing args')
        argcfg_list = [hashstr_(argrepr) for argrepr in argrepr_iter]
        kwdcfg_list =  [hashstr_(kwdrepr) for kwdrepr in kwdrepr_iter]
    else:
        argcfg_list = list(argrepr_iter)
        kwdcfg_list = list(kwdrepr_iter)
    #print('formating args and kwargs')
    cfgstr = cfgstr_fmt % tuple(chain(argcfg_list, kwdcfg_list))
    #print('made cfgstr = %r' % cfgstr)
    return cfgstr


def cached_func(fname=None, cache_dir='default', appname='utool', key_argx=None,
                key_kwds=None, use_cache=None):
    r"""
    Wraps a function with a Cacher object

    uses a hash of arguments as input

    CommandLine:
        python -m utool.util_cache --exec-cached_func

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> def costly_func(a, b, c='d', *args, **kwargs):
        ...    return ([a] * b, c, args, kwargs)
        >>> ans0 = costly_func(41, 3)
        >>> ans1 = costly_func(42, 3)
        >>> closure_ = ut.cached_func('costly_func', appname='utool_test', key_argx=[0, 1])
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
        fname_ = util_inspect.get_funcname(func) if fname is None else fname
        kwdefaults = util_inspect.get_kwdefaults(func)
        argnames   = util_inspect.get_argnames(func)
        cacher = Cacher(fname_, cache_dir=cache_dir, appname=appname)
        if use_cache is None:
            use_cache_ = not util_arg.get_argflag('--nocache-' + fname)
        #_dbgdict = dict(fname_=fname_, key_kwds=key_kwds, appname=appname,
        #                key_argx=key_argx, use_cache_=use_cache_)
        @functools.wraps(func)
        def cached_wraper(*args, **kwargs):
            import utool as ut
            try:
                if True:
                    print('[utool] computing cached function fname_=%s' % (fname_,))
                # Implicitly adds use_cache to kwargs
                cfgstr = get_cfgstr_from_args(func, args, kwargs, key_argx,
                                              key_kwds, kwdefaults, argnames)
                if ut.WIN32:
                    # remove potentially invalid chars
                    cfgstr = '_' + ut.hashstr27(cfgstr)
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
            #except ValueError as ex:
            # handle protocal error
            except Exception as ex:
                _dbgdict2 = dict(key_argx=key_argx, lenargs=len(args), lenkw=len(kwargs),)
                msg = '\n'.join([
                    '+--- UTOOL --- ERROR IN CACHED FUNCTION',
                    #'dbgdict = ' + utool.dict_str(_dbgdict),
                    'dbgdict2 = ' + ut.dict_str(_dbgdict2),
                ])
                ut.printex(ex, msg)
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
    global_cache_dir = util_cplat.get_app_resource_dir(appname,
                                                       meta_util_constants.global_cache_dname)
    if ensure:
        util_path.ensuredir(global_cache_dir)
    return global_cache_dir


def get_global_shelf_fpath(appname='default', ensure=False):
    """ Returns the filepath to the global shelf """
    global_cache_dir = get_global_cache_dir(appname, ensure=ensure)
    shelf_fpath = join(global_cache_dir, meta_util_constants.global_cache_fname)
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


def shelf_open(fpath):
    """
    allows for shelf to be used in with statements

    References:
        http://stackoverflow.com/questions/7489732/easiest-way-to-add-a-function-to-existing-class

    CommandLine:
        python -m utool.util_cache --test-shelf_open

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> import utool as ut
        >>> fpath = ut.unixjoin(ut.ensure_app_resource_dir('utool') 'testshelf.shelf')
        >>> with ut.shelf_open(fpath) as dict_:
        ...     print(ut.dict_str(dict_))
    """
    return contextlib.closing(shelve.open(fpath))


#class YAWShelf(object):
#    def __init__(self, shelf_fpath):
#        self.shelf_fpath = shelf_fpath
#        import shelve
#        self.shelf = shelve.open(shelf_fpath)


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

    def get_fname(self, cfgstr=None, ext=None):
        # convinience
        return basename(self.get_fpath('', cfgstr=cfgstr, ext=ext))

    def get_fpath(self, cachedir, cfgstr=None, ext=None):
        """
        Ignore:
            fname = _fname
            cfgstr = _cfgstr
        """
        _dpath = cachedir
        _fname = self.get_prefix()
        _cfgstr = self.get_cfgstr() if cfgstr is None else cfgstr
        _ext =   self.ext if ext is None else ext
        write_hashtbl = False
        fpath = _args2_fpath(_dpath, _fname, _cfgstr, _ext, write_hashtbl=write_hashtbl)
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

    @profile
    def save(self, cachedir, cfgstr=None, verbose=VERBOSE, quiet=QUIET,
             ignore_keys=None):
        """
        saves query result to directory
        """
        fpath = self.get_fpath(cachedir, cfgstr=cfgstr)
        if verbose:
            print('[Cachable] cache save: %r' % (basename(fpath),))
        if ignore_keys is None:
            save_dict = self.__dict__
        else:
            save_dict = {key: val
                         for (key, val) in six.iteritems(self.__dict__)
                         if key not in ignore_keys}

        util_io.save_cPkl(fpath, save_dict)
        return fpath
        #save_cache(cachedir, '', cfgstr, self.__dict__)
        #with open(fpath, 'wb') as file_:
        #    cPickle.dump(self.__dict__, file_)

    def _unsafe_load(self, fpath, ignore_keys=None):
        loaded_dict = util_io.load_cPkl(fpath)
        if ignore_keys is not None:
            for key in ignore_keys:
                if key in loaded_dict:
                    del loaded_dict[key]
        self.__dict__.update(loaded_dict)
        #with open(fpath, 'rb') as file_:
        #    loaded_dict = cPickle.load(file_)
        #    self.__dict__.update(loaded_dict)

    @profile
    def load(self, cachedir, cfgstr=None, verbose=VERBOSE, quiet=QUIET, ignore_keys=None):
        """
        Loads the result from the given database
        """
        fpath = self.get_fpath(cachedir, cfgstr=cfgstr)
        if verbose:
            print('[Cachable] cache tryload: %r' % (basename(fpath),))
        try:
            self._unsafe_load(fpath, ignore_keys)
            if verbose:
                print('... self cache hit: %r' % (basename(fpath),))
        except ValueError as ex:
            import utool as ut
            msg = '[!Cachable] Cachable(%s) is likely corrupt' % (self.get_cfgstr())
            print('CORRUPT fpath = %s' % (fpath,))
            ut.printex(ex, msg, iswarning=True)
            raise
        except BadZipFile as ex:
            import utool as ut
            msg = '[!Cachable] Cachable(%s) has bad zipfile' % (self.get_cfgstr())
            print('CORRUPT fpath = %s' % (fpath,))
            ut.printex(ex, msg, iswarning=True)
            raise
            #if exists(fpath):
            #    #print('[Cachable] Removing corrupted file: %r' % fpath)
            #    #os.remove(fpath)
            #    raise hsexcept.HotsNeedsRecomputeError(msg)
            #else:
            #    raise Exception(msg)
        except IOError as ex:
            import utool as ut
            if not exists(fpath):
                msg = '... self cache miss: %r' % (basename(fpath),)
                if verbose:
                    print(msg)
                raise
            print('CORRUPT fpath = %s' % (fpath,))
            msg = '[!Cachable] Cachable(%s) is corrupt' % (self.get_cfgstr())
            ut.printex(ex, msg, iswarning=True)
            raise
        except Exception as ex:
            import utool as ut
            ut.printex(ex, 'unknown exception while loading query result')
            raise


def get_lru_cache(max_size=5):
    """
    Args:
        max_size (int):

    References:
        https://github.com/amitdev/lru-dict

    CommandLine:
        python -m utool.util_cache --test-get_lru_cache

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> from utool.util_cache import *  # NOQA
        >>> import utool as ut  # NOQA
        >>> # build test data
        >>> max_size = 5
        >>> # execute function
        >>> cache_obj = get_lru_cache(max_size)
        >>> cache_obj[1] = 1
        >>> cache_obj[2] = 2
        >>> cache_obj[3] = 3
        >>> cache_obj[4] = 4
        >>> cache_obj[5] = 5
        >>> cache_obj[6] = 6
        >>> # verify results
        >>> result = str(ut.dict_str(cache_obj, nl=False))
        >>> print(result)
        {2: 2, 3: 3, 4: 4, 5: 5, 6: 6,}
    """
    USE_C_LRU = False
    if USE_C_LRU:
        #import utool as ut
        import lru
        #lru = ut.tryimport('lru', 'git+https://github.com/amitdev/lru-dict', ensure=True)
        cache_obj = lru.LRU(max_size)
    else:
        cache_obj = LRUDict(max_size)
    return cache_obj


class LRUDict(object):
    """
    Pure python implementation for lru cache fallback

    References:
        http://www.kunxi.org/blog/2014/05/lru-cache-in-python/

    Args:
        max_size (int): (default = 5)

    Returns:
        LRUDict: cache_obj

    CommandLine:
        python -m utool.util_cache --test-LRUDict

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_cache import *  # NOQA
        >>> max_size = 5
        >>> self = LRUDict(max_size)
        >>> for count in range(0, 5):
        ...     self[count] = count
        >>> print(self)
        >>> self[0]
        >>> for count in range(5, 8):
        ...     self[count] = count
        >>> print(self)
        >>> del self[5]
        >>> assert 4 in self
        >>> result = ('self = %r' % (self,))
        >>> print(result)
        self = LRUDict({
            4: 4,
            0: 0,
            6: 6,
            7: 7,
        })
    """

    def __init__(self, max_size):
        self._max_size = max_size
        self._cache = collections.OrderedDict()

    def has_key(self, item):
        return item in self

    def __contains__(self, item):
        return item in self._cache

    def __delitem__(self, key):
        del self._cache[key]

    def __str__(self):
        import utool as ut
        return ut.dict_str(self._cache, nl=False)

    def __repr__(self):
        import utool as ut
        return 'LRUDict(' + ut.dict_str(self._cache) + ')'
        #return repr(self._cache)

    def __iter__(self):
        return iter(self._cache)

    def items(self):
        return self._cache.items()

    def keys(self):
        return self._cache.keys()

    def values(self):
        return self._cache.values()

    def iteritems(self):
        return self._cache.iteritems()

    def iterkeys(self):
        return self._cache.iterkeys()

    def itervalues(self):
        return self._cache.itervalues()

    def clear(self):
        return self._cache.clear()

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, key):
        try:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        except KeyError:
            raise

    def __setitem__(self, key, value):
        try:
            self._cache.pop(key)
        except KeyError:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value


def time_different_diskstores():
    """
    %timeit shelf_write_test()    # 15.1 ms per loop
    %timeit cPickle_write_test()  # 1.26 ms per loop

    %timeit shelf_read_test()     # 8.77 ms per loop
    %timeit cPickle_read_test()   # 2.4 ms per loop
    %timeit cPickle_read_test2()  # 2.35 ms per loop

    %timeit json_read_test()
    %timeit json_write_test()
    """
    import six
    import uuid
    import simplejson as json
    import cPickle
    import utool as ut
    shelf_path = 'test.shelf'
    json_path = 'test.json'
    cpkl_path = 'test.pkl'
    size = 1000
    dict_ = {str(key): str(uuid.uuid4()) for key in range(size)}
    ut.delete(cpkl_path)
    ut.delete(json_path)
    ut.delete(shelf_path)

    def shelf_write_test():
        with ut.shelf_open(shelf_path) as shelf_dict:
            shelf_dict.update(dict_)

    def shelf_read_test():
        with ut.shelf_open(shelf_path) as shelf_dict:
            test = {key: val for key, val in six.iteritems(shelf_dict)}
        assert len(test) > 0

    def json_write_test():
        with open(json_path, 'wb') as outfile:
            json.dump(dict_, outfile)

    def cPickle_write_test():
        with open(cpkl_path, 'wb') as outfile:
            cPickle.dump(dict_, outfile)

    def cPickle_read_test():
        with open(cpkl_path, 'rb') as outfile:
            test = {key: val for key, val in six.iteritems(cPickle.load(outfile))}
        assert len(test) > 0

    def cPickle_read_test2():
        with open(cpkl_path, 'rb') as outfile:
            test = cPickle.load(outfile)
        assert len(test) > 0

    shelf_write_test()
    shelf_read_test()
    #json_write_test()
    #json_read_test()
    cPickle_write_test()
    cPickle_read_test()
    cPickle_read_test2()


@six.add_metaclass(util_class.ReloadingMetaclass)
class LazyDict(object):
    #class LazyDict(collections.Mapping):
    """
    Hacky dictionary where values that are functions are counted as lazy


    CommandLine:
        python -m utool.util_cache --exec-LazyDict

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_cache import *  # NOQA
        >>> import utool as ut
        >>> self = ut.LazyDict()
        >>> self['foo'] = lambda: 5
        >>> self['bar'] = 4
        >>> try:
        >>>     self['foo'] = lambda: 9
        >>>     assert False, 'should not be able to override computable functions'
        >>> except ValueError:
        >>>     pass
        >>> self['biz'] = lambda: 9
        >>> d = {}
        >>> d.update(**self)
        >>> self['spam'] = lambda: 'eggs'
        >>> self.printinfo()
        >>> print(self.tostring(is_eager=False))
    """
    def __init__(self, other=None, is_eager=True, verbose=False, **kwargs):
        # Registered lazy evaluations
        self._eval_funcs = {}
        # Computed results
        self._stored_results = {}
        self.infer_lazy_vals_hack = True
        self._is_eager = is_eager
        self._verbose = verbose
        self.reprkw = dict(is_eager=False, nl=False)
        if other is not None:
            self.update(other)
        if len(kwargs) > 0:
            self.update(kwargs)

    # --- direct interface

    def set_lazy_func(self, key, func):
        from utool import util_type
        assert util_type.is_funclike(func), 'func must be a callable'
        #if key in self._stored_results:
        #    raise ValueError(
        #        ('Cannot add new lazy function for key=%r'
        #         'that has been computed') % (key,))
        #if key in self._stored_results:
        if key in self.reconstructable_keys():
            raise ValueError(
                ('Cannot overwrite lazy function for key=%r') % (key,))
        self._eval_funcs[key] = func

    def setitem(self, key, value):
        from utool import util_type
        # HACK, lazy funcs should all be registered
        # this should should always just set a value
        if key in self.reconstructable_keys():
            raise ValueError(
                ('Cannot overwrite lazy function for key=%r') % (key,))
        if (self.infer_lazy_vals_hack and
             util_type.is_funclike(value)):
            self.set_lazy_func(key, value)
        else:
            self._stored_results[key] = value

    def getitem(self, key, is_eager=None):
        if is_eager is None:
            is_eager = self._is_eager
        if is_eager:
            return self.eager_eval(key)
        else:
            return self.lazy_eval(key)

    def nocache_eval(self, key):
        """ forces function evaluation """
        value = self._eval_funcs[key]()
        return value

    def eager_eval(self, key):
        if key in self._stored_results:
            value  = self._stored_results[key]
        else:
            if self._verbose:
                print('[util_cache] Evaluating key=%r' % (key,))
            value = self.nocache_eval(key)
            self._stored_results[key] = value
        return value

    def lazy_eval(self, key):
        if key in self._stored_results:
            value  = self._stored_results[key]
        else:
            value = self._eval_funcs[key]
        return value

    def clear_evaluated(self):
        for key in list(self.evaluated_keys()):
            del self._stored_results[key]

    def clear_stored(self, keys=None):
        if keys is None:
            keys = list(self.stored_keys())
        for key in keys:
            del self._stored_results[key]

    def stored_keys(self):
        """ keys whose vals that have been explicitly set or evaluated """
        return self._stored_results.keys()

    def reconstructable_keys(self):
        """ only keys whose vals that have been set with a backup func """
        return set(self._eval_funcs.keys())

    def all_keys(self):
        return set(self.stored_keys()).union(set(self.reconstructable_keys()))

    def unevaluated_keys(self):
        """ keys whose vals can be constructed but have not been """
        return set(self.reconstructable_keys()) - set(self.stored_keys())

    def evaluated_keys(self):
        """ only keys whose vals have been evaluated from a stored function """
        return set(self.reconstructable_keys()) - set(self.unevaluated_keys())

    def nonreconstructable_keys(self):
        """ only keys whose vals that have been explicitly set without a backup func """
        return set(self.all_keys()) - self.reconstructable_keys()

    def printinfo(self):
        print('nonreconstructable_keys = %s' % (self.nonreconstructable_keys(),))
        print('reconstructable_keys = %s' % (self.reconstructable_keys(),))
        print('evaluated_keys = %s' % (self.evaluated_keys(),))
        print('unevaluated_keys = %s' % (self.unevaluated_keys(),))

    def asdict(self, is_eager=None):
        dict_ = {key: self.getitem(key, is_eager) for key in self.keys()}
        return dict_

    def tostring(self, is_eager=None, **kwargs):
        import utool as ut
        dict_ = self.asdict(is_eager=is_eager)
        return ut.dict_str(dict_, **kwargs)

    # --- dict interface

    def __setitem__(self, key, value):
        self.setitem(key, value)

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key, *d):
        assert len(d) == 0, 'no support for default yet'
        return self.getitem(key, self._is_eager)

    def update(self, dict_, **kwargs):
        for key, val in six.iteritems(dict_):
            self[key] = val
        for key, val in six.iteritems(kwargs):
            self[key] = val

    def keys(self):
        return self.all_keys()

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def __str__(self):
        return self.tostring()

    def __repr__(self):
        return self.tostring(**self.reprkw)

    #def __getstate__(self):
    #    state_dict = self.asdict()
    #    return state_dict

    #def __setstate__(self, state_dict):
    #    self._stored_results.update(state_dict)


@six.add_metaclass(util_class.ReloadingMetaclass)
class LazyList(object):
    """ very hacky list implemented as a dictionary """
    def __init__(self, **kwargs):
        self._hackstore = LazyDict(**kwargs)

    def __len__(self):
        return len(self._hackstore)

    def __getitem__(self, index):
        try:
            return self._hackstore[index]
        except KeyError:
            raise ValueError('index=%r out of bounds' % (index,))

    def append(self, item):
        self._hackstore[len(self._hackstore)] = item

    def tolist(self):
        return self._hackstore.values()

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_cache; utool.doctest_funcs(utool.util_cache)"
        python -m utool.util_cache
        python -m utool.util_cache --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
