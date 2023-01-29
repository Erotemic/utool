# -*- coding: utf-8 -*-
"""
This module needs serious refactoring and testing
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import shelve
import six
import uuid
import json
import codecs
import os
#import lru
#git+https://github.com/amitdev/lru-dict
#import atexit
#import inspect
import contextlib
import collections
from six.moves import cPickle as pickle
from six.moves import range, zip
from os.path import join, normpath, basename, exists
from functools import partial
from itertools import chain
import zipfile
from utool import util_arg
from utool import util_hash
from utool import util_inject
from utool import util_path
from utool import util_io
from utool import util_str
from utool import util_cplat
from utool import util_inspect
from utool import util_list
from utool import util_class
from utool import util_type
from utool import util_decor  # NOQA
from utool import util_dict
from utool._internal import meta_util_constants
print, rrr, profile = util_inject.inject2(__name__)


# TODO: Remove globalness

VERBOSE = util_arg.VERBOSE
QUIET = util_arg.QUIET
VERBOSE_CACHE = util_arg.NOT_QUIET
USE_CACHE = not util_arg.get_argflag('--nocache')
__APPNAME__ = meta_util_constants.default_appname  # the global application name


class CacheMissException(Exception):
    pass


#class YACacher(object):
# @six.add_metaclass(util_class.ReloadingMetaclass)
@util_class.reloadable_class
class ShelfCacher(object):
    """ yet another cacher """
    def __init__(self, fpath, enabled=True):
        self.verbose = True
        if self.verbose:
            print('[shelfcache] initializing()')
        self.fpath = fpath
        self.shelf = None if not enabled else shelve.open(fpath)

    def __del__(self):
        self.close()

    def __getitem__(self, cachekey):
        return self.load(cachekey)

    def __setitem__(self, cachekey, data):
        return self.save(cachekey, data)

    def keys(self):
        return self.shelf.keys()

    def load(self, cachekey):
        if self.verbose:
            print('[shelfcache] loading %s' % (cachekey,))

        cachekey = cachekey.encode('ascii')
        if self.shelf is None or cachekey not in self.shelf:
            raise CacheMissException(
                'Cache miss cachekey=%r self.fpath=%r' % (cachekey, self.fpath))
        else:
            return self.shelf[cachekey]

    def save(self, cachekey, data):
        if self.verbose:
            print('[shelfcache] saving %s' % (cachekey,))

        cachekey = cachekey.encode('ascii')
        if self.shelf is not None:
            self.shelf[cachekey] = data
        self.shelf.sync()

    def clear(self):
        if self.verbose:
            print('[shelfcache] clearing cache')
        self.shelf.clear()
        self.shelf.sync()

    def close(self):
        if self.verbose:
            print('[shelfcache] closing()')
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


def text_dict_write(fpath, dict_):
    """
    Very naive, but readable way of storing a dictionary on disk
    FIXME: This broke on RoseMary's big dataset. Not sure why. It gave bad
    syntax. And the SyntaxError did not seem to be excepted.
    """
    #dict_ = text_dict_read(fpath)
    #dict_[key] = val
    dict_text2 = util_str.repr4(dict_, strvals=False)
    if VERBOSE:
        print('[cache] ' + str(dict_text2))
    util_io.write_to(fpath, dict_text2)


def consensed_cfgstr(prefix, cfgstr, max_len=128, cfgstr_hashlen=16):
    if len(prefix) + len(cfgstr) > max_len:
        hashed_cfgstr = util_hash.hashstr27(cfgstr, hashlen=cfgstr_hashlen)
        # Hack for prettier names
        if not prefix.endswith('_'):
            fname_cfgstr = prefix + '_' + hashed_cfgstr
        else:
            fname_cfgstr = prefix + hashed_cfgstr
    else:
        fname_cfgstr = prefix + cfgstr
    return fname_cfgstr


def _args2_fpath(dpath, fname, cfgstr, ext):
    r"""
    Ensures that the filename is not too long

    Internal util_cache helper function
    Windows MAX_PATH=260 characters
    Absolute length is limited to 32,000 characters
    Each filename component is limited to 255 characters

    Args:
        dpath (str):
        fname (str):
        cfgstr (str):
        ext (str):

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
        >>> fpath = _args2_fpath(dpath, fname, cfgstr, ext)
        >>> result = str(ut.ensure_unixslash(fpath))
        >>> target = 'F:/data/work/PZ_MTEST/_ibsdb/_ibeis_cache/normalizer_xfylfboirymmcpfg.cPkl'
        >>> ut.assert_eq(result, target)

    """
    if len(ext) > 0 and ext[0] != '.':
        raise ValueError('Please be explicit and use a dot in ext')
    max_len = 128
    # should hashlen be larger?
    cfgstr_hashlen = 16
    prefix = fname
    fname_cfgstr = consensed_cfgstr(prefix, cfgstr, max_len=max_len,
                                    cfgstr_hashlen=cfgstr_hashlen)
    fpath = join(dpath, fname_cfgstr + ext)
    fpath = normpath(fpath)
    return fpath


def save_cache(dpath, fname, cfgstr, data, ext='.cPkl', verbose=None):
    """
    Saves data using util_io, but smartly constructs a filename
    """
    fpath = _args2_fpath(dpath, fname, cfgstr, ext)
    util_io.save_data(fpath, data, verbose=verbose)
    return fpath


def load_cache(dpath, fname, cfgstr, ext='.cPkl', verbose=None, enabled=True):
    """
    Loads data using util_io, but smartly constructs a filename
    """
    if verbose is None:
        verbose = VERBOSE_CACHE
    if not USE_CACHE or not enabled:
        if verbose > 1:
            print('[util_cache] ... cache disabled: dpath=%s cfgstr=%r' %
                    (basename(dpath), cfgstr,))
        raise IOError(3, 'Cache Loading Is Disabled')
    fpath = _args2_fpath(dpath, fname, cfgstr, ext)
    if not exists(fpath):
        if verbose > 0:
            print('[util_cache] ... cache does not exist: dpath=%r fname=%r cfgstr=%r' % (
                basename(dpath), fname, cfgstr,))
        raise IOError(2, 'No such file or directory: %r' % (fpath,))
    else:
        if verbose > 2:
            print('[util_cache] ... cache exists: dpath=%r fname=%r cfgstr=%r' % (
                basename(dpath), fname, cfgstr,))
        import utool as ut
        nbytes = ut.get_file_nBytes(fpath)
        big_verbose = (nbytes > 1E6 and verbose > 2) or verbose > 2
        if big_verbose:
            print('[util_cache] About to read file of size %s' % (ut.byte_str2(nbytes),))
    try:
        with ut.Timer(fpath, verbose=big_verbose and verbose > 3):
            data = util_io.load_data(fpath, verbose=verbose > 2)
    except (EOFError, IOError, ImportError) as ex:
        print('CORRUPTED? fpath = %s' % (fpath,))
        if verbose > 1:
            print('[util_cache] ... cache miss dpath=%s cfgstr=%r' % (
                basename(dpath), cfgstr,))
        raise IOError(str(ex))
    except Exception:
        print('CORRUPTED? fpath = %s' % (fpath,))
        raise
    else:
        if verbose > 2:
            print('[util_cache] ... cache hit')
    return data


def tryload_cache(dpath, fname, cfgstr, verbose=None):
    """
    returns None if cache cannot be loaded
    """
    try:
        return load_cache(dpath, fname, cfgstr, verbose=verbose)
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
def tryload_cache_list_with_compute(use_cache, dpath, fname, cfgstr_list,
                                    compute_fn, *args):
    """
    tries to load data, but computes it if it can't give a compute function
    """
    # Load precomputed values
    if use_cache is False:
        data_list = [None] * len(cfgstr_list)
        ismiss_list = [True] * len(cfgstr_list)
        # Don't load or save, just compute
        data_list = compute_fn(ismiss_list, *args)
        return data_list
    else:
        data_list, ismiss_list = tryload_cache_list(dpath, fname, cfgstr_list,
                                                    verbose=False)
    num_total = len(cfgstr_list)
    if any(ismiss_list):
        # Compute missing values
        newdata_list = compute_fn(ismiss_list, *args)
        newcfgstr_list = util_list.compress(cfgstr_list, ismiss_list)
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
    def __init__(self, fname, cfgstr=None, cache_dir='default',
                 appname='utool', ext='.cPkl', verbose=None,
                 enabled=True):
        if verbose is None:
            verbose = VERBOSE
        if cache_dir == 'default':
            cache_dir = util_cplat.get_app_resource_dir(appname)
        util_path.ensuredir(cache_dir)
        self.dpath = cache_dir
        self.fname = fname
        self.cfgstr = cfgstr
        self.verbose = verbose
        self.ext = ext
        self.enabled = enabled

    def get_fpath(self):
        fpath = _args2_fpath(self.dpath, self.fname, self.cfgstr, self.ext)
        return fpath

    def existing_versions(self):
        """
        Returns data with different cfgstr values that were previously computed
        with this cacher.
        """
        import glob
        pattern = self.fname + '_*' + self.ext
        for fname in glob.glob1(self.dpath, pattern):
            fpath = join(self.dpath, fname)
            yield fpath

    def exists(self, cfgstr=None):
        return exists(self.get_fpath())

    def load(self, cfgstr=None):
        cfgstr = self.cfgstr if cfgstr is None else cfgstr
        # assert cfgstr is not None, 'must specify cfgstr in constructor or call'
        if cfgstr is None:
            import warnings
            warnings.warn('No cfgstr given in Cacher constructor or call')
            cfgstr = ''
        assert self.fname is not None, 'no fname'
        assert self.dpath is not None, 'no dpath'
        # TODO: use the computed fpath from this object instead
        data = load_cache(self.dpath, self.fname, cfgstr, self.ext,
                          verbose=self.verbose, enabled=self.enabled)
        if self.verbose > 1:
            print('[cache] ... ' + self.fname + ' Cacher hit')
        return data

    def tryload(self, cfgstr=None):
        """
        Like load, but returns None if the load fails
        """
        if cfgstr is None:
            cfgstr = self.cfgstr
        if cfgstr is None:
            import warnings
            warnings.warn('No cfgstr given in Cacher constructor or call')
            cfgstr = ''
        # assert cfgstr is not None, (
        #     'must specify cfgstr in constructor or call')
        if not self.enabled:
            if self.verbose > 0:
                print('[cache] ... %s Cacher disabled' % (self.fname))
            return None
        try:
            if self.verbose > 1:
                print('[cache] tryload fname=%s' % (self.fname,))
                # if self.verbose > 2:
                #     print('[cache] cfgstr=%r' % (cfgstr,))
            return self.load(cfgstr)
        except IOError:
            if self.verbose > 0:
                print('[cache] ... %s Cacher miss' % (self.fname))

    def ensure(self, func, *args, **kwargs):
        data = self.tryload()
        if data is None:
            data = func(*args, **kwargs)
            self.save(data)
        return data

    def save(self, data, cfgstr=None):
        if not self.enabled:
            return
        cfgstr = self.cfgstr if cfgstr is None else cfgstr
        # assert cfgstr is not None, 'must specify cfgstr in constructor or call'
        if cfgstr is None:
            import warnings
            warnings.warn('No cfgstr given in Cacher constructor or call')
            cfgstr = ''
        assert self.fname is not None, 'no fname'
        assert self.dpath is not None, 'no dpath'
        if self.verbose > 0:
            print('[cache] ... ' + self.fname + ' Cacher save')
        save_cache(self.dpath, self.fname, cfgstr, data, self.ext)


#@util_decor.memoize
def make_utool_json_encoder(allow_pickle=False):
    """
    References:
        http://stackoverflow.com/questions/8230315/python-sets-are
        http://stackoverflow.com/questions/11561932/why-does-json
        https://github.com/jsonpickle/jsonpickle
        http://stackoverflow.com/questions/24369666/typeerror-b1
        http://stackoverflow.com/questions/30469575/how-to-pickle
    """
    import utool as ut
    PYOBJECT_TAG = '__PYTHON_OBJECT__'
    UUID_TAG = '__UUID__'
    SLICE_TAG = '__SLICE__'

    def decode_pickle(text):
        obj = pickle.loads(codecs.decode(text.encode(), 'base64'))
        return obj

    def encode_pickle(obj):
        try:
            # Use protocol 2 to support both python2.7 and python3
            COMPATIBLE_PROTOCOL = 2
            pickle_bytes = pickle.dumps(obj, protocol=COMPATIBLE_PROTOCOL)
        except Exception:
            raise
        text = codecs.encode(pickle_bytes, 'base64').decode()
        return text

    type_to_tag = collections.OrderedDict([
        (slice, SLICE_TAG),
        (uuid.UUID, UUID_TAG),
        (object, PYOBJECT_TAG),
    ])

    tag_to_type = {tag: type_ for type_, tag in type_to_tag.items()}

    def slice_part(c):
        return '' if c is None else str(c)

    def encode_slice(s):
        parts = [slice_part(s.start), slice_part(s.stop), slice_part(s.step)]
        return ':'.join(parts)

    def decode_slice(x):
        return ut.smart_cast(x, slice)

    encoders = {
        UUID_TAG: str,
        SLICE_TAG: encode_slice,
        PYOBJECT_TAG: encode_pickle,
    }

    decoders = {
        UUID_TAG: uuid.UUID,
        SLICE_TAG: decode_slice,
        PYOBJECT_TAG: decode_pickle,
    }

    if not allow_pickle:
        del encoders[PYOBJECT_TAG]
        del decoders[PYOBJECT_TAG]
        type_ = tag_to_type[PYOBJECT_TAG]
        del tag_to_type[PYOBJECT_TAG]
        del type_to_tag[type_]

    class UtoolJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            need_else = False
            if isinstance(obj, util_type.NUMPY_TYPE_TUPLE):
                return obj.tolist()
            elif six.PY3 and isinstance(obj, bytes):
                return obj.decode('utf-8')
            elif isinstance(obj, (set, frozenset)):
                return list(obj)
                # return json.JSONEncoder.default(self, list(obj))
                # return [json.JSONEncoder.default(o) for o in obj]
            elif isinstance(obj, util_type.PRIMATIVE_TYPES):
                return json.JSONEncoder.default(self, obj)
            elif  hasattr(obj, '__getstate__') and not isinstance(obj, uuid.UUID):
                try:
                    return obj.__getstate__()
                except TypeError:
                    need_else = True
            else:
                need_else = True

            if need_else:
                for type_, tag in type_to_tag.items():
                    if isinstance(obj, type_):
                        #print('----')
                        #print('encoder obj = %r' % (obj,))
                        #print('encoder type_ = %r' % (type_,))
                        func = encoders[tag]
                        text = func(obj)
                        return {tag: text}
                raise TypeError('Invalid serialization type=%r' % (type(obj)))

        @classmethod
        def _json_object_hook(cls, value, verbose=False, **kwargs):
            if len(value) == 1:
                tag, text = list(value.items())[0]
                if tag in decoders:
                    #print('----')
                    #print('decoder tag = %r' % (tag,))
                    func = decoders[tag]
                    obj = func(text)
                    #print('decoder obj = %r' % (obj,))
                    return obj
            else:
                return value
            return value
    return UtoolJSONEncoder


def to_json(val, allow_pickle=False, pretty=False):
    r"""
    Converts a python object to a JSON string using the utool convention

    Args:
        val (object):

    Returns:
        str: json_str

    References:
        http://stackoverflow.com/questions/11561932/why-does-json-dumpslistnp

    CommandLine:
        python -m utool.util_cache --test-to_json
        python3 -m utool.util_cache --test-to_json

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_cache import *  # NOQA
        >>> import utool as ut
        >>> import numpy as np
        >>> import uuid
        >>> val = [
        >>>     '{"foo": "not a dict"}',
        >>>     1.3,
        >>>     [1],
        >>>     # {1: 1, 2: 2, 3: 3}, cant use integer keys
        >>>     {1, 2, 3},
        >>>     slice(1, None, 1),
        >>>     b'an ascii string',
        >>>     np.array([1, 2, 3]),
        >>>     ut.get_zero_uuid(),
        >>>     ut.LazyDict(x='fo'),
        >>>     ut.LazyDict,
        >>>     {'x': {'a', 'b', 'cde'}, 'y': [1]}
        >>> ]
        >>> #val = ut.LazyDict(x='fo')
        >>> allow_pickle = True
        >>> if not allow_pickle:
        >>>     val = val[:-2]
        >>> json_str = ut.to_json(val, allow_pickle=allow_pickle)
        >>> result = ut.repr3(json_str)
        >>> reload_val = ut.from_json(json_str, allow_pickle=allow_pickle)
        >>> # Make sure pickle doesnt happen by default
        >>> try:
        >>>     json_str = ut.to_json(val)
        >>>     assert False or not allow_pickle, 'expected a type error'
        >>> except TypeError:
        >>>     print('Correctly got type error')
        >>> try:
        >>>     json_str = ut.from_json(val)
        >>>     assert False, 'expected a type error'
        >>> except TypeError:
        >>>     print('Correctly got type error')
        >>> print(result)
        >>> print('original = ' + ut.repr3(val, nl=1))
        >>> print('reconstructed = ' + ut.repr3(reload_val, nl=1))
        >>> assert reload_val[6] == val[6].tolist()
        >>> assert reload_val[6] is not val[6]

    Example:
        >>> # test 3.7 safe uuid
        >>> import uuid
        >>> import utool as ut
        >>> ut.to_json([uuid.uuid4()])

    """
    UtoolJSONEncoder = make_utool_json_encoder(allow_pickle)
    json_kw = {}
    json_kw['cls'] = UtoolJSONEncoder
    if pretty:
        json_kw['indent'] = 4
        json_kw['separators'] = (',', ': ')
    json_str = json.dumps(val, **json_kw)
    return json_str


def from_json(json_str, allow_pickle=False):
    """
    Decodes a JSON object specified in the utool convention

    Args:
        json_str (str):
        allow_pickle (bool): (default = False)

    Returns:
        object: val

    CommandLine:
        python -m utool.util_cache from_json --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_cache import *  # NOQA
        >>> import utool as ut
        >>> json_str = 'just a normal string'
        >>> json_str = '["just a normal string"]'
        >>> allow_pickle = False
        >>> val = from_json(json_str, allow_pickle)
        >>> result = ('val = %s' % (ut.repr2(val),))
        >>> print(result)
    """
    if six.PY3:
        if isinstance(json_str, bytes):
            json_str = json_str.decode('utf-8')
    UtoolJSONEncoder = make_utool_json_encoder(allow_pickle)
    object_hook = UtoolJSONEncoder._json_object_hook
    val = json.loads(json_str, object_hook=object_hook)
    return val


def get_func_result_cachekey(func_, args_=tuple(), kwargs_={}):
    """
    TODO: recursive partial definitions
    kwargs = {}
    args = ([],)
    """
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
        >>> import utool as ut
        >>> use_hash = None
        >>> func = consensed_cfgstr
        >>> args = ('a', 'b', 'c', 'd')
        >>> kwargs = {}
        >>> key_argx = [0, 1, 2]
        >>> key_kwds = []
        >>> kwdefaults = ut.util_inspect.get_kwdefaults(func)
        >>> argnames   = ut.util_inspect.get_argnames(func)
        >>> get_cfgstr_from_args(func, args, kwargs, key_argx, key_kwds, kwdefaults, argnames)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cache import *  # NOQA
        >>> import utool as ut
        >>> self = ut.LazyList
        >>> use_hash = None
        >>> func = self.append
        >>> args = ('a', 'b')
        >>> kwargs = {}
        >>> key_argx = [1]
        >>> key_kwds = []
        >>> kwdefaults = ut.util_inspect.get_kwdefaults(func)
        >>> argnames   = ut.util_inspect.get_argnames(func)
        >>> get_cfgstr_from_args(func, args, kwargs, key_argx, key_kwds, kwdefaults, argnames)
    """
    #try:
    #fmt_str = '%s(%s)'
    import utool as ut
    hashstr_ = util_hash.hashstr27
    if key_argx is None:
        key_argx = list(range(len(args)))
    if key_kwds is None:
        key_kwds = ut.unique_ordered(list(kwdefaults.keys()) + list(kwargs.keys()))

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
                key_kwds=None, use_cache=None, verbose=None):
    r"""
    Wraps a function with a Cacher object

    uses a hash of arguments as input

    Args:
        fname (str):  file name (defaults to function name)
        cache_dir (unicode): (default = u'default')
        appname (unicode): (default = u'utool')
        key_argx (None): (default = None)
        key_kwds (None): (default = None)
        use_cache (bool):  turns on disk based caching(default = None)

    CommandLine:
        python -m utool.util_cache --exec-cached_func

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> def costly_func(a, b, c='d', *args, **kwargs):
        ...     return ([a] * b, c, args, kwargs)
        >>> ans0 = costly_func(41, 3)
        >>> ans1 = costly_func(42, 3)
        >>> closure_ = ut.cached_func('costly_func', appname='utool_test',
        >>>                           key_argx=[0, 1])
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
    if verbose is None:
        verbose = VERBOSE_CACHE
    def cached_closure(func):
        # from utool import util_decor
        import utool as ut
        fname_ = util_inspect.get_funcname(func) if fname is None else fname
        kwdefaults = util_inspect.get_kwdefaults(func)
        argnames   = util_inspect.get_argnames(func)
        if ut.is_method(func):
            # ignore self for methods
            argnames = argnames[1:]
        cacher = Cacher(fname_, cache_dir=cache_dir, appname=appname,
                        verbose=verbose)
        if use_cache is None:
            use_cache_ = not util_arg.get_argflag('--nocache-' + fname_)
        else:
            use_cache_ = use_cache
        #_dbgdict = dict(fname_=fname_, key_kwds=key_kwds, appname=appname,
        #                key_argx=key_argx, use_cache_=use_cache_)
        #@functools.wraps(func)
        def cached_wraper(*args, **kwargs):
            """
            Cached Wrapper Function

            Additional Kwargs:
                use_cache (bool) : enables cache
            """
            try:
                if verbose > 2:
                    print('[util_cache] computing cached function fname_=%s' %
                          ( fname_,))
                # Implicitly adds use_cache to kwargs
                cfgstr = get_cfgstr_from_args(func, args, kwargs, key_argx,
                                              key_kwds, kwdefaults, argnames)
                if util_cplat.WIN32:
                    # remove potentially invalid chars
                    cfgstr = '_' + util_hash.hashstr27(cfgstr)
                assert cfgstr is not None, 'cfgstr=%r cannot be None' % (cfgstr,)
                use_cache__ = kwargs.pop('use_cache', use_cache_)
                if use_cache__:
                    # Make cfgstr from specified input
                    data = cacher.tryload(cfgstr)
                    if data is not None:
                        return data
                # Cached missed compute function
                data = func(*args, **kwargs)
                # Cache save
                #if use_cache__:
                # TODO: save_cache
                cacher.save(data, cfgstr)
                return data
            #except ValueError as ex:
            # handle protocal error
            except Exception as ex:
                from utool import util_dbg
                _dbgdict2 = dict(key_argx=key_argx, lenargs=len(args),
                                 lenkw=len(kwargs),)
                msg = '\n'.join([
                    '+--- UTOOL --- ERROR IN CACHED FUNCTION',
                    #'dbgdict = ' + utool.repr4(_dbgdict),
                    'dbgdict2 = ' + util_str.repr4(_dbgdict2),
                ])
                util_dbg.printex(ex, msg)
                raise
        # Give function a handle to the cacher object
        cached_wraper = util_decor.preserve_sig(cached_wraper, func)
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


def shelf_open(fpath):
    """
    allows for shelf to be used in with statements

    References:
        http://stackoverflow.com/questions/7489732/easiest-way-to-add-a-function-to-existing-class

    CommandLine:
        python -m utool.util_cache --test-shelf_open

    Example:
        >>> # DISABLE_DOCTEST
        >>> # UNSTABLE_DOCTEST
        >>> import utool as ut
        >>> fpath = ut.unixjoin(ut.ensure_app_resource_dir('utool'), 'testshelf.shelf')
        >>> with ut.shelf_open(fpath) as dict_:
        ...     print(ut.repr4(dict_))
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
            import dbm
            DBMError = dbm.error
        except Exception:
            DBMError = OSError

        try:
            shelf_fpath = get_global_shelf_fpath(self.appname, ensure=True)
            if VERBOSE:
                print('[cache] open: ' + shelf_fpath)
            self.shelf = shelve.open(shelf_fpath)
        except DBMError as ex:
            from utool import util_dbg
            util_dbg.printex(ex, 'Failed opening shelf_fpath due to bad version, remove and retry',
                             key_list=['shelf_fpath'])
            import utool as ut
            ut.delete(shelf_fpath)
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
        print(util_str.repr4(shelf))


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
    def get_cfgstr(self):
        return getattr(self, 'cfgstr', 'DEFAULT')
        # return 'DEFAULT'
        # raise NotImplementedError('abstract method')

    #@abc.abstractmethod
    def get_prefix(self):
        # import utool as ut
        return self.__class__.__name__ + '_'
        # return ut.get_funcname(self.__class__) + '_'
        # raise NotImplementedError('abstract method')

    def get_cachedir(self, cachedir=None):
        if cachedir is None:
            if hasattr(self, 'cachedir'):
                cachedir = self.cachedir
            else:
                cachedir = '.'
        return cachedir

    def get_fname(self, cfgstr=None, ext=None):
        # convinience
        return basename(self.get_fpath('', cfgstr=cfgstr, ext=ext))

    def get_fpath(self, cachedir=None, cfgstr=None, ext=None):
        """
        Ignore:
            fname = _fname
            cfgstr = _cfgstr
        """
        _dpath = self.get_cachedir(cachedir)
        _fname = self.get_prefix()
        _cfgstr = self.get_cfgstr() if cfgstr is None else cfgstr
        _ext =   self.ext if ext is None else ext
        fpath = _args2_fpath(_dpath, _fname, _cfgstr, _ext)
        return fpath

    def delete(self, cachedir=None, cfgstr=None, verbose=True or VERBOSE or util_arg.VERBOSE):
        """
        saves query result to directory
        """
        fpath = self.get_fpath(cachedir, cfgstr=cfgstr)
        if verbose:
            print('[Cachable] cache delete: %r' % (basename(fpath),))
        os.remove(fpath)

    @profile
    def save(self, cachedir=None, cfgstr=None, verbose=VERBOSE, quiet=QUIET,
             ignore_keys=None):
        """
        saves query result to directory
        """
        fpath = self.get_fpath(cachedir, cfgstr=cfgstr)
        if verbose:
            print('[Cachable] cache save: %r' % (basename(fpath),))

        if hasattr(self, '__getstate__'):
            statedict = self.__getstate__()
        else:
            statedict = self.__dict__

        if ignore_keys is None:
            save_dict = statedict
        else:
            save_dict = {key: val
                         for (key, val) in six.iteritems(statedict)
                         if key not in ignore_keys}

        util_io.save_data(fpath, save_dict)
        return fpath
        #save_cache(cachedir, '', cfgstr, self.__dict__)
        #with open(fpath, 'wb') as file_:
        #    pickle.dump(self.__dict__, file_)

    def _unsafe_load(self, fpath, ignore_keys=None):
        loaded_dict = util_io.load_data(fpath)
        if ignore_keys is not None:
            for key in ignore_keys:
                if key in loaded_dict:
                    del loaded_dict[key]
        if hasattr(self, '__setstate__'):
            self.__setstate__(loaded_dict)
        else:
            self.__dict__.update(loaded_dict)
        #with open(fpath, 'rb') as file_:
        #    loaded_dict = pickle.load(file_)
        #    self.__dict__.update(loaded_dict)

    def glob_valid_targets(self, cachedir=None, partial_cfgstr=''):
        from utool import util_path
        prefix = self.get_prefix()
        pattern = prefix + '*' + partial_cfgstr + '*' + self.ext
        cachedir = self.get_cachedir(cachedir)
        valid_targets = util_path.glob(cachedir, pattern, recursive=False)
        return valid_targets

    def fuzzyload(self, cachedir=None, partial_cfgstr='', **kwargs):
        """
        Try and load from a partially specified configuration string
        """
        valid_targets = self.glob_valid_targets(cachedir, partial_cfgstr)
        if len(valid_targets) != 1:
            import utool as ut
            msg = 'need to further specify target. valid_targets=%s' % (ut.repr3(valid_targets,))
            raise ValueError(msg)
        fpath = valid_targets[0]
        self.load(fpath=fpath, **kwargs)

    @profile
    def load(self, cachedir=None, cfgstr=None, fpath=None, verbose=None,
             quiet=QUIET, ignore_keys=None):
        """
        Loads the result from the given database
        """
        if verbose is None:
            verbose = getattr(self, 'verbose', VERBOSE)
        if fpath is None:
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
        #except BadZipFile as ex:
        except zipfile.error as ex:
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
        >>> # DISABLE_DOCTEST
        >>> # UNSTABLE_DOCTEST
        >>> from utool.util_cache import *  # NOQA
        >>> import utool as ut  # NOQA
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
        >>> result = ut.repr2(dict(cache_obj), nl=False)
        >>> print(result)
        {2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    """
    USE_C_LRU = False
    if USE_C_LRU:
        import lru
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
        return ut.repr4(self._cache, nl=False)

    def __repr__(self):
        import utool as ut
        return 'LRUDict(' + ut.repr4(self._cache) + ')'
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
    import utool as ut
    import simplejson as json
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
            pickle.dump(dict_, outfile)

    def cPickle_read_test():
        with open(cpkl_path, 'rb') as outfile:
            test = {key: val for key, val in six.iteritems(pickle.load(outfile))}
        assert len(test) > 0

    def cPickle_read_test2():
        with open(cpkl_path, 'rb') as outfile:
            test = pickle.load(outfile)
        assert len(test) > 0

    shelf_write_test()
    shelf_read_test()
    #json_write_test()
    #json_read_test()
    cPickle_write_test()
    cPickle_read_test()
    cPickle_read_test2()


class KeyedDefaultDict(util_dict.DictLike):
    def __init__(self, default_func, *args, **kwargs):
        self._default_func = default_func
        self._args = args
        self._kwargs = kwargs
        self._internal = {}

    def setitem(self, key, value):
        self._internal[key] = value

    def getitem(self, key):
        if key not in self._internal:
            value = self._default_func(key, *self._args, **self._kwargs)
            self._internal[key] = value
        return self._internal[key]

    def keys(self):
        return self._internal.keys()

    def values(self):
        return self._internal.values()


# @six.add_metaclass(util_class.ReloadingMetaclass)
@util_class.reloadable_class
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
    def __init__(self, other=None, is_eager=True, verbose=False, reprkw=None,
                 mutable=False,
                 **kwargs):
        # Registered lazy evaluations
        self._eval_funcs = {}
        # Computed results
        self._stored_results = {}
        self.infer_lazy_vals_hack = True
        self._is_eager = is_eager
        self._verbose = verbose
        self.reprkw = dict(is_eager=False, nl=False)
        self._mutable = mutable
        if reprkw is not None:
            self.reprkw.update(**reprkw)
        if other is not None:
            self.update(other)
        if len(kwargs) > 0:
            self.update(kwargs)

    # --- direct interface

    def set_lazy_func(self, key, func):
        assert util_type.is_funclike(func), 'func must be a callable'
        #if key in self._stored_results:
        #    raise ValueError(
        #        ('Cannot add new lazy function for key=%r'
        #         'that has been computed') % (key,))
        #if key in self._stored_results:
        if not self._mutable and key in self.reconstructable_keys():
            raise ValueError(
                ('Cannot overwrite lazy function for key=%r') % (key,))
        self._eval_funcs[key] = func

    def setitem(self, key, value):
        # HACK, lazy funcs should all be registered
        # this should should always just set a value
        if not self._mutable and key in self.reconstructable_keys():
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
        func_ = self._eval_funcs[key]
        value = func_()
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

    def cached_keys(self):
        """ only keys whose vals that have been explicitly set without a backup func """
        return set(self.nonreconstructable_keys()).union(set(self.evaluated_keys()))

    def printinfo(self):
        print('nonreconstructable_keys = %s' % (self.nonreconstructable_keys(),))
        print('reconstructable_keys = %s' % (self.reconstructable_keys(),))
        print('evaluated_keys = %s' % (self.evaluated_keys(),))
        print('unevaluated_keys = %s' % (self.unevaluated_keys(),))

    def asdict(self, is_eager=None):
        dict_ = {key: self.getitem(key, is_eager) for key in self.keys()}
        return dict_

    def tostring(self, is_eager=None, keys=None, **kwargs):
        import utool as ut
        dict_ = self.asdict(is_eager=is_eager)
        class AwakeFaceRepr(object):
            def __repr__(self):
                return '!'
                # return '(o.o)'
                # return "٩(ˊᗜˋ*)و"
        class SleepFaceRepr(object):
            def __repr__(self):
                return 'z'
                # return '(-_-)'
                # return '(ᵕ≀ᵕ)'
        for key in self.evaluated_keys():
            # dict_[key] = '!'
            dict_[key] = AwakeFaceRepr()
        for key in self.unevaluated_keys():
            # dict_[key] = 'z'
            dict_[key] = SleepFaceRepr()
        if keys is not None:
            dict_ = ut.dict_subset(dict_, keys)
        return ut.repr2(dict_, **kwargs)

    # --- dict interface

    def get(self, key, *d):
        if len(d) > 1:
            raise ValueError('can only specify one default')
        elif len(d) == 1:
            #assert len(d) == 0, 'no support for default yet'
            if key not in self:
                return d[0]
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

    def __setitem__(self, key, value):
        self.setitem(key, value)

    def __getitem__(self, key):
        return self.get(key)

    def __delitem__(self, key):
        if key not in self.keys():
            raise KeyError(key)
        if key in self._eval_funcs:
            del self._eval_funcs[key]
        if key in self._stored_results:
            del self._stored_results[key]

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
            #raise ValueError('index=%r out of bounds' % (index,))
            raise ValueError(
                'index=%r out of bounds or error computing lazy value.' % (
                    index,))

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
