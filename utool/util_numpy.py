from __future__ import absolute_import, division, print_function
import numpy as np
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[numpy]')


def tiled_range(range_, cols):
    return np.tile(np.arange(range_), (cols, 1)).T
    #np.tile(np.arange(num_qf).reshape(num_qf, 1), (1, k_vsmany))


def random_indexes(max_index, subset_size):
    subst_ = np.arange(0, max_index)
    np.random.shuffle(subst_)
    subst = subst_[0:min(subset_size, max_index)]
    return subst


def list_index(search_list, to_find_list):
    """ Keep this function
    Searches search_list for each element in to_find_list"""
    try:
        toret = [np.where(search_list == item)[0][0] for item in to_find_list]
    except IndexError as ex1:
        print('ERROR: ' + str(ex1))
        print('item = %r' % (item,))
        raise
    return toret


def npfind(arr):
    found = np.where(arr)[0]
    pos = -1 if len(found) == 0 else found[0]
    return pos


def index_of(item, array):
    'index of [item] in [array]'
    return np.where(array == item)[0][0]


def spaced_indexes(len_, n, trunc=False):
    """ Returns n evenly spaced indexes.
        Returns as many as possible if trunc is true
    </CYTH> """

    if n is None:
        return np.arange(len_)
    all_indexes = np.arange(len_)
    if trunc:
        n = min(len_, n)
    if n == 0:
        return np.empty(0)
    stride = len_ // n
    try:
        indexes = all_indexes[0:-1:stride]
    except ValueError:
        raise ValueError('cannot slice list of len_=%r into n=%r parts' % (len_, n))
    return indexes


def inbounds(arr, low, high):
    flag_low = arr >= low
    flag_high = arr < high if high is not None else flag_low
    flag = np.logical_and(flag_low, flag_high)
    return flag


def intersect2d_numpy(A, B):
    #http://stackoverflow.com/questions/8317022/
    #get-intersecting-rows-across-two-2d-numpy-arrays/8317155#8317155
    # TODO: MOVE to numpy libs
    nrows, ncols = A.shape
    # HACK to get consistent dtypes
    assert A.dtype is B.dtype, 'A and B must have the same dtypes'
    dtype = np.dtype([('f%d' % i, A.dtype) for i in range(ncols)])
    try:
        C = np.intersect1d(A.view(dtype), B.view(dtype))
    except ValueError:
        C = np.intersect1d(A.copy().view(dtype), B.copy().view(dtype))
    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C


def intersect2d(A, B):
    # TODO: MOVE to numpy libs
    Cset  =  set(tuple(x) for x in A).intersection(set(tuple(x) for x in B))
    Ax = np.array([x for x, item in enumerate(A) if tuple(item) in Cset], dtype=np.int)
    Bx = np.array([x for x, item in enumerate(B) if tuple(item) in Cset], dtype=np.int)
    C = np.array(tuple(Cset))
    return C, Ax, Bx


#def unique_keep_order(arr):
    #""" pandas.unique preseves order and seems to be faster due to index overhead """
    # TODO: MOVE to numpy libs
    #import pandas as pd
    #return pd.unique(arr)
    #_, idx = np.unique(arr, return_index=True)
    #return arr[np.sort(idx)]


def deterministic_shuffle(list_, seed=1):
    rand_seed = int(np.random.rand() * np.uint(0 - 2) / 2)
    if not isinstance(list_, (np.ndarray, list)):
        list_ = list(list_)
    seed_ = len(list_) + seed
    np.random.seed(seed_)
    np.random.shuffle(list_)
    np.random.seed(rand_seed)  # reseed
    return list_


def listlike_copy(list_):
    if isinstance(list_, list):
        list2_ = list_[:]
    else:
        list2_ = np.copy(list_)
    return list2_


def random_sample(list_, nSample, strict=False):
    """ Grabs data randomly, but in a repeatable way """
    list2_ = listlike_copy(list_)
    np.random.shuffle(list2_)
    if nSample is None and strict is False:
        return list2_
    if not strict:
        nSample = min(nSample, len(list2_))
    sample_list = list2_[:nSample]
    return sample_list


def deterministic_sample(list_, nSample, seed=1, strict=False):
    """ Grabs data randomly, but in a repeatable way """
    list2_ = listlike_copy(list_)
    deterministic_shuffle(list2_, seed)
    if nSample is None and strict is False:
        return list2_
    if not strict:
        nSample = min(nSample, len(list2_))
    sample_list = list2_[:nSample]
    return sample_list


def spaced_items(list_, n, **kwargs):
    """ Returns n evenly spaced items """
    indexes = spaced_indexes(len(list_), n, **kwargs)
    items = list_[indexes]
    return items


def sample_domain(min_, max_, nSamp, mode='linear'):
    """
    >>> import utool
    >>> min_ = 10
    >>> max_ = 1000
    >>> nSamp  = 7
    >>> utool.sample_domain(min_, max_, nSamp)
    [10, 151, 293, 434, 576, 717, 859]
    """
    if mode == 'linear':
        samples_ = np.rint(np.linspace(min_, max_, nSamp + 1)).astype(np.int64)
    elif mode == 'log':
        base = 2
        logmin = np.log2(min_) / np.log2(base)
        logmax = np.log2(max_) / np.log2(base)
        samples_ = np.rint(np.logspace(logmin, logmax, nSamp + 1, base=base)).astype(np.int64)
    else:
        raise NotImplementedError(mode)
    sample = [index for index in samples_ if index < max_]
    return sample
