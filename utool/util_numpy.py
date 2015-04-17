from __future__ import absolute_import, division, print_function
try:
    import numpy as np
except ImportError as ex:
    pass
from utool import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[util_numpy]')


def tiled_range(range_, cols):
    return np.tile(np.arange(range_), (cols, 1)).T
    #np.tile(np.arange(num_qf).reshape(num_qf, 1), (1, k_vsmany))


def random_indexes(max_index, subset_size):
    subst_ = np.arange(0, max_index)
    np.random.shuffle(subst_)
    subst = subst_[0:min(subset_size, max_index)]
    return subst


#def list_index(search_list, to_find_list):
#    """ Keep this function
#    Searches search_list for each element in to_find_list"""
#    try:
#        toret = [np.where(search_list == item)[0][0] for item in to_find_list]
#    except IndexError as ex1:
#        print('ERROR: ' + str(ex1))
#        print('item = %r' % (item,))
#        raise
#    return toret


def npfind(arr):
    found = np.where(arr)[0]
    pos = -1 if len(found) == 0 else found[0]
    return pos


def index_of(item, array):
    'index of [item] in [array]'
    return np.where(array == item)[0][0]


def spaced_indexes(len_, n, trunc=False):
    """
    Returns n evenly spaced indexes.
    Returns as many as possible if trunc is true
    """

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


def intersect2d(A, B):
    """
    intersect2d

    intersect rows of 2d numpy arrays

    DEPRICATE: use intersect2d in vtool instead

    Args:
        A (ndarray[ndim=2]):
        B (ndarray[ndim=2]):

    Returns:
        tuple: (C, Ax, Bx)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_numpy import *  # NOQA
        >>> import utool as ut
        >>> A = np.array([[1, 2, 3], [1, 1, 1]])
        >>> B = np.array([[1, 2, 3], [1, 2, 14]])
        >>> (C, Ax, Bx) = ut.intersect2d(A, B)
        >>> result = str((C, Ax, Bx))
        >>> print(result)
        (array([[1, 2, 3]]), array([0]), array([0]))
    """
    Cset  =  set(tuple(x) for x in A).intersection(set(tuple(x) for x in B))
    Ax = np.array([x for x, item in enumerate(A) if tuple(item) in Cset], dtype=np.int)
    Bx = np.array([x for x, item in enumerate(B) if tuple(item) in Cset], dtype=np.int)
    C = np.array(tuple(Cset))
    return C, Ax, Bx


#def unique_keep_order(arr):
    #""" pandas.unique preseves order and seems to be faster due to index overhead """
    #import pandas as pd
    #return pd.unique(arr)
    #_, idx = np.unique(arr, return_index=True)
    #return arr[np.sort(idx)]


def deterministic_shuffle(list_, seed=1):
    r"""
    Args:
        list_ (list):
        seed (int):

    Returns:
        list: list_

    CommandLine:
        python -m utool.util_numpy --test-deterministic_shuffle

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_numpy import *  # NOQA
        >>> list_ = [1,2,3,4,5,6]
        >>> seed = 1
        >>> list_ = deterministic_shuffle(list_, seed)
        >>> result = str(list_)
        >>> print(result)
        [4, 6, 1, 3, 2, 5]
    """
    rand_seed = np.uint32(np.random.rand() * np.uint(0 - 2) / 2)
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
    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> min_ = 10
        >>> max_ = 1000
        >>> nSamp  = 7
        >>> result = utool.sample_domain(min_, max_, nSamp)
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


def make_incrementer():
    def incrementer(_mem=[0]):
        _mem[0] += 1
        return _mem[0]
    return incrementer


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_numpy; utool.doctest_funcs(utool.util_numpy, allexamples=True)"
        python -c "import utool, utool.util_numpy; utool.doctest_funcs(utool.util_numpy)"
        python -m utool.util_numpy
        python -m utool.util_numpy --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
