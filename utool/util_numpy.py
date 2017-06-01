# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
import six
import itertools
try:
    import numpy as np
except ImportError as ex:
    pass
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)


def tiled_range(range_, cols):
    return np.tile(np.arange(range_), (cols, 1)).T
    #np.tile(np.arange(num_qf).reshape(num_qf, 1), (1, k_vsmany))


def quantum_random():
    """ returns a 32 bit unsigned integer quantum random number """
    import quantumrandom
    data16 = quantumrandom.uint16(array_length=2)
    assert data16.flags['C_CONTIGUOUS']
    data32 = data16.view(np.dtype('uint32'))[0]
    return data32


def ensure_rng(rng, impl='numpy'):
    """
    Returns a random number generator

    np_rng = np.random.RandomState(seed=0)
    py_rng = random.Random(0)

    for i in range(10):
        np_rng.rand()
        npstate = np_rng.get_state()
        print([npstate[0], npstate[1][[0, 1, 2, -2, -1]], npstate[2], npstate[3], npstate[4]])

    for i in range(10):
        py_rng.random()
        pystate = py_rng.getstate()
        print([pystate[0], pystate[1][0:3] + pystate[1][-2:], pystate[2]])
    """
    import random
    if impl == 'numpy':
        if rng is None:
            rng = np.random
        elif isinstance(rng, int):
            rng = np.random.RandomState(seed=rng)
        elif isinstance(rng, random.Random):
            py_rng = rng
            # Convert python to numpy random state (incomplete)
            py_state = py_rng.getstate()
            np_rng = np.random.RandomState(seed=0)
            np_state = np_rng.get_state()
            new_np_state = (
                np_state[0],
                np.array(py_state[1][0:-1], dtype=np.uint32),
                np_state[2], np_state[3], np_state[4])
            np_rng.set_state(new_np_state)
            rng = np_rng
    else:
        if rng is None:
            rng = random
        elif isinstance(rng, int):
            rng = random.Random(rng)
        elif isinstance(rng, np.random.RandomState):
            np_rng = rng
            # Convert numpy to python random state (incomplete)
            np_state = np_rng.get_state()
            py_rng = random.Random(0)
            py_state = py_rng.getstate()
            new_py_state = (
                py_state[0], tuple(np_state[1].tolist() + [len(np_state[1])]),
                py_state[1]
            )
            py_rng.setstate(new_py_state)
            rng = py_rng
            # seed = rng.randint(sys.maxsize)
            # assert False
    return rng


def random_indexes(max_index, subset_size=None, seed=None, rng=None):
    """ random unrepeated indicies

    Args:
        max_index (?):
        subset_size (None): (default = None)
        seed (None): (default = None)
        rng (RandomState):  random number generator(default = None)

    Returns:
        ?: subst

    CommandLine:
        python -m utool.util_numpy --exec-random_indexes

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_numpy import *  # NOQA
        >>> max_index = 10
        >>> subset_size = None
        >>> seed = None
        >>> rng = np.random.RandomState(0)
        >>> subst = random_indexes(max_index, subset_size, seed, rng)
        >>> result = ('subst = %s' % (str(subst),))
        >>> print(result)
    """
    subst_ = np.arange(0, max_index)
    rng = ensure_rng(seed if rng is None else rng)
    rng.shuffle(subst_)
    if subset_size is None:
        subst = subst_
    else:
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


#def unique_ordered(arr):
    #""" pandas.unique preseves order and seems to be faster due to index overhead """
    #import pandas as pd
    #return pd.unique(arr)
    #_, idx = np.unique(arr, return_index=True)
    #return arr[np.sort(idx)]


def deterministic_shuffle(list_, seed=0, rng=None):
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
        >>> list_ = [1, 2, 3, 4, 5, 6]
        >>> seed = 1
        >>> list_ = deterministic_shuffle(list_, seed)
        >>> result = str(list_)
        >>> print(result)
        [3, 2, 5, 1, 4, 6]
    """
    rng = ensure_rng(seed if rng is None else rng)
    rng.shuffle(list_)
    return list_


def shuffle(list_, seed=0, rng=None):
    r"""
    Shuffles a list inplace and then returns it for convinience

    Args:
        list_ (list or ndarray): list to shuffl
        rng (RandomState or int): seed or random number gen

    Returns:
        list: this is the input, but returned for convinience

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_numpy import *  # NOQA
        >>> list1 = [1, 2, 3, 4, 5, 6]
        >>> list2 = shuffle(list(list1), rng=1)
        >>> assert list1 != list2
        >>> result = str(list2)
        >>> print(result)
        [3, 2, 5, 1, 4, 6]
    """
    rng = ensure_rng(seed if rng is None else rng)
    rng.shuffle(list_)
    return list_


def random_sample(list_, nSample, strict=False, rng=None, seed=None):
    """
    Grabs data randomly

    Args:
        list_ (list):
        nSample (?):
        strict (bool): (default = False)
        rng (module):  random number generator(default = numpy.random)
        seed (None): (default = None)

    Returns:
        list: sample_list

    CommandLine:
        python -m utool.util_numpy --exec-random_sample

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_numpy import *  # NOQA
        >>> list_ = np.arange(10)
        >>> nSample = 4
        >>> strict = False
        >>> rng = np.random.RandomState(0)
        >>> seed = None
        >>> sample_list = random_sample(list_, nSample, strict, rng, seed)
        >>> result = ('sample_list = %s' % (str(sample_list),))
        >>> print(result)
    """
    rng = ensure_rng(seed if rng is None else rng)
    if isinstance(list_, list):
        list2_ = list_[:]
    else:
        list2_ = np.copy(list_)
    if len(list2_) == 0 and not strict:
        return list2_
    rng.shuffle(list2_)
    if nSample is None and strict is False:
        return list2_
    if not strict:
        nSample = min(max(0, nSample), len(list2_))
    sample_list = list2_[:nSample]
    return sample_list


def deterministic_sample(list_, nSample, seed=0, rng=None, strict=False):
    """ Grabs data randomly, but in a repeatable way """
    rng = ensure_rng(seed if rng is None else rng)
    sample_list = random_sample(list_, nSample, strict=strict, rng=rng)
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
    # DEPRICATE FOR ITERTOOLS.COUNT
    return ut.partial(six.next, itertools.count(1))
    # def incrementer(_mem=[0]):
    #     _mem[0] += 1
    #     return _mem[0]
    # return incrementer


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
