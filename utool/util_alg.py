# Licence:
#
# TODO: Rename
# util_science?
#
from __future__ import absolute_import, division, print_function
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # TODO remove numpy
    pass
from collections import defaultdict
import operator
import six
from six.moves import zip, range, reduce  # NOQA
from utool import util_type
from utool import util_inject
import scipy.spatial.distance as spdist
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[alg]')


PHI = 1.61803398875
PHI_A = (1 / PHI)
PHI_B = 1 - PHI_A


def greedy_max_inden_setcover(candidate_sets_dict, items, max_covers=None):
    """
    greedy algorithm for maximum independent set cover

    Covers items with sets from candidate sets. Could be made faster.

    CommandLine:
        python -m utool.util_alg --test-greedy_max_inden_setcover

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> candidate_sets_dict = {'a': [5, 3], 'b': [2, 3, 5],
        ...                        'c': [4, 8], 'd': [7, 6, 2, 1]}
        >>> items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> max_covers = None
        >>> tup = greedy_max_inden_setcover(candidate_sets_dict, items, max_covers)
        >>> (uncovered_items, covered_items_list, accepted_keys) = tup
        >>> result = str((uncovered_items, accepted_keys))
        >>> print(result)
        ([0, 9], set(['a', 'c', 'd']))

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> candidate_sets_dict = {'a': [5, 3], 'b': [2, 3, 5],
        ...                        'c': [4, 8], 'd': [7, 6, 2, 1]}
        >>> items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> max_covers = 1
        >>> tup = greedy_max_inden_setcover(candidate_sets_dict, items, max_covers)
        >>> (uncovered_items, covered_items_list, accepted_keys) = tup
        >>> result = str((uncovered_items, accepted_keys))
        >>> print(result)
        ([0, 3, 4, 5, 8, 9], set(['d']))
    """
    uncovered_set = set(items)
    rejected_keys = set()
    accepted_keys = set()
    covered_items_list = []
    while True:
        # Break if we have enough covers
        if max_covers is not None and len(covered_items_list) >= max_covers:
            break
        maxkey = None
        maxlen = -1
        # Loop over candidates to find the biggested unadded cover set
        for key, candidate_items in six.iteritems(candidate_sets_dict):
            if key in rejected_keys or key in accepted_keys:
                continue
            #print('Checking %r' % (key,))
            lenval = len(candidate_items)
            # len(uncovered_set.intersection(candidate_items)) == lenval:
            if uncovered_set.issuperset(candidate_items):
                if lenval > maxlen:
                    maxkey = key
                    maxlen = lenval
            else:
                rejected_keys.add(key)
        # Add the set to the cover
        if maxkey is None:
            break
        maxval = candidate_sets_dict[maxkey]
        accepted_keys.add(maxkey)
        covered_items_list.append(list(maxval))
        # Add values in this key to the cover
        uncovered_set.difference_update(maxval)
    uncovered_items = list(uncovered_set)
    covertup = uncovered_items, covered_items_list, accepted_keys
    return covertup


def bayes_rule(b_given_a, prob_a, prob_b):
    r"""
    bayes_rule

    P(A | B) = \frac{ P(B | A) P(A) }{ P(B) }

    Args:
        b_given_a (ndarray or float):
        prob_a (ndarray or float):
        prob_b (ndarray or float):

    Returns:
        ndarray or float: a_given_b

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> b_given_a = .1
        >>> prob_a = .3
        >>> prob_b = .4
        >>> a_given_b = bayes_rule(b_given_a, prob_a, prob_b)
        >>> result = a_given_b
        >>> print(result)
        0.075

    """
    a_given_b = (b_given_a * prob_a) / prob_b
    return a_given_b


def estimate_pdf(data, gridsize=1024, adjust=1):
    """
    estimate_pdf

    References;
        http://statsmodels.sourceforge.net/devel/generated/statsmodels.nonparametric.kde.KDEUnivariate.html
        https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

    Args:
        data (?):
        bw_factor (?):

    Returns:
        ?: data_pdf

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import plottool as pt
        >>> data = '?'
        >>> bw_factor = '?'
        >>> data_pdf = estimate_pdf(data, bw_factor)
        >>> pt.plot(data_pdf.cdf)
        >>> print(data_pdf)
    """
    import utool as ut
    #import scipy.stats as spstats
    #import statsmodels
    import numpy as np
    import statsmodels.nonparametric.kde
    try:
        data_pdf = statsmodels.nonparametric.kde.KDEUnivariate(data)
        bw_choices = ['scott', 'silverman', 'normal_reference']
        bw = bw_choices[1]
        fitkw = dict(kernel='gau',
                     bw=bw,
                     fft=True,
                     weights=None,
                     adjust=adjust,
                     cut=3,
                     gridsize=gridsize,
                     clip=(-np.inf, np.inf),)
        data_pdf.fit(**fitkw)
        #density = data_pdf.density
    #try:
    #    data_pdf = spstats.gaussian_kde(data, bw_factor)
    #    data_pdf.covariance_factor = bw_factor
    except Exception as ex:
        ut.printex(ex, '! Exception while estimating kernel density',
                   keys=['data'])
        raise
    return data_pdf


def normalize(array, dim=0):
    return norm_zero_one(array, dim)


def norm_zero_one(array, dim=None):
    """
    normalizes a numpy array from 0 to 1 based in its extent

    Args:
        array (ndarray):
        dim   (int):

    Returns:
        ndarray:

    CommandLine:
        python -m utool.util_alg --test-norm_zero_one

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> array = np.array([ 22, 1, 3, 2, 10, 42, ])
        >>> dim = None
        >>> array_norm = norm_zero_one(array, dim)
        >>> result = np.array_str(array_norm, precision=3)
        >>> print(result)
        [ 0.512  0.     0.049  0.024  0.22   1.   ]
    """
    if not util_type.is_float(array):
        array = array.astype(np.float32)
    array_max  = array.max(dim)
    array_min  = array.min(dim)
    array_exnt = np.subtract(array_max, array_min)
    array_norm = np.divide(np.subtract(array, array_min), array_exnt)
    return array_norm


def find_std_inliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)


def choose(n, k):
    import scipy.misc
    return scipy.misc.comb(n, k, True)


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Args:
        arrays (list of array-like): 1-D arrays to form the cartesian product of
        out (ndarray): Outvar which is modified in place if specified

    Returns:
        out (ndarray): cartesian products formed of input arrays
            2-D array of shape (M, len(arrays))

    References:
        https://gist.github.com/hernamesbarbara/68d073f551565de02ac5

    Example:
        >>> # ENABLE_DOCTEST
        >>> arrays = ([1, 2, 3], [4, 5], [6, 7])
        >>> out = cartesian(arrays)
        >>> result = repr(out.T)
        array([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
               [4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5],
               [6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def euclidean_dist(vecs1, vec2, dtype=None):
    if dtype is None:
        dtype = np.float32
    return np.sqrt(((vecs1.astype(dtype) - vec2.astype(dtype)) ** 2).sum(1))


def negative_minclamp_inplace(arr):
    arr[arr > 0] -= arr[arr > 0].min()
    arr[arr <= 0] = arr[arr > 0].min()
    return arr


def xywh_to_tlbr(bbox, img_wh):
    """ converts xywh format to (tlx, tly, blx, bly) """
    (img_w, img_h) = img_wh
    if img_w == 0 or img_h == 0:
        img_w = 1
        img_h = 1
        msg = '[cc2.1] Your csv tables have an invalid ANNOTATION.'
        print(msg)
        #warnings.warn(msg)
        #ht = 1
        #wt = 1
    # Ensure ANNOTATION is within bounds
    (x, y, w, h) = bbox
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, img_w - 1)
    y2 = min(y + h, img_h - 1)
    return (x1, y1, x2, y2)


def build_reverse_mapping(uid_list, cluster_list):
    """
    Given a list of ids (uid_list) and a corresponding cluster index list
    (cluster_list), this builds a mapping from cluster index to uids
    """
    # Sort by clusterid for cache efficiency
    sortx = cluster_list.argsort()
    cluster_list = cluster_list[sortx]
    uid_list = uid_list[sortx]
    # Initialize dict of lists
    cluster2_uids = defaultdict(list)
    for uid, cluster in zip(uid_list, cluster_list):
        cluster2_uids[cluster].append(uid)
    return cluster2_uids


def group_items(item_list, groupid_list):
    """
    group_items

    Args:
        item_list (list):
        groupid_list (list):

    Returns:
        dict: groupid2_items mapping groupids to a list of items

    SeeAlso:
        vtool.group_indices - much faster numpy grouping algorithm
        vtool.apply_gropuing - second part to faster numpy grouping algorithm

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> item_list    = [ 'ham',      'jam',    'spam',     'eggs', 'cheese', 'bannana']
        >>> groupid_list = ['protein', 'fruit', 'protein',  'protein',  'dairy',   'fruit']
        >>> groupid2_items = ut.group_items(item_list, groupid_list)
        >>> result = ut.dict_str(groupid2_items, newlines=False, strvals=False)
        >>> print(result)
        {'protein': ['eggs', 'ham', 'spam'], 'fruit': ['bannana', 'jam'], 'dairy': ['cheese'],}
    """
    # Sort by groupid for cache efficiency
    sorted_pairs = sorted(list(zip(groupid_list, item_list)))
    # Initialize dict of lists
    groupid2_items = defaultdict(list)
    # Insert each item into the correct group
    for groupid, item in sorted_pairs:
        groupid2_items[groupid].append(item)
    return groupid2_items


def search_utool(pat):
    import utool as ut
    found_list = [name for name in dir(ut) if name.find(pat) >= 0]
    return found_list


def item_hist(list_):
    """ counts the number of times each item appears in the dictionary """
    dict_hist = {}
    # Insert each item into the correct group
    for item in list_:
        if item not in dict_hist:
            dict_hist[item] = 0
        dict_hist[item] += 1
    return dict_hist


def unpack_items_sorted(dict_, sortfn, reverse=True):
    """ Unpacks and sorts the dictionary by sortfn
    """
    items = dict_.items()
    sorted_items = sorted(items, key=sortfn, reverse=reverse)
    sorted_keys, sorted_vals = list(zip(*sorted_items))
    return sorted_keys, sorted_vals


def unpack_items_sorted_by_lenvalue(dict_, reverse=True):
    """ Unpacks and sorts the dictionary by key
    """
    def sort_lenvalue(item):
        return len(item[1])
    return unpack_items_sorted(dict_, sort_lenvalue)


def unpack_items_sorted_by_value(dict_, reverse=True):
    """ Unpacks and sorts the dictionary by key
    """
    def sort_value(item):
        return item[1]
    return unpack_items_sorted(dict_, sort_value)


def flatten_membership_mapping(uid_list, members_list):
    num_members = sum(list(map(len, members_list)))
    flat_uids = [None for _ in range(num_members)]
    flat_members = [None for _ in range(num_members)]
    count = 0
    for uid, members in zip(uid_list, members_list):
        for member in members:
            flat_uids[count]    = uid
            flat_members[count] = member
            count += 1
    return flat_uids, flat_members


def void_rowview_numpy(arr):
    """ returns view of nparray where each row is a single item
    """
    void_dtype = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    arr_void_view = np.ascontiguousarray(arr).view(void_dtype)
    return arr_void_view


def unique_row_indexes(arr):
    """ np.unique on rows """
    arr_void_view = void_rowview_numpy(arr)
    _, unique_rowx = np.unique(arr_void_view, return_index=True)
    # cast back to original dtype
    unique_rowx.sort()
    return unique_rowx


def get_phi():
    """ Golden Ratio: phi = 1 / sqrt(5) / 2.0 = 1.61803398875 """
    #phi = (1.0 + np.sqrt(5)) / 2.0 = 1.61803398875
    # return phi
    return PHI


def get_phi_ratio1():
    return 1.0 / get_phi()


def is_prime(num):
    """
    References:
        http://thelivingpearl.com/2013/01/06/how-to-find-prime-numbers-in-python/
    """
    for j in range(2, num):
        if (num % j) == 0:
            return False
    return True


def deg_to_rad(degree):
    degree %= 360.0
    tau = 2 * np.pi
    return (degree / 360.0) * tau


def rad_to_deg(radians):
    tau = 2 * np.pi
    radians %= tau
    return (radians / tau) * 360.0


def enumerate_primes(max_prime=4100):
    primes = [num for num in range(2, max_prime) if is_prime(num)]
    return primes


def get_nth_prime(n, max_prime=4100, safe=True):
    """ hacky but still brute force algorithm for finding nth prime for small tests """
    if n <= 100:
        first_100_primes = (
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
            67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
            139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
            211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277,
            281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359,
            367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439,
            443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521,
            523, 541, )
        print(len(first_100_primes))
        nth_prime = first_100_primes[n - 1]
    else:
        if safe:
            primes = [num for num in range(2, max_prime) if is_prime(num)]
            nth_prime = primes[n]
        else:
            # This can run for a while... get it? while?
            nth_prime = get_nth_prime_bruteforce(n)
    return nth_prime


def get_nth_prime_bruteforce(n):
    num = 2
    num_primes_found = 0
    while True:
        if is_prime(num):
            num_primes_found += 1
        if num_primes_found == n:
            nth_prime = num
            break
        num += 1
    return nth_prime


def inbounds(num, low, high, eq=False):
    r"""
    Args:
        num (scalar or ndarray):
        low (scalar or ndarray):
        high (scalar or ndarray):
        eq (bool):

    Returns:
        scalar or ndarray: is_inbounds

    CommandLine:
        python -m utool.util_alg --test-inbounds

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> # build test data
        >>> num = np.array([[ 0.   ,  0.431,  0.279],
        ...                 [ 0.204,  0.352,  0.08 ],
        ...                 [ 0.107,  0.325,  0.179]])
        >>> low  = .1
        >>> high = .4
        >>> eq = False
        >>> # execute function
        >>> is_inbounds = inbounds(num, low, high, eq)
        >>> # verify results
        >>> result = ut.numpy_str(is_inbounds)
        >>> print(result)
        np.array([[False, False,  True],
                  [ True,  True, False],
                  [ True,  True,  True]], dtype=bool)

    """
    less    = operator.le if eq else operator.lt
    greater = operator.ge if eq else operator.gt
    and_ = np.logical_and if isinstance(num, np.ndarray) else operator.and_
    is_inbounds = and_(greater(num, low), less(num, high))
    return is_inbounds


#def inbounds(arr, min_, max_):
#    if min_ > 0 and max_ is not None:
#        #if max_ is not None and min
#        islt_max = np.less_equal(arr, max_)
#        isgt_min = np.greater_equal(arr, min_)
#        is_inbounds = np.logical_and(islt_max, isgt_min)
#    elif min_ == 0:
#        is_inbounds = np.less_equal(arr, max_)
#    elif max_ is None:
#        is_inbounds = np.greater_equal(arr, min_)
#    else:
#        assert False
#    return is_inbounds


def almost_eq(arr1, arr2, thresh=1E-11, ret_error=False):
    """ checks if floating point number are equal to a threshold
    """
    error = np.abs(arr1 - arr2)
    passed = error < thresh
    if ret_error:
        return passed, error
    return passed


def knapsack(items, maxweight):
    """
    Solve the knapsack problem by finding the most valuable
    subsequence of `items` subject that weighs no more than
    `maxweight`.

    Args:
        `items` (tuple): is a sequence of tuples `(value, weight, id_)`, where `value`
            is a number and `weight` is a non-negative integer, and `id_` is an
            item identifier.

        `maxweight` (scalar):  is a non-negative integer.

    Returns:
        tuple: a pair whose first element is the sum of values in the most
            valuable subsequence, and whose second element is the subsequence.

    References:
        http://codereview.stackexchange.com/questions/20569/dynamic-programming-solution-to-knapsack-problem

    CommandLine:
        python -m utool.util_alg --test-knapsack

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> items = [(4, 12, 0), (2, 1, 1), (6, 4, 2), (1, 1, 3), (2, 2, 4)]
        >>> result = knapsack(items, 15)
        >>> print(result)
        (11, [(2, 1, 1), (6, 4, 2), (1, 1, 3), (2, 2, 4)])
    """

    # Return the value of the most valuable subsequence of the first i
    # elements in items whose weights sum to no more than j.
    from utool import util_decor
    @util_decor.memoize_nonzero
    def bestvalue(i, j):
        if i == 0:
            return 0
        value, weight = items[i - 1][0:2]
        if weight > j:
            return bestvalue(i - 1, j)
        else:
            return max(bestvalue(i - 1, j),
                       bestvalue(i - 1, j - weight) + value)

    j = maxweight
    result = []
    for i in range(len(items), 0, -1):
        if bestvalue(i, j) != bestvalue(i - 1, j):
            result.append(items[i - 1])
            j -= items[i - 1][1]
    result.reverse()
    return bestvalue(len(items), maxweight), result


def cumsum(num_list):
    """ python cumsum

    References:
        http://stackoverflow.com/questions/9258602/elegant-pythonic-cumsum
    """
    return reduce(lambda acc, itm: operator.iadd(acc, [acc[-1] + itm]), num_list, [0])[1:]


def haversine(latlon1, latlon2):
    """
    #http://gis.stackexchange.com/questions/81551/matching-gps-tracks
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    References:
        http://en.wikipedia.org/wiki/Haversine_formula
        http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import scipy.spatial.distance as spdist
        >>> import utool as ut
        >>> import functools
        >>> gpsarr_track_list_ = [
        ...    np.array([[ -80.21895315, -158.81099213],
        ...              [ -12.08338926,   67.50368014],
        ...              [ -11.08338926,   67.50368014],
        ...              [ -11.08338926,   67.50368014],]
        ...    ),
        ...    np.array([[   9.77816711,  -17.27471498],
        ...              [ -51.67678814, -158.91065495],])
        ...    ]
        >>> latlon1 = gpsarr_track_list_[0][0]
        >>> latlon2 = gpsarr_track_list_[0][1]
        >>> kilometers = ut.haversine(latlon1, latlon2)
        >>> haversin_pdist = functools.partial(spdist.pdist, metric=ut.haversine)
        >>> dist_vector_list = list(map(haversin_pdist, gpsarr_track_list_))
        >>> dist_matrix_list = list(map(spdist.squareform, dist_vector_list))
    """
    # convert decimal degrees to radians
    lat1, lon1 = np.radians(latlon1)
    lat2, lon2 = np.radians(latlon2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))

    EARTH_RADIUS_KM = 6367
    kilometers = EARTH_RADIUS_KM * c
    return kilometers


def unixtime_hourdiff(x, y):
    return np.abs((x - y)) / (60. ** 2)


def safe_pdist(arr, *args, **kwargs):
    if arr is None or len(arr) < 2:
        return None
    else:
        return spdist.pdist(arr, *args, **kwargs)


def safe_div(a, b):
    return None if a is None or b is None else a / b


def safe_max(arr):
    return np.nan if arr is None or len(arr) == 0 else arr.max()


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_alg; utool.doctest_funcs(utool.util_alg, allexamples=True)"
        python -c "import utool, utool.util_alg; utool.doctest_funcs(utool.util_alg)"
        python -m utool.util_alg
        python -m utool.util_alg --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
