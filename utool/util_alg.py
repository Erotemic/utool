# Licence:
#
# TODO: Rename
# util_science
#
from __future__ import absolute_import, division, print_function
import numpy as np
from collections import defaultdict
#import six
from six.moves import zip
#from itertools import izip
from . import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[alg]')


def normalize(array, dim=0):
    return norm_zero_one(array, dim)


def norm_zero_one(array, dim=0):
    """
    normalizes a numpy array from 0 to 1
    </CYTH> """

    array_max  = array.max(dim)
    array_min  = array.min(dim)
    array_exnt = np.subtract(array_max, array_min)
    return np.divide(np.subtract(array, array_min), array_exnt)


def find_std_inliers(data, m=2):
    """ </CYTHE> """
    return abs(data - np.mean(data)) < m * np.std(data)


def choose(n, k):
    import scipy.misc
    return scipy.misc.comb(n, k, True)


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    </CYTHE>
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6], [1, 4, 7], [1, 5, 6], [1, 5, 7],
           [2, 4, 6], [2, 4, 7], [2, 5, 6], [2, 5, 7],
           [3, 4, 6], [3, 4, 7], [3, 5, 6], [3, 5, 7]])
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
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def almost_eq(a, b, thresh=1E-11, ret_error=False):
    """ checks if floating point number are equal to a threshold
    </CYTHE> """
    error = np.abs(a - b)
    passed = error < thresh
    if ret_error:
        return passed, error
    return passed


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
    </CYTHE> """
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
    """ </CYTHE> """
    # Sort by groupid for cache efficiency
    sorted_pairs = sorted(list(zip(groupid_list, item_list)))
    # Initialize dict of lists
    groupid2_items = defaultdict(list)
    for groupid, item in sorted_pairs:
        groupid2_items[groupid].append(item)
    return groupid2_items


def unpack_items_sorted(dict_, sortfn, reverse=True):
    """ Unpacks and sorts the dictionary by sortfn
    </CYTHE> """
    items = dict_.items()
    sorted_items = sorted(items, key=sortfn, reverse=reverse)
    sorted_keys, sorted_vals = list(zip(*sorted_items))
    return sorted_keys, sorted_vals


def unpack_items_sorted_by_lenvalue(dict_, reverse=True):
    """ Unpacks and sorts the dictionary by key
    </CYTHE> """
    def sort_lenvalue(item):
        return len(item[1])
    return unpack_items_sorted(dict_, sort_lenvalue)


def unpack_items_sorted_by_value(dict_, reverse=True):
    """ Unpacks and sorts the dictionary by key
    </CYTHE> """
    def sort_value(item):
        return item[1]
    return unpack_items_sorted(dict_, sort_value)


def flatten_membership_mapping(uid_list, members_list):
    """ </CYTHE> """
    num_members = sum(map(len, members_list))
    flat_uids = [None for _ in xrange(num_members)]
    flat_members = [None for _ in xrange(num_members)]
    count = 0
    for uid, members in zip(uid_list, members_list):
        for member in members:
            flat_uids[count]    = uid
            flat_members[count] = member
            count += 1
    return flat_uids, flat_members


def void_rowview_numpy(arr):
    """ returns view of nparray where each row is a single item
    </CYTHE> """
    void_dtype = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    arr_void_view = np.ascontiguousarray(arr).view(void_dtype)
    return arr_void_view


def unique_row_indexes(arr):
    """ np.unique on rows </CYTHE> """
    arr_void_view = void_rowview_numpy(arr)
    _, unique_rowx = np.unique(arr_void_view, return_index=True)
    # cast back to original dtype
    unique_rowx.sort()
    return unique_rowx

#        (qaids, aids, scores, ranks) = self.cand_match_list
#        # reorder candidates
#        aid_list1 = map(int, qaids)
#        aid_list2 = map(int, aids)
#        from itertools import zip, groupby
#        def groupkey(tup):
#            aid1, aid2 = tup[-2:-1]
#            return (min(aid1, aid2), max(aid1, aid2))
#        zipped = sorted(list(zip(range(len(aid_list1)), aid_list1, aid_list2)), key=lambda tup: tup[1])
#        # key is ordered pair, list is directed pairs and order
#        grouped = [(key, list(subiter)) for key, subiter in groupby(zipped, groupkey)]
#        group_order = sorted([sorted(sublist) for key, sublist in grouped])
#        new_order = [tup[0] for tup in utool.flatten(group_order)]

#        old_graph = defaultdict(list)
#        for edge, dedge_list in grouped:
#            aid1, aid2 = edge
#            old_graph[aid1].append((aid1, aid2))
#        old_graph = {key: list(set(value)) for key, value in six.iteritems(old_graph)}

#        old_graph = dict(list(six.iteritems(old_graph)))

#        def connected_components(neighbors):
#            """ from
#            http://stackoverflow.com/questions/10301000/python-connected-components
#            """
#            seen = set()
#            cc_list = []
#            for node in neighbors:
#                if node not in seen:
#                    node_set = set([node])
#                    c = []
#                    while node_set:
#                        node = node_set.pop()
#                        seen.add(node)
#                        try:
#                            node_set |= neighbors[node] - seen
#                        except KeyError:
#                            pass
#                        c.append(node)
#                    cc_list.append(c)
#                    #yield c

#        new_graph = {node: set(endpt for edge in edges for endpt in edge)
#                     for node, edges in old_graph.items()}\
#        neighbors = new_graph

#        components = []
#        for component in connected_components(new_graph):
#            c = set(component)
#            components.append([edge for edges in old_graph.values()
#                                    for edge in edges
#                                    if c.intersection(edge)])


def get_phi():
    """ Golden Ratio: phi = 1 / sqrt(5) / 2.0 = 1.61803398875
    </CYTHE> """
    #phi = (1.0 + np.sqrt(5)) / 2.0 = 1.61803398875
    # return phi
    return 1.61803398875


def get_phi_ratio1():
    return 1.0 / get_phi()


PHI = get_phi()
PHI_A = (1 / PHI)
PHI_B = 1 - PHI_A


def iceil(num):
    """ Integer ceiling. (because numpy doesn't have it! """
    return int(np.ceil(num))


def iround(num):
    """ Integer round. (because numpy doesn't have it! """
    return int(round(num))
