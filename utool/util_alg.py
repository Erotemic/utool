# TODO Licence:
#
# TODO:  move library intensive functions to vtool
from __future__ import absolute_import, division, print_function, unicode_literals
import operator as op
import decimal
import six
import itertools
from six.moves import zip, range, reduce, map
from collections import defaultdict
import math
from utool import util_type
from utool import util_list
from utool import util_dict
from utool import util_inject
from utool import util_decor
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    # TODO remove numpy
    pass
try:
    import scipy.spatial.distance as spdist
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
print, rrr, profile = util_inject.inject2(__name__)


# Constants
PHI = 1.61803398875
PHI_A = (1 / PHI)
PHI_B = 1 - PHI_A
TAU = 2 * math.pi
# Conversion factors
KM_PER_MILE = 1.609344
MM_PER_INCH = 25.4
FOOT_PER_MILE = 5280


def find_group_differences(groups1, groups2):
    r"""
    Returns a measure of how disimilar two groupings are

    Args:
        groups1 (list): true grouping of items
        groups2 (list): predicted grouping of items

    CommandLine:
        python -m utool.util_alg find_group_differences

    SeeAlso:
        vtool.group_indicies
        vtool.apply_grouping

    Example0:
        >>> from utool.util_alg import *  # NOQA
        >>> groups1 = [[1, 2, 3], [4], [5, 6], [7, 8], [9, 10, 11]]
        >>> groups2 = [[1, 2, 11], [3, 4], [5, 6], [7], [8, 9], [10]]
        >>> total_error = find_group_differences(groups1, groups2)
        >>> result = ('total_error = %r' % (total_error,))
        >>> print(result)
        total_error = 20

    Example1:
        >>> from utool.util_alg import *  # NOQA
        >>> groups1 = [[1, 2, 3], [4], [5, 6]]
        >>> groups2 = [[1, 2, 3], [4], [5, 6]]
        >>> total_error = find_group_differences(groups1, groups2)
        >>> result = ('total_error = %r' % (total_error,))
        >>> print(result)
        total_error = 0

    Example2:
        >>> from utool.util_alg import *  # NOQA
        >>> groups1 = [[1, 2, 3], [4], [5, 6]]
        >>> groups2 = [[1, 2], [4], [5, 6]]
        >>> total_error = find_group_differences(groups1, groups2)
        >>> result = ('total_error = %r' % (total_error,))
        >>> print(result)
        total_error = 4

    Ignore:
        # Can this be done via sklearn label analysis?
        # maybe no... the labels assigned to each component are arbitrary
        # maybe if we label edges? likely too many labels.
        groups1 = [[1, 2, 3], [4], [5, 6], [7, 8], [9, 10, 11]]
        groups2 = [[1, 2, 11], [3, 4], [5, 6], [7], [8, 9], [10]]
    """
    import utool as ut
    # For each group, build mapping from each item to the members the group
    item_to_others1 = {item: set(_group) - {item}
                       for _group in groups1 for item in _group}
    item_to_others2 = {item: set(_group) - {item}
                       for _group in groups2 for item in _group}

    flat_items1 = ut.flatten(groups1)
    flat_items2 = ut.flatten(groups2)

    flat_items = list(set(flat_items1 + flat_items2))

    errors = []
    item_to_error = {}
    for item in flat_items:
        # Determine the number of unshared members in each group
        others1 = item_to_others1.get(item, set([]))
        others2 = item_to_others2.get(item, set([]))
        missing1 = others1 - others2
        missing2 = others2 - others1
        error = len(missing1) + len(missing2)
        if error > 0:
            item_to_error[item] = error
        errors.append(error)
    total_error = sum(errors)
    return total_error


def find_group_consistencies(groups1, groups2):
    r"""
    Returns a measure of group consistency

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> groups1 = [[1, 2, 3], [4], [5, 6]]
        >>> groups2 = [[1, 2], [4], [5, 6]]
        >>> common_groups = sorted(find_group_consistencies(groups1, groups2))
        >>> import ubelt as ub
        >>> print('common_groups = {}'.format(ub.repr2(common_groups, nl=1)))
        common_groups = [
            (4,),
            (5, 6),
        ]
    """
    group1_list = {tuple(sorted(_group)) for _group in groups1}
    group2_list = {tuple(sorted(_group)) for _group in groups2}
    common_groups = list(group1_list.intersection(group2_list))
    return common_groups


def compare_groups(true_groups, pred_groups):
    r"""
    Finds how predictions need to be modified to match the true grouping.

    Notes:
        pred_merges - the merges needed that would need to be done for the
            pred_groups to match true_groups.
        pred_hybrid - the hybrid split/merges needed that would need to be done
            for the pred_groups to match true_groups.

    Ignore:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> true_groups = [
        >>>   [20, 21], [22, 23], [1, 2], [12, 13, 14], [4], [5, 6, 3], [7, 8],
        >>>   [9, 10, 11], [31, 32, 33, 34, 35],   [41, 42, 43, 44], [45], [50]
        >>> ]
        >>> pred_groups = [
        >>>     [20, 21, 22, 23], [1, 2], [12], [13, 14], [3, 4], [5, 6,11],
        >>>     [7], [8, 9], [10], [31, 32], [33, 34, 35], [41, 42, 43, 44, 45]
        >>> ]
        >>> comparisons = ut.compare_groups(true_groups, pred_groups)
        >>> print(comparisons)
        >>> result = ut.repr4(comparisons)
        >>> print(result)
        {
            'common': {{1, 2}},
            'pred_hybrid': {{10}, {3, 4}, {5, 6, 11}, {7}, {8, 9}},
            'pred_merges': [{{12}, {13, 14}}, {{31, 32}, {33, 34, 35}}],
            'pred_splits': [{20, 21, 22, 23}, {41, 42, 43, 44, 45}],
            'true_hybrid': {{3, 5, 6}, {4}, {50}, {7, 8}, {9, 10, 11}},
            'true_merges': [{12, 13, 14}, {31, 32, 33, 34, 35}],
            'true_splits': [{{20, 21}, {22, 23}}, {{41, 42, 43, 44}, {45}}],
        }
    """
    import utool as ut
    true = {frozenset(_group) for _group in true_groups}
    pred = {frozenset(_group) for _group in pred_groups}

    # Find the groups that are exactly the same
    common = true.intersection(pred)

    true_sets = true.difference(common)
    pred_sets = pred.difference(common)

    # connected compoment lookups
    pred_conn = {p: frozenset(ps) for ps in pred for p in ps}
    true_conn = {t: frozenset(ts) for ts in true for t in ts}

    # How many predictions can be merged into perfect pieces?
    # For each true sets, find if it can be made via merging pred sets
    pred_merges = []
    true_merges = []
    for ts in true_sets:
        ccs = set([pred_conn.get(t, frozenset()) for t in ts])
        if frozenset.union(*ccs) == ts:
            # This is a pure merge
            pred_merges.append(ccs)
            true_merges.append(ts)

    # How many predictions can be split into perfect pieces?
    true_splits = []
    pred_splits = []
    for ps in pred_sets:
        ccs = set([true_conn.get(p, frozenset()) for p in ps])
        if frozenset.union(*ccs) == ps:
            # This is a pure merge
            true_splits.append(ccs)
            pred_splits.append(ps)

    pred_merges_flat = ut.flatten(pred_merges)
    true_splits_flat = ut.flatten(true_splits)

    pred_hybrid = frozenset(map(frozenset, pred_sets)).difference(
        set(pred_splits + pred_merges_flat))

    true_hybrid = frozenset(map(frozenset, true_sets)).difference(
        set(true_merges + true_splits_flat))

    comparisons = {
        'common': common,
        # 'true_splits_flat': true_splits_flat,
        'true_splits': true_splits,
        'true_merges': true_merges,
        'true_hybrid': true_hybrid,
        'pred_splits': pred_splits,
        'pred_merges': pred_merges,
        # 'pred_merges_flat': pred_merges_flat,
        'pred_hybrid': pred_hybrid,
    }
    return comparisons


def grouping_delta(old, new, pure=True):
    r"""
    Finds what happened to the old groups to form the new groups.

    Args:
        old (set of frozensets): old grouping
        new (set of frozensets): new grouping
        pure (bool): hybrids are separated from pure merges and splits if
            pure is True, otherwise hybrid cases are grouped in merges and
            splits.

    Returns:
        dict: delta: dictionary of changes containing the merges, splits,
            unchanged, and hybrid cases. Except for unchanged, case a subdict
            with new and old keys.  For splits / merges, one of these contains
            nested sequences to indicate what the split / merge is.

    TODO:
        incorporate addition / deletion of elements?

    Notes:
        merges - which old groups were merged into a single new group.
        splits - which old groups were split into multiple new groups.
        hybrid - which old groups had split/merge actions applied.
        unchanged - which old groups are the same as new groups.

    Ignore:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> old = [
        >>>     [20, 21, 22, 23], [1, 2], [12], [13, 14], [3, 4], [5, 6,11],
        >>>     [7], [8, 9], [10], [31, 32], [33, 34, 35], [41, 42, 43, 44, 45]
        >>> ]
        >>> new = [
        >>>   [20, 21], [22, 23], [1, 2], [12, 13, 14], [4], [5, 6, 3], [7, 8],
        >>>   [9, 10, 11], [31, 32, 33, 34, 35],   [41, 42, 43, 44], [45],
        >>> ]
        >>> delta = ut.grouping_delta(old, new)
        >>> assert set(old[0]) in delta['splits']['old']
        >>> assert set(new[3]) in delta['merges']['new']
        >>> assert set(old[1]) in delta['unchanged']
        >>> result = ut.repr4(delta, nl=2, nobr=True, sk=True)
        >>> print(result)
        unchanged: {
            {1, 2},
        },
        splits: {
            old: [{20, 21, 22, 23}, {41, 42, 43, 44, 45}],
            new: [{{20, 21}, {22, 23}}, {{41, 42, 43, 44}, {45}}],
        },
        merges: {
            old: [{{12}, {13, 14}}, {{31, 32}, {33, 34, 35}}],
            new: [{12, 13, 14}, {31, 32, 33, 34, 35}],
        },
        hybrid: {
            old: {{10}, {3, 4}, {5, 6, 11}, {7}, {8, 9}},
            new: {{3, 5, 6}, {4}, {7, 8}, {9, 10, 11}},
            splits: [{{7}}, {{11}, {5, 6}}, {{10}}, {{3}, {4}}, {{8}, {9}}],
            merges: [{{7}, {8}}, {{4}}, {{3}, {5, 6}}, {{10}, {11}, {9}}],
        },


    Ignore:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> old = [
        >>>     [1, 2, 3], [4], [5, 6, 7, 8, 9], [10, 11, 12]
        >>> ]
        >>> new = [
        >>>     [1], [2], [3, 4], [5, 6, 7], [8, 9, 10, 11, 12]
        >>> ]
        >>> # every case here is hybrid
        >>> pure_delta = ut.grouping_delta(old, new, pure=True)
        >>> assert len(ut.flatten(pure_delta['merges'].values())) == 0
        >>> assert len(ut.flatten(pure_delta['splits'].values())) == 0
        >>> delta = ut.grouping_delta(old, new, pure=False)
        >>> delta = ut.order_dict_by(delta, ['unchanged', 'splits', 'merges'])
        >>> result = ut.repr4(delta, nl=2, sk=True)
        >>> print(result)
        {
            unchanged: {},
            splits: [
                [{2}, {3}, {1}],
                [{8, 9}, {5, 6, 7}],
            ],
            merges: [
                [{4}, {3}],
                [{8, 9}, {10, 11, 12}],
            ],
        }
    """
    import utool as ut
    _old = {frozenset(_group) for _group in old}
    _new = {frozenset(_group) for _group in new}

    _new_items = set(ut.flatten(_new))
    _old_items = set(ut.flatten(_old))
    assert _new_items == _old_items, 'new and old sets must be the same'

    # Find the groups that are exactly the same
    unchanged = _new.intersection(_old)

    new_sets = _new.difference(unchanged)
    old_sets = _old.difference(unchanged)

    # connected compoment lookups
    old_conn = {p: frozenset(ps) for ps in _old for p in ps}
    new_conn = {t: frozenset(ts) for ts in _new for t in ts}

    # How many old sets can be merged into perfect pieces?
    # For each new sets, find if it can be made via merging old sets
    old_merges = []
    new_merges = []
    for ts in new_sets:
        ccs = set([old_conn.get(t, frozenset()) for t in ts])
        if frozenset.union(*ccs) == ts:
            # This is a pure merge
            old_merges.append(ccs)
            new_merges.append(ts)

    # How many oldictions can be split into perfect pieces?
    new_splits = []
    old_splits = []
    for ps in old_sets:
        ccs = set([new_conn.get(p, frozenset()) for p in ps])
        if frozenset.union(*ccs) == ps:
            # This is a pure merge
            new_splits.append(ccs)
            old_splits.append(ps)

    old_merges_flat = ut.flatten(old_merges)
    new_splits_flat = ut.flatten(new_splits)

    old_hybrid = frozenset(map(frozenset, old_sets)).difference(
        set(old_splits + old_merges_flat))

    new_hybrid = frozenset(map(frozenset, new_sets)).difference(
        set(new_merges + new_splits_flat))

    breakup_hybrids = True
    if breakup_hybrids:
        # First split each hybrid
        lookup = {a: n for n, items in enumerate(new_hybrid) for a in items}
        hybrid_splits = []
        for items in old_hybrid:
            nids = ut.take(lookup, items)
            split_part = list(ut.group_items(items, nids).values())
            hybrid_splits.append(set(map(frozenset, split_part)))

        # And then merge them into new groups
        hybrid_merge_parts = ut.flatten(hybrid_splits)
        part_nids = [lookup[next(iter(aids))] for aids in hybrid_merge_parts]
        hybrid_merges = list(map(set, ut.group_items(hybrid_merge_parts,
                                                     part_nids).values()))

    if pure:
        delta = ut.odict()
        delta['unchanged'] = unchanged
        delta['splits'] = ut.odict([
            ('old', old_splits),
            ('new', new_splits),
        ])
        delta['merges'] = ut.odict([
            ('old', old_merges),
            ('new', new_merges),
        ])
        delta['hybrid'] = ut.odict([
            ('old', old_hybrid),
            ('new', new_hybrid),
            ('splits', hybrid_splits),
            ('merges', hybrid_merges),
        ])
    else:
        # Incorporate hybrid partial cases with pure splits and merges
        new_splits2 = [s for s in hybrid_splits if len(s) > 1]
        old_merges2 = [m for m in hybrid_merges if len(m) > 1]
        all_new_splits = new_splits + new_splits2
        all_old_merges = old_merges + old_merges2

        # Don't bother differentiating old and new
        # old_splits2 = [frozenset(ut.flatten(s)) for s in new_splits2]
        # new_merges2 = [frozenset(ut.flatten(m)) for m in old_merges2]
        # all_old_splits = old_splits + old_splits2
        # all_new_merges = new_merges + new_merges2

        splits = all_new_splits
        merges = all_old_merges

        # Sort by split and merge sizes
        splits = ut.sortedby(splits, [len(ut.flatten(_)) for _ in splits])
        merges = ut.sortedby(merges, [len(ut.flatten(_)) for _ in merges])
        splits = [ut.sortedby(_, ut.emap(len, _)) for _ in splits]
        merges = [ut.sortedby(_, ut.emap(len, _)) for _ in merges]

        delta = ut.odict()
        delta['unchanged'] = unchanged
        delta['splits'] = splits
        delta['merges'] = merges
    return delta


def grouping_delta_stats(old, new):
    """
    Returns statistics about grouping changes

    Args:
        old (set of frozenset): old grouping
        new (set of frozenset): new grouping

    Returns:
        pd.DataFrame: df: data frame of size statistics

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> old = [
        >>>     [20, 21, 22, 23], [1, 2], [12], [13, 14], [3, 4], [5, 6,11],
        >>>     [7], [8, 9], [10], [31, 32], [33, 34, 35], [41, 42, 43, 44, 45]
        >>> ]
        >>> new = [
        >>>   [20, 21], [22, 23], [1, 2], [12, 13, 14], [4], [5, 6, 3], [7, 8],
        >>>   [9, 10, 11], [31, 32, 33, 34, 35],   [41, 42, 43, 44], [45],
        >>> ]
        >>> df = ut.grouping_delta_stats(old, new)
        >>> print(df)
    """
    import pandas as pd
    import utool as ut
    group_delta = ut.grouping_delta(old, new)
    stats = ut.odict()
    unchanged = group_delta['unchanged']
    splits = group_delta['splits']
    merges = group_delta['merges']
    hybrid = group_delta['hybrid']
    statsmap = ut.partial(lambda x: ut.stats_dict(map(len, x), size=True))
    stats['unchanged'] = statsmap(unchanged)
    stats['old_split'] = statsmap(splits['old'])
    stats['new_split'] = statsmap(ut.flatten(splits['new']))
    stats['old_merge'] = statsmap(ut.flatten(merges['old']))
    stats['new_merge'] = statsmap(merges['new'])
    stats['old_hybrid'] = statsmap(hybrid['old'])
    stats['new_hybrid'] = statsmap(hybrid['new'])
    df = pd.DataFrame.from_dict(stats, orient='index')
    df = df.loc[list(stats.keys())]
    return df


def upper_diag_self_prodx(list_):
    """
    upper diagnoal of cartesian product of self and self.
    Weird name. fixme

    Args:
        list_ (list):

    Returns:
        list:

    CommandLine:
        python -m utool.util_alg --exec-upper_diag_self_prodx

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> list_ = [1, 2, 3]
        >>> result = upper_diag_self_prodx(list_)
        >>> print(result)
        [(1, 2), (1, 3), (2, 3)]
    """
    return [(item1, item2)
            for n1, item1 in enumerate(list_)
            for n2, item2 in enumerate(list_) if n1 < n2]


def diagonalized_iter(size):
    r"""
    TODO: generalize to more than 2 dimensions to be more like
    itertools.product.

    CommandLine:
        python -m utool.util_alg --exec-diagonalized_iter
        python -m utool.util_alg --exec-diagonalized_iter --size=5

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> size = ut.get_argval('--size', default=4)
        >>> iter_ = diagonalized_iter(size)
        >>> mat = [[None] * size for _ in range(size)]
        >>> for count, (r, c) in enumerate(iter_):
        >>>     mat[r][c] = count
        >>> result = ut.repr2(mat, nl=1, packed=True)
        >>> print(result)
        [[0, 2, 5, 9],
         [1, 4, 8, 12],
         [3, 7, 11, 14],
         [6, 10, 13, 15],]
    """
    for i in range(0, size + 1):
        for r, c in zip(reversed(range(i)), (range(i))):
            yield (r, c)
    for i in range(1, size):
        for r, c in zip(reversed(range(i, size)), (range(i, size))):
            yield (r, c)


def colwise_diag_idxs(size, num=2):
    r"""
    dont trust this implementation or this function name

    Args:
        size (int):

    Returns:
        ?: upper_diag_idxs

    CommandLine:
        python -m utool.util_alg --exec-colwise_diag_idxs --size=5 --num=2
        python -m utool.util_alg --exec-colwise_diag_idxs --size=3 --num=3

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> size = ut.get_argval('--size', default=5)
        >>> num = ut.get_argval('--num', default=2)
        >>> mat = np.zeros([size] * num, dtype=int)
        >>> upper_diag_idxs = colwise_diag_idxs(size, num)
        >>> poses = np.array(upper_diag_idxs)
        >>> idxs = np.ravel_multi_index(poses.T, mat.shape)
        >>> print('poses.T =\n%s' % (ut.repr2(poses.T),))
        >>> mat[tuple(poses.T)] = np.arange(1, len(poses) + 1)
        >>> print(mat)
        poses.T =
        np.array([[0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
                  [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]])
    """
    # diag_idxs = list(diagonalized_iter(size))
    # upper_diag_idxs = [(r, c) for r, c in diag_idxs if r < c]
    # # diag_idxs = list(diagonalized_iter(size))
    import utool as ut
    diag_idxs = ut.iprod(*[range(size) for _ in range(num)])
    #diag_idxs = list(ut.iprod(range(size), range(size)))
    # this is pretty much a simple c ordering
    upper_diag_idxs = [
        tup[::-1] for tup in diag_idxs
        if all([a > b for a, b in ut.itertwo(tup)])
        #if all([a > b for a, b in ut.itertwo(tup[:2])])
    ]
    #upper_diag_idxs = [(c, r) for r, c in diag_idxs if r > c]
    # # upper_diag_idxs = [(r, c) for r, c in diag_idxs if r > c]
    return upper_diag_idxs


def self_prodx(list_):
    return [(item1, item2)
            for n1, item1 in enumerate(list_)
            for n2, item2 in enumerate(list_) if n1 != n2]


def product_nonsame(list1, list2):
    """ product of list1 and list2 where items are non equal """
    for item1, item2 in itertools.product(list1, list2):
        if item1 != item2:
            yield (item1, item2)


def product_nonsame_self(list_):
    return product_nonsame(list_, list_)


def greedy_max_inden_setcover(candidate_sets_dict, items, max_covers=None):
    """
    greedy algorithm for maximum independent set cover

    Covers items with sets from candidate sets. Could be made faster.

    CommandLine:
        python -m utool.util_alg --test-greedy_max_inden_setcover

    Example0:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> candidate_sets_dict = {'a': [5, 3], 'b': [2, 3, 5],
        ...                        'c': [4, 8], 'd': [7, 6, 2, 1]}
        >>> items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> max_covers = None
        >>> tup = greedy_max_inden_setcover(candidate_sets_dict, items, max_covers)
        >>> (uncovered_items, covered_items_list, accepted_keys) = tup
        >>> result = ut.repr4((uncovered_items, sorted(list(accepted_keys))), nl=False)
        >>> print(result)
        ([0, 9], ['a', 'c', 'd'])

    Example1:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> candidate_sets_dict = {'a': [5, 3], 'b': [2, 3, 5],
        ...                        'c': [4, 8], 'd': [7, 6, 2, 1]}
        >>> items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> max_covers = 1
        >>> tup = greedy_max_inden_setcover(candidate_sets_dict, items, max_covers)
        >>> (uncovered_items, covered_items_list, accepted_keys) = tup
        >>> result = ut.repr4((uncovered_items, sorted(list(accepted_keys))), nl=False)
        >>> print(result)
        ([0, 3, 4, 5, 8, 9], ['d'])
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


def setcover_greedy(candidate_sets_dict, items=None, set_weights=None, item_values=None, max_weight=None):
    r"""
    Greedy algorithm for various covering problems.
    approximation gaurentees depending on specifications  like set_weights and item values

    Set Cover: log(len(items) + 1) approximation algorithm
    Weighted Maximum Cover: 1 - 1/e == .632 approximation algorithm
    Generalized maximum coverage is not implemented

    References:
        https://en.wikipedia.org/wiki/Maximum_coverage_problem

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> candidate_sets_dict = {
        >>>     'a': [1, 2, 3, 8, 9, 0],
        >>>     'b': [1, 2, 3, 4, 5],
        >>>     'c': [4, 5, 7],
        >>>     'd': [5, 6, 7],
        >>>     'e': [6, 7, 8, 9, 0],
        >>> }
        >>> max_weight = None
        >>> items = None
        >>> set_weights = None
        >>> item_values = None
        >>> greedy_soln = ut.sort_dict(ut.setcover_greedy(candidate_sets_dict))
        >>> exact_soln = ut.sort_dict(ut.setcover_ilp(candidate_sets_dict))
        >>> print('greedy_soln = %r' % (greedy_soln,))
        >>> print('exact_soln = %r' % (exact_soln,))
    """
    import utool as ut
    solution_cover = {}
    # If candset_weights or item_values not given use the length as defaults
    if items is None:
        items = ut.flatten(candidate_sets_dict.values())
    if set_weights is None:
        get_weight = len
    else:
        def get_weight(solution_cover):
            sum([set_weights[key] for key in solution_cover.keys()])
    if item_values is None:
        get_value = len
    else:
        def get_value(vals):
            sum([item_values[v] for v in vals])
    if max_weight is None:
        max_weight = get_weight(candidate_sets_dict)
    avail_covers = {key: set(val) for key, val in candidate_sets_dict.items()}
    # While we still need covers
    while get_weight(solution_cover) < max_weight and len(avail_covers) > 0:
        # Find candiate set with the most uncovered items
        avail_covers.values()
        uncovered_values = list(map(get_value, avail_covers.values()))
        chosen_idx = ut.argmax(uncovered_values)
        if uncovered_values[chosen_idx] <= 0:
            # needlessly adding value-less items
            break
        chosen_key = list(avail_covers.keys())[chosen_idx]
        # Add values in this key to the cover
        chosen_set = avail_covers[chosen_key]
        solution_cover[chosen_key] = candidate_sets_dict[chosen_key]
        # Remove chosen set from available options and covered items
        # from remaining available sets
        del avail_covers[chosen_key]
        for vals in avail_covers.values():
            vals.difference_update(chosen_set)
    return solution_cover


def setcover_ilp(candidate_sets_dict, items=None, set_weights=None, item_values=None, max_weight=None, verbose=False):
    """
    Set cover / Weighted Maximum Cover exact algorithm

    https://en.wikipedia.org/wiki/Maximum_coverage_problem
    """
    import utool as ut
    import pulp
    if items is None:
        items = ut.flatten(candidate_sets_dict.values())
    if set_weights is None:
        set_weights = {i: 1 for i in candidate_sets_dict.keys()}
    if item_values is None:
        item_values = {e: 1 for e in items}

    if max_weight is None:
        max_weight = sum(ut.take(set_weights, candidate_sets_dict.keys()))

    if False:
        # This is true set coer
        # Formulate integer program
        prob = pulp.LpProblem("Set Cover", pulp.LpMinimize)
        # Solution variable indicates if set it chosen or not
        set_indices = candidate_sets_dict.keys()
        x = pulp.LpVariable.dicts(name='x', indexs=set_indices,
                                  lowBound=0, upBound=1, cat=pulp.LpInteger)
        # minimize the number of sets
        prob.objective = sum(x[i] for i in set_indices)
        # subject to
        for e in items:
            # each element is covered
            containing_sets = [i for i in set_indices if e in candidate_sets_dict[i]]
            prob.add(sum(x[i] for i in containing_sets) >= 1)
        # Solve using with solver like CPLEX, GLPK, or SCIP.
        #pulp.CPLEX().solve(prob)
        pulp.PULP_CBC_CMD().solve(prob)
        # Read solution
        solution_keys = [i for i in set_indices if x[i].varValue]
        solution_cover = {i: candidate_sets_dict[i] for i in solution_keys}
        # Print summary
        if verbose:
            print(prob)
            print('OPT:')
            print('\n'.join(['    %s = %s' % (x[i].name, x[i].varValue) for i in set_indices]))
            print('solution_cover = %r' % (solution_cover,))
    else:
        prob = pulp.LpProblem("Maximum Cover", pulp.LpMaximize)
        # Solution variable indicates if set it chosen or not
        item_indicies = items
        set_indices = candidate_sets_dict.keys()
        x = pulp.LpVariable.dicts(name='x', indexs=set_indices,
                                  lowBound=0, upBound=1, cat=pulp.LpInteger)
        y = pulp.LpVariable.dicts(name='y', indexs=item_indicies,
                                  lowBound=0, upBound=1, cat=pulp.LpInteger)
        r = pulp.LpVariable.dicts(name='r', indexs=item_indicies)
        # maximize the value of the covered items
        primary_objective = sum(item_values[e] * y[e] for e in item_indicies)
        # minimize the number of sets used (make sure it does not influence the chosen primary objective)
        # This is only possible when values are non-negative
        # TODO: minimize redundency
        min_influence = min(item_values.values())
        secondary_weight = min_influence / (1.1 * len(set_indices))
        secondary_objective = (sum(-x[i] for i in set_indices)) * secondary_weight
        #
        prob.objective = primary_objective + secondary_objective
        # subject to
        # no more than the maximum weight
        prob.add(sum(x[i] * set_weights[i] for i in set_indices) <= max_weight)
        # If an item is chosen than at least one set containing it is chosen
        for e in item_indicies:
            containing_sets = [i for i in set_indices if e in candidate_sets_dict[i]]
            if len(containing_sets) > 0:
                prob.add(sum(x[i] for i in containing_sets) >= y[e])
                # record number of times each item is covered
                prob.add(sum(x[i] for i in containing_sets) == r[e])
        # Solve using with solver like CPLEX, GLPK, or SCIP.
        #pulp.CPLEX().solve(prob)
        pulp.PULP_CBC_CMD().solve(prob)
        # Read solution
        solution_keys = [i for i in set_indices if x[i].varValue]
        solution_cover = {i: candidate_sets_dict[i] for i in solution_keys}
        # Print summary
        if verbose:
            print(prob)
            print('OPT:')
            print('\n'.join(['    %s = %s' % (x[i].name, x[i].varValue) for i in set_indices]))
            print('\n'.join(['    %s = %s' % (y[i].name, y[i].varValue) for i in item_indicies]))
            print('solution_cover = %r' % (solution_cover,))
    return solution_cover


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


def item_hist(list_):
    """ counts the number of times each item appears in the dictionary """
    dict_hist = {}
    # Insert each item into the correct group
    for item in list_:
        if item not in dict_hist:
            dict_hist[item] = 0
        dict_hist[item] += 1
    return dict_hist


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


def get_phi():
    """ Golden Ratio: phi = 1 / sqrt(5) / 2.0 = 1.61803398875 """
    #phi = (1.0 + sqrt(5)) / 2.0 = 1.61803398875
    # return phi
    return PHI


def get_phi_ratio1():
    return 1.0 / get_phi()


def is_prime(num):
    """
    naive function for finding primes. Good for stress testing

    References:
        http://thelivingpearl.com/2013/01/06/how-to-find-prime-numbers-in-python/

    CommandLine:
        python -m utool.util_alg --test-is_prime

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> with ut.Timer('isprime'):
        >>>     series = [is_prime(n) for n in range(30)]
        >>> result = ('primes = %s' % (str(ut.list_where(series[0:10])),))
        >>> print(result)
        primes = [2, 3, 5, 7]
    """
    return num >= 2 and not any(num % j == 0 for j in range(2, num))
    # if num < 2:
    #     return False
    # for j in range(2, num):
    #     if (num % j) == 0:
    #         return False
    # return True


def fibonacci_recursive(n):
    """
    CommandLine:
        python -m utool.util_alg --test-fibonacci_recursive

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> with ut.Timer('fib rec'):
        >>>     series = [fibonacci_recursive(n) for n in range(20)]
        >>> result = ('series = %s' % (str(series[0:10]),))
        >>> print(result)
        series = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    """
    if n < 2:
        return n
    return fibonacci_recursive(n - 2) + fibonacci_recursive(n - 1)


def fibonacci_iterative(n):
    """
    Args:
        n (int):

    Returns:
        int: the n-th fibonacci number

    References:
        http://stackoverflow.com/questions/15047116/iterative-alg-fib

    CommandLine:
        python -m utool.util_alg fibonacci_iterative

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> with ut.Timer('fib iter'):
        >>>     series = [fibonacci_iterative(n) for n in range(20)]
        >>> result = ('series = %s' % (str(series[0:10]),))
        >>> print(result)
        series = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    """
    a, b = 0, 1
    for _ in range(0, n):
        a, b = b, a + b
    return a


def fibonacci_approx(n):
    r"""
    approximate value (due to numerical errors) of fib(n) using closed form
    expression

    Args:
        n (int):

    Returns:
        int: the n-th fib number

    CommandLine:
        python -m utool.util_alg fibonacci_approx

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> series = [int(fibonacci_approx(n)) for n in range(20)]
        >>> result = ('series = %s' % (str(series[0:10]),))
        >>> print(result)
    """
    sqrt_5 = math.sqrt(5)
    phi = (1 + sqrt_5) / 2
    return ((phi ** n) - (-phi) ** (-n)) / sqrt_5

fibonacci = fibonacci_iterative


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
        #print(len(first_100_primes))
        nth_prime = first_100_primes[n - 1]
    else:
        if safe:
            primes = [num for num in range(2, max_prime) if is_prime(num)]
            nth_prime = primes[n]
        else:
            # This can run for a while... get it? while?
            nth_prime = get_nth_prime_bruteforce(n)
    return nth_prime


def get_nth_prime_bruteforce(n, start_guess=2, start_num_primes=0):
    """
    Args:
        n (int): the n-th prime (n=2000 takes about a second)

    CommandLine:
        python -m utool.util_alg get_nth_prime_bruteforce --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> n_list = []
        >>> time_list = []
        >>> for n in range(1, 2000 + 2, 500):
        >>>     with ut.Timer(verbose=0) as t:
        >>>         get_nth_prime_bruteforce(n)
        >>>     time_list += [t.ellapsed]
        >>>     n_list += [n]
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.multi_plot(n_list, [time_list], xlabel='prime', ylabel='time')
        >>> ut.show_if_requested()
    """
    guess = start_guess
    num_primes_found = start_num_primes
    while True:
        if is_prime(guess):
            num_primes_found += 1
        if num_primes_found == n:
            nth_prime = guess
            break
        guess += 1
    return nth_prime


def get_prime_index(prime):
    guess = 2
    num_primes_found = 0
    while True:
        if is_prime(guess):
            num_primes_found += 1
            if guess == prime:
                return num_primes_found
        else:
            assert guess != prime, 'input=%r is not prime. has %r primes less than it' % (prime, num_primes_found)
        guess += 1


def generate_primes(stop=None, start_guess=2):
    guess = start_guess
    num_generated = 0
    while True:
        if is_prime(guess):
            num_generated += 1
            yield guess
            if num_generated >= stop:
                break
        guess += 1


def knapsack(items, maxweight, method='recursive'):
    r"""
    Solve the knapsack problem by finding the most valuable subsequence of
    `items` subject that weighs no more than `maxweight`.

    Args:
        items (tuple): is a sequence of tuples `(value, weight, id_)`, where
            `value` is a number and `weight` is a non-negative integer, and
            `id_` is an item identifier.

        maxweight (scalar):  is a non-negative integer.

    Returns:
        tuple: (total_value, items_subset) - a pair whose first element is the
            sum of values in the most valuable subsequence, and whose second
            element is the subsequence. Subset may be different depending on
            implementation (ie top-odwn recusrive vs bottom-up iterative)

    References:
        http://codereview.stackexchange.com/questions/20569/dynamic-programming-solution-to-knapsack-problem
        http://stackoverflow.com/questions/141779/solving-the-np-complete-problem-in-xkcd
        http://www.es.ele.tue.nl/education/5MC10/Solutions/knapsack.pdf

    CommandLine:
        python -m utool.util_alg --test-knapsack

        python -m utool.util_alg --test-knapsack:0
        python -m utool.util_alg --exec-knapsack:1

    Ignore:
        annots_per_view = 2
        maxweight = 2
        items = [
            (0.7005208343554686, 0.7005208343554686, 0),
            (0.669270834329427, 0.669270834329427, 1),
            (0.669270834329427, 0.669270834329427, 2),
            (0.7005208343554686, 0.7005208343554686, 3),
            (0.7005208343554686, 0.7005208343554686, 4),
            (0.669270834329427, 0.669270834329427, 5),
            (0.669270834329427, 0.669270834329427, 6),
            (0.669270834329427, 0.669270834329427, 7),
            (0.669270834329427, 0.669270834329427, 8),
            (0.669270834329427, 0.669270834329427, 9),
            (0.669270834329427, 0.669270834329427, 10),
            (0.669270834329427, 0.669270834329427, 11),
            (0.669270834329427, 0.669270834329427, 12),
            (0.669270834329427, 0.669270834329427, 13),
            (0.669270834329427, 0.669270834329427, 14),
            (0.669270834329427, 0.669270834329427, 15),
            (0.669270834329427, 0.669270834329427, 16),
            (0.669270834329427, 0.669270834329427, 17),
            (0.7005208343554686, 0.7005208343554686, 18),
            (0.7005208343554686, 0.7005208343554686, 19),
            (0.669270834329427, 0.669270834329427, 20),
            (0.7005208343554686, 0.7005208343554686, 21),
            (0.669270834329427, 0.669270834329427, 22),
            (0.669270834329427, 0.669270834329427, 23),
            (0.669270834329427, 0.669270834329427, 24),
            (0.669270834329427, 0.669270834329427, 25),
            (0.669270834329427, 0.669270834329427, 26),
            (0.669270834329427, 0.669270834329427, 27),
            (0.669270834329427, 0.669270834329427, 28),
            (0.7005208343554686, 0.7005208343554686, 29),
            (0.669270834329427, 0.669270834329427, 30),
            (0.669270834329427, 0.669270834329427, 31),
            (0.669270834329427, 0.669270834329427, 32),
            (0.669270834329427, 0.669270834329427, 33),
            (0.7005208343554686, 0.7005208343554686, 34),
            (0.669270834329427, 0.669270834329427, 35),
            (0.669270834329427, 0.669270834329427, 36),
            (0.669270834329427, 0.669270834329427, 37),
            (0.7005208343554686, 0.7005208343554686, 38),
            (0.669270834329427, 0.669270834329427, 39),
            (0.669270834329427, 0.669270834329427, 40),
            (0.7005208343554686, 0.7005208343554686, 41),
            (0.669270834329427, 0.669270834329427, 42),
            (0.669270834329427, 0.669270834329427, 43),
            (0.669270834329427, 0.669270834329427, 44),
        ]
        values = ut.take_column(items, 0)
        weights = ut.take_column(items, 1)
        indices = ut.take_column(items, 2)

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> items = [(4, 12, 0), (2, 1, 1), (6, 4, 2), (1, 1, 3), (2, 2, 4)]
        >>> maxweight = 15
        >>> total_value, items_subset = knapsack(items, maxweight, method='recursive')
        >>> total_value1, items_subset1 = knapsack(items, maxweight, method='iterative')
        >>> result =  'total_value = %.2f\n' % (total_value,)
        >>> result += 'items_subset = %r' % (items_subset,)
        >>> ut.assert_eq(total_value1, total_value)
        >>> ut.assert_eq(items_subset1, items_subset)
        >>> print(result)
        total_value = 11.00
        items_subset = [(2, 1, 1), (6, 4, 2), (1, 1, 3), (2, 2, 4)]

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> # Solve https://xkcd.com/287/
        >>> weights = [2.15, 2.75, 3.35, 3.55, 4.2, 5.8] * 2
        >>> items = [(w, w, i) for i, w in enumerate(weights)]
        >>> maxweight = 15.05
        >>> total_value, items_subset = knapsack(items, maxweight, method='recursive')
        >>> total_value1, items_subset1 = knapsack(items, maxweight, method='iterative')
        >>> total_weight = sum([t[1] for t in items_subset])
        >>> print('total_weight = %r' % (total_weight,))
        >>> result =  'total_value = %.2f' % (total_value,)
        >>> print('items_subset = %r' % (items_subset,))
        >>> print('items_subset1 = %r' % (items_subset1,))
        >>> #assert items_subset1 == items_subset, 'NOT EQ\n%r !=\n%r' % (items_subset1, items_subset)
        >>> print(result)
        total_value = 15.05

    Benchmark:
        >>> # xdoctest: +REQUIRES(module:pulp)
        >>> import utool as ut
        >>> setup = ut.codeblock(
        >>>     '''
                import utool as ut
                weights = [215, 275, 335, 355, 42, 58] * 40
                items = [(w, w, i) for i, w in enumerate(weights)]
                maxweight = 2505
                #import numba
                #knapsack_numba = numba.autojit(ut.knapsack_iterative)
                #knapsack_numba = numba.autojit(ut.knapsack_iterative_numpy)
                ''')
        >>> # Test load time
        >>> stmt_list1 = ut.codeblock(
        >>>     '''
                #ut.knapsack_recursive(items, maxweight)
                ut.knapsack_iterative(items, maxweight)
                ut.knapsack_ilp(items, maxweight)
                #knapsack_numba(items, maxweight)
                #ut.knapsack_iterative_numpy(items, maxweight)
                ''').split('\n')
        >>> ut.util_dev.timeit_compare(stmt_list1, setup, int(5))
    """
    if method == 'recursive':
        return knapsack_recursive(items, maxweight)
    elif method == 'iterative':
        return knapsack_iterative(items, maxweight)
    elif method == 'ilp':
        return knapsack_ilp(items, maxweight)
    else:
        raise NotImplementedError('[util_alg] knapsack method=%r' % (method,))
        #return knapsack_iterative_numpy(items, maxweight)


def knapsack_ilp(items, maxweight, verbose=False):
    """
    solves knapsack using an integer linear program

    CommandLine:
        python -m utool.util_alg knapsack_ilp

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> # Solve https://xkcd.com/287/
        >>> weights = [2.15, 2.75, 3.35, 3.55, 4.2, 5.8, 6.55]
        >>> values  = [2.15, 2.75, 3.35, 3.55, 4.2, 5.8, 6.55]
        >>> indices = ['mixed fruit', 'french fries', 'side salad',
        >>>            'hot wings', 'mozzarella sticks', 'sampler plate',
        >>>            'barbecue']
        >>> items = [(v, w, i) for v, w, i in zip(values, weights, indices)]
        >>> #items += [(3.95, 3.95, 'mystery plate')]
        >>> maxweight = 15.05
        >>> verbose = True
        >>> total_value, items_subset = knapsack_ilp(items, maxweight, verbose)
        >>> print('items_subset = %s' % (ut.repr3(items_subset, nl=1),))
    """
    import pulp
    # Given Input
    values  = [t[0] for t in items]
    weights = [t[1] for t in items]
    indices = [t[2] for t in items]
    # Formulate integer program
    prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)
    # Solution variables
    x = pulp.LpVariable.dicts(name='x', indexs=indices,
                              lowBound=0, upBound=1, cat=pulp.LpInteger)
    # maximize objective function
    prob.objective = sum(v * x[i] for v, i in zip(values, indices))
    # subject to
    prob.add(sum(w * x[i] for w, i in zip(weights, indices)) <= maxweight)
    # Solve using with solver like CPLEX, GLPK, or SCIP.
    #pulp.CPLEX().solve(prob)
    pulp.PULP_CBC_CMD().solve(prob)
    # Read solution
    flags = [x[i].varValue for i in indices]
    total_value = sum([val for val, flag in zip(values, flags) if flag])
    items_subset = [item for item, flag in zip(items, flags) if flag]
    # Print summary
    if verbose:
        print(prob)
        print('OPT:')
        print('\n'.join(['    %s = %s' % (x[i].name, x[i].varValue) for i in indices]))
        print('total_value = %r' % (total_value,))
    return total_value, items_subset


def knapsack_recursive(items, maxweight):
    @util_decor.memoize_nonzero
    def bestvalue(i, j):
        """ Return the value of the most valuable subsequence of the first i
        elements in items whose weights sum to no more than j. """
        if i == 0:
            return 0
        value, weight = items[i - 1][0:2]
        if weight > j:
            return bestvalue(i - 1, j)
        else:
            return max(bestvalue(i - 1, j),
                       bestvalue(i - 1, j - weight) + value)

    j = maxweight
    items_subset = []
    for i in range(len(items), 0, -1):
        if bestvalue(i, j) != bestvalue(i - 1, j):
            items_subset.append(items[i - 1])
            j -= items[i - 1][1]
    items_subset.reverse()
    total_value = bestvalue(len(items), maxweight)
    return total_value, items_subset


def number_of_decimals(num):
    r"""
    Args:
        num (float):

    References:
        stackoverflow.com/questions/6189956/finding-decimal-places

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> num = 15.05
        >>> result = number_of_decimals(num)
        >>> print(result)
        2
    """
    exp = decimal.Decimal(str(num)).as_tuple().exponent
    return max(0, -exp)


def knapsack_iterative(items, maxweight):
    # Knapsack requires integral weights
    weights = [t[1] for t in items]
    max_exp = max([number_of_decimals(w_) for w_ in weights])
    coeff = 10 ** max_exp
    # Adjust weights to be integral
    int_maxweight = int(maxweight * coeff)
    int_items = [(v, int(w * coeff), idx) for v, w, idx in items]
    """
    items = int_items
    maxweight = int_maxweight
    """
    return knapsack_iterative_int(int_items, int_maxweight)


def knapsack_iterative_int(items, maxweight):
    r"""
    Iterative knapsack method

    Math:
        maximize \sum_{i \in T} v_i
        subject to \sum_{i \in T} w_i \leq W

    Notes:
        dpmat is the dynamic programming memoization matrix.
        dpmat[i, w] is the total value of the items with weight at most W
        T is idx_subset, the set of indicies in the optimal solution

    CommandLine:
        python -m utool.util_alg --exec-knapsack_iterative_int --show

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> weights = [1, 3, 3, 5, 2, 1] * 2
        >>> items = [(w, w, i) for i, w in enumerate(weights)]
        >>> maxweight = 10
        >>> items = [(.8, 700, 0)]
        >>> maxweight = 2000
        >>> print('maxweight = %r' % (maxweight,))
        >>> print('items = %r' % (items,))
        >>> total_value, items_subset = knapsack_iterative_int(items, maxweight)
        >>> total_weight = sum([t[1] for t in items_subset])
        >>> print('total_weight = %r' % (total_weight,))
        >>> print('items_subset = %r' % (items_subset,))
        >>> result =  'total_value = %.2f' % (total_value,)
        >>> print(result)
        total_value = 0.80

    Ignore:
        DPMAT = [[dpmat[r][c] for c in range(maxweight)] for r in range(len(items))]
        KMAT  = [[kmat[r][c] for c in range(maxweight)] for r in range(len(items))]
    """
    values  = [t[0] for t in items]
    weights = [t[1] for t in items]
    maxsize = maxweight + 1
    # Sparse representation seems better
    dpmat = defaultdict(lambda: defaultdict(lambda: np.inf))
    kmat = defaultdict(lambda: defaultdict(lambda: False))
    idx_subset = []  # NOQA
    for w in range(maxsize):
        dpmat[0][w] = 0
    # For each item consider to include it or not
    for idx in range(len(items)):
        item_val = values[idx]
        item_weight = weights[idx]
        # consider at each possible bag size
        for w in range(maxsize):
            valid_item = item_weight <= w
            if idx > 0:
                prev_val = dpmat[idx - 1][w]
                prev_noitem_val = dpmat[idx - 1][w - item_weight]
            else:
                prev_val = 0
                prev_noitem_val = 0
            withitem_val = item_val + prev_noitem_val
            more_valuable = withitem_val > prev_val
            if valid_item and more_valuable:
                dpmat[idx][w] = withitem_val
                kmat[idx][w] = True
            else:
                dpmat[idx][w] = prev_val
                kmat[idx][w] = False
    # Trace backwards to get the items used in the solution
    K = maxweight
    for idx in reversed(range(len(items))):
        if kmat[idx][K]:
            idx_subset.append(idx)
            K = K - weights[idx]
    idx_subset = sorted(idx_subset)
    items_subset = [items[i] for i in idx_subset]
    total_value = dpmat[len(items) - 1][maxweight]
    return total_value, items_subset


def knapsack_iterative_numpy(items, maxweight):
    r"""
    Iterative knapsack method

    maximize \sum_{i \in T} v_i
    subject to \sum_{i \in T} w_i \leq W

    Notes:
        dpmat is the dynamic programming memoization matrix.
        dpmat[i, w] is the total value of the items with weight at most W
        T is the set of indicies in the optimal solution
    """
    #import numpy as np
    items = np.array(items)
    weights = items.T[1]
    # Find maximum decimal place (this problem is in NP)
    max_exp = max([number_of_decimals(w_) for w_ in weights])
    coeff = 10 ** max_exp
    # Adjust weights to be integral
    weights = (weights * coeff).astype(int)
    values  = items.T[0]
    MAXWEIGHT = int(maxweight * coeff)
    W_SIZE = MAXWEIGHT + 1

    dpmat = np.full((len(items), W_SIZE), np.inf)
    kmat = np.full((len(items), W_SIZE), 0, dtype=bool)
    idx_subset = []

    for w in range(W_SIZE):
        dpmat[0][w] = 0
    for idx in range(1, len(items)):
        item_val = values[idx]
        item_weight = weights[idx]
        for w in range(W_SIZE):
            valid_item = item_weight <= w
            prev_val = dpmat[idx - 1][w]
            if valid_item:
                prev_noitem_val = dpmat[idx - 1][w - item_weight]
                withitem_val = item_val + prev_noitem_val
                more_valuable = withitem_val > prev_val
            else:
                more_valuable = False
            dpmat[idx][w] = withitem_val if more_valuable else prev_val
            kmat[idx][w] = more_valuable
    K = MAXWEIGHT
    for idx in reversed(range(1, len(items))):
        if kmat[idx, K]:
            idx_subset.append(idx)
            K = K - weights[idx]
    idx_subset = sorted(idx_subset)
    items_subset = [items[i] for i in idx_subset]
    total_value = dpmat[len(items) - 1][MAXWEIGHT]
    return total_value, items_subset


#def knapsack_all_solns(items, maxweight):
#    """
#    TODO: return all optimal solutions to the knapsack problem

#    References:
#        stackoverflow.com/questions/30554290/all-solutions-from-knapsack-dp-matrix

#    >>> items = [(1, 2, 0), (1, 3, 1), (1, 4, 2), (1, 3, 3), (1, 3, 4), (1, 5, 5), (1, 4, 6), (1, 1, 7), (1, 1, 8), (1, 3, 9)]
#    >>> weights = ut.get_list_column(items, 1)
#    >>> maxweight = 6
#    """


def knapsack_greedy(items, maxweight):
    r"""
    non-optimal greedy version of knapsack algorithm
    does not sort input. Sort the input by largest value
    first if desired.

    Args:
        `items` (tuple): is a sequence of tuples `(value, weight, id_)`, where `value`
            is a scalar and `weight` is a non-negative integer, and `id_` is an
            item identifier.

        `maxweight` (scalar):  is a non-negative integer.

    CommandLine:
        python -m utool.util_alg --exec-knapsack_greedy

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> items = [(4, 12, 0), (2, 1, 1), (6, 4, 2), (1, 1, 3), (2, 2, 4)]
        >>> maxweight = 15
        >>> total_value, items_subset = knapsack_greedy(items, maxweight)
        >>> result =  'total_value = %r\n' % (total_value,)
        >>> result += 'items_subset = %r' % (items_subset,)
        >>> print(result)
        total_value = 7
        items_subset = [(4, 12, 0), (2, 1, 1), (1, 1, 3)]
    """
    items_subset = []
    total_weight = 0
    total_value = 0
    for item in items:
        value, weight = item[0:2]
        if total_weight + weight > maxweight:
            continue
        else:
            items_subset.append(item)
            total_weight += weight
            total_value += value
    return total_value, items_subset


def prod(item_list, initial=1.0):
    """
    product of all number in a list (like np.prod)

    Args:
        item_list (list): list of numbers or items supporting mulitplicatiuon
        initial (value): initial identity (default=1)

    Returns:
        float: Multiplied value

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> initial = 1.0
        >>> item_list = [1, 2, 3, 4, 5]
        >>> assert prod(item_list, initial) == 120.0
        >>> assert prod([]) == 1.0
        >>> assert prod([5]) == 5.0
    """
    return reduce(op.mul, item_list, initial)


def cumsum(item_list, initial=0):
    """ python cumsum

    Args:
        item_list (list): list of numbers or items supporting addition
        initial (value): initial zero value

    Returns:
        list: list of accumulated values

    References:
        stackoverflow.com/questions/9258602/elegant-pythonic-cumsum

    CommandLine:
        python -m utool.util_alg cumsum

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> item_list = [1, 2, 3, 4, 5]
        >>> initial = 0
        >>> result = cumsum(item_list, initial)
        >>> assert result == [1, 3, 6, 10, 15]
        >>> print(result)
        >>> item_list = zip([1, 2, 3, 4, 5])
        >>> initial = tuple()
        >>> result2 = cumsum(item_list, initial)
        >>> assert result2 == [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4, 5)]
        >>> print(result2)
    """
    def accum(acc, itm):
        return op.iadd(acc, [acc[-1] + itm])
    return reduce(accum, item_list, [initial])[1:]


def safe_div(a, b):
    return None if a is None or b is None else a / b


def choose(n, k):
    """
    N choose k

    binomial combination (without replacement)
    scipy.special.binom
    """
    import scipy  # NOQA
    try:
        from scipy.special import comb
    except Exception:
        from scipy.misc import comb
    return comb(n, k, exact=True, repetition=False)


def triangular_number(n):
    r"""
    Latex:
        T_n = \sum_{k=1}^{n} k = \frac{n (n + 1)}{2} = \binom{n + 1}{2}

    References:
        en.wikipedia.org/wiki/Triangular_number
    """
    return ((n * (n + 1)) / 2)


# Functions using NUMPY / SCIPY (need to make python only or move to vtool)


def maximin_distance_subset1d(items, K=None, min_thresh=None, verbose=False):
    r"""
    Greedy algorithm, may be exact for 1d case.
    First, choose the first item, then choose the next item that is farthest
    away from all previously chosen items. Iterate.

    CommandLine:
        python -m utool.util_alg --exec-maximin_distance_subset1d

    Notes:
        Given a set of items V.
        Let $E = V \times V$ be the the set of all item pairs.

        The goal is to return the largest subset of item such that the distance
        between any pair of items in the subset is greater than some threshold.

        Let t[u, v] be the distance between u and v.

        Let x[u, v] = 1 if the annotation pair (u, v) is included.

        Let y[u] = 1 if the annotation u is included.

        Objective:
            maximize sum(y[u] for u in V)

        subject to:
            # Annotations pairs are only included if their timedelta is less than
            # the threshold.
            x[u, v] = 0 if t[u, v] > thresh

            # If an edge is exclued at least one of its endpoints must be
            # excluded
            y[u] + y[v] - x[u, v] < 2


    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> from utool.util_alg import *  # NOQA
        >>> #items = [1, 2, 3, 4, 5, 6, 7]
        >>> items = [20, 1, 1, 9, 21, 6, 22]
        >>> min_thresh = 5
        >>> K = None
        >>> result = maximin_distance_subset1d(items, K, min_thresh, verbose=True)
        >>> print(result)
        (array([1, 3, 6]), [1, 9, 22])

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> from utool.util_alg import *  # NOQA
        >>> #items = [1, 2, 3, 4, 5, 6, 7]
        >>> items = [0, 1]
        >>> min_thresh = 5
        >>> K = None
        >>> result = maximin_distance_subset1d(items, K, min_thresh, verbose=True)
        >>> print(result)
    """
    if False:
        import pulp
        # Formulate integer program
        prob = pulp.LpProblem("MaxSizeLargeDistSubset", pulp.LpMaximize)
        # Solution variable indicates if set it chosen or not
        item_indices = list(range(len(items)))
        pair_indices = list(ut.combinations(item_indices, 2))
        x = pulp.LpVariable.dicts(name='x', indexs=pair_indices,
                                  lowBound=0, upBound=1, cat=pulp.LpInteger)
        y = pulp.LpVariable.dicts(name='y', indexs=item_indices,
                                  lowBound=0, upBound=1, cat=pulp.LpInteger)
        # minimize the number of sets
        prob.objective = sum(y[i] for i in item_indices)

        # subject to
        count = 0
        for u, v in pair_indices:
            # Minimum thresh constraint
            if abs(items[u] - items[v]) < min_thresh:
                prob.add(x[(u, v)] == 0, name='thresh_%r' % (count,))
                count += 1

        count = 0
        for u, v in pair_indices:
            prob.add(y[u] + y[v] - x[(u, v)] <= 1, 'exclusion_%r' % (count,))
            count += 1

        pulp.PULP_CBC_CMD().solve(prob)
        # Read solution
        flags = [y[i].varValue >= 1.0 for i in item_indices]
        chosen_items_idxs = ut.where(flags)
        chosen_items = ut.take(items, chosen_items_idxs)

        # total_value = sum([val for val, flag in zip(values, flags) if flag])
        # items_subset = [item for item, flag in zip(items, flags) if flag]
        # each element is covered
        # containing_sets = [i for i in set_indices if e in candidate_sets_dict[i]]
        # prob.add(sum(x[i] for i in containing_sets) >= 1)

    import utool as ut
    try:
        import vtool_ibeis as vt
    except ImportError:
        import vtool as vt
    points = np.array(items)[:, None]
    # Initial sorting of 1d points
    initial_sortx = points.argsort(axis=0).flatten()
    points = points.take(initial_sortx, axis=0)

    if K is None:
        K = len(items)

    def distfunc(x, y):
        return np.abs(x - y)

    assert points.shape[1] == 1
    assert len(points) >= K, 'cannot return subset'
    if K == 1:
        current_idx = [0]
    else:
        current_idx = [0, -1]
        if min_thresh is not None and distfunc(points[0], points[-1])[0] < min_thresh:
            current_idx = [0]
    chosen_mask = vt.index_to_boolmask(current_idx, len(points))

    for k in range(2, K):
        unchosen_idx = np.nonzero(~chosen_mask)[0]
        unchosen_items = points.compress(~chosen_mask, axis=0)
        chosen_items = points.compress(chosen_mask, axis=0)
        distances = distfunc(unchosen_items, chosen_items.T)
        min_distances = distances.min(axis=1)
        argx = min_distances.argmax()
        if min_thresh is not None:
            if min_distances[argx] < min_thresh:
                break
        new_idx = unchosen_idx[argx]
        chosen_mask[new_idx] = True

    # Put chosen mask back in the input order of items
    chosen_items_mask = chosen_mask.take(initial_sortx.argsort())
    chosen_items_idxs = np.nonzero(chosen_items_mask)[0]
    chosen_items = ut.take(items, chosen_items_idxs)
    #current_idx = np.nonzero(chosen_mask)[0]
    if verbose:
        print('Chose subset')
        chosen_points = points.compress(chosen_mask, axis=0)
        distances = (spdist.pdist(chosen_points, distfunc))
        print('chosen_items_idxs = %r' % (chosen_items_idxs,))
        print('chosen_items = %r' % (chosen_items,))
        print('distances = %r' % (distances,))
    return chosen_items_idxs, chosen_items


def maximum_distance_subset(items, K, verbose=False):
    """
    FIXME: I believe this does not work.

    Returns a subset of size K from items with the maximum pairwise distance

    References:
        stackoverflow.com/questions/12278528/subset-elements-furthest-apart-eachother
        stackoverflow.com/questions/13079563/condensed-distance-matrix-pdist

    Recurance:
        Let X[n,k] be the solution for selecting k elements from first n elements items.
        X[n, k] = max( max( X[m, k - 1] + (sum_{p in prev_solution} dist(o, p)) for o < n and o not in prev solution) ) for m < n.

    Example:
        >>> # DISABLE_DOCTEST
        >>> import scipy.spatial.distance as spdist
        >>> items = [1, 6, 20, 21, 22]

    CommandLine:
        python -m utool.util_alg --exec-maximum_distance_subset

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> #items = [1, 2, 3, 4, 5, 6, 7]
        >>> items = [1, 6, 20, 21, 22]
        >>> K = 3
        >>> result = maximum_distance_subset(items, K)
        >>> print(result)
        (42.0, array([4, 3, 0]), array([22, 21,  1]))
    """
    from utool import util_decor
    if verbose:
        print('maximum_distance_subset len(items)=%r, K=%r' % (len(items), K,))

    points = np.array(items)[:, None]

    if False:
        # alternative definition (not sure if works)
        distmat = spdist.squareform(spdist.pdist(points, lambda x, y: np.abs(x - y)))
        D = np.triu(distmat)
        remaining_idxs = np.arange(len(D))
        for count in range(len(points) - K):
            values = D.sum(axis=1) + D.sum(axis=0)
            remove_idx = values.argmin()  # index with minimum pairwise distance
            remaining_idxs = np.delete(remaining_idxs, remove_idx)
            D = np.delete(np.delete(D, remove_idx, axis=0), remove_idx, axis=1)
        value = D.sum()
        subset_idx = remaining_idxs.tolist()
        value, subset_idx
        subset = points.take(subset_idx)
        #print((value, subset_idx, subset))

    sortx = points.T[0].argsort()[::-1]
    sorted_points = points.take(sortx, axis=0)
    pairwise_distance = spdist.pdist(sorted_points, lambda x, y: np.abs(x - y))
    distmat = (spdist.squareform(pairwise_distance))

    def condensed_idx(i, j):
        if i >= len(sorted_points) or j >= len(sorted_points):
            raise IndexError('i=%r j=%r out of range' % (i, j))
        elif i == j:
            return None
        elif j < i:
            i, j = j, i
        return (i * len(sorted_points) + j) - (i * (i + 1) // 2) - (i) - (1)

    def dist(i, j):
        idx = condensed_idx(i, j)
        return 0 if idx is None else pairwise_distance[idx]

    @util_decor.memoize_nonzero
    def optimal_solution(n, k):
        """
        Givem sorted items sorted_points
        Pick subset_idx of size k from sorted_points[:n] with maximum pairwise distance
        Dynamic programming solution
        """
        "# FIXME BROKEN "
        assert n <= len(sorted_points) and k <= len(sorted_points)
        if k < 2 or n < 2 or n < k:
            # BASE CASE
            value, subset_idx =  0, []
        elif k == 2:
            # BASE CASE
            # when k==2 we choose the maximum pairwise pair
            subdist = np.triu(distmat[0:n, 0:n])
            maxpos = subdist.argmax()
            ix, jx = np.unravel_index(maxpos, subdist.shape)
            value = distmat[ix, jx]
            subset_idx = [ix, jx]
        else:
            # RECURSIVE CASE
            value = 0
            subset_idx = None
            # MAX OVER ALL OTHER NODES (might not need a full on loop here, but this will definitely work)
            for m in range(k - 1, n):
                # Choose which point to add would maximize the distance with the previous best answer.
                prev_value, prev_subset_idx = optimal_solution(m, k - 1)
                for o in range(n):
                    if o in prev_subset_idx:
                        continue
                    add_value = sum([distmat[o, px] for px in prev_subset_idx])
                    cur_value = prev_value + add_value
                    if cur_value > value:
                        value = cur_value
                        subset_idx = prev_subset_idx + [o]
        return value, subset_idx

    value, sorted_subset_idx = optimal_solution(len(points), K)
    subset_idx = sortx.take(sorted_subset_idx)
    subset = points.take(subset_idx)
    #print((value, subset_idx, subset))
    return value, subset_idx, subset
    #np.array([[dist(i, k) if k < i else 0 for k in range(len(A))] for i in range(len(A))])
    #raise NotImplementedError('unfinished')


#def safe_max(arr):
#    return np.nan if arr is None or len(arr) == 0 else arr.max()


#def safe_min(arr):
#    return np.nan if arr is None or len(arr) == 0 else arr.min()


def deg_to_rad(degree):
    degree %= 360.0
    return (degree / 360.0) * TAU


def rad_to_deg(radians):
    radians %= TAU
    return (radians / TAU) * 360.0


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
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> num = np.array([[ 0.   ,  0.431,  0.279],
        ...                 [ 0.204,  0.352,  0.08 ],
        ...                 [ 0.107,  0.325,  0.179]])
        >>> low  = .1
        >>> high = .4
        >>> eq = False
        >>> is_inbounds = inbounds(num, low, high, eq)
        >>> result = ut.repr2(is_inbounds, with_dtype=True)
        >>> print(result)
        np.array([[False, False,  True],
                  [ True,  True, False],
                  [ True,  True,  True]], dtype=bool)

    """
    less    = op.le if eq else op.lt
    greater = op.ge if eq else op.gt
    and_ = np.logical_and if isinstance(num, np.ndarray) else op.and_
    is_inbounds = and_(greater(num, low), less(num, high))
    return is_inbounds


def almost_eq(arr1, arr2, thresh=1E-11, ret_error=False):
    """ checks if floating point number are equal to a threshold
    """
    error = np.abs(arr1 - arr2)
    passed = error < thresh
    if ret_error:
        return passed, error
    return passed


def almost_allsame(vals):
    if len(vals) == 0:
        return True
    x = vals[0]
    return np.all([np.isclose(item, x) for item in vals])


def unixtime_hourdiff(x, y):
    r"""
    Args:
        x (?):
        y (ndarray):  labels

    Returns:
        ?:

    CommandLine:
        python -m utool.util_alg --exec-unixtime_hourdiff --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> x = np.nan
        >>> y = 0
        >>> result = unixtime_hourdiff(x, y)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()
    """
    return np.abs((x - y)) / (60. ** 2)


def absdiff(x, y):
    return np.abs(np.subtract(x, y))


def safe_pdist(arr, *args, **kwargs):
    """
    Kwargs:
        metric = ut.absdiff
    SeeAlso:
        scipy.spatial.distance.pdist

    TODO: move to vtool

    """
    if arr is None or len(arr) < 2:
        return None
    else:
        try:
            import vtool_ibeis as vt
        except ImportError:
            import vtool as vt

        arr_ = vt.atleast_nd(arr, 2)
        return spdist.pdist(arr_, *args, **kwargs)


def square_pdist(arr, *args, **kwargs):
    dists = safe_pdist(arr, *args, **kwargs)
    if dists is None:
        return np.zeros((1, 1))
    return spdist.squareform(dists)


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
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> array = np.array([ 22, 1, 3, 2, 10, 42, ])
        >>> dim = None
        >>> array_norm = norm_zero_one(array, dim)
        >>> result = ut.repr2(list(array_norm), precision=3)
        >>> print(result)
        [0.512, 0.000, 0.049, 0.024, 0.220, 1.000]
    """
    if not util_type.is_float(array):
        array = array.astype(np.float32)
    array_max  = array.max(dim)
    array_min  = array.min(dim)
    array_exnt = np.subtract(array_max, array_min)
    array_norm = np.divide(np.subtract(array, array_min), array_exnt)
    return array_norm


def euclidean_dist(vecs1, vec2, dtype=None):
    if dtype is None:
        dtype = np.float32
    return np.sqrt(((vecs1.astype(dtype) - vec2.astype(dtype)) ** 2).sum(1))


def max_size_max_distance_subset(items, min_thresh=0, Kstart=2, verbose=False):
    r"""
    Args:
        items (?):
        min_thresh (int): (default = 0)
        Kstart (int): (default = 2)

    Returns:
        ?: prev_subset_idx

    CommandLine:
        python -m utool.util_alg --exec-max_size_max_distance_subset

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> items = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> min_thresh = 3
        >>> Kstart = 2
        >>> verbose = True
        >>> prev_subset_idx = max_size_max_distance_subset(items, min_thresh,
        >>>                                                Kstart, verbose=verbose)
        >>> result = ('prev_subset_idx = %s' % (str(prev_subset_idx),))
        >>> print(result)
    """
    import utool as ut
    assert Kstart >= 2, 'must start with group of size 2'
    best_idxs = []
    for K in range(Kstart, len(items)):
        if verbose:
            print('Running subset chooser')
        value, subset_idx, subset = ut.maximum_distance_subset(items, K=K,
                                                               verbose=verbose)
        if verbose:
            print('subset = %r' % (subset,))
            print('subset_idx = %r' % (subset_idx,))
            print('value = %r' % (value,))
        distances = ut.safe_pdist(subset[:, None])
        if np.any(distances < min_thresh):
            break
        best_idxs = subset_idx
    return best_idxs


def group_indices(groupid_list):
    """
    groups indicies of each item in ``groupid_list``

    Args:
        groupid_list (list): list of group ids

    SeeAlso:
        vt.group_indices - optimized numpy version
        ut.apply_grouping

    CommandLine:
        python -m utool.util_alg --test-group_indices
        python3 -m utool.util_alg --test-group_indices

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> groupid_list = ['b', 1, 'b', 1, 'b', 1, 'b', 'c', 'c', 'c', 'c']
        >>> (keys, groupxs) = ut.group_indices(groupid_list)
        >>> result = ut.repr3((keys, groupxs), nobraces=1, nl=1)
        >>> print(result)
        [1, 'b', 'c'],
        [[1, 3, 5], [0, 2, 4, 6], [7, 8, 9, 10]],
    """
    item_list = range(len(groupid_list))
    grouped_dict = util_dict.group_items(item_list, groupid_list)
    # Sort by groupid for cache efficiency
    keys_ = list(grouped_dict.keys())
    try:
        keys = sorted(keys_)
    except TypeError:
        # Python 3 does not allow sorting mixed types
        keys = util_list.sortedby2(keys_, keys_)
    groupxs = util_dict.dict_take(grouped_dict, keys)
    return keys, groupxs


def apply_grouping(items, groupxs):
    r"""
    applies grouping from group_indicies
    non-optimized version

    Args:
        items (list): items to group
        groupxs (list of list of ints): grouped lists of indicies

    SeeAlso:
        vt.apply_grouping - optimized numpy version
        ut.group_indices

    CommandLine:
        python -m utool.util_alg --exec-apply_grouping --show

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> idx2_groupid = [2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3]
        >>> items        = [1, 8, 5, 5, 8, 6, 7, 5, 3, 0, 9]
        >>> (keys, groupxs) = ut.group_indices(idx2_groupid)
        >>> grouped_items = ut.apply_grouping(items, groupxs)
        >>> result = ut.repr2(grouped_items)
        >>> print(result)
        [[8, 5, 6], [1, 5, 8, 7], [5, 3, 0, 9]]
    """
    return [util_list.list_take(items, xs) for xs in groupxs]


def iapply_grouping(items, groupxs):
    r"""
    Iterates over groups from group_indicies

    Args:
        items (list): items to group
        groupxs (list of list of ints): grouped lists of indicies

    SeeAlso:
        vt.apply_grouping - optimized numpy version
        ut.group_indices

    CommandLine:
        python -m utool.util_alg --exec-apply_grouping --show

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> idx2_groupid = [2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3]
        >>> items        = [1, 8, 5, 5, 8, 6, 7, 5, 3, 0, 9]
        >>> (keys, groupxs) = ut.group_indices(idx2_groupid)
        >>> grouped_items = list(ut.iapply_grouping(items, groupxs))
        >>> result = ut.repr2(grouped_items)
        >>> print(result)
        [[8, 5, 6], [1, 5, 8, 7], [5, 3, 0, 9]]
    """
    for xs in groupxs:
        yield [items[x] for x in xs]


def ungroup(grouped_items, groupxs, maxval=None, fill=None):
    """
    Ungroups items

    Args:
        grouped_items (list):
        groupxs (list):
        maxval (int): (default = None)

    Returns:
        list: ungrouped_items

    SeeAlso:
        vt.invert_apply_grouping

    CommandLine:
        python -m utool.util_alg ungroup_unique

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> grouped_items = [[1.1, 1.2], [2.1, 2.2], [3.1, 3.2]]
        >>> groupxs = [[0, 2], [1, 5], [4, 3]]
        >>> maxval = None
        >>> ungrouped_items = ungroup(grouped_items, groupxs, maxval)
        >>> result = ('ungrouped_items = %s' % (ut.repr2(ungrouped_items),))
        >>> print(result)
        ungrouped_items = [1.1, 2.1, 1.2, 3.2, 3.1, 2.2]
    """
    if maxval is None:
        # Determine the number of items if unknown
        maxpergroup = [max(xs) if len(xs) else 0 for xs in groupxs]
        maxval = max(maxpergroup) if len(maxpergroup) else 0
    # Allocate an array containing the newly flattened items
    ungrouped_items = [fill] * (maxval + 1)
    # Populate the array
    for itemgroup, xs in zip(grouped_items, groupxs):
        for item, x in zip(itemgroup, xs):
            ungrouped_items[x] = item
    return ungrouped_items


def ungroup_gen(grouped_items, groupxs, fill=None):
    """
    Ungroups items returning a generator.
    Note that this is much slower than the list version and is not gaurenteed
    to have better memory usage.

    Args:
        grouped_items (list):
        groupxs (list):
        maxval (int): (default = None)

    Returns:
        list: ungrouped_items

    SeeAlso:
        vt.invert_apply_grouping

    CommandLine:
        python -m utool.util_alg ungroup_unique

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> grouped_items = [[1.1, 1.2], [2.1, 2.2], [3.1, 3.2]]
        >>> groupxs = [[1, 2], [5, 6], [9, 3]]
        >>> ungrouped_items1 = list(ungroup_gen(grouped_items, groupxs))
        >>> ungrouped_items2 = ungroup(grouped_items, groupxs)
        >>> assert ungrouped_items1 == ungrouped_items2
        >>> grouped_items = [[1.1, 1.2], [2.1, 2.2], [3.1, 3.2]]
        >>> groupxs = [[0, 2], [1, 5], [4, 3]]
        >>> ungrouped_items1 = list(ungroup_gen(grouped_items, groupxs))
        >>> ungrouped_items2 = ungroup(grouped_items, groupxs)
        >>> assert ungrouped_items1 == ungrouped_items2

    Ignore:
        labels = np.random.randint(0, 64, 10000)
        unique_labels, groupxs = ut.group_indices(labels)
        grouped_items = ut.apply_grouping(np.arange(len(labels)), groupxs)
        ungrouped_items1 = list(ungroup_gen(grouped_items, groupxs))
        ungrouped_items2 = ungroup(grouped_items, groupxs)
        assert ungrouped_items2 == ungrouped_items1
        %timeit list(ungroup_gen(grouped_items, groupxs))
        %timeit ungroup(grouped_items, groupxs)
    """
    import utool as ut
    # Determine the number of items if unknown
    #maxpergroup = [max(xs) if len(xs) else 0 for xs in groupxs]
    #maxval = max(maxpergroup) if len(maxpergroup) else 0

    minpergroup = [min(xs) if len(xs) else 0 for xs in groupxs]
    minval = min(minpergroup) if len(minpergroup) else 0

    flat_groupx = ut.flatten(groupxs)
    sortx = ut.argsort(flat_groupx)
    # Indicates the index being yeilded
    groupx_sorted = ut.take(flat_groupx, sortx)
    flat_items = ut.iflatten(grouped_items)

    # Storage for data weiting to be yeilded
    toyeild = {}
    items_yeilded = 0
    # Indicates the index we are curently yeilding
    current_index = 0

    # Determine where fills need to happen
    num_fills_before = [minval] + (np.diff(groupx_sorted) - 1).tolist() + [0]

    # Check if there are fills before the first item
    fills = num_fills_before[items_yeilded]
    if fills > 0:
        for _ in range(fills):
            yield None
            current_index += 1
    # Yield items as possible
    for yeild_at, item in zip(flat_groupx, flat_items):
        if yeild_at > current_index:
            toyeild[yeild_at] = item
        elif yeild_at == current_index:
            # When we find the next element to yeild
            yield item
            current_index += 1
            items_yeilded += 1
            # Check if there are fills before the next item
            fills = num_fills_before[items_yeilded]
            if fills > 0:
                for _ in range(fills):
                    yield None
                    current_index += 1
            # Now yield everything that came before this
            while current_index in toyeild:
                item = toyeild.pop(current_index)
                yield item
                current_index += 1
                items_yeilded += 1
                # Check if there are fills before the next item
                fills = num_fills_before[items_yeilded]
                if fills > 0:
                    for _ in range(fills):
                        yield None
                        current_index += 1


def ungroup_unique(unique_items, groupxs, maxval=None):
    """
    Ungroups unique items to correspond to original non-unique list

    Args:
        unique_items (list):
        groupxs (list):
        maxval (int): (default = None)

    Returns:
        list: ungrouped_items

    CommandLine:
        python -m utool.util_alg ungroup_unique

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> unique_items = [1, 2, 3]
        >>> groupxs = [[0, 2], [1, 3], [4, 5]]
        >>> maxval = None
        >>> ungrouped_items = ungroup_unique(unique_items, groupxs, maxval)
        >>> result = ('ungrouped_items = %s' % (ut.repr2(ungrouped_items),))
        >>> print(result)
        ungrouped_items = [1, 2, 1, 2, 3, 3]
    """
    if maxval is None:
        maxpergroup = [max(xs) if len(xs) else 0 for xs in groupxs]
        maxval = max(maxpergroup) if len(maxpergroup) else 0
    ungrouped_items = [None] * (maxval + 1)
    for item, xs in zip(unique_items, groupxs):
        for x in xs:
            ungrouped_items[x] = item
    return ungrouped_items


def edit_distance(string1, string2):
    """
    Edit distance algorithm. String1 and string2 can be either
    strings or lists of strings

    pip install python-Levenshtein

    Args:
        string1 (str or list):
        string2 (str or list):

    CommandLine:
        python -m utool.util_alg edit_distance --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> string1 = 'hello world'
        >>> string2 = ['goodbye world', 'rofl', 'hello', 'world', 'lowo']
        >>> edit_distance(['hello', 'one'], ['goodbye', 'two'])
        >>> edit_distance('hello', ['goodbye', 'two'])
        >>> edit_distance(['hello', 'one'], 'goodbye')
        >>> edit_distance('hello', 'goodbye')
        >>> distmat = edit_distance(string1, string2)
        >>> result = ('distmat = %s' % (ut.repr2(distmat),))
        >>> print(result)
        >>> [7, 9, 6, 6, 7]
    """

    import utool as ut
    try:
        import Levenshtein
    except ImportError as ex:
        ut.printex(ex, 'pip install python-Levenshtein')
        raise
    #np.vectorize(Levenshtein.distance, [int])
    #vec_lev = np.frompyfunc(Levenshtein.distance, 2, 1)
    #return vec_lev(string1, string2)
    import utool as ut
    isiter1 = ut.isiterable(string1)
    isiter2 = ut.isiterable(string2)
    strs1 = string1 if isiter1 else [string1]
    strs2 = string2 if isiter2 else [string2]
    distmat = [
        [Levenshtein.distance(str1, str2) for str2 in strs2]
        for str1 in strs1
    ]
    # broadcast
    if not isiter2:
        distmat = ut.take_column(distmat, 0)
    if not isiter1:
        distmat = distmat[0]
    return distmat


def get_nth_bell_number(n):
    """
    Returns the (num_items - 1)-th Bell number using recursion.
    The Bell numbers count the number of partitions of a set.

    Args:
        n (int): number of items in a set

    Returns:
        int:

    References:
        http://adorio-research.org/wordpress/?p=11460

    CommandLine:
        python -m utool.util_alg --exec-bell --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> n = 3
        >>> result = get_nth_bell_number(n)
        >>> print(result)
        5
    """
    import utool as ut
    import scipy.special
    @ut.memoize
    def bell_(n):
        if n < 2:
            return 1
        sum_ = 0
        for k in range(1, n + 1):
            sum_ = sum_ + scipy.special.binom(n - 1, k - 1) * bell_(k - 1)
        return sum_
    nth_bell = bell_(n)
    return nth_bell


def num_partitions(num_items):
    return get_nth_bell_number(num_items - 1)


def standardize_boolexpr(boolexpr_, parens=False):
    r"""
    Standardizes a boolean expression into an or-ing of and-ed variables

    Args:
        boolexpr_ (str):

    Returns:
        str: final_expr

    CommandLine:
        sudo pip install git+https://github.com/tpircher/quine-mccluskey.git
        python -m utool.util_alg standardize_boolexpr --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> boolexpr_ = 'not force_opencv and (orient_ or is_gif)'
        >>> result = standardize_boolexpr(boolexpr_, parens=True)
        >>> print(result)
        (orient_ and (not force_opencv)) or (is_gif and (not force_opencv))
    """
    import utool as ut
    import re
    onlyvars = boolexpr_
    onlyvars = re.sub('\\bnot\\b', '', onlyvars)
    onlyvars = re.sub('\\band\\b', '', onlyvars)
    onlyvars = re.sub('\\bor\\b', '', onlyvars)
    onlyvars = re.sub('\\(', '', onlyvars)
    onlyvars = re.sub('\\)', '', onlyvars)
    varnames = ut.remove_doublspaces(onlyvars).strip().split(' ')
    varied_dict = {var: [True, False] for var in varnames}
    bool_states = ut.all_dict_combinations(varied_dict)
    outputs = [eval(boolexpr_, state.copy(), state.copy()) for state in bool_states]
    true_states = ut.compress(bool_states, outputs)
    true_tuples = ut.take_column(true_states, varnames)
    true_cases = [str(''.join([str(int(t)) for t in tup])) for tup in true_tuples]

    # Convert to binary
    ones_bin = [int(x, 2) for x in true_cases]
    #ones_str = [str(x) for x in true_cases]
    from quine_mccluskey.qm import QuineMcCluskey
    qm = QuineMcCluskey()
    result = qm.simplify(ones=ones_bin, num_bits=len(varnames))
    #result = qm.simplify_los(ones=ones_str, num_bits=len(varnames))

    grouped_terms = [dict(ut.group_items(varnames, rs)) for rs in result]
    def parenjoin(char, list_):
        if len(list_) == 0:
            return ''
        else:
            if parens:
                return '(' + char.join(list_) + ')'
            else:
                return char.join(list_)

    if parens:
        expanded_terms = [
            (
                term.get('1', []) +
                ['(not ' + b + ')' for b in term.get('0', [])] +
                [
                    parenjoin(' ^ ', term.get('^', [])),
                    parenjoin(' ~ ', term.get('~', [])),
                ]
            ) for term in grouped_terms
        ]
    else:
        expanded_terms = [
            (
                term.get('1', []) +
                ['not ' + b  for b in term.get('0', [])] +
                [
                    parenjoin(' ^ ', term.get('^', [])),
                    parenjoin(' ~ ', term.get('~', [])),
                ]
            ) for term in grouped_terms
        ]

    final_terms = [[t for t in term if t] for term in expanded_terms]

    products = [parenjoin(' and ', [f for f in form if f]) for form in final_terms]
    final_expr = ' or '.join(products)
    return final_expr


def solve_boolexpr():
    """
    sudo pip install git+https://github.com/tpircher/quine-mccluskey.git
    sudo pip uninstall quine_mccluskey
    pip uninstall quine_mccluskey

    pip install git+https://github.com/tpircher/quine-mccluskey.git


    Args:
        varnames (?):

    Returns:
        ?:

    CommandLine:
        python -m utool.util_alg solve_boolexpr --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> varnames = ['sa', 'said', 'aid']
        >>> result = solve_boolexpr()
        >>> print(result)

    """
    #false_cases = [
    #    int('111', 2),
    #    int('011', 2),
    #    int('001', 2),
    #]
    #true_cases = list(set(range(2 ** 3)) - set(false_cases))
    varnames = ['sa', 'said', 'aid']

    #import utool as ut
    truth_table = [
        dict(sa=True,  said=True,  aid=True,  output=False),
        dict(sa=True,  said=True,  aid=False, output=True),
        dict(sa=True,  said=False, aid=True,  output=True),
        dict(sa=True,  said=False, aid=False, output=True),
        dict(sa=False, said=True,  aid=True,  output=False),
        dict(sa=False, said=True,  aid=False, output=True),
        dict(sa=False, said=False, aid=True,  output=False),
        dict(sa=False, said=False, aid=False, output=True),
    ]
    truth_tuples = [ut.dict_take(d, varnames) for d in truth_table]
    outputs = [d['output'] for d in truth_table]
    true_tuples = ut.compress(truth_tuples, outputs)
    #false_tuples = ut.compress(truth_tuples, ut.not_list(outputs))
    true_cases = [''.join([str(int(t)) for t in tup]) for tup in true_tuples]
    true_cases = [''.join([str(int(t)) for t in tup]) for tup in true_tuples]
    #truth_nums = [int(s, 2) for s in true_cases]

    from quine_mccluskey.qm import QuineMcCluskey
    qm = QuineMcCluskey(use_xor=False)
    result = qm.simplify_los(true_cases, num_bits=len(varnames))
    print(result)
    #ut.chr_range(3)

    #symbol_map = {
    #    '-': '',
    #    '1': '{v}',
    #    '0': 'not {v}',
    #    '^': '^',
    #}

    #'-' don't care: this bit can be either zero or one.
    #'1' the bit must be one.
    #'0' the bit must be zero.
    #'^' all bits with the caret are XOR-ed together.
    #'~' all bits with the tilde are XNOR-ed together.

    #formulas = [[symbol_map[r].format(v=v) for v, r in zip(varnames, rs)] for rs in result]
    grouped_terms = [dict(ut.group_items(varnames, rs)) for rs in result]
    def parenjoin(char, list_):
        if len(list_) == 0:
            return ''
        else:
            return '(' + char.join(list_) + ')'

    expanded_terms = [
        (
            term.get('1', []) +
            ['(not ' + b + ')' for b in term.get('0', [])] +
            [
                parenjoin(' ^ ', term.get('^', [])),
                parenjoin(' ~ ', term.get('~', [])),
            ]
        ) for term in grouped_terms
    ]

    final_terms = [[t for t in term if t] for term in expanded_terms]

    products = [parenjoin(' and ', [f for f in form if f]) for form in final_terms]
    final_expr = ' or '.join(products)
    print(final_expr)


def longest_common_substring(s1, s2):
    """
    References:
        # https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring#Python2
    """
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


@profile
def expensive_task_gen(num=8700):
    r"""
    Runs a task that takes some time

    Args:
        num (int): (default = 8700)

    CommandLine:
        python -m utool.util_alg expensive_task_gen --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> #num = 8700
        >>> num = 40000
        >>> with ut.Timer('expanesive task'):
        >>>     time_list = list(ut.expensive_task_gen(num))
        >>> print(sum(time_list))
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> #pt.plot(time_list)
        >>> from scipy.optimize import curve_fit
        >>> def func(x, a, b, c, d):
        >>>     return a * np.exp(-c * x) + d
        >>> #a*x**3 + b*x**2 +c*x + d
        >>> y = np.array(time_list)
        >>> y = np.array(ut.cumsum(y))
        >>> x = np.arange(len(y))
        >>> #popt, pcov = curve_fit(func, x, y, p0=(1, 1e-6, 1))
        >>> #print('pcov = %r' % (pcov,))
        >>> #print('popt = %r' % (popt,))
        >>> # http://stackoverflow.com/questions/3433486/-curve-fitting-in-python
        >>> pt.plt.plot(x[::num//50], y[::num//50], 'rx', label='measured data')
        >>> #x2 = np.arange(len(y) * 2)
        >>> #pt.plt.plot(x2, func(x2, *popt), 'b', label="Fitted Curve") #same as line above \/
        >>> #pt.plt.legend(loc='upper left')
        >>> ut.show_if_requested()
    """
    import utool as ut
    #time_list = []
    for x in range(0, num):
        with ut.Timer(verbose=False) as t:
            ut.is_prime(x)
        yield t.ellapsed
        #time_list.append(t.ellapsed)
        #print('t.ellapsed = %r' % (t.ellapsed,))
    #return time_list


def factors(n):
    """
    Computes all the integer factors of the number `n`

    Example:
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> result = sorted(ut.factors(10))
        >>> print(result)
        [1, 2, 5, 10]

    References:
        http://stackoverflow.com/questions/6800193/finding-all-the-factors
    """
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_alg
        python -m utool.util_alg --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
