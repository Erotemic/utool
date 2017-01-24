# -*- coding: utf-8 -*-
# UNFINISHED - do not use
from __future__ import print_function, division, absolute_import, unicode_literals
import collections  # NOQA
import networkx as nx
import utool as ut
print, rrr, profile = ut.inject2(__name__)
# import bintrees
# import rbtree


def euler_tour_dfs(G, source=None):
    """ adaptation of networkx dfs """
    if source is None:
        # produce edges for all components
        nodes = G
    else:
        # produce edges for components with source
        nodes = [source]
    yielder = []
    visited = set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, iter(G[start]))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    # yielder += [[parent, child]]
                    yielder += [parent]
                    visited.add(child)
                    stack.append((child, iter(G[child])))
            except StopIteration:
                if stack:
                    last = stack[-1]
                    yielder += [last[0]]
                stack.pop()
    return yielder


@profile
def comparison():
    r"""
    CommandLine:
        python -m utool.experimental.dynamic_connectivity comparison --profile
        python -m utool.experimental.dynamic_connectivity comparison
    """
    n = 12
    a, b = 9, 20
    num = 3

    import utool
    for timer in utool.Timerit(num, 'old bst version (PY)'):
        g = nx.balanced_tree(2, n)
        self = EulerTourTree.from_mst(g, version='bst', fast=False)
        with timer:
            self.delete_edge_bst_version(a, b, bstjoin=False)

    import utool
    for timer in utool.Timerit(num, 'new bst version (PY) (with join)'):
        g = nx.balanced_tree(2, n)
        self = EulerTourTree.from_mst(g, version='bst', fast=False)
        with timer:
            self.delete_edge_bst_version(a, b, bstjoin=True)

    import utool
    for timer in utool.Timerit(num, 'old bst version (C)'):
        g = nx.balanced_tree(2, n)
        self = EulerTourTree.from_mst(g, version='bst', fast=True)
        with timer:
            self.delete_edge_bst_version(a, b, bstjoin=False)

    import utool
    for timer in utool.Timerit(num, 'new bst version (C) (with join)'):
        g = nx.balanced_tree(2, n)
        self = EulerTourTree.from_mst(g, version='bst', fast=True)
        with timer:
            self.delete_edge_bst_version(a, b, bstjoin=True)

    import utool
    for timer in utool.Timerit(num, 'list version'):
        g = nx.balanced_tree(2, n)
        self = EulerTourTree.from_mst(g, version='list')
        with timer:
            self.delete_edge_list_version(a, b)
    pass


class EulerTourTree(object):
    """


    raise NotImplementedError()

    hg clone https://bitbucket.org/mozman/bintrees

    References:
        https://courses.csail.mit.edu/6.851/spring07/scribe/lec05.pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.192.8615&rep=rep1&type=pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.208.2351&rep=rep1&type=pdf
        https://en.wikipedia.org/wiki/Euler_tour_technique

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.dynamic_connectivity import *  # NOQA
        >>> #edges = [(1, 2), (1, 6), (1, 5), (2, 3), (2, 4)]
        >>> #edges = [
        >>> #    ('R', 'A'), ('R', 'B'),
        >>> #    ('B', 'C'), ('C', 'D'), ('C', 'E'),
        >>> #    ('B', 'F'), ('B', 'G'),
        >>> #]
        >>> #mst = nx.Graph(edges)
        >>> mst = nx.balanced_tree(2, 11)
        >>> self = EulerTourTree.from_mst(mst)
        >>> import plottool as pt
        >>> pt.qt4ensure()
        >>> pt.show_nx(mst)

        >>> mst = nx.balanced_tree(2, 4)
    """
    def __init__(self):
        pass

    @classmethod
    @profile
    def from_mst(EulerTourTree, mst, version='bst', fast=True):
        """
        >>> # DISABLE_DOCTEST
        >>> from utool.experimental.dynamic_connectivity import *  # NOQA
        >>> mst = nx.balanced_tree(2, 4)
        >>> self = EulerTourTree.from_mst(mst)
        >>> import plottool as pt
        >>> pt.qt4ensure()
        >>> pt.show_nx(self.to_graph(), pnum=(2, 1, 1), fnum=1)

        >>> a, b = 2, 5
        >>> other = self.delete_edge_bst_version(a, b)
        >>> pt.show_nx(other.to_graph(), pnum=(2, 1, 1), fnum=2)

        """
        tour = euler_tour_dfs(mst)
        self = EulerTourTree.from_tour(tour, version=version, fast=fast)
        return self

    @classmethod
    @profile
    def from_tour(EulerTourTree, tour, version='bst', fast=True):
        import bintrees
        self = EulerTourTree()
        self.fast = fast
        self.version = version

        if version == 'bst':
            # self.tour = tour
            if fast:
                tree = bintrees.FastAVLTree(enumerate(tour))
            else:
                tree = bintrees.AVLTree(enumerate(tour))

            self.first_lookup = first_lookup = {}
            self.last_lookup = last_lookup = {}

            for key, node in tree.iter_items():
                # node = avl_node.value
                if node not in first_lookup:
                    # first_lookup[node] = avl_node.key
                    first_lookup[node] = key
                # last_lookup[node] = avl_node.key
                last_lookup[node] = key
            self.tour_tree = tree
        elif version == 'sortedcontainers':
            """
            >>> from utool.experimental.dynamic_connectivity import *  # NOQA
            >>> mst = nx.balanced_tree(2, 4)
            >>> tour = euler_tour_dfs(mst)
            >>> self = EulerTourTree()

            """
            import sortedcontainers
            tour_order = sortedcontainers.SortedList(enumerate(tour))
            self.first_lookup = dict(i[::-1] for i in tour_order[::-1])
            self.last_lookup = dict(i[::-1] for i in tour_order)
            self.tour_order = tour_order
        else:
            self.tour = tour
            tour_order = list(enumerate(tour))
            self.first_lookup = dict(i[::-1] for i in tour_order[::-1])
            self.last_lookup = dict(i[::-1] for i in tour_order)
            tour_order.bisect_left((7, 0))

        return self

    @profile
    def delete_edge_bst_version(self, a, b, bstjoin=False):
        """
        a, b = (2, 5)
        print(self.first_lookup[a] > self.first_lookup[b])
        tree = self.tour_tree
        list(tree.item_slice(k1, k2))
        """
        if self.first_lookup[a] > self.last_lookup[b]:
            a, b = b, a

        o_a1 = self.first_lookup[a]
        o_a2 = self.last_lookup[a]
        o_b1 = self.first_lookup[b]
        o_b2 = self.last_lookup[b]
        assert o_a1 < o_b1
        # assert o_b1 < o_b2
        assert o_b2 < o_a2

        if bstjoin:
            # splice out the inside contiguous range inplace
            inside, outside = self.tour_tree.splice_inplace(o_b1, o_b2 + 1)
            # Remove unneeded values
            outside = outside.splice_inplace(o_b1, o_a2 + 1)[1]

            other = EulerTourTree()
            other.tour_tree = inside
            # We can reuse these pointers without any modification
            other.first_lookup = self.first_lookup
            other.first_lookup = self.last_lookup
            # Should make an O(n) cleanup step at some point
        else:
            # ET(T2) inner - is given by the interval of ET (o_b1, o_b2)
            # Smaller compoment is reconstructed
            # in amortized O(log(n)) time
            t2_slice = self.tour_tree[o_b1:o_b2 + 1]
            t2_tour = list(t2_slice.values())
            other = EulerTourTree.from_tour(t2_tour, version=self.version,
                                            fast=self.fast)

            # ET(T1) outer - is given by splicing out of ET the sequence
            # (o_b1, o_a2)
            t1_splice = self.tour_tree[o_b1:o_a2 + 1]
            self.tour_tree.remove_items(t1_splice)
        return other

    @profile
    def delete_edge_list_version(self, a, b):
        if self.first_lookup[a] > self.last_lookup[b]:
            a, b = b, a

        o_a1 = self.first_lookup[a]
        o_a2 = self.last_lookup[a]
        o_b1 = self.first_lookup[b]
        o_b2 = self.last_lookup[b]
        assert o_a1 < o_b1
        # assert o_b1 < o_b2
        assert o_b2 < o_a2

        t2_list = self.tour[o_b1:o_b2 + 1]
        other = EulerTourTree.from_tour(t2_list, version=self.version,
                                        fast=self.fast)

        # ET(T1) outer - is given by splicing out of ET the sequence
        # (o_b1, o_a2)
        self.tour = self.tour[:o_b1] + self.tour[o_a2 + 1:]
        # need to recompute lookups O(n) style
        # maybe we can do better?
        # Keep old keys
        if False:
            tour_order = list(enumerate(self.tour))
            self.first_lookup = dict(i[::-1] for i in tour_order[::-1])
            self.last_lookup = dict(i[::-1] for i in tour_order)
        return other

    def reroot(self, s):
        """
        s = 3
        s = 'B'

        Let os denote any occurrence of s.
        Splice out the first part of the sequence ending with the occurrence before os,
        remove its first occurrence (or),
        and tack this on to the end of the sequence which now begins with os.
        Add a new occurrence os to the end.
        """
        # Splice out the first part of the sequence ending with the occurrence before os
        # remove its first occurrence (or),
        o_s1 = self.first_lookup[s]
        splice1 = self.tour[1:o_s1]
        rest = self.tour[o_s1 + 1:]
        new_tour = [s] + rest + splice1 + [s]
        new_tree = EulerTourTree.from_tour(new_tour, version=self.version,
                                           fast=self.fast)
        return new_tree

    def to_graph(self):
        import utool as ut
        return nx.Graph(ut.itertwo(self.tour))

    def join_trees(self, t1, t2, e):
        pass


class EulerTourForest(object):
    pass


class DynamicConnectivity(object):
    """
    Stores a spanning forest with Euler Tour Trees

    References:
        https://courses.csail.mit.edu/6.851/spring14/lectures/L20.pdf
        https://courses.csail.mit.edu/6.851/spring14/lectures/L20.html
        http://cs.stackexchange.com/questions/33595/what-is-the-most-efficient-algorithm-and-data-structure-for-maintaining-connecte

        https://www.cs.princeton.edu/courses/archive/fall03/cs528/handouts/Poly%20logarithmic.pdf
        http://courses.csail.mit.edu/6.851/spring12/scribe/L20.pdf

    Notes:
        Invariant 1 Every connected component of G_i has at most 2^i vertices.
        Invariant 2 F[0] ⊆ F[1] ⊆ F[2] ⊆ ... ⊆ F[log(n)].
            In other words:
                F[i] = F[log(n)] ∩ G_i, and
                F[log(n)] is the minimum spanning forest of G_{log(n)},
                where the weight of an edge is its level.

    CommandLine:
        python -m ibeis.algo.hots.dynamic_connectivity DynamicConnectivity --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.dynamic_connectivity import *  # NOQA
        >>> import utool as ut
        >>> import plottool as pt
        >>> pt.qt4ensure()
        >>> graph = nx.Graph([
        >>>    (0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 4), (2, 3),
        >>>    (5, 6), (5, 7), (5, 8), (6, 8), (7, 9), (8, 9), (7, 8),
        >>> ])
        >>> pt.show_nx(graph)
        >>> ut.show_if_requested()
    """

    def __init__(self, graph):
        # List of forests at each level
        self.graph = graph
        self.n = graph.number_of_nodes()
        # stores the level of each edges
        self.level = {}
        self.F = []

        # First add all tree edges at level 0
        # Then add non-tree edges at higher levels
        # Store each forest as a nx.Graph?

        forests = []
        current_level = list(nx.connected_component_subgraphs(graph))
        while current_level:
            next_level = []
            for subgraph in current_level:
                mst = nx.minimum_spanning_tree(subgraph)
                # mst_tour = find_euler_tour(mst)
                forests.append(mst)
                residual = nx.difference(subgraph, mst)
                if residual.number_of_edges():
                    next_level.append(residual)
            current_level = next_level
            print('current_level = %r' % (current_level,))

    def insert(self, u, v):
        e = (u, v)
        # First set the level of `e` to 0
        self.level[e] = 0
        # update the adjacency lists of u and v
        self.graph.add_edge(u, v)
        # If u and v are in separate trees in F_0, add e to F_0
        F0 = self.F[0]
        if F0.find(u) is not F0.find(v):
            F0.union(u, v)
        # if F0[u]['tree_label'] != F0[v]['tree_label']:
        # if F0[u]['tree_label'] != F0[v]['tree_label']:
        #     F0.add_edge(u, v)

    def delete(self, u, v):
        # Remove edge e = (u, v) from the graph.
        # we first remove e from the adjacency lists of u and v.
        # If e is not in F[log(n)], we’re done
        # Otherwise:
        # 1. Delete e from Fi for all i ≥ level(e).
        #     * Now we want to look for a replacement edge to reconnect u and v.
        #     * Note that the replacement edge cannot be at a level less than
        #     level(e) by Invariant 2 (recall that each Fi is a minimum spanning forest).
        #     * We will start searching for a replacement edge at level(e) to
        #     preserve the Invariant refinv:subset.
        # We will look for this replacement edge by doing the following:
        # 2. For i = level(e) to log n:
        # (a) Let Tu be the tree containing u, and let Tv be the tree
        #     containing v. WLOG, assume |Tv| ≤ |Tu|.
        # (b) By Invariant 1, we know that |Tu| + |Tv| ≤ 2^{i},
        #     so |Tv| ≤ 2^{i−1}.
        #     This means that we can afford to push all edges of Tv down to
        #     level i − 1.
        # (c) For each edge (x, y) at level i with x in Tv:
        #    i. If y is in Tu, add (x, y) to Fi, Fi+1, . . . , Flog n, and stop.
        #    ii. Otherwise set level(x, y) ← i − 1.
        pass

    def is_connected(self, u, v):
        # Check if vertices u and v are connected.
        # Query F_{log(n)} to see if u and v are in the same tree.
        # This can be done by checking F_{log(n)} if Findroot(u) = Findroot(v).
        # This costs O(log n/ log log n) using B-tree based Euler-Tour trees.
        self.F[-1]
        pass

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.hots.dynamic_connectivity
        python -m ibeis.algo.hots.dynamic_connectivity --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
