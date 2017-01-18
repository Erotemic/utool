# -*- coding: utf-8 -*-
# UNFINISHED - do not use
from __future__ import print_function, division, absolute_import, unicode_literals
import collections  # NOQA
import networkx as nx
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


def junk():
    # import blist
    # self.first_visit_idxs = dict(i[::-1] for i in tour_order[::-1])
    # self.last_visit_idxs = dict(i[::-1] for i in tour_order)

    # Store a tree by keeping its Euler tour in a balanced binary search
    # tree, keyed by the index in the tour.

    # class EulerVisit(object):
    #     def __init__(self, num, node):
    #         self.num = num
    #         self.node = node
    #         self.first = None
    #         self.last = None

    #     def __repr__(self):
    #         return ut.repr2(self.__hash__())

    #     def __eq__(self, other):
    #         return self.node == other.node and self.num == other.num

    #     def __lt__(self, other):
    #         if self != other and self.num == other.num:
    #             id(self) < id(other)
    #         return (self.num < other.num)

    #     def __hash__(self):
    #         return (self.node, self.num)

    # # Each node in the represented tree holds pointers to the nodes in the
    # # BST representing the first and last times it was visited
    # visits = ut.lstarmap(EulerVisit, enumerate(tour))
    # groups = ut.group_items(visits, tour)
    # for node, group in groups.items():
    #     first = min(group)
    #     last = max(group)
    #     for visit in group:
    #         visit.first = first
    #         visit.last = last
    # # items = zip(keys, keys)
    # tree = bintrees.AVLTree(zip(tour_order, visits))

    # import collections
    # EulerVisit = collections.namedtuple('EulerVisit',
    #                                     ('node', 'last', 'first'))
    # tree = bintrees.AVLTree(zip(visits, visits))
    # avl_node = tree._root
    # node = avl_node.value.node
    # for avl_node in flat_nodes:
    #     node = avl_node.value.node
    #     first = first_lookup[node]
    #     last = last_lookup[node]
    #     avl_node.value = EulerVisit(node, first, last)

    # self.tour = tour
    # # FIXME: need an implemenation of BBST that allows splicing
    if True:
        """
        for k in mst.nodes():
            print(k, self.first_lookup[k].key, self.last_lookup[k].key)

        print(ut.repr2(tour))

        a, b = 'B', 'C'
        o_a1 = self.first_lookup[a]
        o_b1 = self.first_lookup[b]
        o_a2 = self.last_lookup[a]
        o_b2 = self.last_lookup[b]
        print('o_a1.key = %r' % (o_a1.key,))
        print('o_a2.key = %r' % (o_a2.key,))
        print('o_b1.key = %r' % (o_b1.key,))
        print('o_b2.key = %r' % (o_b2.key,))
        list(tree.item_slice(o_a1, o_b2))

        slice_ = tree[o_a1:o_b2]
        """

    """
    items = list(range(20))
    bst = bintrees.AVLTree(zip(items, items))
    # Slicing is nice because it just accesses the first and last element
    sub_slice = bst[5:15]

    # But it seems like if I do anything with the slice all of its items are
    # expanded and each operation is performed individually

    # I want a subtree (no copying) in O(log(n)), but this seems to take O(n)
    sub_pointer = bintrees.AVLTree(bst[5:15])

    # I want to delete in O(log(n)), but this seems to take O(n)
    del bst[5:15]

        # class EulerVisit(object):
        #     def __init__(self, node, first=None, last=None):
        #         self.node = node
        #         self.last = last
        #         self.first = first

        #     def __repr__(self):
        #         return ut.repr2(self.node)
        # tour_values = [(k, EulerVisit(v)) for k, v in tour_order]
    """
    pass


def traverse_avl_nodes_recursive(avl_node):
    if not avl_node:
        raise StopIteration()
    for _ in traverse_avl_nodes(avl_node.left):
        yield _
    yield avl_node
    for _ in traverse_avl_nodes(avl_node.right):
        yield _


def traverse_avl_nodes(root):
    stack = []
    node = root
    while stack or node is not None:
        if node is not None:
            stack.append(node)
            node = node.left
        else:
            node = stack.pop()
            yield node
            node = node.right


def avl_find_node(self, key):
    node = self._root
    while node is not None:
        if key == node.key:
            return node
        elif key < node.key:
            node = node.left
        else:
            node = node.right


def show_avl_tree(tree, fnum=None, pnum=None):
    """
    >>> show_avl_tree(tree, pnum=(2, 1, 2), fnum=1)
    >>> pt.show_nx(mst, pnum=(2, 1, 1), fnum=1)

    """
    import networkx as nx
    # import igraph as igraphs
    G = nx.Graph()
    G.add_node(0)
    queue = [[tree._root, 0]]
    index = 0
    while queue:
        node = queue[0][0]  # Select front of queue.
        node_index = queue[0][1]
        G.node[node_index]['label'] = '%s,%s' % (node.key, node.value)
        if node.left is not None:
            G.add_node(node_index)
            G.add_edges_from([(node_index, index + 1)])
            queue.append([node.left, index + 1])
            index += 1
        if node.right is not None:
            G.add_node(node_index)
            G.add_edges_from([(node_index, index + 1)])
            queue.append([node.right, index + 1])
            index += 1
        queue.pop(0)
    import plottool as pt
    pt.qt4ensure()
    pt.show_nx(G, fnum=fnum, pnum=pnum)


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
    def from_mst(EulerTourTree, mst):
        """
        >>> # DISABLE_DOCTEST
        >>> from utool.experimental.dynamic_connectivity import *  # NOQA
        >>> mst = nx.balanced_tree(2, 3)
        >>> self = EulerTourTree.from_mst(mst)
        >>> import plottool as pt
        >>> pt.qt4ensure()
        >>> show_avl_tree(self.tour_tree, pnum=(2, 1, 2), fnum=1)
        >>> pt.show_nx(self.mst, pnum=(2, 1, 1), fnum=1)
        """
        self = EulerTourTree()
        self.mst = mst
        tour = euler_tour_dfs(mst)
        self = EulerTourTree.from_tour(tour)
        # if True:
        # else:
        #     self.first_lookup = dict(
        #         i[::-1] for i in tour_order[::-1])
        #     self.last_lookup = dict(
        #         i[::-1] for i in tour_order)
        return self

    @classmethod
    def from_tour(EulerTourTree, tour):
        import bintrees
        self = EulerTourTree()
        self.tour = tour
        tree = bintrees.AVLTree(list(enumerate(tour)))

        self.first_lookup = first_lookup = {}
        self.last_lookup = last_lookup = {}

        for avl_node in traverse_avl_nodes(tree._root):
            node = avl_node.value
            if node not in first_lookup:
                first_lookup[node] = avl_node.key
            last_lookup[node] = avl_node.key
        self.tour_tree = tree

    def delete_edge(self, a, b):
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

        # if False:
        #     t2_list = self.tour[o_b1:o_b2 + 1]
        #     t1_list = self.tour[:o_b1] + self.tour[o_a2 + 1:]

        if True:
            tree = self.tour_tree
            # ET(T2) inner - is given by the interval of ET (o_b1, o_b2)
            # Smaller compoment is reconstructed
            # in amortized O(log(n)) time
            t2_slice = tree[o_b1:o_b2 + 1]
            t2_tour = list(t2_slice.values())
            other = EulerTourTree.from_tour(t2_tour)

            # ET(T1) outer - is given by splicing out of ET the sequence
            # (o_b1, o_a2)
            t1_splice = tree[o_b1:o_a2 + 1]
            tree.remove_items(t1_splice)
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
        new_tree = EulerTourTree.from_tour(new_tour)
        return new_tree

    def to_graph(self):
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
