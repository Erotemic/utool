# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
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


class EulerTourTree(object):
    """
    hg clone https://bitbucket.org/mozman/bintrees

    References:
        https://courses.csail.mit.edu/6.851/spring07/scribe/lec05.pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.192.8615&rep=rep1&type=pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.208.2351&rep=rep1&type=pdf
        https://en.wikipedia.org/wiki/Euler_tour_technique

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.dynamic_connectivity import *  # NOQA
        >>> edges = [(1, 2), (1, 6), (1, 5), (2, 3), (2, 4)]
        >>> edges = [
        >>>     ('R', 'A'), ('R', 'B'),
        >>>     ('B', 'C'), ('C', 'D'), ('C', 'E'),
        >>>     ('B', 'F'), ('B', 'G'),
        >>> ]
        >>> #edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
        >>> mst = nx.Graph(edges)
        >>> self = EulerTourTree()
        >>> import plottool as pt
        >>> pt.qt4ensure()
        >>> pt.show_nx(mst)
    """
    def __init__(self):
        pass

    @classmethod
    def from_mst(EulerTourTree, tour):
        # import blist
        self = EulerTourTree()
        tour = euler_tour_dfs(mst)
        tour_order = list(enumerate(tour))

        # self.first_visit_idxs = dict(i[::-1] for i in tour_order[::-1])
        # self.last_visit_idxs = dict(i[::-1] for i in tour_order)

        self.tour_order = tour_order
        self.tour = tour

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

        import bintrees
        # # items = zip(keys, keys)
        # tree = bintrees.AVLTree(zip(tour_order, visits))

        # import collections
        # EulerVisit = collections.namedtuple('EulerVisit',
        #                                     ('node', 'last', 'first'))
        class EulerVisit(object):
            def __init__(self, node, first, last):
                self.node = node
                self.last = last
                self.first = first

            def __repr__(self):
                return ut.repr2(self.node)

        # tree = bintrees.AVLTree(zip(visits, visits))
        # avl_node = tree._root
        def traverse_avl_nodes(avl_node):
            if not avl_node:
                raise StopIteration()
            for _ in traverse_avl_nodes(avl_node.left):
                yield _
            yield avl_node
            for _ in traverse_avl_nodes(avl_node.right):
                yield _

        tour_values = [(k, EulerVisit(v, None, None)) for k, v in tour_order]
        tree = bintrees.AVLTree(tour_values)

        import treap
        treap = treap.treap()

        self.first_lookup = first_lookup = {}
        self.last_lookup = last_lookup = {}
        flat_nodes = list(traverse_avl_nodes(tree._root))

        for avl_node in flat_nodes:
            # node = avl_node.value.node
            node = avl_node.value
            if node not in first_lookup:
                first_lookup[node] = avl_node
            last_lookup[node] = avl_node

        # for avl_node in flat_nodes:
        #     node = avl_node.value.node
        #     first = first_lookup[node]
        #     last = last_lookup[node]
        #     avl_node.value = EulerVisit(node, first, last)

        # self.tour = tour
        # # FIXME: need an implemenation of BBST that allows splicing
        self.tour_tree = tree
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

        slice_ = bintrees.treeslice.TreeSlice(tree, o_a1, o_b2)
        """
        return self

    def show_bbst(self, tree):
        import networkx as nx
        # import igraph as igraphs
        G = nx.Graph()
        root = tree._root
        G.add_node(0)
        queue = [[tree._root, 0]]
        index = 0
        while queue:
            node = queue[0][0]  # Select front of queue.
            node_index = queue[0][1]
            G.node[node_index]['label'] = (node.key, node.value)
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
        pt.show_nx(G)

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
    """

    def cut_edge(self, a, b):
        """
        a, b = 'B', 'C'
        print(self.first_lookup[a].key > self.first_lookup[b].key)
        tree = self.tour_tree
        list(tree.item_slice(k1, k2))
        """
        if self.first_lookup[a].key > self.last_lookup[b].key:
            a, b = b, a
        o_a1 = self.first_lookup[a]
        o_a2 = self.last_lookup[a]
        o_b1 = self.first_lookup[b]
        o_b2 = self.last_lookup[b]
        assert o_a1.key < o_b1.key
        assert o_b1.key < o_b2.key
        assert o_b2.key < o_a2.key

        o_b1.value

        tree = self.tour_tree
        t2_slice = tree[o_b1.key:o_b2.key + 1]
        t2 = bintrees.AVLTree(t2_slice)
        t1_not_keys = tree[o_b1.key, o_a2.key]

        list(bintrees.treeslice.TreeSlice(tree, o_b1.key, o_b2.key + 1))

        t2_values = list(tree.key_slice(o_b1.key, o_b2.key + 1))
        print('t2_values = %r' % (t2_values,))
        t1_not_keys = list(tree.key_slice(o_b1, o_a2 + 1))
        t1_values = tree.difference(t1_not_values)
        list(tree.value_slice(o_b1, o_a2))
        list(tree.value_slice(tree.min_key(), o_a1))
        list(tree.value_slice(o_a2, tree.max_key()))
        # if tree[('f', u)] > tree[('f', v)]:
        #     #     u, v = v, u

        if False:
            outer1 = self.tour[:o_a1 + 1]
            outer2 = self.tour[o_b2 + 1:]
            outer = outer1 + outer2
            inner = self.tour[o_a1 + 1:o_b2 + 1]

            new1 = EulerTourTree.from_tour(outer)
            new2 = EulerTourTree.from_tour(inner)
            return new1, new2
        # bintrees.AVLTree(outer)
        # bintrees.AVLTree(inner)

        # list(tree.item_slice(tree.min_item(), tree.max_item()))
        # list(tree.item_slice(k1, k2))
        # list(tree.key_slice(k1, k2))
        # tour_tree._root

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
        o_s1 = self.first_visit_idxs[s]
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
