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
        self = TestETT.from_tree(g, version='bst', fast=False)
        with timer:
            self.delete_edge_bst_version(a, b, bstjoin=False)

    import utool
    for timer in utool.Timerit(num, 'new bst version (PY) (with join)'):
        g = nx.balanced_tree(2, n)
        self = TestETT.from_tree(g, version='bst', fast=False)
        with timer:
            self.delete_edge_bst_version(a, b, bstjoin=True)

    import utool
    for timer in utool.Timerit(num, 'old bst version (C)'):
        g = nx.balanced_tree(2, n)
        self = TestETT.from_tree(g, version='bst', fast=True)
        with timer:
            self.delete_edge_bst_version(a, b, bstjoin=False)

    import utool
    for timer in utool.Timerit(num, 'new bst version (C) (with join)'):
        g = nx.balanced_tree(2, n)
        self = TestETT.from_tree(g, version='bst', fast=True)
        with timer:
            self.delete_edge_bst_version(a, b, bstjoin=True)

    import utool
    for timer in utool.Timerit(num, 'list version'):
        g = nx.balanced_tree(2, n)
        self = TestETT.from_tree(g, version='list')
        with timer:
            self.delete_edge_list_version(a, b)
    pass


class TestETT(object):
    """


    raise NotImplementedError()

    hg clone https://bitbucket.org/mozman/bintrees

    git clone git@github.com:Erotemic/bintrees.git

    References:
        https://courses.csail.mit.edu/6.851/spring07/scribe/lec05.pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.192.8615&rep=rep1&type=pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.208.2351&rep=rep1&type=pdf
        https://en.wikipedia.org/wiki/Euler_tour_technique

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.experimental.dynamic_connectivity import *  # NOQA
        >>> edges = [(1, 2), (1, 6), (1, 5), (2, 3), (2, 4)]
        >>> edges = [
        >>>     ('R', 'A'), ('R', 'B'),
        >>>     ('B', 'C'), ('C', 'D'), ('C', 'E'),
        >>>     ('B', 'F'), ('B', 'G'),
        >>> ]
        >>> mst = nx.Graph(edges)
        >>> #mst = nx.balanced_tree(2, 11)
        >>> self = TestETT.from_tree(mst)
        >>> import plottool_ibeis as pt
        >>> pt.qt4ensure()
        >>> pt.show_nx(mst)

        >>> mst = nx.balanced_tree(2, 4)
    """
    def __init__(self):
        pass

    @classmethod
    @profile
    def from_tree(TestETT, mst, version='bst', fast=True):
        """
        >>> # DISABLE_DOCTEST
        >>> from utool.experimental.dynamic_connectivity import *  # NOQA
        >>> mst = nx.balanced_tree(2, 4)
        >>> self = TestETT.from_tree(mst)
        >>> import plottool_ibeis as pt
        >>> pt.qt4ensure()
        >>> pt.show_nx(self.to_networkx(), pnum=(2, 1, 1), fnum=1)

        >>> a, b = 2, 5
        >>> other = self.delete_edge_bst_version(a, b)
        >>> pt.show_nx(other.to_networkx(), pnum=(2, 1, 1), fnum=2)

        """
        tour = euler_tour_dfs(mst)
        self = TestETT.from_tour(tour, version=version, fast=fast)
        return self

    @classmethod
    @profile
    def from_tour(TestETT, tour, version='bst', fast=True):
        import bintrees
        self = TestETT()
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
            >>> self = TestETT()

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

            other = TestETT()
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
            other = TestETT.from_tour(t2_tour, version=self.version,
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
        other = TestETT.from_tour(t2_list, version=self.version,
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
        new_tree = TestETT.from_tour(new_tour, version=self.version,
                                           fast=self.fast)
        return new_tree

    def to_networkx(self):
        import utool as ut
        return nx.Graph(ut.itertwo(self.tour))

    def join_trees(self, t1, t2, e):
        pass


# class AVLMaster(object):
#     def __init__(self, master=None):
#         self.master = None


# class AVLKey(object):
#     def __init__(self, master=None):
#         self.master = None


class EulerTourList(object):
    """
    load-list representation of an Euler tour inspired by sortedcontainers

    this doesnt work for the same reason keyed bintrees dont work

    the indexing needs to be implicit, but this has explicit indexes
    """

    def __init__(self, iterable, load=1000):
        self.first = {}
        self.last = {}
        self._len = 0
        self._cumlen = []
        self._lists = []
        self._load = load
        self._twice = load * 2
        self._half = load >> 1
        self._offset = 0

        if iterable is not None:
            self.update(iterable)

    def __iter__(self):
        import itertools as it
        return it.chain.from_iterable(self._lists)

    def __repr__(self):
        return 'EulerTourList(' + str(list(self)) + ')'

    def join(self, other):
        r"""
        Args:
            other (?):

        CommandLine:
            python -m sortedcontainers.sortedlist join2

        Example:
            >>> from utool.experimental.dynamic_connectivity import *  # NOQA
            >>> self = EulerTourList([1, 2, 3, 2, 4, 2, 1], load=3)
            >>> other = EulerTourList([0, 5, 9, 5, 0], load=3)
            >>> result = self.join(other)
            >>> print(result)
        """
        assert self._load == other._load, 'loads must be the same'
        self._lists.extend(other._lists)
        self._cumlen.extend([c + self._len for c in other._cumlen])
        self._len += other._len

    def split(self, pos, idx):
        # (pos, idx) = self._pos(idx)
        left_part = self._lists[0:pos + 1]
        right_part = self._lists[pos + 1:0]

        left_last = left_part[-1][:idx]
        right_first = left_part[-1][idx:]

        if left_last:
            left_part[-1] = left_last
        else:
            del left_part[-1]
        if right_first:
            right_part.insert(0, right_first)

        other = EulerTourList()
        other._list = right_part

    def append(self, value):
        pass

    def update(self, iterable):
        _lists = self._lists
        values = list(iterable)

        _load = self._load
        _lists.extend(values[pos:(pos + _load)]
                      for pos in range(0, len(values), _load))
        self._cumlen = ut.cumsum(map(len, _lists))
        self._len = len(values)


class EulerTourTree(object):
    """
    CommandLine:
        python -m utool.experimental.dynamic_connectivity EulerTourTree --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.experimental.dynamic_connectivity import *  # NOQA
        >>> #mst = nx.balanced_tree(2, 2)
        >>> edges = [
        >>>     ('R', 'A'), ('R', 'B'),
        >>>     ('B', 'C'), ('C', 'D'), ('C', 'E'),
        >>>     ('B', 'F'), ('B', 'G'),
        >>> ]
        >>> mst = nx.Graph(edges)
        >>> self = EulerTourTree.from_tree(mst)
        >>> import plottool_ibeis as pt
        >>> pt.qt4ensure()
        >>> fnum = 1
        >>> pnum_ = pt.make_pnum_nextgen(1, 3)
        >>> pt.show_nx(mst, pnum=pnum_(), fnum=fnum)
        >>> pt.show_nx(self.to_networkx(), pnum=pnum_(), fnum=fnum)
        >>> pt.show_nx(self.tour_tree.to_networkx(labels=['key', 'value']), pnum=pnum_(), fnum=fnum)
        >>> print(self.tour)
        >>> print(self.first_lookup)
        >>> print(self.last_lookup)
        >>> ut.show_if_requested()
    """
    def __init__(self):
        # node attributes in reprsented graph
        self.first_lookup = {}
        self.last_lookup = {}

        # structure stored in auxillary graph
        self.tour_tree = None
        # self.tour = None

    def to_networkx(self):
        import utool as ut

        # n = list(self.tour_tree._traverse_nodes())[0]

        # return nx.Graph(ut.itertwo(self.tour))
        # In order traversal of the tree is the tour order
        # return nx.Graph(ut.itertwo(self.tour_tree.values()))
        tour = (n.value for n in self.tour_tree._traverse_nodes())
        graph = nx.Graph(ut.itertwo(tour))
        return graph

    def find_root(self, node):

        pass

    @classmethod
    @profile
    def from_tree(EulerTourTree, mst, fast=True, start=0):
        tour = euler_tour_dfs(mst)
        self = EulerTourTree.from_tour(tour, fast=fast, start=0)
        return self

    @classmethod
    @profile
    def from_tour(EulerTourTree, tour, fast=False, start=0):
        import bintrees
        self = EulerTourTree()
        self.fast = fast

        if fast:
            tour_tree = bintrees.FastAVLTree(enumerate(tour, start=start))
        else:
            tour_tree = bintrees.AVLTree(enumerate(tour, start=start))

        self.first_lookup = first_lookup = {}
        self.last_lookup = last_lookup = {}

        for key, node in tour_tree.iter_items():
            # node = avl_node.value
            if node not in first_lookup:
                # first_lookup[node] = avl_node.key
                first_lookup[node] = key
            # last_lookup[node] = avl_node.key
            last_lookup[node] = key

        self.tour_tree = tour_tree
        # self.tour = tour
        tour_order = list(enumerate(tour))
        self.first_lookup = dict(i[::-1] for i in tour_order[::-1])
        self.last_lookup = dict(i[::-1] for i in tour_order)
        # tour_order.bisect_left((7, 0))
        return self

    def join(self, other):
        pass

    @profile
    def cut(self, a, b, bstjoin=False):
        """
        cuts edge (a, b) into two parts because this is a tree

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
        new_tree = TestETT.from_tour(new_tour, fast=self.fast)
        return new_tree


class BinaryNode(object):
    def __init__(self, value, parent=None, left=None, right=None):
        self.value = value
        self.parent = parent
        self.children = [left, right]
        self._dir = 0

    @property
    def left(self):
        return self.children[0]

    @left.setter
    def left(self, other):
        self.children[0] = other

    @property
    def right(self):
        return self.children[1]

    @right.setter
    def right(self, other):
        self.children[1] = other

    def set_child(self, other, dir_):
        other.parent = self
        self.children[dir_] = other


class EulerTourForest(object):
    def __init__(self):
        self.aux_trees = {}
        self.first = {}
        self.last = {}
        self.n_nodes = 0
        # import bintrees
        # self._cls = bintrees.AVLTree

    def has_node(self, node):
        return node in self.first

    def add_node(self, node):
        if not self.has_node(node):
            binnode = BinaryNode(node)
            self.aux_trees[node] = binnode
            self.first[node] = binnode
            self.last[node] = binnode

    def find_root(self, node):
        return self.first[node]

    def reroot(self, old, new):
        assert old == new
        return new

    def add_edge(self, u, v):
        """
        self = EulerTourForest()
        self.add_node(1)
        self.add_node(2)
        u, v = 1, 2
        """
        # ubin = self.find_root(u)
        ru = self.find_root(u)
        rv = self.find_root(v)
        ru = self.reroot(ru, u)
        rv = self.reroot(rv, v)
        # ubin.set_child(vbin)
        pass


class DummyEulerTourForest(object):
    """
    maintain a forests of euler tour trees

    This is a bad implementation, but will let us use the DynConnGraph api
    """
    def __init__(self, nodes=None):
        # mapping from root node to tree
        self.trees = {}
        if nodes is not None:
            for node in nodes:
                self.add_node(node)

    def _check_node_type(self, node):
        if not isinstance(node, (int, str)):
            raise ValueError('only primative int/str objects can be nodes')

    def add_node(self, node):
        self._check_node_type(node)
        root = self.find_root(node)
        if root is None:
            # self.trees[node] = EulerTourTree(node)
            new_root = nx.Graph()
            new_root.add_node(node)
            self.trees[node] = new_root

    def has_edge(self, u, v):
        return any(tree.has_edge(u, v) for tree in self.components())

    def find_root(self, node):
        for root, subgraph in self.trees.items():
            if subgraph.has_node(node):
                return root

    def subtree(self, node):
        root = self.find_root(node)
        subtree = self.trees[root]
        return subtree

    def remove_edge(self, u, v):
        ru = self.find_root(u)
        rv = self.find_root(v)
        assert ru == rv
        subtree = self.trees[ru]
        del self.trees[ru]
        subtree.remove_edge(u, v)
        for new_tree in nx.connected_component_subgraphs(subtree):
            if new_tree.has_node(u):
                self.trees[u] = new_tree
            elif new_tree.has_node(v):
                self.trees[v] = new_tree
        # raise NotImplementedError('remove edge')

    def add_edge(self, u, v):
        print('[euler_tour_forest] add_edge(%r, %r)' % (u, v))
        if self.has_edge(u, v):
            return
        ru = self.find_root(u)
        rv = self.find_root(v)
        if ru is None:
            self.add_node(u)
            ru = u
        if rv is None:
            self.add_node(v)
            rv = v
        assert ru is not rv, (
            'u=%r, v=%r not disjoint, can only join disjoint edges' % (u, v))
        assert ru in self.trees, 'ru must be a root node'
        assert rv in self.trees, 'rv must be a root node'
        subtree1 = self.trees[ru]
        subtree2 = self.trees[rv]
        del self.trees[rv]
        new_tree = nx.compose(subtree1, subtree2)
        new_tree.add_edge(u, v)
        self.trees[ru] = new_tree
        print(list(new_tree.nodes()))
        assert nx.is_connected(new_tree)
        assert nx.is_tree(new_tree)

    def components(self):
        return self.trees.values()

    def to_networkx(self):
        graph = nx.compose_all(self.components())
        return graph


class DynConnGraph(object):
    """
    Stores a spanning forest with Euler Tour Trees

    References:
        https://courses.csail.mit.edu/6.851/spring14/lectures/L20.pdf
        https://courses.csail.mit.edu/6.851/spring14/lectures/L20.html
        http://cs.stackexchange.com/questions/33595/what-is-the-most-efficient-algorithm-and-data-structure-for-maintaining-connecte

        https://www.cs.princeton.edu/courses/archive/fall03/cs528/handouts/Poly%20logarithmic.pdf

        DEFINES ET-Trees
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.192.8615&rep=rep1&type=pdf

        https://www.cs.princeton.edu/courses/archive/fall03/cs528/handouts/Poly%20logarithmic.pdf
        http://courses.csail.mit.edu/6.851/spring12/scribe/L20.pdf
        http://courses.csail.mit.edu/6.854/16/Projects/B/dynamic-graphs-survey.pdf
        http://dl.acm.org/citation.cfm?id=502095
        http://delivery.acm.org/10.1145/510000/502095/p723-holm.pdf?ip=128.213.17.14&id=502095&acc=ACTIVE%20SERVICE&key=7777116298C9657D%2EAF047EA360787914%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=905563744&CFTOKEN=37809688&__acm__=1488222284_4ae91dd7a761430ee714f0c69c17b772

    Notes:
        Paper uses level 0 at top, but video lecture uses floor(log(n)) as top.

        All edges start at level floor(log(n)).
        The level of each edge will change over time, but cannot decrease below zero.


    Notes:
        Going to store

        Let G[i] = subgraph graph at level i. (
            contains only edges of level i or greater)
        Let F[i] be Euler tour forest to correspond with G[i].

        G[log(n)] = full graph

    Notes:
        Invariant 1 Every connected component of G_i has at most 2^i vertices.
        Invariant 2 F[0] ⊆ F[1] ⊆ F[2] ⊆ ... ⊆ F[log(n)].
            In other words:
                F[i] = F[log(n)] ∩ G_i, and
                F[log(n)] is the minimum spanning forest of G_{log(n)},
                where the weight of an edge is its level.

                F[0] is a maximum spanning forest if using 0 as top level

    CommandLine:
        python -m utool.experimental.dynamic_connectivity DynConnGraph --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.experimental.dynamic_connectivity import *  # NOQA
        >>> import networkx as nx
        >>> import utool as ut
        >>> import plottool_ibeis as pt
        >>> graph = nx.Graph([
        >>>    (0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 4), (2, 3),
        >>>    (5, 6), (5, 7), (5, 8), (6, 8), (7, 9), (8, 9), (7, 8),
        >>> ])
        >>> graph = nx.generators.cycle_graph(5)
        >>> self = DynConnGraph(graph)
        >>> pt.qtensure()
        >>> pt.show_nx(self.graph, fnum=1)
        >>> self.show_internals(fnum=2)
        >>> self.remove_edge(1, 2)
        >>> self.show_internals(fnum=3)
        >>> self.remove_edge(3, 4)
        >>> self.show_internals(fnum=4)
        >>> ut.show_if_requested()
    """

    def show_internals(self, fnum=None):
        import plottool_ibeis as pt
        pt.qtensure()

        pnum_ = pt.make_pnum_nextgen(nRows=1, nCols=len(self.forests))
        for level, forest in enumerate(self.forests):
            pt.show_nx(forest.to_networkx(), title='level=%r' % (level,),
                       fnum=fnum, pnum=pnum_())

    def _init_forests():
        """
        F[i] is a spanning forest of G[i].
        F[i] is stored as an EulerTourTree

        F[floor(log(n))] - used to support connectivity queries

        F[0] has fewest edges
        F[log(n)] has most edges

        Invariant 2 F[0] ⊆ F[1] ⊆ F[2] ⊆ ... ⊆ F[log(n)].
            In other words:
                F[i] = F[log(n)] ∩ G_i, and
                F[log(n)] is the minimum spanning forest of G_{log(n)},
                where the weight of an edge is its level.
            F[i] is a min. spanning forest w.r.t level, otherwise invariant 2 is not satisfied
        """
        pass

    def __init__(self, graph):
        # List of forests at each level
        # self.graph = graph
        self.graph = nx.Graph()
        self.level = {}
        self.forests = [DummyEulerTourForest()]

        for u, v in graph.edges():
            self.add_edge(u, v)

        # self.n = graph.number_of_nodes()
        # stores the level of each edges
        # First add all tree edges at level 0
        # Then add non-tree edges at higher levels
        # Store each forest as a nx.Graph?

        # forests = []
        # current_level = list(nx.connected_component_subgraphs(graph))
        # while current_level:
        #     next_level = []
        #     for subgraph in current_level:
        #         mst = nx.minimum_spanning_tree(subgraph)
        #         # mst_tour = find_euler_tour(mst)
        #         forests.append(mst)
        #         residual = nx.difference(subgraph, mst)
        #         if residual.number_of_edges():
        #             next_level.append(residual)
        #     current_level = next_level
        #     print('current_level = %r' % (current_level,))

    def add_edge(self, u, v):
        """
        O(log(n))
        """
        # print('add_edge u, v = %r, %r' % (u, v,))
        if self.graph.has_edge(u, v):
            return
        for node in (u, v):
            if not self.graph.has_node(node):
                self.graph.add_node(node)
            for Fi in self.forests:
                Fi.add_node(node)
        # First set the level of (u, v) to 0
        self.level[(u, v)] = 0
        # update the adjacency lists of u and v
        self.graph.add_edge(u, v)
        # If u and v are in separate trees in F_0, add e to F_0
        ru = self.forests[0].find_root(u)
        rv = self.forests[0].find_root(v)
        if ru is not rv:
            # If they are in different connected compoments merge compoments
            self.forests[0].add_edge(u, v)

    def remove_edge(self, u, v):
        """
        Using notation where 0 is top level

        Intuitively speaking, when the level of a nontree edge is increased, it
        is because we have discovered that its end points are close enough in F
        to fit in a smaller tree on a higher level.
        """
        # Remove (u, v) from represented graph
        print('Dynamically removing uv=(%r, %r)' % (u, v))
        self.graph.remove_edge(u, v)
        e = (u, v)
        # Remove edge e = (u, v) from all graphs.
        if not self.forests[0].has_edge(u, v):
            # If (u, v) is a non-tree edge, simply delete it.
            # Nothing else to do.
            return
        # If (u, v) is a tree edge we delete it and search for a replacement.
        # Delete from all higher levels
        for i in reversed(range(0, self.level[e] + 1)):
            self.forests[i].remove_edge(u, v)

        # Determine if another edge that connects u and v exists.
        # (This must be an edge r, level[r] <= level[e])
        # (Find max possible level[r] <= level[e])
        for i in reversed(range(0, self.level[e] + 1)):
            # Tu != Tw b/c (u, v) was just deleted from all forests
            Tu = self.forests[i].subtree(u)
            print('Tu = %r' % (list(Tu.nodes()),))
            Tv = self.forests[i].subtree(v)
            print('Tv = %r' % (list(Tv.nodes()),))
            # Relabel so len(Tu) <= len(Tv)
            # This ensures len(Tu) < 2 ** (floor(log(n)) - i)
            if len(Tu) > len(Tv):
                Tu, Tv = Tv, Tu
                # Note len(Tu) <= 2 * (len(Tu) + len(Tv) + 1)
            # We can afford to push all of Tu's edges to the next level and
            # still preserve invariant 1.
            seen_ = set([])
            for x in Tu.nodes():
                # Visit all edges INCIDENT (in real graph) to nodes in Tu.
                # This lets us find non-tree edges to make a tree edge
                seen_.add(x)
                for y in self.graph.neighbors(x):
                    if y in seen_:
                        continue
                    # print('Check replacement edge xy=(%r, %r)' % (x, y))
                    if y in Tv:
                        print('* Found replacement xy=(%r, %r)' % (x, y))
                        # edge (x, y) is a replacement edge.
                        # add (x, y) to prev forests F[0:i+1]
                        # This is the only place edges are added to forets of
                        # higher levels.
                        if len(self.forests) == i + 1:
                            self.forests.append(DummyEulerTourForest(self.graph.nodes()))
                        for j in range(0, i + 2):
                            print('* Add replacment to F[j=%r]' % (j,))
                            # Need euler tree augmentation for outgoing level edges
                            self.forests[j].add_edge(x, y)
                        return
                    else:
                        print('* Charging xy=(%r, %r)' % (x, y))
                        # charge --- add (x, y) to next level
                        # this pays for our search in an amortized sense
                        # (ie, the next search at this level wont consider this)
                        if len(self.forests) == i + 1:
                            self.forests.append(DummyEulerTourForest(self.graph.nodes()))
                        if self.forests[i].has_edge(x, y):
                            self.forests[i + 1].add_edge(x, y)
                        #     # assert False, 'we got it, should add it?'
                        self.level[(x, y)] = i + 1

    def is_connected(self, u, v):
        """
        Check if vertices u and v are connected.
        Query top level forest F[0] to see if u and v are in the same tree.
        This can be done by checking F_{log(n)} if Findroot(u) = Findroot(v).
        This costs O(log(n) / log(log(n))) using B-tree based Euler-Tour trees.
        but this trades off with a O(log^2(n)/log(log(n))) update
        This is O(log(n)) otherwise
        """
        ru = self.forests[0].find_root(u)
        rv = self.forests[0].find_root(v)
        return ru == rv


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
