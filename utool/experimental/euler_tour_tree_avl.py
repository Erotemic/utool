# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import networkx as nx
import utool as ut

MAXSTACK = 32


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
            parent, kids = stack[-1]
            try:
                child = next(kids)
                if child not in visited:
                    # yielder += [[parent, child]]
                    yielder += [parent]
                    visited.add(child)
                    stack.append((child, iter(G[child])))
            except StopIteration:
                if len(stack) > 0:
                    last = stack[-1]
                    yielder += [last[0]]
                stack.pop()
    return yielder


def euler_tour(G, node=None, seen=None, visited=None):
    """
    definition from
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.192.8615&rep=rep1&type=pdf

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.experimental.euler_tour_tree_avl import *  # NOQA
        >>> edges = [
        >>>     ('R', 'A'), ('R', 'B'),
        >>>     ('B', 'C'), ('C', 'D'), ('C', 'E'),
        >>>     ('B', 'F'), ('B', 'G'),
        >>> ]
        >>> G = nx.Graph(edges)
        >>> node = list(G.nodes())[0]
        >>> et1 = euler_tour(G, node)
        >>> et2 = euler_tour_dfs(G, node)
    """
    if node is None:
        node = next(G.nodes())
    if visited is None:
        assert nx.is_tree(G)
        visited = []
    if seen is None:
        seen = set([])
    visited.append(node)
    for c in G.neighbors(node):
        if c in seen:
            continue
        seen.add(c)
        euler_tour(G, c, seen, visited)
        visited.append(node)
    return visited


class Node(ut.NiceRepr):
    """Internal object, represents a tree node."""
    # __slots__ = ['left', 'right', 'parent', 'balance', 'key', 'value']

    def __init__(self, key=None, value=None):
        self.left = None
        self.right = None
        self.parent = None
        if key is None:
            key = object()  # for networkx
        self.key = key
        self.value = value
        self.balance = 0

    @property
    def kids(self):
        return (self.left, self.right)

    def __iter__(self):
        return iter(EulerTourTree(root=self))

    @property
    def val(self):
        return self.value

    def __nice__(self):
        with_neighbors = False
        if with_neighbors:
            def value(node):
                return None if node is None else node.value
            return '({})-{}-({}, {})'.format(
                value(self.parent), self.value, value(self.left), value(self.right))
        else:
            return str(self.value)

    @property
    def xdata(self):
        """ compatibility with the C node_t struct """
        return self.balance

    @xdata.setter
    def xdata(self, data):
        """ compatibility with the C node_t struct """
        self.balance = data

    def set_child(self, direction, other):
        if other is not None:
            other.parent = self
        self[direction] = other

    def __getitem__(self, direction):
        """ direction is 0 (left) or 1 (right)."""
        return self.left if direction == 0 else self.right

    def __setitem__(self, direction, other):
        """ key is 0 (left) or 1 (right)."""
        if direction == 0:
            self.left = other
        else:
            self.right = other

    def free(self):
        """Remove all references."""
        self.left = None
        self.right = None
        self.key = None
        self.value = None


class EulerTourTree(ut.NiceRepr):
    """
    TODO: generalize out the binary tree sequence part

    CommandLine:
        python -m utool.experimental.euler_tour_tree_avl EulerTourTree

    References:
        Randomized Dynamic Graph Algorithms with Polylogarithmic Time per Operation
        Henzinger and King 1995
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.192.8615&rep=rep1&type=pdf

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> from utool.experimental.euler_tour_tree_avl import *  # NOQA
        >>> ETT = EulerTourTree
        >>> self = ETT(['a', 'b', 'c', 'b', 'd', 'b', 'a'])
        >>> self._assert_nodes()
        >>> other = ETT(['E', 'F', 'G', 'F', 'E'])
        >>> other2 = ETT(['E', 'F', 'G', 'F', 'E'])
        >>> new = self + other + other2
        >>> print(self)
        >>> print(other)
        >>> print(self + other)
        >>> print(new)
        >>> print(new + self + self)
        >>> self.print_tree()
        >>> #other.print_tree()
        >>> #self.print_tree()

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> import networkx as nx
        >>> from utool.experimental.euler_tour_tree_avl import *  # NOQA
        >>> edges = [
        >>>     ('R', 'A'), ('R', 'B'),
        >>>     ('B', 'C'), ('C', 'D'), ('C', 'E'),
        >>>     ('B', 'F'), ('B', 'G'),
        >>> ]
        >>> tour = euler_tour(nx.Graph(edges))
        >>> print(tour)
        >>> self = EulerTourTree(tour)
        >>> print(self)
        >>> assert list(self) == tour
    """
    def __init__(self, iterable=None, root=None):
        self.root = root
        if iterable is not None:
            for value in iterable:
                self.root = avl_insert_dir(self.root, Node(value=value))

    def join(self, other):
        self.root = avl_join2(self.root, other.root)
        other.root = None
        return self

    def min_elem(self):
        if self.root is None:
            raise ValueError('no min element')
        node = self.root
        while node.left is not None:
            node = node.left
        return node.value

    def reroot(self, first_node, last_node):
        """
        Notes:
            ● Pick any occurrence of the new root r.
            ● Split the tour into A and B, where B is the
            part of the tour before r.
            ● Delete the first node of A and append r.
            ● Concatenate B and A.

            To change the root of T from r to s:
                Let os denote any occurrence of s.
                Splice out the first part of the sequence ending with the
                occurrence before or,
                remove its first occurrence (or), and
                tack this on to the end of the sequence which now begins with os.
                Add a new occurrence os to the end.

        CommandLine:
            python -m utool.experimental.euler_tour_tree_avl reroot

        Ignore:
            >>> # DISABLE_DOCTEST
            >>> import networkx as nx
            >>> from utool.experimental.euler_tour_tree_avl import *  # NOQA
            >>> edges = [
            >>>     ('R', 'A'), ('R', 'B'),
            >>>     ('B', 'C'), ('C', 'D'), ('C', 'E'),
            >>>     ('B', 'F'), ('B', 'G'),
            >>> ]
            >>> edges = list(nx.balanced_tree(2, 2).edges())
            >>> tour = euler_tour(nx.Graph(edges))
            >>> self = EulerTourTree(tour)
            >>> print('old_tour = %r' % (self,))
            >>> nodes = list(self._traverse_nodes())
            >>> self.first_lookup = {node.value: node for node in nodes[::-1]}
            >>> self.last_lookup = {node.value: node for node in nodes}
            >>> new_root_val = list(self)[445 % (len(tour) - 1)]
            >>> new_root_val = 5
            >>> print('new_root_val = %r' % (new_root_val,))
            >>> first_node = self.first_lookup[new_root_val]
            >>> last_node = self.last_lookup[new_root_val]
            >>> self.reroot(first_node, last_node)
            >>> print('new_tour = %r' % (self,))
            >>> ut.quit_if_noshow()
            >>> ut.show_if_requested()
        """
        min_elem = self.min_elem()
        if min_elem  == first_node.value:
            print('Already rooted there')
            return
        # tour = list(self)
        # print('tour     = %r' % (tour,))
        # B is the part before R
        # A is the part after R (with first element removed)
        B, A, first_node = avl_split(self.root, first_node)
        print('Splice out first part of sequence ending before os')
        print('B = %r' % ([] if B is None else list(B),))
        print('Remove its first occurrence or')
        B, old_root = (B, B) if B is None else avl_split_first(B)
        print('B = %r' % ([] if B is None else list(B),))
        print('The rest of the sequence now begins with os')
        A = avl_insert_dir(A, first_node, 0)
        print('A = %r' % (list(A),))
        print('Tack the first part onto the end')
        EulerTourTree(root=A)._assert_nodes('A')
        EulerTourTree(root=B)._assert_nodes('B')
        C = avl_join2(A, B)
        EulerTourTree(root=C)._assert_nodes('C')
        print('C = %r' % (list(C),))
        print('Add a new occurrence os to the end')
        new_last = Node(value=last_node.value)
        C = avl_insert_dir(C, new_last, 1)
        print('C = %r' % (list(C),))

        EulerTourTree(root=B)._assert_nodes()
        EulerTourTree(root=A)._assert_nodes()
        # EulerTourTree(root=first_node)._assert_nodes()

        # EulerTourTree(root=B).print_tree()
        # EulerTourTree(root=A).print_tree()
        # EulerTourTree(root=first_node).print_tree()

        # B = avl_insert_dir(B, new_last, 1)
        # print('B = %r' % ([] if B is None else list(B),))
        # print('A = %r' % (list(A),))

        # EulerTourTree(root=A).print_tree()

        # old_tour_parts = [S1, R, S2]
        # old_tour = ut.flatten([list(p) for p in old_tour_parts if p])
        # print('old_tour = %r' % (old_tour,))
        # assert tour == old_tour
        # new_tour_parts = [A, B]
        # new_tour = ut.flatten([list(p) for p in new_tour_parts if p])
        print('new_tour = %r' % (list(C)))
        self.root = C

        # TODO: fix lookups
        self.last_lookup[new_last.value] = new_last

        nodes = list(self._traverse_nodes())
        new_first_lookup = {node.value: node for node in nodes[::-1]}
        new_last_lookup = {node.value: node for node in nodes[::1]}

        for key in new_last_lookup.keys():
            old_last = self.last_lookup[key]
            new_last = new_last_lookup[key]
            if old_last is not new_last:
                print('key=%r needs LAST_DICT update' % (key,))

        for key in new_last_lookup.keys():
            old_first = self.first_lookup[key]
            new_first = new_first_lookup[key]
            if old_first is not new_first:
                print('key=%r needs FIRST_DICT update' % (key,))

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def __nice__(self):
        if self.root is None:
            return '[]'
        else:
            return str(list(self))

    def __add__(self, other):
        return self.copy().join(other.copy())

    def __iadd__(self, other):
        return self.join(other.copy())

    def values(self):
        for node in self._traverse_nodes():
            yield node.value

    def __iter__(self):
        return self.values()

    def __getitem__(self, index):
        node = self.get_node(index)
        return node.value

    def get_node(self, index):
        for count, node in enumerate(self._traverse_nodes()):
            if count == index:
                return node

    def _assert_nodes(self, name=None):
        if False and self.root is not None:
            if self.root.parent is not None:
                treestr = self.get_ascii_tree()
                msg = ut.codeblock(
                    r'''
                    Root cannot have a parent.
                    name = {}
                    root = {}
                    root.parent = {}
                    '''.format(name, self.root, self.root.parent)
                )
                msg = msg + '\n' + treestr
                raise AssertionError(msg)
            assert self.root.parent is None, 'must be root'
        for count, node in enumerate(self._traverse_nodes()):
            if node.left:
                assert node.left.parent is node, 'left child problem, %d' % count
            if node.right:
                assert node.right.parent is node, 'right child problem, %d' % count
            if node.parent:
                assert node in node.parent.kids, 'parent problem, %d' % count
        if name:
            print('Nodes in {} are ok'.format(name))

    def _traverse_nodes(self):
        """ Debugging function (exposes cython nodes as dummy nodes) """
        node = self.root
        stack = []
        while stack or node is not None:
            if node is not None:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                yield node
                node = node.right

    def to_networkx(self, labels=None, edge_labels=False):
        """ Get a networkx representation of the binary search tree. """
        import networkx as nx
        graph = nx.DiGraph()
        for node in self._traverse_nodes():
            u = node.key
            graph.add_node(u)  # Minor redundancy
            # Set node properties
            graph.nodes[u]['value'] = node.value
            if labels is not None:
                label = ','.join([str(getattr(node, k)) for k in labels])
                graph.nodes[u]['label'] = label
            if node.left is not None:
                v = node.left.key
                graph.add_node(v)
                graph.add_edge(u, v)
                if edge_labels:
                    graph.edge[u][v]['label'] = 'L'
            if node.right is not None:
                v = node.right.key
                graph.add_node(v)
                graph.add_edge(u, v)
                if edge_labels:
                    graph.edge[u][v]['label'] = 'R'
        return graph

    @property
    def repr_tree(self):
        """
        reconstruct represented tree as a DiGraph to
        preserve the current rootedness
        """
        import utool as ut
        import networkx as nx
        repr_tree = nx.DiGraph()
        for u, v in ut.itertwo(self.values()):
            if not repr_tree.has_edge(v, u):
                repr_tree.add_edge(u, v)
        return repr_tree

    def show_nx(self, labels=['value'], edge_labels=False, fnum=None):
        import plottool as pt
        graph = self.to_networkx(labels=labels, edge_labels=edge_labels)
        pt.show_nx(graph, fnum=fnum)

    def print_tree(self):
        ascii_tree(self.root)

    def get_ascii_tree(self):
        import drawtree
        import ubelt as ub
        root = self.root
        with ub.CaptureStdout() as cap:
            drawtree.drawtree.drawtree(root)
        return cap.text


def ascii_tree(root, name=None):
    import drawtree
    import ubelt as ub
    if hasattr(root, 'root'):
        root = root.root
    with ub.CaptureStdout() as cap:
        drawtree.drawtree.drawtree(root)
    if name is not None:
        print('+---')
        print('Tree(%s)' % (name,))
    print(cap.text)
    # if False:
    #     # Modified BFS with placeholders
    #     yielder = []
    #     queue = [root]
    #     while queue:
    #         node = queue.pop(0)
    #         if node is None:
    #             yielder.append(None)
    #         else:
    #             yielder.append(node)
    #             queue.append(node.left)
    #             queue.append(node.right)
    #     # Generate code for drawtree
    #     # import itertools as it
    #     # counter = it.count(0)
    #     # import six
    #     # c = ut.partial(six.next, counter)
    #     sequence = [
    #         # '#' if n is None else str(c())
    #         '#' if n is None else str(n.value)
    #         for n in yielder
    #     ]
    #     code = ','.join(sequence)
    #     # code = code.rstrip('#')
    #     if name is not None:
    #         print('+---')
    #         print('Tree(%s)' % (name,))
    #     drawtree.draw_level_order('{' + code + '}')
    #     print([(n.value, n.balance) for n in yielder if n is not None])
    if name is not None:
        print('L___')


def height(node):
    return node.balance if node is not None else -1


def avl_release_kids(node):
    """
    splits a node from its kids maintaining parent pointers
    """
    left, right = node.left, node.right
    if left is not None:
        # assert left.parent is node
        left.parent = None
    if right is not None:
        # assert right.parent is node
        right.parent = None
    node.balance = 0
    node.left = None
    node.right = None
    return node, left, right


def avl_release_parent(node):
    """
    removes the parent of a child
    """
    parent = node.parent
    if parent is not None:
        if parent.right is node:
            parent.right = None
        elif parent.left is node:
            parent.left = None
        else:
            raise AssertionError('impossible state')
        node.parent = None
        parent.balance = max(height(parent.right), height(parent.left)) + 1
    return node, parent


def avl_rotate_single(root, direction):
    r"""
    Single rotation, either 0 (left) or 1 (right).

    Figure:
                a,0 (left)
                ---------->
          a                   b
           \                /   \
            b             a       c
             \
              c

    a = root
    save = root.right
    """
    other_side = 1 - direction
    save = root[other_side]
    save.parent = root.parent
    # root[other_side] = save[direction]
    # save[direction] = root
    root.set_child(other_side, save[direction])
    save.set_child(direction, root)
    rlh = height(root.left)
    rrh = height(root.right)
    slh = height(save[other_side])
    root.balance = max(rlh, rrh) + 1
    save.balance = max(slh, root.balance) + 1
    return save


def avl_rotate_double(root, direction):
    r"""
    Double rotation, either 0 (left) or 1 (right).

    Figure:
                    c,1 (right)
                    ----------->
           a              a             c
          /      b,0     /     a,1    /   \
         b       --->   b      -->  b      a
          \            /
           c          c
    """
    other_side = 1 - direction
    root[other_side] = avl_rotate_single(root[other_side], other_side)
    return avl_rotate_single(root, direction)


DEBUG_JOIN = 0
_DEBUG_JOIN_DIR = 0


def avl_join_dir_recursive(t1, t2, node, direction):
    """
    Recursive version of join_left and join_right
    TODO: make this iterative using a stack
    """
    other_side = 1 - direction
    if _DEBUG_JOIN_DIR:
        print('--JOIN DIR (dir=%r) --' % (direction,))
        ascii_tree(t1, 't1')
        ascii_tree(t2, 't2')

    if direction == 0:
        large, small = t2, t1
    elif direction == 1:
        large, small = t1, t2
    else:
        assert False

    # Follow the spine of the larger tree
    spine = large[direction]
    rest = large[other_side]
    # k_, v_ = large.key, large.value

    hsmall = height(small)
    hspine = height(spine)
    hrest = height(rest)

    if _DEBUG_JOIN_DIR:
        ascii_tree(spine, 'spine')
        ascii_tree(rest, 'rest')
        ascii_tree(small, 'small')

    if hspine <= hsmall + 1:
        t_ = avl_new_top(small, spine, node, direction)
        if _DEBUG_JOIN_DIR:
            print('JOIN DIR (BASE)')
            ascii_tree(t_, 't_')
        if height(t_) <= hrest + 1:
            if _DEBUG_JOIN_DIR:
                print('JOIN DIR (Case 1)')
            return avl_new_top(t_, rest, large, direction)
        else:
            # Double rotation, but with a new node
            if _DEBUG_JOIN_DIR:
                print('JOIN DIR (Case 2)')
            t_rotate = avl_rotate_single(t_, direction)
            if _DEBUG_JOIN_DIR:
                ascii_tree(t_rotate, 't_rotate')
                EulerTourTree(root=t_rotate)._assert_nodes('t_rotate')
            t_merge = avl_new_top(rest, t_rotate, large, other_side)
            if _DEBUG_JOIN_DIR:
                ascii_tree(t_merge, 't_merge')
                EulerTourTree(root=t_merge)._assert_nodes('t_merge')
            new_root = avl_rotate_single(t_merge, other_side)
            if _DEBUG_JOIN_DIR:
                ascii_tree(new_root, 'new_root')
                EulerTourTree(root=new_root)._assert_nodes('new_root')
            return new_root
    else:
        # Traverse down the spine in the appropriate direction
        if _DEBUG_JOIN_DIR:
            print('JOIN DIR (RECURSE)')
        if direction == 0:
            t_ = avl_join_dir_recursive(small, spine, node, direction)
        elif direction == 1:
            t_ = avl_join_dir_recursive(spine, t2, node, direction)
        else:
            raise AssertionError('invalid direction')
        t__ = avl_new_top(t_, rest, large, direction)
        if height(t_) <= hrest + 1:
            if _DEBUG_JOIN_DIR:
                print('JOIN DIR (Case 3)')
            return t__
        else:
            if _DEBUG_JOIN_DIR:
                print('JOIN DIR (Case 4)')
            return avl_rotate_single(t__, other_side)
    assert False, 'should never get here'


def avl_join(t1, t2, node):
    """
    Joins two trees `t1` and `t1` with an intermediate key-value pair

    CommandLine:
        python -m utool.experimental.euler_tour_tree_avl avl_join

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.experimental.euler_tour_tree_avl import *  # NOQA
        >>> self = EulerTourTree(['a', 'b', 'c', 'b', 'd', 'b', 'a'])
        >>> other = EulerTourTree(['E', 'F', 'G', 'F', 'E'])
        >>> node = Node(value='Q')
        >>> root = avl_join(self.root, other.root, node)
        >>> new = EulerTourTree(root=root)
        >>> print('new = %r' % (new,))
        >>> ut.quit_if_noshow()
        >>> self.print_tree()
        >>> other.print_tree()
        >>> new.print_tree()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.experimental.euler_tour_tree_avl import *  # NOQA
        >>> self = EulerTourTree(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
        >>> other = EulerTourTree(['X'])
        >>> node = Node(value='Q')
        >>> root = avl_join(self.root, other.root, node)
        >>> new = EulerTourTree(root=root)
        >>> print('new = %r' % (new,))
        >>> ut.quit_if_noshow()
        >>> ut.qtensure()
        >>> #self.show_nx(fnum=1)
        >>> #other.show_nx(fnum=2)
        >>> new.show_nx()

    Running Time:
        O(abs(r(t1) - r(t2)))
        O(abs(height(t1) - height(t2)))
    """
    if DEBUG_JOIN:
        print('-- JOIN node=%r' % (node,))

    if t1 is None and t2 is None:
        if DEBUG_JOIN:
            print('Join Case 1')
        top = node
    elif t1 is None:
        # FIXME keep track of count if possible
        if DEBUG_JOIN:
            print('Join Case 2')
        top = avl_insert_dir(t2, node, 0)
    elif t2 is None:
        if DEBUG_JOIN:
            print('Join Case 3')
        top = avl_insert_dir(t1, node, 1)
    else:
        h1 = height(t1)
        h2 = height(t2)
        if h1 > h2 + 1:
            if DEBUG_JOIN:
                print('Join Case 4')
            top = avl_join_dir_recursive(t1, t2, node, 1)
            if DEBUG_JOIN:
                ascii_tree(t1, 'top')
        elif h2 > h1 + 1:
            if DEBUG_JOIN:
                print('Join Case 5')
                ascii_tree(t1)
                ascii_tree(t2)

            top = avl_join_dir_recursive(t1, t2, node, 0)
            if DEBUG_JOIN:
                ascii_tree(top)
        else:
            if DEBUG_JOIN:
                print('Join Case 6')
            # Insert at the top of the tree
            top = avl_new_top(t1, t2, node, 0)
    return top


def avl_split_last(root):
    """
    Removes the maximum element from the tree

    Returns:
        tuple: new_root, last_node

    O(log(n)) = O(height(root))
    """
    if root is None:
        raise IndexError('Empty tree has no maximum element')
    root, left, right = avl_release_kids(root)
    if right is None:
        new_root, last_node = left, root
    else:
        new_right, last_node = avl_split_last(right)
        new_root = avl_join(left, new_right, root)
    return (new_root, last_node)


def avl_split_first(root):
    """
    Removes the minimum element from the tree

    Returns:
        tuple: new_root, first_node

    O(log(n)) = O(height(root))
    """
    if root is None:
        raise IndexError('Empty tree has no maximum element')
    root, left, right = avl_release_kids(root)
    if left is None:
        new_root, first_node = right, root
    else:
        new_left, first_node = avl_split_first(left)
        new_root = avl_join(new_left, right, root)
    return (new_root, first_node)


def avl_join2(t1, t2):
    """
    join two trees without any intermediate key

    Returns:
        Node: new_root

    O(log(n) + log(m)) = O(r(t1) + r(t2))

    For AVL-Trees the rank r(t1) = height(t1) - 1
    """
    if t1 is None and t2 is None:
        new_root = None
    elif t2 is None:
        new_root = t1
    elif t1 is None:
        new_root = t2
    else:
        new_left, last_node = avl_split_last(t1)

        debug = 0

        if debug:
            EulerTourTree(root=new_left)._assert_nodes('new_left')
            EulerTourTree(root=last_node)._assert_nodes('last_node')
            EulerTourTree(root=t2)._assert_nodes('t2')

            print('new_left')
            EulerTourTree(root=new_left).print_tree()

            print('last_node')
            EulerTourTree(root=last_node).print_tree()

            print('t2')
            EulerTourTree(root=t2).print_tree()

        new_root = avl_join(new_left, t2, last_node)

        if debug:
            print('new_root')
            EulerTourTree(root=new_root).print_tree()
            EulerTourTree(root=last_node)._assert_nodes('new_root')
    return new_root


def avl_new_top(t1, t2, top, direction=0):
    """
    if direction == 0:
        (t1, t2) is (left, right)
    if direction == 1:
        (t1, t2) is (right, left)
    """
    top.parent = None
    assert top.parent is None, str(top.parent.value)
    top.set_child(direction, t1)
    top.set_child(1 - direction, t2)
    top.balance = max(height(t1), height(t2)) + 1
    return top


DEBUG_SPLIT = 0


def backtrace_root(node):
    """
    Ignore:
        >>> from utool.experimental.euler_tour_tree_avl import *  # NOQA
        >>> self = EulerTourTree(range(10))
        >>> self._assert_nodes()
        >>> root = self.root
        >>> node = self.get_node(5)
        >>> self.print_tree()
        >>> print('node = %r' % (node,))
        >>> rpath = backtrace_root(node)
        >>> print('rpath = %r' % (rpath,))
    """
    # Trace path to the root
    rpath = []
    prev = node
    now = node.parent
    while now is not None:
        if now.left is prev:
            rpath.append((now, 0))
        elif now.right is prev:
            rpath.append((now, 1))
        else:
            raise AssertionError('impossible state')
        prev = now
        now = now.parent
    return rpath


def test_avl_split(verbose=1):
    for num in range(0, 20):
        for index in range(num):
            if verbose:
                print('------')
                print('num = %r' % (num,))
                print('index = %r' % (index,))
            try:
                tree0 = EulerTourTree(ut.chr_range(num))
                tour = list(tree0)
                tree0._assert_nodes()
                if verbose >= 2:
                    tree0.print_tree()
                if verbose:
                    print('tree0 = %r' % (tree0,))
                node = tree0.get_node(index)
                if verbose:
                    print('node = %s' % (node,))
                part1, part2, bnode = avl_split(tree0.root, node)
                tree1 = EulerTourTree(root=part1)
                tree2 = EulerTourTree(root=part2)
                if verbose >= 2:
                    tree1.print_tree(), tree2.print_tree()
                if verbose:
                    print('tree1 = %r' % (tree1,))
                    print('tree2 = %r' % (tree2,))
                # Should correspond to a split in the tour list
                assert bnode.left is None, 'bnode must be split'
                assert bnode.right is None, 'bnode must be split'
                assert bnode.parent is None, 'bnode must be split'
                assert bnode is node, 'node must be same'
                ut.assert_eq(list(tree1), tour[:index])
                ut.assert_eq(list(tree2), tour[index + 1:])
                tree1._assert_nodes(), tree2._assert_nodes()
            except Exception:
                print('num = %r' % (num,))
                print('index = %r' % (index,))
                raise


def avl_split(root, node):
    """
    O(log(n))

    Args:
        root (Node): tree root
        node (Node): node to split at

    Returns:
        puple: (tl, tr, node)
            tl contains all keys in the tree less than node
            tr contains all keys in the tree greater than node
            node is the node we split out

    CommandLine:
        python -m utool.experimental.euler_tour_tree_avl avl_split

    Ignore:
        >>> from utool.experimental.euler_tour_tree_avl import *  # NOQA
        >>> self = EulerTourTree(ut.chr_range(10))
        >>> self.print_tree()
        >>> node = self.get_node(5)
        >>> part1, part2, bnode = avl_split(self.root, node)
        >>> ascii_tree(part1)
        >>> ascii_tree(part2)
        >>> ascii_tree(bnode)

    Ignore:
        >>> from utool.experimental.euler_tour_tree_avl import *  # NOQA
        >>> test_avl_split(verbose=2)
    """
    DEBUG_SPLIT = 0

    # Get the backtrace to the root
    rpath = backtrace_root(node)
    if len(rpath) > 0:
        assert rpath[-1][0] is root
    if DEBUG_SPLIT:
        print('======== SPLIT (PY)')
        print('rpath = %s' % (rpath,))
        print('node = %s' % (node,))

    # We start by knowing where the node is
    # This is the base case of the recursive function
    bnode, part1, part2 = avl_release_kids(node)
    assert bnode is node
    if DEBUG_SPLIT:
        print('bnode = %s' % (bnode,))
        print(' * part1 = %s' % (part1,))
        print(' * part2 = %s' % (part2,))
    avl_release_parent(bnode)

    # We have split out the node we care about.
    # Now, we need to recombine the tree in an ordered fashion

    # Retrace the the stack that would have been
    # generated by the old recursive key-based split
    for count, (node, direction) in enumerate(rpath):
        if DEBUG_SPLIT:
            print('+--- Iter {}'.format(count))
            print(' * node = %s' % (node,))
            print(' * direction = %r' % (direction,))
        node, left, right = avl_release_kids(node)
        avl_release_parent(node)
        if DEBUG_SPLIT:
            print(' * left = %s' % (left,))
            print(' * right = %s' % (right,))
        # At `node` we would have decided to go `direction`
        if direction == 0:
            # left is case 1
            if DEBUG_SPLIT:
                print(' * Case 1')
                print(' * Join %s + %s + %s' % (part2, node, right))
            new_right = avl_join(part2, right, node)
            part1 = part1
            part2 = new_right
        elif direction == 1:
            # right is case 1
            if DEBUG_SPLIT:
                print(' * Case 2')
                print(' * Join %s + %s + %s' % (left, node, part1))
            new_left = avl_join(left, part1, node)
            part1 = new_left
            part2 = part2
        else:
            raise AssertionError('impossible state')
        if DEBUG_SPLIT:
            print('     * part1 = %s' % (part1,))
            print('     * part2 = %s' % (part2,))
            print('+--- End Iter {}'.format(count))
    if DEBUG_SPLIT:
        print('RETURN')
        print(' * part1 = %s' % (part1,))
        print(' * part2 = %s' % (part2,))
        print(' * bnode = %s' % (bnode,))
    return (part1, part2, bnode)


def avl_split_old(root, key):
    if DEBUG_SPLIT:
        print('-- SPLIT (PY)')
        print('root = %r' % (root if root is None else root.key,))
        print('key = %r' % (key,))
        pass
    # TODO: keep track of the size of the sets being avl_split if possible
    if root is None:
        if DEBUG_SPLIT:
            print("Split Case None")
        part1 = root
        part2 = root
        b = False
        bv = None
    else:
        left, right = root.left, root.right
        t_key = root.key
        t_val = root.value
        if key == t_key:
            if DEBUG_SPLIT:
                print('Split Case Hit')
            part1 = left
            part2 = right
            b = True
            bv = t_val
        elif key < t_key:
            if DEBUG_SPLIT:
                print('Split Case Recurse 1')
            ll, lr, b, bv = avl_split(left, key)
            if DEBUG_SPLIT:
                print('Split Case Up 1')
            new_right = avl_join(lr, right, t_key, t_val)
            part1 = ll
            part2 = new_right
        else:
            if DEBUG_SPLIT:
                print('Split Case Recurse 2')
            rl, rr, b, bv = avl_split(right, key)
            if DEBUG_SPLIT:
                print('Split Case Up 2')
            new_left = avl_join(left, rl, t_key, t_val)
            part1 = new_left
            part2 = rr
    if DEBUG_SPLIT:
        print('part1 = %r' % (None if part1 is None else part1.key,))
        print('part2 = %r' % (None if part2 is None else part2.key,))
    return (part1, part2, b, bv)


def avl_insert_dir(root, new_node, direction=1):
    """
    Inserts a single node all the way to the left (direction=1) or right (direction=1)

    CommandLine:
        python -m utool.experimental.euler_tour_tree_avl avl_insert_dir --show
        python -m utool.experimental.euler_tour_tree_avl avl_insert_dir

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.experimental.euler_tour_tree_avl import *  # NOQA
        >>> import utool as ut
        >>> root = Node(value='A')
        >>> new_node = Node(value='B')
        >>> new_root = avl_insert_dir(root, new_node, direction=1)
        >>> new_root = avl_insert_dir(root, Node(value='Z'), direction=1)
        >>> EulerTourTree(root=new_root)._assert_nodes()
        >>> for v in ut.chr_range(5, base='C'):
        >>>     new_root = avl_insert_dir(new_root, Node(value=v), direction=1)
        >>>     self = EulerTourTree(root=new_root)
        >>>     self._assert_nodes()
        >>> new = EulerTourTree(root=new_root)
        >>> print(new)
        >>> ut.quit_if_noshow()
        >>> ut.qtensure()
        >>> new.show_nx(edge_labels=True)
        >>> ut.show_if_requested()
        >>> #ascii_tree(root)
        >>> #print(result)
    """
    if root is None:
        return new_node
    assert new_node.parent is None, str((new_node, new_node.parent))
    assert new_node.left is None
    assert new_node.right is None
    assert root.parent is None

    node_stack = []  # node stack
    # dir_stack = array('I')  # direction stack
    done = False
    top = 0
    # Move all the way to the right/left in tree1
    node = root
    # search for an empty link, save path
    while True:
        # Always move to the right
        # dir_stack.append(direction)
        node_stack.append(node)
        if node[direction] is None:
            break
        node = node[direction]
    extreme_node = node

    # Insert a new node at the bottom of the tree
    extreme_node.set_child(direction, new_node)
    new_root = root

    # Walk back up the search path
    # (which for joining orderless structures was always right)
    other_side = 1 - direction
    top = len(node_stack) - 1
    while (top >= 0) and not done:
        # direction = dir_stack[top]
        # other_side = 1 - direction
        top_node = node_stack[top]
        left_height = height(top_node[direction])
        right_height = height(top_node[other_side])

        # Terminate or rebalance as necessary
        if left_height - right_height == 0:
            done = True
        if left_height - right_height >= 2:
            a = top_node[direction][direction]
            b = top_node[direction][other_side]

            # Determine which rotation is required
            if height(a) >= height(b):
                node_stack[top] = avl_rotate_single(top_node, other_side)
            else:
                node_stack[top] = avl_rotate_double(top_node, other_side)

            # Fix parent
            if top != 0:
                # d_ = dir_stack[top - 1]
                d_ = direction
                node_stack[top - 1].set_child(d_, node_stack[top])
            else:
                new_root = node_stack[0]
                new_root.parent = None
            done = True

        # Update balance factors
        top_node = node_stack[top]
        left_height = height(top_node[direction])
        right_height = height(top_node[other_side])

        top_node.balance = max(left_height, right_height) + 1
        top -= 1
    assert new_root.parent is None
    return new_root


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m utool.experimental.euler_tour_tree_avl
        python -m utool.experimental.euler_tour_tree_avl --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
