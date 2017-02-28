from __future__ import absolute_import, division, print_function
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


class Node(ut.NiceRepr):
    """Internal object, represents a tree node."""
    # __slots__ = ['left', 'right', 'parent', 'balance', 'key', 'value']

    def __init__(self, key=None, value=None):
        self.left = None
        self.right = None
        self.parent = None
        if key is None:
            key = object()
        self.key = key
        self.value = value
        self.balance = 0

    @property
    def children(self):
        return (self.left, self.right)

    @property
    def val(self):
        return self.value

    def __nice__(self):
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
        if other:
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
    CommandLine:
        python -m utool.experimental.euler_tour_tree_avl EulerTourTree

    Example:
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

    Example:
        >>> # DISABLE_DOCTEST
        >>> import networkx as nx
        >>> from utool.experimental.euler_tour_tree_avl import *  # NOQA
        >>> edges = [
        >>>     ('R', 'A'), ('R', 'B'),
        >>>     ('B', 'C'), ('C', 'D'), ('C', 'E'),
        >>>     ('B', 'F'), ('B', 'G'),
        >>> ]
        >>> tour = euler_tour_dfs(nx.Graph(edges))
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

    def _assert_nodes(self):
        for count, node in enumerate(self._traverse_nodes()):
            if node.left:
                assert node.left.parent is node, 'left child problem, %d' % count
            if node.right:
                assert node.right.parent is node, 'right child problem, %d' % count
            if node.parent:
                assert node in node.parent.children, 'parent problem, %d' % count

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
            graph.node[u]['value'] = node.value
            if labels:
                label = ','.join([str(getattr(node, k)) for k in labels])
                graph.node[u]['label'] = label
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

    def show_nx(self, labels=['value'], edge_labels=False, fnum=None):
        import plottool as pt
        graph = self.to_networkx(labels=labels, edge_labels=edge_labels)
        pt.show_nx(graph, fnum=fnum)

    def join(self, other):
        self.root = avl_join2(self.root, other.root)
        other.root = None
        return self

    def print_tree(self):
        ascii_tree(self.root)


def ascii_tree(root, name=None):
    import drawtree
    import ubelt as ub
    if hasattr(root, 'root'):
        root = root.root
    with ub.CaptureStdout() as cap:
        drawtree.drawtree.drawtree(root)
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


def avl_rotate_single(root, direction):
    """
    Single rotation, either 0 (left) or 1 (right).

    Figure:
                a,0 (left)
                ---------->
          a                   b
           \                /   \
            b             a       c
             \
              c
    """
    other_side = 1 - direction
    save = root[other_side]
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
    """
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
            t_merge = avl_new_top(rest, t_rotate, large, other_side)
            if _DEBUG_JOIN_DIR:
                ascii_tree(t_merge, 't_merge')
            new_root = avl_rotate_single(t_merge, other_side)
            if _DEBUG_JOIN_DIR:
                ascii_tree(new_root, 'new_root')
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
            assert False
        t__ = avl_new_top(t_, rest, large, direction)
        if height(t_) <= hrest + 1:
            if _DEBUG_JOIN_DIR:
                print('JOIN DIR (Case 3)')
            return t__
        else:
            if _DEBUG_JOIN_DIR:
                print('JOIN DIR (Case 4)')
            return avl_rotate_single(t__, other_side)


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
        return node
    elif t1 is None:
        # FIXME keep track of count if possible
        if DEBUG_JOIN:
            print('Join Case 2')
        return avl_insert_dir(t2, node, 1)
    elif t2 is None:
        if DEBUG_JOIN:
            print('Join Case 3')
        return avl_insert_dir(t1, node, 0)

    h1 = height(t1)
    h2 = height(t2)
    if h1 > h2 + 1:
        if DEBUG_JOIN:
            print('Join Case 4')
        top = avl_join_dir_recursive(t1, t2, node, 1)
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
    root, left, right = avl_release_children(root)
    if right is None:
        new_root, last_node = left, root
    else:
        new_right, last_node = avl_split_last(right)
        new_root = avl_join(left, new_right, root)
    return (new_root, last_node)


def avl_join2(t1, t2):
    """
    join two trees without any intermediate key

    Returns:
        Node: new_root

    O(log(n) + log(m)) = O(r(t1) + r(t2))

    For AVL-Trees the rank r(t1) = height(t1) - 1
    """
    if t1 is None:
        return t2
    else:
        new_left, last_node = avl_split_last(t1)
        return avl_join(new_left, t2, last_node)


def avl_release_children(root):
    left, right = root.left, root.right
    if left:
        left.parent = None
    if right:
        right.parent = None
    root.balance = 0
    root.left = None
    root.right = None
    return root, left, right


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
    Example:
        >>> from utool.experimental.euler_tour_tree_avl import *  # NOQA
        >>> self = EulerTourTree(range(10))
        >>> self._assert_nodes()
        >>> root = self.root
        >>> node = self.get_node(5)
        >>> self.print_tree()
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


def avl_split(root, node):
    """
    O(log(n))

    Args:
        root (Node): tree root
        node (Node): node to split at
    Returns:
        puple: (tl, tr)
            tl contains all keys in the tree less than node
            tr contains all keys in the tree greater than node
    """
    # Get the backtrace to the root
    rpath = backtrace_root(node)
    # We start by knowing where the node is
    l, r = avl_release_children(node)


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
        l, r = root.left, root.right
        t_key = root.key
        t_val = root.value
        if key == t_key:
            if DEBUG_SPLIT:
                print('Split Case Hit')
            part1 = l
            part2 = r
            b = True
            bv = t_val
        elif key < t_key:
            if DEBUG_SPLIT:
                print('Split Case Recurse 1')
            ll, lr, b, bv = avl_split(l, key)
            if DEBUG_SPLIT:
                print('Split Case Up 1')
            new_right = avl_join(lr, r, t_key, t_val)
            part1 = ll
            part2 = new_right
        else:
            if DEBUG_SPLIT:
                print('Split Case Recurse 2')
            rl, rr, b, bv = avl_split(r, key)
            if DEBUG_SPLIT:
                print('Split Case Up 2')
            new_left = avl_join(l, rl, t_key, t_val)
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
    assert new_node.parent is None
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
