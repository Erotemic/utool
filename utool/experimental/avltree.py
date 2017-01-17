"""
https://triangleinequality.wordpress.com/2014/07/15/how-to-balance-your-binary-search-trees-avl-trees/
"""


class AVLNode(object):

    def __init__(self, key):
        self.key = key
        self.right_child = None
        self.left_child = None
        self.parent = None
        self.height = 0
        self.balance = 0

    def update_height(self, upwards=True):
        # If upwards we go up the tree correcting heights and balances,
        # if not we just correct the given node.
        if self.left_child is None:
            # Empty left tree.
            left_height = 0
        else:
            left_height = self.left_child.height + 1
        if self.right_child is None:
            # Empty right tree.
            right_height = 0
        else:
            right_height = self.right_child.height + 1
        # Note that the balance can change even when the height does not,
        # so change it before checking to see if height needs updating.
        self.balance = left_height - right_height
        height = max(left_height, right_height)
        if self.height != height:
            self.height = height
            if self.parent is not None:
                # We only need to go up a level if the height changes.
                if upwards:
                    self.parent.update_height()

    def is_left(self):
        # Handy to find out whether a node is a left or right child or neither.
        if self.parent is None:
            return self.parent
        else:
            return self is self.parent.left_child


class AVLTree(object):

    def __init__(self):
        self.root = None

    def insert(self, key, node=None):
        # The first call is slightly different.
        if node is None:
            # First call, start node at root.
            node = self.root
            if node is None:
                # Empty tree, create root.
                node = AVLNode(key=key)
                self.root = node
                return node
            else:
                ret = self.insert(key=key, node=node)
                self.balance(ret)
                return ret
        # Not a first call.
        if node.key == key:
            # No need to insert, key already present.
            return node
        elif node.key > key:
            child = node.left_child
            if child is None:
                # Reached the bottom, insert node and update heights.
                child = AVLNode(key=key)
                child.parent = node
                node.left_child = child
                node.update_height()
                return child
            else:
                return self.insert(key=key, node=child)
        elif node.key < key:
            child = node.right_child
            if child is None:
                # Reached the bottom, insert node and update heights.
                child = AVLNode(key=key)
                child.parent = node
                node.right_child = child
                return child
            else:
                return self.insert(key=key, node=child)
        else:
            raise AssertionError("This shouldn't happen.")

    def find(self, key, node=None):
        if node is None:
            # First call.
            node = self.root
            if self.root is None:
                return None
            else:
                return self.find(key, self.root)
        # Now we handle nonfirst calls.
        elif node.key == key:
            # Found the node.
            return node
        elif key < node.key:
            if node.left_child is None:
                # If key not in tree, we return a node that would be its
                # parent.
                return node
            else:
                return self.find(key, node.left_child)
        else:
            if node.right_child is None:
                return node
            else:
                return self.find(key, node.right_child)

    def delete(self, key, node=None):
        # Delete key from tree.
        if node is None:
            # Initial call.
            node = self.find(key)
            if (node is None) or (node.key != key):
                # Empty tree or key not in tree.
                return

        if (node.left_child is None) and (node.right_child is not None):
            # Has one right child.
            right_child = node.right_child
            left = node.is_left()
            if left is not None:
                parent = node.parent
                if not left:
                    parent.right_child = right_child
                else:
                    parent.left_child = right_child
                right_child.parent = parent
                self.balance(parent)
            else:
                right_child.parent = None
                self.root = right_child
                # No need to update heights or rebalance.

        elif (node.left_child is not None) and (node.right_child is None):
            # Has one left child.
            left_child = node.left_child
            left = node.is_left()
            if left is not None:
                parent = node.parent
                if left:
                    parent.left_child = left_child
                else:
                    parent.right_child = right_child
                left_child.parent = parent

                self.balance(parent)
            else:
                left_child.parent = None
                self.root = left_child
        elif node.left_child is None:
            # Has no children.
            parent = node.parent
            if parent is None:
                # Deleting a lone root, set tree to empty.
                self.root = None
            else:
                if parent.left_child is node:
                    parent.left_child = None
                else:
                    parent.right_child = None
                self.balance(parent)
        else:
            # Node has two childen, swap keys with successor node
            # and delete successor node.
            right_most_child = self.find_leftmost(node.right_child)
            node.key = right_most_child.key
            self.delete(key=node.key, node=right_most_child)
            # Note that updating the heights will be handled in the next
            # call of delete.

    def find_rightmost(self, node):
        if node.right_child is None:
            return node
        else:
            return self.find_rightmost(node.right_child)

    def find_leftmost(self, node):
        if node.left_child is None:
            return node
        else:
            return self.find_leftmost(node.left_child)

    def find_next(self, key):
        node = self.find(key)
        if (node is None) or (node.key != key):
            # Key not in tree.
            return None
        else:
            right_child = node.right_child
            if right_child is not None:
                node = self.find_leftmost(right_child)
            else:
                parent = node.parent
                while(parent is not None):
                    if node is parent.left_child:
                        break
                    node = parent
                    parent = node.parent
                node = parent
            if node is None:
                # Key is largest in tree.
                return node
            else:
                return node.key

    def find_prev(self, key):
        node = self.find(key)
        if (node is None) or (node.key != key):
            # Key not in tree.
            return None
        else:
            left_child = node.left_child
            if left_child is not None:
                node = self.find_rightmost(left_child)
            else:
                parent = node.parent
                while(parent is not None):
                    if node is parent.right_child:
                        break
                    node = parent
                    parent = node.parent
                node = parent
            if node is None:
                # Key is largest in tree.
                return node
            else:
                return node.key

    def right_rotation(self, root):
        left = root.is_left()
        pivot = root.left_child
        if pivot is None:
            return
        root.left_child = pivot.right_child
        if pivot.right_child is not None:
            root.left_child.parent = root
        pivot.right_child = root
        pivot.parent = root.parent
        root.parent = pivot
        if left is None:
            self.root = pivot
        elif left:
            pivot.parent.left_child = pivot
        else:
            pivot.parent.right_child = pivot
        root.update_height(False)
        pivot.update_height(False)

    def left_rotation(self, root):
        left = root.is_left()
        pivot = root.right_child
        if pivot is None:
            return
        root.right_child = pivot.left_child
        if pivot.left_child is not None:
            root.right_child.parent = root
        pivot.left_child = root
        pivot.parent = root.parent
        root.parent = pivot
        if left is None:
            self.root = pivot
        elif left:
            pivot.parent.left_child = pivot
        else:
            pivot.parent.right_child = pivot
        root.update_height(False)
        pivot.update_height(False)

    def balance(self, node):
        node.update_height(False)
        if node.balance == 2:
            if node.left_child.balance != -1:
                # Left-left case.
                self.right_rotation(node)
                if node.parent.parent is not None:
                    # Move up a level.
                    self.balance(node.parent.parent)
            else:
                # Left-right case.
                self.left_rotation(node.left_child)
                self.balance(node)
        elif node.balance == -2:
            if node.right_child.balance != 1:
                # Right-right case.
                self.left_rotation(node)
                if node.parent.parent is not None:
                    self.balance(node.parent.parent)
            else:
                # Right-left case.
                self.right_rotation(node.right_child)
                self.balance(node)
        else:
            if node.parent is not None:
                self.balance(node.parent)

    def sort(lst, ascending=True):
        A = AVLTree()
        for item in lst:
            A.insert(item)
        ret = []
        if ascending:
            node = A.find_leftmost(A.root)
            if node is not None:
                key = node.key
            else:
                key = node
            while (key is not None):
                ret.append(key)
                key = A.find_next(key)
        else:
            node = A.find_rightmost(A.root)
            if node is not None:
                key = node.key
            else:
                key = node
            while (key is not None):
                ret.append(key)
                key = A.find_prev(key)
        return ret

    def plot(self, balance=False):
        """
        #I also include a new plotting routine to show the balances or keys of
        the node.
        """
        # Builds a copy of the BST in igraphs for plotting.
        # Since exporting the adjacency lists loses information about
        # left and right children, we build it using a queue.
        import networkx as nx
        # import igraph as igraphs
        G = nx.Graph()
        if self.root is not None:
            G.add_node(0)
        queue = [[self.root, 0]]
        # Queue has a pointer to the node in our BST, and its index
        # in the igraphs copy.
        index = 0

        while queue:
            # At each iteration, we label the head of the queue with its key,
            # then add any children into the igraphs graph,
            # and into the queue.

            node = queue[0][0]  # Select front of queue.
            node_index = queue[0][1]
            if not balance:
                G.node[node_index]['label'] = node.key
            else:
                G.node[node_index]['label'] = node.balance
            if index == 0:
                # Label root green.
                # G.node[node_index]['color'] = 'green'
                pass
            if node.left_child is not None:
                G.add_node(node_index)
                G.add_edges_from([(node_index, index + 1)])
                queue.append([node.left_child, index + 1])
                # G.node[index + 1]['color'] = 'red'  # Left children are red.
                index += 1
            if node.right_child is not None:
                G.add_node(node_index)
                G.add_edges_from([(node_index, index + 1)])
                # G.node[index + 1]['color'] = 'blue'
                queue.append([node.right_child, index + 1])
                index += 1

            queue.pop(0)
        import plottool as pt
        pt.show_nx(G)
        # layout = G.layout_reingold_tilford(root=0)
        # igraphs.plot(G, layout=layout)


def demo_avl_tree():
    lst = [1, 1, 1, 1, 1]
    B = AVLTree()
    for item in lst:
        print("inserting", item)
        B.insert(item)
        B.plot(True)
    # print("End of inserts")
    # print("Deleting 5")
    # B.plot(True)
    # B.delete(5)
    # print("Deleting 1")
    # B.plot(True)
    # B.delete(1)
    # B.plot(False)
    # print(B.root.key == 4)
    # print(B.find_next(3) == 4)
    # print(B.find_prev(7) == 4.5)
    # print(B.find_prev(1) is None)
    # print(B.find_prev(7) == 4.5)
    # print(B.find_prev(2) is None)
    # print(B.find_prev(11) == 7)
    import utool as ut
    import plottool as pt
    pt.present()
    ut.show_if_requested()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.hots.avltree demo_avl_tree --show
        python -m ibeis.algo.hots.avltree --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
