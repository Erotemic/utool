from __future__ import absolute_import, division, print_function
from six.moves import zip, map, range  # NOQA
import collections
import weakref
from utool import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[set]')


class _Link(object):
    __slots__ = 'prev', 'next', 'key', '__weakref__'


class OrderedSet(collections.MutableSet):
    """ Set the remembers the order elements were added
     Big-O running times for all methods are the same as for regular sets.
     The internal self.__map dictionary maps keys to links in a doubly linked list.
     The circular doubly linked list starts and ends with a sentinel element.
     The sentinel element never gets deleted (this simplifies the algorithm).
     The prev/next links are weakref proxies (to prevent circular references).
     Individual links are kept alive by the hard reference in self.__map.
     Those hard references disappear when a key is deleted from an OrderedSet.

    References:
        http://code.activestate.com/recipes/576696/
        http://code.activestate.com/recipes/576694/
        http://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
    """

    def __init__(self, iterable=None):
        self.__root = root = _Link()         # sentinel node for doubly linked list
        root.prev = root.next = root
        self.__map = {}                     # key --> link
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.__map)

    def __contains__(self, key):
        return key in self.__map

    def add(self, key):
        # Store new key in a new link at the end of the linked list
        if key not in self.__map:
            self.__map[key] = link = _Link()
            root = self.__root
            last = root.prev
            link.prev, link.next, link.key = last, root, key
            last.next = root.prev = weakref.proxy(link)

    def append(self, key):
        # Alias for add
        return self.add(key)

    def discard(self, key):
        # Remove an existing item using self.__map to find the link which is
        # then removed by updating the links in the predecessor and successors.
        if key in self.__map:
            link = self.__map.pop(key)
            link.prev.next = link.next
            link.next.prev = link.prev

    def __iter__(self):
        # Traverse the linked list in order.
        root = self.__root
        curr = root.next
        while curr is not root:
            yield curr.key
            curr = curr.next

    def __reversed__(self):
        # Traverse the linked list in reverse order.
        root = self.__root
        curr = root.prev
        while curr is not root:
            yield curr.key
            curr = curr.prev

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = next(reversed(self)) if last else next(iter(self))
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return not self.isdisjoint(other)


# alias
oset = OrderedSet
