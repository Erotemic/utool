# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from six.moves import zip, map, range  # NOQA
try:
    from collections.abc import MutableSet
except Exception:
    from collections import MutableSet
import weakref
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)


class _Link(object):
    __slots__ = ('prev', 'next', 'key', '__weakref__')


class OrderedSet(MutableSet):
    """ Set the remembers the order elements were added
     Big-O running times for all methods are the same as for regular sets.
     The internal self._map dictionary maps keys to links in a doubly linked list.
     The circular doubly linked list starts and ends with a sentinel element.
     The sentinel element never gets deleted (this simplifies the algorithm).
     The prev/next links are weakref proxies (to prevent circular references).
     Individual links are kept alive by the hard reference in self._map.
     Those hard references disappear when a key is deleted from an OrderedSet.

    References:
        http://code.activestate.com/recipes/576696/
        http://code.activestate.com/recipes/576694/
        http://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
    """

    def __init__(self, iterable=None):
        self._root = root = _Link()  # sentinel node for doubly linked list
        root.prev = root.next = root
        self._map = {}  # key --> link
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self._map)

    def __contains__(self, key):
        return key in self._map

    def add(self, key):
        """ Store new key in a new link at the end of the linked list """
        if key not in self._map:
            self._map[key] = link = _Link()
            root = self._root
            last = root.prev
            link.prev, link.next, link.key = last, root, key
            last.next = root.prev = weakref.proxy(link)

    def append(self, key):
        """ Alias for add """
        return self.add(key)

    def discard(self, key):
        # Remove an existing item using self._map to find the link which is
        # then removed by updating the links in the predecessor and successors.
        if key in self._map:
            link = self._map.pop(key)
            link.prev.next = link.next
            link.next.prev = link.prev

    def __iter__(self):
        # Traverse the linked list in order.
        root = self._root
        curr = root.next
        while curr is not root:
            yield curr.key
            curr = curr.next

    def __reversed__(self):
        # Traverse the linked list in reverse order.
        root = self._root
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

    @classmethod
    def union(cls, *sets):
        """
        >>> from utool.util_set import *  # NOQA
        """
        import utool as ut
        lists_ = ut.flatten([list(s) for s in sets])
        return cls(lists_)

    def update(self, other):
        """ union update """
        for item in other:
            self.add(item)

    def __getitem__(self, index):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> import utool as ut
            >>> self = ut.oset([1, 2, 3])
            >>> assert self[0] == 1
            >>> assert self[1] == 2
            >>> assert self[2] == 3
            >>> ut.assert_raises(IndexError, self.__getitem__, 3)
            >>> assert self[-1] == 3
            >>> assert self[-2] == 2
            >>> assert self[-3] == 1
            >>> ut.assert_raises(IndexError, self.__getitem__, -4)
        """
        if index < 0:
            iter_ = self.__reversed__
            index_ = -1 - index
        else:
            index_ = index
            iter_ = self.__iter__
        if index_ >= len(self):
            raise IndexError('index %r out of range %r' % (index, len(self)))
        for count, item in zip(range(index_ + 1), iter_()):
            pass
        return item

    def index(self, item):
        """
        Find the index of `item` in the OrderedSet

        Example:
            >>> # ENABLE_DOCTEST
            >>> import utool as ut
            >>> self = ut.oset([1, 2, 3])
            >>> assert self.index(1) == 0
            >>> assert self.index(2) == 1
            >>> assert self.index(3) == 2
            >>> ut.assert_raises(ValueError, self.index, 4)
        """
        for count, other in enumerate(self):
            if item == other:
                return count
        raise ValueError('%r is not in OrderedSet' % (item,))


# alias
oset = OrderedSet
