"""O(1) LRU cache implemented with a dict and doubly linked list."""


class _Node:
    """Internal doubly linked list node."""

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """Least Recently Used cache.

    get(key) returns the stored value or -1 when absent.
    put(key, value) inserts or updates a value and evicts the least recently
    used item when capacity is exceeded.

    Both operations are O(1). This implementation does not use OrderedDict or
    functools.lru_cache.
    """

    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("capacity must be non-negative")

        self.capacity = capacity
        self._cache = {}

        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

    def get(self, key):
        node = self._cache.get(key)
        if node is None:
            return -1

        self._move_to_front(node)
        return node.value

    def put(self, key, value):
        node = self._cache.get(key)

        if node is not None:
            node.value = value
            self._move_to_front(node)
            return

        if self.capacity == 0:
            return

        node = _Node(key, value)
        self._cache[key] = node
        self._add_to_front(node)

        if len(self._cache) > self.capacity:
            lru = self._remove_lru()
            del self._cache[lru.key]

    def _add_to_front(self, node):
        current_first = self._head.next

        node.prev = self._head
        node.next = current_first
        self._head.next = node
        current_first.prev = node

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _move_to_front(self, node):
        self._remove(node)
        self._add_to_front(node)

    def _remove_lru(self):
        lru = self._tail.prev
        self._remove(lru)
        return lru

    def __len__(self):
        return len(self._cache)
