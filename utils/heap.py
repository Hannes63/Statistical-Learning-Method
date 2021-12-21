import random

import numpy as np


class Heap:
    """
    Min-binary heap property: A[parent[i]] <= A[i].
    Max-binary heap property: A[parent[i]] >= A[i]
    parent[i] = (i-1)//2
    left[i] = 2*i - 1
    right[i] = 2*i
    """

    def __init__(self, heap_type='max'):
        assert heap_type in ('max', 'min')
        self.data = []
        self.size = 0
        self.type = heap_type

    @staticmethod
    def left(i):
        return i*2 + 1

    @staticmethod
    def right(i):
        return i*2 + 2

    @staticmethod
    def parent(i):
        return (i - 1) // 2

    def heapify(self, idx):
        """Heapify a node of a heap to maintain the heap property in \Theta(log n) time"""
        if self.type == 'min':
            # min-heapify
            while idx < self.size:
                exchange = idx
                if self.left(idx) < self.size and self.data[self.left(idx)] < self.data[exchange]:
                    exchange = self.left(idx)
                if self.right(idx) < self.size and self.data[self.right(idx)] < self.data[exchange]:
                    exchange = self.right(idx)
                if exchange == idx:
                    break
                self.data[exchange], self.data[idx] = self.data[idx], self.data[exchange]
                idx = exchange
        else:
            # max-heapify
            while idx < self.size:
                exchange = idx
                if self.left(idx) < self.size and self.data[self.left(idx)] > self.data[exchange]:
                    exchange = self.left(idx)
                if self.right(idx) < self.size and self.data[self.right(idx)] > self.data[exchange]:
                    exchange = self.right(idx)
                if exchange == idx:
                    break
                self.data[exchange], self.data[idx] = self.data[idx], self.data[exchange]
                idx = exchange

    def build_heap(self, data):
        """Build a new heap using data in \Theta(n) expected time"""
        self.data = data
        self.size = len(data)
        for i in range(self.size // 2 - 1, -1, -1):
            self.heapify(i)

    def heap_sort(self, data):
        """
        Sort the list data in \Theta(nlog n) time.
        If heap is a min-heap, the result is in descending order; else is in ascending order.
        """
        self.build_heap(data)
        for i in range(self.size, 1, -1):
            self.data[0], self.data[self.size - 1] = self.data[self.size - 1], self.data[0]
            self.size -= 1
            self.heapify(0)


class PriorityQueue(Heap):
    def __init__(self, pq_type='max'):
        super().__init__(heap_type=pq_type)

    def top(self):
        """return the maximum/minimum element of the pq if it is a max/min-pq"""
        return self.data[0]

    def extract_top(self):
        """return and delete the maximum/minimum element of the pq"""
        top = self.data[0]
        self.delete(0)
        return top

    def insert(self, value):
        """insert a new element into the pq"""
        self.data.append(float('-inf') if self.type == 'max' else float('inf'))
        self.size += 1
        idx = self.size - 1
        if self.type == 'max':
            while idx > 0 and value > self.data[self.parent(idx)]:
                self.data[idx] = self.data[self.parent(idx)]
                idx = self.parent(idx)
        else:
            while idx > 0 and value < self.data[self.parent(idx)]:
                self.data[idx] = self.data[self.parent(idx)]
                idx = self.parent(idx)
        self.data[idx] = value

    def delete(self, idx):
        """delete the element of specified index"""
        self.data[idx] = self.data[-1]
        self.size -= 1
        self.data.pop()
        self.heapify(idx)


if __name__ == '__main__':
    def tst_Heap():
        a = [5, 3, 8, 0, 8, 9]
        h1 = Heap('max')
        h1.heap_sort(a)
        print(a)
        h2 = Heap('min')
        h2.heap_sort(a)
        print(a)

    def tst_priority_queue():
        p1 = PriorityQueue('max')
        a = np.array([5, 3, 7, 6, 4, 9, 3])
        for i in a:
            p1.insert(i)
        print(p1.data)
        p1.delete(2)
        print(p1.data)
        p2 = PriorityQueue('min')
        for i in a:
            p2.insert(i)
        print(p2.data)
        p2.delete(2)
        print(p2.data)

    def tst_object():
        class Tup:
            def __init__(self, a: int, b):
                self.a = a
                self.b = b

            def __lt__(self, other):
                return self.a < other.a

            def __eq__(self, other):
                return self.a == other.a

        p = PriorityQueue()
        p.insert(Tup(2, 3))
        p.insert(Tup(5, 1))
        p.insert(Tup(4, 0))
        print(p.extract_top().a)
        print(p.extract_top().a)
        print(p.extract_top().a)



    # tst_Heap()
    # tst_priority_queue()
    tst_object()


