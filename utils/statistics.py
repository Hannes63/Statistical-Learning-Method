import numpy as np
import random
import math


def select(data: np.ndarray, n, axis=0):
    """
    Use quick select algorithm to select the n-smallest number in specified dimension.
    Time complexity: Expected case: Θ(num), worst case: Θ(n); the randomization helps avoid extreme case.
    Space complexity: Θ(num), out-place alg.; it could be modified into in-place alg.,
    but would mess up the original data.
    :param data: an ndarray of numbers in 1-dimension or 2-dimension.
    :param n: select the n-smallest number among the input (in specified axis if data.ndim == 2).
    :param axis: specified axis if data.ndim == 2.
    :return: the subscript of n-smallest element.
    """

    def partition(tups, left, right):
        # randomization to avoid worst case
        rd = random.randrange(left, right + 1)
        tups[rd], tups[right] = tups[right], tups[rd]

        key = tups[right][0]
        i = left - 1
        for j in range(left, right):
            if tups[j][0] < key:
                i += 1
                tups[j], tups[i] = tups[i], tups[j]
        i += 1
        tups[right], tups[i] = tups[i], tups[right]
        return i

    n -= 1
    num = len(data)
    if data.ndim == 2:
        tuples = list(zip(data[:, axis], np.arange(num)))
    elif data.ndim == 1:
        tuples = list(zip(data, np.arange(num)))
    else:
        assert 0
    left, right = 0, num - 1
    while left < right:
        mid = partition(tuples, left, right)
        if n == mid:
            break
        elif n < mid:
            right = mid - 1
        else:
            left = mid + 1
    return tuples[n][1]


def median(data: np.ndarray, axis=0, mod='lower'):
    """
    Find the index of median of an 1d or 2d array.
    :param data: an ndarray of numbers in 1-dimension or 2-dimension
    :param axis: the median of specified axis if data.ndim == 2
    :param mod: choose the upper median or lower median if the number of median is even
    :return: the index of median
    """
    assert mod in ['lower', 'upper']
    num = len(data) + 1
    mid = math.floor(num/2) if mod == 'lower' else math.ceil(num/2)
    return select(data, mid, axis=axis)


if __name__ == '__main__':
    def tst_select():
        data = np.array([5, 3, 7, 9])
        ans = select(data, 2)
        print(ans)

        data2 = np.array([[2, 3, 4], [4, 1, 3], [7, 5, 2]])
        ans2 = select(data2, 1, axis=1)
        print(ans2)

    def tst_median():
        data = np.array([[2, 3, 4], [4, 1, 3], [7, 5, 2], [0, 0, 0]])
        ans = median(data, axis=0, mod='upper')
        ans2 = median(data, axis=0, mod='lower')
        print(ans, ans2)

    def tst_time():
        import time
        cnt = 0
        for i in range(100):
            start_time = time.time()
            data = np.random.rand(40000, 10)
            median(data, axis=1, mod='upper')
            cnt += time.time() - start_time
        print('Total running time:', cnt, 's')

    tst_time()
