import numpy as np
from utils.statistics import median
from utils import heap
from time import time


class KDTree:
    c1 = 0
    c2 = 0

    def __init__(self, X, y, dim):
        """
        Information of a kd-tree node, and kd-tree operation functions.
        Compared with brute-force method (sorting method), kd-tree could decrease the time of one prediction
        from O(n log n) to O(log k * log n), where Θ(log n) is the height of the kd-tree, and O(log k) is the
        operation time of priority queue.
        :param X: ndarray of shape (n_samples, n_features).
        :param y: ndarray of shape (n_samples, ), category of X
        :param dim: the compared dimension of this node.
        """
        self.parent = None
        self.left = None
        self.right = None
        self.dim = dim
        self.X = X
        self.y = y

    @classmethod
    def construct_kd_tree(cls, X, y, axis=0, parent=None):
        """
        Construct an kd-tree recursively with input data.
        :param X: ndarray of shape (n_samples, n_features).
        :param y: ndarray of shape (n_samples,).
        :param axis: the current axis that is to calculate the median.
        :param parent: the parent node of this call.
        :return: the root of constructed tree/subtree.
        """
        if X.size <= 0:
            return None
        num, dim = X.shape
        s1 = time()
        mid = median(X, axis=axis, mod='upper')
        KDTree.c1 += time() - s1
        mid_value = X[mid][axis]
        node = KDTree(X[mid], y[mid], axis)
        node.parent = parent

        s2 = time()
        lX, ly, rX, ry = [], [], [], []
        for i in range(num):
            if X[i][axis] <= mid_value and i != mid:  # left subtree
                lX.append(X[i])
                ly.append(y[i])
            elif X[i][axis] > mid_value:  # right subtree
                rX.append(X[i])
                ry.append(y[i])
        lX, ly, rX, ry = np.array(lX), np.array(ly), np.array(rX), np.array(ry)
        KDTree.c2 += time() - s2

        node.left = cls.construct_kd_tree(lX, ly, (axis + 1) % dim, node)
        node.right = cls.construct_kd_tree(rX, ry, (axis + 1) % dim, node)
        return node

    @classmethod
    def k_nearest_neighbors(cls, node, X0, k):
        """
        Find k nearest neighbors of input value in a kd-tree.
        :param k: The number of nearest neighbors.
        :param node: root node of kd-tree.
        :param X0: array-like of shape (n_features, ), the input data point.
        :return: a list of k values whose L2-distances are the nearest.
        """
        assert k > 0
        axis = 0
        dim = len(X0)
        son = node

        # Find the leaf node in kd-tree that is the nearest to input value
        while son is not None:
            node = son
            if X0[axis] <= node.X[axis]:
                son = node.left
            else:
                son = node.right
            axis = (axis + 1) % dim
        pq = heap.PriorityQueue('max')

        # Used for priority queue
        class Tup:
            def __init__(self, d, kdnode: KDTree):
                self.d = d
                self.node = kdnode

            def __eq__(self, other):
                return self.d == other.d

            def __lt__(self, other):
                return self.d < other.d

        # L2 distance without sqrt between input and the node nd
        L_const = 2
        L2 = lambda nd: np.sum((nd.X - X0) ** L_const)

        def traversal(node):
            # traverse the tree downward
            if node is None:
                return

            nonlocal pq, X0
            if pq.size < k:
                dis = L2(node)
                pq.insert(Tup(dis, node))
                traversal(node.left)
                traversal(node.right)
            elif abs(node.X[node.dim] - X0[node.dim])**L_const > pq.top().d:
                return
            else:
                dis = L2(node)
                if dis < pq.top().d:
                    pq.extract_top()
                    pq.insert(Tup(dis, node))
                traversal(node.left)
                traversal(node.right)

        pq.insert(Tup(L2(node), node))
        while node is not None:
            # traverse the tree upward, if condition satisfied, then downward by calling traversal(node)
            son = node
            node = node.parent
            if node and (pq.size < k or abs(node.X[node.dim] - X0[node.dim])**L_const <= pq.top().d):
                dis = L2(node)
                if pq.size < k:
                    pq.insert(Tup(dis, node))
                elif dis < pq.top().d:
                    pq.extract_top()
                    pq.insert(Tup(dis, node))

                if son == node.left:
                    traversal(node.right)
                else:  # son == node.right
                    traversal(node.left)
        k_list = [i.node for i in pq.data]
        return k_list


class KNN:
    def __init__(self, k_neighbors=1):
        """
        Predict a sample by referring the k nearest neighbors, and making decision by majority.
        :param k_neighbors: the number of nearest neighbors
        """
        self.k_neighbors = k_neighbors
        self.kd_root = None

    def fit(self, X, y):
        """
        Construct the kd-tree using the input X.
        :param X: array-like of shape (n_samples, n_features)
        :param y: array-like of shape (n_samples, ), the tags of corresponding sample in X
        """
        X = np.array(X)
        y = np.array(y)
        self.kd_root = KDTree.construct_kd_tree(X, y)

    def predict(self, X0):
        """
        Predict the category of input.
        :param X0: array-like of shape (n_features, )
        :return: the category of the input X0
        """
        k_list = KDTree.k_nearest_neighbors(self.kd_root, X0, self.k_neighbors)
        vote = dict()
        for i in k_list:
            if i.y not in vote:
                vote[i.y] = 1
            else:
                vote[i.y] += 1
        largest = 0
        pred_cat = None
        for category, count in vote.items():
            if count > largest:
                largest = count
                pred_cat = category
        return pred_cat



