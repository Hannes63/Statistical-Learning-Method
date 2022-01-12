
import numpy as np


class Node:
    def __init__(self):
        """
        child: a dictionary of children of a node
        parent: the parent node of a node
        group: if node.group is not None, then the node is a leaf node, and it represents the group tag
        feature: if node.feature is not None, then this node is an internal node, and it represents that
        this feature (the index of the feature, 0, 1, etc.) decides the branch of this node.
        """
        self.child = {}
        self.parent = None
        self.group = None
        self.feature = None


class DecisionTree:
    # The pruning function is not implemented yet.
    def __init__(self, min_gain=0.1, algorithm='C4.5'):
        """
        Use ID3 or C4.5 algorithm to construct a decision tree. (Algorithm 5.2 and 5.3 in page 76-78)
        :param min_gain: the threshold in step (4).
        :param algorithm: choose ID3 or C4.5 algorithm. It recommends to use ID3 when the values of each feature
        is small; otherwise, use C4.5 algo.
        """
        self.root = None
        self.min_gain = min_gain
        self.algorithm = algorithm
        assert algorithm in ('ID3', 'C4.5')

    def fit(self, X, y):
        """
        Fit the data.
        :param X: array-like of shape (n_samples, n_features)
        :param y: array-like of shape (n_samples, )
        """
        features = list(range(len(X[0])))
        self.root = self._feature_selection(X, y, features, None)
        self._pruning()

    def predict(self, X0):
        """
        Predict the group of X0 by searching the decision tree.
        :param X0: array-like of shape (n_samples, )
        :return: the predicted group of X0
        """
        assert self.root is not None
        node = self.root
        while node.group is None:
            if X0[node.feature] not in node.child:
                # when there is no child node for this value of the feature, it could choose any current child nodes
                node = next(iter(node.child.values()))
            else:
                node = node.child[X0[node.feature]]
        return node.group

    def _feature_selection(self, X, y, features, parent):
        """
        Recursively construct the decision tree.
        :param X: array-like of shape (n_samples, n_features)
        :param y: array-like of shape (n_samples, )
        :param features: list of integers, represent the remaining features, like [0, 1, 3, 4]
        :param parent: the pointer to the parent node (will be used in pruning process)
        :return: the current node
        """
        node = Node()
        H = self.entropy(y)
        N = len(y)
        # if use ID3 algorithm, max_info_gain stores the maximum information gain
        # if use C4.5 algorithm, max_info_gain means the maximum information gain ratio
        max_index = 0
        max_info_gain = 0

        y_count = {}
        for i in range(N):
            if y[i] not in y_count:
                y_count[y[i]] = 1
            else:
                y_count[y[i]] += 1
        if len(y_count) <= 1:
            # algorithm 5.2, step (1)
            node.group = y[0]
            return node

        y_max = 0
        y_max_group = 0
        for key, value in y_count.items():
            if value > y_max:
                y_max = value
                y_max_group = key
        if len(features) == 0:
            # step (2)
            node.group = y_max_group
            return node

        for i in range(len(features)):
            # step (3)
            info_gain = H
            cond_list = {}
            for k in range(N):
                if X[k][i] not in cond_list:
                    cond_list[X[k][i]] = []
                cond_list[X[k][i]].append(y[k])
            for value in cond_list.values():
                h = self.entropy(value)
                info_gain -= len(value) / N * h
            if self.algorithm == 'C4.5':
                x_i = [X[k][i] for k in range(N)]
                Hx_i = self.entropy(x_i)
                if Hx_i != 0:
                    # if Hx_i == 0, then there is only one group of property i, so info_gain = 0.
                    info_gain /= Hx_i
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_index = i

        if max_info_gain < self.min_gain:
            # step (4)
            node.group = y_max_group
            return node
        else:
            # step (5), (6)
            node.feature = max_index
            features_removed = features.copy()
            features_removed.remove(max_index)
            branch_X = {}
            branch_y = {}
            for i in range(N):
                if X[i][max_index] not in branch_X:
                    branch_X[X[i][max_index]] = [X[i]]
                    branch_y[X[i][max_index]] = [y[i]]
                else:
                    branch_X[X[i][max_index]].append(X[i])
                    branch_y[X[i][max_index]].append(y[i])
            for key in branch_X.keys():
                node.child[key] = self._feature_selection(branch_X[key], branch_y[key], features_removed, node)
        return node

    def _pruning(self):
        pass

    @staticmethod
    def entropy(x):
        """
        Calculate the empirical entropy of x.
        H = -\sum_1^n p_i log p_i
        :param x: array-like of shape (N, )
        :return: the empirical entropy of x.
        """
        count = {}
        N = len(x)
        for key in x:
            if key not in count:
                count[key] = 1
            else:
                count[key] += 1
        ret = 0
        for n in count.values():
            p = n / N
            ret -= p * np.log2(p)
        return ret

