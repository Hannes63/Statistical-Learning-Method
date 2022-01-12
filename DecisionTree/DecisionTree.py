
import numpy as np


class Node:
    def __init__(self):
        self.child = {}
        self.parent = None
        self.group = None
        self.feature = None


class DecisionTree:
    def __init__(self, min_gain=0.1):
        self.root = None
        self.min_gain = min_gain

    def construct_decision_tree(self, X, y):
        X = np.ndarray(X)
        y = np.ndarray(y)
        self.root = Node()
        features = list(range(X.shape[1]))
        self._feature_selection(X, y, features, None)

    def _feature_selection(self, X, y, features, parent):
        node = Node()
        H = self.entropy(y)
        N = len(y)
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
            info_gain = H
            cond_list = {}
            for k in range(N):
                if X[k][i] not in cond_list:
                    cond_list[X[k][i]] = []
                cond_list[X[k][i]].append(y[k])
            for value in cond_list.values():
                h = self.entropy(value)
                info_gain -= len(value) / N * h
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_index = i

        if max_info_gain < self.min_gain:
            node.group = y_max_group
            return node
        else:
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

    @staticmethod
    def entropy(x):
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

