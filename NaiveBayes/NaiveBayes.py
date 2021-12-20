
import numpy as np


class NaiveBayes:
    def __init__(self, const=1.0):
        """
        :param const: the bayesian estimation constant, if const=0, it is equal to MLE; if const=1, it is called
        Laplacian smoothing.
        """
        self.cond_prob = []
        # a list of dictionaries, where cond_prob[i][(j, k)] indicates the conditional probability
        # of P(X^(i) = j | Y = k).
        self.priori = {}
        # a dictionary, where priori[k] indicates the probability P(Y = k).
        self.const = const

    def fit(self, X, y):
        """
        Fit the data.
        :param X: Discrete feature vectors, array-like of shape (n_samples, n_features).
        :param y: Classification tags, array-like of shape (n_samples, )
        """
        X = np.array(X)
        n1 = X.shape[0]
        n2 = X.shape[1]
        list_X = [set([X[i][j] for i in range(n1)]) for j in range(n2)]
        set_y = set(y)
        for set_X in list_X:
            dic = {}
            for feature in set_X:
                for tag in set_y:
                    dic[(feature, tag)] = 0
            self.cond_prob.append(dic)
        for tag in set_y:
            self.priori[tag] = 0

        for i in range(n1):
            for j in range(n2):
                self.cond_prob[j][(X[i][j], y[i])] += 1
            self.priori[y[i]] += 1

        for i in range(len(self.cond_prob)):
            S_i = len(self.cond_prob[i]) / len(set_y)
            for tup in self.cond_prob[i].keys():
                self.cond_prob[i][tup] = (self.cond_prob[i][tup] + self.const) / \
                                         (self.priori[tup[1]] + S_i * self.const)
        for key in self.priori:
            self.priori[key] = (self.priori[key] + self.const) / (n1 + len(self.priori) * self.const)

    def predict(self, X0):
        """
        Predict the category of input X0.
        :param X0: array-like of shape (n_features, )
        :return: the most-likely classification
        """
        X0 = np.array(X0)
        max_y = None
        max_prob = 0
        for y in self.priori:
            prob = self.priori[y]
            for j in range(len(X0)):
                prob *= self.cond_prob[j][(X0[j], y)]
            if prob > max_prob:
                max_prob = prob
                max_y = y
        return max_y


