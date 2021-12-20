
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle


class Perceptron:
    def __init__(self, learning_rate=0.1, plot='last'):
        """
        The perceptron model learns the parameter w of shape (n_features + 1, ), sign(w @ (x 1)) predicates the class
        of input x.
        :param learning_rate: learning rate of the model
        :param plot: plot the learning process of a data set (2 features only)
        """
        self.lr = learning_rate
        self.plot = plot

        self.X = None
        self.y = None
        self.num = 0
        self.dim = 0
        self.w = None
        assert self.plot in ['last', 'every', 'off']

    def fit(self, X, y):
        """
        :param X: array-like of shape (n_samples, n_features)
        :param y: array-like of shape (n_samples, )
        :return:
        """
        X = np.array(X)
        y = np.array(y)
        self.num = X.shape[0]
        self.dim = X.shape[1]
        self.X = np.concatenate((X, np.ones((self.num, 1))), axis=1)
        self.y = y
        self.w = np.zeros(self.dim + 1)
        it = list(range(self.num))
        it_cnt = 0
        it_cnt_max = 1e5
        modified = True
        while modified:
            it_cnt += 1
            if it_cnt > it_cnt_max:
                break
            modified = False
            shuffle(it)
            for i in it:
                if self.y[i] * self.w @ self.X[i] <= 0:
                    self.w = self.w + self.lr * self.y[i] * self.X[i]
                    modified = True
                    if self.plot == 'every':
                        self.plot2d(self.w)
        if self.plot == 'last':
            self.plot2d(self.w)
        print('The algorithm terminates with loss ', self.loss(self.w))

    def predict(self, X0):
        """
        Predict the positive/negative class of the input samples.
        :param X0: array-like of shape (n_samples, n_features)
        :return: the list of predicted tags
        """
        X0 = np.concatenate((X0, np.ones((len(X0), 1))), axis=1)
        y = [1 if i > 0 else -1 for i in X0 @ self.w.T]
        return y

    def loss(self, w):
        """loss function"""
        loss = 0
        for i in range(self.num):
            if self.y[i] * w @ self.X[i] <= 0:
                loss -= self.y[i] * w * self.X[i]
        return loss

    def plot2d(self, w):
        """plot mode can only show data set of 2 features"""
        assert self.X.ndim == 2
        eps = 1e-6
        plt.title('Perceptron demo')
        plt.xlabel('x1')
        plt.ylabel('x2')
        x = np.arange(0, 8)
        y = -w[0] / (w[1] + eps) * x - w[2] / (w[1] + eps)
        for j in range(self.num):
            plt.plot(self.X[j][0], self.X[j][1], 'o' if self.y[j] == 1 else 'x')
        plt.plot(x, y)
        plt.show()
        print('w1 = {0}, w2 = {1}, b = {2}'.format(w[0], w[1], w[2]))
        # import os
        # os.system('pause')

