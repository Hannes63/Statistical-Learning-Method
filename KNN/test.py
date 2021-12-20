import numpy as np
from KNN import *


def tst_kd_tree():
    def preorder_traverse(node):
        if node is None:
            print()
            return
        print(node.X, node.y)
        print('left:', end='')
        preorder_traverse(node.left)
        print('right:', end='')
        preorder_traverse(node.right)
        print('back')

    X = np.array([[1, 4], [7, 2], [8, 1], [2, 3], [4, 7], [9, 6]])
    y = ['a', 'a', 'b', 'b', 'a', 'b']
    root = KDTree.construct_kd_tree(X, y)
    preorder_traverse(root)


def tst_knn():
    X = np.array([[12, 7], [22, 12], [81, 41], [25, 36], [49, 77], [93, 60], [25, 56], [67, 12], [70, 80], [50, 32],
                  [34, 90], [23, 55], [56, 3], [34, 76], [45, 44], [60, 60], [4, 12], [3, 43], [25, 87], [83, 25],
                  [90, 12], [43, 88], [89, 91], [33, 25], [39, 97], [71, 56], [25, 76], [50, 50], [6, 79], [58, 2]])
    y = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0,
                  1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  1, 1, 1, 0, 1, 1, 1, 0, 1, 0])
    from matplotlib import pyplot as plt
    plt.title('KNN Example')
    plt.xlabel('x1')
    plt.ylabel('x2')
    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], 'or' if y[i] == 0 else 'xb')

    neigh = KNN(k_neighbors=5)
    neigh.fit(X, y)
    X0 = [50, 52]

    plt.plot(X0[0], X0[1], '^g')
    y0 = neigh.predict(X0)
    plt.show()

    print(y0)


def tst_knn2():
    X = np.array([[0, 97, 15], [12, 12, 87], [81, 40, 77], [25, 36, 12], [4, 77, 14],
                  [93, 60, 88], [25, 59, 32], [67, 12, 25], [70, 80, 14], [55, 32, 12],
                  [34, 90, 67], [23, 55, 78], [66, 3, 33], [34, 76, 98], [45, 44, 76],
                  [60, 60, 77], [4, 12, 33], [3, 43, 67], [25, 87, 83], [83, 25, 38],
                  [90, 12, 27], [43, 88, 32], [89, 91, 77], [33, 25, 87], [39, 97, 38],
                  [71, 56, 25], [25, 76, 88], [37, 73, 36], [6, 79, 44], [58, 2, 23],
                  [65, 78, 33], [79, 87, 14], [81, 30, 68], [31, 12, 24], [77, 32, 90],
                  [32, 40, 9], [9, 40, 23], [88, 70, 2], [67, 89, 72], [41, 13, 88]])
    y = np.array([2, 1, 5, 0, 2, 7, 2, 4, 6, 4,
                  3, 3, 4, 3, 1, 7, 0, 1, 3, 4,
                  4, 2, 1, 7, 2, 4, 3, 2, 2, 4,
                  6, 6, 5, 0, 5, 0, 0, 6, 7, 1])
    neigh = KNN(k_neighbors=3)
    neigh.fit(X, y)
    X0 = [93, 93, 3]

    y0 = neigh.predict(X0)

    print(y0)


# tst_kd_tree()
tst_knn()
