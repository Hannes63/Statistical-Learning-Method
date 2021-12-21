import numpy as np
import random
import time
from KNN import KNN


def tst_knn1():
    X = np.array([[12, 7], [22, 12], [81, 41], [25, 36], [49, 77], [93, 60], [25, 56], [67, 12], [70, 80], [50, 32],
                  [34, 90], [23, 55], [56, 3], [34, 76], [45, 44], [60, 60], [4, 12], [3, 43], [25, 87], [83, 25],
                  [90, 12], [43, 88], [89, 91], [33, 25], [39, 97], [71, 56], [25, 76], [50, 50], [6, 79], [58, 2],
                  [6, 60], [41, 3], [13, 20], [29, 12], [63, 34], [40, 19], [72, 2], [80, 83], [16, 89], [57, 83]])
    y = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0,
                  1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  1, 1, 1, 0, 1, 1, 1, 0, 1, 0,
                  1, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    from matplotlib import pyplot as plt
    plt.title('KNN Example')
    plt.xlabel('x1')
    plt.ylabel('x2')
    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], 'or' if y[i] == 0 else 'xb')

    neigh = KNN(k_neighbors=5)
    neigh.fit(X, y)
    X0 = [38, 60]

    plt.plot(X0[0], X0[1], '^g')
    y0 = neigh.predict(X0)
    plt.show()

    print(y0)


def load_data(file_path, size=0):
    print('start read file', file_path)
    with open(file_path) as f:
        features = []
        labels = []
        lines = f.readlines()[:size] if size else f.readlines()
        random.shuffle(lines)
        for line in lines:
            cur_line = line.strip().split(',')
            features.append([int(num) for num in cur_line[1:]])
            labels.append(int(cur_line[0]))

        print('end read file, items number:', len(labels), 'feature number:', len(features[0]))
        return features, labels


def tst_knn2():
    X, y = load_data('../data_set/rand_train.csv', 160000)
    start_time = time.time()
    model = KNN(k_neighbors=25)
    print('fit data...')
    model.fit(X, y)
    end_time = time.time()
    print('Total training time: {}s.'.format(end_time - start_time))

    Xt, yt = load_data('../data_set/rand_test.csv', 1000)
    start_time = time.time()
    y0 = []
    print('start prediction...')
    for i in range(len(Xt)):
        if i % 200 == 0 and i > 0:
            print('finish {} predictions, used time {}s'.format(i, time.time() - start_time))
        y0.append(model.predict(Xt[i]))
    precision = np.sum([int(yt[i] == y0[i]) for i in range(len(yt))]) / len(yt)
    print('precision: {}%'.format(precision * 100))
    end_time = time.time()
    print('Total testing time: {}'.format(end_time - start_time))


tst_knn1()
# tst_knn2()
