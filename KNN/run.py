import KNN
import numpy as np
import time
import random


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

        print('end read file, items number:', len(labels))
        return features, labels


if __name__ == '__main__':
    X, y = load_data('../data_set/mnist_train.csv')
    start_time = time.time()
    model = KNN.KNN(k_neighbors=25)
    print('fit data...')
    model.fit(X, y)
    end_time = time.time()
    print('Total training time: {}s.'.format(end_time - start_time))

    Xt, yt = load_data('../data_set/mnist_test.csv')
    start_time = time.time()
    y0 = []
    print('start prediction')
    for i in range(len(Xt)):
        if i % 1000 == 0:
            print('finish {} predictions'.format(i))
        y0.append(model.predict(Xt[i]))
    precision = np.sum([int(yt[i] == y0[i]) for i in range(len(yt))]) / len(yt)
    print('precision: {}%'.format(precision * 100))
    end_time = time.time()
    print('Total testing time: {}'.format(end_time - start_time))
    print('median calculating time: {}s, array splitting time: {}s'.format(KNN.KDTree.c1, KNN.KDTree.c2))

