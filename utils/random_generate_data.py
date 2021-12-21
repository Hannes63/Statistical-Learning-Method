
import numpy as np


def random_generate_data(file_path, x, y):
    data = np.random.randint(0, 100, (x, y))
    with open(file_path, 'w') as f:
        print('start writing {}...'.format(file_path))
        np.savetxt(file_path, data, delimiter=',', fmt='%d')
        print('finish writing')


random_generate_data('../data_set/rand_train.csv', 160000, 3)
random_generate_data('../data_set/rand_test.csv', 10000, 3)


