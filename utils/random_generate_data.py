
import numpy as np
import csv


def random_generate_data(file_path):
    data = np.random.rand(10, 10)
    with open(file_path, 'w') as f:
        print('start writing...')
        writer = csv.writer(f)
        writer.writerows(data)
        print('finish writing')


random_generate_data('data_set/mnist_train.csv')
