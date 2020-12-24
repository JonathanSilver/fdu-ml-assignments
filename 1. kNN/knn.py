import numpy as np
import struct
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from collections import Counter

from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process


# code for binary file IO is adapted from:
# https://blog.csdn.net/qq_35014850/article/details/80914850


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('Magic: %d, Total Images: %d, Size: %d*%d' % (magic_number, num_images, num_rows, num_cols))
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('Magic: %d, Total Images: %d' % (magic_number, num_images))
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i], = struct.unpack_from(fmt_image, bin_data, offset)
        offset += struct.calcsize(fmt_image)
    return labels


######################## END OF ADAPTATION ########################


class KNNClassifier:

    def __init__(self):
        self.x = None
        self.y = None
        self.n_features = -1
        self.n_classes = -1

    def train(self, x, y):
        self.x = x
        self.y = y
        self.n_features = x.shape[1]
        self.n_classes = np.max(y) + 1
        return self

    def predict_one(self, x, num_neighbors):
        result_list = np.zeros(self.y.shape)
        for i, x0 in enumerate(self.x):
            result_list[i] = np.sqrt(np.sum((x - x0) ** 2))
        idx = np.argsort(result_list)
        result_list = [self.y[i] for i in idx[:num_neighbors]]
        counter = Counter(result_list)
        result_i = -1
        for k, v in counter.items():
            if result_i == -1 or v > counter[result_i]:
                result_i = k
        return result_i


# Sharing numpy objects between processes uses SharedMemory.
# Reference:
# https://docs.python.org/3/library/multiprocessing.shared_memory.html


def create_buffer_from_array(a: np.ndarray):
    buffer = SharedMemory(create=True, size=a.nbytes)
    b = np.ndarray(shape=a.shape, dtype=a.dtype, buffer=buffer.buf)
    b[:] = a[:]
    return buffer, buffer.name, a.shape, a.dtype


def create_array_from_buffer_data(buffer_name, shape, dtype):
    buffer = SharedMemory(name=buffer_name)
    return buffer, np.ndarray(shape=shape, dtype=dtype, buffer=buffer.buf)


def release_memory(buffer_dat):
    buffer_dat[0].close()
    buffer_dat[0].unlink()


def work(train_images_dat,
         train_labels_dat,
         test_images_dat,
         result_dat,
         start, end,
         num_neighbors):
    # get handles from shared memory
    train_images_buffer, train_images_ = create_array_from_buffer_data(*train_images_dat)
    train_labels_buffer, train_labels_ = create_array_from_buffer_data(*train_labels_dat)
    test_images_buffer, test_images_ = create_array_from_buffer_data(*test_images_dat)
    result_buffer_, result_ = create_array_from_buffer_data(*result_dat)
    # make predictions
    classifier = KNNClassifier().train(x=train_images_,
                                       y=train_labels_)
    for i in range(start, end):
        result_[i] = classifier.predict_one(test_images_[i], num_neighbors)
        if i != start and i % 100 == 0:
            print(start, end, i)
    # close handles
    train_images_buffer.close()
    train_labels_buffer.close()
    test_images_buffer.close()
    result_buffer_.close()


def prepare_dataset(images_file, labels_file, size, num=None):
    images = decode_idx3_ubyte(images_file)
    images = images.reshape((-1, size))
    labels = decode_idx1_ubyte(labels_file)
    labels = np.array(labels, dtype=np.int)
    # shuffle
    idx = np.arange(labels.shape[0])
    np.random.shuffle(idx)
    if num is not None:
        idx = idx[:num]
    images = images[idx]
    labels = labels[idx]
    return images, labels


if __name__ == '__main__':
    # constants
    train_images_idx3_ubyte_file = './mnist/train-images.idx3-ubyte'
    train_labels_idx1_ubyte_file = './mnist/train-labels.idx1-ubyte'
    test_images_idx3_ubyte_file = './mnist/t10k-images.idx3-ubyte'
    test_labels_idx1_ubyte_file = './mnist/t10k-labels.idx1-ubyte'
    SIZE = 28 * 28
    N_COMPONENTS = 100
    N_PROCESSES = 4
    # datasets
    train_images, train_labels = prepare_dataset(train_images_idx3_ubyte_file,
                                                 train_labels_idx1_ubyte_file, SIZE)
    test_images, test_labels = prepare_dataset(test_images_idx3_ubyte_file,
                                               test_labels_idx1_ubyte_file, SIZE)
    # pre-processes images
    if N_COMPONENTS > 0:
        pca = PCA(n_components=N_COMPONENTS)
        train_images = pca.fit_transform(train_images)
        test_images = pca.transform(test_images)
        print('PCA: done.')
    # shared memory for storing numpy objects
    train_images_data = create_buffer_from_array(train_images)
    train_labels_data = create_buffer_from_array(train_labels)
    test_images_data = create_buffer_from_array(test_images)
    # place to store the predictions
    result_data = create_buffer_from_array(np.zeros(test_labels.shape))
    result_buffer, result = create_array_from_buffer_data(*result_data[1:])
    # calculate the number of images to predict for each process
    n, = test_labels.shape
    group = n // N_PROCESSES
    # release unused memory
    del train_images
    del train_labels
    del test_images
    # multiprocessing
    for n_neighbors in range(1, 13, 2):
        processes = []
        for k in range(0, n, group):
            p = Process(target=work, args=(train_images_data[1:],
                                           train_labels_data[1:],
                                           test_images_data[1:],
                                           result_data[1:],
                                           k, min(k + group, n),
                                           n_neighbors))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print('k:', n_neighbors)
        print(classification_report(test_labels, result, digits=4))
        print()
    # release shared memory
    del result
    result_buffer.close()
    release_memory(train_images_data)
    release_memory(train_labels_data)
    release_memory(test_images_data)
    release_memory(result_data)
