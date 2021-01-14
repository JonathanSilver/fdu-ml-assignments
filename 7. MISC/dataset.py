import numpy as np
import struct


train_images_idx3_ubyte_file = './mnist/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = './mnist/train-labels.idx1-ubyte'

test_images_idx3_ubyte_file = './mnist/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = './mnist/t10k-labels.idx1-ubyte'

SIZE = 28 * 28
N_CLASSES = 10


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
    return np.array(labels, np.int)


def get_dataset():
    train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)

    test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)

    train_images /= 255
    test_images /= 255
    train_images = train_images.reshape(-1, SIZE)
    test_images = test_images.reshape(-1, SIZE)
    
    train_labels = np.eye(N_CLASSES)[train_labels]
    test_labels = np.eye(N_CLASSES)[test_labels]

    return train_images, train_labels, test_images, test_labels
