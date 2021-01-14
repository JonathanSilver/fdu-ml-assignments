"""
This is a self-implemented Feed-forward/Deep Neural Network.

Though many beginners would love to implement their own,
I shall say that knowing how it works is quite different
from building it from scratch.

If you are just a beginner, you should refer to this code,
or whatever you can find on the Internet, because there are
tricks you have never seen, and pitfalls you are not aware.
What's more, tuning the parameters is non-trivial. This is
NO easy task, definitely. You will feel frustrated if your
code doesn't work, or it performs poorly. I felt difficult
when I wrote this one, too. So, I also recommend that you
may skip it as a beginner, come back and re-write it when
you know more about machine/deep learning.

If you are a master already and find anything wrong in my
code, please let me know ^_^.

By the way, do not expect this tutorial-like script to out-
perform Tensorflow/PyTorch. This is obviously not possible.
"""


import numpy as np
import struct
from copy import deepcopy
from sklearn.metrics import classification_report


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


train_images_idx3_ubyte_file = './mnist/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = './mnist/train-labels.idx1-ubyte'

test_images_idx3_ubyte_file = './mnist/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = './mnist/t10k-labels.idx1-ubyte'

train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)

test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)

SIZE = 28 * 28
train_images /= 255
test_images /= 255
train_images = train_images.reshape(-1, SIZE)
test_images = test_images.reshape(-1, SIZE)

N_CLASSES = 10
train_labels = np.eye(N_CLASSES)[train_labels]
test_labels = np.eye(N_CLASSES)[test_labels]


def sigmoid(z):
    """
    Sigmoid Function: 1 / (1 + exp(-z))

    For z >= 0, 0 < exp(-z) <= 1, so you can directly calculate the result.

    For z < 0, exp(-z) > 1, so you may encounter overflow easily.
    Then, you can use the alternative way to do it: exp(z) / (exp(z) + 1).
    """
    r = np.zeros_like(z)
    pos_mask = (z >= 0)
    r[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
    neg_mask = (z < 0)
    e = np.exp(z[neg_mask])
    r[neg_mask] = e / (e + 1)
    return r


def softmax(z):
    """
    Softmax Function: exp(z) / sum(exp(z))

    To avoid overflow, we subtract the maximum value m in z.
    You may easily discover that the result is the same:
    exp(z - m) / sum(exp(z - m)) == exp(z) / sum(exp(z))
    Since z - m <= 0, we have 0 < exp(z - m) <= 1,
    which is how we avoid overflow.
    """
    m = np.max(z, axis=-1, keepdims=True)
    z = np.exp(z - m)
    return z / np.sum(z, axis=-1, keepdims=True)


def SGD(eta):
    """
    Stochastic Gradient Descent (SGD)

    parameter update rule:
    w = w - eta * dw
    """
    def optimizer(grads, vars):
        def step():
            for key in vars:
                if key in grads:
                    vars[key] -= eta * grads[key]
        return step
    return optimizer


def RMSProp(eta, gamma=0.9, eps=1e-6):
    """
    RMSProp

    parameter update rule:
    r = gamma * r + (1 - gamma) * dw ** 2
    w = w - eta * dw / (sqrt(r) + eps)
    with r initialized to 0
    """
    def optimizer(grads, vars):
        states = {}
        for key in vars:
            states[key] = np.zeros_like(vars[key])
        def step():
            for key in vars:
                if key in grads:
                    g = grads[key]
                    r = gamma * states[key] + (1 - gamma) * g ** 2
                    vars[key] -= eta * g / (np.sqrt(r) + eps)
                    states[key] = r
        return step
    return optimizer


class DNNClassifier:

    """
    Feed-forward/Deep Neural Network (FNN/DNN)

    input: x
    output: y
    label: y'

    forward:
    z^[l] = a^[l-1] * w^[l] + b^[l]
    a^[l] = sigmoid(z^[l])
    a^[0] = x
    y = softmax(z^[N])
    l = 1, ..., N

    cross entropy loss:
    L = -MEAN(y' log(y))

    backward:
    dL/d(z^[N]) = (softmax(z^[N]) - y') / N_CLASSES
    dL/d(z^[l]) = dL/d(z^[l+1]) * w^[l+1].T sigmoid_prime(z^[l])
                = dL/d(z^[l+1]) * w^[l+1].T a^[l] (1 - a^[l])
    dL/d(w^[l]) = a[l-1].T * dL/d(z^[l]) / BATCH_SIZE
    dL/d(b^[l]) = MEAN(dL/d(z^[l]), axis=0)
    """

    def __init__(self, shape):
        assert len(shape) > 1
        self.__shape = shape
        self.__weight = {}
        self.__grad = {}
        for i in range(1, len(shape)):
            self.__weight['w' + str(i)] = np.random.randn(shape[i - 1], shape[i])
            self.__weight['b' + str(i)] = np.zeros(shape[i])

    def __forward(self, x):
        cache_a = [x]
        for i in range(1, len(self.__shape)):
            z = np.matmul(cache_a[-1], self.__weight['w' + str(i)]) \
                + self.__weight['b' + str(i)]
            cache_a.append(sigmoid(z) if i != len(self.__shape) - 1 else softmax(z))
        return cache_a
    
    def __call__(self, x):
        return self.__forward(x)[-1]

    def __backward(self, cache_a, y):
        batch_size = cache_a[-1].shape[0]
        dz = (cache_a[-1] - y) / self.__shape[-1]
        for i in reversed(range(1, len(self.__shape))):
            a = cache_a[i - 1]
            self.__grad['w' + str(i)] = np.matmul(a.T, dz) / batch_size
            self.__grad['b' + str(i)] = np.mean(dz, axis=0)
            w = self.__weight['w' + str(i)]
            dz = np.matmul(dz, w.T) * a * (1 - a)

    def train(self, train_x, train_y, val_x, val_y,
              batch_size, n_epochs, patience, optimizer):
        optimize = optimizer(self.__grad, self.__weight)
        n_train, n_val = train_x.shape[0], val_x.shape[0]
        y_true = np.argmax(val_y, axis=-1)
        idx = np.arange(n_train)
        epoch, best_epoch, best_correct = [0] * 3
        best_weight = None
        while epoch < n_epochs:
            epoch += 1
            np.random.shuffle(idx)
            loss, correct = [0] * 2
            for k in range(0, n_train, batch_size):
                batch_idx = idx[k:min(k + batch_size, n_train)]
                cache_a = self.__forward(train_x[batch_idx])
                batch_y = train_y[batch_idx]
                self.__backward(cache_a, batch_y)
                optimize()
                loss += np.sum(-batch_y * np.log(cache_a[-1]))
                correct += np.sum(np.argmax(cache_a[-1], axis=-1) 
                                == np.argmax(batch_y, axis=-1))
            y_pred = np.argmax(self(val_x), axis=-1)
            val_correct = np.sum(y_true == y_pred)
            print('epoch: %d, train loss: %.6f, '
                  'train accuracy: %.6f, val accuracy: %.6f'
                % (epoch, loss, correct / n_train, val_correct / n_val))
            if val_correct > best_correct:
                best_correct = val_correct
                best_epoch = epoch
                best_weight = deepcopy(self.__weight)
            if epoch - best_epoch >= patience:
                print('early stopping: '
                      'restoring weights from epoch', best_epoch)
                self.__weight = best_weight
                break
        return self


if __name__ == '__main__':
    np.random.seed(1)
    y_true = np.argmax(test_labels, axis=-1)
    model = DNNClassifier([SIZE, 100, N_CLASSES]) \
        .train(train_images, train_labels, test_images, test_labels,
               batch_size=100, n_epochs=100, patience=5,
               optimizer=RMSProp(1e-2))
    y_pred = np.argmax(model(test_images), axis=-1)
    print(classification_report(y_true, y_pred, digits=4))

    # using the above settings, in my computer, it
    # achieved an accuracy of 97.24% on the test set.
