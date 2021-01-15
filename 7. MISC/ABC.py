"""
This is a self-implemented Multi-class AdaBoost Classifier (SAMME).

Decision Tree is used as the weak classifier.

Algorithm (SAMME):
1) For each sample, x[i] and y[i], i = 1, ..., n.
   Initialize its weight w[i] = 1 / n.
2) For m = 1 to M:
   (a) Fit a weak classifier T[m] using weights w.
   (b) Let err[m] = sum_i{ w[i] * [[ T[m](x[i]) != y[i] ]] }
   (c) Let a[m] = log((1 - err[m]) / err[m]) + log(K - 1)
   (d) Update w[i] = w[i] * exp(a[m] * [[ T[m](x[i]) != y[i] ]])
   (e) Normalize w[i] = w[i] / sum_i{ w[i] }
3) The prediction for x is:
   argmax_k{ sum_m{ a[m] * [[ T[m](x) == k ]] } }

Note:
    K is the number of classes.
    [[ x ]] = 1, if x == True,
              0, otherwise.
    sum_i{ w[i] } = w[1] + ... + w[n].
    argmax_k{ a[k] } returns the k which maximizes a[k].

For theoretical justification, please refer to the paper:
Ji Zhu, Saharon Rosset, Hui Zou, et al. Multi-class AdaBoost.
"""

import numpy as np
from sklearn.metrics import classification_report
import math

from DTC import DecisionTreeClassifier
from dataset import SIZE, N_CLASSES, get_dataset


class AdaBoostClassifier:

    def __init__(self, n_features, n_classes, n_estimators, max_depth):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def __call__(self, x):
        s = np.zeros((len(x), self.n_classes))
        for T, a in zip(self.T, self.a):
            s += a * np.eye(self.n_classes)[T(x)]
        return np.argmax(s, axis=-1)
    
    def train(self, train_x, train_y):
        n = len(train_x)
        assert n > 0
        labels = np.argmax(train_y, axis=-1)
        weights = np.ones(n) / n
        self.T = []
        self.a = []
        for i in range(self.n_estimators):
            print('fitting estimator:', i + 1, '/', self.n_estimators)
            T = DecisionTreeClassifier(self.n_features, self.n_classes,
                                       self.max_depth) \
                                           .train(train_x, train_y, weights)
            incorrect_mask = (T(train_x) != labels)
            err = np.sum(weights[incorrect_mask])
            a = math.log((1 - err) / err) + math.log(self.n_classes - 1)
            weights[incorrect_mask] *= math.exp(a)
            weights /= np.sum(weights)
            self.T.append(T)
            self.a.append(a)
            print('weighted accuracy:', 1 - err)
            print('estimator weight:', a)
        return self


if __name__ == '__main__':
    np.random.seed(1)
    train_images, train_labels, test_images, test_labels = get_dataset()
    # For simplicity, the pixels are either 0 or 1.
    train_images[train_images <= 0.05] = 0
    train_images[train_images != 0] = 1
    test_images[test_images <= 0.05] = 0
    test_images[test_images != 0] = 1

    y_true = np.argmax(test_labels, axis=-1)
    y_pred = AdaBoostClassifier(SIZE, N_CLASSES, 10, 6) \
        .train(train_images, train_labels)(test_images)
    print(classification_report(y_true, y_pred, digits=4))

    # using the above settings, in my computer, it
    # achieved an accuracy of 87.73% on the test set.
