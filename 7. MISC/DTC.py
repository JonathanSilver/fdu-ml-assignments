"""
This is a self-implemented Decision Tree Classifier (ID3).

Algorithm (ID3):
If the maximum depth is reached or all the samples
in the node have the same label, then return. Otherwise,
1) For all the splitting points, split the samples
   into two halves and calculate the information gain.
2) Split the samples using the splitting point that
   leads to the maximum information gain.
3) Recurse into each of the two halves.
"""

import numpy as np
from sklearn.metrics import classification_report
import math

from dataset import SIZE, N_CLASSES, get_dataset


class DecisionTreeClassifier:

    def __init__(self, n_features, n_classes, max_depth):
        self.n_features = n_features
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.max_nodes = 2 ** (max_depth + 1) - 1
        self.n_nodes = 0
    
    @staticmethod
    def _predict_single(node, x):
        if 'value' in node:
            return node['value']
        if x[node['k']] <= node['v']:
            return DecisionTreeClassifier._predict_single(node['left'], x)
        else:
            return DecisionTreeClassifier._predict_single(node['right'], x)
    
    def __call__(self, X):
        pred = [DecisionTreeClassifier._predict_single(self.root, x) for x in X]
        return np.array(pred)
    
    @staticmethod
    def _calc_entropy(weights, labels):
        unique_labels = np.unique(labels)
        weight_sum = np.sum(weights)
        entropy = 0
        for y in unique_labels:
            probability = np.sum(weights[labels == y]) / weight_sum
            entropy -= probability * math.log2(probability)
        return entropy
    
    def build_tree(self, depth, node, x, y, v, w, labels):
        self.n_nodes += 1
        print('building:', self.n_nodes, '/', self.max_nodes)
        if depth == self.max_depth \
                or np.sum(labels == labels[0]) == len(labels):
            node['value'] = np.argmax(np.sum(y * w, axis=0))
            return
        best_k = None
        best_i = None
        best_e = math.inf
        for k in range(self.n_features):
            for i in range(len(v[k])):
                left_idx = np.where(x[:, k] <= v[k][i])
                right_idx = np.where(x[:, k] > v[k][i])
                left_w = w[left_idx]
                right_w = w[right_idx]
                entropy = np.sum(left_w) * self._calc_entropy(left_w, labels[left_idx]) \
                    + np.sum(right_w) * self._calc_entropy(right_w, labels[right_idx])
                if entropy < best_e:
                    best_e = entropy
                    best_k = k
                    best_i = i
        assert best_k is not None and best_i is not None
        node['k'] = best_k
        node['v'] = v[best_k][best_i]
        left_idx = np.where(x[:, best_k] <= v[best_k][best_i])
        right_idx = np.where(x[:, best_k] > v[best_k][best_i])
        node['left'] = {}
        node['right'] = {}
        left_v = v[:]
        left_v[best_k] = left_v[best_k][:best_i + 1]
        right_v = v
        right_v[best_k] = right_v[best_k][best_i + 1:]
        self.build_tree(depth + 1, node['left'], x[left_idx], y[left_idx],
                        left_v, w[left_idx], labels[left_idx])
        self.build_tree(depth + 1, node['right'], x[right_idx], y[right_idx],
                        right_v, w[right_idx], labels[right_idx])

    def train(self, train_x, train_y, weights=None):
        n = len(train_x)
        assert n > 0
        labels = np.argmax(train_y, axis=-1).reshape(-1, 1)
        if weights is None:
            weights = np.ones(n) / n
        weights = weights.reshape(-1, 1)
        values = [np.unique(train_x[:, k]) for k in range(self.n_features)]
        self.root = {}
        self.build_tree(0, self.root, train_x, train_y, values, weights, labels)
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
    y_pred = DecisionTreeClassifier(SIZE, N_CLASSES, 10) \
        .train(train_images, train_labels)(test_images)
    print(classification_report(y_true, y_pred, digits=4))

    # using the above settings, in my computer, it
    # achieved an accuracy of 87.28% on the test set.
