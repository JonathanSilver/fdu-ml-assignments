// filename: knn.cpp
// compile with: cl /EHsc knn.cpp /std:c++17 /O2
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <map>
#include <vector>
#include <valarray>
#include <algorithm>
#include <thread>
using namespace std;
int read_int(ifstream & fin) {
    int x;
    char * p = reinterpret_cast<char *>(&x);
    fin.read(p + 3, sizeof(char));
    fin.read(p + 2, sizeof(char));
    fin.read(p + 1, sizeof(char));
    fin.read(p, sizeof(char));
    return x;
}
auto decode_idx3_ubyte(const char * filename) {
    ifstream fin(filename, ios_base::binary);
    int magic = read_int(fin), n = read_int(fin);
    int n_row = read_int(fin), n_col = read_int(fin);
    int size = n_row * n_col;
    // elements in valarray is int because they involve being squared later
    vector<valarray<int> > images(n, valarray<int>(size));
    cout << "number of images: " << n << ", magic number: " << magic << endl;
    cout << "number of rows: " << n_row << ", number of columns: " << n_col << endl;
    static char buffer[1 << 10];
    for (int i = 0; i < n; i++) {
        fin.read(buffer, sizeof(char) * size);
        for (int j = 0; j < size; j++)
            images[i][j] = (unsigned char)buffer[j]; // casting is necessary
    }
    return images;
}
auto decode_idx1_ubyte(const char * filename) {
    ifstream fin(filename, ios_base::binary);
    int magic = read_int(fin), n = read_int(fin);
    valarray<char> labels(n);
    cout << "number of labels: " << n << ", magic number: " << magic << endl;
    for (int i = 0; i < n; i++)
        fin.read(&labels[i], sizeof(labels[i]));
    return labels;
}
class kNN_classifier {
    const vector<valarray<int> > x;
    const valarray<char> y;
    char predict_one(const valarray<int> & vec, int k) {
        vector<double> dist(x.size());
        for (int i = 0; i < x.size(); i++) {
            auto t = vec - x[i];
            t *= t;
            dist[i] = sqrt(t.sum());
        }
        vector<int> idx(x.size());
        for (int i = 0; i < idx.size(); i++) idx[i] = i;
        sort(idx.begin(), idx.end(), [&] (auto i, auto j) { return dist[i] < dist[j]; });
        map<char, int> counter;
        for (int i = 0; i < k; i++) counter[y[idx[i]]]++;
        char key = counter.begin()->first;
        int value = counter.begin()->second;
        for (auto & p : counter) {
            if (p.second > value) {
                key = p.first;
                value = p.second;
            }
        }
        return key;
    }
public:
    kNN_classifier(vector<valarray<int> > && x, valarray<char> && y)
        : x{move(x)}, y{move(y)} {}
    auto predict(const vector<valarray<int> > & x, int k, int n_threads = 4) {
        int total = x.size();
        valarray<char> result(total);
        int group_size = total / n_threads;
        vector<thread> threads;
        for (int i = 0; i < n_threads; i++) {
            threads.emplace_back([&] (int start, int end) {
                // synchronization is not needed,
                // as no position will be visited by two distinct threads concurrently
                for (int i = start; i < end; i++) {
                    result[i] = predict_one(x[i], k);
                    if (i != start && i % 500 == 0) cout << i << endl;
                }
            }, i * group_size, i == n_threads - 1 ? total : i * group_size + group_size);
        }
        for (int i = 0; i < n_threads; i++) threads[i].join();
        return result;
    }
};
int main() {
    auto train_images = decode_idx3_ubyte("./mnist/train-images.idx3-ubyte");
    auto train_labels = decode_idx1_ubyte("./mnist/train-labels.idx1-ubyte");
    auto test_images = decode_idx3_ubyte("./mnist/t10k-images.idx3-ubyte");
    auto test_labels = decode_idx1_ubyte("./mnist/t10k-labels.idx1-ubyte");
    
    // for (int i = test_images.size() - 5; i < test_images.size(); i++) {
    //     for (int j = 0; j < 28; j++) {
    //         for (int k = 0; k < 28; k++)
    //             printf("%4d", test_images[i][j * 28 + k]);
    //         puts("");
    //     }
    //     cout << "labels: " << (int)test_labels[i] << endl << endl;
    // }
    
    kNN_classifier classifier{move(train_images), move(train_labels)};
    for (int k = 1; k <= 101; k += 10) {
        auto result = classifier.predict(test_images, k);
        int correct = 0;
        for (int i = 0; i < test_labels.size(); i++) {
            if (result[i] == test_labels[i])
                correct++;
        }
        cout << "k = " << k << ", accuracy: " << correct << " / " << test_labels.size() << " = "
            << 1. * correct / test_labels.size() << endl;
    }
}
