#pragma once
#include "tensor.hpp"
#include <any>
#include <random>

pair<vector<vector<any>>, vector<any>> shuffle_data(const vector<vector<any>> &X, const vector<any> &y) {
    vector<size_t> indices(X.size());
    iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ..., X.size() - 1
    shuffle(indices.begin(), indices.end(), mt19937{random_device{}()});

    vector<vector<any>> X_shuffled;
    vector<any> y_shuffled;
    for (int idx : indices) {
        X_shuffled.push_back(X[idx]);
        y_shuffled.push_back(y[idx]);
    }
    return {X_shuffled, y_shuffled};
}

tuple<vector<vector<any>>, vector<vector<any>>, vector<any>, vector<any>>
train_test_split(const vector<vector<any>> &X, const vector<any> &y, float test_size = 0.2) {
    pair<vector<vector<any>>, vector<any>> shuffled_data = shuffle_data(X, y);
    vector<vector<any>> X_shuffled = shuffled_data.first;
    vector<any> y_shuffled = shuffled_data.second;

    size_t train_size = X.size() * (1 - test_size);
    vector<vector<any>> X_train(X_shuffled.begin(), X_shuffled.begin() + train_size);
    vector<vector<any>> X_test(X_shuffled.begin() + train_size, X_shuffled.end());

    vector<any> y_train(y_shuffled.begin(), y_shuffled.begin() + train_size);
    vector<any> y_test(y_shuffled.begin() + train_size, y_shuffled.end());
    return {X_train, X_test, y_train, y_test};
}

class TensorConverter {
public:
    static vector<TensorPtr> to_1d_tensors(const vector<double> &data) {
        size_t n_samples = data.size();
        vector<TensorPtr> tensors_1d(n_samples);
        for (size_t i = 0; i < n_samples; ++i)
            tensors_1d[i] = make_shared<Tensor>(data[i]);
        return tensors_1d;
    }

    static vector<vector<TensorPtr>> to_2d_tensors(const vector<vector<double>> &data) {
        size_t n_samples = data.size();
        int n_features = data[0].size();
        vector<vector<TensorPtr>> tensors_2d(n_samples, vector<TensorPtr>(n_features));
        for (size_t i = 0; i < n_samples; ++i)
            for (int j = 0; j < n_features; ++j)
                tensors_2d[i][j] = make_shared<Tensor>(data[i][j]);
        return tensors_2d;
    }

    static vector<vector<TensorPtr>> to_onehot_tensors(const vector<int> &y_raw, int num_classes) {
        size_t n_samples = y_raw.size();
        vector<vector<TensorPtr>> encoded_y(n_samples, vector<TensorPtr>(num_classes, make_shared<Tensor>(0.0)));
        for (int i = 0; i < n_samples; ++i)
            encoded_y[i][y_raw[i]] = make_shared<Tensor>(1.0);
        return encoded_y;
    }
};

class StandardScaler {
private:
    vector<double> means, stds;
    const double any_to_double(const any &value) {
        if (value.type() == typeid(double)) return any_cast<double>(value);
        else if (value.type() == typeid(int)) return any_cast<int>(value);
        throw runtime_error("Unsupported data type in StandardScaler");
    }

public:
    vector<vector<TensorPtr>> fit_transform_to_tensors(const vector<vector<any>> &X) {
        vector<vector<double>> X_scaled = fit_transform(X);
        return TensorConverter::to_2d_tensors(X_scaled);
    }

    vector<vector<TensorPtr>> transform_to_tensors(const vector<vector<any>> &X) {
        vector<vector<double>> X_scaled = transform(X);
        return TensorConverter::to_2d_tensors(X_scaled);
    }

    vector<vector<double>> fit_transform(const vector<vector<any>> &X) {
        int n_samples = X.size(), n_features = X[0].size();
        means.resize(n_features, 0.0);
        stds.resize(n_features, 0.0);

        // Calculate mean and std for each feature
        for (int f = 0; f < n_features; ++f) {
            for (const vector<any> &sample : X)
                means[f] += any_to_double(sample[f]);
            means[f] /= n_samples;

            for (const vector<any> &sample : X)
                stds[f] += pow(any_to_double(sample[f]) - means[f], 2);
            stds[f] = sqrt(stds[f] / n_samples);
            if (stds[f] == 0) stds[f] = 1.0; // Prevent division by 0
        }
        return transform(X);
    }

    vector<vector<double>> transform(const vector<vector<any>> &X) {
        int n_samples = X.size(), n_features = X[0].size();
        vector<vector<double>> X_scaled(n_samples, vector<double>(n_features));

        for (int i = 0; i < n_samples; ++i)
            for (int f = 0; f < n_features; ++f)
                X_scaled[i][f] = (any_to_double(X[i][f]) - means[f]) / stds[f];
        return X_scaled;
    }
};