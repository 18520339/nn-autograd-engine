#pragma once
#include "converters.hpp"
#include <fstream>
#include <random>
#include <unordered_map>

pair<vector<vector<any>>, vector<any>> Xy_from_csv(const string &file_path, bool skip_header = true) {
    vector<vector<any>> X_raw;
    vector<any> y_raw;
    unordered_map<string, int> class_to_index = {};
    int class_index = 0;

    ifstream file(file_path);
    string line, cell_value;
    if (skip_header) getline(file, line); // Skip header

    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream line_stream(line);
        vector<any> row;

        // Auto-detect the data type of each cell and store it in the row
        while (getline(line_stream, cell_value, ','))
            row.push_back(str_to_any(cell_value));
        X_raw.emplace_back(row.begin(), row.end() - 1);

        if (row.back().type() == typeid(string)) {
            string class_label = any_cast<string>(row.back());
            if (class_to_index.find(class_label) == class_to_index.end()) // Not found
                class_to_index[class_label] = class_index++;
            y_raw.push_back(class_to_index[class_label]);
        } else y_raw.push_back(row.back());
    }
    return {X_raw, y_raw};
}

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
        return to_2d_tensors(X_scaled);
    }

    vector<vector<TensorPtr>> transform_to_tensors(const vector<vector<any>> &X) {
        vector<vector<double>> X_scaled = transform(X);
        return to_2d_tensors(X_scaled);
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