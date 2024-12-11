#pragma once
#include "tensor.hpp"
#include <any>

any str_to_any(const string &str) { // Convert string to int, double, or string
    try {
        size_t pos; // Number of characters processed
        int int_value = stoi(str, &pos);
        if (pos == str.size()) return int_value;
    } catch (invalid_argument) {}
    try {
        size_t pos;
        double double_value = stod(str, &pos);
        if (pos == str.size()) return double_value;
    } catch (invalid_argument) {}
    return str;
}

vector<int> anys_to_ints(const vector<any> &input) {
    vector<int> result;
    result.reserve(input.size());
    for (const any &a : input)
        result.push_back(any_cast<int>(a));
    return result;
}

vector<double> anys_to_doubles(const vector<any> &input) {
    vector<double> result;
    result.reserve(input.size());
    for (const any &a : input)
        result.push_back(any_cast<double>(a));
    return result;
}

vector<TensorPtr> to_1d_tensors(const vector<double> &data) {
    size_t n_samples = data.size();
    vector<TensorPtr> tensors_1d(n_samples);
    for (size_t i = 0; i < n_samples; ++i)
        tensors_1d[i] = make_shared<Tensor>(data[i]);
    return tensors_1d;
}

vector<vector<TensorPtr>> to_2d_tensors(const vector<vector<double>> &data) {
    size_t n_samples = data.size();
    int n_features = data[0].size();
    vector<vector<TensorPtr>> tensors_2d(n_samples, vector<TensorPtr>(n_features));
    for (size_t i = 0; i < n_samples; ++i)
        for (int j = 0; j < n_features; ++j)
            tensors_2d[i][j] = make_shared<Tensor>(data[i][j]);
    return tensors_2d;
}

vector<vector<TensorPtr>> to_onehot_tensors(const vector<int> &y_raw, int num_classes) {
    size_t n_samples = y_raw.size();
    vector<vector<TensorPtr>> encoded_y(n_samples, vector<TensorPtr>(num_classes, make_shared<Tensor>(0.0)));
    for (int i = 0; i < n_samples; ++i)
        encoded_y[i][y_raw[i]] = make_shared<Tensor>(1.0);
    return encoded_y;
}