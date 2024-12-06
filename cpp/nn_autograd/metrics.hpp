// metrics.hpp
#pragma once
#include "tensor.hpp"
#include <algorithm>
#include <iterator>
#include <numeric>
template <typename OutputType>

double accuracy(const OutputType &y_trues, const OutputType &y_preds) {
    if constexpr (is_same_v<OutputType, vector<vector<TensorPtr>>>) { // Check for one-hot encoded labels
        int correct = 0;
        for (size_t i = 0; i < y_trues.size(); ++i) {
            auto true_max = max_element(y_trues[i].begin(), y_trues[i].end()); // Find index of max value in true labels
            auto pred_max = max_element(                                       // Find index of max value in predictions
                y_preds[i].begin(), y_preds[i].end(),
                [](const auto &a, const auto &b) { return a->data < b->data; }
            );
            size_t true_idx = distance(y_trues[i].begin(), true_max);
            size_t pred_idx = distance(y_preds[i].begin(), pred_max);
            if (true_idx == pred_idx) correct++;
        }
        return static_cast<double>(correct) / y_trues.size();
    }
    return inner_product(
       y_trues.begin(), y_trues.end(), y_preds.begin(), 0.0, plus<>(),
       [](const TensorPtr &y_true, const TensorPtr &y_pred) { return y_true->data == y_pred->data ? 1.0 : 0.0; }
    ) / y_trues.size();
}