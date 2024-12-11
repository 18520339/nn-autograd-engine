#pragma once
#include "tensor.hpp"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <variant>

template <typename OutputType>
class Metrics {
public: // TODO: Inject calculation into training loop
    static double accuracy(const vector<OutputType> &y_trues, const vector<OutputType> &y_preds) {
        if constexpr (is_same_v<OutputType, vector<TensorPtr>>) { // Check for one-hot encoded labels
            int correct = 0;
            for (size_t i = 0; i < y_trues.size(); ++i) {
                auto true_max = max_element(y_trues[i].begin(), y_trues[i].end()); // Find index of max value in true labels
                auto pred_max = max_element(                                       // Find index of max value in predictions
                    y_preds[i].begin(), y_preds[i].end(),
                    [](const TensorPtr &a, const TensorPtr &b) { return a->data < b->data; });
                size_t true_idx = distance(y_trues[i].begin(), true_max);
                size_t pred_idx = distance(y_preds[i].begin(), pred_max);
                if (true_idx == pred_idx) correct++;
            }
            return static_cast<double>(correct) / y_trues.size();
        }
        auto binary_op2 = [](const variant<TensorPtr, vector<TensorPtr>> &y_true,
                             const variant<TensorPtr, vector<TensorPtr>> &y_pred) {
            return get<0>(y_true) == get<0>(y_pred) ? 1.0 : 0.0;
        };
        return inner_product(y_trues.begin(), y_trues.end(), y_preds.begin(), 0.0, plus<>(), binary_op2) / y_trues.size();
    }
};
