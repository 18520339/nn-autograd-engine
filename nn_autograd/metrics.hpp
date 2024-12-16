#pragma once
#include "tensor.hpp"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <variant>

using YTruesVariant = variant<vector<double>, vector<vector<int>>>;
using YPredsVariant = variant<vector<double>, vector<vector<double>>>;

class Metrics {
public:
    static double accuracy(const YTruesVariant &y_trues, const YPredsVariant &y_preds) {
        if (holds_alternative<vector<vector<int>>>(y_trues) && get<1>(y_trues)[0].size() > 0) { // Check for one-hot encoded labels
            int correct = 0, n_samples = get<1>(y_trues).size();
            for (size_t i = 0; i < n_samples; ++i) {
                vector<int> y_true = get<1>(y_trues)[i];
                vector<double> y_pred = get<1>(y_preds)[i];
                auto true_max = max_element(y_true.begin(), y_true.end(), [](const int &a, const int &b) { return a < b; });
                auto pred_max = max_element(y_pred.begin(), y_pred.end(), [](const double &a, const double &b) { return a < b; });

                // Find index of max value in true labels and predictions
                int true_idx = distance(y_true.begin(), true_max), pred_idx = distance(y_pred.begin(), pred_max);
                if (true_idx == pred_idx) correct++;
            }
            return static_cast<double>(correct) / n_samples;
        }
        // When activation is sigmoid and loss is binary_crossentropy, convert probabilities to binary values
        auto binary_op2 = [](const int &y_true, const double &y_pred) {
            int y_pred_binary = y_pred > 0.5 ? 1 : 0;
            return y_true == y_pred_binary ? 1 : 0;
        };
        vector<double> trues = get<0>(y_trues), preds = get<0>(y_preds);
        return inner_product(trues.begin(), trues.end(), preds.begin(), 0.0, plus<>(), binary_op2) / trues.size();
    }
};
