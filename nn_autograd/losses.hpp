#pragma once
#include "tensor.hpp"

class Loss {
public:
    static TensorPtr mean_squared_error(const vector<TensorPtr> &y_trues, const vector<TensorPtr> &y_preds) {
        if (y_trues.size() != y_preds.size() || y_trues.empty()) throw invalid_argument("Number of samples must match and non-zero");
        TensorPtr sum = make_shared<Tensor>(0.0); // Initialize sum to 0

        for (size_t i = 0; i < y_trues.size(); ++i) {
            TensorPtr diff = y_trues[i] - y_preds[i];
            sum = sum + (diff * diff);
        }
        return sum / make_shared<Tensor>(static_cast<double>(y_trues.size()));
    }

    static TensorPtr binary_crossentropy(const vector<TensorPtr> &y_trues, const vector<TensorPtr> &y_preds) {
        if (y_trues.size() != y_preds.size() || y_trues.empty()) throw invalid_argument("Number of samples must match and non-zero");
        TensorPtr sum = make_shared<Tensor>(0.0); // Initialize sum to 0

        for (size_t i = 0; i < y_trues.size(); ++i) {
            TensorPtr term1 = y_trues[i] * log(y_preds[i]);
            TensorPtr term2 = (make_shared<Tensor>(1.0) - y_trues[i]) * log(make_shared<Tensor>(1.0) - y_preds[i]);
            sum = sum + (term1 + term2);
        }
        return -sum / make_shared<Tensor>(static_cast<double>(y_trues.size())); // Return negative average
    }

    static TensorPtr categorical_crossentropy(const vector<vector<TensorPtr>> &y_trues, const vector<vector<TensorPtr>> &y_preds) {
        if (y_trues.size() != y_preds.size() || y_trues.empty()) throw invalid_argument("Number of samples must match and non-zero");
        TensorPtr sum = make_shared<Tensor>(0.0); // Initialize sum to 0

        for (size_t i = 0; i < y_trues.size(); ++i) {
            if (y_trues[i].size() != y_preds[i].size() || y_trues[i].empty()) throw invalid_argument("One-hot vector dimensions must match");
            TensorPtr sample_loss = make_shared<Tensor>(0.0); // Initialize sum to 0

            for (size_t j = 0; j < y_trues[i].size(); ++j) // For each class in the one-hot vector
                if (y_trues[i][j]->data > 0)               // Only accumulate loss where true label is 1 (one-hot encoded)
                    sample_loss = sample_loss + (y_trues[i][j] * log(y_preds[i][j]));
            sum = sum + sample_loss;
        }
        return -sum / make_shared<Tensor>(static_cast<double>(y_trues.size())); // Return negative average
    }
};