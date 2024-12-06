#pragma once
#include "tensor.hpp"
template <typename T> // Handle different input types

class Loss {
private:
    static TensorPtr validate_and_initialize_sum(const vector<T> &y_trues, const vector<T> &y_preds, string message = "Number of samples must match and > 0") {
        if (y_trues.size() != y_preds.size() || y_trues.empty()) throw invalid_argument(message);
        return make_shared<Tensor>(0.0); // Initialize sum to 0
    }

public:
    static TensorPtr mean_squared_error(const vector<TensorPtr> &y_trues, const vector<TensorPtr> &y_preds) {
        TensorPtr sum = validate_and_initialize_sum(y_trues, y_preds);
        for (size_t i = 0; i < y_trues.size(); ++i) {
            TensorPtr diff = y_trues[i] - y_preds[i];
            sum = sum + (diff * diff);
        }
        return sum / make_shared<Tensor>(static_cast<double>(y_trues.size()));
    }

    static TensorPtr binary_crossentropy(const vector<TensorPtr> &y_trues, const vector<TensorPtr> &y_preds) {
        TensorPtr sum = validate_and_initialize_sum(y_trues, y_preds);
        for (size_t i = 0; i < y_trues.size(); ++i) {
            TensorPtr term1 = y_trues[i] * log(y_preds[i]);
            TensorPtr term2 = (make_shared<Tensor>(1.0) - y_trues[i]) * log(make_shared<Tensor>(1.0) - y_preds[i]);
            sum = sum + (term1 + term2);
        }
        return -sum / make_shared<Tensor>(static_cast<double>(y_trues.size())); // Return negative average
    }

    static TensorPtr categorical_crossentropy(const vector<vector<TensorPtr>> &y_trues, const vector<vector<TensorPtr>> &y_preds) {
        TensorPtr sum = validate_and_initialize_sum(y_trues, y_preds);
        for (size_t i = 0; i < y_trues.size(); ++i) {
            TensorPtr sample_loss = validate_and_initialize_sum(y_trues[i], y_preds[i], "One-hot vector dimensions must match");
            for (size_t j = 0; j < y_trues[i].size(); ++j) // For each class in the one-hot vector
                if (y_trues[i][j]->data > 0)               // Only accumulate loss where true label is 1 (one-hot encoded)
                    sample_loss = sample_loss + (y_trues[i][j] * log(y_preds[i][j]));
            sum = sum + sample_loss;
        }
        return -sum / make_shared<Tensor>(static_cast<double>(y_trues.size())); // Return negative average
    }
};