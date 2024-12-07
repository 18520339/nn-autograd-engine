#include "nn_autograd/layers.hpp"
#include "nn_autograd/losses.hpp"
#include "nn_autograd/models.hpp"

int main() {
    vector<vector<TensorPtr>> X_train; // Features: [2x³, 3x², -3x]
    vector<TensorPtr> y_train;         // Target: 2x³ + 3x² - 3x

    for (double x = -2.0; x <= 1.0; x += 0.1) {
        X_train.push_back({make_shared<Tensor>(2 * pow(x, 3)),
                           make_shared<Tensor>(3 * pow(x, 2)),
                           make_shared<Tensor>(-3 * x)});
        y_train.push_back(make_shared<Tensor>(2 * pow(x, 3) + 3 * pow(x, 2) - 3 * x));
    }

    Sequential<TensorPtr> model( // Initialize model
        {Dense(3, 4, "relu", Initializers::he_uniform, "Dense0"),
         Dense(4, 3, "relu", Initializers::he_uniform, "Dense1"),
         Dense(3, 1, "linear", Initializers::he_uniform, "Dense2")},
        Loss<TensorPtr>::mean_squared_error);
    model.summary();

    double learning_rate = 0.05;
    int epochs = 100, batch_size = X_train.size();
    model.train(X_train, y_train, epochs, learning_rate, batch_size);
    return 0;
}