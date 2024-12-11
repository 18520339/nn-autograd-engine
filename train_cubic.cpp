#include "nn_autograd/models.hpp"

int main() {
    vector<vector<TensorPtr>> X_train; // Features: [2x³, 3x², -3x]
    vector<TensorPtr> y_train;         // Target: 2x³ + 3x² - 3x

    for (double x = -2.0; x <= 1.0; x += 0.1) {
        X_train.push_back({Tensor::create(2 * pow(x, 3)), // 2x³
                           Tensor::create(3 * pow(x, 2)), // 3x²
                           Tensor::create(-3 * x)});      // -3x
        y_train.push_back(Tensor::create(2 * pow(x, 3) + 3 * pow(x, 2) - 3 * x));
    }

    Sequential<TensorPtr> model( // Initialize model
        {Dense(3, 4, "relu", Initializers::he_uniform, "Dense0"),
         Dense(4, 3, "relu", Initializers::he_uniform, "Dense1"),
         Dense(3, 1, "linear", Initializers::he_uniform, "Dense2")},
        Loss::mean_squared_error);
    model.summary();

    int epochs = 100, batch_size = X_train.size();
    LearningRateScheduler *lr_scheduler = new WarmUpAndDecayScheduler(0.05, 5, 10, 0.9);
    model.train(X_train, y_train, epochs, lr_scheduler, batch_size);

    delete lr_scheduler;
    return 0;
}