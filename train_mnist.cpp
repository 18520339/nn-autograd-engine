#include "nn_autograd/losses.hpp"
#include "nn_autograd/metrics.hpp"
#include "nn_autograd/models.hpp"
#include "nn_autograd/preprocess.hpp"

int main() {
    // Load the MNIST dataset
    cout << "Loading the MNIST dataset..." << endl;
    auto [X_train, y_train] = Xy_from_csv("data/mnist_train.csv", 0, false);
    auto [X_test, y_test] = Xy_from_csv("data/mnist_test.csv", 0, false);

    // Standardize numerical features
    cout << "Standardizing numerical features..." << endl;
    StandardScaler scaler;
    auto X_train_scaled = scaler.fit_transform(X_train);
    auto X_test_scaled = scaler.transform(X_test);

    // Build a simple neural network model
    const int num_classes = 10;
    using OutputType = vector<TensorPtr>;
    Sequential<OutputType> model(
        {Dense(X_train[0].size(), 10, "relu", Initializers::he_uniform, "Dense0"),
         Dense(10, num_classes, "softmax", Initializers::he_uniform, "Dense1")},
        Loss::categorical_crossentropy, {{"accuracy", Metrics<OutputType>::accuracy}});
    model.summary();

    // Train the model
    int epochs = 5, batch_size = 64, steps_per_epoch = ceil(static_cast<double>(X_train.size()) / batch_size);
    LearningRateScheduler *lr_scheduler = new WarmUpAndDecayScheduler(0.1, steps_per_epoch, steps_per_epoch, 0.95);
    model.train(X_train_scaled, anys_to_1hots(y_train, num_classes), epochs, lr_scheduler, batch_size);

    // Evaluate on test data
    vector<OutputType> y_pred_test = model.predict(X_test_scaled);
    double accuracy_test = Metrics<OutputType>::accuracy(anys_to_1hot_tensors(y_test, num_classes), y_pred_test);
    cout << "Accuracy on Test set: " << fixed << setprecision(4) << accuracy_test << endl;

    delete lr_scheduler;
    return 0;
}