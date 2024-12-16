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

    // One-hot encode target labels
    const int num_classes = 10;
    auto y_train_1hots = anys_to_1hots(y_train, num_classes);
    auto y_test_1hots = anys_to_1hots(y_test, num_classes);

    // Build a simple neural network model
    Sequential<vector<TensorPtr>> model(
        {Dense(X_train[0].size(), 10, "relu", Initializers::he_uniform, "Dense0"),
         Dense(10, num_classes, "softmax", Initializers::he_uniform, "Dense1")},
        Loss::categorical_crossentropy, {{"accuracy", Metrics::accuracy}});
    model.summary();

    // Train the model
    int epochs = 5, batch_size = 64, steps_per_epoch = ceil(static_cast<double>(X_train.size()) / batch_size);
    LearningRateScheduler *lr_scheduler = new WarmUpAndDecayScheduler(0.1, steps_per_epoch, steps_per_epoch, 0.95);
    model.train(X_train_scaled, y_train_1hots, epochs, lr_scheduler, batch_size);

    // Evaluate on train data
    auto y_pred_train = model.predict(X_train_scaled);
    double accuracy_train = Metrics::accuracy(y_train_1hots, y_pred_train);
    cout << "Accuracy on Train set: " << fixed << setprecision(4) << accuracy_train << endl;

    // Evaluate on test data
    auto y_pred_test = model.predict(X_test_scaled);
    double accuracy_test = Metrics::accuracy(y_test_1hots, y_pred_test);
    cout << "Accuracy on Test set: " << fixed << setprecision(4) << accuracy_test << endl;

    delete lr_scheduler;
    return 0;
}