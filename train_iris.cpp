#include "nn_autograd/models.hpp"
#include "nn_autograd/preprocess.hpp"

int main() {
    // Load and split the Iris dataset
    const auto [X_raw, y_raw] = Xy_from_csv("data/iris.csv", -1, false);
    auto [X_train, X_test, y_train, y_test] = train_test_split(X_raw, y_raw, 0.2);

    // Standardize numerical features
    StandardScaler scaler;
    auto X_train_scaled = scaler.fit_transform_to_tensors(X_train);
    auto X_test_scaled = scaler.transform_to_tensors(X_test);

    // One-hot encode target labels
    const int num_classes = 3;
    auto y_train_onehot = to_onehot_tensors(anys_to_ints(y_train), num_classes);
    auto y_test_onehot = to_onehot_tensors(anys_to_ints(y_test), num_classes);

    // Build a simple neural network model
    using OutputType = vector<TensorPtr>;
    Sequential<OutputType> model(
        {Dense(X_train_scaled[0].size(), 8, "relu", Initializers::he_uniform, "Dense0"),
         Dense(8, 4, "relu", Initializers::he_uniform, "Dense1"),
         Dense(4, num_classes, "softmax", Initializers::he_uniform, "Dense2")},
        Loss::categorical_crossentropy, {{"accuracy", Metrics<OutputType>::accuracy}});
    model.summary();

    // Train the model
    int epochs = 30, batch_size = 40; // Should be divisible by X_train size
    LearningRateScheduler *lr_scheduler = new WarmUpAndDecayScheduler(0.1, 5, 10, 0.9);
    model.train(X_train_scaled, y_train_onehot, epochs, lr_scheduler, batch_size);

    // Evaluate on training data
    vector<OutputType> y_pred_train = model.predict(X_train_scaled);
    double accuracy_train = Metrics<OutputType>::accuracy(y_train_onehot, y_pred_train);
    cout << "Training accuracy: " << fixed << setprecision(4) << accuracy_train << endl;

    // Evaluate on test data
    vector<OutputType> y_pred_test = model.predict(X_test_scaled);
    double accuracy_test = Metrics<OutputType>::accuracy(y_test_onehot, y_pred_test);
    cout << "Test accuracy: " << fixed << setprecision(4) << accuracy_test << endl;

    delete lr_scheduler;
    return 0;
}