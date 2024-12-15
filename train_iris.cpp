#include "nn_autograd/losses.hpp"
#include "nn_autograd/metrics.hpp"
#include "nn_autograd/models.hpp"
#include "nn_autograd/preprocess.hpp"

int main() {
    // Load and split the Iris dataset
    const auto [X_raw, y_raw] = Xy_from_csv("data/iris.csv", -1, false);
    auto [X_train, X_test, y_train, y_test] = train_test_split(X_raw, y_raw, 0.2);

    // Standardize numerical features
    StandardScaler scaler;
    auto X_train_scaled = scaler.fit_transform(X_train);
    auto X_test_scaled = scaler.transform(X_test);

    // Build a simple neural network model
    const int num_classes = 3;
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
    model.train(X_train_scaled, anys_to_1hots(y_train, num_classes), epochs, lr_scheduler, batch_size);

    // Evaluate on test data
    vector<OutputType> y_pred_test = model.predict(X_test_scaled);
    double accuracy_test = Metrics<OutputType>::accuracy(anys_to_1hot_tensors(y_test, num_classes), y_pred_test);
    cout << "Accuracy on Test set: " << fixed << setprecision(4) << accuracy_test << endl;

    delete lr_scheduler;
    return 0;
}