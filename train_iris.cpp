#include "n2n_autograd/models.hpp"
#include "n2n_autograd/preprocess.hpp"

int main() {
    // Load and split the Iris dataset
    const auto [X_raw, y_raw] = Xy_from_csv("data/iris.csv", -1, false);
    auto [X_train, X_test, y_train, y_test] = train_test_split(X_raw, y_raw, 0.2);

    // Standardize numerical features
    StandardScaler scaler;
    auto X_train_scaled = scaler.fit_transform(X_train);
    auto X_test_scaled = scaler.transform(X_test);

    // One-hot encode target labels
    const int num_classes = 3;
    auto y_train_1hots = anys_to_1hots(y_train, num_classes);
    auto y_test_1hots = anys_to_1hots(y_test, num_classes);

    // Build a simple neural network model
    Sequential<vector<TensorPtr>> model(
        {Dense(X_train_scaled[0].size(), 8, "relu", Initializers::he_uniform, "Dense0"),
         Dense(8, 4, "relu", Initializers::he_uniform, "Dense1"),
         Dense(4, num_classes, "softmax", Initializers::he_uniform, "Dense2")},
        Loss::categorical_crossentropy, {{"accuracy", Metrics::accuracy}});
    model.summary();

    // Train the model
    int epochs = 30, batch_size = 40; // Should be divisible by X_train size
    LearningRateScheduler *lr_scheduler = new WarmUpAndDecayScheduler(0.1, 5, 10, 0.9);
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