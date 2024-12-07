// #pragma once
#include "layers.hpp"
#include "losses.hpp"
#include "metrics.hpp"
#include <any>
#include <iomanip>
#include <unordered_map>
#include <variant>
template <typename OutputType>

class Sequential {
private:
    vector<TensorPtr> parameters;
    vector<Dense> layers; // Currently only support Fully connected layers
    function<TensorPtr(const vector<OutputType> &, const vector<OutputType> &)> loss_func;
    unordered_map<string, function<double(const vector<OutputType> &, const vector<OutputType> &)>> metric_funcs;
    unordered_map<string, vector<any>> history;

public:
    vector<TensorPtr> &get_parameters() { return parameters; }
    vector<Dense> &get_layers() { return layers; }
    unordered_map<string, vector<any>> &get_history() { return history; }

    Sequential(const vector<Dense> &_layers, // Constructor
               function<TensorPtr(const vector<OutputType> &, const vector<OutputType> &)> _loss_func,
               unordered_map<string, function<double(const vector<OutputType> &, const vector<OutputType> &)>> _metric_funcs = {})
        : layers(_layers), loss_func(_loss_func), metric_funcs(_metric_funcs) {
        // Collect all parameters across layers
        for (auto &layer : layers) {
            vector<TensorPtr> layer_params = layer.get_parameters();
            parameters.insert(parameters.end(), layer_params.begin(), layer_params.end());
        }
        // Initialize history with default keys
        history = {{"epoch", {}}, {"step", {}}, {"loss", {}}};
        for (const auto &[name, _] : metric_funcs)
            any_cast<vector<double>>(history[name]) = vector<double>(); // Add metric keys
    }

    variant<TensorPtr, vector<TensorPtr>> forward(vector<TensorPtr> inputs) { // Forward propagation for input features
        for (int i = 0; i < layers.size(); ++i)
            inputs = layers[i].forward(inputs);
        if constexpr (is_same_v<OutputType, TensorPtr>) return inputs[0];
        return inputs;
    }

    void summary() {
        // Calculate dynamic widths based on content
        size_t layer_width = 10, shape_width = 15, params_width = 12; // Minimum width
        for (auto &layer : layers) {                                  // Find maximum widths needed
            layer_width = max(layer_width, layer.get_name().length() + 2);
            shape_width = max(shape_width, to_string(layer.get_output_size()).length() + 2);
            params_width = max(params_width, to_string(layer.get_parameters().size()).length() + 2);
        }
        int total_width = layer_width + shape_width + params_width + 4; // Calculate total width for the table

        // Print header
        cout << "Model Summary:" << endl
             << string(total_width, '=') << endl;
        cout << left << setw(layer_width) << " Layer"
             << right << setw(shape_width) << " Output Shape"
             << right << setw(params_width) << "Param #" << endl;
        cout << string(total_width, '-') << endl;

        // Print layers
        size_t total_params = 0;
        for (auto &layer : layers) {
            size_t num_params = layer.get_parameters().size();
            total_params += num_params;

            cout << left << setw(layer_width) << (" " + layer.get_name())
                 << right << setw(shape_width) << (" " + to_string(layer.get_output_size()))
                 << right << setw(params_width) << num_params << endl;
        }

        // Print footer
        cout << string(total_width, '=') << endl;
        cout << "Total params: " << total_params << endl;
        cout << string(total_width, '=') << endl;
    }

    void train(const vector<vector<TensorPtr>> &X_train, const vector<OutputType> &y_train,
               int epochs = 100, double learning_rate = 0.01, int batch_size = 1, double clip_value = 0.0) {
        int data_size = X_train.size(), steps_per_epoch = data_size / batch_size;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            history["epoch"].push_back(epoch);

            for (auto [batch_start, step] = make_tuple(0, 0); batch_start < data_size; batch_start += batch_size, ++step) {
                // Prevents index out of range when batch_size doesn't divide the data size
                int last_idx = min(batch_start + batch_size, data_size);
                vector<vector<TensorPtr>> X_batch(X_train.begin() + batch_start, X_train.begin() + last_idx);
                vector<OutputType> y_batch(y_train.begin() + batch_start, y_train.begin() + last_idx);

                // Forward pass
                vector<OutputType> predictions;
                for (const vector<TensorPtr> &inputs : X_batch)
                    predictions.push_back(get<(is_same_v<OutputType, TensorPtr>) ? 0 : 1>(forward(inputs)));

                // Backpropagation and update parameters with Gradient Descent
                TensorPtr loss = loss_func(y_batch, predictions);
                loss->backward();

                for (TensorPtr &param : parameters) {
                    // Clip gradients to ensure gradients stay within a manageable range, especially for ReLU
                    // For example: clip(500, -10, 10) = 10
                    if (clip_value > 0.0) param->gradient = max(-clip_value, min(clip_value, param->gradient));
                    param->data -= learning_rate * param->gradient; // Update parameter
                    param->gradient = 0.0;                          // Reset gradient
                }

                // Update history and metrics
                history["step"].push_back(step);
                history["loss"].push_back(loss->data);
                cout << "Epoch " << epoch + 1 << "/" << epochs
                     << " - Step " << step + 1 << "/" << steps_per_epoch
                     << " - Loss: " << setprecision(4) << loss->data;

                for (const auto &[name, func] : metric_funcs) {
                    double metric_value = func(y_batch, predictions);
                    history[name].push_back(metric_value);
                    cout << " - " << name << ": " << setprecision(4) << metric_value;
                }
                cout << endl;
            }
        }
    }

    // vector<vector<TensorPtr>> predict(const vector<vector<TensorPtr>> &X) {
    //     vector<vector<TensorPtr>> predictions;
    //     for (const vector<TensorPtr> &inputs : X)
    //         predictions.push_back(this->forward(inputs));
    //     return predictions;
    // }
};
