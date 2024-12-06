// layers.hpp
#pragma once
#include "tensor.hpp"
#include <random>
#include <variant>
using ForwardType = variant<TensorPtr, vector<TensorPtr>>;

class Initializers {
public:
    static random_device random_seed;
    static mt19937 gen;

    static double random_uniform(double low, double high) {
        return uniform_real_distribution<double>(low, high)(Initializers::gen);
    }

    static double he_uniform(int fan_in, int fan_out) {
        // https://github.com/keras-team/keras/blob/master/keras/src/initializers/random_initializers.py#L525
        double limit = sqrt(6.0 / fan_in);
        return uniform_real_distribution<double>(-limit, limit)(Initializers::gen);
    }

    static double glorot_uniform(int fan_in, int fan_out) {
        // https://github.com/keras-team/keras/blob/master/keras/src/initializers/random_initializers.py#326
        double limit = sqrt(6.0 / (fan_in + fan_out));
        return uniform_real_distribution<double>(-limit, limit)(Initializers::gen);
    }
};

class ParamsContainer {
protected:
    function<double()> initializer;
    vector<TensorPtr> parameters;
    string activation, name;

public:
    ParamsContainer(const string &_activation = "", const string &_name = "", function<double()> init_func = nullptr)
        : activation(_activation), name(_name), initializer(init_func) {}
    vector<TensorPtr> &get_parameters() { return parameters; }
    const string &get_activation() const { return activation; }
    const string &get_name() const { return name; }
    virtual ForwardType forward(const vector<TensorPtr> &inputs) = 0;
};

class Neuron : public ParamsContainer {
private:
    vector<TensorPtr> weights;
    TensorPtr bias;

public:
    Neuron(int input_size, const string &_activation = "", function<double()> init_func = nullptr, const string &_name = "Neuron")
        : ParamsContainer(_activation, _name, init_func ? init_func : []() { return Initializers::random_uniform(-1.0, 1.0); }) {
        // If no initializer is provided:
        // Randomly initialize weights and bias with a uniform distribution between -1 and 1
        // Uniform distribution means that all values have an equal chance of being selected
        for (int i = 0; i < input_size; ++i)
            weights.push_back(make_shared<Tensor>(initializer(), name + "_w" + to_string(i)));
        bias = make_shared<Tensor>(initializer(), name + "_b");

        parameters = weights;
        parameters.push_back(bias);
    }

    ForwardType forward(const vector<TensorPtr> &inputs) {
        TensorPtr z = bias;
        for (size_t i = 0; i < inputs.size(); ++i)
            z = z + weights[i] * inputs[i];

        if (activation.empty() || activation == "softmax") return z;
        if (activation == "sigmoid") return sigmoid(z);
        if (activation == "tanh") return tanh(z);
        if (activation == "relu") return relu(z);
        throw runtime_error("Unknown activation function: " + activation);
    }
};

class Dense : public ParamsContainer { // Fully connected layer
private:
    vector<Neuron> neurons;
    int input_size, output_size;

public:
    Dense(int input_size, int output_size, const string &_activation = "", function<double(int, int)> init_func = nullptr, const string &_name = "Dense")
        : input_size(input_size), output_size(output_size), ParamsContainer(_activation, _name) {
        if (init_func) initializer = [=]() { return init_func(input_size, output_size); };
        for (int i = 0; i < output_size; ++i)
            // Construct the element directly, avoiding the overhead of creating
            // a temporary Neuron object and then moving or copying it into the vector
            neurons.emplace_back(input_size, activation, initializer, name + "_N" + to_string(i));
    }

    ForwardType forward(const vector<TensorPtr> &inputs) {
        vector<TensorPtr> outputs;

        if (activation == "softmax") { // Softmax is a special case where the activation is applied across entire layer instead of individual neurons
            vector<TensorPtr> exp_outputs;
            TensorPtr sum = make_shared<Tensor>(0.0);

            for (Neuron &neuron : neurons) {
                exp_outputs.push_back(exp(neuron.forward(inputs)));
                sum = sum + exp_outputs.back();
            }
            for (size_t i = 0; i < exp_outputs.size(); ++i) {
                outputs.push_back(exp_outputs[i] / sum);
                parameters.insert(parameters.end(), neurons[i].get_parameters().begin(), neurons[i].get_parameters().end());
            }
        } else {
            for (size_t i = 0; i < neurons.size(); ++i) {
                outputs.push_back(neurons[i].forward(inputs));
                parameters.insert(parameters.end(), neurons[i].get_parameters().begin(), neurons[i].get_parameters().end());
            }
        }
        return outputs.size() == 1 ? outputs[0] : outputs;
    }
};