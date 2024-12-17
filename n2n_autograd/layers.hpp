#pragma once
#include "tensor.hpp"
#include <random>

class Initializers {
public:
    static double random_uniform(double low, double high) {
        static random_device random_seed;
        static mt19937 gen(random_seed()); // Mersenne Twister engine
        return uniform_real_distribution<double>(low, high)(gen);
    }

    static double he_uniform(int fan_in, int fan_out) {
        // https://github.com/keras-team/keras/blob/master/keras/src/initializers/random_initializers.py#L525
        double limit = sqrt(6.0 / fan_in);
        return Initializers::random_uniform(-limit, limit);
    }

    static double glorot_uniform(int fan_in, int fan_out) {
        // https://github.com/keras-team/keras/blob/master/keras/src/initializers/random_initializers.py#326
        double limit = sqrt(6.0 / (fan_in + fan_out));
        return Initializers::random_uniform(-limit, limit);
    }
};

class Neuron {
private:
    function<double()> initializer;
    vector<TensorPtr> weights, parameters;
    TensorPtr bias;
    string activation, name;

public:
    vector<TensorPtr> &get_parameters() { return parameters; }

    Neuron(int input_size, const string &_activation = "", function<double()> init_func = nullptr, const string &_name = "Neuron")
        : activation(_activation), name(_name) {
        // If no initializer is provided:
        // Randomly initialize weights and bias with a uniform distribution between -1 and 1
        // Uniform distribution means that all values have an equal chance of being selected
        initializer = init_func ? init_func : []() { return Initializers::random_uniform(-1.0, 1.0); };
        for (int i = 0; i < input_size; ++i)
            weights.push_back(make_shared<Tensor>(initializer(), name + "_w" + to_string(i)));
        bias = make_shared<Tensor>(initializer(), name + "_b");

        parameters.insert(parameters.end(), weights.begin(), weights.end());
        parameters.push_back(bias);
    }

    TensorPtr forward(const vector<TensorPtr> &inputs) {
        TensorPtr z = bias;
        for (size_t i = 0; i < inputs.size(); ++i)
            z = z + weights[i] * inputs[i];

        if (activation == "softmax" || activation == "linear" || activation.empty()) return z;
        if (activation == "sigmoid") return sigmoid(z);
        if (activation == "tanh") return tanh(z);
        if (activation == "relu") return relu(z);
        throw runtime_error("Unknown activation function: " + activation);
    }
};

class Dense { // Fully connected layer
private:
    function<double()> initializer;
    vector<Neuron> neurons;
    vector<TensorPtr> parameters;
    int input_size, output_size;
    string activation, name;

public:
    vector<TensorPtr> &get_parameters() { return parameters; }
    const int &get_input_size() const { return input_size; }
    const int &get_output_size() const { return output_size; }
    const string &get_name() const { return name; }
    const string &get_activation() const { return activation; }

    Dense(int _input_size, int _output_size, const string &_activation = "", function<double(int, int)> init_func = nullptr, const string &_name = "Dense")
        : input_size(_input_size), output_size(_output_size), activation(_activation), name(_name) {
        if (init_func) initializer = [=]() { return init_func(input_size, output_size); };
        for (int i = 0; i < output_size; ++i) {
            // Construct the element directly, avoiding the overhead of creating
            // a temporary Neuron object and then moving or copying it into the vector
            neurons.emplace_back(input_size, activation, initializer, name + "_N" + to_string(i));
            parameters.insert(parameters.end(), neurons.back().get_parameters().begin(), neurons.back().get_parameters().end());
        }
    }

    vector<TensorPtr> forward(const vector<TensorPtr> &inputs) {
        vector<TensorPtr> outputs;

        if (activation == "softmax") { // Softmax is a special case where the activation is applied across entire layer instead of individual neurons
            vector<TensorPtr> exp_outputs;
            TensorPtr sum = make_shared<Tensor>(0.0);

            for (Neuron &neuron : neurons) {
                exp_outputs.push_back(exp(neuron.forward(inputs)));
                sum = sum + exp_outputs.back();
            }
            for (size_t i = 0; i < exp_outputs.size(); ++i)
                outputs.push_back(exp_outputs[i] / sum);
        } else {
            for (size_t i = 0; i < neurons.size(); ++i)
                outputs.push_back(neurons[i].forward(inputs));
        }
        return outputs;
    }
};