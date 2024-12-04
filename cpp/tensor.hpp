#pragma once
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
using namespace std;

class Tensor : public enable_shared_from_this<Tensor> {
private:
    function<void(const Tensor *)> local_backward = [](const Tensor *) {}; // Backward lambda function for local gradient computation
    set<TensorPtr> children;                                               // set<> is to prevent duplicates for operations like a + a
    string operation;                                                      // Operation that produced this node

    vector<Tensor *> topological_sort() { // Topological sort to get the order of nodes for backpropagation
        set<Tensor *> visited;
        vector<Tensor *> sorted_nodes; // Builds a list of tensors in topological order
        function<void(Tensor *)> dfs = [&](Tensor *node) {
            if (visited.find(node) == visited.end()) { // If the node has not been visited
                visited.insert(node);
                for (auto &child : node->children)
                    dfs(child.get());
                sorted_nodes.push_back(node);
            }
        };
        dfs(this); // Depth-first search
        return sorted_nodes;
    }

public:
    double data;           // Stores the value of the tensor
    double gradient = 0.0; // Gradient accumulated during backpropagation.
    string label = "";     // Optional Label for the tensor (e.g., "a", "b").

    // Constructor
    Tensor(double _data, const string _label = "") : data(_data) {
        if (_label.empty()) {
            ostringstream ss;
            ss.precision(3);
            ss << fixed << _data;
            label = ss.str();
        } else label = _label;
    }
    Tensor(double _data, const set<TensorPtr> &_children, const string &_operation) : data(_data), children(_children), operation(_operation) {}

    // Utility to print the tensor (similar to Python's __repr__)
    friend ostream &operator<<(ostream &os, const Tensor &tensor) {
        os << "Tensor[data=" << tensor.data << ", gradient=" << tensor.gradient << ", label=" << tensor.label << "]";
        return os;
    }

    // Overloading '+' operator for addition. Using friend to access private members of the class
    friend TensorPtr operator+(const TensorPtr &lhs, const TensorPtr &rhs) {
        auto output = make_shared<Tensor>(lhs->data + rhs->data, set<TensorPtr>{lhs, rhs}, "+");
        output->label = lhs->label + " + " + rhs->label;
        output->local_backward = [lhs, rhs](const Tensor *parent_ptr) {
            // The gradient needs to be accumulated to avoid overwriting when performing operations for the same node
            // For example, if b = a + a, then b.gradient = 2
            // If we don't accumulate, then b.gradient = 1
            lhs->gradient += 1.0 * parent_ptr->gradient; // d(a+b)/da = 1
            rhs->gradient += 1.0 * parent_ptr->gradient; // d(a+b)/db = 1
        };
        return output;
    }

    // Overloading '*' operator for multiplication
    friend TensorPtr operator*(const TensorPtr &lhs, const TensorPtr &rhs) {
        auto output = make_shared<Tensor>(lhs->data * rhs->data, set<TensorPtr>{lhs, rhs}, "*");
        output->label = (lhs->children.size() > 0 ? "(" + lhs->label + ")" : lhs->label) + " * ";
        output->label += (rhs->children.size() > 0 ? "(" + rhs->label + ")" : rhs->label);
        output->local_backward = [lhs, rhs](const Tensor *parent_ptr) {
            lhs->gradient += rhs->data * parent_ptr->gradient; // d(a*b)/da = b
            rhs->gradient += lhs->data * parent_ptr->gradient; // d(a*b)/db = a
        };
        return output;
    }

    // Power operator
    friend TensorPtr pow(const TensorPtr &base, const TensorPtr &exponent) {
        if (base->data == 0.0 && exponent->data == 0.0) throw invalid_argument("0^0 is undefined.");
        if (base->data == 0.0 && exponent->data < 0.0) throw invalid_argument("Division by 0.");

        string operation = (exponent->data > 0 ? "^" + exponent->label + "\n(Division)" : "^" + exponent->label);
        auto output = make_shared<Tensor>(std::pow(base->data, exponent->data), set<TensorPtr>{base, exponent}, operation);
        output->label = (base->children.size() > 0 ? "(" + base->label + ")" : base->label) + "^";
        output->label += (exponent->children.size() > 0 ? "(" + exponent->label + ")" : exponent->label);

        output->local_backward = [base, exponent](const Tensor *parent_ptr) {
            base->gradient += exponent->data * std::pow(base->data, exponent->data - 1) * parent_ptr->gradient;       // d(a^b)/da = b * a^(b-1)
            exponent->gradient += std::log(base->data) * std::pow(base->data, exponent->data) * parent_ptr->gradient; // d(a^b)/db = a^b * ln(a)
        };
        return output;
    }

    // Overloading '/' operator for division and '-' operator for subtraction
    friend TensorPtr operator/(const TensorPtr &numerator, const TensorPtr &denominator) {
        TensorPtr power_minus_one = pow(denominator, make_shared<Tensor>(-1.0));
        if (numerator->data == 1.0) return power_minus_one;         // 1 / b = b^-1
        return numerator * power_minus_one; // a / b = a * b^-1
    }

    friend TensorPtr operator-(const TensorPtr &lhs, const TensorPtr &rhs) {
        return lhs + (-rhs); // a - b = a + (-b)
    }

    // UNARY OPERATORS
    friend TensorPtr operator-(const TensorPtr &tensor) {  // Negation function
        return tensor * make_shared<Tensor>(-1.0); // -a = a * -1
    }

    friend TensorPtr exp(const TensorPtr &tensor) { // Exponential function
        auto output = make_shared<Tensor>(std::exp(tensor->data), set<TensorPtr>{tensor}, "exp");
        output->label = "exp(" + tensor->label + ")";
        output->local_backward = [tensor](const Tensor *parent_ptr) {
            tensor->gradient += parent_ptr->data * parent_ptr->gradient;
        };
        return output;
    }

    friend TensorPtr log(const TensorPtr &tensor) { // Logarithm function
        auto output = make_shared<Tensor>(std::log(tensor->data), set<TensorPtr>{tensor}, "log");
        output->label = "log(" + tensor->label + ")";
        output->local_backward = [tensor](const Tensor *parent_ptr) {
            tensor->gradient += (1.0 / tensor->data) * parent_ptr->gradient;
        };
        return output;
    }

    // ACTIVATION FUNCTIONS
    friend TensorPtr sigmoid(const TensorPtr &tensor) {
        TensorPtr one = make_shared<Tensor>(1.0);
        return one / (one + exp(-tensor));
    }

    friend TensorPtr tanh(const TensorPtr &tensor) {
        TensorPtr one = make_shared<Tensor>(1.0);
        TensorPtr exp2x = exp(tensor * make_shared<Tensor>(2.0));
        return (exp2x - one) / (exp2x + one);
    }

    friend TensorPtr relu(const TensorPtr &tensor) {
        return tensor * make_shared<Tensor>(double(tensor->data > 0), "data > 0");
    }

    // BACKPROPAGATION
    void backward() {
        vector<Tensor *> sorted_nodes = topological_sort();
        gradient = 1.0; // Starting point (usually dL/dL = 1)
        for (long i = sorted_nodes.size() - 1; i >= 0; i--)
            sorted_nodes[i]->local_backward(sorted_nodes[i]); // Compute local gradients
    }
};