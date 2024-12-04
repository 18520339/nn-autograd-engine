#include "tensor.hpp"

int main() {
    auto x1 = Tensor::create(2.0, "x1"), x2 = Tensor::create(3.0, "x2");  // Inputs
    auto w1 = Tensor::create(-4.0, "w1"), w2 = Tensor::create(1.0, "w2"); // Weights
    auto b = Tensor::create(5.0, "b");                                    // Bias

    auto x1w1 = x1 * w1, x2w2 = x2 * w2;
    auto x1w1_plus_x2w2 = x1w1 + x2w2;
    auto y = x1w1_plus_x2w2 + b;
    auto z = sigmoid(y);
    z->backward();

    for (auto node : {x1, x2, w1, w2, b})
        cout << *node << endl;
    return 0;
}