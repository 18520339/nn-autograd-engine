#include "tensor.hpp"

int main() {
    auto x1 = make_shared<Tensor>(2.0, "x1"), x2 = make_shared<Tensor>(3.0, "x2");  // Inputs
    auto w1 = make_shared<Tensor>(-4.0, "w1"), w2 = make_shared<Tensor>(1.0, "w2"); // Weights
    auto b = make_shared<Tensor>(5.0, "b");                                         // Bias

    auto x1w1 = x1 * w1, x2w2 = x2 * w2;
    auto x1w1_plus_x2w2 = x1w1 + x2w2;
    auto y = x1w1_plus_x2w2 + b;
    auto z = sigmoid(y);
    z->backward();

    for (TensorPtr node : {x1, x2, w1, w2, b})
        cout << *node << endl;
    return 0;
}