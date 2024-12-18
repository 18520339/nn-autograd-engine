# **C++17** Autograd Neural Network Framework

A flexible and extensible framework in pure **C++17** designed to facilitate the construction, training, and evaluation of Neural Networks. Inspired by modern Deep Learning frameworks like [PyTorch](https://pytorch.org) and [TensorFlow](https://www.tensorflow.org), this project provides:
- A collection of modular components with an **automatic differentiation engine** as essential building blocks for experimenting with custom model architectures.
- A foundational understanding how Neural Network and its **computational graph** can be implemented from scratch, offering insights into the underlying mechanics of forward and backward propagation, gradient computation using chain rule, and its optimization using Gradient Descent.

This project serves both educational purposes for those interested in understanding the internals of Neural Networks and practical applications where a lightweight, efficient, and customizable framework is needed.

## Key Features

- **Pure C++17 Implementation**: No external dependencies, leveraging modern C++ features for efficient. Memory management is handled using smart pointers (`std::shared_ptr`), minimizing the risk of memory leaks.
- **Tensor Operations**: Support for tensor arithmetic with automatic gradient tracking, which performs and manipulates mathematical operations with tensors, like `+`, `*`, or activation functions, etc.
  - **Automatic Differentiation**: Automatically compute gradients efficiently during backpropagation.
  - **Activation Functions**: Include common activation functions like `Sigmoid`, `Tanh`, `ReLU`, and `Softmax`.
- **Sequential Model**: A high-level API similar to [TensorFlow](https://www.tensorflow.org/guide/keras/sequential_model) for building and training Neural Network models using a sequential stack of layers.
  - **Batch Processing**: Support for training models with **Mini-batch Gradient Descent**.
  - **Loss Functions**: Implementations of standard loss functions like **Mean Squared Error** or **Binary/Categorical Cross-Entropy** Loss.
  - **Evaluation Metrics**: Functions to evaluate the performance of models using metrics like `accuracy`.
  - **Learning Rate Scheduler**: Offer schedulers for dynamic learning rate adjustment during training.
  - **Logging**: Tools for model summarizing, monitoring, and logging training progress like [TensorFlow's Model.summary()](https://www.tensorflow.org/api_docs/python/tf/keras/Model#summary)
- **Data Preprocessing**: Utilities for loading, shuffling, splitting, scaling, and encoding datasets like in [scikit-learn](https://scikit-learn.org/stable).

## Getting Started

### 1. Prerequisites

- **C++ Compiler**: Ensure you have a **C++17** (or higher) compatible compiler.
- **Data Files**: Place any required data files in the appropriate directory (e.g., `iris.csv` for the Iris dataset).

### 2. Build and Run the Examples

> I included some example scripts demonstrating how to use the engine. Compile any of these following and run the executable:

<table>
<tr>
  <th>Example</th>
  <th>Description</th>
  <th>Compile and Run</th>
</tr>
<tr>
  <td><a href="./backward_test.cpp">backward_test.cpp</a></td>
  <td>Demonstrate/verify the correctness of auto-differentiation by calculating the gradients of simple computation graph.</td>
  <td>
  
  ```bash
  g++ backward_test.cpp -o verify
  ./verify
  ```
  </td>
</tr>
<tr>
  <td><a href="./train_cubic.cpp">train_cubic.cpp</a></td>
  <td>Train a Neural Network to approximate a cubic function y = 2x³ + 3x² - 3x, demonstrating regression capabilities.</td>
  <td>
  
  ```bash
  g++ train_cubic.cpp -o train_cubic
  ./train_cubic
  ```
  </td>
</tr>
<tr>
  <td><a href="./train_iris.cpp">train_iris.cpp</a></td>
  <td>Load, preprocess, and train a Neural Network for multi-class classification on the <a href="https://www.kaggle.com/datasets/arshid/iris-flower-dataset">Iris</a> dataset using one-hot encoding.</td>
  <td>
  
  ```bash
  g++ train_iris.cpp -o train_iris
  ./train_iris
  ```
  </td>
</tr>
<tr>
  <td><a href="./train_mnist.cpp">train_mnist.cpp</a></td>
  <td>Similar to <a href="./train_iris.cpp">train_iris.cpp</a>, but train a model on the <a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv">MNIST</a> dataset for digit recognition with pixels as input features.</td>
  <td>
  
```bash
g++ train_mnist.cpp -o train_mnist
./train_mnist
```
  </td>
</tr>
</table>

***Note**: Before running, ensure that the **Iris** and **MNIST** dataset are available in the specified data directory. Here, I just simply use their `.csv` files directly from **Kaggle**.

## Core Components

The engine is organized into several header files (`.hpp`) located in the [n2n_autograd](./n2n_autograd/) folder, each including the classes and functions responsible for different aspects of the Neural Network and auto-differentiation operations.

## Potential Improvements

- [ ] **Extend Tensor Support**: Implement support for multi-dimensional `Tensor` (`Tensor` with more than 1 dimension).
- [ ] **Additional Layers**: Add more types of layers such as convolutional layers and recurrent layers.
- [ ] **Optimizers**: Implement more sophisticated optimization algorithms like Adam or RMSProp.
- [ ] **Concurrency**: The code currently runs on a single thread. Multi-threading or GPU acceleration can be explored for more computational efficiency or performance improvements on large datasets.
- [ ] **Model Serialization**: Add functionality to save and load trained models.