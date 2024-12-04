import numpy as np
from tensor import Tensor


class Initializers:
    def he_uniform(fan_in, fan_out):
        # https://github.com/keras-team/keras/blob/master/keras/src/initializers/random_initializers.py#L525
        limit = np.sqrt(6 / fan_in)
        return np.random.uniform(-limit, limit)

    def glorot_uniform(fan_in, fan_out): # Xavier initialization
        # https://github.com/keras-team/keras/blob/master/keras/src/initializers/random_initializers.py#326
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit)
    
    
class Neuron:
    def __init__(self, input_size, activation=None, initializer=None, name='Neuron'):
        # If no initializer is provided:
        # Randomly initialize weights and bias with a uniform distribution between -1 and 1
        # Uniform distribution means that all values have an equal chance of being selected
        self.initializer = initializer if initializer else lambda: np.random.uniform(-1, 1)
        self.weights = [Tensor(self.initializer(), f'{name}_w{idx}') for idx in range(input_size)]
        self.bias = Tensor(self.initializer(), f'{name}_b')
        self.activation = activation
        self.name = name

    def __call__(self, inputs):
        z = sum([w * x for w, x in zip(self.weights, inputs)]) + self.bias # Future work: w @ x + b
        if self.activation == None or self.activation == 'softmax': return z
        if self.activation == 'sigmoid': return z.sigmoid()
        elif self.activation == 'tanh': return z.tanh()
        elif self.activation == 'relu': return z.relu()
        raise ValueError(f'Unknown activation function: {self.activation}')

    def parameters(self):
        return self.weights + [self.bias]


class Dense: # Fully connected layer
    def __init__(self, input_size, output_size, activation=None, initializer=None, name='Dense'):
        self.initializer = lambda: initializer(input_size, output_size) if initializer else None
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.name = name
        self.neurons = [
            Neuron(input_size, activation, self.initializer, name=f'{self.name}_N{neuron_idx}')
            for neuron_idx in range(output_size)
        ]

    def __call__(self, inputs):
        if self.activation == 'softmax':
            # Softmax is a special case where the activation is applied across the entire layer instead of individual neurons
            outputs = [neuron(inputs).exp() for neuron in self.neurons]
            outputs = [output / sum(outputs) for output in outputs]
        else: outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]