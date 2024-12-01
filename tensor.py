import numpy as np
from graphviz import Digraph


class Tensor:
    def __init__(self, data, label='', _children=(), _operation=''):
        self.label = label
        self.data = data
        self.gradient = 0.0 # Accumulated gradient
        self._backward = lambda: None # Default to no-op
        self._children = set(_children) # set() is to prevent duplicates for operations like a + a
        self._operation = _operation # Operation that produced this node

    def __repr__(self):
        return f'Tensor({self.data}, gradient={self.gradient}, label={self.label})'

    def __str__(self):
        return str(self.data)


    # ADD OPERATORS
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, label=str(other))
        output = Tensor(self.data + other.data, _children=(self, other), _operation='+')
        output.label = f'{self.label} + {other.label}'

        def _backward():
            # The gradient needs to be accumulated to avoid overwriting when performing operations for the same node
            # For example, if b = a + a, then b.gradient = 2
            # If we don't accumulate, then b.gradient = 1
            self.gradient += 1.0 * output.gradient # d(a + b)/da = 1
            other.gradient += 1.0 * output.gradient # d(a + b)/db = 1

        output._backward = _backward
        return output

    def __radd__(self, other):
        return self + other # For scalar addition where the scalar is on the left. Ex: 2 + a


    # MULTIPLICATION OPERATORS
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, label=str(other))
        output = Tensor(self.data * other.data, _children=(self, other), _operation='*')
        output.label = f'({self.label})' if len(self._children) > 0 else self.label
        output.label += f' * ({other.label})' if len(other._children) > 0 else f' * {other.label}'

        def _backward():
            self.gradient += other.data * output.gradient # d(a * b)/da = b
            other.gradient += self.data * output.gradient # d(a * b)/db = a

        output._backward = _backward
        return output

    def __rmul__(self, other):
        return self * other # For scalar multiplication where the scalar is on the left. Ex: 2 * a


    # POWER OPERATORS
    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'Currently only supporting int or float'
        other = other if isinstance(other, Tensor) else Tensor(other, label=str(other))
        output = Tensor(self.data**other.data, _children=(self, other))
        output._operation = f'^{other.data}' if other.data > 0 else f'^{other.data}\n(Division)'
        output.label = f'({self.label})' if len(self._children) > 0 else self.label
        output.label += f'^({other.label})' if len(other._children) > 0 else f'^{other.label}'

        def _backward():
            self.gradient += other.data * self.data**(other.data - 1) * output.gradient

        output._backward = _backward
        return output

    def __rpow__(self, other):
        return other**self


    # SUBTRACTION and DIVISION OPERATORS
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __truediv__(self, other):
        if self.data == 1: return other**-1
        return self * other**-1 # a / b = a * b^-1

    def __rtruediv__(self, other):
        if other == 1: return self**-1
        return other * self**-1


    # UNARY OPERATORS
    def __neg__(self):
        return self * -1

    def exp(self):
        output = Tensor(np.exp(self.data), f'exp({self.label})', _children=(self,), _operation='exp')
        def _backward():
            self.gradient += output.data * output.gradient
        output._backward = _backward
        return output

    def log(self):
        output = Tensor(np.log(self.data), f'log({self.label})', _children=(self,), _operation='log')
        def _backward():
            self.gradient += (1 / self.data) * output.gradient
        output._backward = _backward
        return output


    # ACTIVATION FUNCTIONS
    def sigmoid(self):
        return 1 / (1 + (-self).exp())

    def tanh(self):
        exp2x = (2 * self).exp()
        return (exp2x - 1) / (exp2x + 1)

    def relu(self):
        return self * (self.data > 0)


    # BACKPROPAGATION
    def backward(self):
        sorted_nodes, _ = self._topological_sort()
        self.gradient = 1.0 # starting point (usually dL/dL = 1)
        for node in reversed(sorted_nodes):
            node._backward()


    def _topological_sort(self):
        visited, edges = set(), set() # builds a set of all nodes and edges in a graph
        sorted_nodes = [] # builds a list of tensors in topological order

        def dfs(node): # Depth-first search
            if node not in visited: # If the node has not been visited
                visited.add(node)
                for child in node._children:
                    edges.add((child, node))
                    dfs(child)
                sorted_nodes.append(node)
        dfs(self)
        return sorted_nodes, edges


    def draw_computation_graph(self, width=5):
        graph = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
        nodes, edges = self._topological_sort()

        for tensor in nodes: # For any value in the graph, create a rectangular ('record') node for it
            label = tensor.label.split(' ')
            label = '\\n'.join(' '.join(label[i:i+width]) for i in range(0, len(label), width))
            label = f'{label} | Data: {tensor.data:.3f} | Gradient: {tensor.gradient:.3f}'

            uid = str(id(tensor))
            graph.node(name=uid, label=label, shape='record')
            if tensor._operation: # If this value is a result of some operation, create an operation node for it
                graph.node(name=uid + tensor._operation, label=tensor._operation)
                graph.edge(uid + tensor._operation, uid) # and connect this node to it

        for n1, n2 in edges:
            graph.edge(str(id(n1)), str(id(n2)) + n2._operation) # Connect n1 to the operation node of n2
        return graph