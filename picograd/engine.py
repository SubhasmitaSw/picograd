import numpy as np
import graphviz
from typing import Union, Tuple, Set, Dict
import os

class DistributedTensor:
    """
    A class representing a distributed tensor.

    This class wraps numpy arrays and provides a way to handle distributed computations.
    """

    def __init__(self, data: Union[np.ndarray, float, int], dtype=np.float32):
        """
        Initialize a DistributedTensor.

        Args:
            data (Union[np.ndarray, float, int]): The data to be stored in the tensor.
            dtype (numpy.dtype, optional): The data type of the tensor. Defaults to np.float32.
        """
        self.data = np.array(data, dtype=dtype)
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    @classmethod
    def from_array(cls, arr: Union[np.ndarray, float, int]):
        """
        Create a DistributedTensor from an array-like object.

        Args:
            arr (Union[np.ndarray, float, int]): The array-like object to create the tensor from.

        Returns:
            DistributedTensor: A new DistributedTensor instance.
        """
        return cls(arr)

    def __repr__(self):
        return f"DistributedTensor(data={self.data}, shape={self.shape})"

class Value:
    """
    A class representing a value in the computational graph.

    This class is used to build and manipulate the computational graph for automatic differentiation.
    """

    _ops: Dict[str, Set['Value']] = {}

    def __init__(self, data, _children=(), _op='', label=None):
        """
        Initialize a Value object.

        Args:
            data: The data to be stored in the Value object.
            _children (tuple, optional): Child nodes in the computational graph. Defaults to ().
            _op (str, optional): The operation that produced this Value. Defaults to ''.
            label (str, optional): A label for the Value. Defaults to None.
        """
        self.data = DistributedTensor(data) if not isinstance(data, DistributedTensor) else data
        self.grad = DistributedTensor(np.zeros_like(self.data.data))
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.is_constant = not _children
        self.label = label or self._generate_label()

    def _generate_label(self):
        """Generate a label for the Value based on its data."""
        if np.prod(self.data.shape) == 1:
            return f"Value({self.data.data.item():.4f})"
        else:
            return f"Value(shape={self.data.shape})"

    def __add__(self, other):
        """Add two Values."""
        return self._binary_op(other, '+', np.add)

    def __radd__(self, other):
        """Reverse add two Values."""
        return self + other

    def __sub__(self, other):
        """Subtract two Values."""
        return self + (-other)

    def __rsub__(self, other):
        """Reverse subtract two Values."""
        return other + (-self)

    def __mul__(self, other):
        """Multiply two Values."""
        return self._binary_op(other, '*', np.multiply)

    def __rmul__(self, other):
        """Reverse multiply two Values."""
        return self * other

    def __neg__(self):
        """Negate a Value."""
        return self * -1

    def __truediv__(self, other):
        """Divide two Values."""
        return self * other**-1

    def __rtruediv__(self, other):
        """Reverse divide two Values."""
        return other * self**-1

    def __pow__(self, other):
        """Raise a Value to a power."""
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        return self._binary_op(other, f'**{other}', lambda x, y: x ** y)

    def relu(self):
        """Apply the ReLU function to this Value."""
        return self._unary_op('ReLU', lambda x: np.maximum(0, x))

    def tanh(self):
        """Apply the tanh function to this Value."""
        return self._unary_op('tanh', np.tanh)

    def exp(self):
        """Apply the exponential function to this Value."""
        return self._unary_op('exp', np.exp)

    def log(self):
        """Apply the natural logarithm function to this Value."""
        return self._unary_op('log', np.log)

    def _binary_op(self, other, op_symbol, op_func):
        """
        Perform a binary operation.

        Args:
            other: The other Value to perform the operation with.
            op_symbol (str): A symbol representing the operation.
            op_func (function): The function to perform the operation.

        Returns:
            Value: The result of the operation.
        """
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(op_func(self.data.data, other.data.data), (self, other), op_symbol)
        
        def _backward():
            if op_symbol == '+':
                self.grad.data += out.grad.data
                other.grad.data += out.grad.data
            elif op_symbol == '*':
                self.grad.data += other.data.data * out.grad.data
                other.grad.data += self.data.data * out.grad.data
            elif op_symbol.startswith('**'):
                power = float(op_symbol[2:])
                self.grad.data += (power * self.data.data**(power-1)) * out.grad.data
        out._backward = _backward
        return out

    def _unary_op(self, op_symbol, op_func):
        """
        Perform a unary operation.

        Args:
            op_symbol (str): A symbol representing the operation.
            op_func (function): The function to perform the operation.

        Returns:
            Value: The result of the operation.
        """
        out = Value(op_func(self.data.data), (self,), op_symbol)
        
        def _backward():
            if op_symbol == 'ReLU':
                self.grad.data += (out.data.data > 0) * out.grad.data
            elif op_symbol == 'tanh':
                self.grad.data += (1 - out.data.data**2) * out.grad.data
            elif op_symbol == 'exp':
                self.grad.data += out.data.data * out.grad.data
            elif op_symbol == 'log':
                self.grad.data += (1 / self.data.data) * out.grad.data
        out._backward = _backward
        return out

    def backward(self):
        """Perform backpropagation starting from this Value."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad.data = np.ones_like(self.data.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return self.label

def optimize_graph(root: Value):
    """
    Optimize the computational graph.

    This function performs various optimizations on the graph, such as fusing ReLU and Add operations.

    Args:
        root (Value): The root node of the computational graph.

    Returns:
        Value: The root node of the optimized graph.
    """
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(root)

    for v in topo:
        if v._op == 'ReLU' and len(v._prev) == 1:
            prev = next(iter(v._prev))
            if prev._op == '+':
                fused = Value(np.maximum(0, prev.data.data), prev._prev, 'FusedReLUAdd')
                def _backward():
                    grad = (fused.data.data > 0) * fused.grad.data
                    for child in prev._prev:
                        child.grad.data += grad
                fused._backward = _backward
                v._prev = fused._prev
                v.data = fused.data
                v._op = 'FusedReLUAdd'

    return root

def visualize_graph(root: Value, filename='computational_graph'):
    """
    Visualize the computational graph.

    This function creates a visual representation of the computational graph and saves it as a PNG file.

    Args:
        root (Value): The root node of the computational graph.
        filename (str, optional): The name of the file to save the visualization. Defaults to 'computational_graph'.
    """
    # Create a 'graph' folder if it doesn't exist
    graph_root = 'graphs'
    os.makedirs(graph_root, exist_ok=True)
    
    # Prepare the full file path
    file_path = os.path.join(graph_root, filename)
    
    dot = graphviz.Digraph(comment='Computational Graph')
    dot.attr(rankdir='LR')
    
    visited = set()
    
    def add_nodes(v: Value):
        if v not in visited:
            visited.add(v)
            label = v.label
            if np.prod(v.data.shape) == 1:
                label += f"\nvalue={v.data.data.item():.4f}"
                if np.prod(v.grad.shape) == 1:
                    label += f"\ngrad={v.grad.data.item():.4f}"
            dot.node(str(id(v)), label, shape='box')
            if v._op:
                dot.node(str(id(v)) + v._op, v._op, shape='ellipse')
                dot.edge(str(id(v)) + v._op, str(id(v)))
            for child in v._prev:
                add_nodes(child)
                dot.edge(str(id(child)), str(id(v)) + v._op)
    
    add_nodes(root)
    dot.render(file_path, view=True, format='png')
    print(f"Graph visualization saved as {file_path}.png")