import math
from typing import Union, Set, List, Tuple, Callable


class Value:
    """
    A wrapper around a scalar value that enables automatic differentiation.
    
    This class tracks operations performed on scalar values and can compute
    gradients via backpropagation.
    """
    
    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = '', label: str = ''):
        """
        Initialize a Value object.
        
        Args:
            data: The scalar value
            _children: Tuple of parent Value objects that created this value
            _op: String representation of the operation that created this value
            label: Optional label for debugging/visualization
        """
        self.data = data
        self.grad = 0.0  # Gradient of this value
        self._backward = lambda: None  # Function to compute gradients of children
        self._prev = set(_children)  # Set of parent values
        self._op = _op  # Operation that created this value
        self.label = label
    
    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other: Union['Value', float, int]) -> 'Value':
        """Addition operation."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    def __radd__(self, other: Union[float, int]) -> 'Value':
        """Right addition (when Value is on the right side)."""
        return self + other
    
    def __mul__(self, other: Union['Value', float, int]) -> 'Value':
        """Multiplication operation."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __rmul__(self, other: Union[float, int]) -> 'Value':
        """Right multiplication."""
        return self * other
    
    def __sub__(self, other: Union['Value', float, int]) -> 'Value':
        """Subtraction operation."""
        return self + (-other)
    
    def __rsub__(self, other: Union[float, int]) -> 'Value':
        """Right subtraction."""
        return other + (-self)
    
    def __neg__(self) -> 'Value':
        """Negation operation."""
        return self * -1
    
    def __pow__(self, other: Union['Value', float, int]) -> 'Value':
        """Power operation."""
        assert isinstance(other, (int, float)), "Only int/float powers supported for now"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other: Union['Value', float, int]) -> 'Value':
        """Division operation."""
        return self * other**-1
    
    def __rtruediv__(self, other: Union[float, int]) -> 'Value':
        """Right division."""
        return other * self**-1
    
    def exp(self) -> 'Value':
        """Exponential function."""
        out = Value(math.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad  # derivative of exp(x) is exp(x)
        out._backward = _backward
        
        return out
    
    def log(self) -> 'Value':
        """Natural logarithm."""
        out = Value(math.log(self.data), (self,), 'log')
        
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self) -> 'Value':
        """Hyperbolic tangent activation function."""
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self) -> 'Value':
        """ReLU activation function."""
        out = Value(max(0, self.data), (self,), 'relu')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out
    
    def sigmoid(self) -> 'Value':
        """Sigmoid activation function."""
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        """
        Perform backpropagation to compute gradients.
        
        This implements reverse-mode automatic differentiation by traversing
        the computation graph in topological order and applying the chain rule.
        """
        # Build topological order of all nodes in the graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Set gradient of output to 1 and propagate backwards
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = 0.0


# Convenience functions for common operations
def exp(x: Value) -> Value:
    """Exponential function."""
    return x.exp()

def log(x: Value) -> Value:
    """Natural logarithm."""
    return x.log()

def tanh(x: Value) -> Value:
    """Hyperbolic tangent."""
    return x.tanh()

def relu(x: Value) -> Value:
    """ReLU activation."""
    return x.relu()

def sigmoid(x: Value) -> Value:
    """Sigmoid activation."""
    return x.sigmoid()