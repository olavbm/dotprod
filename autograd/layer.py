import random
from typing import List
from .value import Value, relu, sigmoid, tanh


class Neuron:
    """
    A single neuron with weights, bias, and activation function.
    """
    
    def __init__(self, nin: int, activation: str = 'linear'):
        """
        Initialize a neuron.
        
        Args:
            nin: Number of input connections
            activation: Activation function ('linear', 'relu', 'sigmoid', 'tanh')
        """
        # Initialize weights randomly
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation
    
    def __call__(self, x: List[Value]) -> Value:
        """
        Forward pass through the neuron.
        
        Args:
            x: List of input Values
            
        Returns:
            Output Value after activation
        """
        # Compute weighted sum: w1*x1 + w2*x2 + ... + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        
        # Apply activation function
        if self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        elif self.activation == 'tanh':
            return act.tanh()
        else:  # linear activation
            return act
    
    def parameters(self) -> List[Value]:
        """Return all parameters (weights and bias) of this neuron."""
        return self.w + [self.b]


class Layer:
    """
    A layer of neurons (fully connected/dense layer).
    """
    
    def __init__(self, nin: int, nout: int, activation: str = 'linear'):
        """
        Initialize a layer.
        
        Args:
            nin: Number of inputs
            nout: Number of neurons (outputs)
            activation: Activation function for all neurons in this layer
        """
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
    
    def __call__(self, x: List[Value]) -> List[Value]:
        """
        Forward pass through the layer.
        
        Args:
            x: List of input Values
            
        Returns:
            List of output Values (one per neuron)
        """
        outs = [n(x) for n in self.neurons]
        return outs
    
    def parameters(self) -> List[Value]:
        """Return all parameters of this layer."""
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    Multi-Layer Perceptron (fully connected neural network).
    """
    
    def __init__(self, nin: int, nouts: List[int], activations: List[str] = None):
        """
        Initialize an MLP.
        
        Args:
            nin: Number of inputs
            nouts: List of layer sizes (number of neurons per layer)
            activations: List of activation functions per layer
        """
        sz = [nin] + nouts
        
        # Default to ReLU for hidden layers and linear for output
        if activations is None:
            activations = ['relu'] * (len(nouts) - 1) + ['linear']
        
        assert len(activations) == len(nouts), "Must provide activation for each layer"
        
        self.layers = [Layer(sz[i], sz[i+1], activations[i]) 
                      for i in range(len(nouts))]
    
    def __call__(self, x: List[Value]) -> List[Value]:
        """
        Forward pass through the entire network.
        
        Args:
            x: List of input Values
            
        Returns:
            List of output Values
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> List[Value]:
        """Return all parameters of the network."""
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        """Reset gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = 0.0


# Loss functions
def mse_loss(predictions: List[Value], targets: List[float]) -> Value:
    """
    Mean Squared Error loss function.
    
    Args:
        predictions: List of predicted Values
        targets: List of target values (floats)
        
    Returns:
        MSE loss as a Value
    """
    losses = [(pred - target)**2 for pred, target in zip(predictions, targets)]
    return sum(losses) / len(losses)


def cross_entropy_loss(logits: List[Value], target_idx: int) -> Value:
    """
    Cross-entropy loss for classification (single target).
    
    Args:
        logits: Raw output scores from the network
        target_idx: Index of the correct class
        
    Returns:
        Cross-entropy loss as a Value
    """
    # Compute softmax probabilities
    # First subtract max for numerical stability
    max_logit = max(logit.data for logit in logits)
    exp_logits = [logit.exp() for logit in logits]  # No need to subtract max in autograd
    sum_exp = sum(exp_logits)
    
    # Get probability of correct class
    correct_prob = exp_logits[target_idx] / sum_exp
    
    # Return negative log probability
    return -correct_prob.log()


def accuracy(predictions: List[Value], targets: List[int]) -> float:
    """
    Calculate accuracy for classification.
    
    Args:
        predictions: List of prediction Value lists (one per sample)
        targets: List of target class indices
        
    Returns:
        Accuracy as float between 0 and 1
    """
    correct = 0
    total = len(targets)
    
    for pred, target in zip(predictions, targets):
        # Find predicted class (argmax)
        pred_class = max(range(len(pred)), key=lambda i: pred[i].data)
        if pred_class == target:
            correct += 1
    
    return correct / total