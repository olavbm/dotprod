from value import Value
from layer import Neuron, Layer, MLP, mse_loss, cross_entropy_loss

def test_neuron():
    """Test single neuron."""
    print("Testing single neuron...")
    
    # Create a neuron with 2 inputs
    neuron = Neuron(2, activation='relu')
    
    # Test input
    x = [Value(1.0), Value(2.0)]
    output = neuron(x)
    
    print(f"Neuron output: {output.data}")
    
    # Test backward pass
    output.backward()
    print(f"Input gradients: [{x[0].grad}, {x[1].grad}]")
    print(f"Number of parameters: {len(neuron.parameters())}")
    
    assert isinstance(output, Value)
    assert len(neuron.parameters()) == 3  # 2 weights + 1 bias
    
    print("✓ Neuron test passed\n")

def test_layer():
    """Test a layer of neurons."""
    print("Testing layer...")
    
    # Create a layer: 3 inputs, 2 outputs
    layer = Layer(3, 2, activation='relu')
    
    # Test input
    x = [Value(1.0), Value(0.5), Value(-1.0)]
    outputs = layer(x)
    
    print(f"Layer outputs: [{outputs[0].data:.4f}, {outputs[1].data:.4f}]")
    print(f"Number of parameters: {len(layer.parameters())}")
    
    # Test backward pass
    loss = outputs[0] + outputs[1]  # Simple loss
    loss.backward()
    
    print(f"Input gradients: [{x[0].grad:.4f}, {x[1].grad:.4f}, {x[2].grad:.4f}]")
    
    assert len(outputs) == 2
    assert len(layer.parameters()) == 8  # 2 neurons * (3 weights + 1 bias) = 8
    
    print("✓ Layer test passed\n")

def test_mlp():
    """Test multi-layer perceptron."""
    print("Testing MLP...")
    
    # Create a simple network: 3 -> 4 -> 2
    mlp = MLP(3, [4, 2], ['relu', 'linear'])
    
    # Test input
    x = [Value(1.0), Value(0.5), Value(-0.5)]
    outputs = mlp(x)
    
    print(f"MLP outputs: [{outputs[0].data:.4f}, {outputs[1].data:.4f}]")
    print(f"Number of parameters: {len(mlp.parameters())}")
    
    # Expected parameters: (3*4 + 4) + (4*2 + 2) = 16 + 10 = 26
    assert len(mlp.parameters()) == 26
    
    # Test backward pass
    loss = outputs[0]**2 + outputs[1]**2
    loss.backward()
    
    print(f"Loss: {loss.data:.4f}")
    print(f"Input gradients: [{x[0].grad:.4f}, {x[1].grad:.4f}, {x[2].grad:.4f}]")
    
    print("✓ MLP test passed\n")

def test_loss_functions():
    """Test loss functions."""
    print("Testing loss functions...")
    
    # Test MSE loss
    predictions = [Value(1.5), Value(2.8)]
    targets = [1.0, 3.0]
    
    mse = mse_loss(predictions, targets)
    print(f"MSE Loss: {mse.data:.4f}")
    
    mse.backward()
    print(f"MSE gradients: [{predictions[0].grad:.4f}, {predictions[1].grad:.4f}]")
    
    # Test cross-entropy loss
    logits = [Value(2.0), Value(1.0), Value(0.5)]  # 3 classes
    target_class = 0  # First class is correct
    
    ce_loss = cross_entropy_loss(logits, target_class)
    print(f"Cross-entropy Loss: {ce_loss.data:.4f}")
    
    ce_loss.backward()
    print(f"CE gradients: [{logits[0].grad:.4f}, {logits[1].grad:.4f}, {logits[2].grad:.4f}]")
    
    print("✓ Loss function tests passed\n")

def test_training_step():
    """Test a complete training step."""
    print("Testing complete training step...")
    
    # Create a small network for XOR-like problem
    mlp = MLP(2, [3, 1], ['relu', 'linear'])
    
    # Sample data point
    x = [Value(1.0), Value(0.0)]
    target = 1.0
    
    # Forward pass
    pred = mlp(x)
    loss = (pred[0] - target)**2
    
    print(f"Initial prediction: {pred[0].data:.4f}")
    print(f"Initial loss: {loss.data:.4f}")
    
    # Backward pass
    mlp.zero_grad()
    loss.backward()
    
    # Simple gradient descent step
    learning_rate = 0.01
    for p in mlp.parameters():
        p.data -= learning_rate * p.grad
    
    # Forward pass again to see improvement
    mlp.zero_grad()
    x_new = [Value(1.0), Value(0.0)]
    pred_new = mlp(x_new)
    loss_new = (pred_new[0] - target)**2
    
    print(f"After one step prediction: {pred_new[0].data:.4f}")
    print(f"After one step loss: {loss_new.data:.4f}")
    
    print("✓ Training step test passed\n")

if __name__ == "__main__":
    test_neuron()
    test_layer()
    test_mlp()
    test_loss_functions()
    test_training_step()
    print("All layer tests passed! 🎉")