from value import Value

def test_basic_operations():
    """Test basic arithmetic operations and gradients."""
    print("Testing basic operations...")
    
    # Test addition
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    c = a + b
    c.backward()
    print(f"a + b = {c.data}, da/dc = {a.grad}, db/dc = {b.grad}")
    assert c.data == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0
    
    # Reset gradients
    a.zero_grad()
    b.zero_grad()
    
    # Test multiplication
    d = a * b
    d.backward()
    print(f"a * b = {d.data}, da/dd = {a.grad}, db/dd = {b.grad}")
    assert d.data == 6.0
    assert a.grad == 3.0  # db/da = b
    assert b.grad == 2.0  # da/db = a
    
    print("✓ Basic operations passed\n")

def test_complex_expression():
    """Test a more complex expression: f(x,y) = x*y + x^2."""
    print("Testing complex expression: f(x,y) = x*y + x^2")
    
    x = Value(2.0, label='x')
    y = Value(3.0, label='y')
    
    # f = x * y + x**2
    f = x * y + x**2
    f.backward()
    
    print(f"f = x*y + x^2 = {f.data}")
    print(f"df/dx = {x.grad} (expected: y + 2*x = {y.data + 2*x.data})")
    print(f"df/dy = {y.grad} (expected: x = {x.data})")
    
    assert abs(f.data - 10.0) < 1e-6  # 2*3 + 2^2 = 10
    assert abs(x.grad - 7.0) < 1e-6   # 3 + 2*2 = 7
    assert abs(y.grad - 2.0) < 1e-6   # 2
    
    print("✓ Complex expression passed\n")

def test_activation_functions():
    """Test activation functions."""
    print("Testing activation functions...")
    
    # Test ReLU
    x1 = Value(2.0)
    y1 = x1.relu()
    y1.backward()
    print(f"relu(2.0) = {y1.data}, gradient = {x1.grad}")
    assert y1.data == 2.0
    assert x1.grad == 1.0
    
    x2 = Value(-1.0)
    y2 = x2.relu()
    y2.backward()
    print(f"relu(-1.0) = {y2.data}, gradient = {x2.grad}")
    assert y2.data == 0.0
    assert x2.grad == 0.0
    
    # Test sigmoid
    x3 = Value(0.0)
    y3 = x3.sigmoid()
    y3.backward()
    print(f"sigmoid(0.0) = {y3.data}, gradient = {x3.grad}")
    assert abs(y3.data - 0.5) < 1e-6
    assert abs(x3.grad - 0.25) < 1e-6  # sigmoid'(0) = 0.25
    
    print("✓ Activation functions passed\n")

def test_neural_network_example():
    """Test a simple 2-layer neural network computation."""
    print("Testing simple neural network computation...")
    
    # Input
    x = Value(0.5, label='x')
    
    # First layer: w1*x + b1
    w1 = Value(0.8, label='w1')
    b1 = Value(0.1, label='b1')
    h1 = w1 * x + b1  # Linear transformation
    a1 = h1.tanh()    # Activation
    
    # Second layer: w2*a1 + b2
    w2 = Value(-1.2, label='w2')
    b2 = Value(0.5, label='b2')
    output = w2 * a1 + b2
    
    # Backward pass
    output.backward()
    
    print(f"Network output: {output.data}")
    print(f"Gradients:")
    print(f"  dL/dx = {x.grad}")
    print(f"  dL/dw1 = {w1.grad}")
    print(f"  dL/db1 = {b1.grad}")
    print(f"  dL/dw2 = {w2.grad}")
    print(f"  dL/db2 = {b2.grad}")
    
    # Verify some basic properties
    assert isinstance(output.data, float)
    assert all(isinstance(v.grad, float) for v in [x, w1, b1, w2, b2])
    
    print("✓ Neural network example passed\n")

if __name__ == "__main__":
    test_basic_operations()
    test_complex_expression()
    test_activation_functions()
    test_neural_network_example()
    print("All tests passed! 🎉")