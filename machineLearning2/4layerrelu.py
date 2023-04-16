import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x**2

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Modified XOR input and output
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoded labels

input_dim, hidden_dim, output_dim = 2, 4, 2
learning_rate = 0.1
epochs = 30000

# Initialize weights
W1 = np.random.uniform(size=(input_dim, hidden_dim))
W2 = np.random.uniform(size=(hidden_dim, output_dim))

losses = []

for epoch in range(epochs):
    # Forward propagation
    layer0 = X
    layer1 = tanh(np.dot(layer0, W1))
    layer2 = softmax(np.dot(layer1, W2))
    
    # Calculate loss
    loss = -np.sum(y * np.log(layer2)) / len(X)
    losses.append(loss)

    # Backpropagation
    layer2_error = layer2 - y
    layer2_delta = layer2_error
    
    layer1_error = np.dot(layer2_delta, W2.T)
    layer1_delta = layer1_error * tanh_derivative(layer1)
    
    # Update weights
    W2 -= learning_rate * np.dot(layer1.T, layer2_delta) / len(X)
    W1 -= learning_rate * np.dot(layer0.T, layer1_delta) / len(X)

# Plot loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plot decision surface
x1_vals = np.linspace(-2, 2, 100)
x2_vals = np.linspace(-2, 2, 100)

x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))

layer1_grid = tanh(np.dot(X_grid, W1))
layer2_grid = softmax(np.dot(layer1_grid, W2))
y_grid = np.argmax(layer2_grid, axis=1).reshape(x1_grid.shape)

plt.contourf(x1_grid, x2_grid, y_grid, cmap="coolwarm", alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), cmap="coolwarm", s=100, edgecolors="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
