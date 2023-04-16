import numpy as np
import matplotlib.pyplot as plt

# modified XOR
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y = np.array([[1,0], [0,1], [0,1], [1,0]])

def relu(x):
    return np.maximum(0, x)

def reluDerivative(x):
    return (x > 0).astype(float)

def softmax(x):
    # shiftX = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)), axis=1, keepdims=True)

def tanh(x):
    return np.tanh(x)

def tanhDerivative(x):
    return 1 - x**2

inputDim, hiddenDim, outputDim = 2, 2, 2
# learning rate
rho = 0.01
epochs = 10000

w1 = np.random.uniform(size=(inputDim, hiddenDim))
w2 = np.random.uniform(size=(hiddenDim, outputDim))

losses = []

for epoch in range(epochs):
    # forward prop
    layer0 = X
    layer1 = tanh(np.dot(layer0, w1))
    layer2 = softmax(np.dot(layer1, w2))
    
    loss = -np.sum(y * np.log(layer2)) / len(X)
    losses.append(loss)

    # backprop
    layer2error = layer2 - y
    layer2delta = layer2error
    
    layer1error = np.dot(layer2delta, w2.T)
    layer1delta = layer1error * tanhDerivative(layer1)
    
    w2 -= rho * np.dot(layer1.T, layer2delta) / len(X)
    w1 -= rho * np.dot(layer0.T, layer1delta) / len(X)

# Plot loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plot decision surface in 3D
x1_vals = np.linspace(-2, 2, 100)
x2_vals = np.linspace(-2, 2, 100)

x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))

layer1_grid = tanh(np.dot(X_grid, w1))
layer2_grid = softmax(np.dot(layer1_grid, w2))
y_grid = np.argmax(layer2_grid, axis=1).reshape(x1_grid.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot decision surface
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
ax.plot_surface(x1_grid, x2_grid, y_grid, cmap="coolwarm", alpha=0.6)

# Plot data points
ax.scatter(X[:, 0], X[:, 1], np.argmax(y, axis=1), c=np.argmax(y, axis=1), cmap="coolwarm", s=100, edgecolors="k")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Class")
plt.show()

