import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_loss(losses, title):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)

# Function to plot the decision surface
def plot_decision_surface(X, y, w1, w2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x1_vals = np.linspace(-2, 2, 100)
    x2_vals = np.linspace(-2, 2, 100)

    x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
    X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))

    layer1_grid = tanh(np.dot(X_grid, w1))
    layer2_grid = softmax(np.dot(layer1_grid, w2))
    y_grid = np.argmax(layer2_grid, axis=1).reshape(x1_grid.shape)

    ax.plot_surface(x1_grid, x2_grid, y_grid, cmap="coolwarm", alpha=0.6)

    ax.scatter(X[:, 0], X[:, 1], np.argmax(y, axis=1), c=np.argmax(y, axis=1), cmap="coolwarm", s=100, edgecolors="k")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Class")
    plt.title("Decision Surface")
# Function to plot the regression results
def plot_regression(X, T, x_plot, y_plot, y_pred, title):
    plt.figure()
    plt.plot(x_plot, y_plot, label="True function")
    plt.plot(x_plot, y_pred, label="Predicted function")
    plt.scatter(X, T, color="red", label="Input data")
    plt.legend()
    plt.title(title)

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


plot_loss(losses, "Loss for XOR Problem")
plot_decision_surface(X, y, w1, w2)

###### GENERATED DATASET ########

# Read Excel file
df = pd.read_excel("Proj5Dataset.xlsx")

# Extract input and target data
X = df.iloc[:, 0].values.reshape((-1,1))
T = df.iloc[:, 1].values.reshape((-1,1))

print("X:", X.shape)
print("T:", T.shape)

for hiddenDim, title in zip((3, 20), ("Regression with 3 hidden units", "Regression with 20 hidden units")):
    # Define network architecture
    inputDim, outputDim = 1, 1

    # Define learning rate and number of epochs
    rho = 0.001
    # Initialize weights
    w1 = np.random.uniform(size=(inputDim, hiddenDim))
    w2 = np.random.uniform(size=(hiddenDim, outputDim))

    # Train the network
    losses = []
    for epoch in range(epochs):
        # forward prop
        layer0 = X
        layer1 = tanh(np.dot(layer0, w1))
        layer2 = tanh(np.dot(layer1, w2))
        
        loss = np.mean((layer2 - T)**2)
        losses.append(loss)

        # backprop
        layer2_error = layer2 - T
        layer2_delta = layer2_error * tanhDerivative(layer2)
        
        layer1_error = np.dot(layer2_delta, w2.T)
        layer1_delta = layer1_error * tanhDerivative(layer1)
        
        w2 -= rho * np.dot(layer1.T, layer2_delta) / len(X)
        w1 -= rho * np.dot(layer0.T, layer1_delta) / len(X)

    # Plot the model together with the input data
    x_plot = np.linspace(-1, 1, 100).reshape((100,1))
    y_plot = np.sin(2 * np.pi * x_plot)
    y_pred = tanh(np.dot(tanh(np.dot(x_plot, w1)), w2))

    plot_loss(losses, title)
    plot_regression(X, T, x_plot, y_plot, y_pred, title)

plt.show()