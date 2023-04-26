import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh_derivative(x):
    return 1 - x**2

# Neural Network
class NeuralNetwork:
    def __init__(self, input_units, hidden_units, output_units, activation, activation_derivative):
        self.W1 = np.random.randn(input_units, hidden_units)
        self.W2 = np.random.randn(hidden_units, output_units)
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1)
        self.A1 = self.activation(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2)
        self.A2 = self.activation(self.Z2)
        return self.A2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        dZ2 = (self.A2 - y) * self.activation_derivative(self.A2)
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        dZ1 = np.dot(dZ2, self.W2.T) * self.activation_derivative(self.A1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)

        self.W1 -= learning_rate * dW1
        self.W2 -= learning_rate * dW2

    def train(self, X, y, learning_rate, epochs):
        loss_history = []
        for epoch in range(epochs):
            self.forward(X)
            loss = np.mean((y - self.A2)**2)
            loss_history.append(loss)
            self.backward(X, y, learning_rate)
        return loss_history

# XOR classification problem
X_xor = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
y_xor = np.array([[1], [1], [0], [0]])

nn_xor = NeuralNetwork(2, 2, 1, sigmoid, sigmoid_derivative)
loss_history_xor = nn_xor.train(X_xor, y_xor, learning_rate=0.5, epochs=10000)

plt.figure()
plt.plot(loss_history_xor)
plt.title("Loss function value for XOR classification problem")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Regression problem
X_reg = np.linspace(-2, 2, 21).reshape(-1, 1)
y_reg = X_reg**3 + 0.5 * (X_reg**2) - 2 * X_reg - 1

nn_reg_3 = NeuralNetwork(1, 3, 1, tanh, tanh_derivative)
loss_history_reg_3 = nn_reg_3.train(X_reg, y_reg, learning_rate=0.01, epochs=10000)

nn_reg_20 = NeuralNetwork(1, 20, 1, tanh, tanh_derivative)
loss_history_reg_20 = nn_reg_20.train(X_reg, y_reg, learning_rate=0.01, epochs=10000)

plt.figure()
plt.plot(loss_history_reg_3, label="3 Hidden Units")
plt.plot(loss_history_reg_20, label="20 Hidden Units")
plt.title("Loss function value for regression problem")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot regression models
y_pred_reg_3 = nn_reg_3.forward(X_reg)
y_pred_reg_20 = nn_reg_20.forward(X_reg)

plt.figure()
plt.scatter(X_reg, y_reg, label="Input Data", color="blue")
plt.plot(X_reg, y_pred_reg_3, label="3 Hidden Units", color="red")
plt.plot(X_reg, y_pred_reg_20, label="20 Hidden Units", color="green")
plt.title("Regression models with input data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Report training errors
mse_reg_3 = mean_squared_error(y_reg, y_pred_reg_3)
mse_reg_20 = mean_squared_error(y_reg, y_pred_reg_20)

print(f"Training error for 3 hidden units: {mse_reg_3}")
print(f"Training error for 20 hidden units: {mse_reg_20}")

