import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target.astype(int)

# Normalize the data
X /= 255.0

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
y_onehot = np.eye(10)[y]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Softmax function
def softmax(z):
    z_exp = np.exp(z)
    return z_exp / np.sum(z_exp, axis=1, keepdims=True)

# Training parameters
learning_rate = 0.01
epochs = 100
n_features = X_train.shape[1]
n_classes = y_onehot.shape[1]

# Initialize weights and biases
W = np.random.randn(n_features, n_classes)
b = np.zeros((1, n_classes))

# Training loop
for epoch in range(epochs):
    # Forward pass
    z = np.dot(X_train, W) + b
    y_hat = softmax(z)

    # Compute gradients
    dL_dz = y_hat - y_train
    dL_dW = np.dot(X_train.T, dL_dz)
    dL_db = np.sum(dL_dz, axis=0, keepdims=True)

    # Update weights and biases
    W -= learning_rate * dL_dW
    b -= learning_rate * dL_db

    # Calculate loss and accuracy
    loss = -np.mean(np.sum(y_train * np.log(y_hat + 1e-15), axis=1))
    accuracy = accuracy_score(np.argmax(y_train, axis=1), np.argmax(y_hat, axis=1))
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

# Test the model
z_test = np.dot(X_test, W) + b
y_test_hat = softmax(z_test)
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_test_hat, axis=1))
print(f"Test accuracy: {test_accuracy:.4f}")
