import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

class Autoencoder:
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Initialize weights and biases for the encoder and decoder
        self.W1 = np.random.randn(input_dim, latent_dim)
        self.b1 = np.random.randn(latent_dim)
        self.W2 = np.random.randn(latent_dim, input_dim)
        self.b2 = np.random.randn(input_dim)

    def encode(self, X):
        return sigmoid(np.dot(X, self.W1) + self.b1)

    def decode(self, Z):
        return sigmoid(np.dot(Z, self.W2) + self.b2)

    def forward(self, X):
        Z = self.encode(X)
        X_hat = self.decode(Z)
        return X_hat

    def train(self, X, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            Z = self.encode(X)
            X_hat = self.decode(Z)
            
            # Calculate loss
            loss = mse(X, X_hat)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

            # Backward pass
            dX_hat = mse_derivative(X, X_hat)
            dZ = np.dot(dX_hat * sigmoid_derivative(X_hat), self.W2.T)
            
            # Update weights and biases for the decoder
            self.W2 -= learning_rate * np.dot(Z.T, dX_hat * sigmoid_derivative(X_hat))
            self.b2 -= learning_rate * np.sum(dX_hat * sigmoid_derivative(X_hat), axis=0)
            
            # Update weights and biases for the encoder
            self.W1 -= learning_rate * np.dot(X.T, dZ * sigmoid_derivative(Z))
            self.b1 -= learning_rate * np.sum(dZ * sigmoid_derivative(Z), axis=0)

# Generate some synthetic data
X = np.random.rand(100, 10)

# Define autoencoder parameters
input_dim = 10
latent_dim = 2
epochs = 500
learning_rate = 0.01

# Create and train the autoencoder
autoencoder = Autoencoder(input_dim, latent_dim)
autoencoder.train(X, epochs, learning_rate)

# Encode and decode the data
encoded_data = autoencoder.encode(X)
decoded_data = autoencoder.decode(encoded_data)