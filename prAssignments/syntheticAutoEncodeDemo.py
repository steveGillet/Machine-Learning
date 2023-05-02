import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model

# Generate synthetic data: noisy sinusoidal
n_samples = 300
x = np.linspace(0, 2 * np.pi, n_samples)
y = np.sin(x) + np.random.normal(0, 0.1, n_samples)
data = np.column_stack((x, y))

# Normalize the data
data = (data - data.min()) / (data.max() - data.min())

# Define the autoencoder architecture
input_dim = data.shape[1]
latent_dim = 1

input_img = Input(shape=(input_dim,))
encoded = Dense(latent_dim, activation='tanh')(input_img)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder on the synthetic data
autoencoder.fit(data, data, epochs=500, batch_size=32)

# Create a separate model for the encoder
encoder = Model(input_img, encoded)

# Encode the synthetic data
encoded_data = encoder.predict(data)

# Reconstruct the synthetic data
reconstructed_data = autoencoder.predict(data)

# Plot the original data, encoded data, and reconstructed data
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].scatter(data[:, 0], data[:, 1], c='b')
axes[0].set_title("Original Data")
axes[0].axis('equal')

axes[1].scatter(encoded_data[:, 0], np.zeros_like(encoded_data[:, 0]), c='r')
axes[1].set_title("Encoded Data")
axes[1].axis('equal')

axes[2].scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], c='g')
axes[2].set_title("Reconstructed Data")
axes[2].axis('equal')

plt.show()
