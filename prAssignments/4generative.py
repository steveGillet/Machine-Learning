import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Generate synthetic dataset
n_points = 1000
angles = np.random.uniform(0, 2 * np.pi, size=n_points)
radii = 2 + 0.1 * np.random.randn(n_points)
x_train = np.array([radii * np.cos(angles), radii * np.sin(angles)]).T

# Autoencoder architecture
input_data = Input(shape=(2,))
encoded = Dense(8, activation='relu')(input_data)
latent = Dense(1, activation='linear')(encoded)
decoded = Dense(8, activation='relu')(latent)
output_data = Dense(2, activation='linear')(decoded)

autoencoder = Model(input_data, output_data)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=200, batch_size=32, shuffle=True)

# Build the generator model
latent_input = Input(shape=(1,))
generator = Model(latent_input, autoencoder.layers[-1](autoencoder.layers[-2](latent_input)))

# Sample from the latent space and generate new points
n = 100
latent_samples = np.random.normal(loc=0.0, scale=1.0, size=(n, 1))
generated_points = generator.predict(latent_samples)

# Display original and generated points
plt.figure(figsize=(6, 6))
plt.scatter(x_train[:, 0], x_train[:, 1], c='b', alpha=0.5, label='Original points')
plt.scatter(generated_points[:, 0], generated_points[:, 1], c='r', alpha=0.5, label='Generated points')
plt.legend()
plt.show()
