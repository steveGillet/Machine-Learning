import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the input data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the input data
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Define the autoencoder architecture
input_dim = x_train.shape[1]
latent_dim = 2

input_img = Input(shape=(input_dim,))
encoded = Dense(latent_dim, activation='relu')(input_img)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Create a separate model for the encoder
encoder = Model(input_img, encoded)

# Encode the test data
encoded_data = encoder.predict(x_test)

# Visualize the encoded data
plt.figure(figsize=(12, 10))
plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=y_test, cmap='viridis')
plt.colorbar()
plt.show()
