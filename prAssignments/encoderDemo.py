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
latent_dim = 32

input_img = Input(shape=(input_dim,))
encoded = Dense(latent_dim, activation='relu')(input_img)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder on the entire MNIST dataset
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_data=(x_test, x_test))

# Create a separate model for the encoder
encoder = Model(input_img, encoded)

# Test the autoencoder on another image
test_image = x_test[1]
reconstructed_image = autoencoder.predict(test_image.reshape(1, -1)).reshape(28, 28)

# Encode the test image
encoded_image = encoder.predict(test_image.reshape(1, -1))

# Reshape the encoded image for visualization
encoded_image_reshaped = encoded_image.reshape(4, 8)

# Visualize the test image, the reconstructed image, and the encoded image
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(test_image.reshape(28, 28), cmap='gray')
axes[0].set_title("Test Image")
axes[0].axis('off')

axes[1].imshow(reconstructed_image, cmap='gray')
axes[1].set_title("Reconstructed Image")
axes[1].axis('off')

axes[2].imshow(encoded_image_reshaped, cmap='gray')
axes[2].set_title("Encoded Image")
axes[2].axis('off')

plt.show()
