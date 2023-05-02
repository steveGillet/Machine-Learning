import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten

(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

input_shape = x_train.shape[1:]
hidden_units = 64

input_layer = Input(shape=input_shape)
flatten_layer = Flatten()(input_layer)
encoder_layer = Dense(hidden_units, activation='relu')(flatten_layer)

decoder_layer = Dense(np.prod(input_shape), activation='sigmoid')(encoder_layer)
output_layer = Reshape(input_shape)(decoder_layer)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))

weights = autoencoder.layers[2].get_weights()[0]  # Get the weights of the encoder layer
weights = weights.reshape(input_shape + (hidden_units,))

# Calculate the number of rows and columns needed for the grid
grid_rows = int(np.ceil(np.sqrt(hidden_units)))
grid_cols = int(np.ceil(hidden_units / grid_rows))

plt.figure(figsize=(grid_rows * 2, grid_cols * 2))

for i in range(hidden_units):
    plt.subplot(grid_rows, grid_cols, i + 1)
    plt.imshow(weights[:, :, i], cmap='viridis', interpolation='nearest')
    plt.axis('off')

# Number of images to display
num_images = 16

plt.figure(figsize=(4, 4))

for i in range(num_images):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_train[i], cmap='gray', interpolation='nearest')
    plt.axis('off')

plt.suptitle('Original Images')

print(weights[:, :, 0].shape)
print(x_train[0].shape)

plt.show()
