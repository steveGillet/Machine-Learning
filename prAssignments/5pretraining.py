import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist

np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.
x_test = x_test.reshape(-1, 784).astype('float32') / 255.

# Define the autoencoder
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

# Train the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Pretrain the classifier
classifier_output = Dense(10, activation='softmax')(encoded)
pretrained_classifier = Model(input_img, classifier_output)  # Fix the error here

# Train the classifier with pretrained weights
pretrained_classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_pretrained = pretrained_classifier.fit(x_train, y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test))

# Train the classifier with random weights
random_classifier = Model(input_img, Dense(10, activation='softmax')(input_img))
random_classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_random = random_classifier.fit(x_train, y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test))

# Plotting the results
plt.plot(history_pretrained.history['val_accuracy'], label='Pretrained')
plt.plot(history_random.history['val_accuracy'], label='Random')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.show()
