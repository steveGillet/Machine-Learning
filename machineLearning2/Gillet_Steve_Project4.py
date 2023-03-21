import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

# Load the training images and labels
train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

# Load the test images and labels
test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

train_labels = np.eye(10)[train_labels]
M=10
K = 10
w = np.random.randn(K,M+1)
phi = np.ones((M+1, 1))
s=0.1
for j in range(1,M+1):
    phi[j] = np.exp((-(train_images[0].flatten()[0]-j/(M+1))**2)/(2*s**2))
y = np.exp(w.dot(phi))/np.sum(np.exp(w.dot(phi)))
print(y)