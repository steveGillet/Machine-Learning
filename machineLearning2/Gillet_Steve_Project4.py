import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = img[:,:,0]
            img = np.resize(img, (28,28))
            (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img = img / 255
            images.append(img)
    return np.array(images)

folder_path = './Celegans_ModelGen/0'
negImages = load_images_from_folder(folder_path)
# negImages = negImages[:,:,:,0]
# negImages = negImages / 255

folder_path = './Celegans_ModelGen/1'
posImages = load_images_from_folder(folder_path)
# posImages = posImages[:,:,:,0]
# posImages = posImages / 255

trainingPercent = 0.8

nTrain = int(2222*trainingPercent + 2222*trainingPercent) -1
xTrain = np.vstack((negImages[:int(nTrain/2)], posImages[:int(nTrain/2)]))

tTrain = np.vstack((np.hstack((np.zeros((int(nTrain/2),1)), np.ones((int(nTrain/2),1)))), np.hstack((np.ones((int(nTrain/2),1)), np.zeros((int(nTrain/2),1))))))


# xTrain = np.load('DataXProcessed.npz')['DataX']  # Load preprocessed numpy matrix for images
# tTrain = np.load('DataTProcessed.npz')['DataT']
# nTrain = len(xTrain)

# plt.imshow(xTrain[2].reshape(28,28))
# plt.show()

shuffleIndices = np.arange(tTrain.shape[0])
np.random.shuffle(shuffleIndices)
xTrain = xTrain[shuffleIndices]
tTrain = tTrain[shuffleIndices]
# print(tTrain)

beta = 0.8
rho = 0.0001
K=2

def softmax(x):
    xShift = np.exp(x-np.max(x, axis=1, keepdims=True))
    return xShift / np.sum(xShift, axis=1, keepdims=True)

X = np.hstack((np.reshape(xTrain,(-1, xTrain.shape[1]*xTrain.shape[2])), np.ones((nTrain,1))))
# X = xTrain
w = np.random.randn(X.shape[1], K)

y = softmax(X.dot(w))
v = np.random.randn(X.shape[1], K)
epochs = 2000
loss = np.zeros(epochs)
for i in range(epochs):
  loss[i] = np.mean(-np.sum(tTrain*np.log(y+1e-15), axis=1))

  grad = X[:,:X.shape[1]].T.dot(y) - X[:,:X.shape[1]].T.dot(tTrain)
  
  v = beta * v + (1-beta) * grad
  w = w - rho * v
  y = softmax(X.dot(w))

  loss[i] = np.mean(-np.sum(tTrain*np.log(y+1e-15), axis=1))
  print(loss[i])

x0 = np.linspace(0,epochs,epochs)
plt.plot(x0, loss)
plt.show()

misses=0
for i in range(len(y)):
   if np.argmax(y[i]) != np.argmax(tTrain[i]):
       misses +=1

accuracy = (len(y) - misses) / len(y)       

print(accuracy)
np.save('weightsWorms', w)



###############################################################################################################################

# # Load the training images and labels
# train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
# train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

# # Load the test images and labels
# test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
# test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

# t = np.eye(10)[train_labels]
# nTrain = len(train_images) 
# M=10
# K = 10
# # w = np.random.randn(K,M+1)
# # phi = np.ones((M+1, 1))
# s=0.1
# beta = 0.9
# rho = 0.01
# # for j in range(1,M+1):
# #     phi[j] = np.exp((-(train_images[0].flatten()[0]-j/(M+1))**2)/(2*s**2))
# # y = np.exp(w.dot(phi))/np.sum(np.exp(w.dot(phi)))
# # print(y)

# def softmax(x):
#     xShift = np.exp(x-np.max(x, axis=1, keepdims=True))
#     return xShift / np.sum(xShift, axis=1, keepdims=True)

# X = np.hstack((np.reshape(train_images,(-1, train_images.shape[1]*train_images.shape[2])), np.ones((nTrain,1))))

# w = np.random.randn(X.shape[1], K)

# y = softmax(X.dot(w))
# v = np.random.randn(X.shape[1], K)
# print(y)
# for i in range(200):
#   loss = np.mean(-np.sum(t*np.log(y+1e-15), axis=1))
#   print(np.mean(-np.sum(t*np.log(y+1e-15), axis=1)))
#   grad = X[:,:X.shape[1]].T.dot(y) - X[:,:X.shape[1]].T.dot(t)
#   print(grad.shape)
#   v = beta * v + (1-beta) * grad
#   w = w - rho * v
#   y = softmax(X.dot(w))
#   print(y)
#   loss = np.mean(-np.sum(t*np.log(y+1e-15), axis=1))
#   print(loss)

# misses=0
# for i in range(len(y)):
#    if np.argmax(y[i]) != np.argmax(t[i]):
#        misses +=1

# accuracy = (len(y) - misses) / len(y)       

# print(accuracy)
# np.save('weightsMNIST', w)