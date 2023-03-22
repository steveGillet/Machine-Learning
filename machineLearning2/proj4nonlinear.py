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

beta = 0.8
rho = 0.0001
K=2
trainingPercent = 0.8

nTrain = int(2222*trainingPercent + 2222*trainingPercent) -1
xTrain = np.vstack((negImages[:int(nTrain/2)], posImages[:int(nTrain/2)]))

tTrain = np.vstack((np.hstack((np.ones((int(nTrain/2),1)), np.zeros((int(nTrain/2),1)))), np.hstack((np.zeros((int(nTrain/2),1)), np.ones((int(nTrain/2),1))))))

nFeatures = xTrain.shape[1]*xTrain.shape[2]
X = np.reshape(xTrain,(-1, nFeatures))
M=2
s=0.1
phi = np.ones((nTrain,nFeatures,M+1))
for i in range(nTrain):
  for j in range(nFeatures):
    for k in range(M):
      phi[i][j][k] = np.exp((-(X[i][j]-(k+1)/(M+1))**2)/(2*s**2))

w = np.random.randn(M+1, K, nFeatures)

shuffleIndices = np.arange(len(tTrain))
np.random.shuffle(shuffleIndices)
phi = phi[shuffleIndices]
tTrain = tTrain[shuffleIndices]


def softmax(x):
    xShift = np.exp(x-np.max(x, axis=1, keepdims=True))
    return xShift / np.sum(xShift, axis=1, keepdims=True)

a = w.dot(phi)
print(a)
print(a.shape)
y = softmax(a)
print(y)
print(y.shape)
# v = np.random.randn(X.shape[1], K)
# epochs = 10000
# loss = np.zeros(epochs)
# for i in range(epochs):
#   loss[i] = np.mean(-np.sum(tTrain*np.log(y+1e-15), axis=1))

#   grad = X[:,:X.shape[1]].T.dot(y) - X[:,:X.shape[1]].T.dot(tTrain)
  
#   v = beta * v + (1-beta) * grad
#   w = w - rho * v
#   y = softmax(X.dot(w))

#   loss[i] = np.mean(-np.sum(tTrain*np.log(y+1e-15), axis=1))
#   print(loss[i])

# x0 = np.linspace(0,epochs,epochs)
# plt.plot(x0, loss)
# plt.show()

# misses=0
# for i in range(len(y)):
#    if np.argmax(y[i]) != np.argmax(tTrain[i]):
#        misses +=1

# accuracy = (len(y) - misses) / len(y)       

# print(accuracy)
# np.save('weightsWorms', w)
