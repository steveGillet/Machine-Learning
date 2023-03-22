import numpy as np
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# # Load the training images and labels
# train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
# train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

# # Load the test images and labels
# test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
# test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

# t = np.eye(10)[test_labels]
# nTrain = len(test_images) 
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

# X = np.hstack((np.reshape(test_images,(-1, test_images.shape[1]*test_images.shape[2])), np.ones((nTrain,1))))

# w = np.load('weights.npy')
# y = softmax(X.dot(w))
# print(y[1])
# print(X[1])
# print(t[1])
# plt.imshow(test_images[1])
# plt.show()

########################################################################################################

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

trainingPercent = 0.8

nTest = int(len(negImages)*(1-trainingPercent) + len(posImages)*(1-trainingPercent))-1
nTrain = int(len(negImages)*trainingPercent + len(posImages)*trainingPercent)
print(nTest)
print(nTrain)
xTest = np.vstack((negImages[int(nTrain/2):], posImages[int(nTrain/2):]))

tTest = np.vstack((np.hstack((np.zeros((int(nTest/2),1)), np.ones((int(nTest/2),1)))), np.hstack((np.ones((int(nTest/2),1)), np.zeros((int(nTest/2),1))))))
shuffleIndices = np.arange(tTest.shape[0])
np.random.shuffle(shuffleIndices)
xTest = xTest[shuffleIndices]
tTest = tTest[shuffleIndices]

beta = 0.9
rho = 0.01
K=2

def softmax(x):
    xShift = np.exp(x-np.max(x, axis=1, keepdims=True))
    return xShift / np.sum(xShift, axis=1, keepdims=True)

X = np.hstack((np.reshape(xTest,(-1, xTest.shape[1]*xTest.shape[2])), np.ones((nTest,1))))
w = np.load('weightsWorms.npy')

y = softmax(X.dot(w))

loss = np.mean(-np.sum(tTest*np.log(y+1e-15), axis=1))
print(loss)


misses=0
for i in range(len(y)):
   if np.argmax(y[i]) != np.argmax(tTest[i]):
       misses +=1

accuracy = (len(y) - misses) / len(y)       

print(accuracy)
np.save('weightsWorms', w)
