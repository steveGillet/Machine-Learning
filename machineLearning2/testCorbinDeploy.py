import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd

model = 0
while model != 1 and model != 2:
    model = int(input('Worms(1) or MNIST(2)? '))
if model == 1:
    w = np.load('trainedModelWorms.npz')['x']
elif model == 2:
    w = np.load('trainedModelMNIST.npz')['x']

folderPath = input('Name the directory where the images are located: ')

def softmax(x):
    # Compute softmax using log-sum-exp trick
    shift_x = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shift_x)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return probs

#deployImages = load_images_from_folder(folderPath)
images = []
filenames = []
filenames = os.listdir(folderPath)
for filename in filenames:
    img = cv2.imread(os.path.join(folderPath, filename))
    images.append(img)
deployImages = np.array(images)
resizedImages = [cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA) for img in deployImages]
grayscaleImages = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in resizedImages]
blurredImages = [cv2.GaussianBlur(img, (3,3), 0) for img in grayscaleImages]
sobelx = [cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) for img in blurredImages]
sobely = [cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) for img in blurredImages]
sobelxy = [cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) for img in blurredImages]
edges = np.array([cv2.Canny(image=img, threshold1=60, threshold2=140) for img in blurredImages])

linearizeImages = edges.reshape(edges.shape[0], edges.shape[1] * edges.shape[2])

X = linearizeImages / 255
X =  np.insert(X, X.shape[1], 1, axis=1)
y = softmax(X.dot(w))    
  
df = pd.DataFrame({'File Names': filenames, 'Predictions': np.argmax(y, axis=1)})
outputPath = 'output.xlsx'
df.to_excel(outputPath, index=False)
i = 1
for prediction in np.argmax(y, axis=1):
    print(f'Prediction for image {i}: {prediction}')
    i+=1
print(f'Output File: {outputPath}')