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

images = []
filenames = []
filenames = os.listdir(folderPath)
for filename in filenames:
    img = cv2.imread(os.path.join(folderPath, filename))
    images.append(img)
deployImages = np.array(images)
if model == 1:
    resizedImages = [cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA) for img in deployImages]
    grayscaleImages = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in resizedImages]
    blurredImages = [cv2.GaussianBlur(img, (3,3), 0) for img in grayscaleImages]
    edges = np.array([cv2.Canny(image=img, threshold1=60, threshold2=140) for img in blurredImages])
    linearizeImages = edges.reshape(edges.shape[0], edges.shape[1] * edges.shape[2])
else:
    deployImages = deployImages[:,:,:,0]
    linearizeImages = deployImages.reshape(deployImages.shape[0], deployImages.shape[1] * deployImages.shape[2])


X = linearizeImages / 255
X = np.insert(X, X.shape[1], 1, axis=1)

y = softmax(X.dot(w))    
  
predictions = np.argmax(y, axis=1) 
predictionTotals = np.unique(predictions, return_counts=True)
df1 = pd.DataFrame({'File Names': filenames, 'Predictions': predictions})
df2 = pd.DataFrame({'Unique Values': predictionTotals[0], 'Counts': predictionTotals[1]})

if model == 1:
    outputPath = 'outputWorms.xlsx'
else:
    outputPath = 'outputMNIST.xlsx'

writer = pd.ExcelWriter(outputPath, engine='openpyxl')
df1.to_excel(writer, sheet_name='Sheet1', index=False)
df2.to_excel(writer, sheet_name='Sheet1', index=False, startcol=3)
writer.save()

i = 0
# misclass = 0
for prediction in predictions:
    print(f'Prediction for image {filenames[i]}: {prediction}')
    # if prediction != 0:
    #     misclass +=1
    i+=1
print(f'Output File: {outputPath}')
# print(1-misclass/len(predictions))
