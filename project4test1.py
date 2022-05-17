import time
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import PIL
import glob

random.seed(3)

startTime = time.time()

N = 10000
xTrain = np.zeros((N,101,101))
xTrainSmol = np.zeros((N,67,67))
t = np.ones((N,2))
t[0:5000] = [1,0]
t[5000:N] = [0,1]
# m=10
# s=0.1
# learning rate
rho = 0.00000000001

for i in range (5000):
    img = Image.open(r"C:\Users\steph\OneDrive\Desktop\School\Machine Learning\Celegans_ModelGen\0\image_{}.png".format(i+1))
    xTrainSmol[i] = img.resize((67,67))
    img = Image.open(r"C:\Users\steph\OneDrive\Desktop\School\Machine Learning\Celegans_ModelGen\1\image_{}.png".format(i+1))
    xTrainSmol[i+5000]

temp = list(zip(xTrainSmol, t))
random.shuffle(temp)
xTrainSmol, t = zip(*temp)
xTrainSmol, t = list(xTrainSmol), list(t)
xTrainSmol = np.array(xTrainSmol)
t = np.array(t)

xTrainSmol = xTrainSmol / 255.0

classes = np.unique(t)
numClass = len(classes)

xFlat = np.zeros((N,67*67))
for i in range(N):
    xFlat[i] = xTrainSmol[i].flatten()

w = np.random.rand(67*67) * 2 - 1

# phi = np.ones((N,101*101,m+1))
# for k in range(N):
#     for j in range(101*101):
#         for i in range(1,m+1):
#             phi[k][j][i] = np.exp((-(xFlat[k][j] - (i-1)/(m-1))**2)/(2*s**2))

# phiFile = open("phiFile", "wb")
# np.save(phiFile, phi)
# phiFile.close

# phiFile = open("phiFile", "rb")
# phi = np.load(phiFile)

a = np.ones((N))
for i in range(N):
    a[i] = np.sum(xFlat[i]*w)

y = np.ones((N,numClass))
for i in range(N):
    y[i][0] = 1/(1+np.exp(-a[i]))
    y[i][1] = 1-y[i][0]

e = np.ones((N,numClass))
for i in range(N):
    e[i][0] = -t[i][0]*np.log(y[i][0])-(1-t[i][0])*np.log(1-y[i][0])

gradient = np.zeros((N,67*67))
for i in range(N):
    gradient[i] = (y[i][0]-t[i][0])*(xFlat[i])

beta = 0.9
v = np.sum(gradient)
v = beta*v+(1-beta)*np.sum(gradient)
print(w)
w = w - rho*v
print(w)
print(np.mean(e))

counter = 0
while(counter < 150):
    gradient = np.zeros((N,67*67))
    for i in range(N):
        a[i] = np.sum(xFlat[i]*w)
        y[i][0] = 1/(1+np.exp(-a[i]))
        y[i][1] = 1-y[i][0]
        e[i][0] = -t[i][0]*np.log(y[i][0])-(1-t[i][0])*np.log(1-y[i][0])
        gradient[i] = (y[i][0]-t[i][0])*(xFlat[i])
    v = beta*v+(1-beta)*np.sum(gradient)
    w = w - rho*v
    print(np.mean(e))
    counter += 1

print(w)

wFile = open("wFile2", "wb")
np.save(wFile, w)
wFile.close

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))