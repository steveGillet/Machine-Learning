import time
from PIL import Image
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score

startTime = time.time()

random.seed(3)

####################################################

# dir1 = input("Enter Directory Location:")
# list1 = os.listdir(dir1) 
# numFiles = len(list1)

# t = np.ones((numFiles,2))
# t[0:10] = [1,0]
# t[10:20] = [0,1]

# xTest = np.zeros((numFiles,101,101))
# xTestSmol = np.zeros((numFiles,67,67))

# for i in range (numFiles):
#     img = Image.open(dir1 + "/" + list1[i])
#     xTestSmol[i] = img.resize((67,67))
# xTest = np.array(xTest)

########################################################

N = 1000
numFiles = N
t = np.ones((numFiles,2))
t[0:500] = [1,0]
t[500:1000] = [0,1]

xTest = np.zeros((N,101,101))
xTestSmol = np.zeros((numFiles,67,67))

for i in range (int(N/2)):
    img = Image.open(r"C:\Users\steph\OneDrive\Desktop\School\Machine Learning\Celegans_ModelGen\0\image_{}.png".format(i+5000+1))
    xTestSmol[i] = img.resize((67,67))
    img = Image.open(r"C:\Users\steph\OneDrive\Desktop\School\Machine Learning\Celegans_ModelGen\1\image_{}.png".format(i+5000+1))
    xTestSmol[i+int(N/2)] = img.resize((67,67))

# ########################################################################################

# temp = list(zip(xTest, t))
# random.shuffle(temp)
# xTest, t = zip(*temp)
# xTest, t = list(xTest), list(t)
# xTest = np.array(xTest)
# t = np.array(t)

xTestSmol = xTestSmol/255.0

classes = np.unique(t)
numClass = len(classes)

wFile = open("wFile2", "rb")
w = np.load(wFile)

print(w)

xFlat = np.zeros((numFiles,67*67))
for i in range(numFiles):
    xFlat[i] = xTestSmol[i].flatten()

# phi = np.ones((N,101*101,m+1))
# for k in range(N):
#     for j in range(101*101):
#         for i in range(1,m+1):
#             phi[k][j][i] = np.exp((-(xFlat[k][j] - (i-1)/(m-1))**2)/(2*s**2))
a = np.zeros((numFiles))
y = np.zeros((numFiles,numClass))
yTest = np.zeros((numFiles,numClass))
for i in range(numFiles):
    a[i] = np.sum(xFlat[i]*w)
    y[i][0] = 1/(1+np.exp(-a[i]))
    y[i][1] = 1-y[i][0]

print(y)
for i in range(numFiles):
    yTest[i] = np.round(y[i])

print(yTest)
print(t)
print("Test  Accuracy : {:.3f}".format(accuracy_score(yTest, t)))

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))