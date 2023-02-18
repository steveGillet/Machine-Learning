import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataFrame = pd.read_excel('Proj1DataSet.xlsx')
setosa = dataFrame.query("species == 'setosa'").to_numpy()
versicolor = dataFrame.query("species == 'versicolor'").to_numpy()
virginica = dataFrame.query("species == 'virginica'").to_numpy()

dataArray = dataFrame.to_numpy()

dataArray = np.where(dataArray != 'setosa', dataArray, 1)
dataArray = np.where(dataArray != 'versicolor', dataArray, 2)
dataArray = np.where(dataArray != 'virginica', dataArray, 3)

sepLen = dataArray[:,0]
sepWid = dataArray[:,1]
petalLen = dataArray[:,2]
petalWid = dataArray[:,3]
classFlower = dataArray[:,4]

sepalLenMin = sepLen.min()
print('Sepal Length Minimum: ', sepalLenMin)
sepalWidMin = sepWid.min()
print('Sepal Width Minimum: ', sepalWidMin)
petalLenMin = petalLen.min()
print('Petal Length Minimum: ', petalLenMin)
petalWidMin = petalWid.min()
print('Petal Width Minimum: ', petalWidMin)

sepalLenMax = sepLen.max()
print('Sepal Length Maximum: ', sepalLenMax)
sepalWidMax = sepWid.max()
print('Sepal Width Maximum: ', sepalWidMax)
petalLenMax = petalLen.max()
print('Petal Length Maximum: ', petalLenMax)
petalWidMax = petalWid.max()
print('Petal Width Maximum: ', petalWidMax)

sepalLenMean = sepLen.mean()
print('Sepal Length Mean: ', sepalLenMean)
sepalWidMean = sepWid.mean()
print('Sepal Width Mean: ', sepalWidMean)
petalLenMean = petalLen.mean()
print('Petal Length Mean: ', petalLenMean)
petalWidMean = petalWid.mean()
print('Petal Width Mean: ', petalWidMean)

sepalLenVar = sepLen.var()
print('Sepal Length Variance: ', sepalLenVar)
sepalWidVar = sepWid.var()
print('Sepal Width Variance: ', sepalWidVar)
petalLenVar = petalLen.var()
print('Petal Length Variance: ', petalLenVar)
petalWidVar = petalWid.var()
print('Petal Width Variance: ', petalWidVar)

swSetosa = 0
swVersicolor = 0
swVirginica = 0
for i in range(4):
  swSetosa += setosa[:,i].var() * 1/3
  swVersicolor += versicolor[:,i].var() * 1/3
  swVirginica += virginica[:,i].var() * 1/3
print('Within-Class Variance Setosa: ', swSetosa)
print('Within-Class Variance Versicolor: ', swVersicolor)
print('Within-Class Variance Virginica: ', swVirginica)


sbSetosa = 0
sbVersicolor = 0
sbVirginica = 0
for i in range(4):
  sbSetosa += 1/3 * (setosa[:,i].mean() - dataArray[:,i].mean())**2
  sbVersicolor += 1/3 * (versicolor[:,i].mean() - dataArray[:,i].mean())**2
  sbVirginica += 1/3 * (virginica[:,i].mean() - dataArray[:,i].mean())**2

print('Between-Class Variance Setosa: ', sbSetosa)
print('Between-Class Variance Versicolor: ', sbVersicolor)
print('Between-Class Variance Virginica: ', sbVirginica)

corrcoef = np.corrcoef(dataArray.astype(int), rowvar=False)

plt.imshow(corrcoef, cmap='jet')
plt.colorbar()
plt.xticks([0,1, 2, 3, 4],['SepL', 'SepW', 'PetL', 'PetW', 'Class'])
plt.yticks([0,1, 2, 3, 4],['SepL', 'SepW', 'PetL', 'PetW', 'Class'])
plt.show()

plt.subplot(2,2,1,title='SepL vs. Class')
plt.scatter(sepLen, classFlower, marker='x', c='red')
plt.xlim(0,8)
plt.ylim(1,3)
plt.subplot(2,2,2,title='SepW vs. Class')
plt.scatter(sepWid, classFlower, marker='x', c='red')
plt.xlim(0,8)
plt.ylim(1,3)
plt.subplot(2,2,3,title='PetL vs. Class')
plt.scatter(petalLen, classFlower, marker='x', c='red')
plt.xlim(0,8)
plt.ylim(1,3)
plt.subplot(2,2,4,title='PetW vs. Class')
plt.scatter(petalWid, classFlower, marker='x', c='red')
plt.xlim(0,8)
plt.ylim(1,3)
plt.show()

########################################################################
print('Setosa Vs. Versi + Virgi, All Features') 
rho = 0.001
x = np.append(np.append(dataArray[:50, :4], np.ones((50,1)),1), np.append(-dataArray[50:, :4], -np.ones((100,1)),1), 0)
w = np.random.rand(5,1)
epochs = 0
flag = 1
while flag:
  flag = 0
  for j in range(len(x)):
    if w.transpose().dot(x[j].reshape(5,1))[0] < 0:
      w = w + rho*(x[j].reshape(5,1))
      flag = 1
  epochs +=1
print('Epochs: ', epochs)
print('Weights: ', w)

x = np.append(dataArray[:,:4], np.ones((150,1)),1)
t = np.append(np.ones((50,1)), -np.ones((100,1)), 0)
w = np.linalg.pinv(x.astype(int)).dot(t)
print('Weights (LS): ', w)
misclassified = 0
for x in x[:50]:
  if w.transpose().dot(x.reshape(5,1)) < 0:
    misclassified+=1
for x in x[50:]:
  if w.transpose().dot(x.reshape(5,1)) > 0:
    misclassified+=1
print('Misclassifications: ', misclassified)

########################################################################
print('Setosa Vs. Versi + Virgi, Features 3 and 4') 
plt.title('Setosa Vs. Versi + Virgi, Features 3 and 4') 
rho = 0.001
x = np.append(np.append(dataArray[:50, 2:4], np.ones((50,1)),1), np.append(-dataArray[50:, 2:4], -np.ones((100,1)),1), 0)
w = np.random.rand(3,1)
epochs = 0
flag = 1
while flag:
  flag = 0
  for j in range(len(x)):
    if w.transpose().dot(x[j].reshape(3,1))[0] < 0:
      w = w + rho*(x[j].reshape(3,1))
      flag = 1
  epochs +=1
print('Epochs: ', epochs)
print('Weights: ', w)

plt.scatter(setosa[:,2],setosa[:,3], c='red', label='Setosa')
plt.scatter(dataArray[50:,2],dataArray[50:,3], label='Versi+Virgi')

x0 = np.linspace(np.min(dataArray[:,2]), np.max(dataArray[:,2]))
plt.plot(x0, (-w[0,0]*x0-w[2,0])/w[1,0], c='green', label='Batch Perceptron Decision Boundary')



x = np.append(dataArray[:,2:4], np.ones((150,1)),1)
t = np.append(np.ones((50,1)), -np.ones((100,1)), 0)
w = np.linalg.pinv(x.astype(int)).dot(t)
print('Weights (LS): ', w)
misclassified = 0
for x in x[:50]:
  if w.transpose().dot(x.reshape(3,1)) < 0:
    misclassified+=1
for x in x[50:]:
  if w.transpose().dot(x.reshape(3,1)) > 0:
    misclassified+=1
print('Misclassifications: ', misclassified)

plt.plot(x0, (-w[0,0]*x0-w[2,0])/w[1,0], c='orange', label='Least Squares Decision Boundary')
# plt.xlim(np.min(dataArray[:,2]), np.max(dataArray[:,2]))
plt.ylim(np.min(dataArray[:,3]) - 0.5, np.max(dataArray[:,3]) + 0.5)
plt.legend()
plt.xlabel('Feature 3 (Petal Length)')
plt.ylabel('Feature 4 (Petal Width)')
plt.show()

########################################################################
print('Virgi Vs. Versi + Setosa, All Features') 
rho = 0.001
x = np.append(np.append(dataArray[:100, :4], np.ones((100,1)),1), np.append(-dataArray[100:, :4], -np.ones((50,1)),1), 0)
D = len(x[0])
w = np.random.rand(D,1)
epochs = 0
flag = 1
nonConvergence = 0
while flag and nonConvergence < 1000:
  nonConvergence += 1
  flag = 0
  for j in range(len(x)):
    if w.transpose().dot(x[j].reshape(D,1))[0] < 0:
      w = w + rho*(x[j].reshape(D,1))
      flag = 1
  epochs +=1
print('Epochs: ', epochs)
print('Weights: ', w)

x = np.append(dataArray[:,:4], np.ones((150,1)),1)
t = np.append(np.ones((100,1)), -np.ones((50,1)), 0)
w = np.linalg.pinv(x.astype(int)).dot(t)
print('Weights (LS): ', w)
misclassified = 0
for i in range(0,100):
  if w.transpose().dot(x[i].reshape(D,1)) < 0:
    misclassified += 1

for i in range(100,150):
  if w.transpose().dot(x[i].reshape(D,1)) > 0:
    misclassified += 1  
print('Misclassifications: ', misclassified)

########################################################################
print('Virgi Vs. Versi + Setosa, Features 3 and 4')
plt.title('Virgi Vs. Versi + Setosa, Features 3 and 4') 
rho = 0.001
x = np.append(np.append(dataArray[:100, 2:4], np.ones((100,1)),1), np.append(-dataArray[100:, 2:4], -np.ones((50,1)),1), 0)
D = len(x[0])
w = np.random.rand(D,1)
epochs = 0
flag = 1
nonConvergence = 0
while flag and nonConvergence < 1000:
  nonConvergence += 1
  flag = 0
  for j in range(len(x)):
    if w.transpose().dot(x[j].reshape(D,1))[0] < 0:
      w = w + rho*(x[j].reshape(D,1))
      flag = 1
  epochs +=1
print('Epochs: ', epochs)
print('Weights: ', w)

x = np.append(dataArray[:,2:4], np.ones((150,1)),1)
t = np.append(np.ones((100,1)), -np.ones((50,1)), 0)
w = np.linalg.pinv(x.astype(int)).dot(t)
print('Weights (LS): ', w)
misclassified = 0

for i in range(0,100):
  if w.transpose().dot(x[i].reshape(D,1)) < 0:
    misclassified += 1

for i in range(100,150):
  if w.transpose().dot(x[i].reshape(D,1)) > 0:
    misclassified += 1  

print('Misclassifications: ', misclassified)
print(w)
plt.scatter(virginica[:,2],virginica[:,3], c='red', label='Virginica')
plt.scatter(dataArray[:100,2],dataArray[:100,3], label='Versi+Setosa')
plt.plot(x0, (-w[0,0]*x0-w[2,0])/w[1,0], c='orange', label='Least Squares Decision Boundary')
# plt.xlim(np.min(dataArray[:,2]), np.max(dataArray[:,2]))
plt.ylim(np.min(dataArray[:,3]) - 0.5, np.max(dataArray[:,3]) + 0.5)
plt.legend()
plt.xlabel('Feature 3 (Petal Length)')
plt.ylabel('Feature 4 (Petal Width)')
plt.show()

########################################################################
print('Setosa Vs. Versi Vs. Virgi, Features 3 and 4')
plt.title('Setosa Vs. Versi Vs. Virgi, Features 3 and 4') 

x = np.append(dataArray[:,2:4], np.ones((150,1)),1)
D = len(x[0])
t = np.zeros((150,3))
t[:50, 0] = 1
t[50:100, 1] = 1
t[100:150, 2] = 1

w = np.linalg.pinv(x.astype(int)).dot(t)

print('Weights (LS): ', w)
misclassified = 0

for i in range(0,50):
  if w.transpose()[0].dot(x[i].reshape(D,1)) < w.transpose()[1].dot(x[i].reshape(D,1)) or w.transpose()[0].dot(x[i].reshape(D,1)) < w.transpose()[2].dot(x[i].reshape(D,1)):
    misclassified += 1

for i in range(50,100):
  if w.transpose()[1].dot(x[i].reshape(D,1)) < w.transpose()[0].dot(x[i].reshape(D,1)) or w.transpose()[1].dot(x[i].reshape(D,1)) < w.transpose()[2].dot(x[i].reshape(D,1)):
    misclassified += 1

for i in range(100,150):
  if w.transpose()[2].dot(x[i].reshape(D,1)) < w.transpose()[1].dot(x[i].reshape(D,1)) or w.transpose()[2].dot(x[i].reshape(D,1)) < w.transpose()[0].dot(x[i].reshape(D,1)):
    misclassified += 1

print('Misclassifications: ', misclassified)

plt.scatter(setosa[:,2],setosa[:,3], c='red', label='Setosa')
plt.scatter(versicolor[:,2],versicolor[:,3], c='green', label='Versicolor')
plt.scatter(virginica[:,2],virginica[:,3], c='blue', label='Virginica')

plt.plot(x0, (-x0*(w[0,0]-w[0,1])-w[2,0]+w[2,1])/(w[1,0]-w[1,1]), c='orange', label='Setosa Vs Versicolor Decision Boundary')
plt.plot(x0, (-x0*(w[0,1]-w[0,2])-w[2,1]+w[2,2])/(w[1,1]-w[1,2]), c='yellow', label='Versicolor Vs Virginica Decision Boundary')
plt.plot(x0, (-x0*(w[0,0]-w[0,2])-w[2,0]+w[2,2])/(w[1,0]-w[1,2]), c='purple', label='Setosa Vs Virginica Decision Boundary')
plt.legend()
plt.xlabel('Feature 3 (Petal Length)')
plt.ylabel('Feature 4 (Petal Width)')
plt.ylim(np.min(dataArray[:,3]) - 0.5, np.max(dataArray[:,3]) + 0.5)
plt.show()
