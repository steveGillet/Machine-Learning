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
sepalWidMin = sepWid.min()
petalLenMin = petalLen.min()
petalWidMin = petalWid.min()

sepalLenMax = sepLen.max()
sepalWidMax = sepWid.max()
petalLenMax = petalLen.max()
petalWidMax = petalWid.max()

sepalLenMean = sepLen.mean()
sepalWidMean = sepWid.mean()
petalLenMean = petalLen.mean()
petalWidMean = petalWid.mean()

sepalLenVar = sepLen.var()
sepalWidVar = sepWid.var()
petalLenVar = petalLen.var()
petalWidVar = petalWid.var()

swSetosa = 0
swVersicolor = 0
swVirginica = 0
for i in range(4):
  swSetosa += setosa[:,i].var() * 1/3
  swVersicolor += versicolor[:,i].var() * 1/3
  swVirginica += virginica[:,i].var() * 1/3


sbSetosa = 0
sbVersicolor = 0
sbVirginica = 0
for i in range(4):
  sbSetosa += 1/3 * (setosa[:,i].mean() - dataArray[:,i].mean())**2
  sbVersicolor += 1/3 * (versicolor[:,i].mean() - dataArray[:,i].mean())**2
  sbVirginica += 1/3 * (virginica[:,i].mean() - dataArray[:,i].mean())**2

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