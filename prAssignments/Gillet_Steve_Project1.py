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
pedalLen = dataArray[:,2]
pedalWid = dataArray[:,3]
classFlower = dataArray[:,4]

sepalLenMin = sepLen.min()
sepalWidMin = sepWid.min()
pedalLenMin = pedalLen.min()
pedalWidMin = pedalWid.min()

sepalLenMax = sepLen.max()
sepalWidMax = sepWid.max()
pedalLenMax = pedalLen.max()
pedalWidMax = pedalWid.max()

sepalLenMean = sepLen.mean()
sepalWidMean = sepWid.mean()
pedalLenMean = pedalLen.mean()
pedalWidMean = pedalWid.mean()

sepalLenVar = sepLen.var()
sepalWidVar = sepWid.var()
pedalLenVar = pedalLen.var()
pedalWidVar = pedalWid.var()

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
plt.scatter(pedalLen, classFlower, marker='x', c='red')
plt.xlim(0,8)
plt.ylim(1,3)
plt.subplot(2,2,4,title='PetW vs. Class')
plt.scatter(pedalWid, classFlower, marker='x', c='red')
plt.xlim(0,8)
plt.ylim(1,3)
plt.show()

x = setosa[:, :4]
print(x)