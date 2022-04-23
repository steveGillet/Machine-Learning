import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import seed
from random import random

seed()

plt.subplot(1,2,1)
carBigData = pd.read_excel('proj1Dataset.xlsx')
cleanCarBig = carBigData[carBigData['Horsepower'] > 0]

x = np.ones((len(cleanCarBig), 2))
x[:,0] = cleanCarBig['Weight']

xP = np.linalg.pinv(x)

t = np.ones((len(cleanCarBig),1))
t[:,0] = cleanCarBig['Horsepower']
w = xP.dot(t)

print(w)

plt.scatter(x[:,0],t)
x0 = np.arange(1500,6000)
y = w[0]*x0+w[1]

plt.plot(x0,y, color="orange", label='Closed Form')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.legend()

plt.subplot(1,2,2)
xT = x.transpose()
xTmean = np.mean(xT[0])
xTstd = np.std(xT[0])

for i in range(len(xT[0])):
    xT[0][i] = (xT[0][i] - xTmean) / xTstd

for i in range(len(t)):
    t[i] = (t[i] - np.mean(t)) / np.std(t)

wK = np.ones((2,1))
wK[0]=random()*10
wK[1]=random()*10

wKT = wK.transpose()
tT = t.transpose()
p = 0.00025

while(True): 
    wK1 = wK - p*(2*wKT.dot(xT).dot(x)-2*tT.dot(x)).transpose()
    if((abs(wK-wK1)<.0001).all()):
        break
    wK = wK1
    wKT = wK.transpose()  
plt.scatter(x[:,0],t)

for i in range(len(xT[0])):
    xT[0][i] = xT[0][i] * xTstd + xTmean    
for i in range(len(t)):
    t[i] = t[i] * np.std(t) + np.mean(t)

print(wK)
    
plt.scatter(x[:,0],t)

y = wK[0]*x0+wK[1]
plt.plot(x0, y, color="purple", label='Gradient Descent')
plt.legend()
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.tight_layout()
plt.show()    
    

