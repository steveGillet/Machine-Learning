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
loss1 = (y-t)**2

plt.plot(x0,y, color="orange", label='Closed Form')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.legend()

plt.subplot(1,2,2)
xT = x.transpose()
xTmin = np.min(xT[0])
xTmax = np.max(xT[0])
tMax = np.max(t)
tMin = np.min(t)
xTav = np.mean(xT)
tAv = np.mean(t)

for i in range(len(xT[0])):
    xT[0][i] = (xT[0][i] - xTmin) / (xTmax - xTmin)

for i in range(len(t)):
    t[i] = (t[i] - tMin) / (tMax - tMin)

xTavS = np.mean(xT)
tAvS = np.mean(t)

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

xT[0] = xT[0] * (xTmax - xTmin) + xTmin
t = t * (tMax - tMin) + tMin

plt.scatter(x[:,0],t)

x0 = np.arange(1500,6000)

wK = wK * tMax/xTmax

print(wK)
y = wK[0]*x0+wK[1]
loss = (y - t)**2
plt.plot(x0, y, color="purple", label='Gradient Descent')
plt.legend()
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.tight_layout()
plt.show()    
    

