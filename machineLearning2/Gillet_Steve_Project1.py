import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

plt.subplot(1,2,1, title='Car Big Dataset')

def normalize(matrix):
  norm = np.linalg.norm(matrix)
  return matrix / norm

dataFrame = pd.read_excel('proj1Dataset.xlsx').query('Horsepower.notna()')

dataArray = np.asarray(dataFrame)

x = np.append(dataArray[:,0].reshape((len(dataArray), 1)), np.ones((len(dataArray),1)), axis=1)
Xn = np.append(normalize(dataArray[:,0].reshape((len(dataArray), 1))), np.ones((len(dataArray),1)), axis=1)

Xp = np.linalg.pinv(x)

t = dataArray[:,1].reshape((len(dataArray),1))

w = Xp.dot(t)
print('Closed Form Weights(Pinv):', w)

x0 = np.linspace(np.min(x[:,0]), np.max(x[:,0]), num=400)

plt.scatter(x[:,0], t, marker='x', c='red')
plt.plot(x0, x0*w[0] + w[1], label='Pinv Closed Form')
plt.legend(loc='upper right')
plt.xlabel('Weight')
plt.ylabel('Horsepower')

plt.subplot(1,2,2, title='Car Big Dataset')

random.seed()
w[0] = random.random() * 20 - 10
w[1] = random.random() * 20 - 10

rho = .0024

# Tn = normalize(t)
# print(Xn)
# print(Tn)

Wk = w
termCrit = False
while not termCrit:
  Wk0 = Wk
  Wk = Wk - rho*(2*Wk.transpose().dot(Xn.transpose()).dot(Xn) - 2*t.transpose().dot(Xn)).transpose()
  # could increase termination criteria by a couple orders of magnitude for faster, or decrease for more accurate. Sufficiently close either way.
  termCrit = ((abs(Wk0 - Wk)) < .0000001).all()
  # print(Wk)

Wk[0] = Wk[0]/np.linalg.norm(x)

print('Gradient Descent Weights:', Wk)
plt.scatter(x[:,0], t, marker='x', c='red')
plt.plot(x0, x0*Wk[0] + Wk[1], c='lime', label='Gradient Descent')
plt.legend(loc='upper right')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.show()