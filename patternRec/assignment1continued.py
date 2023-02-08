import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataFrame = pd.read_excel('Proj1DataSet.xlsx')
dataArray = dataFrame.to_numpy()
# print(dataArray)

virginica = dataArray[dataArray[:, 4] == 'virginica']
versicolor = dataArray[dataArray[:, 4] == 'versicolor']
setosa = dataArray[dataArray[:, 4] == 'setosa']

versivirgi = np.append(virginica, versicolor, 0)
print(versivirgi)
w = np.ones((1,5))
print(w)
wk = w
rho = 0.0001
# print(np.sum(setosa[:,0]) + np.sum(-versivirgi[:,0]))
for i in range(200): 
  wk = wk + rho*(np.array([np.sum(-versivirgi[:,0])+ np.sum(setosa[:,0]), np.sum(-versivirgi[:,1])+ np.sum(setosa[:,1]), np.sum(-versivirgi[:,2])+ np.sum(setosa[:,2]),np.sum(-versivirgi[:,3]) + np.sum(setosa[:,3]), 0]))

print(versivirgi[:,0])
print(np.sum(-versivirgi[:,0]))
print(-np.sum(versivirgi[:,0]))
print(wk.dot(np.array([np.sum(-versivirgi[:,0])+ np.sum(setosa[:,0]), np.sum(-versivirgi[:,1])+ np.sum(setosa[:,1]), np.sum(-versivirgi[:,2])+ np.sum(setosa[:,2]),np.sum(-versivirgi[:,3]) + np.sum(setosa[:,3]), 0])))

wtest = np.random.rand(1,3)
# np.append(wtest, np.array([1]), 1)
plt.scatter(setosa[:,0],setosa[:,1], c='red')
plt.scatter(virginica[:,0],virginica[:,1])
# plt.show()

for i in range(50): 
  wtest = wtest + rho*(np.array([np.sum(setosa[:,0])+ np.sum(-virginica[:,0]), np.sum(setosa[:,1])+ np.sum(-virginica[:,1]), 0]))
print(wtest)

x0 = np.linspace(4,8)
y = x0*wtest[0][0] + x0*wtest[0][1] + wtest[0][2]
plt.plot(x0, y)
plt.show()

wlintest = np.linalg.pinv(np.array(setosa[:,0]))

print(wtest[0][0])