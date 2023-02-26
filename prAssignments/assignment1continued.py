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
# print(versivirgi)
w = np.ones((1,5))
# print(w)
wk = w
rho = 0.0001
# print(np.sum(setosa[:,0]) + np.sum(-versivirgi[:,0]))
for i in range(200): 
  wk = wk + rho*(np.array([np.sum(-versivirgi[:,0])+ np.sum(setosa[:,0]), np.sum(-versivirgi[:,1])+ np.sum(setosa[:,1]), np.sum(-versivirgi[:,2])+ np.sum(setosa[:,2]),np.sum(-versivirgi[:,3]) + np.sum(setosa[:,3]), 0]))

# print(versivirgi[:,0])
# print(np.sum(-versivirgi[:,0]))
# print(-np.sum(versivirgi[:,0]))
# print(wk.dot(np.array([np.sum(-versivirgi[:,0])+ np.sum(setosa[:,0]), np.sum(-versivirgi[:,1])+ np.sum(setosa[:,1]), np.sum(-versivirgi[:,2])+ np.sum(setosa[:,2]),np.sum(-versivirgi[:,3]) + np.sum(setosa[:,3]), 0])))

wtest = np.random.rand(3,1)
# np.append(wtest, np.array([1]), 1)
plt.scatter(setosa[:,0],setosa[:,1], c='red')
plt.scatter(virginica[:,0],virginica[:,1])
# plt.show()
print(np.append(np.ones((len(virginica), 1)), -np.ones((len(virginica), 1))))
Xbp = np.append(np.append(np.append(setosa[:,0], -virginica[:,0], 0).reshape(len(virginica)*2, 1), np.append(setosa[:,1], -virginica[:,1], 0).reshape(len(virginica)*2, 1), 1), np.append(np.ones((len(virginica), 1)), -np.ones((len(virginica), 1))).reshape(len(virginica)*2, 1) , 1)
print(Xbp)
print(wtest.transpose(), Xbp[0].reshape(3,1))
print(wtest.transpose().dot(Xbp[0].reshape(3,1)))
flag = 1
while flag:
  flag = 0
  for j in range(len(Xbp)):
    if wtest.transpose().dot(Xbp[j].reshape(3,1))[0] < 0:
      wtest = wtest + rho*(Xbp[j].reshape(3,1))
      flag = 1
print(wtest)

x0 = np.linspace(4,8)
plt.plot(x0, (-wtest[0]*x0-wtest[2])/wtest[1])
plt.show()

X = np.append(np.append(np.append(setosa[:,0], virginica[:,0], 0).reshape(len(virginica)*2, 1), np.append(setosa[:,1], virginica[:,1], 0).reshape(len(virginica)*2, 1), 1), np.ones((len(virginica)*2, 1)), 1)
# print(X)
t = np.append(np.ones((50,1)), -np.ones((50,1))).reshape(len(virginica)*2, 1)
# print(t)
wlintest = np.linalg.pinv(X.astype(int)).dot(t)

plt.scatter(setosa[:,0],setosa[:,1], c='red')
plt.scatter(virginica[:,0],virginica[:,1])

print(wlintest)
print(wlintest[2,0])
plt.plot(x0, (-wlintest[0,0]*x0-wlintest[2,0])/wlintest[1,0])
plt.show()
