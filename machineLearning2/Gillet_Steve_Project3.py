import numpy as np
import matplotlib.pyplot as plt

testN = 1000
L = 100
N = 25
s = 0.1
X = np.random.uniform(0,1,(L,N))
t = np.sin(2*np.pi*X) + np.random.normal(0,.3,(L,N))
x0 = np.linspace(0,1,N)
hx = np.sin(2*np.pi*x0)
Xtest = np.random.uniform(0,1,(testN))
tTest = np.sin(2*np.pi*Xtest) + np.random.normal(0,.3,(testN))
# plt.plot(x0, hx)



M = 4
I = np.identity(M+1)
lamb = np.linspace(0,3,N)
phi = np.ones((N, M+1))
yMean = np.zeros((N, N))
bias = np.zeros((N, 1))
variance = np.zeros((N, 1))
testError = np.zeros((N, 1))
phiTest = np.ones((N, M+1))
for i in range(N):
  for j in range(1,M+1):
    phiTest[i,j] = np.exp((-(Xtest[i]-(j)/(M+1))**2)/(2*s**2))

for g in range(N):
  y = np.zeros((L, N))
  wTest = 0
  for h in range(L):
    for i in range(N):
      for j in range(1,M+1):
        phi[i,j] = np.exp((-(X[h,i]-(j)/(M+1))**2)/(2*s**2))

    w = (np.linalg.inv(phi.transpose().dot(phi)+lamb[g]*I)).dot(phi.transpose()).dot(t[h].reshape(N,1))
    wTest += w

    y[h]+=w[0]

    for j in range(1,M+1):
      y[h] += w[j]*np.exp((-(x0-j/(M+1))**2)/(2*s**2))

  wTest /= L


  yMean[g] = np.mean(y, axis=0)
  for h in range(N):
    bias[g] += (yMean[g, h]-hx[h])**2
    for i in range(L):
      variance[g] += (y[i,h] - yMean[g,h])**2
    variance[g] /= L
  bias[g] /= N
  variance[g] /= N
  # print(wTest)
  # print(phiTest)
  testError[g] = (np.sqrt((np.linalg.norm(phiTest.dot(wTest)-tTest, ord =2)**2)/testN))
  
  # plt.scatter(X,t,c='orange')
  # plt.plot(x0,yMean[g], c='black')
# print(testError)
plt.plot(np.log(lamb), bias)
plt.plot(np.log(lamb), variance)
plt.plot(np.log(lamb), bias+variance)
plt.plot(np.log(lamb), testError)
plt.show()
