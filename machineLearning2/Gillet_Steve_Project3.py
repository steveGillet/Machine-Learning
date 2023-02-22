import numpy as np
import matplotlib.pyplot as plt

L = 100
N = 25
s = 0.1
X = np.random.uniform(0,1,(L,N))
t = np.sin(2*np.pi*X) + np.random.normal(0,.3,(L,N))
plt.scatter(X[0],t[0])

M = 2
I = np.identity(M+1)
lamb = np.linspace(0,3,N)
phi = np.ones((N, M+1))

for i in range(N):
  for j in range(1,M+1):
    phi[i,j] = np.exp((-(X[0,i]-(j)/(M+1))**2)/(2*s**2))

w = (np.linalg.inv(phi.transpose().dot(phi)+lamb[0]*I)).dot(phi.transpose()).dot(t[0].reshape(N,1))

y = np.ones(N)
print(y)
x0 = np.linspace(0,1,N)
y[0] =w[0]
for i in range(N):
  for j in range(1,M+1):
    y[i] += w[j]*np.exp((-(x0-j/(M+1))**2)/(2*s**2))

# y[j] = w[j][0]
#         for k in range(1, m+1):
#             y[j] += w[j][k]*np.exp(-((x0 - k/m)**2)/(2*s**2))
# print(y)
plt.plot(x0,y)
plt.show()