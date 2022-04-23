import numpy as np
import matplotlib.pyplot as plt

D = 1
N = 50

X = 2*np.random.uniform(0,D,N)-1
X = np.array(X)

T = X**2

print(X)
print(T)
plt.scatter(X,T)
plt.show()

n = [D, 20, 1]
n = np.array(n)
print(n)

L = len(n)

print(L)

MaxEp = 3000
rho = 0.1
beta = 0.9

print(X.dot(T.transpose()))
print(np.dot(X, T.transpose()))