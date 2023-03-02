import numpy as np
from cvxopt import matrix, solvers
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_excel('Proj2DataSet.xlsx', header=None)
X = X.to_numpy()


C = 0.1
y=X[:,2]

# Construct the matrices for the QP problem
N = len(X) # number of training examples
K = np.dot(X, X.T) # kernel matrix
P = matrix(np.outer(y, y) * K)
q = matrix(-np.ones(N))
G = matrix(np.vstack((-np.eye(N), np.eye(N))))
h = matrix(np.hstack((np.zeros(N), C*np.ones(N))))
A = matrix(y.reshape(1, -1))
b = matrix(0.0)

# Solve the QP problem
sol = solvers.qp(P, q, G, h, A, b)

# Extract the solution
lambdas = np.array(sol['x']).reshape(-1)

sv = []
w=np.zeros((1,2))
for i in range(N):
  if lambdas[i] > 0.01 and lambdas[i] < C:
    w+=lambdas[i]*y[i]*X[i,:2]
    sv.append(i)

wl = 0


for i in sv:
  print(i)
  wl+=(1/y[i]-w.dot(X[i,:2]))

wl /= len(sv)
print(w)

plt.scatter(X[:60,0], X[:60,1])
plt.scatter(X[60:,0], X[60:,1], c='red')
plt.scatter(X[sv,0], X[sv,1], c='green', marker='x')
plt.ylim(-1, 6)
x0 = np.linspace(-1,6,len(sv))

y = (-wl-w[0,0]*x0)/w[0,1]
plt.plot(x0, y)
plt.plot(x0, y + 1)
plt.plot(x0, y - 1)
plt.show()

print(wl)