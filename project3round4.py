import numpy as np
import matplotlib.pyplot as plt

# number data points
N = 25
# number datasets
L = 100
# model complexity
m = 26
# test error data set size
testN = 1000
s = 0.1
D = np.zeros((L,2,N))
for i in range(L):
    X = np.zeros(N)
    X = np.random.uniform(0,1,N)
    t = np.sin(2*np.pi*X) + np.random.normal(0,0.3,N)
    D[i] = [X,t]    

lamb = np.linspace(0,5,N)
yTot = np.zeros((L,N))
variance = np.zeros(N)
bias = np.zeros(N)
tErr = np.zeros(N)
for l in range (N):
    avg = 0
    for k in range(L):
        phi = np.ones((N,m+1))
        for j in range(N):
            for i in range(1,m+1):
                phi[j][i] = np.exp((-(D[k][0][j] - (i-1)/(m-1))**2)/(2*s**2))


        w = np.zeros((N,m+1))
        phiT = np.transpose(phi)
        # phiT = np.reshape(phiT,(m+1,1))
        # phiTemp = np.reshape(phi[0],(1,m+1))
        # print(D[0][1])
        # print(np.reshape(D[0][1],(1,25)))
        phiP = np.linalg.pinv(phi)
        w = (np.linalg.inv(phiT.dot(phi) + lamb[l] * np.identity(m+1)).dot(phiT)).dot(D[k][1])

        x0 = np.linspace(0,1,N)
        y=w[0]
        for i in range(1,m+1):
            y += w[i]*np.exp((-(x0-(i-1)/(m-1))**2)/(2*s**2))
        yTot[k] = y

    for i in range(L):
        avg += yTot[i]
    avg /= L
    # plt.plot(x0,avg)
    # plt.scatter(D[k][0],D[k][1])
    # plt.show()
    for i in range(N):
        bias[l] += (avg[i] - np.sin(2*np.pi*avg[i]))**2
    bias[l] /= N*50
    difference = 0
    for j in range(N):
        for i in range(L):
            difference += (yTot[i][j] - avg[j])**2
        difference /= L
        variance[l] += difference
    variance[l] /= N
    tErr[l] = (bias[l] + lamb[l]*np.linalg.norm(w,2)) / 25
    
plt.plot(np.log(lamb),bias, label = 'Bias')
plt.plot(np.log(lamb),variance, label = 'Variance')
plt.plot(np.log(lamb),bias+variance, label = 'Bias + Variance')
plt.plot(np.log(lamb),tErr, label = 'Test Error')
plt.xlabel("ln λ")
plt.legend()
plt.show()