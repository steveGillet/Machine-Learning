import numpy as np
import matplotlib.pyplot as plt

testN = 1000
L = 100
N = 25
D = np.ones((L,N))
D[:,:] = np.random.uniform(0,1,(L,N))
t = np.ones((L,N))
t[:,:] = np.sin(2*np.pi*D) + np.random.normal(0,0.3,(L,N))
s = 0.1
lam = np.linspace(0,2,N)
bias = np.linspace(0,2,N)
variance = np.linspace(0,2,N)
tErr = np.linspace(0,2,N)
m = 4
x0 = np.linspace(0,1,N)
h = np.linspace(0, 1, N)
h = np.sin(2*np.pi*x0)
h = np.array(h).reshape(N,1)

for l in range(0,N):
    phi = np.ones((N,m+1))
    w = np.ones((L,m+1))
    y = np.ones((L,len(x0)))

    for j in range(0,L):
        for i in range(0,N):
            for k in range(1,m+1):
                phi[i,k] = np.exp(-((D[j][i] - k/m)**2)/(2*s**2))
        phiT = np.transpose(phi)
        tCurr = np.array(t[j]).reshape((N,1))
        w[j] = (np.linalg.inv(phiT.dot(phi)+lam[l]*np.identity(m+1))).dot(phiT).dot(t[j])
        y[j] = w[j][0]
        for k in range(1, m+1):
            y[j] += w[j][k]*np.exp(-((x0 - k/m)**2)/(2*s**2))
    sumY = np.ones((1,len(x0)))
    for i in range(0,N):
        for j in range(0,L):
            sumY[0][i] += y[j][i]

    avgY = sumY / L
    avgY = np.transpose(avgY)

    for i in range(0,N):
        bias[l] += (avgY[i]-h[i])**2

    bias[l] = bias[l]/N

    for i in range(0,N):
        for j in range(0,L):
            variance[l] += (y[j][i] - avgY[i])**2
        variance[l] /= L
    variance[l] /= N

    # plt.plot(x0,(avgY+variance[l]), color = "red")
    # plt.plot(x0,(avgY-variance[l]), color = "red")
    # plt.plot(x0,h, color = "orange")
    # plt.plot(x0,avgY, color = "purple")
    # plt.show()

    xTest = np.random.uniform(0,1,testN)

    tTest = np.ones(testN)
    tTest = np.sin(2*np.pi*xTest) + np.random.normal(0,0.3,testN)

    phiTest = np.ones((testN, m+1))
    for i in range(0,testN):
        for m in range(1,m+1):
                phiTest[i,m] = np.exp(-((xTest[i] - m/4)**2)/(2*s**2))

    tErr[l] = (np.sqrt((np.linalg.norm(phiTest.dot(w[l])-tTest, ord =2)**2)/testN))

# plt.scatter(D,t)
# plt.plot(x0,avgY,color="orange")
# plt.show()

plt.plot(lam, tErr, color = "orange", label = 'Test Error')
# plt.plot(lam, ((bias-np.min(bias))/(np.max(bias)-np.min(bias))),color = "blue")
plt.plot(lam, bias, color = "blue", label = "Bias")
plt.plot(lam, variance, color = "yellow", label = "Variance")
plt.plot(lam, (variance+bias),color = "green", label = "Variance + Bias")
plt.legend()
plt.show()
