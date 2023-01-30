import numpy as np
import matplotlib.pyplot as plt

N=25
X = np.random.uniform(0,1,N)
print(X)
t = np.ones((N,1))
t[:,0] = np.sin(2*np.pi*X) + np.random.normal(0,0.3,N)
x0 = np.linspace(0,1)
h = np.sin(2*np.pi*x0)


print(t)



s = 0.1
m = 5
phi = np.ones((N,m+1))
phiPoly = np.ones((N,m+1))
for i in range (0,N):
    for j in range(m-1, -1, -1):
        phi[i][j] = np.exp((-(X[i]-(j)/(m-1))**2)/(2*s**2))
        phiPoly[i][j] = X[i]**(m-j)

phiT = np.transpose(phi)
phiPolyT = np.transpose(phiPoly)

bias = np.ones((N,1))
lam = np.linspace(-2,2,25)
for i in range (N):
    
    phiP = np.linalg.pinv(phi)
    wP = phiP.dot(t)

    A = phiT.dot(phi)+ lam[i] * np.identity(m+1)
    b = phiT.dot(t)
    w = np.linalg.lstsq(A, b, rcond=None)
    print(w)
    w = w[0]

    print(wP)

    phiPolyP = np.linalg.pinv(phiPoly)
    wPolyP = phiPolyP.dot(t)
    Apoly = phiPolyT.dot(phiPoly)+ lam[i] * np.identity(m+1)
    
    
    bPoly = phiPolyT.dot(t)
    


    wPoly = np.linalg.lstsq(Apoly, bPoly, rcond=None)
    wPoly = wPoly[0] 

    print(wPoly)

    y = 0
    yPoly = 0

    for j in range(0, m+1):
        e = 1
        if(j == 4):
            e = 0
        y += w[j]*np.exp((-(x0-(j)/(m-1))**2)/(2*s**2))**e
        yPoly += wPolyP[j]*x0**(m-j)
        
    for j in range (N):
        bias[i] += (y[j] - h[j])**2
    bias[i] /= N
    
    plt.scatter(X,t)
    plt.plot(x0,y)
    plt.plot(x0,yPoly, color = "red")
    plt.show()

plt.plot(lam,bias)
plt.show()
