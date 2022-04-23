import numpy as np
import matplotlib.pyplot as plt

L = 100
N = 25
lowL = -1
hiL = 3
s = 0.1
m = 4
lam = np.linspace(lowL,hiL,N)

D = np.ones((L,N))
D[:,:] = np.random.uniform(0,1,(L,N))

t = np.ones((L,N))
t[:,:] = np.sin(2*np.pi*D) + np.random.normal(0,0.3,(L,N))

x0 = np.linspace(0,1,N)

h = np.sin(2*np.pi*x0)

phi = np.ones((N,m+1))
w = np.ones((N, m+1))
w1 = np.ones((N, m+1))
w2 = np.ones((N, m+1))
fX = np.ones((L,N))
fX1 = np.ones((L,N))
fX2 = np.ones((L,N))

for k in range(0,N):
    for l in range(1,m+1):
        phi[k][l] = np.exp((-(D[0][k] - l/m)**2)/(2*s**2))
    phiT = np.transpose(phi)
    lamInv = 0 if lam[24] == 0 else 1/lam[24]
    w[k] = (np.linalg.pinv(phi).dot(t[0].reshape(25,1))+(lamInv*np.identity(m+1)).dot(phiT).dot(t[0].reshape(25,1))).reshape(1,5)
    w1[k] = np.linalg.inv(phiT.dot(phi)+lam[24]*np.identity(m+1)).dot(phiT).dot(t[0].reshape(25,1)).reshape(1,5)
    tempW = np.linalg.inv(phiT.dot(phi)+(lam[24]*np.identity(m+1)))
    w2[k] = tempW.dot(phiT).dot(t[0].reshape(25,1)).reshape(1,5)

for k in range(0,N):
    fX[0][k] = w[k][4]
    fX1[0][k] = w1[k][4]
    fX2[0][k] = w2[k][4]
    for l in range(m-1,0,-1):
        fX[0][k] += w[k][l]*np.exp((-(x0[k]-l/m)**2)/(2*s**2))
        fX1[0][k] += w1[k][l]*np.exp((-(x0[k]-l/m)**2)/(2*s**2))
        fX2[0][k] += w2[k][l]*np.exp((-(x0[k]-l/m)**2)/(2*s**2))

print(w)
print(w1)
print(w2)
print(fX[0])
print(fX1[0])
print(fX2[0])

plt.scatter(D[0],t[0])
plt.plot(x0,fX[0],color="orange")
plt.plot(x0,fX1[0],color="pink")
plt.plot(x0,fX2[0],color="red")
plt.show()

# for i in range(0,N):
#     fXsum = np.zeros((1,N))
#     fX1sum = np.zeros((1,N))
#     for j in range(0,L):
#         for k in range(0,N):
#             for l in range(1,m+1):
#                 phi[k][l] = np.exp((-(D[j][k] - l/m)**2)/(2*s**2))
#             phiT = np.transpose(phi)
#             lamInv = 0 if lam[i] == 0 else 1/lam[i]
#             w[k] = (np.linalg.pinv(phi).dot(t[j].reshape(25,1))+(lamInv*np.identity(m+1)).dot(phiT).dot(t[j].reshape(25,1))).reshape(1,5)
#             w1[k] = np.linalg.inv(phiT.dot(phi)+lam[i]*np.identity(m+1)).dot(phiT).dot(t[j].reshape(25,1)).reshape(1,5)


#         for k in range(0,N):
#             fX[j][k] = w[k][0]
#             fX1[j][k] = w1[k][0]
#             for l in range(1,m+1):
#                 fX[j][k] += w[k][l]*np.exp((-(x0[k]-l/m)**2)/(2*s**2))
#                 fX1[j][k] += w1[k][l]*np.exp((-(x0[k]-l/m)**2)/(2*s**2))
#     for j in range(0,L):
#         fXsum += fX[j]                      
#         fX1sum += fX1[j]
#     fXav = fXsum / L                  
#     fX1av = fX1sum / L
#     plt.scatter(D,t)
#     plt.plot(x0, h, color="orange")
#     plt.plot(x0,fXav.reshape(25,1), color="yellow")
#     plt.plot(x0,fX1av.reshape(25,1), color="green")
#     plt.show()

  