import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('titanic.csv', delimiter=',', usecols=(1,5,6), skip_header=1, skip_footer=1)
j = 0
for i in range(len(data)):
  if(np.isnan(data[j, 0])):
    data = np.delete(data, (j), axis=0)
    j -= 1
  j+=1
j = 0
for i in range(len(data)):
  if(np.isnan(data[j, 1])):
    data = np.delete(data, (j), axis=0)
    j -= 1
  j+=1
j = 0
for i in range(len(data)):
  if(np.isnan(data[j, 2])):
    data = np.delete(data, (j), axis=0)
    j -= 1
  j+=1

aliveDataAge = []
aliveDataSib = []
deadDataAge = []
deadDataSib = []

j = 0
k = 0
for i in range(len(data)):
  if data[i, 0] == 1:
    aliveDataAge.append(data[i, 1])
    aliveDataSib.append(data[i, 2])
    j+=1
  else:
    deadDataAge.append(data[i, 1])
    deadDataSib.append(data[i, 2])
    k+=1

aliveData = np.array([aliveDataAge, aliveDataSib])    
deadData = np.array([deadDataAge, deadDataSib])    

plt.scatter(aliveData[0,:], aliveData[1,:], label="alive")
plt.scatter(deadData[0,:], deadData[1,:], label="dead")
plt.legend()


# w = np.linalg.pinv(np.array([[4, 3, 2, 1, 1, 5],[1, 12, 3, 5, 6, 7]]))

data = np.append(data, np.ones((len(data),1)), axis=1)

w = np.linalg.pinv(np.array(data[:, [1,2,3]]).reshape((len(data),3))).dot(data[:, 0].reshape((len(data),1)))
print(w)
x1 = np.linspace(0, 80)
x2 = np.linspace(0, 8)
plt.plot(x1,x2,x1*w[0]+x2*w[1]+w[2])
plt.show()