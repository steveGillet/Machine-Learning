#####################################################################

# nTrain = 100
# Xtrain = np.random.uniform(0,1,nTrain)

# tTrain = np.sin(2*np.pi*Xtrain) + np.random.normal(0,0.3,nTrain)

# # plt.show()

# nTest = 100
# Xtest = np.random.uniform(0,1,nTest)
# tTest = np.sin(2*np.pi*Xtest) + np.random.normal(0,0.3,nTest)
# # plt.show()

# trainError = []
# testError = []

# for k in range(0, 10):
#   # plt.scatter(Xtrain, tTrain)
#   # plt.scatter(Xtest, tTest, c='red')
#   M = k
#   phiTrain = np.ones((nTrain, M+1))

#   for i in range(nTrain):
#     for j in range(0,M):
#       phiTrain[i, j] = Xtrain[i]**(M-j)

#   w = np.linalg.pinv(phiTrain).dot(tTrain)

#   x0 = np.linspace(0,1)

#   y = 0
#   m = M
#   for i in range(len(w)):
#       y+= w[i]*x0**m
#       m = m-1


#   # plt.plot(x0,y)
#   # plt.show()

#   phiTest = np.ones((nTest, M+1))

#   for i in range(nTest):
#     for j in range(0,M):
#       phiTest[i, j] = Xtest[i]**(M-j)

#   # print(phiTest)

#   trainError.append(np.sqrt((np.linalg.norm(phiTrain.dot(w) - tTrain)**2)/nTrain))
#   testError.append(np.sqrt((np.linalg.norm(phiTest.dot(w) - tTest)**2)/nTest))

# # print(trainError)
# # print(testError)

# xError = np.linspace(0,9,10)
# plt.plot(xError, trainError, fillstyle='none', marker='o', c='blue', label='Training')
# plt.plot(xError, testError, fillstyle='none', marker='o', c='red', label='Test')
# plt.xlim(0,9)
# plt.ylim(0,1)
# plt.xlabel("M")
# plt.ylabel("$\mathregular{E_{RMS}}$")
# plt.legend(loc='upper left')
# plt.show()