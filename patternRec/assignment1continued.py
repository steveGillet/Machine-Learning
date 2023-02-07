import pandas as pd
import numpy as np

dataFrame = pd.read_excel('Proj1DataSet.xlsx')
dataArray = dataFrame.to_numpy()
# print(dataArray)

virginica = dataArray[dataArray[:, 4] == 'virginica']
versicolor = dataArray[dataArray[:, 4] == 'versicolor']
setosa = dataArray[dataArray[:, 4] == 'setosa']

versivirgi = np.append(virginica, versicolor, 0)
print(versivirgi)
w = np.ones((1,5))
print(w)
wk = w
rho = 0.01
# print(np.sum(setosa[:,0]) + np.sum(-versivirgi[:,0]))
wk = wk + rho*(np.array([np.sum(-versivirgi[:,0])+ np.sum(setosa[:,0]), np.sum(-versivirgi[:,1])+ np.sum(setosa[:,1]), np.sum(-versivirgi[:,2])+ np.sum(setosa[:,2]),np.sum(-versivirgi[:,3]) + np.sum(setosa[:,3]), 1]))
wk = wk + rho*(np.array([np.sum(-versivirgi[:,0])+ np.sum(setosa[:,0]), np.sum(-versivirgi[:,1])+ np.sum(setosa[:,1]), np.sum(-versivirgi[:,2])+ np.sum(setosa[:,2]),np.sum(-versivirgi[:,3]) + np.sum(setosa[:,3]), 1]))
wk = wk + rho*(np.array([np.sum(-versivirgi[:,0])+ np.sum(setosa[:,0]), np.sum(-versivirgi[:,1])+ np.sum(setosa[:,1]), np.sum(-versivirgi[:,2])+ np.sum(setosa[:,2]),np.sum(-versivirgi[:,3]) + np.sum(setosa[:,3]), 1]))
print(wk)