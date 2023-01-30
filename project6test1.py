import torch
import torchvision
from tensorflow import keras
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import SGD, RMSprop, Adam
from PIL import Image
import numpy as np
from numpy import array
import random
from matplotlib import pyplot as plt
import time
import os

startTime = time.time()

list1 = os.listdir("Celegans_ModelGen/0") # dir is your directory path
numFiles = len(list1)

xTrain = np.zeros((10000,101,101))
yTrain = np.zeros((10000))
yTrain[0:5000] = 1

for i in range (5000):
    xTrain[i] = Image.open(r"C:\Users\steph\OneDrive\Desktop\School\Machine Learning\Celegans_ModelGen\0\image_{}.png".format(i+1))
    xTrain[i+5000] = Image.open(r"C:\Users\steph\OneDrive\Desktop\School\Machine Learning\Celegans_ModelGen\1\image_{}.png".format(i+1))

temp = list(zip(xTrain, yTrain))
random.shuffle(temp)
xTrain, yTrain = zip(*temp)
xTrain, yTrain = list(xTrain), list(yTrain)
xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

# print(yTrain[0])
# plt.imshow(xTrain[0], interpolation='nearest')
# plt.show()

xTrain, yTrain = torch.tensor(xTrain,dtype=torch.float32), \
                 torch.tensor(yTrain,dtype=torch.long)

xTrain = xTrain.reshape(-1,1,101,101)
xTrain = xTrain/255.0

classes = yTrain.unique()

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(11,11), padding="valid"),
            nn.ReLU(),

            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(11,11), padding="valid"),
            nn.ReLU(),

            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(11,11), padding="valid"),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(24*71*71, len(classes)),
            #nn.Softmax(dim=1)            
        )

    def forward(self, x_batch):
        preds = self.seq(x_batch)
        return preds
conv_net = ConvNet()

def TrainModelInBatches(model, loss_func, optimizer, X, Y, batch_size=32, epochs=5):
    for i in range(epochs):
        batches = torch.arange((X.shape[0]//batch_size)+1) ### Batch Indices

        losses = [] ## Record loss of each batch
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
            else:
                start, end = int(batch*batch_size), None

            X_batch, Y_batch = X[start:end], Y[start:end] ## Single batch of data

            preds = model(X_batch) ## Make Predictions by forward pass through network

            loss = loss_func(preds, Y_batch) ## Calculate Loss
            losses.append(loss) ## Record Loss

            optimizer.zero_grad() ## Zero weights before calculating gradients
            loss.backward() ## Calculate Gradients
            optimizer.step() ## Update Weights

        print("Categorical Cross Entropy : {:.3f}".format(torch.tensor(losses).mean()))

loss = nn.CrossEntropyLoss()

torch.manual_seed(42) ##For reproducibility.This will make sure that same random weights are initialized each time.

epochs = 25
learning_rate = torch.tensor(1/1e3) # 0.001
batch_size=18

cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = SGD(params=conv_net.parameters(), lr=learning_rate)

TrainModelInBatches(conv_net,
                    cross_entropy_loss,
                    optimizer,
                    xTrain, yTrain,
                    batch_size=batch_size,
                    epochs=epochs)

# conv_net_file = open('conv_net.pkl', 'wb') 
# pickle.dump(conv_net, conv_net_file)
# conv_net_file.close()

torch.save(conv_net.state_dict(), 'conv_net_model2.ckpt')

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))