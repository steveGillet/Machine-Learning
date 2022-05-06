import torch
import torchvision
from tensorflow import keras
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import SGD, RMSprop, Adam
import pickle
from PIL import Image
import numpy as np
from numpy import array
import random
from sklearn.metrics import accuracy_score
import time
import os

startTime = time.time()

dir1 = input("Enter Directory Location:")
list1 = os.listdir(dir1) # dir is your directory path
numFiles = len(list1)

xTest = np.zeros((numFiles,101,101))

for i in range (numFiles):
    xTest[i] = Image.open(dir1 + "/" + list1[i])

xTest = torch.tensor(xTest, dtype=torch.float32)
                                   
xTest = xTest.reshape(-1,1,101,101)

xTest = xTest/255.0

classes = (0,1)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(11,11), padding="valid", stride = 2),
            nn.ReLU(),

            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(11,11), padding="valid", stride = 2),
            nn.ReLU(),

            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(11,11), padding="valid", stride = 2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(24*4*4, len(classes)),
            #nn.Softmax(dim=1)            
        )

    def forward(self, x_batch):
        preds = self.seq(x_batch)
        return preds

conv_net = ConvNet()
conv_net.load_state_dict(torch.load('conv_net_model2.ckpt'))

def MakePredictions(model, input_data, batch_size=32):
    batches = torch.arange((input_data.shape[0]//batch_size)+1) ### Batch Indices

    with torch.no_grad(): ## Disables automatic gradients calculations
        preds = []
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
            else:
                start, end = int(batch*batch_size), None

            X_batch = input_data[start:end]

            preds.append(model(X_batch))

    return preds

test_preds = MakePredictions(conv_net, xTest, batch_size=18) ## Make Predictions on test dataset

test_preds = torch.cat(test_preds) ## Combine predictions of all batches

test_preds = test_preds.argmax(dim=1)

for i in range(numFiles):
    print("File 1: {}".format(list1[i]))
    print("Prediction: {}".format(test_preds[i]))


print("Number of Images with Worms    (1) : {:.3f}".format(list(test_preds).count(1)))
print("Number of Images without Worms (0) : {:.3f}".format(list(test_preds).count(0)))

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))