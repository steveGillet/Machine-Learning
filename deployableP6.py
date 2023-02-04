import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import SGD, RMSprop, Adam
from PIL import Image
import numpy as np
from numpy import array
import time
import os

# Starts timer of execution time
startTime = time.time()

# input for directory, list1 is list of files in directory
dir1 = input("Enter Directory Location:")
list1 = os.listdir(dir1) 
numFiles = len(list1)

# makes array for input
xTest = np.zeros((numFiles,101,101))

# goes index by index and turns the images into matrices, takes the directory name adds a slash and the names of the files taken from list1
for i in range (numFiles):
    xTest[i] = Image.open(dir1 + "/" + list1[i])

# converts to tensor
xTest = torch.tensor(xTest, dtype=torch.float32)

# reshapes input so that there is a dimension for channels                                   
xTest = xTest.reshape(-1,1,101,101)

# regularizing
xTest = xTest/255.0

classes = (0,1)

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
        )

    def forward(self, xBatch):
        preds = self.seq(xBatch)
        return preds

convNet = ConvNet()
convNet.load_state_dict(torch.load('conv_net_model2.ckpt'))

def MakePredictions(model, inputData, batchSize=32):
    batches = torch.arange((inputData.shape[0]//batchSize)+1)

    with torch.no_grad(): 
        preds = []
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch*batchSize), int(batch*batchSize+batchSize)
            else:
                start, end = int(batch*batchSize), None

            xBatch = inputData[start:end]

            preds.append(model(xBatch))

    return preds

testPreds = MakePredictions(convNet, xTest, batchSize=18) 

testPreds = torch.cat(testPreds) 

testPreds = testPreds.argmax(dim=1)

print("Results")
print("_________________________________")
fixedPreds = np.zeros(numFiles)
for i in range(numFiles):
    if(testPreds[i] == 0):
        fixedPreds[i] = 1

for i in range(numFiles):
    print("File {:2d}: {} | Prediction: {}".format(i+1,list1[i],fixedPreds[i]))


print("Number of Images with Worms    (1) : {:.3f}".format(list(fixedPreds).count(1)))
print("Number of Images without Worms (0) : {:.3f}".format(list(fixedPreds).count(0)))

params = convNet.parameters()

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

