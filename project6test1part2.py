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
from sklearn.metrics import multilabel_confusion_matrix, classification_report

startTime = time.time()

xTrain = np.zeros((10000,101,101))
xTest = np.zeros((1000,101,101))
yTrain = np.zeros((10000))
yTest = np.zeros((1000))
yTrain[0:5000] = 1
yTest[0:500] = 1

for i in range (5000):
    xTrain[i] = Image.open(r"C:\Users\steph\OneDrive\Desktop\School\Machine Learning\Celegans_ModelGen\0\image_{}.png".format(i+1))
    xTrain[i+5000] = Image.open(r"C:\Users\steph\OneDrive\Desktop\School\Machine Learning\Celegans_ModelGen\1\image_{}.png".format(i+1))
for i in range (500):
    xTest[i] = Image.open(r"C:\Users\steph\OneDrive\Desktop\School\Machine Learning\Celegans_ModelGen\0\image_{}.png".format(i+5001))
    xTest[i+500] = Image.open(r"C:\Users\steph\OneDrive\Desktop\School\Machine Learning\Celegans_ModelGen\1\image_{}.png".format(i+5001))

temp1 = list(zip(xTrain, yTrain))
random.shuffle(temp1)
xTrain, yTrain = zip(*temp1)
xTrain, yTrain = list(xTrain), list(yTrain)
xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

temp2 = list(zip(xTest, yTest))
random.shuffle(temp2)
xTest, yTest = zip(*temp2)
xTest, yTest = list(xTest), list(yTest)
xTest = np.array(xTest)
yTest = np.array(yTest)

xTrain, xTest, yTrain, yTest = torch.tensor(xTrain, dtype=torch.float32),\
                                   torch.tensor(xTest, dtype=torch.float32),\
                                   torch.tensor(yTrain, dtype=torch.long),\
                                   torch.tensor(yTest, dtype=torch.long)

xTrain, xTest = xTrain.reshape(-1,1,101,101), xTest.reshape(-1,1,101,101)

xTrain, xTest = xTrain/255.0, xTest/255.0


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

train_preds = MakePredictions(conv_net, xTrain, batch_size=18) ## Make Predictions on train dataset

train_preds = torch.cat(train_preds)

train_preds = train_preds.argmax(dim=1)

print(test_preds[494:499], train_preds[:5])
print(yTest[494:499], yTrain[:5])


print("Train Accuracy : {:.3f}".format(accuracy_score(yTrain, train_preds)))
print("Test  Accuracy : {:.3f}".format(accuracy_score(yTest, test_preds)))

cf_matrix = multilabel_confusion_matrix(yTest, test_preds, labels=[False, True])
print(cf_matrix)
print(classification_report(yTest,test_preds))

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))