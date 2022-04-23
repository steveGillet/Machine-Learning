import matplotlib.pyplot as plt
import numpy as np
import time

def sigmoid(x): #Defining the sigmoid function for computation
    return 1.0/(1.0 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetworkXOR: #Class to be called for initializing the Neural Network

    def __init__(self, nInput, nHiddenLayer, nOutputLayer):
        self.alpha = 0.5 #Learning Rate of the network

        self.nInput = nInput #Number of Neurons in the input layer
        self.nHiddenLayer = nHiddenLayer #Number of Neurons in the Hidden Layer
        self.nOutputLayer = nOutputLayer #Number of Neurons in the Output Layer
        
        #Initializing weight matrices to serve as connections between Input and Hidden layer
        self.hiddenWeights = np.random.random((self.nHiddenLayer, self.nInput+1)) 
        print(self.hiddenWeights)
        #Initializing weight matrices to serve as connections between Hidden layer and output neuron
        self.outputWeights = np.random.random((self.nOutputLayer, self.nHiddenLayer+1))
        print(self.outputWeights)
       
        #Serves as a container, for matrix multiplication of weights and input values
        self.hiddenLayerActivation = np.zeros((self.nHiddenLayer, 1), dtype=float) 
        #Servers as a container, for matrix multiplication of output weights and resultant output from the hidden layer
        self.outputLayerActivation = np.zeros((self.nOutputLayer, 1), dtype=float) 
        #Container of input values to be initialized
        self.initialOutput = np.zeros((self.nInput+1, 1), dtype=float)
        #Container of hidden layer values after multiplication and sigmoid activation
        self.hiddenLayerOutput = np.zeros((self.nHiddenLayer+1, 1), dtype=float)
        #Container of Final output following forward propagation and final application of Sigmoid, for output.
        self.outputLayerOutput = np.zeros((self.nOutputLayer, 1),  dtype=float) 

        #Error propgated from hidden layer to weights between input and hidden layer
        self.hiddenLayerDelta = np.zeros(self.nHiddenLayer, dtype=float) 
        #Error propagated from output from final layer to the weights between hidden layer and output layer
        self.outputLayerDelta = np.zeros(self.nOutputLayer, dtype=float)

    def forward(self, input):
        self.initialOutput[:-1, 0] = input #Initializing all but the topmost part of the hidden layer as Input
        self.initialOutput[-1:, 0] = 1.0 #Initializing bias neuron value

        #Matrix computation where weights are multiplied by the weights between input and hidden layer
        self.hiddenLayerActivation = np.dot(self.hiddenWeights, self.initialOutput) 
         #Applying sigmoid to the the matrix computation
        self.hiddenLayerOutput[:-1, :] = sigmoid(self.hiddenLayerActivation)
        #Initializing the first value of the hidden neurons to 1
        self.hiddenLayerOutput[-1:, :] = 1.0 
        
        #Matrix multiplication of hidden layer computation and the weights b/w hidden layer and output neuron
        self.outputLayerActivation = np.dot(self.outputWeights, self.hiddenLayerOutput)
         #Applying sigmoid activation to output neuron 
        self.outputLayerOutput = sigmoid(self.outputLayerActivation)

    """Backward Propagation"""
    def backward(self, teach):
         #Calculation of error in output of Forward Propagation and the expected output
        self.error = self.outputLayerOutput - np.array(teach, dtype=float)
         #Computation of error between output and hidden layer, using formula z * g(ij)
        self.outputLayerDelta = (1 - sigmoid(self.outputLayerActivation)) * sigmoid(self.outputLayerActivation) * self.error      
        
        
        #Computation of error done simultaneously between hidden output and input weights
        smalldelta_output = np.dot(self.outputWeights[:, :-1].transpose(), self.outputLayerDelta)
        self.hiddenLayerDelta = (1 - sigmoid(self.outputLayerActivation)) * sigmoid(self.hiddenLayerActivation) *  smalldelta_output
        
        #Updating weights between hidden layer and input layer
        self.hiddenWeights -= self.alpha * np.dot(self.hiddenLayerDelta, self.initialOutput.transpose()) 
        #Updating weights between hidden layer and output layer
        self.outputWeights -= self.alpha * np.dot(self.outputLayerDelta, self.hiddenLayerOutput.transpose())
#Defining output function
    def getOutput(self): 
        return (self.outputLayerOutput+1)/2

if __name__ == '__main__': 

    xorSet = [[-1, -1], [-1, 1], [1, 1], [1, -1]] #Input for training
    xorTeach = [[-1], [1], [-1], [1]] #Corresponding output values to be taught

    nn = NeuralNetworkXOR(2, 2, 1) #Calling the class

    count = 0
    while True:
        rnd = np.random.randint(0, 4) #Generating random number between 0 and 4

        nn.forward(xorSet[rnd]) #Random training data picked up over the given cases

        nn.backward(xorTeach[rnd]) #Propagating backwards on expected outputs
        print (count, xorSet[rnd], nn.getOutput()[0], nn.outputWeights)
        #Parameters for output determination
        count+= 1
        if nn.getOutput()[0] > 0.8: 
            print ('TRUE')
        elif nn.getOutput()[0] < 0.2:
            print ('FALSE')
        if(count > 3000):
            break

    x = np.outer(np.linspace(-2, 2, 10), np.ones(10))
    y = x.copy().T
    z = np.cos(x ** 2 + y ** 3)
    
    print("x: ", x)
    print("y: ", y)
    print("z: ", z)

    fig = plt.figure()
 
    # syntax for 3-D plotting
    ax = plt.axes(projection ='3d')
 
    # syntax for plotting
    ax.plot_surface(x, y, z, cmap ='viridis', edgecolor ='green')
    ax.set_title('Surface plot geeks for geeks')
    plt.show()    

    X = np.outer(np.linspace(-2, 2, 10), np.ones(10))
    Y = x.copy().T
    Z = np.ones((10,10))
    for i in range(10):
        for j in range(10):
            nn.forward([X[i][j],Y[i][j]])
            Z[i][j] = nn.getOutput()

    print(X)
    print(Y)
    print(Z)

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
 
    ax.plot_surface(X, Y, Z, cmap ='viridis', edgecolor ='green')

    plt.show()