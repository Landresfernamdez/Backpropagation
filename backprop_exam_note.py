import numpy as np
import os
# X = (hours sleeping, hours studying), y = score on test
trainingInputs = []
trainingOutputs = []

class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 6000
        self.outputSize = 4
        self.hiddenSize = 3
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))
    
    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights


    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        output = self.sigmoid(self.z3) # final activation function
        return output
    
    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)


def cargarTrainingSet():
    lista = os.listdir(os.path.dirname(
        os.path.abspath(__file__)) + "\\inputs\\")
    for folderfilenames in lista:
        listaFolders=os.listdir(os.path.dirname(
        os.path.abspath(__file__)) + "\\inputs\\"+folderfilenames)
        for filename in listaFolders:
            currentPath = os.path.dirname(os.path.abspath(__file__)) + "\\inputs\\"+folderfilenames+"\\"
            index = listaFolders.index(filename)
            outputNN = []
            inputNN = []
            trainsetElement = []
            outputNN.append(int(folderfilenames))
            with open(currentPath + filename, "r") as f:
                linea = f.read()
                for elemento in linea:
                    if(elemento == '1' or elemento == '0'):
                        binary = int(elemento)
                        inputNN.append(binary)
            trainsetElement.append(inputNN)
            trainingOutputs.append(outputNN)
            trainingInputs.append(trainsetElement)

cargarTrainingSet()
NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
  print("Input: \n" + str(trainingInputs))
  print ("Actual Output: \n" + str(trainingOutputs)) 
  print ("Predicted Output: \n" + str(NN.forward(trainingInputs))) 
  print ("Loss: \n" + str(np.mean(np.square(trainingOutputs - NN.forward(trainingInputs))))) # mean sum squared loss
  print ("\n")
  NN.train(trainingInputs, trainingOutputs)
cargarTrainingSet()


