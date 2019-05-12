import os
import sys

training_sets = []
lista = os.listdir(os.path.dirname(os.path.abspath(__file__)) + "\\inputs\\")
for filename in lista:
    trainsetElement = []
    currentPath = os.path.dirname(os.path.abspath(__file__)) + "\\inputs\\"
    index = lista.index(filename)
    outputNN = []
    inputNN = []
    if(index % 2 == 0):
        outputNN.append(0)
    else:
        outputNN.append(1)
    with open(currentPath + filename, "r") as f:
        linea = f.read()
        for elemento in linea:
            if(elemento=='1' or elemento=='0'):
                binary=int(elemento)
                inputNN.append(binary)
    trainsetElement.append(inputNN)
    trainsetElement.append(outputNN)
    training_sets.append(trainsetElement)
    print(trainsetElement)
