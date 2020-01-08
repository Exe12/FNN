import numpy as np
from FNN import FNN


if __name__ == "__main__":
    myFNN = FNN(neuronsInput=4,layersHidden=1,neuronsHidden=4,neuronsOutput=4,activationFunction="sigmoid",learningRate=0.2)
    print("---------------------------------------------------------")
    print("Test Predict")
    #print(myFNN.predict([1,0,0,0,0,0,0,0,0]))
    print("Input [1,0]: ")
    print(myFNN.predict([1,0,1,1]))
    print("Gradients [1,1,0,1]-->1: ")
    print(myFNN.train([1,0,1,1],[1,1,0,1]))
    print("---------------------------------------------------------")