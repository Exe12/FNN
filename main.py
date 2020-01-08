import numpy as np
from FNN import FNN


if __name__ == "__main__":
    myFNN = FNN(neuronsInput=9,layersHidden=10,neuronsHidden=20,neuronsOutput=9,activationFunction="sigmoid")
    print("---------------------------------------------------------")
    print("Test Predict")
    print(myFNN.predict([1,0,0,0,0,0,0,0,0]))
    print(myFNN.predict([0,1,0,0,0,0,0,0,0]))
    print("---------------------------------------------------------")