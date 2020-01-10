import numpy as np
from NN import FNN


if __name__ == "__main__":
    myFNN = FNN(neuronsInput=2,layersHidden=1,neuronsHidden=2,neuronsOutput=1	,activationFunction="sigmoid",learningRate=0.1)
    print("---------------------------------------------------------")
    print("[0,0]: "+str(myFNN.predict([-1,-1])))
    print("[0,1]: "+str(myFNN.predict([-1,1])))
    print("[1,0]: "+str(myFNN.predict([1,-1])))
    print("[1,1]: "+str(myFNN.predict([1,1])))
    print()
    data = []
    print(myFNN.train([[0,1],[1]]))
    print()
    print("[0,0]: "+str(myFNN.predict([-1,-1])))
    print("[0,1]: "+str(myFNN.predict([-1,1])))
    print("[1,0]: "+str(myFNN.predict([1,-1])))
    print("[1,1]: "+str(myFNN.predict([1,1])))
    print("---------------------------------------------------------")