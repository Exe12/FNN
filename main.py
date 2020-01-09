import numpy as np
from FNN import FNN


if __name__ == "__main__":
    myFNN = FNN(neuronsInput=2,layersHidden=1,neuronsHidden=2,neuronsOutput=1,activationFunction="sigmoid",learningRate=0.1)
    print("---------------------------------------------------------")
    #print("[0,0]: "+str(myFNN.predict([0,0])))
    #print("[0,1]: "+str(myFNN.predict([0,1])))
    print("[1,0]: "+str(myFNN.predict([1,0])))
    #print("[1,1]: "+str(myFNN.predict([1,1])))
    print("")
    data = [[[0,0],[0]],[[0,1],[1]],[[1,0],[1]],[[1,1],[0]]]
    print("Training....")
    for i in range(10000):
        test = data[np.random.randint(0,4)]
        myFNN.train(test)
    print("")
    #print("[0,0]: "+str(myFNN.predict([0,0])))
    #print("[0,1]: "+str(myFNN.predict([0,1])))
    print("[1,0]: "+str(myFNN.predict([1,0])))
    #print("[1,1]: "+str(myFNN.predict([1,1])))
    print("---------------------------------------------------------")