import numpy as np
from NN import FNN


if __name__ == "__main__":
    myFNN = FNN(neuronsInput=2,layersHidden=1,neuronsHidden=50,neuronsOutput=1,activationFunction="sigmoid",learningRate=0.1)
    print("---------------------------------------------------------")
    print("[0,0]: "+str(myFNN.predict([1,1])))
    print("[0,1]: "+str(myFNN.predict([1,2])))
    print("[1,0]: "+str(myFNN.predict([2,1])))
    print("[1,1]: "+str(myFNN.predict([2,2])))
    print()
    data = [[[1,1],[0]],[[1,2],[1]],[[2,1],[1]],[[2,2],[0]]]
    for i in range(10000):
        x = np.random.randint(0,4)
        myFNN.train(data[x])
    print()
    print("[0,0]: "+str(myFNN.predict([1,1])))
    print("[0,1]: "+str(myFNN.predict([1,2])))
    print("[1,0]: "+str(myFNN.predict([2,1])))
    print("[1,1]: "+str(myFNN.predict([2,2])))
    print("---------------------------------------------------------")
    print()
    print()
    '''
    print("REAL")
    print("---------------------------------------------------------")
    print("[0,0]: "+str(myFNN.predict([0,0])))
    print("[0,1]: "+str(myFNN.predict([0,1])))
    print("[1,0]: "+str(myFNN.predict([1,0])))
    print("[1,1]: "+str(myFNN.predict([1,1])))
    print()
    
    data = [[[0,0],[0]],[[0,1],[1]],[[1,0],[1]],[[1,1],[0]]]
    for i in range(1):
        x = np.random.randint(0,4)
        myFNN.train(data[x])
    
    #myFNN.train([[0,1],[1]])
    print()
    print("[0,0]: "+str(myFNN.predict([0,0])))
    print("[0,1]: "+str(myFNN.predict([0,1])))
    print("[1,0]: "+str(myFNN.predict([1,0])))
    print("[1,1]: "+str(myFNN.predict([1,1])))
    print("---------------------------------------------------------")
    '''