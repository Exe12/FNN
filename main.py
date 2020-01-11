import numpy as np
from NN import FNN


if __name__ == "__main__":
    myFNN = FNN(neuronsInput=2,layersHidden=1,neuronsHidden=100,neuronsOutput=1,activationFunction="sigmoid",learningRate=0.1)
    # 1.000.000 -> ~ 209Sek (3Min29Sek)
    # 100.000 -> ~ 20Sek
    # 10.000 -> ~ 3Sek
    '''
    print("---------------------------------------------------------")
    iterations = 100000
    print("TicTacToe - Iterations: "+str(iterations))
    print("---------------------------------------------------------")
    data = [[[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0]],[[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]],[[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0]],[[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0]],[[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0]]]
    print("1.Feld-->2.Feld: "+str(np.round(myFNN.predict(data[0][0]),3).T))
    print("2.Feld-->3.Feld: "+str(np.round(myFNN.predict(data[1][0]),3).T))
    print("3.Feld-->4.Feld: "+str(np.round(myFNN.predict(data[2][0]),3).T))
    print("4.Feld-->5.Feld: "+str(np.round(myFNN.predict(data[3][0]),3).T))
    print("5.Feld-->6.Feld: "+str(np.round(myFNN.predict(data[4][0]),3).T))
    print()
    for i in range(iterations):
        x = np.random.randint(0,5)
        myFNN.train(data[x])
    print()
    print("1.Feld-->2.Feld: "+str(np.round(myFNN.predict(data[0][0]),3).T))
    print("2.Feld-->3.Feld: "+str(np.round(myFNN.predict(data[1][0]),3).T))
    print("3.Feld-->4.Feld: "+str(np.round(myFNN.predict(data[2][0]),3).T))
    print("4.Feld-->5.Feld: "+str(np.round(myFNN.predict(data[3][0]),3).T))
    print("5.Feld-->6.Feld: "+str(np.round(myFNN.predict(data[4][0]),3).T))
    print("---------------------------------------------------------")
    '''
    #'''
    print("---------------------------------------------------------")
    iterations = 10000
    print("XOR - Iterations: "+str(iterations))
    print("---------------------------------------------------------")
    print("[0,0]: "+str(myFNN.predict([0,0])))
    print("[0,1]: "+str(myFNN.predict([0,1])))
    print("[1,0]: "+str(myFNN.predict([1,0])))
    print("[1,1]: "+str(myFNN.predict([1,1])))
    print()
    data = [[[0,0],[0]],[[0,1],[1]],[[1,0],[1]],[[1,1],[0]]]
    for i in range(iterations):
        x = np.random.randint(0,4)
        myFNN.train(data[x])
    print()
    print("[0,0]: "+str(myFNN.predict([0,0])))
    print("[0,1]: "+str(myFNN.predict([0,1])))
    print("[1,0]: "+str(myFNN.predict([1,0])))
    print("[1,1]: "+str(myFNN.predict([1,1])))
    print("---------------------------------------------------------")
    #'''
    