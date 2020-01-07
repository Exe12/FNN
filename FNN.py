import numpy as np

class FNN:
    def __init__(self,neuronsInput,layersHidden,neuronsHidden,neuronsOutput,learningRate):
        #Set object vars
        self.nIS = neuronsInput
        self.lHS = layersHidden
        self.nHS = neuronsHidden
        self.nOS = neuronsOutput
        self.lR = learningRate
        self.vWS = 1+self.lHS+1-1
        self.bias = np.random.rand(self.vWS)
        #Initialise input and output layers
        self.layerInput = np.zeros(shape=(self.nIS,1))                  #Input Layer array of neurons with zeros
        self.layerOutput = np.zeros(shape=(self.nOS,1))                 #Output Layer array of neurons with zeros
        self.layerHidden = []                                           #Array for Hidden Layers array of neurons
        self.valuesWeights = []                                         #Array for Weight Matrixs of weight values
        for i in range(self.lHS):                                       #Fill the Array for Hidden Layers
            self.layerHidden.append(np.zeros(shape=(self.nHS,1)))       #Create Hidden Layer with zeros
        self.valuesWeights.append(np.random.rand(self.nIS,self.nHS))    #Create and add Weight Matrix for Input-->Hidden_1
        for i in range(self.vWS-2):                                     #Fill the Array for Weight Matrixs
            self.valuesWeights.append(np.random.rand(self.nHS,self.nHS))#Create and add Weight Matrix for Hidden_1-->...-->Hidden_n
        self.valuesWeights.append(np.random.rand(self.nHS,self.nOS))    #Create and add Weight Matrix for Hidden_n-->Output
        


    def calcNextLayerNeurons(self,layer1,weights,layer2,bias):
        pass



    #Test, dump and debugging functions
    def dump_layers(self):
        print("InL-->OutL")
        spacer1 = ""
        spacer2 = ""
        count = 0
        exception = 1
        control = ""
        for x,y in zip(str(self.layerInput),str(self.layerInput)):
            control+=x
            if x == "\n":
                continue
            if exception == 1:
                exception = 0
                continue
            if count == 3:
                count=0
                print(spacer1+"-->"+spacer2)
                spacer1 = ""
                spacer2 = ""
            else:
                spacer1+=x
                spacer2+=y
                count+=1
    def dump_vars(self):
        dump = str(vars(self)).split(", '")
        for x in dump:
            print(x.replace("{","").replace("}","").replace("'",""))
    def testClass(self):
        print("Class seems to work...")












