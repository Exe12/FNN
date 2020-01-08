import numpy as np
#Activation: RELU
class FNN:
    def __init__(self,neuronsInput,layersHidden,neuronsHidden,neuronsOutput,activationFunction):
        #Set object vars
        self.nIS = neuronsInput
        self.lHS = layersHidden
        self.nHS = neuronsHidden
        self.nOS = neuronsOutput
        self.tNL = 1+self.lHS+1                                         #Number of total layers
        self.vWS = self.tNL-1                                           #Number of weight matrixs
        self.bias = np.random.rand(self.vWS)
        self.actFunc = _sigmoid() if activationFunction == "sigmoid" else error
        #Initialise input and output layers
        self.layerInput = np.zeros(shape=(self.nIS,1))                  #Input Layer array of neurons with zeros
        self.layerOutput = np.zeros(shape=(self.nOS,1))                 #Output Layer array of neurons with zeros
        self.layerHidden = []                                           #Array for Hidden Layers array of neurons
        self.matrixWeights = []                                         #Array for Weight Matrixs of weight values
        for i in range(self.lHS):                                       #Fill the Array for Hidden Layers
            self.layerHidden.append(np.zeros(shape=(self.nHS,1)))       #Create Hidden Layer with zeros
        self.matrixWeights.append(np.random.rand(self.nIS,self.nHS))    #Create and add Weight Matrix for Input-->Hidden_1
        for i in range(self.vWS-2):                                     #Fill the Array for Weight Matrixs
            self.matrixWeights.append(np.random.rand(self.nHS,self.nHS))#Create and add Weight Matrix for Hidden_1-->...-->Hidden_n
        self.matrixWeights.append(np.random.rand(self.nHS,self.nOS))    #Create and add Weight Matrix for Hidden_n-->Output
        
    def _sigmoid(self,x):                                               #Sigmoid - activation function
        return 1/(1+np.exp(-x))                                         #__Applay sigmoid function to x and return it: sigmoid(x) = 1/1(1+exp(-x))

    def _calcNextLayer(self,layer1,weights,bias):                #Calculate next Layer with layer bevor and weights and bias
        self.layer2 = np.matmul(weights,layer1)+bias                    #__Calculate next layer: Layer2_vector = (Weights_matrix * Layer1_vector) + bias
        for i in range(len(self.layer2)):                               #__Foreach element in Layer2_vector
            self.layer2[i][0] = self._sigmoid(self.layer2[i][0])        #____Apply activation function
        return self.layer2                                              #__Return new Layer2_vector

    def predict(self,inputNeurons):                                     #Predict Output from Input through FNN
        for j in range(len(inputNeurons)):
            self.layerInput[j][0] = inputNeurons[j]
        self.allLayers = []                                             #__Create array for all layers - collection
        self.allLayers.append(self.layerInput)                          #__Add InputLayer to collection
        for x in self.layerHidden:                                      #__Foreach element in Hidden Layers (HiddenLayer_1,...,HiddenLayer_n) 
            self.allLayers.append(x)                                    #__Add each HiddenLayer to collection
        self.allLayers.append(self.layerOutput)                         #__Add OutputLayer to collection
        for i in range(self.tNL-1):                                     #__Iterate over all Layers(collection)
            self.allLayers[i+1] = self.actFunc(self.allLayers[i],self.matrixWeights[i],self.bias[i])#__Calculate next Layer and set it in the collection (all Layers)
        return self.allLayers[self.tNL-1]                               #__Return OutputLayer (last layer of collection)











