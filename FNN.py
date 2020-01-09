import numpy as np
#Activation: RELU
class FNN:
    def __init__(self,neuronsInput,layersHidden,neuronsHidden,neuronsOutput,activationFunction,learningRate):
        #Set object vars
        self.nIS = neuronsInput
        self.lHS = layersHidden
        self.nHS = neuronsHidden
        self.nOS = neuronsOutput
        self.lr = learningRate
        self.tNL = 1+self.lHS+1                                         #Number of total layers
        self.vWS = self.tNL-1                                           #Number of weight matrixs
        self.bias = []
        for i in range(self.vWS-1):
            self.bias.append(np.random.rand(self.nHS,1))
        self.bias.append(np.random.rand(self.nOS,1))
        self.actFuncName = activationFunction
        self.aLLP = []                                                  #"All Layers Last Prediction" stores the layers and neurons from last prediction for the training (backpropagation)            
        #Initialise input and output layers
        self.layerInput = np.zeros(shape=(self.nIS,1))                  #Input Layer array of neurons with zeros
        self.layerOutput = np.zeros(shape=(self.nOS,1))                 #Output Layer array of neurons with zeros
        self.layerHidden = []                                           #Array for Hidden Layers array of neurons
        self.matrixWeights = []                                         #Array for Weight Matrixs of weight values
        for i in range(self.lHS):                                       #Fill the Array for Hidden Layers
            self.layerHidden.append(np.zeros(shape=(self.nHS,1)))       #Create Hidden Layer with zeros
        self.matrixWeights.append(np.random.rand(self.nHS,self.nIS))    #Create and add Weight Matrix for Input-->Hidden_1
        for i in range(self.vWS-2):                                     #Fill the Array for Weight Matrixs
            self.matrixWeights.append(np.random.rand(self.nHS,self.nHS))#Create and add Weight Matrix for Hidden_1-->...-->Hidden_n
        self.matrixWeights.append(np.random.rand(self.nOS,self.nHS))    #Create and add Weight Matrix for Hidden_n-->Output

    def _sigmoid(self,x):                                               #Sigmoid - activation function
        return 1/(1+np.exp(-x))                                         #__Applay sigmoid function to x and return it: sigmoid(x) = 1/1(1+exp(-x))

    def _dsigmoidPreSigmoid(self,x):
        return x*(1-x)                                                  #dsigmoid = sigmoid(x)*(1-sigmoid(x)), but sigmoid is already apllied on x

    def _actFunc(self,x):                                               #Define activation function
        if self.actFuncName == "sigmoid":
            return self._sigmoid(x)

    def _dActFuncPreActFunc(self,x):                           #Define derivative of activation function, but activation function is already apllied to x
        if self.actFuncName == "sigmoid":
            return self._dsigmoidPreSigmoid(x)

    def _calcNextLayer(self,layer1,weights,bias):                      #Calculate next Layer with layer bevor and weights and bias
        self.layer2 = np.matmul(weights,layer1)+bias                    #__Calculate next layer: Layer2_vector = (Weights_matrix * Layer1_vector) + bias
        for i in range(len(self.layer2)):                               #__Foreach element in Layer2_vector
            self.layer2[i][0] = self._actFunc(self.layer2[i][0])        #____Apply activation function
        return self.layer2                                              #__Return new Layer2_vector

    def predict(self,inputNeurons):                                     #Predict Output from Input through FNN
        for j in range(len(inputNeurons)):
            self.layerInput[j][0] = inputNeurons[j]
        self.allLayers = []                                             #__Create array for all layers - collection
        self.allLayers.append(self.layerInput)                          #__Add InputLayer to collection
        for x in self.layerHidden:                                      #__Foreach element in Hidden Layers (HiddenLayer_1,...,HiddenLayer_n) 
            self.allLayers.append(x)                                    #____Add each HiddenLayer to collection
        self.allLayers.append(self.layerOutput)                         #__Add OutputLayer to collection
        for i in range(self.tNL-1):                                     #__Iterate over all Layers(collection)
            self.allLayers[i+1] = self._calcNextLayer(self.allLayers[i],self.matrixWeights[i],self.bias[i])#____Calculate next Layer and set it in the collection (all Layers)
        self.aLLP = self.allLayers                                      #__Spacer for the collection, needed for the training to get the last collection neurons for backpropagation
        return self.allLayers[self.tNL-1]                               #__Return OutputLayer (last layer of collection)

    def _train_calcErrors(self,expectedResult,prediction):              #Calculate all error_vectors foreach layer
        self.errors = np.zeros(shape=(len(expectedResult),1))           #__Create error_vector aka error_vector_output
        for i in range(len(self.errors)):                               #__Calculate error foreach prediction and add to error_vector
            self.errors[i][0] = expectedResult[i] - self.prediction[i]
        self.allLayersError = []                                        #__Create array (aka error_collection) for error_vectors foreach Layer, so every Layer has a error_vector
        self.allLayersError.append(np.zeros(shape=(self.nIS,1)))
        for i in range(self.lHS):
            self.allLayersError.append(np.zeros(shape=(self.nHS,1)))
        self.allLayersError.append(np.zeros(shape=(self.nOS,1)))
        self.allLayersError[len(self.allLayersError)-1] = self.errors   #__Set last error-layer as calculated error_vector (output-error)
        for i in range(self.tNL-1):                                     #__Calcluate each error_vector from last layer to first (output-->hidden-->input)
            x = self.tNL-1 - i                                          #____Because from back to front
            self.allLayersError[x-1] = np.matmul(self.matrixWeights[x-1].T, self.allLayersError[x]) #.T
        return self.allLayersError

    def _mapMatrixFunc(self,matrix,function):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] = function(matrix[i][j])
        return matrix

    def _vectorMultiplication(self,x,y):
        self.newVector = []
        for self.x,self.y in zip(x,y):
            self.newVector.append(self.x[0]*self.y[0])
        print("####")
        print(x)
        print("")
        print(y)
        print("")
        print(self.newVector)
        return self.newVector

    def _train_calcWeightGradientsAndBias(self,allLayers,allLayersError):      #Calculate all Weight Gradient Matrixs foreach Weight Matrix
        self.matrixWeightsGradients = []                                       #__Create Array for the Weight Gradients Matrixs
        for i in range(self.vWS):
            '''
            print("####")
            print(allLayersError[i+1])
            print("")
            print(self._mapMatrixFunc(allLayers[i+1], self._dActFuncPreActFunc))
            print("")
            print((allLayersError[i+1] * self._mapMatrixFunc(allLayers[i+1], self._dActFuncPreActFunc)))
            '''
            self.newBias = self.lr * (allLayersError[i+1] * self._mapMatrixFunc(allLayers[i+1], self._dActFuncPreActFunc))
            #self.newBias = np.matmul(self.lr * allLayersError[i+1], self._mapMatrixFunc(allLayers[i+1],self._dActFuncPreActFunc))
            self.matrixWeightsGradients.append(self.lr * np.dot((allLayersError[i+1] * self._mapMatrixFunc(allLayers[i+1], self._dActFuncPreActFunc)), allLayers[i].T))#np.matmul(self.newBias, allLayers[i].T))#self.newBias * allLayers[i].T)
            self.bias[i] = np.add(self.bias[i],self.newBias)#self.bias[i] + self.newBias
        return self.matrixWeightsGradients       


    def train(self,input):
        self.inputData = input[0]
        self.expectedResult = input[1]
        self.prediction = self.predict(self.inputData)                       #__Calculate prediction
        self.allLayers = self.aLLP                                           #__Load spacer for collection (all neurons from last prediction) to get the neurons/layers 
        self.allLayersError = self._train_calcErrors(self.expectedResult,self.prediction)
        self.matrixWeightsGradients = self._train_calcWeightGradientsAndBias(self.allLayers,self.allLayersError)
        for i in range(self.vWS):
            self.matrixWeights[i] += self.matrixWeightsGradients[i]#self.matrixWeights[i] + self.matrixWeightsGradients[i]

   










