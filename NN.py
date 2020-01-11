import math
from scipy import special
import numpy as np

class FNN:

	def __init__(self,neuronsInput,layersHidden,neuronsHidden,neuronsOutput,activationFunction,learningRate):
		self.nIS = neuronsInput
		self.lHS = layersHidden
		self.nHS = neuronsHidden
		self.nOS = neuronsOutput
		self.actFuncName = activationFunction
		self.lr = learningRate
		self.weights = [np.random.normal(0.0, self.nHS**0.5, (self.nHS, self.nIS))]
		self.weights.extend([np.random.normal(0.0, self.nHS**0.5, (self.nHS, self.nHS)) for x in range(self.lHS-1)])
		self.weights.append(np.random.normal(0.0, self.nHS**0.5, (self.nOS, self.nHS)))
		self.bias = [np.random.normal(0.0, self.nIS**0.5, (self.nIS, 1))]
		self.bias.extend([np.random.normal(0.0, self.nHS**0.5, (self.nHS, 1)) for x in range(self.lHS)])
		self.bias.append(np.random.normal(0.0, self.nOS**0.5, (self.nOS, 1)))
		self.layerMemory = []

	def sigmoid(self, x):
		return special.expit(x)#return 1 / (1+math.exp(-x))

	def dsigmoid(self,x):
		return x*(1.0-x)

	def predict(self,inputNeurons):
		self.layerMemory = []
		self.layer = np.array(inputNeurons, ndmin=2).T
		self.layerMemory.append(self.layer)
		self.actFunc = np.vectorize(self.sigmoid)
		for i in range(self.lHS+1):
			self.layer = self.actFunc(np.dot(self.weights[i], self.layer)+self.bias[i+1])
			self.layerMemory.append(self.layer)
		return self.layer

	def train(self,inputAndExpected):
		self.inputExpected = np.array(inputAndExpected[1], ndmin=2).T
		self.prediction = self.predict(inputAndExpected[0])
		self.layerError = [self.inputExpected - self.prediction]
		self.dactFunc = np.vectorize(self.dsigmoid)
		for i in range(self.lHS+1):
			self.bias[len(self.bias)-1-i] += self.lr * self.layerError[0]*self.dactFunc(self.layerMemory[len(self.layerMemory)-1-i])
			self.weights[len(self.weights)-1-i] += np.dot(self.lr * self.layerError[0]*self.dactFunc(self.layerMemory[len(self.layerMemory)-1-i]), self.layerMemory[len(self.layerMemory)-2-i].T)
			self.newError = np.dot(self.weights[len(self.weights)-1-i].T, self.layerError[0])
			self.spacer = []
			self.spacer.append(self.newError)
			self.spacer.append(self.layerError)
			self.layerError = self.spacer

	def trainer(self):
		pass

	def exportWeights(self,path):
		np.savez_compressed(path,weights=self.weights)
	def importWeights(self,path):
		self.weights = np.load(path,allow_pickle=True)["weights"]
	def exportBias(self,path):
		np.savez_compressed(path,bias=self.bias)
	def importBias(self,path):
		self.bias = np.load(path,allow_pickle=True)["bias"]
	def exportFNN(self,path):
		np.savez_compressed(path,weights=self.weights,bias=self.bias)
	def importFNN(self,path):
		self.data = np.load(path,allow_pickle=True)
		self.weights = self.data["weights"]
		self.bias = self.data["bias"]
			
