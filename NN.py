import math
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
		self.layerMemory = []

	def sigmoid(self, x):
		return 1.0 / (1.0+math.exp(-x))
	def dsigmoid(self,x):
		return x*(1.0-x)
	def predict(self,inputNeurons):
		self.layerMemory = []
		self.layer = np.array(inputNeurons, ndmin=2).T
		self.layerMemory.append(self.layer)
		self.actFunc = np.vectorize(self.sigmoid)
		for i in range(self.lHS+1):
			self.layer = np.dot(self.weights[i], self.layer)
			self.layerMemory.append(self.actFunc(self.layer))
		return self.layer

	def train(self,inputAndExpected):
		self.inputExpected = np.array(inputAndExpected[1], ndmin=2).T
		self.prediction = self.predict(inputAndExpected[0])
		self.layerError = [self.inputExpected - self.prediction]
		self.dactFunc = np.vectorize(self.dsigmoid)
		for i in range(self.lHS+1):
			self.weights[len(self.weights)-1-i] += self.lr * np.dot((self.layerError[0]*self.dactFunc(self.layerMemory[len(self.layerMemory)-1-i])), self.layerMemory[len(self.layerMemory)-2-i].T)
			self.newError = [np.dot(self.weights[len(self.weights)-1-i].T, self.layerError[0])]
			print("######")
			print(self.newError)
			print()
			print(self.layerError)
			self.layerError = np.concatenate((self.newError, self.layerError))
			print()
			print(self.layerError)
			
