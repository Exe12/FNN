import numpy as np

class FNN:
    def __init__(self,neuronsInput,layersHidden,neuronsHidden,neuronsOutput,learningRate):
    	#Set object vars
    	self.nIS = neuronsInput
    	self.lHS = layersHidden
    	self.nHS = neuronsHidden
    	self.nOS = neuronsOutput
    	self.lR = learningRate
    	#Initialise input and output layers
    	self.layerInput = np.zeros(shape=(self.nIS,1))
    	self.layerOutput = np.zeros(shape=(self.nOS,1))
    	self.layerHidden = []
    	self.valuesWeights = []
    	for i in range(self.lHS):
    		self.layerHidden.append(np.zeros(shape=(self.nHS,1)))
    	for i in range(1+1+lHS-1):
    		






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












