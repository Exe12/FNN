import numpy as np

class FNN:
    def __init__(self,neuronsInput,neuronsHidden,neuronsOutput,learningRate):
    	#Set object vars
    	self.nI = neuronsInput
    	self.nH = neuronsHidden
    	self.nO = neuronsOutput
    	self.lR = learningRate
    	#Initialise input and output layers
    	self.layerInput = np.array(np.arange(self.nI), ndmin=2).T
    	self.layerOutput = np.array(np.arange(self.nO), ndmin=2).T





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












