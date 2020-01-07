import numpy as np
from FNN import FNN


if __name__ == "__main__":
    myFNN = FNN(9,2,9,9,0.5)
    myFNN.testClass()
    myFNN.dump_vars()
    #myFNN.dump_layers()
    	