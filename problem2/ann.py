import sys
import copy
import numpy as np
from neuralNetwork import *

# example: python ann ../testData/spam/spam 1 10 .01 10000
def parseCommandLineToNeuralNetwork():
    # sys.argv[0] is the name of the script so we need 6 args for 5 options.
    if (len(sys.argv) is not 6):
        raise ValueError('You must run with 5 options.')
    dataPath = sys.argv[1]
    useCrossValidation = int(sys.argv[2]) == 0
    numberOfHiddenNodes = int(sys.argv[3])
    weightDecayCoeff = float(sys.argv[4])
    numberOfTrainingIterations = int(sys.argv[5])
    if numberOfTrainingIterations == 0:  # If the arg is 0, we want to run until convergence
        numberOfTrainingIterations = -1  # But it's more convenient to represent this as a -1 internally
    return NeuralNetwork(dataPath, useCrossValidation, numberOfHiddenNodes, weightDecayCoeff, numberOfTrainingIterations)

# MAIN
np.random.seed(12345)
neuralNetwork = parseCommandLineToNeuralNetwork()
