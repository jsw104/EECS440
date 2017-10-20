import sys
import os
import random
from mldata import *
from bayesianNetwork import *
from exampleManager import *

# example: python nbayes ../testData/spam/spam 1 5 0
def parseCommandLine():
    # sys.argv[0] is the name of the script so we need 5 args for 4 options.
    if (len(sys.argv) is not 5):
        raise ValueError('You must run with 4 options.')

    dataPath = sys.argv[1]
    if type(dataPath) is not str:
        raise ValueError('The data path must be a string')

    useCrossValidation = int(sys.argv[2]) == 0
    numberOfBins = int(sys.argv[3])
    mEstimate = int(sys.argv[4])

    if numberOfBins < 2:
        raise ValueError('There must be at least 2 bins.')

    return dataPath, useCrossValidation, numberOfBins, mEstimate

def getExamplesFromDataPath(dataPath):
    # Read data file
    fileName = os.path.basename(dataPath)
    rootDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath[0:-(len(fileName) + 1)])
    exampleSet = parse_c45(fileName, rootDirectory)
    return exampleSet

random.seed(12345)
dataPath, useCrossValidation, numberOfBins, mEstimate = parseCommandLine()
exampleSet = getExamplesFromDataPath(dataPath)

accuracies = []
precisions = []
recalls = []

exampleManager = ExampleManager(exampleSet.examples, useCrossValidation)
if not useCrossValidation:
    trainingExamples, testExamples = exampleManager.getUnfoldedExamples()
    bayesianNetwork = BayesianNetwork(trainingExamples, exampleSet.schema, numberOfBins, mEstimate)
    bayesianNetwork.evaluateExamples(testExamples)

else:
    for i in range(0, exampleManager.numFolds()):
        trainingExamples, testExamples = exampleManager.getCrossValidationExamples(i)
        bayesianNetwork = BayesianNetwork(trainingExamples, exampleSet.schema, numberOfBins, mEstimate)
        bayesianNetwork.evaluateExamples(testExamples) 


