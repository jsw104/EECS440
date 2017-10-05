import sys
import copy
import numpy as np
from neuralNetwork import *
from continuousAttributeStandardizer import *
from mldata import *

# example: python ann ../testData/spam/spam 1 10 .01 10000
def parseCommandLine():
    # sys.argv[0] is the name of the script so we need 6 args for 5 options.
    if (len(sys.argv) is not 6):
        raise ValueError('You must run with 5 options.')
    
    dataPath = sys.argv[1]
    if type(dataPath) is not str:
        raise ValueError('The data path must be a string')
    
    useCrossValidation = int(sys.argv[2]) == 0
    numberOfHiddenNodes = int(sys.argv[3])
    weightDecayCoeff = float(sys.argv[4])
    numberOfTrainingIterations = int(sys.argv[5])
    
    if numberOfTrainingIterations == 0:  # If the arg is 0, we want to run until convergence
        numberOfTrainingIterations = -1  # But it's more convenient to represent this as a -1 internally
        
    return dataPath, useCrossValidation, numberOfHiddenNodes, weightDecayCoeff, numberOfTrainingIterations

class NormalizedExample:
    def __init__(self, example, schema, nominalAttributeHashes, continuousAttributeHash):
        inputsList = []
        targetsList = []
        for i in range(0, len(example)):
            if schema.features[i].type == Feature.Type.NOMINAL:
                nominalAttributeHash = nominalAttributeHashes[schema.features[i]]
                feature = example[i]
                if example[i] in nominalAttributeHash:
                    feature = nominalAttributeHash[feature]
                inputsList.append(feature) 
            elif (schema.features[i].type == Feature.Type.BINARY):
                inputsList.append(example[i])
            elif (schema.features[i].type == Feature.Type.CONTINUOUS):
                inputsList.append((continuousAttributeHash[schema.features[i]]).standardizeInput(example[i]))
            elif schema.features[i].type == Feature.Type.CLASS:
                targetsList.append(example[i])                    
        self.inputs = np.array(inputsList)
        self.targets = np.array(targetsList)


class NeuralNetworkManager:                        
    def __init__(self, dataPath, numberOfHiddenNodes, weightDecayCoeff):
        # Read data file
        fileName = os.path.basename(dataPath)
        rootDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath[0:-(len(fileName) + 1)])
        exampleSet = parse_c45(fileName, rootDirectory)
            
        # Construct nominal attribute hash and count the total number of features
        numUsefulFeatures = 0
        self.nominalAttributeHashes = {}
        self.continuousAttributeHash = {}
        for i in range(0,len(exampleSet.schema.features)):
            nominalAttributeHash = None
            if exampleSet.schema.features[i].type == Feature.Type.NOMINAL or exampleSet.schema.features[i].type == Feature.Type.BINARY or exampleSet.schema.features[i].type == Feature.Type.CONTINUOUS:
                numUsefulFeatures = numUsefulFeatures + 1
            if exampleSet.schema.features[i].type == Feature.Type.NOMINAL:
                for value in exampleSet.schema[i].values:
                    if nominalAttributeHash is None:
                        nominalAttributeHash = {}
                        self.nominalAttributeHashes[exampleSet.schema.features[i]] = nominalAttributeHash
                    if value not in nominalAttributeHash:
                        nominalAttributeHash[value] = len(nominalAttributeHash.keys())
            elif exampleSet.schema.features[i].type == Feature.Type.CONTINUOUS:
                self.continuousAttributeHash[exampleSet.schema.features[i]] = ContinuousAttributeStandardizer(exampleSet.examples, i)
         
        print str(numUsefulFeatures) + ' useful features' 
                              
        # Normalize all examples  
        self.normalizedExamples = []
        for example in exampleSet.examples:
            self.normalizedExamples.append(NormalizedExample(example, exampleSet.schema, self.nominalAttributeHashes, self.continuousAttributeHash))
        
        # Construct the neural network
        numberOfOutputNodes = len(self.normalizedExamples[0].targets)
        layerSizesList = [numberOfHiddenNodes, numberOfOutputNodes] # Only a single hidden layer
        if layerSizesList[0] == 0: # If no hidden layer
            layerSizesList = [numberOfOutputNodes]
        self.neuralNetwork = NeuralNetwork(layerSizesList, numUsefulFeatures, weightDecayCoeff)
        
    def trainNetwork(self, numIterations):
        sumSquaredErrors, numCorrect = self.evaluateNetworkPerformance(self.normalizedExamples)
        print 'INITIAL:'
        print 'SUM-SQUARED-ERRORS: ' + str(sumSquaredErrors) + '; NUM CORRECT: ' + str(numCorrect) + '/' + str(len(self.normalizedExamples))
        for i in range(0, numIterations):
            self.neuralNetwork.executeTrainingIteration(self.normalizedExamples)
            if (i+1) % 10 == 0:
                sumSquaredErrors, numCorrect = self.evaluateNetworkPerformance(self.normalizedExamples)
                print 'AFTER ' + str(i+1) + ' TRAINING ITERATIONS:'
                print 'SUM-SQUARED-ERRORS: ' + str(sumSquaredErrors) + '; NUM CORRECT: ' + str(numCorrect) + '/' + str(len(self.normalizedExamples))

        sumSquaredErrors, numCorrect = self.evaluateNetworkPerformance(self.normalizedExamples)
        print 'FINAL:'
        print 'SUM-SQUARED-ERRORS: ' + str(sumSquaredErrors) + '; NUM CORRECT: ' + str(numCorrect) + '/' + str(len(self.normalizedExamples))

    def evaluateExampleError(self, example):
        outputs = self.neuralNetwork.stimulateNetwork(example.inputs)
        rawErrors = outputs - example.targets                                       
        binaryErrors = np.absolute(np.rint(outputs) - example.targets) # 0 => Correct; 1 => Wrong
        return rawErrors, binaryErrors

    def evaluateNetworkPerformance(self, examples):
        numCorrect = 0
        sumSquaredErrors = 0
        for example in examples:
            rawErrors, binaryErrors = self.evaluateExampleError(example)
            sumSquaredErrors = sumSquaredErrors + 0.5 * np.sum(rawErrors*rawErrors)
            if(np.sum(binaryErrors) == 0):
                numCorrect = numCorrect + 1      
        return sumSquaredErrors, numCorrect
    
    
# MAIN
np.random.seed(12345)
dataPath, useCrossValidation, numberOfHiddenNodes, weightDecayCoeff, numberOfTrainingIterations = parseCommandLine()
neuralNetworkManager = NeuralNetworkManager(dataPath, numberOfHiddenNodes, weightDecayCoeff)
neuralNetworkManager.trainNetwork(numberOfTrainingIterations)

