import sys
import copy
import numpy as np
from neuralNetwork import *
from continuousAttributeStandardizer import *
from exampleManager import *
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
    def __init__(self, dataPath, numberOfHiddenNodes, weightDecayCoeff, useCrossValidation):
        # Read data file
        fileName = os.path.basename(dataPath)
        rootDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath[0:-(len(fileName) + 1)])
        exampleSet = parse_c45(fileName, rootDirectory)
            
        # Construct nominal attribute hashes and continuousAttributeHash and count the total number of features
        self.nominalAttributeHashes, self.continuousAttributeHash, numUsefulFeatures = self.createAttributeHashes(exampleSet)

        # Normalize all examples  
        normalizedExamples = []
        for example in exampleSet.examples:
            normalizedExamples.append(NormalizedExample(example, exampleSet.schema, self.nominalAttributeHashes, self.continuousAttributeHash))

        #create example manager to handle constructing folds and delineating test and training examples
        self.useCrossValidation = useCrossValidation
        self.exampleManager = ExampleManager(normalizedExamples, useCrossValidation)

        # Construct the neural network
        numberOfOutputNodes = len(normalizedExamples[0].targets)
        layerSizesList = [numberOfHiddenNodes, numberOfOutputNodes] # Only a single hidden layer
        if layerSizesList[0] == 0: # If no hidden layer
            layerSizesList = [numberOfOutputNodes]

        self.neuralNetworks = []
        for i in range(0, self.exampleManager.numFolds()):
            self.neuralNetworks.append(NeuralNetwork(layerSizesList, numUsefulFeatures, weightDecayCoeff))

    #nominalAttributeHashes creates direct mapping of inputs to normalized values for each feature
    #continuousAttributeHash creates a continuousAttributeStandardizer for each feature
    def createAttributeHashes(self, exampleSet):
        numUsefulFeatures = 0
        nominalAttributeHashes = {}
        continuousAttributeHash = {}
        for i in range(0, len(exampleSet.schema.features)):
            nominalAttributeHash = None
            if exampleSet.schema.features[i].type == Feature.Type.NOMINAL or exampleSet.schema.features[
                i].type == Feature.Type.BINARY or exampleSet.schema.features[i].type == Feature.Type.CONTINUOUS:
                numUsefulFeatures = numUsefulFeatures + 1
            if exampleSet.schema.features[i].type == Feature.Type.NOMINAL:
                for value in exampleSet.schema[i].values:
                    if nominalAttributeHash is None:
                        nominalAttributeHash = {}
                        nominalAttributeHashes[exampleSet.schema.features[i]] = nominalAttributeHash
                    if value not in nominalAttributeHash:
                        nominalAttributeHash[value] = len(nominalAttributeHash.keys())
            elif exampleSet.schema.features[i].type == Feature.Type.CONTINUOUS:
                continuousAttributeHash[exampleSet.schema.features[i]] = ContinuousAttributeStandardizer(
                    exampleSet.examples, i)

        return nominalAttributeHashes, continuousAttributeHash, numUsefulFeatures

    def train(self, numIterations):
        if self.useCrossValidation:
            for i in range(0, self.exampleManager.numFolds()):
                trainingExamples, testingExamples = self.exampleManager.getCrossValidationExamples(i)
                self.trainNetwork(self.neuralNetworks[i], numIterations, trainingExamples, testingExamples)
        else:
            trainingExamples, testingExamples = self.exampleManager.getUnfoldedExamples()
            self.trainNetwork(self.neuralNetworks[0], numIterations, trainingExamples, testingExamples)

    def trainNetwork(self, neuralNetwork, numIterations, trainingExamples, testingExamples):
        pr = neuralNetwork.evaluatePerformance(testingExamples)
        print 'INITIAL: ' + 'Sum-Squared-Errors=' + str(pr.sumSquaredErrors) + '; Accuracy=' + str(pr.accuracy()) + '; Precision=' + str(pr.precision()) + '; Recall=' + str(pr.recall())
        for i in range(0, numIterations):
            neuralNetwork.executeTrainingIteration(trainingExamples)
            if (i+1) % 10 == 0:
                pr = neuralNetwork.evaluatePerformance(testingExamples)
                print 'AFTER ' + str(i+1) + ' TRAINING EPOCHS: ' + 'Sum-Squared-Errors=' + str(pr.sumSquaredErrors) + '; Accuracy=' + str(pr.accuracy()) + '; Precision=' + str(pr.precision()) + '; Recall=' + str(pr.recall())

        pr = neuralNetwork.evaluatePerformance(testingExamples)
        print 'FINAL: ' + 'Sum-Squared-Errors=' + str(pr.sumSquaredErrors) + '; Accuracy=' + str(pr.accuracy()) + '; Precision=' + str(pr.precision()) + '; Recall=' + str(pr.recall())
    
    
# MAIN
np.random.seed(12345)
dataPath, useCrossValidation, numberOfHiddenNodes, weightDecayCoeff, numberOfTrainingIterations = parseCommandLine()
neuralNetworkManager = NeuralNetworkManager(dataPath, numberOfHiddenNodes, weightDecayCoeff, useCrossValidation)
neuralNetworkManager.train(numberOfTrainingIterations)

