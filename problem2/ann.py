import sys
import copy
import numpy as np
from neuralNetwork import *
from continuousAttributeStandardizer import *
from exampleManager import *
from mldata import *
from scipy.constants.codata import precision
from numpy.lib.function_base import append

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
    
    if numberOfTrainingIterations <= 0:  # If the arg is 0 or negative, we want to run until convergence
        numberOfTrainingIterations = -1  # But it's more convenient to represent this as a -1 internally
        
    return dataPath, useCrossValidation, numberOfHiddenNodes, weightDecayCoeff, numberOfTrainingIterations

def computePooledAROC(listPerformanceEvalResults):
    # WRT the first neuron in the output layer, if there is more than one
    allConfidences = []
    for pr in listPerformanceEvalResults:        
        for outputs in pr.outputs:
            allConfidences.append(outputs[0][0])
        
    rocPoints = (len(allConfidences)+2)*[None] #[(0.0,0.0,1.0),(1.0,1.0,0.0)]
    rocPoints[0] = (0.0,0.0,1.0)
    rocPoints[-1] = (1.0,1.0,0.0)
    appendIndex = 1
    for confidence in allConfidences:
        totalTP = 0.0
        totalFP = 0.0
        totalTN = 0.0
        totalFN = 0.0
        for pr in listPerformanceEvalResults:
            tp,tn,fp,fn = pr.itermediateStatistics(confidence)
            totalTP = totalTP + tp
            totalFP = totalFP + fp
            totalTN = totalTN + tn
            totalFN = totalFN + fn
        fpRate = 0.0 if totalFP + totalTN == 0 else totalFP/(totalFP + totalTN)
        tpRate = 0.0 if totalTP + totalFN == 0 else totalTP/(totalTP + totalFN)
        rocPoints[appendIndex] = (fpRate, tpRate, confidence)
        appendIndex = appendIndex + 1                      
    
    rocPoints.sort(key=lambda tup: tup[0])
    
    areaUnderROC = 0
    for i in range(0, len(rocPoints)-1):
        if rocPoints[i+1][0] == rocPoints[i][0]:
            continue
        else:
            trapezoidArea = 0.5 * (rocPoints[i][1] + rocPoints[i+1][1]) * (rocPoints[i+1][0] - rocPoints[i][0])
            areaUnderROC = areaUnderROC + trapezoidArea
            
    return areaUnderROC 

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

    def train(self, numIterations, debuggingOutput=True):
        if debuggingOutput:
            print 'useCrossValidation=' + str(self.useCrossValidation)
            
        prs = []
        if self.useCrossValidation:
            for i in range(0, self.exampleManager.numFolds()):
                if debuggingOutput:
                    print '------------------------ Training on Fold ' + str(i+1) + ' ------------------------'
                trainingExamples, testingExamples = self.exampleManager.getCrossValidationExamples(i)
                pr = self.trainNetwork(self.neuralNetworks[i], numIterations, trainingExamples, testingExamples, debuggingOutput=debuggingOutput)
                prs.append(pr)
        else:
            trainingExamples, testingExamples = self.exampleManager.getUnfoldedExamples()
            if debuggingOutput:
                print '------------------------ Training on Full Dataset ------------------------'
            pr = self.trainNetwork(self.neuralNetworks[0], numIterations, trainingExamples, testingExamples, debuggingOutput=debuggingOutput)
            prs.append(pr)
         
        if debuggingOutput:
            print 'Computing performance statistics...'
                   
        accuracies = []
        precisions = []
        recalls = []
        for pr in prs:
            accuracies.append(pr.accuracy())
            precisions.append(pr.precision())
            recalls.append(pr.recall())
        
        avgAccuracy = np.mean(accuracies)
        stdAccuracy = np.std(accuracies)
        avgPrecision = np.mean(precisions)
        stdPrecision = np.std(precisions)
        avgRecall = np.mean(recalls)
        stdRecall = np.std(recalls)
        areaUnderROC = computePooledAROC(prs) 
        
        if debuggingOutput:
            print '=============================================================================================='
        
        print 'Accuracy: ' + str(avgAccuracy) + ' ' + str(stdAccuracy)
        print 'Precision: ' + str(avgPrecision) + ' ' + str(stdPrecision)
        print 'Recall: ' + str(avgRecall) + ' ' + str(stdRecall) 
        print 'Area under ROC: ' + str(areaUnderROC)

    def trainNetwork(self, neuralNetwork, numIterations, trainingExamples, testingExamples, debuggingOutput=True):
        pr = neuralNetwork.evaluatePerformance(testingExamples)
        if debuggingOutput:
            print 'INITIAL: ' + 'Sum-Squared-Errors=' + str(pr.sumSquaredErrors) + '; Accuracy=' + str(pr.accuracy()) + '; Precision=' + str(pr.precision()) + '; Recall=' + str(pr.recall())
        for i in range(0, numIterations):
            converged = neuralNetwork.executeTrainingIteration(trainingExamples)
            if debuggingOutput and (i+1) % 10 == 0:
                pr = neuralNetwork.evaluatePerformance(testingExamples)
                print 'AFTER ' + str(i+1) + ' TRAINING EPOCHS: ' + 'Sum-Squared-Errors=' + str(pr.sumSquaredErrors) + '; Accuracy=' + str(pr.accuracy()) + '; Precision=' + str(pr.precision()) + '; Recall=' + str(pr.recall())
            if converged:
                print 'CONVERGED ON ITERATION ' + str(i)
                break
            
        pr = neuralNetwork.evaluatePerformance(testingExamples)
        if debuggingOutput:
            print 'FINAL: ' + 'Sum-Squared-Errors=' + str(pr.sumSquaredErrors) + '; Accuracy=' + str(pr.accuracy()) + '; Precision=' + str(pr.precision()) + '; Recall=' + str(pr.recall())
        return pr
    
# MAIN
np.random.seed(12345)
dataPath, useCrossValidation, numberOfHiddenNodes, weightDecayCoeff, numberOfTrainingIterations = parseCommandLine()
neuralNetworkManager = NeuralNetworkManager(dataPath, numberOfHiddenNodes, weightDecayCoeff, useCrossValidation)
neuralNetworkManager.train(numberOfTrainingIterations, debuggingOutput=True)

