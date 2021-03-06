import sys
import os
import random
import numpy as np
from utils import *
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


np.random.seed(12345)
random.seed(12345)
dataPath, useCrossValidation, numberOfBins, mEstimate = parseCommandLine()
exampleSet = getExamplesFromDataPath(dataPath)

trainingSets = []
testSets = []

accuracies = []
precisions = []
recalls = []
listsTargetOutputPairs = []

exampleManager = ExampleManager(exampleSet.examples, useCrossValidation)
if not useCrossValidation:
    trainingExamples, testExamples = exampleManager.getUnfoldedExamples()
    trainingSets.append(trainingExamples)
    testSets.append(testExamples)
else:
    for i in range(0, exampleManager.numFolds()):
        trainingExamples, testExamples = exampleManager.getCrossValidationExamples(i)
        trainingSets.append(trainingExamples)
        testSets.append(testExamples) 
        
for i in range(0, len(trainingSets)):
    trainingExamples = trainingSets[i]
    testExamples = testSets[i]
    if useCrossValidation:
        print 'Processing Fold ' + str(i+1) 
    bayesianNetwork = BayesianNetwork(trainingExamples, exampleSet.schema, numberOfBins, mEstimate)
    tp, fp, tn, fn, targetOutputPairs = bayesianNetwork.evaluateExamples(testExamples)
    accuarcy, precision, recall = computeStatistics(tp, fp, tn, fn)
    accuracies.append(accuarcy)
    precisions.append(precision)
    recalls.append(recall)
    listsTargetOutputPairs.append(targetOutputPairs)
    
avgAccuracy = np.mean(accuracies)
stdAccuracy = np.std(accuracies)
avgPrecision = np.mean(precisions)
stdPrecision = np.std(precisions)
avgRecall = np.mean(recalls)
stdRecall = np.std(recalls)
aroc = computePooledAROC(listsTargetOutputPairs)

print 'Accuracy: ' + str(avgAccuracy) + ' ' + str(stdAccuracy)
print 'Precision: ' + str(avgPrecision) + ' ' + str(stdPrecision)
print 'Recall: ' + str(avgRecall) + ' ' + str(stdRecall) 
print 'Area under ROC: ' + str(aroc)