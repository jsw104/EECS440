import sys
import os
import random
import numpy as np
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

def computeStatistics(tp, fp, tn, fn):
    numAll = tp+tn+fp+fn
    numScoredPos = tp+fp
    numActualPos = tp+fn
    accuracy = 0.0 if numAll == 0 else float(tp+tn)/numAll
    precision = 0.0 if numScoredPos == 0 else float(tp)/numScoredPos
    recall = 0.0 if numActualPos == 0 else float(tp)/numActualPos
    return accuracy, precision, recall

def computePooledAROC(listsTargetOutputPair):
    allTargetOutputPairs = []
    for pairList in listsTargetOutputPair:        
        allTargetOutputPairs = allTargetOutputPairs + pairList
        
    rocPoints = (len(allTargetOutputPairs)+2)*[None]
    rocPoints[0] = (0.0,0.0,1.0)
    rocPoints[-1] = (1.0,1.0,0.0)
    appendIndex = 1
    allTargetOutputPairs.sort(key = lambda x:x[1])

    totalTP = 0.0
    totalFP = 0.0
    totalTN = 0.0
    totalFN = 0.0
    #start with threshold of zero confidence.
    for targetOutputPair in allTargetOutputPairs:
        if targetOutputPair[0]:
            totalTP = totalTP + 1
        else:
            totalFP = totalFP + 1

    #incrementally move confidence level over to the right
    for targetOutputPair in allTargetOutputPairs:
        if targetOutputPair[0]:
            totalTP = totalTP - 1
            totalFN = totalFN + 1
        else:
            totalFP = totalFP - 1
            totalTN = totalTN + 1
        fpRate = 0.0 if totalFP + totalTN == 0 else totalFP/(totalFP + totalTN)
        tpRate = 0.0 if totalTP + totalFN == 0 else totalTP/(totalTP + totalFN)
        rocPoints[appendIndex] = (fpRate, tpRate)
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

random.seed(12345)
dataPath, useCrossValidation, numberOfBins, mEstimate = parseCommandLine()
exampleSet = getExamplesFromDataPath(dataPath)

accuracies = []
precisions = []
recalls = []
listsTargetOutputPairs = []

exampleManager = ExampleManager(exampleSet.examples, useCrossValidation)
if not useCrossValidation:
    trainingExamples, testExamples = exampleManager.getUnfoldedExamples()
    bayesianNetwork = BayesianNetwork(trainingExamples, exampleSet.schema, numberOfBins, mEstimate)
    tp, fp, tn, fn, targetOutputPairs = bayesianNetwork.evaluateExamples(testExamples)
    accuarcy, precision, recall = computeStatistics(tp, fp, tn, fn)
    accuracies.append(accuarcy)
    precisions.append(precision)
    recalls.append(recall)
    listsTargetOutputPairs.append(targetOutputPairs)

else:
    for i in range(0, exampleManager.numFolds()):
        print 'Processing Fold ' + str(i+1)
        trainingExamples, testExamples = exampleManager.getCrossValidationExamples(i)
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