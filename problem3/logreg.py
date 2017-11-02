import sys
import numpy as np
import random
from exampleManager import *
from utils import *
from logregClassifier import *

# example: python nbayes ../testData/spam/spam 1 0.001
def parseCommandLine():
    # sys.argv[0] is the name of the script so we need 4 args for 3 options.
    if (len(sys.argv) is not 4):
        raise ValueError('You must run with 3 options.')

    dataPath = sys.argv[1]
    if type(dataPath) is not str:
        raise ValueError('The data path must be a string')

    useCrossValidation = int(sys.argv[2]) == 0
    
    const_lambda = float(sys.argv[3])
    if const_lambda < 0:
        raise ValueError('lambda must be non-negative')

    return dataPath, useCrossValidation, const_lambda

        
np.random.seed(12345)
random.seed(12345)
dataPath, useCrossValidation, const_lambda = parseCommandLine()
exampleSet = getExamplesFromDataPath(dataPath)
exampleNormalizer = ExampleNormalizer(exampleSet)
normalizedExamples = exampleNormalizer.normalizeExamples(exampleSet)
exampleManager = ExampleManager(normalizedExamples, useCrossValidation)

trainingSets = []
testSets = []

accuracies = []
precisions = []
recalls = []
listsTargetOutputPairs = []

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
    lrc = LogRegClassifier(len(trainingExamples[0].inputs), const_lambda)
    lrc.train(trainingExamples)   
    tp, fp, tn, fn, targetOutputPairs = lrc.evaluateExamples(testExamples)
    print 'True positives: ' + str(tp)
    print 'False positives: ' + str(fp)
    print 'True negatives: ' + str(tn)
    print 'False negatives: ' + str(fn)
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
    
