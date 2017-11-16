import sys
import numpy as np
import random
from exampleManager import *
from utils import *
from boostedClassifier import *

# example: python nbayes ../testData/spam/spam 0 ann
def parseCommandLine():
    # sys.argv[0] is the name of the script so we need 4 args for 3 options.
    if (len(sys.argv) is not 5):
        raise ValueError('You must run with 4 options.')

    dataPath = sys.argv[1]
    if type(dataPath) is not str:
        raise ValueError('The data path must be a string')

    useCrossValidation = int(sys.argv[2]) == 0

    learning_alg = sys.argv[3].upper()
    if learning_alg not in ['DTREE', 'ANN', 'NBAYES', 'LOGREG']:
        raise ValueError('learning_alg must be DTREE, ANN, NBAYES, or LOGREG')

    numIterations = int(sys.argv[4])

    return dataPath, useCrossValidation, learning_alg, numIterations


np.random.seed(12345)
random.seed(12345)
dataPath, useCrossValidation, learning_alg, numIterations = parseCommandLine()

exampleSet = getExamplesFromDataPath(dataPath)

exampleNormalizer = ExampleNormalizer(exampleSet, learning_alg)
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
    exampleNormalizer.resetWeights(trainingExamples)
    exampleNormalizer.resetWeights(testExamples)
    
    if useCrossValidation:
        print 'Processing Fold ' + str(i + 1)

    boostedClf = BoostedClassifierManager(learning_alg, numIterations, exampleSet.schema)
    boostedClf.train(trainingExamples, testExamples)
    tp, fp, tn, fn, targetOutputPairs = boostedClf.evaluateExamples(testExamples)
    print 'tp: ' + str(tp)
    print 'fp: ' + str(fp)
    print 'tn: ' + str(tn)
    print 'fn: ' + str(fn)
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
