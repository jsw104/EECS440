import sys
import numpy as np
import random
from exampleManager import *
from utils import *
from dtree.dtree import DTree
from ann.neuralNetwork import NeuralNetwork
from nbayes.bayesianNetwork import BayesianNetwork
from logreg.logregClassifier import LogRegClassifier


#class Classifier:
#    def __init__(self):
        

# example: python nbayes ../testData/spam/spam 0 ann
def parseCommandLine():
    # sys.argv[0] is the name of the script so we need 4 args for 3 options.
    if (len(sys.argv) is not 4):
        raise ValueError('You must run with 3 options.')

    dataPath = sys.argv[1]
    if type(dataPath) is not str:
        raise ValueError('The data path must be a string')

    useCrossValidation = int(sys.argv[2]) == 0
    
    learning_alg = sys.argv[3].upper()
    if learning_alg not in ['DTREE', 'ANN', 'NBAYES', 'LOGREG']:
        raise ValueError('learning_alg must be DTREE, ANN, NBAYES, or LOGREG')

    return dataPath, useCrossValidation, learning_alg



np.random.seed(12345)
random.seed(12345)
dataPath, useCrossValidation, learning_alg = parseCommandLine()

exampleSet = getExamplesFromDataPath(dataPath)

exampleManager = None
if learning_alg == 'DTREE' or learning_alg == 'NBAYES':
    exampleManager = ExampleManager(exampleSet.examples, useCrossValidation)
else:
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
    clf = None
    if learning_alg == 'DTREE':
        clf = DTree(exampleSet.schema, maxDepth=1, useInformationGain=False)
    elif learning_alg == 'ANN':
        clf = NeuralNetwork([1], len(trainingExamples[0].inputs), weightDecayCoeff=0, maxTrainingIterations=-1)
    elif learning_alg == 'NBAYES':
        clf =  BayesianNetwork(exampleSet.schema, numberOfBins=15, mEstimate=1)
    else: #learning_alg == 'LOGREG'
        clf = LogRegClassifier(len(trainingExamples[0].inputs), const_lambda=0.01)
    
    clf.train(trainingExamples)   
    tp, fp, tn, fn, targetOutputPairs = clf.evaluateExamples(testExamples)
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


