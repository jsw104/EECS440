import random
from mldata import *
from dtree.dtree import DTree
from ann.neuralNetwork import NeuralNetwork
from nbayes.bayesianNetwork import BayesianNetwork
from logreg.logregClassifier import LogRegClassifier

class BoostedClassifier:
    def __init__(self, learning_alg, numBags, schema):

        numInputs = 0
        for i in range(0, len(schema.features)):
            nominalAttributeHash = None
            if schema.features[i].type == Feature.Type.NOMINAL or schema.features[i].type == Feature.Type.BINARY or \
                            schema.features[i].type == Feature.Type.CONTINUOUS:
                numInputs = numInputs + 1

        self.classifiers = []
        for i in range(0, numBags):
            if learning_alg == 'DTREE':
                self.classifiers.append(DTree(schema, maxDepth=1, useInformationGain=False))
            elif learning_alg == 'ANN':
                self.classifiers.append(NeuralNetwork([1], numInputs, weightDecayCoeff=0, maxTrainingIterations=-1))
            elif learning_alg == 'NBAYES':
                self.classifiers.append(BayesianNetwork(schema, numberOfBins=15, mEstimate=1))
            else:  # learning_alg == 'LOGREG'
                self.classifiers.append(LogRegClassifier(numInputs, const_lambda=0.01))

    def train(self, trainingExamples, testingExamples):
        examples = trainingExamples
        for i in range(0, len(self.classifiers)):
            clf = self.classifiers[i]
            clf.train(examples)
            tp, tn, fp, fn, targetOutputPairs = clf.evaluateExamples(testingExamples)

    def createErrorResidual(self, testingExamples, targetOutputPairs):
        for example in testingExamples:
            print str(example.target)
