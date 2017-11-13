import random
import math
from mldata import *
from dtree.dtree import DTree
from ann.neuralNetwork import NeuralNetwork
from nbayes.bayesianNetwork import BayesianNetwork
from logreg.logregClassifier import LogRegClassifier

class BoostedClassifierManager:
    def __init__(self, learning_alg, numIterations, schema):
        self.totalIterations = numIterations
        numInputs = 0
        for i in range(0, len(schema.features)):
            nominalAttributeHash = None
            if schema.features[i].type == Feature.Type.NOMINAL or schema.features[i].type == Feature.Type.BINARY or \
                            schema.features[i].type == Feature.Type.CONTINUOUS:
                numInputs = numInputs + 1

        self.weightedTrainingError = -1
        self.boostedClassifiers = []
        self.numInputs = numInputs
        self.learning_alg = learning_alg
        self.schema = schema

    def _createBoostedClassifier(self):
       if self.learning_alg == 'DTREE':
            return BoostedClassifier(DTree(self.schema, maxDepth=1, useInformationGain=False))
       elif self.learning_alg == 'ANN':
            return BoostedClassifier(NeuralNetwork([1], self.numInputs, weightDecayCoeff=0, maxTrainingIterations=-1))
       elif self.learning_alg == 'NBAYES':
            return BoostedClassifier(BayesianNetwork(self.schema, numberOfBins=15, mEstimate=1))
       else:  # learning_alg == 'LOGREG'
            return BoostedClassifier(LogRegClassifier(self.numInputs, const_lambda=0.01))

    def train(self, trainingExamples, testingExamples):
        examples = trainingExamples
        for i in range(0, self.totalIterations):
            boostedClf = self._createBoostedClassifier()
            self.boostedClassifiers.append(boostedClf)
            clf = boostedClf.classifier
            clf.train(examples)
            tp, tn, fp, fn, targetOutputPairs = clf.evaluateExamples(testingExamples)
            boostedClf.setNewClassifierWeight(testingExamples, targetOutputPairs)

    def evaluateExamples(self, examples):
        classifierSummation = self._totalClassifierWeightSummation()
        for bclf in self.boostedClassifiers:
            bclf.setTargetOutputPairs(examples)

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        targetOutputPairs = []

        for i in range(0, len(examples)):
            evaluationFunctionResult = self._evalutaionFunction(i, bclf.classifierWeight, classifierSummation, self.boostedClassifiers)
            output = round(evaluationFunctionResult)
            target = examples[i].target
            targetOutputPairs.append(list((target, output)))
            if target and output:
                tp = tp + 1
            elif (not target) and (not output):
                tn = tn + 1
            elif (not target) and output:
                fp = fp + 1
            elif target and not output:
                fn = fn + 1
        return tp, fp, tn, fn, targetOutputPairs


    def _evalutaionFunction(self, exampleIndex, classifierWeight, classifierSummation, boostedClassifiers):
        result = 0.0
        for bclf in boostedClassifiers:
            result = result + (classifierWeight / classifierSummation) * float(bclf.targetOutputPairs[exampleIndex][1])
        return result

    def _totalClassifierWeightSummation(self):
        sum = 0.0
        for bclf in self.boostedClassifiers:
            sum = sum + bclf.classifierWeight
        return sum

class BoostedClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
        self.classifierWeight = -1
        self.targetOutputPairs = []

    def _roundOutputs(self, targetOutputPairs):
        for targetOutputPair in targetOutputPairs:
            targetOutputPair[1] = round(targetOutputPair[1])

    def _calculateWeightedTrainingError(self, testingExamples, targetOutputPairs):
        weightedTrainingError = 0.0
        self._roundOutputs(targetOutputPairs)
        for i in range(0, len(targetOutputPairs)):
            weightedTrainingError = weightedTrainingError + testingExamples[i].weight * (targetOutputPairs[i][0] != targetOutputPairs[i][1])
        return weightedTrainingError

    def setNewClassifierWeight(self, testingExamples, targetOutputPairs):
        weightedTrainingError = self._calculateWeightedTrainingError(testingExamples, targetOutputPairs)
        self.classifierWeight = 0.5 * math.log((1-weightedTrainingError)/weightedTrainingError)

    def setTargetOutputPairs(self, testingExamples):
        tp, fp, tn, fn, targetOutputPairs = self.classifier.evaluateExamples(testingExamples)
        self.targetOutputPairs = targetOutputPairs
