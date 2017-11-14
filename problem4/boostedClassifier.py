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
        for i in range(0, self.totalIterations):
            boostedClf = self._createBoostedClassifier()
            self.boostedClassifiers.append(boostedClf)
            clf = boostedClf.classifier
            clf.train(trainingExamples)
            tp, tn, fp, fn, trainingTargetOutputPairs = clf.evaluateExamples(trainingExamples)
            tp, tn, fp, fn, testingTargetOutputPairs = clf.evaluateExamples(testingExamples)
            boostedClf.roundOutputs(trainingTargetOutputPairs) #rounding makes outputs equal to the classifier guess
            boostedClf.roundOutputs(testingTargetOutputPairs)
            boostedClf.setNewClassifierWeight(testingExamples, testingTargetOutputPairs)
            self._updateWeights(trainingExamples, trainingTargetOutputPairs, boostedClf.classifierWeight)

    def _updateWeights(self, examples, targetOutputPairs, classifierWeight):
        newWeightSummation = 0.0
        for exampleIndex in range(0, len(examples)):
            exponentSign = 1.0
            if (targetOutputPairs[exampleIndex][0] == targetOutputPairs[exampleIndex][1]):
                exponentSign = -1.0
            exponent = classifierWeight * exponentSign
            newWeight = examples[exampleIndex].weight * math.exp(exponent)
            examples[exampleIndex].weight = newWeight
            newWeightSummation = newWeightSummation + newWeight

        #normalize weights to add up to 1
        for example in examples:
            example.weight = example.weight / newWeightSummation



    def evaluateExamples(self, examples):
        for bclf in self.boostedClassifiers:
            bclf.setTargetOutputPairs(examples)

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        targetOutputPairs = []

        for i in range(0, len(examples)):
            evaluationFunctionResult = self._evaluationFunction(i, self.boostedClassifiers)
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


    def _evaluationFunction(self, exampleIndex, boostedClassifiers):
        result = 0.0
        classifierSummation = self._totalClassifierWeightSummation()
        for bclf in boostedClassifiers:
            result = result + (bclf.classifierWeight / classifierSummation) * float(bclf.targetOutputPairs[exampleIndex][1])
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

    def roundOutputs(self, targetOutputPairs):
        for targetOutputPair in targetOutputPairs:
            targetOutputPair[1] = round(targetOutputPair[1])

    def _calculateWeightedTrainingError(self, testingExamples, targetOutputPairs):
        weightedTrainingError = 0.0
        for i in range(0, len(targetOutputPairs)):
            weightedTrainingError = weightedTrainingError + testingExamples[i].weight * (targetOutputPairs[i][0] != targetOutputPairs[i][1])
        return weightedTrainingError

    def setNewClassifierWeight(self, examples, targetOutputPairs):
        weightedTrainingError = self._calculateWeightedTrainingError(examples, targetOutputPairs)
        self.classifierWeight = 0.5 * math.log((1-weightedTrainingError)/weightedTrainingError)

    def setTargetOutputPairs(self, examples):
        tp, fp, tn, fn, targetOutputPairs = self.classifier.evaluateExamples(examples)
        self.targetOutputPairs = targetOutputPairs
