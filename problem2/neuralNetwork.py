from neuralNode import *
import os
from mldata import *

class NeuralNetwork:
    def __init__(self, dataPath, useCrossValidation, numberOfHiddenNodes, weightDecay, numberOfTrainingIterations):
        if type(dataPath) is not str:
            raise ValueError('The data path must be a string')
        fileName = os.path.basename(dataPath)
        rootDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath[0:-(len(fileName) + 1)])

        exampleSet = parse_c45(fileName, rootDirectory)
        self.schema = exampleSet.schema
        self.useCrossValidation = useCrossValidation
        self.hiddenLayerNodes = []
        self.weightDecay = weightDecay
        self.numberOfTrainingIterations = numberOfTrainingIterations

        self.nominalAttributeHash = {}
        self.constructNominalAttributeHash()

        for i in range(0, numberOfHiddenNodes):
            #the first and last feature are not useful
            self.hiddenLayerNodes.append(NeuralNode(len(self.schema.features) - 2))
        self.outputNode = NeuralNode(len(self.hiddenLayerNodes))

        for i in range (0, numberOfTrainingIterations):
            print("Currently on " + str(i) + " training iteration")
            self.executeTrainingIteration(exampleSet.examples)

    def constructNominalAttributeHash(self):
        for i in range(1, len(self.schema) - 1):
            if self.schema[i].type == 'NOMINAL':
                for value in self.schema[i].values:
                    if value not in self.nominalAttributeHash:
                        self.nominalAttributeHash[value] = len(self.nominalAttributeHash.keys())

    def executeTrainingIteration(self, trainingExamples):
        for example in trainingExamples:
            expectedOutput = example[-1]
            example = self.normalizeExample(example)
            self.evaluateExample(example, expectedOutput)

    def normalizeExample(self, example):
        inputList = []
        for i in range(1, len(example) - 1):
            inputList.append(self.normalizeFeature(example[i])) if (self.schema[i].type == 'NOMINAL') else inputList.append(example[i])
        return inputList

    def normalizeFeature(self, feature):
        if feature in self.nominalAttributeHash:
            return self.nominalAttributeHash[feature]
        return feature

    def evaluateExample(self, example, expectedOutput):
        hiddenLayerOutputs = []
        for hiddenLayerNode in self.hiddenLayerNodes:
            summation = hiddenLayerNode.calculateInputWeightSummation(example)
            hiddenLayerOutputs.append(hiddenLayerNode.calculateSigmoid(summation))
        outputNodeSummation = self.outputNode.calculateInputWeightSummation(hiddenLayerOutputs)
        output = self.outputNode.calculateSigmoid(outputNodeSummation)
        print "performance: " + str(self.calculatePerformance(output, expectedOutput))

    def calculatePerformance(self, output, expectedOutput):
        return 0.5 * (output - expectedOutput)**2