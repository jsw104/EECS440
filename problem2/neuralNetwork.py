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
        for i in range(1, numberOfHiddenNodes):
            self.hiddenLayerNodes.append(NeuralNode(len(self.schema.features)))
        self.outputNode = NeuralNode(len(self.hiddenLayerNodes))
        self.weightDecay = weightDecay
        self.numberOfTrainingIterations = numberOfTrainingIterations
