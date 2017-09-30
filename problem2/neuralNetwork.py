from neuralNode import *
from neuralNetworkLayer import *
import os
from mldata import *
import numpy as np

class NeuralNetwork:
    def __init__(self, dataPath, useCrossValidation, numberOfHiddenNodes, weightDecayCoeff, numberOfTrainingIterations):
        if type(dataPath) is not str:
            raise ValueError('The data path must be a string')
        fileName = os.path.basename(dataPath)
        rootDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath[0:-(len(fileName) + 1)])

        exampleSet = parse_c45(fileName, rootDirectory)
        self.examples = exampleSet.examples
        self.schema = exampleSet.schema
        
        self.featureUsefullnessMask = np.zeros(len(self.schema.features))
        self.numUsefulFeatures = 0
        for i in range(0,len(self.schema.features)-1):
            if self.schema.features[i].type == Feature.Type.NOMINAL or self.schema.features[i].type == Feature.Type.BINARY or self.schema.features[i].type == Feature.Type.CONTINUOUS:
                self.featureUsefullnessMask[i] = 1
                self.numUsefulFeatures = self.numUsefulFeatures + 1
        
        self.layers = []
        self.hiddenLayer = NeuralNetworkLayer(numberOfHiddenNodes, self.numUsefulFeatures)
        self.layers.append(self.hiddenLayer)
        self.outputLayer = NeuralNetworkLayer(1, numberOfHiddenNodes)
        self.layers.append(self.outputLayer)
        
        self.useCrossValidation = useCrossValidation
        
        self.weightDecayCoeff = weightDecayCoeff
        self.numberOfTrainingIterations = numberOfTrainingIterations

        self.nominalAttributeHash = {}
        self._constructNominalAttributeHash()

        #for i in range(1, numberOfHiddenNodes):
            #the first and last feature are not useful
            #self.hiddenLayerNodes.append(NeuralNode(len(self.schema.features) - 2))
        
        #self.outputNode = NeuralNode(len(self.hiddenLayerNodes))
        
        self.evaluateNetworkPerformance(self.examples)
        for i in range (0, numberOfTrainingIterations):
            print("Currently on training iteration " + str(i+1))
            self.executeTrainingIteration(self.examples)
            self.evaluateNetworkPerformance(self.examples)
            for layer in self.layers:
                print 'biases'
                print layer.biases
                print 'weights'
                print layer.weights
                

    def _constructNominalAttributeHash(self):
        for i in range(1, len(self.schema) - 1):
            if self.schema[i].type == 'NOMINAL':
                for value in self.schema[i].values:
                    if value not in self.nominalAttributeHash:
                        self.nominalAttributeHash[value] = len(self.nominalAttributeHash.keys())

    def _train(self, inputs, targets):
        layerIndex = 0
        layerInputs = []
        layerActivationDerivs = []
        layerInputs.append(inputs)
        layerActivationDerivs.append(None) #Dummy entry to keep indexes aligned
        while True:
            layerOutputs, activationDerivs = self.layers[layerIndex].getOutputs(layerInputs[-1])
            layerInputs.append(layerOutputs)
            layerActivationDerivs.append(activationDerivs)
            if layerIndex+1 < len(self.layers):
                layerIndex = layerIndex + 1
            else:
                break      
                
        errors = targets - layerInputs[layerIndex+1] 
        downstreamBiasSensitivities = errors #might need to transpose this if we have multiple outputs
        downstreamWeights = np.ones(len(errors)) #might need to transpose this if we have multiple outputs
        
        while layerIndex >= 0:
            print '============================================'
            layer = self.layers[layerIndex]
            print layer.numInputs, layer.numNodesThisLayer
            print layer.biases
            print layer.weights
            #print 'LAYER:' + str(layerIndex) + ' nodes: ' + str(layer.numNodesThisLayer) + ' inputs: ' + str(layer.numInputs)
            downstreamWeights, downstreamBiasSensitivities = layer.backpropagate(layerInputs[layerIndex+1],layerActivationDerivs[layerIndex+1],downstreamBiasSensitivities,downstreamWeights)
            layerIndex = layerIndex - 1
            print '============================================'

    
    def executeTrainingIteration(self, trainingExamples):
        for example in trainingExamples:
            #expectedOutput = example[-1]
            #example = self.normalizeExample(example)
            #self.evaluateExample(example, expectedOutput)
            inputs = self.normalizeExample(example)
            targets = np.array([1.0]) if example[-1] else np.array([0.0])
            self._train(inputs, targets)      
            break #for debug only -- remove this
        
        

    def normalizeExample(self, example):
        inputList = []
        for i in range(1, len(example) - 1): #This 'skip the first and last' logic should be de-hardcoded
            if (self.schema[i].type == 'NOMINAL'):
                inputList.append(self.normalizeFeature(example[i])) 
            else:
                 inputList.append(example[i])
        return np.array(inputList)

    def normalizeFeature(self, feature):
        if feature in self.nominalAttributeHash:
            return self.nominalAttributeHash[feature]
        return feature

    def evaluateExample(self, exampleInputs, targetOutputs):
        values = exampleInputs
        for layer in self.layers:
            values, derivs = layer.getOutputs(values)
            
        errors = targetOutputs - values
        return errors

    def evaluateNetworkPerformance(self, examples):
        numCorrect = 0
        sumSumSquaredErrors = 0
        for i in range(0,len(examples)-1):
            exampleInputs = self.normalizeExample(examples[i]) #Let's not have to normalize every time
            targetOutputs = np.array([1.0]) if examples[i][-1] else np.array([0.0])
            rawErrors = self.evaluateExample(exampleInputs, targetOutputs)
            if(abs(np.sum(rawErrors)) < 0.5):
                numCorrect = numCorrect + 1
            sumSumSquaredErrors = sumSumSquaredErrors + 0.5 * np.sum(rawErrors*rawErrors)
        print 'SUM-SUM-SQUARED-ERRORS: ' + str(sumSumSquaredErrors)
        print 'NUM CORRECT: ' + str(numCorrect) + '/' + str(len(examples))

