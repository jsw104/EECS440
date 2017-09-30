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
        
        self.evaluateNetworkPerformance(self.examples)
        for i in range (0, numberOfTrainingIterations):
            print("Currently on training iteration " + str(i+1))
            self.executeTrainingIteration(self.examples)
            self.evaluateNetworkPerformance(self.examples)
            #for layer in self.layers:
            #    print 'biases'
            #    print layer.biases
            #    print 'weights'
            #    print layer.weights
                

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
        #layerActivationDerivs.append(None) #Dummy entry to keep indexes aligned
        while True:
            layerOutputs, binaryLayerOutputs, activationDerivs = self.layers[layerIndex].getOutputs(layerInputs[-1])
            layerInputs.append(layerOutputs)
            layerActivationDerivs.append(activationDerivs)
            if layerIndex+1 < len(self.layers):
                layerIndex = layerIndex + 1
            else:
                break      
                
        errors = layerInputs[layerIndex+1] - targets 
        downstreamBiasSensitivities = errors #might need to transpose this if we have multiple outputs
        downstreamWeights = np.ones(len(errors)) #might need to transpose this if we have multiple outputs
        
        while layerIndex >= 0:
            layer = self.layers[layerIndex]
            downstreamWeights, downstreamBiasSensitivities = layer.backpropagate(layerInputs[layerIndex],layerActivationDerivs[layerIndex],downstreamBiasSensitivities,downstreamWeights)
            layerIndex = layerIndex - 1
    
    def executeTrainingIteration(self, trainingExamples):
        for example in trainingExamples:
            inputs = self.normalizeExample(example)
            targets = np.array([1.0]) if example[-1] else np.array([0.0])
            self._train(inputs, targets)      
            #break #for debug only -- remove this     

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
            values, binaryValues, derivs = layer.getOutputs(values)
                                        
        rawErrors = targetOutputs - values
        binaryErrors = np.absolute(np.rint(values) - targetOutputs) # 0 => Correct; 1 => Wrong
        return rawErrors, binaryErrors

    def evaluateNetworkPerformance(self, examples):
        numCorrect = 0
        sumSquaredErrors = 0
        for i in range(0,len(examples)-1):
            exampleInputs = self.normalizeExample(examples[i]) #Let's not have to normalize every time
            targetOutputs = np.array([1.0]) if examples[i][-1] else np.array([0.0])
            rawErrors, binaryErrors = self.evaluateExample(exampleInputs, targetOutputs)
            sumSquaredErrors = sumSquaredErrors + 0.5 * np.sum(rawErrors*rawErrors)
            if(np.sum(binaryErrors) == 0):
                numCorrect = numCorrect + 1
            
        print 'SUM-SQUARED-ERRORS: ' + str(sumSquaredErrors)
        print 'NUM CORRECT: ' + str(numCorrect) + '/' + str(len(examples))

