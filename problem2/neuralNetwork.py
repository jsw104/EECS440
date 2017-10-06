from neuralNetworkLayer import *
import os
from mldata import *
import numpy as np

class NeuralNetwork:
    def __init__(self, layerSizesList, numFirstLayerInputs, weightDecayCoeff):       
        self.layers = []
        
        numLastLayerOutputs = numFirstLayerInputs
        for layerSize in layerSizesList:
            self.layers.append(NeuralNetworkLayer(layerSize, numLastLayerOutputs))
            numLastLayerOutputs = layerSize
               
        self.weightDecayCoeff = weightDecayCoeff                

    def _train(self, inputs, targets):
        layerIndex = 0
        layerInputs = []
        layerActivationDerivs = []
        layerInputs.append(inputs)
        while True:
            layerOutputs, activationDerivs = self.layers[layerIndex].getOutputs(layerInputs[-1])
            layerInputs.append(layerOutputs)
            layerActivationDerivs.append(activationDerivs)
            if layerIndex+1 < len(self.layers):
                layerIndex = layerIndex + 1
            else:
                break      
                
        errors = layerInputs[layerIndex+1] - targets 
        downstreamBiasSensitivities = errors     #might need to transpose this if we have multiple outputs
        downstreamWeights = np.ones(len(errors)) #might need to transpose this if we have multiple outputs
        while layerIndex >= 0:
            layer = self.layers[layerIndex]
            downstreamWeights, downstreamBiasSensitivities = layer.backpropagate(layerInputs[layerIndex],layerActivationDerivs[layerIndex],downstreamBiasSensitivities,downstreamWeights,weightDecayFactor=self.weightDecayCoeff)
            layerIndex = layerIndex - 1
    
    def executeTrainingIteration(self, trainingExamples):
        for example in trainingExamples:
            self._train(example.inputs, example.targets)      

    def stimulateNetwork(self, inputs):
        values = inputs
        for layer in self.layers:
            values, derivs = layer.getOutputs(values)
        return values

    def __evaluateExampleError(self, example):
        outputs = self.stimulateNetwork(example.inputs)
        rawErrors = outputs - example.targets
        binaryErrors = np.absolute(np.rint(outputs) - example.targets) # 0 => Correct; 1 => Wrong
        return rawErrors, binaryErrors

    def evaluatePerformance(self, examples):
        numCorrect = 0
        sumSquaredErrors = 0
        for example in examples:
            rawErrors, binaryErrors = self.__evaluateExampleError(example)
            sumSquaredErrors = sumSquaredErrors + 0.5 * np.sum(rawErrors*rawErrors)
            if(np.sum(binaryErrors) == 0):
                numCorrect = numCorrect + 1
        
        return sumSquaredErrors, numCorrect
