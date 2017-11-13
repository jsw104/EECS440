from neuralNetworkLayer import *
import os
from mldata import *
import numpy as np
from targetOutputPair import *

class NeuralNetwork:
    def __init__(self, layerSizesList, numFirstLayerInputs, weightDecayCoeff, maxTrainingIterations):       
        self.layers = []
        
        numLastLayerOutputs = numFirstLayerInputs
        for layerSize in layerSizesList:
            self.layers.append(NeuralNetworkLayer(layerSize, numLastLayerOutputs))
            numLastLayerOutputs = layerSize
               
        self.weightDecayCoeff = weightDecayCoeff
        self.maxTrainingIterations = maxTrainingIterations #-1 => always train until convergence
        self.learningRate = 0.01                

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
            downstreamWeights, downstreamBiasSensitivities = layer.backpropagate(layerInputs[layerIndex],layerActivationDerivs[layerIndex],downstreamBiasSensitivities,downstreamWeights,learningRate=self.learningRate)
            layerIndex = layerIndex - 1
    
    def train(self, trainingExamples): #negative numIterations => train until convergence
        
        trainUntilConvervence = self.maxTrainingIterations < 0
        iterationCounter = 0
        allLayersConverged = False
        
        while(True):
            if allLayersConverged or (not trainUntilConvervence and iterationCounter >= self.maxTrainingIterations):
                break
            
            initialBiasValues = []
            initialWeightValues = []
            for layer in self.layers:
                initialBiasValues.append(layer.biases)
                initialWeightValues.append(layer.weights)
                
                if self.weightDecayCoeff > 0.0: # WEIGHT DECAY
                    layer.biases = (1 - 2*self.weightDecayCoeff*self.learningRate) * layer.biases
                    layer.weights = (1 - 2*self.weightDecayCoeff*self.learningRate) * layer.weights
                
            for example in trainingExamples:
                self._train(example.inputs, example.target)  
            
            allLayersConverged = True
            for i in range(0,len(self.layers)):
                allLayersConverged = allLayersConverged and self.layers[i].checkConvergence(initialBiasValues[i], initialWeightValues[i]) 
            
            iterationCounter = iterationCounter + 1
                    
    def stimulateNetwork(self, inputs):
        values = inputs
        for layer in self.layers:
            values, derivs = layer.getOutputs(values)
        return values       

    def evaluateExamples(self, examples): 
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        targetOutputPairs = []
        
        for example in examples:
            output = self.stimulateNetwork(example.inputs)[0][0]
            booleanOutput = int(round(output)) == 1
            if booleanOutput and example.target:
                tp = tp + 1
            elif booleanOutput and not example.target:
                fp = fp + 1
            elif not booleanOutput and example.target:
                fn = fn + 1
            else:
                tn = tn + 1
            targetOutputPairs.append(list((example.target, output)))
        
        return tp, fp, tn, fn, targetOutputPairs