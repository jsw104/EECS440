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
        initialBiasValues = []
        initialWeightValues = []
        for layer in self.layers:
            initialBiasValues.append(layer.biases)
            initialWeightValues.append(layer.weights)
            
        for example in trainingExamples:
            self._train(example.inputs, example.targets)  
        
        allLayersConverged = True
        for i in range(0,len(self.layers)-1):
            allLayersConverged = allLayersConverged and self.layers[i].checkConvergence(initialBiasValues[i], initialWeightValues[i]) 
            
        return allLayersConverged  

    def stimulateNetwork(self, inputs):
        values = inputs
        for layer in self.layers:
            values, derivs = layer.getOutputs(values)
        return values       

    def evaluatePerformance(self, examples): 
        outputs = []
        for example in examples:
            outputs.append(self.stimulateNetwork(example.inputs))
        
        return PerformanceEvaluationResult(examples, outputs)
    
    
class PerformanceEvaluationResult:
        
    def __init__(self, examples, outputs):
        self.examples = examples
        self.outputs = outputs
        
        targetShape = examples[0].targets.shape
        numTruePositives = np.zeros(targetShape)
        numFalsePositives = np.zeros(targetShape)
        numTrueNegatives = np.zeros(targetShape)
        numFalseNegatives = np.zeros(targetShape) 
        sumSquaredErrors = 0
        
        for i in range(0,len(examples)):
            example = examples[i]
            output = outputs[i]
            rawErrors = output - example.targets
            signedBinaryErrors = np.rint(output) - example.targets # 0 => Correct (TP or TN); +1 => Wrong (FP); -1 => Wrong (FN)    
            sumSquaredErrors = sumSquaredErrors + 0.5 * np.sum(rawErrors*rawErrors) 
            numTruePositives = numTruePositives + np.multiply(np.equal(signedBinaryErrors, np.zeros(targetShape)), example.targets)
            numFalsePositives = numFalsePositives + np.equal(signedBinaryErrors, np.ones(targetShape))
            numTrueNegatives = numTrueNegatives + np.multiply(np.equal(signedBinaryErrors, np.zeros(targetShape)), np.invert(example.targets))
            numFalseNegatives = numFalseNegatives + np.equal(signedBinaryErrors, np.full(targetShape, -1))
        
        self.sumSquaredErrors = sumSquaredErrors
        self.tp = numTruePositives
        self.tn = numTrueNegatives
        self.fp = numFalsePositives
        self.fn = numFalseNegatives
        self.totalCorrect = self.tp + self.tn
        self.totalWrong = self.fp + self.fn
        self.totalExamples = len(examples)
        
    def accuracy(self):
        a = np.divide(self.totalCorrect, self.totalExamples)
        if a.shape == (1,1):
            return a[0][0]
        return a
    
    def precision(self):
        p = np.divide(self.tp, self.tp+self.fp)
        if p.shape == (1,1):
            return p[0][0]
        return p
    
    def recall(self):
        r = np.divide(self.tp, self.tp+self.fn)
        if r.shape == (1,1):
            return r[0][0]
        return r
    
    def areaUnderROC(self):
        print 'TODO'
