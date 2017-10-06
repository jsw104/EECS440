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
        signedBinaryErrors = np.rint(outputs) - example.targets # 0 => Correct (TP or TN); +1 => Wrong (FP); -1 => Wrong (FN)
        return rawErrors, signedBinaryErrors

    def evaluatePerformance(self, examples):  
        targetShape = examples[0].targets.shape
        numTruePositives = np.zeros(targetShape)
        numFalsePositives = np.zeros(targetShape)
        numTrueNegatives = np.zeros(targetShape)
        numFalseNegatives = np.zeros(targetShape) 
        #numFullyCorrect = 0
        sumSquaredErrors = 0
        for example in examples:
            rawErrors, signedBinaryErrors = self.__evaluateExampleError(example)
            sumSquaredErrors = sumSquaredErrors + 0.5 * np.sum(rawErrors*rawErrors)
            #if(np.sum(signedBinaryErrors) == 0):
            #    numFullyCorrect = numFullyCorrect + 1 
            numTruePositives = numTruePositives + np.multiply(np.equal(signedBinaryErrors, np.zeros(targetShape)), example.targets)
            numFalsePositives = numFalsePositives + np.equal(signedBinaryErrors, np.ones(targetShape))
            numTrueNegatives = numTrueNegatives + np.multiply(np.equal(signedBinaryErrors, np.zeros(targetShape)), np.invert(example.targets))
            numFalseNegatives = numFalseNegatives + np.equal(signedBinaryErrors, np.full(targetShape, -1))
        
        return PerformanceEvaluationResult(sumSquaredErrors, numTruePositives, numFalsePositives, numTrueNegatives, numFalseNegatives)
    
    
    
class PerformanceEvaluationResult:
    def __init__(self, sumSquaredErrors, tp, fp, tn, fn):
        self.sumSquaredErrors = sumSquaredErrors
        self.tp = np.array(tp)
        self.fp = np.array(fp)
        self.tn = np.array(tn)
        self.fn = np.array(fn)
        self.totalCorrect = self.tp + self.tn
        self.totalWrong = self.fp + self.fn
        self.totalExamples = self.totalCorrect + self.totalWrong
        
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
