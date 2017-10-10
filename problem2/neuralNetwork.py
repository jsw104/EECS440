from neuralNetworkLayer import *
import os
from mldata import *
import numpy as np
from targetOutputPair import *

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
        for i in range(0,len(self.layers)):
            allLayersConverged = allLayersConverged and self.layers[i].checkConvergence(initialBiasValues[i], initialWeightValues[i]) 
            
        return allLayersConverged  

    def stimulateNetwork(self, inputs):
        values = inputs
        for layer in self.layers:
            values, derivs = layer.getOutputs(values)
        return values       

    def evaluatePerformance(self, examples): 
        return PerformanceEvaluation(self, examples)
    
    
class PerformanceEvaluation:
        
    def __init__(self, neuralNet, examples):
        self.neuralNet = neuralNet
        self.examples = examples
        
        self.targetOutputPairs = []
        for example in self.examples:
            self.targetOutputPairs.append(TargetOutputPair(example.targets, self.neuralNet.stimulateNetwork(example.inputs)))
        
        sumSquaredErrors = 0
        for targetOutputPair in self.targetOutputPairs:
            rawErrors = targetOutputPair.outputs - targetOutputPair.targets
            sumSquaredErrors = sumSquaredErrors + 0.5 * np.sum(rawErrors*rawErrors) 
        self.sumSquaredErrors = sumSquaredErrors
        
    def itermediateStatistics(self, decisionThresh=0.5):
        if self.neuralNet.layers[-1].numNodesThisLayer != 1:
            raise RuntimeError('intermediateStatistics not available with multiple-output networks')
        else:
            targetShape = self.examples[0].targets.shape
            tp = 0
            tn = 0
            fp = 0
            fn = 0         
        
            for targetOutputPair in self.targetOutputPairs:
                target = targetOutputPair.targets[0]
                output = targetOutputPair.outputs[0]
                signedBinaryError = int((1-decisionThresh)+output) - (1 if target else 0) # 0 => Correct (TP or TN); +1 => Wrong (FP); -1 => Wrong (FN)   
                if signedBinaryError == 1:
                    fp = fp + 1
                elif signedBinaryError == -1:
                    fn = fn + 1
                elif target:
                    tp = tp + 1
                else:
                    tn = tn + 1
            
            return tp, tn, fp, fn
        
    def accuracy(self, decisionThresh=0.5):
        tp,tn,fp,fn = self.itermediateStatistics(decisionThresh)
        numAll = tp+tn+fp+fn
        if numAll == 0:
            return 0.0
        return float(tp+tn)/(tp+tn+fp+fn)
    
    def precision(self, decisionThresh=0.5):
        tp,tn,fp,fn = self.itermediateStatistics(decisionThresh)
        numScoredPos = tp+fp
        if numScoredPos == 0:
            return 0.0
        return float(tp)/(tp+fp)
    
    def recall(self, decisionThresh=0.5):
        tp,tn,fp,fn = self.itermediateStatistics(decisionThresh)
        numActualPos = tp+fn
        if numActualPos == 0:
            return 0.0
        return float(tp)/(tp+fn)
