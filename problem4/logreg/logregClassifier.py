import numpy as np
import math

class LogRegClassifier():
    
    def __init__(self, num_inputs, const_lambda):
        self.const_lambda = const_lambda
        self.bias = np.random.uniform(-0.1,0.1)
        self.weights = np.random.uniform(-0.1,0.1,size=(num_inputs))
        self.learningRate = .0025
        
    def _sigmoid(self, x):
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            expx = math.exp(x)
            return expx / (1 + expx)
    
    def calculateOutput(self, example):
        weightedInputsSummation = example.inputs.dot(self.weights) + self.bias
        output = self._sigmoid(weightedInputsSummation)
        deriv = output * (1-output)
        return output, deriv
        
    def train(self, trainingExamples, maxIter=1000):
        absDiffBias = 1
        maxAbsDiffWeights = 1
        iterationCtr = 0
        while iterationCtr < maxIter and (absDiffBias > 0.015 or maxAbsDiffWeights > 0.015):
            initialBias = self.bias
            initialWeights = self.weights
            self._trainingIteration(trainingExamples)
            absDiffBias = abs(self.bias - initialBias)
            maxAbsDiffWeights = np.max(np.abs(self.weights - initialWeights))
            iterationCtr = iterationCtr + 1
            
    def evaluateExamples(self, testExamples):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        targetOutputPairs = []
        for example in testExamples:
            output, sigmoid_deriv = self.calculateOutput(example)
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
               
    def _trainingIteration(self, trainingExamples):
        for example in trainingExamples:
            output, sigmoid_deriv = self.calculateOutput(example)
                        
            magnitudeWeightsSquared = self.weights.dot(self.weights) + self.bias*self.bias
            magnitudeWeights = np.sqrt(magnitudeWeightsSquared)
            error = len(trainingExamples) * example.weight * (output - example.target)
            
            dE_dBias = sigmoid_deriv * error + self.const_lambda * magnitudeWeights * 2 * self.bias * (0.5 * self.const_lambda * magnitudeWeightsSquared)
            dE_dWeights = sigmoid_deriv * error * example.inputs + self.const_lambda * magnitudeWeights * 2 * self.weights * (0.5 * self.const_lambda * magnitudeWeightsSquared)   
    
            self.bias = self.bias - (self.learningRate * dE_dBias)
            self.weights = self.weights - (self.learningRate * dE_dWeights)
            