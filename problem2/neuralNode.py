import random
import math

class NeuralNode:
    def __init__(self, numberOfWeights):
        self.weights = []
        for i in range(0, numberOfWeights):
            self.weights.append(random.uniform(0, 1))

    def calculateInputWeightSummation(self, inputs):
        if (len(inputs) != len(self.weights)):
            print "We have a problem"
        sum = 0
        for i in range(0, len(self.weights)):
            sum = sum + inputs[i] * self.weights[i]
        return sum

    #use this for evaluation phase.
    def willFire(self, summation):
        return (self.calculateSigmoid(summation) > 0.5)

    #use this for training phase
    def calculateSigmoid(self, x):
        return 1 / (1 + math.exp(-x))