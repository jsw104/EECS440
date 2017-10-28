import math
class ContinuousAttributeStandardizer:
    def __init__(self, examples, featureIndex):
        self.mean = self.computeMean(examples, featureIndex)
        self.standardDeviation = self.computeStandardDeviation(examples, featureIndex, self.mean)

    def computeMean(self, examples, featureIndex):
        sum = 0
        for example in examples:
            sum = sum + example[featureIndex]
        return sum / len(examples)

    def computeStandardDeviation(self, examples, featureIndex, mean):
        squaredSum = 0
        for example in examples:
            squaredSum = squaredSum + abs(example[featureIndex] - mean)**2
        return math.sqrt(squaredSum / len(examples))

    def standardizeInput(self, input):
        return (input - self.mean) / self.standardDeviation

