from mldata import *
from bayesianBin import *
class BayesianFeature:
    def __init__(self, examples, featureIndex, featureType, numberOfBins):
        self.countersForClassification = {}
        self.featureIndex = featureIndex
        self.featureType = featureType
        self.numberOfBins = numberOfBins
        if self.featureType == Feature.Type.CONTINUOUS:
            self.bins = {}
            self.createBins(examples)
        self.calculateProbabilities(examples)

    def calculateProbabilities(self, examples):
        print "do something"

    def createBins(self, examples):
        if self.featureType == Feature.Type.CONTINUOUS:
            min = examples[0][self.featureIndex]
            max = examples[0][self.featureIndex]
            for example in examples:
                if example[self.featureIndex] < min:
                    min = example[self.featureIndex]
                elif example[self.featureIndex] > max:
                    max = example[self.featureIndex]
            stepSize = (float(max) - float(min)) / float(self.numberOfBins)
            for i in range(0, self.numberOfBins):
                self.bins[i] = BayesianBin(min + stepSize * i, min + stepSize * (i + 1))
