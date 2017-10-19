from mldata import *
from bayesianBin import *
class BayesianFeature:
    def __init__(self, examples, featureIndex, featureType, numberOfBins):
        self.countersForClassification = {}
        self.featureIndex = featureIndex
        self.featureType = featureType
        self.numberOfBins = numberOfBins
        self.examples = examples
        if self.featureType == Feature.Type.CONTINUOUS:
            self.bins = {}
            self.createBins(examples)
        self.calculateProbabilities(examples)

    def calculateProbabilities(self, examples):
        self.determinePossibleClassifications(examples)
        if self.featureType == Feature.Type.CONTINUOUS:
            for example in examples:
                binIndex = self.binIndexForValue(example[self.featureIndex])
                if binIndex not in self.countersForClassification[example[-1]]:
                    self.countersForClassification[example[-1]][binIndex] = 0
                else:
                    self.countersForClassification[example[-1]][binIndex] = self.countersForClassification[example[-1]][binIndex] + 1
        else:
            for example in examples:
                if example[self.featureIndex] not in self.countersForClassification[example[-1]]:
                    self.countersForClassification[example[-1]][example[self.featureIndex]] = 0
                else:
                    self.countersForClassification[example[-1]][example[self.featureIndex]] = self.countersForClassification[example[-1]][example[self.featureIndex]] + 1

    def determinePossibleClassifications(self, examples):
        for example in examples:
            if example[-1] not in self.countersForClassification:
                self.countersForClassification[example[-1]] = {}

    def binIndexForValue(self, value):
        for i in range(0, len(self.bins)):
            if self.bins[i].belongsInBin(value):
                return i

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
                self.bins[i] = BayesianBin(min + stepSize * i, min + stepSize * (i + 1), i == self.numberOfBins - 1)

    def probabilityOfAttributeGivenClassification(self, attribute, classification):
        if self.featureType == Feature.Type.CONTINUOUS:
            binIndex = self.binIndexForValue(attribute)
            return self.countersForClassification[classification][binIndex]/len(self.examples)
        return self.countersForClassification[classification][attribute]/len(self.examples)