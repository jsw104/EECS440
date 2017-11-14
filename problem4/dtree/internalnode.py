from mldata import *
import entropy
from continiousAttributeSplitFinder import *
from boundary import *

class InternalNode:
    
    def __init__(self, inputIndex, feature):
        self.parent = None
        self.inputIndex = int(inputIndex)
        self.feature = feature
        self.featureType = feature.type
        self.boundaryValue = None
        self.children = {}
    
    def setParent(self, parent):
        self.parent = parent
        
    def addChild(self, childNode, key):
        if key in self.children.keys():
            print 'Uh Oh, we are going to overwrite ' + str(self.children[key]) + ' with ' + str(childNode)
        self.children[key] = childNode

    def analyzeSplit(self, examples):
        """
        Return the entropy of using the split on the examples and the resulting binned examples. Does not modify any attributes of self.
        """
        binnedExamples, boundaryValue = self.binExamples(examples)
            
        prospectiveEntropy = 0
        for featureValue in binnedExamples.keys():
            prospectiveEntropy = prospectiveEntropy + (float(sum(example.weight for example in binnedExamples[featureValue])) / float(sum(example.weight for example in examples))) * entropy.entropy_class_label(binnedExamples[featureValue])

        attributeEntropy = entropy.entropy_attribute(examples, self.inputIndex)
        
        return prospectiveEntropy, attributeEntropy, binnedExamples, boundaryValue
    
    def binExamples(self, examples):
        """
        Find the bins if the provided examples are split using this node's feature. Does not modify any attributes of self
        """

        binnedExamples = {}
        bestSplitFeatureValue = -1

        if self.featureType is Feature.Type.BINARY or self.featureType is Feature.Type.NOMINAL:
            for featureValue in self.feature.values:
                binnedExamples[featureValue] = []
                
            for example in examples: 
                binnedExamples[example.inputs[self.inputIndex]].append(example)
    
        elif self.featureType is Feature.Type.CONTINUOUS:
            possibleSplitFinder = ContiniousAttributeSplitFinder()
            sortedFeatureValues, featureValueCounter = possibleSplitFinder.sortFeatureValues(examples, self.inputIndex)
            currentBoundary = Boundary(sortedFeatureValues[0], examples)
            for featureValue in sortedFeatureValues:
                if True in featureValueCounter[featureValue]:
                    currentBoundary.greaterThanTrueWeight = currentBoundary.greaterThanTrueWeight + featureValueCounter[featureValue][True]
                if False in featureValueCounter[featureValue]:
                    currentBoundary.greaterThanFalseWeight = currentBoundary.greaterThanFalseWeight + featureValueCounter[featureValue][False]

            bestSplitFeatureValueIndex = -1
            bestEntropy = -1
            for featureValue in sortedFeatureValues:
                trueCount = 0
                falseCount = 0
                if True in featureValueCounter[featureValue]:
                    trueCount = featureValueCounter[featureValue][True]
                if False in featureValueCounter[featureValue]:
                    falseCount = featureValueCounter[featureValue][False]
                currentBoundary.greaterThanTrueWeight = currentBoundary.greaterThanTrueWeight - trueCount
                currentBoundary.greaterThanFalseWeight = currentBoundary.greaterThanFalseWeight - falseCount
                currentBoundary.lessThanTrueWeight = currentBoundary.lessThanTrueWeight + trueCount
                currentBoundary.lessThanFalseWeight = currentBoundary.lessThanFalseWeight + falseCount
                currentEntropy = currentBoundary.calculateLessThanEntropy() + currentBoundary.calculateGreaterThanEntropy()
                if (bestEntropy < 0 or currentEntropy < bestEntropy):
                    bestEntropy = currentEntropy
                    bestSplitFeatureValueIndex = sortedFeatureValues.index(featureValue)

            bestSplitFeatureValue = sortedFeatureValues[bestSplitFeatureValueIndex + 1] if len(sortedFeatureValues) > bestSplitFeatureValueIndex + 1 else sortedFeatureValues[bestSplitFeatureValueIndex] + 1
            binnedExamples[">="] = []
            binnedExamples["<"] = []

            for example in examples:
                if example.inputs[self.inputIndex] >= bestSplitFeatureValue:
                    binnedExamples[">="].append(example)
                else:
                    binnedExamples["<"].append(example)
        return binnedExamples, bestSplitFeatureValue
        