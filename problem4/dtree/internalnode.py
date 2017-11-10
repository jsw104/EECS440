from mldata import *
import entropy
from continiousAttributeSplitFinder import *
from boundary import *

class InternalNode:
    
    def __init__(self, schema, featureIndex):
        self.parent = None
        self.schema = schema
        self.featureIndex = int(featureIndex)
        self.featureType = self.schema.features[self.featureIndex].type
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
            prospectiveEntropy = prospectiveEntropy + (float(len(binnedExamples[featureValue])) / len(examples)) * entropy.entropy_class_label(binnedExamples[featureValue])

        attributeEntropy = entropy.entropy_attribute(examples, self.featureIndex)
        
        return prospectiveEntropy, attributeEntropy, binnedExamples, boundaryValue
    
    def binExamples(self, examples):
        """
        Find the bins if the provided examples are split using this node's feature. Does not modify any attributes of self
        """

        feature = self.schema.features[self.featureIndex]
        binnedExamples = {}
        bestSplitFeatureValue = -1

        if feature.type is Feature.Type.BINARY or feature.type is Feature.Type.NOMINAL:
            for featureValue in feature.values:
                binnedExamples[featureValue] = []
                
            for example in examples: 
                binnedExamples[example.features[self.featureIndex]].append(example)
    
        elif feature.type is Feature.Type.CONTINUOUS:
            possibleSplitFinder = ContiniousAttributeSplitFinder(self.schema)
            sortedFeatureValues, featureValueCounter = possibleSplitFinder.sortFeatureValues(examples, self.featureIndex)
            currentBoundary = Boundary(sortedFeatureValues[0], len(examples))
            for featureValue in sortedFeatureValues:
                if True in featureValueCounter[featureValue]:
                    currentBoundary.greaterThanTrueCount = currentBoundary.greaterThanTrueCount + featureValueCounter[featureValue][True]
                if False in featureValueCounter[featureValue]:
                    currentBoundary.greaterThanFalseCount = currentBoundary.greaterThanFalseCount + featureValueCounter[featureValue][False]

            bestSplitFeatureValueIndex = -1
            bestEntropy = -1
            for featureValue in sortedFeatureValues:
                trueCount = 0
                falseCount = 0
                if True in featureValueCounter[featureValue]:
                    trueCount = featureValueCounter[featureValue][True]
                if False in featureValueCounter[featureValue]:
                    falseCount = featureValueCounter[featureValue][False]
                currentBoundary.greaterThanTrueCount = currentBoundary.greaterThanTrueCount - trueCount
                currentBoundary.greaterThanFalseCount = currentBoundary.greaterThanFalseCount - falseCount
                currentBoundary.lessThanTrueCount = currentBoundary.lessThanTrueCount + trueCount
                currentBoundary.lessThanFalseCount = currentBoundary.lessThanFalseCount + falseCount
                currentEntropy = currentBoundary.calculateLessThanEntropy() + currentBoundary.calculateGreaterThanEntropy()
                if (bestEntropy < 0 or currentEntropy < bestEntropy):
                    bestEntropy = currentEntropy
                    bestSplitFeatureValueIndex = sortedFeatureValues.index(featureValue)

            bestSplitFeatureValue = sortedFeatureValues[bestSplitFeatureValueIndex + 1] if len(sortedFeatureValues) > bestSplitFeatureValueIndex + 1 else sortedFeatureValues[bestSplitFeatureValueIndex] + 1
            binnedExamples[">="] = []
            binnedExamples["<"] = []

            for example in examples:
                if example[self.featureIndex] >= bestSplitFeatureValue:
                    binnedExamples[">="].append(example)
                else:
                    binnedExamples["<"].append(example)
        return binnedExamples, bestSplitFeatureValue
        