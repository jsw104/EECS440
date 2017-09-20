from mldata import *
import entropy
from continiousAttributeSplitFinder import *
from boundary import *

class InternalNode:
    
    def __init__(self, schema, featureIndex, possibleSplitThresholds=None):
        self.parent = None
        self.schema = schema
        self.featureIndex = int(featureIndex)
        self.featureType = self.schema.features[self.featureIndex].type
        self.possibleSplitThresholds = possibleSplitThresholds
        self.chosenThresholdIndex = None
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
        binnedExamples, bestThresholdIndex = self.binExamples(examples)
            
        prospectiveEntropy = 0
        for featureValue in binnedExamples.keys():
            prospectiveEntropy = prospectiveEntropy + (float(len(binnedExamples[featureValue])) / len(examples)) * entropy.entropy_class_label(binnedExamples[featureValue])

        attributeEntropy = entropy.entropy_attribute(examples, self.featureIndex)
        
        return prospectiveEntropy, attributeEntropy, binnedExamples, bestThresholdIndex
    
    def binExamples(self, examples):
        """
        Find the bins if the provided examples are split using this node's feature. Does not modify any attributes of self
        """

        feature = self.schema.features[self.featureIndex]
        binnedExamples = {}
        bestThresholdIndex = -1
        
        if feature.type is Feature.Type.BINARY or feature.type is Feature.Type.NOMINAL:
            for featureValue in feature.values:
                binnedExamples[featureValue] = ExampleSet(self.schema)
                
            for example in examples: 
                binnedExamples[example.features[self.featureIndex]].append(example)
    
        elif feature.type is Feature.Type.CONTINUOUS:
            possibleBoundaries = []

            for threshold in self.possibleSplitThresholds:
                possibleBoundaries.append(Boundary(threshold))

            for example in examples:
                for possibleBoundary in possibleBoundaries:
                    if float(example[self.featureIndex]) >= possibleBoundary.boundaryValue:
                        possibleBoundary.greaterThanOrEqualExamples.append(example)
                    else:
                        possibleBoundary.lessThanExamples.append(example)

            bestEntropy = -1
            bestBoundary = None
            for possibleBoundary in possibleBoundaries:
                prospectiveEntropy = 0
                prospectiveEntropy = prospectiveEntropy + (float(len(possibleBoundary.greaterThanOrEqualExamples)) / len(
                    examples)) * entropy.entropy_class_label(possibleBoundary.greaterThanOrEqualExamples)
                prospectiveEntropy = prospectiveEntropy + (float(len(possibleBoundary.lessThanExamples)) / len(
                    examples)) * entropy.entropy_class_label(possibleBoundary.lessThanExamples)
                if(bestEntropy < 0 or prospectiveEntropy < bestEntropy):
                    bestEntropy = prospectiveEntropy
                    bestBoundary = possibleBoundary
                    bestThresholdIndex = possibleBoundaries.index(possibleBoundary)

            binnedExamples[">="] = bestBoundary.greaterThanOrEqualExamples
            binnedExamples["<"] = bestBoundary.lessThanExamples
        
        return binnedExamples, bestThresholdIndex
        