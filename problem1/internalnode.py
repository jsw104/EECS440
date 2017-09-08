from mldata import *
import entropy

class InternalNode:
    
    def __init__(self, schema, featureIndex, boundaryValue=None):
        self.parent = None
        self.schema = schema
        self.featureIndex = featureIndex
        self.boundaryValue = boundaryValue
        self.children = []
    
    def setParent(self, parent):
        self.parent = parent
        
    """Return the entropy of using the split on the examples and the resulting binned examples. Does not modify any attributes of self."""
    def analyzeSplit(self, examples):
        for featureIndex in range(1,len(self.schema.features)-1):
            binnedExamples = self.binExamples(examples)
            
        prospectiveEntropy = 0
        for featureValue in binnedExamples.keys():
            prospectiveEntropy = prospectiveEntropy + (float(len(binnedExamples[featureValue])) / len(examples)) * entropy.entropy_class_label(binnedExamples[featureValue])
        
        return prospectiveEntropy, binnedExamples
    
    """Find the bins if the provided examples are split using this node's feature. Does not modify any attributes of self"""
    def binExamples(self, examples):
        feature = self.schema.features[self.featureIndex]
    
        binnedExamples = {}
        
        if feature.type is Feature.Type.BINARY or feature.type is Feature.Type.NOMINAL:
            for featureValue in feature.values:
                binnedExamples[featureValue] = ExampleSet(self.schema)
                
            for example in examples:
                binnedExamples[example.features[self.featureIndex]].append(example)
    
        elif feature.type is Feature.Type.CONTINUOUS:
            #TODO
            print 'CONTINUOUS features not yet implemented'
        
        return binnedExamples 
        