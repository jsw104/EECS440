import sys
import os
import copy
from mldata import *
import entropy
from internalnode import *
from continiousAttributeSplitFinder import *
from leafnode import *
import random

class DTree:

    def __init__(self, schema, maxDepth, useInformationGain):
        self.schema = schema
        self.useInformationGain = useInformationGain
        self.maxDepth = maxDepth
        self.rootNode = None
    
    def train(self, trainingExamples):
        # Identify the possible candidate tests. We pre-construct all possible nodes we may
        # place in the tree for easier bookkeeping later.
        possibleNodes = []
        possibleSplitFinder = ContiniousAttributeSplitFinder(self.schema)

        for featureIndex in range(1, len(self.schema.features) - 1):
            feature = self.schema.features[featureIndex]

            if feature.type is Feature.Type.BINARY or feature.type is Feature.Type.NOMINAL:
                possibleNodes.append(InternalNode(self.schema, featureIndex))

            elif feature.type is Feature.Type.CONTINUOUS:
                possibleNodes.append(InternalNode(self.schema, featureIndex))

        overallMajorityClass, overallMajorityClassFraction = entropy.majority_class(trainingExamples)
        self.rootNode = self._buildTree(trainingExamples, self.schema, possibleNodes, self.maxDepth, overallMajorityClass)

    def evaluateExamples(self, examples):
        if self.rootNode is None:
            raise RuntimeError('Tree not yet built; run train first')
         
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        targetOutputPairs = [] 
            
        numCorrect = 0
        for example in examples:
            node = self.rootNode
            while hasattr(node, 'children'):
                featureValue = example.features[node.featureIndex]
                if node.featureType == Feature.Type.BINARY or node.featureType == Feature.Type.NOMINAL:
                    node = node.children[featureValue]
                
                elif node.featureType == Feature.Type.CONTINUOUS:
                    if example[node.featureIndex] >= node.boundaryValue:
                        node = node.children['>=']
                    else:
                        node = node.children['<']
            
            if example.features[-1] and node.classLabel:
                tp = tp + 1
            elif (not example.features[-1]) and (not node.classLabel):
                tn = tn + 1
            elif (not example.features[-1]) and node.classLabel:
                fp = fp + 1 
            elif example.features[-1] and not node.classLabel:
                fn = fn + 1
            targetOutputPairs.append((example.features[-1],node.classLabel))
            
        return tp, fp, tn, fn, targetOutputPairs


    def countNodes(self):
        if self.rootNode == None:
            return 0, 0
        
        internalNodeCount = 0
        leafNodeCount = 0
        queue = []
        queue.append(self.rootNode)
        
        while(len(queue) > 0):
            node = queue.pop(0)
            if hasattr(node, 'children'):
                internalNodeCount = internalNodeCount + 1
                for bin, childNode in node.children.items():
                    queue.append(childNode)
            else:
                leafNodeCount = leafNodeCount + 1
                
        return internalNodeCount, leafNodeCount
    
    def findMaxDepth(self):
        if self.rootNode == None:
            return 0    
        return self._findMaxDepth(self.rootNode)
            
    def _findMaxDepth(self, node):
        maxChildDepth = 0;
        if hasattr(node, 'children'):
            for key, child in node.children.items():
                maxChildDepth = max(maxChildDepth, self._findMaxDepth(child))
        else:
            return 0  #Base Case; 0 since the leaf node doesn't count towards the depth
                
        return maxChildDepth + 1
    
    def getFirstFeatureName(self):
        if self.rootNode == None:
            return None
        return self.rootNode.schema.features[self.rootNode.featureIndex].name
    
    #if a child bin is continuous and we
    def _removeUnnecessaryNodes(self, possibleSplitNodes, bestNode):
        if bestNode.featureType == Feature.Type.NOMINAL:
            possibleSplitNodes.remove(bestNode)
            #print 'removed ' + str(bestNode.schema.features[bestNode.featureIndex].name)
    
    def _buildTree(self, examples, schema, possibleSplitNodes, depthRemaining, parentMajorityClass):
        initialClassLabelEntropy = entropy.entropy_class_label(examples)
        #print('len(possibleSplitNodes)=' + str(len(possibleSplitNodes)))
        #Check for empty node
        if len(examples) == 0:
            return LeafNode(parentMajorityClass, 0.0) #Base Case
            
        majorityClass, majorityClassFraction = entropy.majority_class(examples)
            
        #Check for pure node
        if initialClassLabelEntropy == 0:
            classLabel = examples[0].features[-1]
            return LeafNode(classLabel, 1.0) #Base Case
        
        #Check for no depth remaining
        if depthRemaining == 0:
            return LeafNode(majorityClass, majorityClassFraction) #Base Case
        
        #Of the the decision nodes we can choose, identify the one with the lowest entropy after splitting
        bestNode = None
        bestNodeInformationGain = -1
        bestNodeGainRatio = -1
        bestNodeBinnedExamples = None
        bestBoundaryValue = -1

        for i in range(0, len(possibleSplitNodes)):
            #if i % 50 == 0:
            #    print('i=' + str(i))
            possibleSplitNode = possibleSplitNodes[i]
            
            splitClassLabelEntropy, attributeEntropy, binnedExamples, boundaryValue = possibleSplitNode.analyzeSplit(examples)
            informationGain = initialClassLabelEntropy - splitClassLabelEntropy
            
            if informationGain > 0 and attributeEntropy > 0:
                gainRatio = informationGain / attributeEntropy      
                if (
                        (self.useInformationGain and (bestNodeInformationGain < 0 or informationGain > bestNodeInformationGain)) or
                        ((not self.useInformationGain) and (bestNodeInformationGain < 0 or gainRatio > bestNodeGainRatio))
                    ):
                        bestNode = possibleSplitNode
                        bestNodeInformationGain = informationGain
                        bestNodeGainRatio = gainRatio
                        bestNodeBinnedExamples = binnedExamples
                        bestBoundaryValue = boundaryValue
    
        #Check for no information gain
        if not (bestNodeInformationGain > 0):
            return LeafNode(majorityClass, majorityClassFraction) #Base Case

        cloneBestNode = InternalNode(bestNode.schema, bestNode.featureIndex)
        cloneBestNode.boundaryValue = bestBoundaryValue
                        
        #if self.useInformationGain:
        #    print 'Selected Split: (Feature Index ' + str(bestNode.featureIndex) + ') ' + bestNode.schema.features[bestNode.featureIndex].name + ' [InformationGain=' + str(bestNodeInformationGain) + '] ' + bestNode.schema.features[bestNode.featureIndex].type
        #else:
        #    print 'Selected Split: (Feature Index ' + str(bestNode.featureIndex) + ') ' + bestNode.schema.features[bestNode.featureIndex].name + ' [GainRatio=' + str(bestNodeGainRatio) + '] ' + bestNode.schema.features[bestNode.featureIndex].type 
        
        #Add the child nodes corresponding to this choice
        if depthRemaining > 0:
            depthRemaining = depthRemaining - 1
    
        for bin, binnedExamples in bestNodeBinnedExamples.items():
            newPossibleSplitNodes = list(possibleSplitNodes)
            self._removeUnnecessaryNodes(newPossibleSplitNodes, bestNode)
            
            #Recurse and add result as child node
            childNode = self._buildTree(binnedExamples, schema, newPossibleSplitNodes, depthRemaining, majorityClass)
            cloneBestNode.addChild(childNode, bin)          
              
        return cloneBestNode    
