import sys
import os
import copy
from mldata import *
import entropy
from internalnode import *
from continiousAttributeSplitFinder import *
from leafnode import *
import random
from lib2to3.pytree import Leaf
from copy import deepcopy

class DTree:

    def __init__(self, dataPath, noCrossValidation, maxDepth, useInformationGain):
        self.totalNodesCounter = 0

        if type(dataPath) is not str:
            raise ValueError('The data path must be a string')
        fileName = os.path.basename(dataPath)
        rootDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath[1:-(len(fileName) + 1)])

        #self.exampleSet = parse_c45(fileName, rootDirectory)
        exampleSet = parse_c45(fileName, rootDirectory)
        
        self.examples = exampleSet.examples
        self.schema = exampleSet.schema
        self.useCrossValidation = not noCrossValidation
        self.maxDepth = maxDepth
        self.useInformationGain = useInformationGain
        self.rootNode = None
        
        #Identify the possible candidate tests. We pre-construct all possible nodes we may
        # place in the tree for easier bookkeeping later.
        possibleNodes = []
        possibleSplitFinder = ContiniousAttributeSplitFinder(self.schema)
        
        for featureIndex in range(1,len(self.schema.features)-1):    
            feature = self.schema.features[featureIndex]
            
            if feature.type is Feature.Type.BINARY or feature.type is Feature.Type.NOMINAL:
                possibleNodes.append(InternalNode(self.schema, featureIndex))
                
            elif feature.type is Feature.Type.CONTINUOUS:
                possibleSplitThresholds = possibleSplitFinder.findPossibleSplitValues(self.examples, featureIndex)
                possibleNodes.append(InternalNode(self.schema, featureIndex, possibleSplitThresholds))
        
        overallMajorityClass, overallMajorityClassFraction = entropy.majority_class(self.examples)
        self.rootNode = self._buildTree(self.examples, self.schema, possibleNodes, self.maxDepth, overallMajorityClass)
    
    def evaluateExamples(self, examples):
        numCorrect = 0
        for example in examples:
            node = self.rootNode
            while hasattr(node, 'children'):
                featureValue = example.features[node.featureIndex]
                if node.featureType == Feature.Type.BINARY or node.featureType == Feature.Type.NOMINAL:
                    node = node.children[featureValue]
                
                elif node.featureType == Feature.Type.CONTINUOUS:
                    print 'TODO'
            
            if example.features[-1] == node.classLabel:
                numCorrect = numCorrect + 1
        
        return float(numCorrect)/len(examples)
    
    def countNodes(self):
        if self.rootNode == None:
            return 0
        
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
        return self._findMaxDepth(self.rootNode)-1 #The leaf node doesn't count towards the depth
            
    def _findMaxDepth(self, node):
        maxChildDepth = 0;
        if hasattr(node, 'children'):
            for key, child in node.children.items():
                maxChildDepth = max(maxChildDepth, self._findMaxDepth(child))        
        return maxChildDepth + 1
    
    def getFirstFeatureName(self):
        if self.rootNode == None:
            return None
        return self.rootNode.schema.features[self.rootNode.featureIndex].name
    
    #if a child bin is continuous and we
    def _removeUnnecessaryNodes(self, possibleSplitNodes, bestNode, bestNodeThresholdIndex, bin):
        if bestNode.featureType == Feature.Type.CONTINUOUS:
            if(bin == ">="):
                bestNodeCopy = copy.deepcopy(bestNode)
                bestNodeCopy.possibleSplitThresholds = list(bestNode.possibleSplitThresholds[bestNodeThresholdIndex + 1:])
                possibleSplitNodes.remove(bestNode)
                if(len(bestNodeCopy.possibleSplitThresholds) > 0):
                    possibleSplitNodes.append(bestNodeCopy)
            elif(bin == "<"):
                bestNodeCopy = copy.deepcopy(bestNode)
                bestNodeCopy.possibleSplitThresholds = list(bestNode.possibleSplitThresholds[
                                                   :bestNodeThresholdIndex - len(bestNode.possibleSplitThresholds)])
                possibleSplitNodes.remove(bestNode)
                if(len(bestNodeCopy.possibleSplitThresholds) > 0):
                    possibleSplitNodes.append(bestNodeCopy)

        else:
            possibleSplitNodes.remove(bestNode)
    
    def _buildTree(self, examples, schema, possibleSplitNodes, depthRemaining, parentMajorityClass):
        initialClassLabelEntropy = entropy.entropy_class_label(examples)
        print('len(possibleSplitNodes)=' + str(len(possibleSplitNodes)))
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
        bestThresholdIndex = -1
    
        for i in range(0, len(possibleSplitNodes)):
            if i % 50 == 0:
                print('i=' + str(i))
            possibleSplitNode = possibleSplitNodes[i]
            
            splitClassLabelEntropy, attributeEntropy, binnedExamples, thresholdIndex = possibleSplitNode.analyzeSplit(examples)
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
                        bestThresholdIndex = thresholdIndex          
    
        #Check for no information gain
        if not (bestNodeInformationGain > 0):
            return LeafNode(majorityClass, majorityClassFraction) #Base Case
        
        cloneBestNode = InternalNode(bestNode.schema, bestNode.featureIndex, bestNode.possibleSplitThresholds)
        cloneBestNode.chosenThresholdIndex = bestThresholdIndex 
                        
        if self.useInformationGain:
            print 'Selected Split: (Feature Index ' + str(bestNode.featureIndex) + ') ' + bestNode.schema.features[bestNode.featureIndex].name + ' [InformationGain=' + str(bestNodeInformationGain) + ']'
        else:
            print 'Selected Split: (Feature Index ' + str(bestNode.featureIndex) + ') ' + bestNode.schema.features[bestNode.featureIndex].name + ' [GainRatio=' + str(bestNodeGainRatio) + ']' 
        
        #Add the child nodes corresponding to this choice
        if depthRemaining > 0:
            depthRemaining = depthRemaining - 1
    
        for bin, binnedExamples in bestNodeBinnedExamples.items():
            if len(binnedExamples) > 0:
                newPossibleSplitNodes = list(possibleSplitNodes)
                self._removeUnnecessaryNodes(newPossibleSplitNodes, bestNode, bestThresholdIndex, bin)
                
                #Recurse and add result as child node
                childNode = self._buildTree(binnedExamples, schema, newPossibleSplitNodes, depthRemaining, majorityClass)
                self.totalNodesCounter = self.totalNodesCounter + 1
                cloneBestNode.addChild(childNode, bin)          
              
        return cloneBestNode    
    
#example: python dtree /../testData/spam/spam 1 0 1
def parseCommandLineToTree():
    #sys.argv[0] is the name of the script so we need 5 args for 4 options.
    if (len(sys.argv) is not 5):
        raise ValueError('You must run with 4 options.')
    dataPath = sys.argv[1]
    noCrossValidation = int(sys.argv[2]) == 0
    maxDepth = int(sys.argv[3])
    if maxDepth == 0: #If the arg is 0, we want to grow the full tree
        maxDepth = -1 #But it's more convenient to represent this as a -1 internally
    useInformationGain = int(sys.argv[4]) == 0

    return DTree(dataPath, noCrossValidation, maxDepth, useInformationGain)        
      
#MAIN        
dtree = parseCommandLineToTree()
print '======'
print 'Accuracy: ' + '!!!! TODO !!!!' #See work started in Dtree.evaluateExamples()
print 'Size: ' + str(dtree.countNodes()[0]) #Not counting leaf nodes
print 'Maximum Depth: ' + str(dtree.findMaxDepth())
print 'First Feature: ' + str(dtree.getFirstFeatureName())