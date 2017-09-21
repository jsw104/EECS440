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

    def __init__(self, dataPath, useCrossValidation, maxDepth, useInformationGain):
        self.totalNodesCounter = 0

        if type(dataPath) is not str:
            raise ValueError('The data path must be a string')
        fileName = os.path.basename(dataPath)
        rootDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath[1:-(len(fileName) + 1)])

        #self.exampleSet = parse_c45(fileName, rootDirectory)
        exampleSet = parse_c45(fileName, rootDirectory)

        self.schema = exampleSet.schema
        self.useCrossValidation = useCrossValidation
        self.maxDepth = maxDepth
        self.useInformationGain = useInformationGain
        self.rootNode = None
        self.accuracy = 0

        self.examples = exampleSet.examples

        if self.useCrossValidation:
            foldArray = self.constructFolds(self.examples)
            for i in range(0, len(foldArray)):
                testExamples = []
                trainingExamples = []
                testExamples = foldArray[i]
                for j in range(0, len(foldArray)):
                    if j != i:
                        trainingExamples = trainingExamples + foldArray[j]
                rootNode = self.createRootNode(trainingExamples)
                self.accuracy = self.accuracy + self.evaluateExamples(rootNode, testExamples)
            self.accuracy = self.accuracy/len(foldArray)
            
        else:
            self.rootNode = self.createRootNode(self.examples)
            self.accuracy = self.evaluateExamples(self.rootNode, self.examples)
            
    def createRootNode(self, trainingExamples):
        # Identify the possible candidate tests. We pre-construct all possible nodes we may
        # place in the tree for easier bookkeeping later.
        possibleNodes = []
        possibleSplitFinder = ContiniousAttributeSplitFinder(self.schema)

        for featureIndex in range(1, len(self.schema.features) - 1):
            feature = self.schema.features[featureIndex]

            if feature.type is Feature.Type.BINARY or feature.type is Feature.Type.NOMINAL:
                possibleNodes.append(InternalNode(self.schema, featureIndex))

            elif feature.type is Feature.Type.CONTINUOUS:
                possibleSplitThresholds = possibleSplitFinder.findPossibleSplitValues(trainingExamples, featureIndex)
                possibleNodes.append(InternalNode(self.schema, featureIndex, possibleSplitThresholds))

        overallMajorityClass, overallMajorityClassFraction = entropy.majority_class(trainingExamples)
        return self._buildTree(trainingExamples, self.schema, possibleNodes, self.maxDepth, overallMajorityClass)

    def evaluateExamples(self, rootNode, examples):
        numCorrect = 0
        for example in examples:
            node = rootNode
            while hasattr(node, 'children'):
                featureValue = example.features[node.featureIndex]
                if node.featureType == Feature.Type.BINARY or node.featureType == Feature.Type.NOMINAL:
                    node = node.children[featureValue]
                
                elif node.featureType == Feature.Type.CONTINUOUS:
                    splitThresh = node.possibleSplitThresholds[node.chosenThresholdIndex]
                    if example[node.chosenThresholdIndex] >= splitThresh:
                        node = node.children['>=']
                    else:
                        node = node.children['<']
            
            if example.features[-1] == node.classLabel:
                numCorrect = numCorrect + 1
        
        return float(numCorrect)/len(examples)

    def constructFolds(self, examples):
        trueClassificationArray, falseClassificationArray = self.partitionByClass(examples)
        random.seed(12345)
        random.shuffle(trueClassificationArray)
        random.shuffle(falseClassificationArray)
        numberOfFolds = 5
        foldArray = [None] * numberOfFolds
        for i in range(0,numberOfFolds):
            foldArray[i] = []
            for j in range((len(trueClassificationArray) * i)/numberOfFolds , (len(trueClassificationArray) * (i+1))/numberOfFolds - 1):
                foldArray[i].append(trueClassificationArray[j])
            for j in range((len(falseClassificationArray) * i)/numberOfFolds , (len(falseClassificationArray) * (i+1))/numberOfFolds - 1):
                foldArray[i].append(falseClassificationArray[j])
        return foldArray

    def partitionByClass(self, examples):
        trueClassificationArray = []
        falseClassificationArray = []
        for example in examples:
            trueClassificationArray.append(example) if (example[-1] == True) else falseClassificationArray.append(example)
        return trueClassificationArray, falseClassificationArray


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
    useCrossValidation = int(sys.argv[2]) == 0
    maxDepth = int(sys.argv[3])
    if maxDepth == 0: #If the arg is 0, we want to grow the full tree
        maxDepth = -1 #But it's more convenient to represent this as a -1 internally
    useInformationGain = int(sys.argv[4]) == 0

    return DTree(dataPath, useCrossValidation, maxDepth, useInformationGain)
      
#MAIN        
dtree = parseCommandLineToTree()
print '======'
print 'Accuracy: ' + str(dtree.accuracy)
countInternalNodes, countLeafNodes = dtree.countNodes()
print 'Size: ' + str(countInternalNodes) #Not counting leaf nodes
print 'Maximum Depth: ' + str(dtree.findMaxDepth())
print 'First Feature: ' + str(dtree.getFirstFeatureName())