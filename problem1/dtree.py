import sys
import os
import copy
from mldata import *
import entropy
from internalnode import *
from continiousAttributeSplitFinder import *
from leafnode import *
import random
from Queue import *

class DTree:

    def __init__(self, dataPath, useCrossValidation, maxDepth, useInformationGain):

        if type(dataPath) is not str:
            raise ValueError('The data path must be a string')
        fileName = os.path.basename(dataPath)
        rootDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath[0:-(len(fileName) + 1)])

        exampleSet = parse_c45(fileName, rootDirectory)

        self.schema = exampleSet.schema
        self.useCrossValidation = useCrossValidation
        self.maxDepth = maxDepth
        self.useInformationGain = useInformationGain
        self.rootNode = None
        self.accuracy = 0

        self.examples = exampleSet.examples

        #Build the tree using the full set of examples. The final tree is built the same
        #way even if cross-validating to get a better accuracy estimate.
        self.rootNode = self.createRootNode(self.examples)
        
        #Get accuracy estimate
        if self.useCrossValidation:
            foldArray = self.constructFolds(self.examples)
            for i in range(0, len(foldArray)):
                testExamples = []
                trainingExamples = []
                testExamples = foldArray[i]
                for j in range(0, len(foldArray)):
                    if j != i:
                        trainingExamples = trainingExamples + foldArray[j]
                foldTreeRootNode = self.createRootNode(trainingExamples)
                self.accuracy = self.accuracy + self.evaluateExamples(foldTreeRootNode, testExamples)
                self.printTree(foldTreeRootNode)
            self.accuracy = self.accuracy/len(foldArray)
            
        else:
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
                possibleNodes.append(InternalNode(self.schema, featureIndex))

        overallMajorityClass, overallMajorityClassFraction = entropy.majority_class(trainingExamples)
        return self._buildTree(trainingExamples, self.schema, possibleNodes, self.maxDepth, overallMajorityClass)

    def printTree(self, rootNode):
        q = Queue(maxsize=0)
        next_level = 1
        q.put([rootNode, next_level])
        while (q.empty() is not True):
            n = q.get()
            if next_level == n[1]:
                next_level = next_level + 1
                print "\n"
            if isinstance(n[0], LeafNode):
                print str(n[0].classLabel) + "(count:" + str(n[0].evaluationCount) + ")" + "      ",
                continue
            else:
                print n[0].schema.features[n[0].featureIndex].name + "(count:" + str(n[0].evaluationCount) + "boundary:" + str(n[0].boundaryValue) + ")" + "      ",
            for childNode in n[0].children.itervalues():
                q.put([childNode, n[1] + 1])


    def evaluateExamples(self, rootNode, examples):
        numCorrect = 0
        for example in examples:
            node = rootNode
            while hasattr(node, 'children'):
                featureValue = example.features[node.featureIndex]
                node.incrementEvaluationCount()
                if node.featureType == Feature.Type.BINARY or node.featureType == Feature.Type.NOMINAL:
                    node = node.children[featureValue]
                
                elif node.featureType == Feature.Type.CONTINUOUS:
                    if example[node.featureIndex] >= node.boundaryValue:
                        node = node.children['>=']
                    else:
                        node = node.children['<']
            node.incrementEvaluationCount()
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

#print '======' 
#print str(dtree.evaluateExamples(dtree.rootNode, dtree.examples))
#print '======'

print 'Accuracy: ' + str(dtree.accuracy)
countInternalNodes, countLeafNodes = dtree.countNodes()
print 'Size: ' + str(countInternalNodes) #Not counting leaf nodes
print 'Maximum Depth: ' + str(dtree.findMaxDepth())
print 'First Feature: ' + str(dtree.getFirstFeatureName())
dtree.printTree()