import sys
import os
from mldata import *
import entropy
from internalnode import *
from continiousAttributeSplitFinder import *
from leafnode import *

class DTree:
    def __init__(self, dataPath, noCrossValidation, maxDepth, useInformationGain):
        if type(dataPath) is not str:
            raise ValueError('The data path must be a string')
        fileName = os.path.basename(dataPath)
        rootDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath[1:-(len(fileName) + 1)])

        self.exampleSet = parse_c45(fileName, rootDirectory)
        self.useCrossValidation = not noCrossValidation
        self.maxDepth = maxDepth
        self.useInformationGain = useInformationGain

#example: python dtree /../testData/spam/spam 1 0 1
def parseCommandLineToTree():
    #the first argument is technically the name of the script so we need 5 args for 4 options.
    if (len(sys.argv) is not 5):
        raise ValueError('You must run with 4 options.')
    dataPath = sys.argv[1]
    noCrossValidation = sys.argv[2]
    maxDepth = int(sys.argv[3])
    if maxDepth == 0: #If the arg is 0, we want to grow the full tree
        maxDepth = -1 #But it's more convenient to represent this as a -1 internally
    useInformationGain = sys.argv[4]

    return DTree(dataPath, noCrossValidation, maxDepth, useInformationGain)
 

def buildTree(examples, schema, possibleSplitNodes, depthRemaining, parentMajorityClass):
    initialClassLabelEntropy = entropy.entropy_class_label(examples)
    
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
    bestNodeEntropy = -1
    bestNodeBinnedExamples = None

    for possibleNode in possibleSplitNodes:
        prospectiveEntropy, binnedExamples = possibleNode.analyzeSplit(examples)
        if bestNodeEntropy < 0 or prospectiveEntropy < bestNodeEntropy:
            bestNodeEntropy = prospectiveEntropy
            bestNode = possibleNode
            bestNodeBinnedExamples = binnedExamples

    #Check for no information gain
    informationGain = initialClassLabelEntropy - bestNodeEntropy
    if not (informationGain > 0):
        return LeafNode(majorityClass, majorityClassFraction) #Base Case
                    
    print 'Selected Split: (Feature Index ' + str(bestNode.featureIndex) + ') ' + bestNode.schema.features[bestNode.featureIndex].name + ' [Entropy=' + str(bestNodeEntropy) + ']'
     
    #Add the child nodes corresponding to this choice
    if depthRemaining > 0:
        depthRemaining = depthRemaining - 1
    
    for bin in bestNodeBinnedExamples.keys():
        newPossibleSplitNodes = list(possibleSplitNodes)
        newPossibleSplitNodes.remove(bestNode)
        #Recurse and add result as child node
        bestNode.addChild(buildTree(bestNodeBinnedExamples[bin], schema, newPossibleSplitNodes, depthRemaining, majorityClass), bin)
        
    return bestNode
        
      
        
dtree = parseCommandLineToTree() 

#Identify the possible candidate tests. We pre-construct all possible nodes we may
# place in the tree for easier bookkeeping later.
possibleNodes = []
examples = dtree.exampleSet.examples
possibleSplitFinder = ContiniousAttributeSplitFinder(examples, dtree.exampleSet.schema)

for featureIndex in range(1,len(dtree.exampleSet.schema.features)-1):    
    feature = dtree.exampleSet.schema.features[featureIndex]
    
    if feature.type is Feature.Type.BINARY or feature.type is Feature.Type.NOMINAL:
        possibleNodes.append(InternalNode(dtree.exampleSet.schema, featureIndex))
        
    elif feature.type is Feature.Type.CONTINUOUS:
        possibleSplitThresholds = possibleSplitFinder.findPossibleSplitValues(featureIndex)
        for possibleSplitThreshold in possibleSplitThresholds:
            possibleNodes.append(InternalNode(dtree.exampleSet.schema, featureIndex, possibleSplitThreshold))

overallMajorityClass, overallMajorityClassFraction = entropy.majority_class(examples)

rootNode = buildTree(examples, dtree.exampleSet.schema, possibleNodes, dtree.maxDepth, overallMajorityClass)
