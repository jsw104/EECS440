import sys
import os
from mldata import *
import entropy
from internalnode import *

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

def parseCommandLineToTree():
    #the first argument is the name of the script.
    if (len(sys.argv) is not 5):
        raise ValueError('You must run with 4 options.')
    dataPath = sys.argv[1]
    noCrossValidation = sys.argv[2]
    maxDepth = sys.argv[3]
    useInformationGain = sys.argv[4]

    return DTree(dataPath, noCrossValidation, maxDepth, useInformationGain)
    
        
dtree = parseCommandLineToTree()



### The following will eventually be put in a function and invoked recursively.
### But first let's get the entropy calculations and split selection ironed down. 

#Calculate Initial Entropy of Class Labels
initialClassLabelEntropy = entropy.entropy_class_label(dtree.exampleSet)

#Identify the possible candidate tests. We pre-construct all possible nodes we may
# place in the tree for easier bookkeeping later.
possibleNodes = []
for featureIndex in range(1,len(dtree.exampleSet.schema.features)-1):    
    feature = dtree.exampleSet.schema.features[featureIndex]
    
    if feature.type is Feature.Type.BINARY or feature.type is Feature.Type.NOMINAL:
        possibleNodes.append(InternalNode(dtree.exampleSet.schema, featureIndex))
        
    elif feature.type is Feature.Type.CONTINUOUS:
        #TODO
        print 'CONTINUOUS features not yet implemented'

#Of the the decision nodes we can choose, identify the one with the lowest entropy after splitting
bestNode = None
bestNodeEntropy = -1
for possibleNode in possibleNodes:
    prospectiveEntropy, binnedExamples = possibleNode.analyzeSplit(dtree.exampleSet.examples)
    if bestNodeEntropy < 0 or prospectiveEntropy < bestNodeEntropy:
        bestNodeEntropy = prospectiveEntropy
        bestNode = possibleNode
        
print 'Selected Split: (Feature Index ' + str(bestNode.featureIndex) + ') ' + bestNode.schema.features[bestNode.featureIndex].name + ' [Entropy=' + str(bestNodeEntropy) + ']'

#Add best node to tree here
