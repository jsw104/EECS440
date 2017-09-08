import sys
import os
import mldata

class DTree:
    def __init__(self, dataPath, noCrossValidation, maxDepth, useInformationGain):
        if type(dataPath) is not str:
            raise ValueError('The data path must be a string')
        fileName = os.path.basename(dataPath)
        rootDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath[1:-(len(fileName) + 1)])

        self.exampleSet = mldata.parse_c45(fileName, rootDirectory)
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
print dtree.exampleSet