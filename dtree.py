import sys

class DTree:
    class TreeConfiguration:
        def __init__(self, dataPath, noCrossValidation, maxDepth, useInformationGain):
            if type(dataPath) is not str:
                raise ValueError('The data path must be a string')
            self.dataPath = dataPath
            self.useCrossValidation = not noCrossValidation
            self.maxDepth = maxDepth
            self.useInformationGain = useInformationGain

    def __init__(self, treeConfiguration):
        self.treeConfiguration = treeConfiguration


if (len(sys.argv) is not 5):
    raise ValueError('You must run with 4 options.')
dataPath = sys.argv[1]
noCrossValidation = sys.argv[2]
maxDepth = sys.argv[3]
useInformationGain = sys.argv[4]

treeConfiguration = DTree.TreeConfiguration(sys.argv[2], sys.argv[3], sys.argv[4])
dtree = DTree(treeConfiguration)
print dtree.treeConfiguration