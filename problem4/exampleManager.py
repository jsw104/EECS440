import random
from mldata import Example
from exampleNormalization import *

class ExampleManager:
    def __init__(self, examples, useCrossValidation):
        self.unfoldedExamples = examples
        self.numberOfFolds = 1
        if useCrossValidation:
            self.numberOfFolds = 5
            self.foldArray = self.__constructFolds(examples, self.numberOfFolds)

    def __constructFolds(self, examples, numberOfFolds):
        trueClassificationArray, falseClassificationArray = self.__partitionByClass(examples)
        random.shuffle(trueClassificationArray)
        random.shuffle(falseClassificationArray)
        foldArray = [None] * numberOfFolds
        for i in range(0, numberOfFolds):
            foldArray[i] = []
            for j in range((len(trueClassificationArray) * i) / numberOfFolds,
                           (len(trueClassificationArray) * (i + 1)) / numberOfFolds - 1):
                foldArray[i].append(trueClassificationArray[j])
            for j in range((len(falseClassificationArray) * i) / numberOfFolds,
                           (len(falseClassificationArray) * (i + 1)) / numberOfFolds - 1):
                foldArray[i].append(falseClassificationArray[j])
        return foldArray

    def __partitionByClass(self, examples):
        trueClassificationArray = []
        falseClassificationArray = []
        for example in examples:
            if isinstance(example, NormalizedExample):
                trueClassificationArray.append(example) if (example.target == True) else falseClassificationArray.append(example)
            else: #for mldata.Example types
                trueClassificationArray.append(example) if example[-1] else falseClassificationArray.append(example)
        return trueClassificationArray, falseClassificationArray

    def getCrossValidationExamples(self, testingFoldIndex):
        trainingExamples = []
        testingExamples = self.foldArray[testingFoldIndex]
        for i in range(0, len(self.foldArray)):
            if i != testingFoldIndex:
                trainingExamples = trainingExamples + self.foldArray[i]
        return trainingExamples, testingExamples

    def getUnfoldedExamples(self):
        return self.unfoldedExamples, self.unfoldedExamples

    def numFolds(self):
        return self.numberOfFolds


