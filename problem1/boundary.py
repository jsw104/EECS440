import math
class Boundary:
    def __init__(self, boundaryValue, totalNumberExamples):
        self.totalNumberExamples = totalNumberExamples
        self.lessThanTrueCount = 0
        self.greaterThanTrueCount = 0
        self.lessThanFalseCount = 0
        self.greaterThanFalseCount = 0
        self.boundaryValue = boundaryValue

    def calculateLessThanEntropy(self):
        if (self.lessThanTrueCount + self.lessThanFalseCount) == 0:
            return 0
        trueProbability = float(self.lessThanTrueCount / float(self.lessThanFalseCount + self.lessThanTrueCount))
        falseProbability = 1.0 - trueProbability
        trueEntropy = 0 if trueProbability == 0 else trueProbability*math.log(trueProbability,2)
        falseEntropy = 0 if falseProbability == 0 else falseProbability*math.log(falseProbability,2)

        return -1.0 * (self.lessThanTrueCount + self.lessThanFalseCount) * (trueEntropy + falseEntropy)/float(self.totalNumberExamples)

    def calculateGreaterThanEntropy(self):
        if (self.greaterThanTrueCount + self.greaterThanFalseCount) == 0:
            return  0
        trueProbability = float(self.greaterThanTrueCount / float(self.greaterThanFalseCount + self.greaterThanTrueCount))
        falseProbability = 1.0 - trueProbability
        trueEntropy = 0 if trueProbability == 0 else trueProbability*math.log(trueProbability,2)
        falseEntropy = 0 if falseProbability == 0 else falseProbability*math.log(falseProbability,2)
        return -1.0 * (self.greaterThanTrueCount + self.greaterThanFalseCount) * (trueEntropy + falseEntropy)/float(self.totalNumberExamples)