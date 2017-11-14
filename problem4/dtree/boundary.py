import math
class Boundary:
    def __init__(self, boundaryValue, examples):
        self.totalExampleWeight = sum(example.weight for example in examples)
        self.lessThanTrueWeight = 0
        self.greaterThanTrueWeight = 0
        self.lessThanFalseWeight = 0
        self.greaterThanFalseWeight = 0
        self.boundaryValue = boundaryValue

    def calculateLessThanEntropy(self):
        if (self.lessThanTrueWeight + self.lessThanFalseWeight) == 0:
            return 0
        trueProbability = float(self.lessThanTrueWeight / float(self.lessThanFalseWeight + self.lessThanTrueWeight))
        falseProbability = 1.0 - trueProbability
        trueEntropy = 0 if trueProbability == 0 else trueProbability*math.log(trueProbability,2)
        falseEntropy = 0 if falseProbability == 0 else falseProbability*math.log(falseProbability,2)

        return -1.0 * (self.lessThanTrueWeight + self.lessThanFalseWeight) * (trueEntropy + falseEntropy) / float(self.totalExampleWeight)

    def calculateGreaterThanEntropy(self):
        if (self.greaterThanTrueWeight + self.greaterThanFalseWeight) == 0:
            return  0
        trueProbability = float(self.greaterThanTrueWeight / float(self.greaterThanFalseWeight + self.greaterThanTrueWeight))
        falseProbability = 1.0 - trueProbability
        trueEntropy = 0 if trueProbability == 0 else trueProbability*math.log(trueProbability,2)
        falseEntropy = 0 if falseProbability == 0 else falseProbability*math.log(falseProbability,2)
        return -1.0 * (self.greaterThanTrueWeight + self.greaterThanFalseWeight) * (trueEntropy + falseEntropy) / float(self.totalExampleWeight)