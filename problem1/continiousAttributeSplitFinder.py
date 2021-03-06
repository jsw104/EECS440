
class ContiniousAttributeSplitFinder:
    
    def __init__(self, schema):
        self.schema = schema

    def sortFeatureValues(self, examples, featureIndex):
        featureValueCounter = {}
        for example in examples:
            if example[featureIndex] not in featureValueCounter:
                featureValueCounter[example[featureIndex]] = {}

            if example[-1] not in featureValueCounter[example[featureIndex]]:
                featureValueCounter[example[featureIndex]][example[-1]] = 0

            featureValueCounter[example[featureIndex]][example[-1]] = featureValueCounter[example[featureIndex]][
                                                                          example[-1]] + 1
        return sorted(featureValueCounter.keys()), featureValueCounter

    def findPossibleSplitValues(self, examples, featureIndex):
        sortedFeatureValues, featureValueCounter = self.sortFeatureValues(examples, featureIndex)
        possibleSplitValues = []
        for i in range(1,len(sortedFeatureValues)-1):
            if set(featureValueCounter[sortedFeatureValues[i]].keys()) != set(featureValueCounter[sortedFeatureValues[i-1]].keys()) or len(featureValueCounter[sortedFeatureValues[i]]) > 1 or len(featureValueCounter[sortedFeatureValues[i-1]]) > 1:
                possibleSplitValues.append(float(sortedFeatureValues[i] + sortedFeatureValues[i-1]) / 2)
        
        return possibleSplitValues
