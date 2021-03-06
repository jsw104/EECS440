
class ContiniousAttributeSplitFinder:

    def sortFeatureValues(self, examples, inputIndex):
        featureValueCounter = {}
        for example in examples:
            if example.inputs[inputIndex] not in featureValueCounter:
                featureValueCounter[example.inputs[inputIndex]] = {}

            if example.target not in featureValueCounter[example.inputs[inputIndex]]:
                featureValueCounter[example.inputs[inputIndex]][example.target] = 0

            featureValueCounter[example.inputs[inputIndex]][example.target] = featureValueCounter[example.inputs[inputIndex]][
                                                                          example.target] + 1
        return sorted(featureValueCounter.keys()), featureValueCounter

    def findPossibleSplitValues(self, examples, featureIndex):
        sortedFeatureValues, featureValueCounter = self.sortFeatureValues(examples, featureIndex)
        possibleSplitValues = []
        for i in range(1,len(sortedFeatureValues)-1):
            if set(featureValueCounter[sortedFeatureValues[i]].keys()) != set(featureValueCounter[sortedFeatureValues[i-1]].keys()) or len(featureValueCounter[sortedFeatureValues[i]]) > 1 or len(featureValueCounter[sortedFeatureValues[i-1]]) > 1:
                possibleSplitValues.append(float(sortedFeatureValues[i] + sortedFeatureValues[i-1]) / 2)
        
        return possibleSplitValues
