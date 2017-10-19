from mldata import *
from bayesianFeature import *
class BayesianNetwork:

    def __init__(self, trainingExamples, schema, numberOfBins, mEstimate):
        self.numberOfBins = numberOfBins
        self.mEstimate = mEstimate
        self.bayesianFeatures = self.constructFeatureProbabilities(trainingExamples, schema)

    def constructFeatureProbabilities(self, trainingExamples, schema):
        bayesianFeatures = {}
        for i in range(0, len(schema.features)):
            if schema.features[i].type == Feature.Type.NOMINAL or schema.features[
                i].type == Feature.Type.BINARY or schema.features[i].type == Feature.Type.CONTINUOUS:
                bayesianFeatures[i] = BayesianFeature(trainingExamples, i, schema.features[i].type, self.numberOfBins)
        return bayesianFeatures

    def evaluateExamples(self, testingExamples):
        for example in testingExamples:
            self.attributeProbabilitiesGivenClassification(example, example[-1])

    def attributeProbabilitiesGivenClassification(self, example, classification):
        attributeProbabilities = []
        for featureIndex in self.bayesianFeatures:
            attribute = example[featureIndex]
            attributeProbabilities.append(self.bayesianFeatures[featureIndex].probabilityOfAttributeGivenClassification(attribute, classification))
        return attributeProbabilities