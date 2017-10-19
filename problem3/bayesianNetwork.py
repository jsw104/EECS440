from mldata import *
from bayesianFeature import *
class BayesianNetwork:

    def __init__(self, trainingExamples, schema, numberOfBins, mEstimate):
        self.numberOfBins = numberOfBins
        self.mEstimate = mEstimate
        self.constructFeatureProbabilities(trainingExamples, schema)

    def constructFeatureProbabilities(self, trainingExamples, schema):
        bayesianFeatures = []
        for i in range(0, len(schema.features)):
            if schema.features[i].type == Feature.Type.NOMINAL or schema.features[
                i].type == Feature.Type.BINARY or schema.features[i].type == Feature.Type.CONTINUOUS:
                bayesianFeatures[i] = BayesianFeature(trainingExamples, i, self.numberOfBins)
