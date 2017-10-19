from mldata import *
from bayesianFeature import *
class BayesianNetwork:

    def __init__(self, trainingExamples, schema, numberOfBins, mEstimate):
        self.numberOfBins = numberOfBins
        self.mEstimate = mEstimate
        self.bayesianFeatures = self.constructFeatureProbabilities(trainingExamples, schema)
        self.classificationProbabilities = self.constructClassificationProbabilities(trainingExamples)

    def constructFeatureProbabilities(self, trainingExamples, schema):
        bayesianFeatures = {}
        for i in range(0, len(schema.features)):
            if schema.features[i].type == Feature.Type.NOMINAL or schema.features[
                i].type == Feature.Type.BINARY or schema.features[i].type == Feature.Type.CONTINUOUS:
                bayesianFeatures[i] = BayesianFeature(trainingExamples, i, schema.features[i].type, self.numberOfBins)
        return bayesianFeatures

    def constructClassificationProbabilities(self, trainingExamples):
        classificationCounter = {}
        for example in trainingExamples:
            if example[-1] not in classificationCounter:
                classificationCounter[example[-1]] = 0
            else:
                classificationCounter[example[-1]] = classificationCounter[example[-1]] + 1
        classificationProbabilities = {}
        for classification in classificationCounter:
            classificationProbabilities[classification] = float(classificationCounter[classification])/float(len(trainingExamples))
        return classificationProbabilities

    def evaluateExamples(self, testingExamples):
        totalCorrect = 0
        totalIncorrect = 0
        for example in testingExamples:
            classificationHypothesis = self.evaluateExample(example)
            if classificationHypothesis == example[-1]:
                totalCorrect = totalCorrect + 1
            else:
                totalIncorrect = totalIncorrect + 1
        accuracy = float(totalCorrect) / float(totalCorrect + totalIncorrect)
        print "ACCURACY IS " + str(accuracy)


    def evaluateExample(self, example):
        #doesnt change depending on classification we are testing...
        #exampleAttributeProbabilities = self.attributeProbabilitiesForExample(example)

        bestClassification = None
        bestClassificationResult = -1
        for classification in self.classificationProbabilities:
            attributeProbabilitiesGivenClassification = self.attributeProbabilitiesGivenClassification(example, classification)
            classificationProbability = self.classificationProbabilities[classification]
            hypothesisResult = self.performHypothesisTest(attributeProbabilitiesGivenClassification, classificationProbability)
            if bestClassificationResult == -1 or hypothesisResult > bestClassificationResult:
                bestClassification = classification
                bestClassificationResult = hypothesisResult
        return bestClassification

    def performHypothesisTest(self, attributeProbabilitiesGivenClassification, classificationProbability):
        result = 1
        for i in range(0, len(attributeProbabilitiesGivenClassification)):
            result = result * (attributeProbabilitiesGivenClassification[i])
        result = result * classificationProbability
        return result

    def attributeProbabilitiesGivenClassification(self, example, classification):
        attributeProbabilitiesGivenClassification = []
        for featureIndex in self.bayesianFeatures:
            attribute = example[featureIndex]
            attributeProbabilitiesGivenClassification.append(self.bayesianFeatures[featureIndex].probabilityOfAttributeGivenClassification(attribute, classification))
        return attributeProbabilitiesGivenClassification

    def attributeProbabilitiesForExample(self, example):
        attributeProbabilities = []
        for featureIndex in self.bayesianFeatures:
            attribute = example[featureIndex]
            attributeProbabilities.append(self.bayesianFeatures[featureIndex].probabilityOfAttribute(attribute))
        return attributeProbabilities