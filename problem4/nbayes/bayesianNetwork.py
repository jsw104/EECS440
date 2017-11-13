from mldata import *
from bayesianFeature import *
import math
class BayesianNetwork:

    def __init__(self, schema, numberOfBins, mEstimate):
        self.inputFeatures = schema.features[1:-1]
        self.numberOfBins = numberOfBins
        self.mEstimate = mEstimate
        self.bayesianFeatures = None
        self.classificationProbabilities = None
        
    def train(self, trainingExamples):
        self.bayesianFeatures = self._constructFeatureProbabilities(trainingExamples)
        self.classificationProbabilities = self._constructClassificationProbabilities(trainingExamples)

    def _constructFeatureProbabilities(self, trainingExamples):
        bayesianFeatures = {}
        for i in range(0, len(self.inputFeatures)):
            if (self.inputFeatures[i].type == Feature.Type.NOMINAL or self.inputFeatures[
                i].type == Feature.Type.BINARY or self.inputFeatures[i].type == Feature.Type.CONTINUOUS) and i > 1:
                bayesianFeatures[i] = BayesianFeature(trainingExamples, i, self.inputFeatures[i].type, self.numberOfBins)
        return bayesianFeatures

    def _constructClassificationProbabilities(self, trainingExamples):
        classificationCounter = {}
        for example in trainingExamples:
            if example.target not in classificationCounter:
                classificationCounter[example.target] = 1
            else:
                classificationCounter[example.target] = classificationCounter[example.target] + 1
        classificationProbabilities = {}
        for classification in classificationCounter:
            classificationProbabilities[classification] = float(classificationCounter[classification])/float(len(trainingExamples))
        return classificationProbabilities

    def evaluateExamples(self, testingExamples):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        targetOutputPairs = []
        for example in testingExamples:
            classificationHypothesis, confidence = self.evaluateExample(example)
            if classificationHypothesis == example.target and example.target:
                tp = tp + 1
            elif classificationHypothesis == example.target and not example.target:
                tn = tn + 1
            elif classificationHypothesis != example.target and example.target:
                fn = fn + 1
            else:
                fp = fp + 1
        
            confExampleIsTrue = confidence if classificationHypothesis else 1.0-confidence    
            targetOutputPairs.append((example.target, confExampleIsTrue))

        return tp, fp, tn, fn, targetOutputPairs


    def evaluateExample(self, example):
        #doesnt change depending on classification we are testing...
        #exampleAttributeProbabilities = self.attributeProbabilitiesForExample(example)

        bestClassification = None
        bestClassificationResult = -1
        sumHypothesisResults = 0
        for classification in self.classificationProbabilities:
            attributeProbabilitiesGivenClassification = self.attributeProbabilitiesGivenClassification(example, classification)
            classificationProbability = self.classificationProbabilities[classification]
            hypothesisResult = self.performHypothesisTest(attributeProbabilitiesGivenClassification, classificationProbability)
            
            sumHypothesisResults = sumHypothesisResults + hypothesisResult
            if bestClassificationResult == -1 or hypothesisResult > bestClassificationResult:
                bestClassification = classification
                bestClassificationResult = hypothesisResult

        confidence = bestClassificationResult/sumHypothesisResults if sumHypothesisResults != 0 else 0.0 
        return bestClassification, confidence

    def performHypothesisTest(self, attributeProbabilitiesGivenClassification, classificationProbability):
        result = 0
        for i in range(0, len(attributeProbabilitiesGivenClassification)):
            if attributeProbabilitiesGivenClassification[i] == 0:
                return 0.0 #Any one zero factor will zero this whole computation
            result = result - math.log(attributeProbabilitiesGivenClassification[i], 2)
        result = result - math.log(classificationProbability, 2)
        return math.pow(2, -1*result)

    def attributeProbabilitiesGivenClassification(self, example, classification):
        attributeProbabilitiesGivenClassification = []
        for featureIndex in self.bayesianFeatures:
            attribute = example.inputs[featureIndex]
            attributeProbabilitiesGivenClassification.append(self.bayesianFeatures[featureIndex].probabilityOfAttributeGivenClassification(attribute, classification, self.classificationProbabilities, self.mEstimate))
        return attributeProbabilitiesGivenClassification

    def attributeProbabilitiesForExample(self, example):
        attributeProbabilities = []
        for featureIndex in self.bayesianFeatures:
            attribute = example[featureIndex]
            attributeProbabilities.append(self.bayesianFeatures[featureIndex].probabilityOfAttribute(attribute))
        return attributeProbabilities