import random
from mldata import *
from dtree.dtree import DTree
from ann.neuralNetwork import NeuralNetwork
from nbayes.bayesianNetwork import BayesianNetwork
from logreg.logregClassifier import LogRegClassifier
            
class BaggedClassifier:
    def __init__(self, learning_alg, numBags, schema):
        
        numInputs = 0
        for i in range(0, len(schema.features)):
            nominalAttributeHash = None
            if schema.features[i].type == Feature.Type.NOMINAL or schema.features[i].type == Feature.Type.BINARY or schema.features[i].type == Feature.Type.CONTINUOUS:
                numInputs = numInputs + 1 
        
        self.classifiers = []
        for i in range(0, numBags):
            if learning_alg == 'DTREE':
                self.classifiers.append(DTree(schema, maxDepth=1, useInformationGain=False))
            elif learning_alg == 'ANN':
                self.classifiers.append(NeuralNetwork([1], numInputs, weightDecayCoeff=0, maxTrainingIterations=-1))
            elif learning_alg == 'NBAYES':
                self.classifiers.append(BayesianNetwork(schema, numberOfBins=15, mEstimate=1))
            else: #learning_alg == 'LOGREG'
                self.classifiers.append(LogRegClassifier(numInputs, const_lambda=0.01))
    
    def _bag(self, examples, numBags):
        bags = []
        for i in range(0, numBags):
            bag = []
            for j in range(0,len(examples)):
                bag.append(examples[random.randrange(0, len(examples))])
            bags.append(bag)
        return bags
                   
    def train(self, trainingExamples):
        bags = self._bag(trainingExamples, len(self.classifiers))
        for i in range(0, len(self.classifiers)):
            bag = bags[i]
            clf = self.classifiers[i]
            clf.train(bag)
        
    def evaluateExamples(self, examples):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        baggedTargetOutputPairs = []
        
        targetValuePairOutputs = []
        for clf in self.classifiers:
            targetValuePairOutputs.append(clf.evaluateExamples(examples)[4])
        
        for i in range(0, len(examples)):
            example = examples[i]
            new_score = 0
            for tvp in targetValuePairOutputs:
                new_score = new_score + tvp[i][1]
            new_score = float(new_score) / len(self.classifiers)
            
            classification = int(round(new_score)) == 1
            target = example.target if hasattr(example,'target') else example[-1]
            
            if classification and target:
                tp = tp + 1
            elif classification and not target:
                fp = fp + 1
            elif not classification and target:
                fn = fn + 1
            else:
                tn = tn + 1
            baggedTargetOutputPairs.append((target, new_score))
                
        return tp, fp, tn, fn, baggedTargetOutputPairs
        
        

            