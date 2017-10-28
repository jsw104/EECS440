from mldata import *
import numpy as np
from continuousAttributeStandardizer import *

class NormalizedExample:
    def __init__(self, example, schema, nominalAttributeHashes, continuousAttributeHash):
        inputsList = []
        for i in range(0, len(example)):
            if schema.features[i].type == Feature.Type.NOMINAL:
                nominalAttributeHash = nominalAttributeHashes[schema.features[i]]
                feature = example[i]
                if example[i] in nominalAttributeHash:
                    feature = nominalAttributeHash[feature]
                inputsList.append(feature) 
            elif (schema.features[i].type == Feature.Type.BINARY):
                inputsList.append(example[i])
            elif (schema.features[i].type == Feature.Type.CONTINUOUS):
                inputsList.append((continuousAttributeHash[schema.features[i]]).standardizeInput(example[i]))
            elif schema.features[i].type == Feature.Type.CLASS:
                self.target = example[i]
        self.inputs = np.array(inputsList)
        
class ExampleNormalizer:      
    def __init__(self, exampleSet):
        self.numUsefulFeatures = 0
        self.nominalAttributeHashes = {}
        self.continuousAttributeHash = {}
        for i in range(0, len(exampleSet.schema.features)):
            nominalAttributeHash = None
            if exampleSet.schema.features[i].type == Feature.Type.NOMINAL or exampleSet.schema.features[
                i].type == Feature.Type.BINARY or exampleSet.schema.features[i].type == Feature.Type.CONTINUOUS:
                self.numUsefulFeatures = self.numUsefulFeatures + 1
            if exampleSet.schema.features[i].type == Feature.Type.NOMINAL:
                for value in exampleSet.schema[i].values:
                    if nominalAttributeHash is None:
                        nominalAttributeHash = {}
                        self.nominalAttributeHashes[exampleSet.schema.features[i]] = nominalAttributeHash
                    if value not in nominalAttributeHash:
                        nominalAttributeHash[value] = len(nominalAttributeHash.keys())
            elif exampleSet.schema.features[i].type == Feature.Type.CONTINUOUS:
                self.continuousAttributeHash[exampleSet.schema.features[i]] = ContinuousAttributeStandardizer(exampleSet.examples, i)
                
    def normalizeExamples(self, exampleSet):
        normalizedExamples = []
        for example in exampleSet.examples:
            normalizedExamples.append(NormalizedExample(example, exampleSet.schema, self.nominalAttributeHashes, self.continuousAttributeHash))
        return normalizedExamples