import random
class NeuralNode:
    def __init__(self, numberOfWeights):
        self.weights = []
        for i in range(1, numberOfWeights):
            self.weights.append(random.uniform(0, 1))

