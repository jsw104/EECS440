import numpy as np
import math
from audioop import bias

class NeuralNetworkLayer:
	def __init__(self, numNodes, numInputs):
		self.numNodesThisLayer = numNodes
		self.numInputs = numInputs	
		self.biases = np.random.uniform(-0.1,0.1,size=(numNodes,1))
		self.weights = np.random.uniform(-0.1,0.1,size=(numNodes,numInputs))
		
	def calculateInputWeightSummations(self, layerInputs):
		layerInputs = np.reshape(layerInputs, (1,-1))
		return np.reshape(layerInputs.dot(self.weights.transpose()),(-1,1)) + self.biases
		 
	def applyActivationFunction(self, weightedSums):
		outputs = np.copy(weightedSums)
		derivsActivationFunc = np.copy(outputs)
		for i in range(0, len(weightedSums)):
			outputs[i] = self._sigmoid(weightedSums[i])
			derivsActivationFunc[i] = outputs[i] * (1.0-outputs[i])
		return outputs, derivsActivationFunc
	
	def _sigmoid(self, x):
		if x >= 0:
			return 1 / (1 + math.exp(-x))
		else:
			expx = math.exp(x)
			return expx / (1 + expx)
	
	def getOutputs(self, inputs):
		weightedSums = self.calculateInputWeightSummations(inputs)
		layerOutputs, derivsActivationFunc = self.applyActivationFunction(weightedSums)
		return layerOutputs, derivsActivationFunc
	
	def checkConvergence(self, oldBiases, oldWeights, convergenceThresh=0.0075):
		diffBiases = np.absolute(self.biases - oldBiases)
		diffWeights = np.absolute(self.weights - oldWeights)
		biasesConverged = (diffBiases >= convergenceThresh).sum() == 0
		weightsConverged = (np.reshape(diffWeights,(1,-1)) >= convergenceThresh).sum() == 0
		return (biasesConverged and weightsConverged)
		
	def backpropagate(self, layerInputs, derivs, downstreamBiasSensitivities, downstreamWeights, learningRate=0.01):
		# For the output layer, set downstreamBiasSensitivities to the error (output-target) and downstreamWeights to a vector of ones
		
		# VECTORIZED BACKPROPAGATION
		derivs = np.reshape(derivs,(-1,1))
		biasSensitivities = np.multiply(downstreamWeights.transpose().dot(np.reshape(downstreamBiasSensitivities,(1,-1))), derivs)
		weightSensitivities = biasSensitivities.dot(np.reshape(layerInputs,(1,-1)))
		oldWeights = np.copy(self.weights)
		oldBiases = np.copy(self.biases)
		self.biases = self.biases - (learningRate * biasSensitivities)
		self.weights = self.weights - (learningRate * weightSensitivities)
		
		return oldWeights, biasSensitivities
		
		