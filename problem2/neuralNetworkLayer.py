import numpy as np
import math
from audioop import bias

class NeuralNetworkLayer:
	def __init__(self, numNodes, numInputs):
		self.numNodesThisLayer = numNodes
		self.numInputs = numInputs	
		self.biases = np.random.uniform(-0.1,0.1,size=(numNodes,1))
		self.weights = np.random.uniform(-0.1,0.1,size=(numNodes,numInputs))
		#self.biases = np.zeros(shape=(numNodes,1)) #for debugging only
		#self.weights = np.zeros(shape=(numNodes,numInputs)) #for debugging only
		
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
		
	#Vectorized backpropagation
	def backpropagate(self, layerInputs, derivs, downstreamBiasSensitivities, downstreamWeights, learningRate = 0.01):
		# For the output layer, set downstreamBiasSensitivities to the error (output-target) and downstreamWeights to a vector of ones
		derivs = np.reshape(derivs,(-1,1))
		biasSensitivities = np.multiply(downstreamWeights.transpose().dot(np.reshape(downstreamBiasSensitivities,(1,-1))), derivs)
		weightSensitivities = biasSensitivities.dot(np.reshape(layerInputs,(1,-1)))
		oldWeights = np.copy(self.weights)
		self.biases = self.biases - (learningRate * biasSensitivities)
		self.weights = self.weights - (learningRate * weightSensitivities)
		
		# DEBUGGING OUTPUTS
		#print '*********************************************************'
		#print 'layerNodes: ' + str(self.numNodesThisLayer) + '; layerInputs: ' + str(self.numInputs)
		#print '======= layerInputs ======='
		#print layerInputs
		#print '======= derivs ======='
		#print derivs
		#print '======= downstreamBiasSensitivities ======='
		#print downstreamBiasSensitivities
		#print '======= downstreamWeights ======='
		#print downstreamWeights
		#print '======= biasSensitivities ======='
		#print biasSensitivities
		#print '======= weightSensitivities ======='
		#print weightSensitivities
		#print '======= oldBiases ======='
		#print oldBiases
		#print '======= biases =======' 
		#print self.biases
		#print '======= oldWeights ======='
		#print oldWeights
		#print '======= weights =======' 
		#print self.weights
		
		return oldWeights, biasSensitivities
		
		