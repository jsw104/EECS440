import numpy as np
import math
from audioop import bias

class NeuralNetworkLayer:
	def __init__(self, numNodes, numInputs):
		self.numNodesThisLayer = numNodes
		self.numInputs = numInputs	
		#self.biases = np.random.uniform(-0.1,0.1,size=(numNodes))
		#self.weights = np.random.uniform(-0.1,0.1,size=(numNodes,numInputs))
		self.biases = np.zeros(shape=(numNodes))
		self.weights = np.zeros(shape=(numNodes,numInputs))
		
	def calculateInputWeightSummations(self, layerInputs):
		#print np.array(layerInputs)
		#print self.weights.transpose() 
		#print self.biases
		#print np.array(layerInputs).dot(self.weights.transpose()) + self.biases
		#print np.transpose(np.array(layerInputs).dot(self.weights.transpose()) + self.biases)
		#raise RuntimeError
		return np.array(layerInputs).dot(self.weights.transpose()) + self.biases
	
	#use this for evaluation phase.
	def willFire(self, weightedSums):
		outputs, derivs = self.applyActivationFunction(weightedSums)
		return np.equal(np.rint(outputs),np.ones(len(weightedSums)))
		 
	def applyActivationFunction(self, weightedSums):
		#print weightedSums
		outputs = np.copy(np.array(weightedSums))
		derivsActivationFunc = np.copy(outputs)	
		for i in range(0, len(weightedSums)):
			outputs[i] = 1.0 / (1.0 + math.exp(-1.0*weightedSums[i]))
			derivsActivationFunc[i] = outputs[i] * (1.0-outputs[i])
		return outputs, derivsActivationFunc
	
	def getOutputs(self, inputs):
		weightedSums = self.calculateInputWeightSummations(inputs)
		layerOutputs, derivsActivationFunc = self.applyActivationFunction(weightedSums)
		return layerOutputs, derivsActivationFunc
	
	#def backpropagateAsFinalLayer(self, layerInputs, targets, learningRate):
	#	layerOutputs, derivs = self.getOutputs(layerInputs)
	#	errors = targets - layerOutputs
	#	biasSensitivities = np.multiply(derivs, errors)
	#	weightSensitivities = layerInputs.dot(biasSensitivites)
	#	
	#	oldBiases = np.copy(self.biases)
	#	oldWeights = np.copy(self.weights)
	#		
	#	self.biases = self.biases - (learningRate * biasSensitivities)
	#	self.weights = self.weights - (learningRate * weightSensitivities)
	#	
	#	return oldWeights, biasSensitivities, weightSensitivities
		
	def backpropagate(self, layerInputs, derivs, downstreamBiasSensitivities, downstreamWeights, learningRate = 0.01):
		print 'layerInputs: ' + str(layerInputs)
		print 'derivs: ' + str(derivs)
		print 'downstreamBiasSensitivities: ' + str(downstreamBiasSensitivities)
		print 'downstreamWeights: ' + str(downstreamWeights)
		
		biasSensitivities = np.multiply(downstreamWeights.transpose().dot(downstreamBiasSensitivities), derivs)
		
		print 'biasSensitivities: ' + str(biasSensitivities)
		
		weightSensitivities = np.copy(self.weights)
		
		for row in weightSensitivities:
			for i in range(0, len(row)-1):
				row[i] = biasSensitivities[i]
				
				
		
		weightSensitivities = biasSensitivities.dot(layerInputs)
		
		print 'weightSensitivities: ' + str(weightSensitivities)
		
		oldBiases = np.copy(self.biases)
		oldWeights = np.copy(self.weights)
		
		self.biases = self.biases - (learningRate * biasSensitivities)
		self.weights = self.weights - (learningRate * weightSensitivities)
		
		#print 'oldBiases: ' + str(oldBiases)
		#print 'biases: ' + str(self.biases)
		#print 'oldWeights: ' + str(oldWeights)
		#print 'weights: ' + str(self.weights)
		
		return oldWeights, biasSensitivities
		
		
		