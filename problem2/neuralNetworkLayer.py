import numpy as np
import math
from audioop import bias

class NeuralNetworkLayer:
	def __init__(self, numNodes, numInputs):
		self.numNodesThisLayer = numNodes
		self.numInputs = numInputs	
		self.biases = np.random.uniform(-0.1,0.1,size=(numNodes,1))
		self.weights = np.random.uniform(-0.1,0.1,size=(numNodes,numInputs))
		#self.biases = np.zeros(shape=(numNodes,1))
		#self.weights = np.zeros(shape=(numNodes,numInputs))
		
	def calculateInputWeightSummations(self, layerInputs):
		layerInputs = np.reshape(layerInputs, (1,-1))
		return np.reshape(layerInputs.dot(self.weights.transpose()),(-1,1)) + self.biases
	
	#use this for evaluation phase.
	def willFire(self, weightedSums):
		outputs, derivs = self.applyActivationFunction(weightedSums)
		return np.equal(np.rint(outputs),np.ones(len(weightedSums)))
		 
	def applyActivationFunction(self, weightedSums):
		#print weightedSums
		outputs = np.copy(weightedSums)
		derivsActivationFunc = np.copy(outputs)	
		#print outputs
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
		#print 'layerInputs: ' + str(layerInputs)
		derivs = np.reshape(derivs,(-1,1))
		#print 'derivs: ' + str(derivs)
		#print 'downstreamBiasSensitivities: ' + str(downstreamBiasSensitivities)
		#print 'downstreamWeights: ' + str(downstreamWeights)
		#print 'lefthalfofmultiply: ' + str(downstreamWeights.transpose().dot(np.reshape(downstreamBiasSensitivities,(1,-1))))
		biasSensitivities = np.multiply(downstreamWeights.transpose().dot(np.reshape(downstreamBiasSensitivities,(1,-1))), derivs)
				
		#print 'biasSensitivities: ' + str(biasSensitivities)
				
		weightSensitivities = biasSensitivities.dot(np.reshape(layerInputs,(1,-1)))
		
		#print 'weightSensitivities: ' + str(weightSensitivities)
		
		oldBiases = np.copy(self.biases)
		oldWeights = np.copy(self.weights)
		
		self.biases = self.biases - (learningRate * biasSensitivities)
		self.weights = self.weights - (learningRate * weightSensitivities)
		
		#print '======= oldBiases ======='
		#print oldBiases
		#print '======= biases =======' 
		#print self.biases
		#print '======= oldWeights ======='
		#print oldWeights
		#print '======= weights =======' 
		#print self.weights
		
		return oldWeights, biasSensitivities
		
		
		