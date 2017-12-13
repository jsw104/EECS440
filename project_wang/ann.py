from example import *
import tensorflow as tf
import numpy as np
import math
class Network:
    def constructTrainedClassifier(self, trainingExamples, numSteps):
        # Specify that all features have real-value data
        feature_columns = [tf.feature_column.numeric_column("x", shape=[len(trainingExamples[0].tensor)])]

        # Build 3 layer DNN with 10, 20, 10 units respectively.
        classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=7)
        targets = [ex.target for ex in trainingExamples]
        features = [ex.tensor for ex in trainingExamples]
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(features)},
            y=np.array(targets),
            num_epochs=None,
            shuffle=False)
        classifier.train(input_fn=train_input_fn, steps=numSteps)
        return classifier

    def testClassifier(self, classifier, testingExamples):
        targets = [ex.target for ex in testingExamples]
        features = [ex.tensor for ex in testingExamples]
        # Define the test inputs
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(features)},
            y=np.array(targets),
            num_epochs=1,
            shuffle=False)

        # Evaluate accuracy.
        accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
        print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
        return accuracy_score

    def _constructFeatureColumns(self):
        featureColumns = []
        for col in range(1, 5000):
            featureColumns.append(tf.feature_column.numeric_column(col))


class ExampleManager:
    def _constructBagOfWordsDict(self, line):
        bagOfWordsDict = {}
        count = 0
        for word in line.split(','):
            if (count < 2):
                count = count + 1
            else:
                keyValue = word.split(':')
                bagOfWordsDict[int(keyValue[0])] = int(keyValue[1])
        return bagOfWordsDict

    def exampleSetStatistics(self, examples):
        musicGenreCounts = {}
        for example in examples:
            if example.target not in musicGenreCounts:
                musicGenreCounts[example.target] = 1
            else:
                musicGenreCounts[example.target] = musicGenreCounts[example.target] + 1
        for key in musicGenreCounts.keys():
            print(str(key) + ": " +  str(musicGenreCounts[key]))
        return musicGenreCounts

    def _getTotalWordCounts(self, examples):
        totalWordCounts = np.zeros(5000)
        for example in examples:
            totalWordCounts = totalWordCounts + example.tensor
        return totalWordCounts

    def calculateEntropyByGenre(self, examples):
        wordCountsByGenre = self._calculateOccurencesByGenre(examples)
        totalWordCounts = self._getTotalWordCounts(examples)
        numGenres = len(wordCountsByGenre)
        entropies = np.zeros(len(wordCountsByGenre[1]))
        for i in range(0, len(wordCountsByGenre[1])):
            featureEntropy = 0
            for genreWords in wordCountsByGenre.values():
                if genreWords[i] > 0:
                    featureEntropy = featureEntropy - (float(genreWords[i])/float(totalWordCounts[i])) * math.log(float(genreWords[i])/float(totalWordCounts[i]), numGenres)
            entropies[i] = featureEntropy
        return entropies


    def calculateStandardDeviationsByGenre(self, examples):
        wordCountsByGenre = self._calculateOccurencesByGenre(examples)
        musicGenreSongCounts = self.exampleSetStatistics(examples)
        wordAveragesByGenre = {}
        for i in range(1, len(wordCountsByGenre) + 1):
            wordAveragesByGenre[i] = wordCountsByGenre[i] / musicGenreSongCounts[i]
        totalWordCounts = self._getTotalWordCounts(examples)
        totalWordAverages = totalWordCounts / len(examples)
        stds = np.zeros(5000)
        for i in range(0, len(totalWordAverages)):
            meanSquaredSum = 0
            for genreWords in wordAveragesByGenre.values():
                meanSquaredSum = meanSquaredSum + math.pow(genreWords[i] - totalWordAverages[i], 2)
            std = math.sqrt(meanSquaredSum/(len(wordCountsByGenre) - 1))
            stds[i] = std
        return stds

    def _calculateOccurencesByGenre(self, examples):
        wordCountsByGenre = {}
        for example in examples:
            if(wordCountsByGenre.has_key(example.target)):
                wordCountsByGenre[example.target] = wordCountsByGenre[example.target] + example.tensor
            else:
                wordCountsByGenre[example.target] = example.tensor
        return wordCountsByGenre

    def determineIndicesToRemove(self, k, stds):
        #remove k lowest stds
        print("Running with k: " + str(k))

        idx = np.argpartition(stds, k)
        return idx[0 : k]

        #remove according to threshold, 's'
        #idx = []
        #print ("Threshold: " + str(s))
        #for i in range(0, len(stds)):
        #    if stds[i] < s:
        #        idx.append(i)
        #print ("Removing this many buckets: " + str(len(idx)))
        #return idx
#
    def removeIndicesFromExamples(self, examples, indices):
        for example in examples:
            example.tensor = np.delete(example.tensor, indices)

    def createExamplesFromFiles(self, songLyricsFile, genresFile):
        examples = []
        f = open(songLyricsFile, 'r')
        g = open(genresFile, 'r')
        while True:
            line = f.readline()
            line = line.rstrip()
            if not line: break
            if line[0] == '#' or line[0] == '%': continue
            else:
                genre = g.readline().rstrip()
                if genre != '' and genre != 'None' and genre != 'Mult':
                    bagOfWordsDict = self._constructBagOfWordsDict(line)
                    example = Example(bagOfWordsDict, genre)
                    examples.append(example)
        return examples


k = 2000
exampleManager = ExampleManager()
trainingSongLyricsFile = 'mxm_dataset_train.txt'
trainingGenresFile = 'train_genres.txt'
trainingExamples = exampleManager.createExamplesFromFiles(trainingSongLyricsFile, trainingGenresFile)
exampleManager.exampleSetStatistics(trainingExamples)
entropies = exampleManager.calculateEntropyByGenre(trainingExamples)
indices = exampleManager.determineIndicesToRemove(k, entropies)
exampleManager.removeIndicesFromExamples(trainingExamples, indices)
print "Number of training examples: " + str(len(trainingExamples))
testingSongLyricsFile = 'mxm_dataset_test.txt'
testingGenresFile = 'test_genres.txt'
testingExamples = exampleManager.createExamplesFromFiles(testingSongLyricsFile, testingGenresFile)
exampleManager.removeIndicesFromExamples(testingExamples, indices)
for i in range(1, 10):
    numSteps = i * 100
    network = Network()
    totalAccuracies = 0
    numIterations = 20
    for i in range(0, numIterations):
        classifier = network.constructTrainedClassifier(trainingExamples, numSteps)
        totalAccuracies = totalAccuracies + network.testClassifier(classifier, testingExamples)
    meanAccuracy = totalAccuracies / float(numIterations)
    print("numsteps: " + str(numSteps))
    print "Mean accuracy for " + str(k) + "prunings: " + str(meanAccuracy)