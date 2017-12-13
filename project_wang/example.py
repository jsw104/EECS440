import tensorflow as tf
import numpy

validGenres = {'Pop': 1, 'Alternative': 2, 'Rock': 3, 'Country': 4, 'Heavy Metal': 5, 'Hip Hop/Rap': 6}

class Example:
    def __init__(self, bagsOfWordsDict, targetGenre):
        if targetGenre == '':
            print 'whoops'
        self.target = validGenres[targetGenre]
        self.tensor = self._createTensor(bagsOfWordsDict)

    def _createTensor(self, bagOfWordsDict):
        tensor = numpy.zeros(5000)
        for key in bagOfWordsDict:
            tensor[key - 1] = bagOfWordsDict[key]
        return tensor

