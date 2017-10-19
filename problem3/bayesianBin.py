class BayesianBin:
    def __init__(self, minimumValue, maximumValue):
        #greater than or equal to minimum and less than maximum...
        self.minimumValue = minimumValue
        self.maximumValue = maximumValue

    def belongsInBin(self, value):
        return (value >= self.minimumValue && value < self.maximumValue)