class BayesianBin:
    def __init__(self, minimumValue, maximumValue, lastBin):
        #greater than or equal to minimum and less than maximum...
        self.minimumValue = minimumValue
        self.maximumValue = maximumValue
        self.lastBin = lastBin

    def belongsInBin(self, value):
        #must include less than or equal to maximum if the very last bin...
        if self.lastBin:
            return (value >= self.minimumValue and value <= self.maximumValue)
        else:
            return (value >= self.minimumValue and value < self.maximumValue)