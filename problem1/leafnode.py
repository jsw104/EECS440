
class LeafNode:
    
    def __init__(self, classLabel, classLabelFraction):
        self.classLabel = classLabel
        self.classLabelFraction = classLabelFraction
        self.evaluationCount = 0

    def incrementEvaluationCount(self):
        self.evaluationCount = self.evaluationCount + 1