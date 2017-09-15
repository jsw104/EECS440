import math
import mldata

def entropy_class_label(exampleSet):
    """
    Compute the class label entropy of the example set
    """

    label_counts = {}
    for example in exampleSet:
        if example is not None and len(example) > 0:
            label = str(example[-1])
            if(label not in label_counts):
                label_counts[label] = 1
            else:
                label_counts[label] = label_counts[label] + 1
    
    entropy = 0
                
    for label in label_counts:
        label_count = label_counts[label]
        p = float(label_count)/len(exampleSet)
        entropy = entropy + (p * math.log(p,2))  
            
    return -1 * entropy

"""Find the majority class"""
def majority_class(examples):
    
    classCounter = {}
    for example in examples:
        if example[-1] not in classCounter:
            classCounter[example[-1]] = 0
        classCounter[example[-1]] = classCounter[example[-1]] + 1
            
    majorityClass = None
    for classLabel in classCounter:
        if majorityClass is None or classCounter[classLabel] > majorityClass:
            majorityClass = classLabel
    
    majorityClassFraction = float(classCounter[majorityClass]) / len(examples)
            
    return majorityClass, majorityClassFraction