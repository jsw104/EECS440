import math
import mldata

def entropy_attribute(examples, attributeIndex):
    """
    Compute the entropy of the attribute with the specified attributeIndex for the example set
    """
    attr_counts = {}
    for example in examples:
        if example is not None and len(example) > 0:
            attr = str(example[attributeIndex])
            if(attr not in attr_counts):
                attr_counts[attr] = 1
            else:
                attr_counts[attr] = attr_counts[attr] + 1
    
    entropy = 0
                
    for attr in attr_counts:
        attr_count = attr_counts[attr]
        p = float(attr_count)/len(examples)
        entropy = entropy + (p * math.log(p,2))  
            
    return -1 * entropy

def entropy_class_label(examples):
    """
    Compute the class label entropy of the example set, which is the last atrribute
    """
    if len(examples) == 0:
        return 0

    return entropy_attribute(examples, len(examples[0])-1)
    

"""Find the majority class"""
def majority_class(examples):
    
    classCounter = {}
    for example in examples:
        if example[-1] not in classCounter:
            classCounter[example[-1]] = 0
        classCounter[example[-1]] = classCounter[example[-1]] + 1
            
    majorityClass = None
    majorityClassCount = 0
    for classLabel, count in classCounter.items():
        if majorityClassCount < count:
            majorityClass = classLabel
            majorityClassCount = count
    
    majorityClassFraction = float(majorityClassCount) / len(examples)
            
    return majorityClass, majorityClassFraction