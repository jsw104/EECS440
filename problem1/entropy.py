import math
import mldata

"""Compute the class label entropy of the example set"""
def entropy_class_label(exampleSet):
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