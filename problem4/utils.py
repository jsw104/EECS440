import os
import numpy as np
from mldata import *


def getExamplesFromDataPath(dataPath):
    # Read data file
    fileName = os.path.basename(dataPath)
    rootDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath[0:-(len(fileName) + 1)])
    exampleSet = parse_c45(fileName, rootDirectory)
    return exampleSet

def computeStatistics(tp, fp, tn, fn):
    numAll = tp+tn+fp+fn
    numScoredPos = tp+fp
    numActualPos = tp+fn
    accuracy = 0.0 if numAll == 0 else float(tp+tn)/numAll
    precision = 0.0 if numScoredPos == 0 else float(tp)/numScoredPos
    recall = 0.0 if numActualPos == 0 else float(tp)/numActualPos
    return accuracy, precision, recall

def computePooledAROC(listsTargetOutputPair):
    allTargetOutputPairs = []
    for pairList in listsTargetOutputPair:        
        allTargetOutputPairs = allTargetOutputPairs + pairList
        
    rocPoints = (len(allTargetOutputPairs)+2)*[None]
    rocPoints[0] = (0.0,0.0,1.0)
    rocPoints[-1] = (1.0,1.0,0.0)
    appendIndex = 1
    allTargetOutputPairs.sort(key = lambda x:x[1])

    totalTP = 0.0
    totalFP = 0.0
    totalTN = 0.0
    totalFN = 0.0
    #start with threshold of zero confidence.
    for targetOutputPair in allTargetOutputPairs:
        if targetOutputPair[0]:
            totalTP = totalTP + 1
        else:
            totalFP = totalFP + 1

    #incrementally move confidence level over to the right
    for targetOutputPair in allTargetOutputPairs:
        if targetOutputPair[0]:
            totalTP = totalTP - 1
            totalFN = totalFN + 1
        else:
            totalFP = totalFP - 1
            totalTN = totalTN + 1
        fpRate = 0.0 if totalFP + totalTN == 0 else totalFP/(totalFP + totalTN)
        tpRate = 0.0 if totalTP + totalFN == 0 else totalTP/(totalTP + totalFN)
        rocPoints[appendIndex] = (fpRate, tpRate)
        appendIndex = appendIndex + 1
    
    rocPoints.sort(key=lambda tup: tup[0])
    
    areaUnderROC = 0
    for i in range(0, len(rocPoints)-1):
        if rocPoints[i+1][0] == rocPoints[i][0]:
            continue
        else:
            trapezoidArea = 0.5 * (rocPoints[i][1] + rocPoints[i+1][1]) * (rocPoints[i+1][0] - rocPoints[i][0])
            areaUnderROC = areaUnderROC + trapezoidArea
            
    return areaUnderROC