from numpy import *
import operator

def fileMatrix(filePath):
    f = open(filePath)
    arrayLines = f.readlines()
    arrayRow = len(arrayLines)

    returnMat = zeros((arrayRow,3))
    labelVec = []
    index = 0;

    for line in arrayLines:
        line = line.strip()
        lineList = line.split('\t')
        returnMat[index :] = lineList[:3]
        labelVec.append(int(lineList[-1]))
        index += 1
    return returnMat, labelVec

def normaData(dataSet):
    minData = dataSet.min(0)
    maxData = dataSet.max(0)
    extreNum = maxData - minData

    normaDataSet = zeros(shape(dataSet))
    row = dataSet.shape[0]

    normaDataSet = dataSet - tile(minData, (row, 1))
    normaDataSet /= tile(extreNum,(m,1))
    return normaDataSet,extreNum,minData

def classfiy()


fileMatrix('datingTestSet2.txt')

