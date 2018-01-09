from numpy import *
from os import listdir
import operator

#将32x32的二进制图像矩阵转换为1x1024的向量
def imgVector(filename):
    returnVect = zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        listStr = f.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(listStr[j])
    return returnVect

def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return  sortClassCount[0][0]

def handwritingClassTest():
    hwlabels = []
    #返回指定路径下的文件和文件夹列表。
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    #将trainingDigits中的二进制图像转换为1x1024的向量，并根据文件名将对应的数字保存到hwlabels中
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        labelNum = int(fileStr.split('_')[0])
        hwlabels.append(labelNum)
        trainingMat[i, :] = imgVector('trainingDigits\%s'%fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for j in range(mTest):
        fileNameStr = testFileList[j]
        fileStr = fileNameStr.split('.')[0]
        labelNum = int(fileStr.split('_')[0])
        testVector = imgVector('testDigits\%s'%fileNameStr)
        classifiyResult = classify(testVector, trainingMat, hwlabels, 3)
        #print('识别结果为：%d, 真实结果为：%d'%(classifiyResult,labelNum))
        if int(classifiyResult) != labelNum:
            errorCount = errorCount + 1.0
            #print('识别结果为：%d, 真实结果为：%d'%(classifiyResult,labelNum))
    print('一共错了：%f个'%errorCount)
    rightRatio = 1 - (errorCount/float(mTest))
    print('正确率为：%f %%'%(rightRatio*100))


handwritingClassTest()
