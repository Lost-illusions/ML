from math import log
import operator

#计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not  in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        #计算概率
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#c创建数据集
def createDataSet():
    #1.不浮出水面是否可以生存 2.是否有脚蹼 3.是否属于鱼类
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    #特征有两类，1.不浮出水面是否可以生存 2.是否有脚蹼
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

#按照数据集中第axis个特征值进行划分，返回第axis个特征值是value的数据集(将第axis列去除的）
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    #原始数据集的香农熵
    baseEnt = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1

    for i in range(numFeatures):
        #返回的是第i列值的列表
        featList = [example[i] for example in dataSet]
        #返回一个队列（不含重复值），即第i列可取的值的队列
        uniqueVals = set(featList)
        newEnt = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            #计算第i个特征值为value的发生概率
            prob = len(subDataSet)/float(len(dataSet))
            newEnt += prob * calcShannonEnt(subDataSet)
        #信息增益,信息增益越大说明词特征的选择越合理，而香农熵越小数据越有序
        inforGain = baseEnt - newEnt
        if inforGain > bestInfoGain:
            bestInfoGain = inforGain
            bestFeature = i
        return bestFeature

#返回出现次数最多的元素，classList保存的labels中的最后一列元素
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

#创建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #count方法统计元素个数
    if classList.count(classList[0]) == len(classList):
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return  myTree



dataSet, labels = createDataSet()
#print(calcShannonEnt(dataSet))
#a =splitDataSet(dataSet, 0, 1)
#b = chooseBestFeatureToSplit(dataSet)
print(createTree(dataSet, labels))
