from numpy import *
import operator

'''
1. strip()
    描述
        Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）。
    语法
        str.strip([chars]);
    参数
        chars -- 移除字符串头尾指定的字符。
    返回值
        返回移除字符串头尾指定的字符生成的新字符串。
        
2.numpy-tile()
    函数形式： tile(A，rep) 
    功能：重复A的各个维度 
    参数类型： 
        - A: Array类的都可以 
        - rep：A沿着各个维度重复的次数（可以是数字或者元组），
            按照从类向外的维度，（2,3）就是最内层的维度重复2次，外层重复3次

'''



#将文本文件中的数据存储到数组矩阵中
def fileMatrix(filename):
    '''

    :param filename: 数据文件的路径
    :return: 经过分割后的数据集（numpy数组）与类别集（列表）
    '''

    f = open(filename)
    arrarylines = f.readlines()
    #得到文件的行数
    linesNum = len(arrarylines)
    #创建空的返回矩阵,因为有三个特征值，故列数为3
    returnMat = zeros((linesNum, 3))

    labelVector = []
    index = 0

    for line in arrarylines:
        #用于移除字符串头尾指定的字符（默认为空格）
        line = line.strip()
        #按照制表符分割一行，分成含有四个元素的列表
        listLine = line.split('\t')
        returnMat[index, :] = listLine[0: 3]
        labelVector.append(int(listLine[-1]))
        index = index + 1
    return returnMat, labelVector

#为避免某个数值因为数值本身过大或（过小）对结果造成不好的影响，故进行归一化，使数值处在-1-1(0-1)之间
#newValue = (oldValue - min) / (max - min)
def normaData(dataSet):
    '''

    :param dataSet: 数据集（numpy数组）
    :return: 经过归一化后的数据集（numpy数组）， 每一列最大值与最小值之差的列表，每列最小值的列表
    '''

    #dataSet.min(0)表示每列的最小值，为1表示每行的最小值
    minData = dataSet.min(0)
    maxData = dataSet.max(0)

    extreMum = maxData - minData
    normaDataSet = zeros(shape(dataSet))
    #计算矩阵行数
    m = dataSet.shape[0]
    normaDataSet = dataSet - tile(minData, (m,1))
    normaDataSet = normaDataSet / (tile(extreMum, (m, 1)))
    return  normaDataSet, extreMum, minData

#KNN
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

#测试分类器的正确率,传入参数k为临近数据选择数修改k以找到使准确率最高的k值
def datingClassTest(k):
    #测试用的数据占总数据的百分比
    testRatio = 0.10
    datingDataMat, datingLabels = fileMatrix('datingTestSet2.txt')
    normMat, extreMum, minData = normaData(datingDataMat)
    m = normMat.shape[0]
    numTest = int(m * testRatio)
    errorCount = 0.0
    for i in range(numTest):
        classfiyResult = classify(normMat[i, :], normMat[numTest:m, :], datingLabels[numTest:m], k)
        print('分类器结果：%d,真实结果：%d'%(classfiyResult, datingLabels[i]))
        if classfiyResult != datingLabels[i]:
            errorCount = errorCount + 1.0
    rightRadio = 1 - errorCount/float(numTest)
    print('正确率为：%f %%'%(rightRadio*100))

def classifyPerson():
    resultList = ['令人讨厌', '一丢丢吧', '我的我的我的！']
    ffMiles = float(input('每年获得的飞行常客里程数：'))
    percentTats = float(input('玩视频游戏所耗时间百分比：'))
    iceCream = float(input('每周消费的冰淇淋公升数：'))

    datingDataMat, datinglabels = fileMatrix('datingTestSet2.txt')
    normMat, extreMum, minData = normaData(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    #对输入数据进行归一化
    normInArr = (inArr - minData)/extreMum
    classifyResult = classify(normInArr, normMat, datinglabels,3)
    print('或许你对这个人的感觉是：%s'%resultList[classifyResult-1])

#classifyPerson()
datingDataMat, datinglabels = fileMatrix('datingTestSet2.txt')
