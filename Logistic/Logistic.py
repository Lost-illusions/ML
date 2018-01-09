import matplotlib.pyplot as plt
import numpy as np

def loadDataSet():
    '''
    载入数据
    :return: 列表数据集，类别列表
    '''
    dataMat = []
    lableMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        lableMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, lableMat

def plotBestFit(weights):
    '''
    绘制边界曲线
    :param weights:权重列表
    :return:
    '''
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1 :
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    #等同于ax = fig.add_subplot(1, 1, 1)
    ax = fig.add_subplot(111)
    # s代表点的大小，marker表示点的形状，alpha代表透明度
    ax.scatter(xcord1, ycord1, s=10, c='red', marker='o', alpha=.5)
    ax.scatter(xcord2, ycord2, s=10, c='green', alpha=.5)
    x = np.arange(-3.0, 3.0, 0.1)
    # x即x1，y即X2,根据 0 = w0 * x0 + w1 * x1 + w2 * x2 解出x2与x1的关系
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x,y)
    plt.title('BestFit')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def sigmoid(inX):
    '''
    阶跃函数
    :param inX:
    :return: 介于0-1之间的值
    '''
    return 1.0/(1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    '''
    梯度上升算法
    :param dataMatIn: 数据集
    :param classLabels: 类别集
    :return: 权重列表
    '''
    dataMatrix = np.mat(dataMatIn)
    #将类标签转换为numpy的列向量
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    #步长
    alpha = 0.001
    #最大迭代次数
    maxCycles = 500
    #系数矩阵
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    #将矩阵转化为数组
    return weights.getA()


dataMat, labelMat = loadDataSet()
weights = gradAscent(dataMat, labelMat)
plotBestFit(weights)
