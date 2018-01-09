


def sigmoid(inX):
    '''
    阶跃函数
    :param inX:
    :return: 介于0-1之间的值
    '''
    return 1.0/(1 + np.exp(-inX))

def stocGradAscent1(dataMatrix, classLabels, numlter=150):
    '''
    随机梯度上升算法
    :param dataMatrix: 数据集(转换为np数组）
    :param classLabels: 类别集
    :param numlter: 最大迭代次数
    :return: 权重列表
    '''
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numlter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]
    return weights

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
    return weights.getA()
