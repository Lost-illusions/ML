import feedparser
import operator
import numpy as np
import random
import re


def   createVocabList(dataSet):
    '''
    将dataSet中的数据即分割后的单词存储到一个列表中，通过set过滤掉重复的单词
    :param dataSet: 原始数据集，将每个文档字符串化后，作为一个列表加入其中的列表的集合。
                    即一个列表中包含多个列表，每个列表
    :return:经过过滤后的词汇集（不包含重复的词汇）
    '''
    vocabList = set([])
    for document in dataSet:
        vocabList = vocabList | set(document)
    return list(vocabList)

def setOfWords2Vec(vocabList, inputSet):
    '''
    将未知（在本例还是已知数据集中抽取的部分数据作为测试）的词汇表，转化为出现置1否则为0的列表
    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabularty!'% word)
    return  returnVec

#词袋模型,只是简单修改了setOfWords2Vec,使返回值统计了每个词出现的次数
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return  returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords= len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    #为防止 计算多个概率的乘积以获得文档属于某个类别的概率 时因一个概率为0而导致结果为0，故而做以下操作
    p0Num = np.ones(numWords)  #p0Num = np.zeros(numWords)
    p1Num = np.ones(numWords)  #p1Num = np.zeros(numWords)
    p0Denom = 2.0   #p0Denom = 0.0
    p1Denom = 2.0   #p1Denom = 0.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #通过取对数防止结果下溢而不精确
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #因为在trainNB0中返回的是取对数后的值，故而这里的两个对数相加其实是不取对数的乘： logA * B = logA + logB
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def textParse(bigString):
    listOfTokens =  re.split(r'\W', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []; classList = []; fullText = []

    for i in range(1,26):
        wordList = textParse(open(r'D:/Python/ML/NB/email/spam/%d.txt'%i,'r',encoding='gbk').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open(r'D:/Python/ML/NB/email/ham/%i.txt'%i,'r',encoding='gbk').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    #存储训练集和测试集索引值的列表
    trainingSet = list(range(50)); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0.0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print('分类错误的测试集：', docList[docIndex])
    print('错误率：%.2f %%'%(float(errorCount)/len(testSet)*100))

def calcMostFreq(vocabList, fullText):
    '''
    :param vocabList: 经过过滤的词汇集（不重复）
    :param fullText: 将文档划分为单独词后的列表（有重复）
    :return: 前30个高频词列表
    '''
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    '''
    通过已知来自两个不同地方的数据集，判断未知文档（言论）来自两个地方的概率
    1：代表纽约地区
    2：旧金山海湾地区
    :param feed1:纽约的RSS源
    :param feed0:旧金山海湾地区的RSS源
    :return:词汇集，属于1类的概率，属于2类的概率
    '''
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        #访问RSS源中第i个词条的summary
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)

        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])

    #随机生成10个测试集
    trainingSet = list(range(2 * minLen)); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0.0
    for docIndex in testSet:
        wordVector =  bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    accuracyRat = (1.00 - float(errorCount)/float(len(testSet)))*100
    print('正确率为：%.2f %%'%accuracyRat)
    return vocabList, p0V, p1V


ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
localWords(ny,sf)
