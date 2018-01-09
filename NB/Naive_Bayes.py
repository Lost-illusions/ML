'''
数据spam中的为垃圾邮件，ham为非垃圾邮件
'''

import numpy as np
import random
import re

def loadDataSet():
    #进行词条分割后的文档集合
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #类别标签，0=正常言论，1=侮辱性言论
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

#将dataSet中的数据即分割后的单词存储到一个列表中，通过set过滤掉重复的单词
def   createVocabList(dataSet):
    vocabList = set([])
    for document in dataSet:
        vocabList = vocabList | set(document)
    return list(vocabList)

def setOfWords2Vec(vocabList, inputSet):
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

def testingNB():
    listOPosts,listClasses = loadDataSet()                                  #创建实验样本
    myVocabList = createVocabList(listOPosts)                               #创建词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))             #将实验样本向量化
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))        #训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']                                 #测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))              #测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')                                        #执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')                                       #执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']                                       #测试样本2

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))              #测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')                                        #执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')                                       #执行分类并打印分类结果

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

spamTest()
