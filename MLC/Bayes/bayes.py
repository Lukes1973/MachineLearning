from numpy import *

def loadDataSet():
	#一些评论样本数据
	postingList=[['my', 'dog', 'has', 'flea', \
						 'problems', 'help', 'please'],
						 ['maybe', 'not', 'take', 'him', \
						  'to', 'dog', 'park', 'stupid'],
						 ['my', 'dalmation', 'is', 'so', 'cute', \
						   'I', 'love', 'him'],
						 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
						 ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
						   'to', 'stop', 'him'],
						 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	
    #样本对应的分类数据，0代表正常言论，1代表侮辱性言论
	classVec = [0,1,0,1,0,1]
	return postingList,classVec

#构造词语字典
def createVocabList(dataSet):
	#创建一个空集
	vocabSet = set([])
	#遍历所有的数据
	for document in dataSet:
		#获取两个集合的并集，用来确保词语的唯一性
		vocabSet = vocabSet | set(document)
	#返回词典表
	return list(vocabSet)

#根据已有的词典，构造词向量
def setOfWords2Vec(vocabList,inputSet):
	#创造一个和词典个数相同，但是所有元素都为0的向量
	returnVec = [0]*len(vocabList)
	#遍历输入评论
	for word in inputSet:
		#判断词是否在字典里
		if word in vocabList:
			#如果存在，那么对应词典的位置标记为1
			returnVec[vocabList.index(word)] = 1
		# 不存在则显示不在词典里面
		else:print("the word : %s is not in my Vocalbulary!" %word)
	#返回词向量
	return returnVec


def trainNB0(trainMatrix,trainCategory):
	#获取词向量的个数
	numTrainDocs = len(trainMatrix)
	#获取构造每个词向量词典词数
	numWords = len(trainMatrix[0])
	#计算归类为1的概率
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	#构造零向量
	p0Num = zeros(numWords);p1Num = zeros(numWords)
	#初始化词典中每个词向量累加总数
	p0Denom = 0.0;p1Denom = 0.0
	#遍历每一条词向量
	for i in range(numTrainDocs):
        #判断对应的类别
		if trainCategory[i] == 1:
			#如果类别为1，对应向量自相加并保存在p1Num
			p1Num += trainMatrix[i]
			#累计总数，便于后续计算条件概率
			p1Denom += sum(trainMatrix[i])
		else:
			#如果类别为0，对应向量自相加并保存在p0Num
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	#分别计算对应类别下的条件概率
	p1Vect = p1Num/p1Denom
	p0Vect = p0Num/p0Denom
	return p0Vect,p1Vect,pAbusive

# Modify Version of TrainNB0

def trainNB0M(trainMatrix,trainCategory):
	#获取词向量的个数
	numTrainDocs = len(trainMatrix)
	#获取构造每个词向量词典词数
	numWords = len(trainMatrix[0])
	#计算归类为1的概率
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	#构造零向量
	p0Num = ones(numWords);p1Num = ones(numWords)
	#初始化词典中每个词向量累加总数
	p0Denom = 2.0;p1Denom = 2.0
	#遍历每一条词向量
	for i in range(numTrainDocs):
        #判断对应的类别
		if trainCategory[i] == 1:
			#如果类别为1，对应向量自相加并保存在p1Num
			p1Num += trainMatrix[i]
			#累计总数，便于后续计算条件概率
			p1Denom += sum(trainMatrix[i])
		else:
			#如果类别为0，对应向量自相加并保存在p0Num
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	#分别计算对应类别下的条件概率
	p1Vect = log(p1Num/p1Denom)
	p0Vect = log(p0Num/p0Denom)
	return p0Vect,p1Vect,pAbusive


def classifyNB(vec2Classify,p0Vect,p1Vect,pClass):
	#计算同一条对应在不同分类下的概率
	p1 = sum(vec2Classify*p1Vect) + log(pClass)
	p0 = sum(vec2Classify*p0Vect) + log(1.0 - pClass)
	#比较不同类别下对应概率，根据概率大小给出结论
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	#获取训练词典数据集
	lsitOPosts,listClasses = loadDataSet()
	#构造训练词典数据集字典
	myVocabList = createVocabList(lsitOPosts)
	#初始化训练集数据
	trainMat = []
	#获取每一条评论并转换成词向量
	for postinDoc in lsitOPosts:
		#添加词向量到训练集
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
	#计算属于侮辱性文档的概率PAb，以及两个类别的概率向量
	p0V,p1V,pAb = trainNB0M(array(trainMat),array(listClasses))
	#测试评论数据
	testEntry = ['love','my','dalmation']
	#转化为词向量
	thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
	#通过函数classifyNB，判别该评论是否为侮辱性
	print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
	#测试评论
	testEntry = ['stupid','garbage']
	thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
	print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

#词袋模型
def bagOfWords2VecMN(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

#示例：过滤垃圾邮件

#使用非数字，字母作为分割符号，划分单词
def textParse(bigString):
	#引入正则表达式
	import re
	#非数字，非字母正则表达式
	regEx = re.compile('\\W*')
	#分割字段
	listOfTokens = regEx.split(bigString)
	#返回分割字段
	return [tok.lower() for tok in listOfTokens if len(tok) > 0]

def spamTest():
	#初始化
	docList = [];classList = []; fullText = []
	#遍历所有文件
	for i in range(1,26):
		#切分字段
		wordList = textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		#print(i)
		wordList = textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	#print(docList)
	#构造全词数据字典
	vocabList = createVocabList(docList)
	#使用交叉验证机制，随机选择20%数据作为测试数据，80%作为训练数据，先确定指针
	trainingSet = list(range(50));testSet = []
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	#初始化训练数据，和对应的标签
	trainMat = [];trainClasses = []
	#根据随机选择的数据，选择对应的邮件，同时转换为词向量
	for docIndex in trainingSet:
		#添加对应词向量
		trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
		#添加对应的标签
		trainClasses.append(classList[docIndex])
	#print(array(trainMat))
	#计算属于欺诈邮件的概率pSpam，以及两个类别的概率向量
	p0V,p1V,pSpam = trainNB0M(array(trainMat),array(trainClasses))
	errorCount = 0
	#通过测试集计算验证正确率
	for docIndex in testSet:
		#选择测试字段，然后转换为词向量
		wordVector = setOfWords2Vec(vocabList,docList[docIndex])
		#如果结果和实际不一致，则错误数加1
		if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
			errorCount += 1
    #计算最后错误的比例
	print('the error rate is: ',float(errorCount)/len(testSet))
















