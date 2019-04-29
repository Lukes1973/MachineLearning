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
	p1 = sum(vec2Classify*p1Vect) + log(pClass)
	p0 = sum(vec2Classify*p0Vect) + log(1.0 - pClass)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	lsitOPosts,listClasses = loadDataSet()
	myVocabList = createVocabList(lsitOPosts)
	trainMat = []
	for postinDoc in lsitOPosts:
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
		p0V,p1V,pAb = trainNB0M(array(trainMat),array(listClasses))
		testEntry = ['love','my','dalmation']
		thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
		print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
		testEntry = ['stupid','garbage']
		thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
		print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))




























