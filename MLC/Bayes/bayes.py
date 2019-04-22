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
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num = zeros(numWords);p1Num = zeros(numWords)
	p0Denom = 0.0;p1Denom = 0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = p1Num/p1Denom
	p0Vect = p0Num/p0Denom
	return p0Vect,p1Vect,pAbusive
	pass




























