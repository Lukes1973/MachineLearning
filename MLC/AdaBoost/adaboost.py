from numpy import *
def loadSimpData():
	datMat = matrix([[ 1. ,  2.1],
[2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return datMat,classLabels


#单层决策树生成函数

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	#先初始化所有分类为1
	retArray = ones((shape(dataMatrix)[0],1))
	#判断条件
	if threshIneq=='lt':
		#根据阈值情况，修改对应行的分类
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	return retArray

def buildStump(dataArr,classLabels,D):
	#将数据转换成为矩阵格式
	dataMatrix = mat(dataArr);labelMat = mat(classLabels).T
	#获取dataMatrix的行列数
	m,n = shape(dataMatrix)
	#初始化相关参数
	numSteps= 10.0;bestStump = {};bestClasEst = mat(zeros((m,1)))
	#设置最小错误率为+∞
	minError = inf
	#遍历每个特征
	for i in range(n):
		#取对应特征列中的最小值，最大值
		rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max();
		#确定步长
		stepSize = (rangeMax-rangeMin)/numSteps
		#循环递增步长
		for j in range(-1,int(numSteps)+1):
			#设置不等参数,双验证机制
			for inequal in ['lt','gt']:
				#设置判断阈值
				threshVal = (rangeMin + float(j)*stepSize)
				#通过阈值，调用stumpClassify进行分类
				predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
				#先假设所有的分类都是错误的，初始化都为1的列向量
				errArr = mat(ones((m,1)))
				#将判断正确的，设置为0，即代表着判断正确
				errArr[predictedVals==labelMat]=0
				#获取此轮分类的weightedError
				weightedError = D.T*errArr
				#print("split:dim%d,thresh %.2f,thresh inequal: %s ,the weightedError is %.3f"%\
					#(i,threshVal,inequal,weightedError))
				# 获取最优分类，对应最佳决策分类值thresh，最佳分类特征i，最佳分类条件ineq
				if weightedError<minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim']=i
					bestStump['thresh']=threshVal
					bestStump['ineq']=inequal
	return bestStump,minError,bestClasEst



#----------------------------------------------------------------------------------


#full version adaboost algorithm

def adaboostTrainDS(dataArr,classLabels,numIt=40):
	#初始化参数
	weakClassArr = []
	#获取数据集的行数
	m = shape(dataArr)[0]
	#依据行数初始化列向量D，注意D中所有数字之和是1
	D = mat(ones((m,1))/m)
	#初始化类别估计累计值
	aggClassEst = mat(zeros((m,1)))
	#对数据进行循环迭代
	for i in range(numIt):
		#利用buildstump建立一个弱分类器
		bestStump,error,classEst = buildStump(dataArr,classLabels,D)
		#打印当前的D向量
		print("D:",D.T)
		#利用得到的error计算alpha
		alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
		#记录当前的alpha值
		bestStump['alpha'] = alpha
		#添加当前分类器到weakClassArr
		weakClassArr.append(bestStump)
		#打印出当前预测值
		print("classEst:",classEst.T)
		#根据新的alpha值更新D
		expon = multiply(-1*alpha*mat(classLabels).T,classEst)
		D = multiply(D,exp(expon))
		#确保D向量所有值相加为1
		D=D/D.sum()
		#累加aggClassEst
		aggClassEst += alpha*classEst
		print("aggClassEst: ",aggClassEst)
		#利用列向量的方式显示分类是否错误，为1为时，表示分类错误
		aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
		#计算错误率
		errorRate = aggErrors.sum()/m
		#打印错误率
		print("total error :",errorRate,"\n")
		if errorRate == 0.0:break
	return weakClassArr,aggClassEst

# 测试adaboost 算法

def adaClassify(datToClass,classifierArr):
	#矩阵化需要分类的数据集
	dataMatrix = mat(datToClass)
	#获取数据集的行数
	m = shape(dataMatrix)[0]
	#初始化类别估计累计值
	aggClassEst = mat(zeros((m,1)))
	#遍历每一个弱分类器
	for i in range(len(classifierArr)):
		#获取当前分类器下的预测值
		classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
			classifierArr[i]['thresh'],classifierArr[i]['ineq'])
		#累加aggClassEst
		aggClassEst += classifierArr[i]['alpha']*classEst
		print(aggClassEst)
	#返回预测值
	return sign(aggClassEst)


#example 第四章战马生还预测

#数据加载函数,不用指定行数
def loadDataSet(fileName):
	#获得列数，也就是特征列和预测列数之和
	numFeat = len(open(fileName).readline().split('\t'))
	#初始化相关变量
	dataMat = [];labelMat = []
	#打开文件
	fr = open(fileName)
	#遍历每一行
	for line in fr.readlines():
		#临时变量
		lineArr = []
		#获取当前行数据
		curLine = line.strip().split('\t')
		#遍历每一个数字，添加到lineArr中
		for i in range(numFeat-1):
			lineArr.append(float(curLine[i]))
		#添加至dataMat作为完整数据
		dataMat.append(lineArr)
		#获取需要预测的标签值
		labelMat.append(float(curLine[-1]))
    #返回特征值和标签向量
	return dataMat,labelMat



#样本非均衡分类问题

def plotROC(predStrengths,classLabels):
	#倒入matplotlib库
	import matplotlib.pyplot as plt
	#初始化数据点
	cur = (1.0,1.0)
	#用于计算AUC值
	ySum = 0.0
	#统计标签值为1的个数
	numPosClas = sum(array(classLabels) == 1.0)
	#y轴步长
	yStep = 1/float(numPosClas)
	#x轴步长
	xStep = 1/float(len(classLabels)-numPosClas)
	#对累积值进行排序,从小到大
	sortedIndicies = predStrengths.argsort()
	#打开一张画布
	fig = plt.figure()
	fig.clf()
	#设置子轴
	ax = plt.subplot(111)
	#去累积值（预测强度）中每个数
	for index in sortedIndicies.tolist()[0]:
		#这里条件说明如果和实际不相符，则真阳率下降，也就是说明在Y轴方向减小
		if classLabels[index] == 1.0:
			delX = 0; delY = yStep;
		else:
			delX = xStep; delY = 0;
			#累加返回新的高度值，用于后面计算AUC面积
			ySum += cur[1]
		#绘制变化后对点
		ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
		#返回当前的点的坐标值
		cur = (cur[0]-delX,cur[1]-delY)
	#绘制一条参考虚线
	ax.plot([0,1],[0,1],'b--')
	plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
	plt.title('ROC curve for adaboost Horse Colic Detection System')
	ax.axis([0,1,0,1])
	plt.show()
	print("the Area Under the Curve is: ",ySum*xStep)





