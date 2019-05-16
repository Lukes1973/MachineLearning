from numpy import *

import matplotlib.pyplot as plt
#加载数据，获取特征值和标签值
def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t'))-1
	dataMat = [];labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat

#计算最佳线性拟合直线
def standRegres(xArr,yArr):
	#把数据转化为矩阵格式
	xMat = mat(xArr);yMat = mat(yArr).T
	#计算XTX
	xTx = xMat.T*xMat
	#判断矩阵的行列式是否为零，判断矩阵是否可逆
	if linalg.det(xTx) == 0.0:
		print("This matrix is singular,cannot do inverser")
		return
	#计算ws参数
	ws = xTx.I * (xMat.T*yMat)
	return ws

#绘制线性回归图形
def plot():	
	#加载数据
	xArr, yArr = loadDataSet('ex0.txt')
	#转化为矩阵格式
	xMat = mat(xArr);yMat = mat(yArr)
	#调用standRegres计算出ws
	ws = standRegres(xArr,yArr)
	#初始化画布
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#绘制原始数据散点图
	ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
	#复制一份特征值数据
	xCopy=xMat.copy()
	#从小到大排序
	xCopy.sort(0)
	#根据计算出的ws进行预测
	yHat = xCopy*ws
	#绘制出预测直线
	ax.plot(xCopy[:,1],yHat)
	plt.show()


