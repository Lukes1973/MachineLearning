from numpy import *

import matplotlib.pyplot as plt

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

def standRegres(xArr,yArr):
	xMat = mat(xArr);yMat = mat(yArr).T
	xTx = xMat.T*xMat
	if linalg.det(xTx) == 0.0:
		print("This matrix is singular,cannot do inverser")
		return
	ws = xTx.I * (xMat.T*yMat)
	return ws



def plot():
	#xArr, yArr = loadDataSet('MLC\Regression\ex0.txt')
	xArr, yArr = loadDataSet('ex0.txt')
	xMat = mat(xArr);yMat = mat(yArr)
	ws = standRegres(xArr,yArr)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
	xCopy=xMat.copy()
	xCopy.sort(0)
	yHat = xCopy*ws
	ax.plot(xCopy[:,1],yHat)
	plt.show()


