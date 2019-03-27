from numpy import *


# help function for SMO

#读取文件
def loadDataSet(fileName):
    dataMat = [];labelMat=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        #获取特征量
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        #获取目标量
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

# make sure i not equal to j
def selectJrand(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

#
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

#simple version of SMO
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = mat(dataMatIn);labelMat = mat(classLabels).transpose()
    b= 0 ;m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphasPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei = fXi -float(labelMat[i])


