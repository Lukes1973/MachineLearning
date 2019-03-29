
from numpy import *

def loadDataSet(fileName):
    dataMat = [];labelMat=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

#simple version of SMO
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = mat(dataMatIn);labelMat = mat(classLabels).transpose()
    b = 0;m,n =shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter=0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # 求的当前alphas下，第i条数据对应的y值
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            # 比较计算值和实际值之间的误差
            Ei = fXi - float(labelMat[i])
            #根据KTT条件，判断是否满足需要优化的条件
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and \
                                                                   (alphas[i] > 0)):
                #在0和m之间选择j,确保i不等于j
                j = selectJrand(i,m)
                #在当前alphas值下，计算第j条数据对应y值
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                #比较计算值和当前值的误差
                Ej = fXi -float(labelMat[j])
                #根据python引用传递的规则，需要重新分配内存保存旧的alphas，从而实现比较新旧alphas的目的
                alphasIold = alphas[i].copy(); alphasJold = alphas[j].copy();
                #根据y值是否相等，分别计算不同边界值L，H，用于将alphas调整至0至C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] -C)
                    H = min(C,alphas[j] + alphas[i])
                #如果L等于H，则不需要调整alphas，结束本次循环
                if L==H:print("L=H");continue
                #计算alphas[j]的最佳修改值，hh
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T-\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:print("eta>=0");continue
                alphas[i] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphasJold)<0.00001):print("j not moving enough");continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphasJold-alphas[j])
                b1 = b - Ei -labelMat[i]*(alphas[i]-alphasIold)*dataMatrix[i,:]*dataMatrix[i,:].T-\
                labelMat[j]*(alphas[j]-alphasIold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b- Ej - labelMat[i]*(alphas[i]-alphasIold)*dataMatrix[i,:]*dataMatrix[j,:].T-\
                    labelMat[j]*(alphas[j]-alphasJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0< alphas[i]) and (C > alphas[j]):b=b1
                elif (0 < alphas[j]) and (C>alphas[j]):b=b2
                else:b=(b1+b2)/2.0
                alphaPairsChanged +=1
                print("iter:%d i:%d,pairs changed %d"%(iter,i,alphaPairsChanged))
        if (alphaPairsChanged==0):iter+=1
        else:iter=0
        print("iteration number:%d"% iter)
    return b,alphas

