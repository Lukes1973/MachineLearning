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

#====================================================
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
                Ej = fXj -float(labelMat[j])
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
                #计算alphas[j]的最佳修改值
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T-\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:print("eta>=0");continue
                #在eta<0的时候，对计算新的alphas值
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                #使用chipAlpha对alphas[j]进行修正，确保其在区间[L,H]
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphasJold)<0.00001):print("j not moving enough");continue
                #当alpha[j]发生足够大变化时，同时改变alphas[i]，但是改变的方向相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphasJold-alphas[j])
                #分别计算不同alphas对应的b值
                #根据Ei和oS.X[i,:]计算b1
                b1 = b - Ei -labelMat[i]*(alphas[i]-alphasIold)*dataMatrix[i,:]*dataMatrix[i,:].T-\
                labelMat[j]*(alphas[j]-alphasJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                #根据Ej和oS.X[j,:]计算b1
                b2 = b- Ej - labelMat[i]*(alphas[i]-alphasIold)*dataMatrix[i,:]*dataMatrix[j,:].T-\
                    labelMat[j]*(alphas[j]-alphasJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0< alphas[i]) and (C > alphas[i]):b=b1
                elif (0 < alphas[j]) and (C>alphas[j]):b=b2
                else:b=(b1+b2)/2.0
                alphaPairsChanged +=1
                print("iter:%d i:%d,pairs changed %d"%(iter,i,alphaPairsChanged))
        if (alphaPairsChanged==0):iter+=1
        else:iter=0
        print("iteration number:%d"% iter)
    return b,alphas

#==============================================================================================

# full version of SMO without kernal

#构造对象，保存重要的参数
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        #保存训练数据集
        self.X = dataMatIn
        #保存数据标签类别
        self.labelMat = classLabels
        #设置松弛变量C，用来允许一定程度上的错误分类
        self.C = C
        #设置每一次长度
        self.tol = toler
        #获取数据集的函数
        self.m = shape(dataMatIn)[0]
        #构造一个列向量alphas
        self.alphas = mat(zeros((self.m,1)))
        #初始化b值
        self.b = 0
        #缓存误差，包含两列，第一列是标记误差是否有效，第二列保存误差值
        self.eCache = mat(zeros((self.m,2)))

#计算误差
def calcEk(oS,k):
    #根据当前alphas值，计算第k条数据预测的classlabel值
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    #计算误差值
    Ek = fXk -float(oS.labelMat[k])
    return Ek

#根据下标i和误差Ei来计算第二个alphas的值，用来保证每次优化中可以获得最大的步长
def selectJ(i,oS,Ei):
    #初始化参数
    maxK = -1;maxDeltaE = 0;Ej = 0
    #更新缓存误差值，并设置有效值为1
    oS.eCache[i] = [1,Ei]
    #获取有效的误差值
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    #判断有效误差值的个数
    if (len(validEcacheList))>1:
        #遍历所有的有效误差值，选择误差值deltaE最大时对应的Ek,保证此次优化获得最大步长
        for k in validEcacheList:
            #跳过k=i的条件
            if k==i:continue
            #计算误差
            Ek = calcEk(oS,k)
            #获取和Ei误差相对值
            deltaE = abs(Ei-Ek)
            #确保误差最大
            if (deltaE>maxDeltaE):
                maxK = k;maxDeltaE = deltaE;Ej=Ek
        return maxK,Ej
    #一般第一次，随机选择j
    else:
        #选择j，是的i<>j,同时j在区间（0，m)之间
        j = selectJrand(i,oS.m)
        #计算行数为j对应的误差
        Ej = calcEk(oS,j)
    return j,Ej

#更新误差值到eCache中
def updateEk(oS,k):
    #计算误差值
    Ek = calcEk(oS,k)
    #更新误差值，标记有效性
    oS.eCache[k]=[1,Ek]

#内部循环函数，返回为0或者1
def innerL(i,oS):
    #计算误差值
    Ei = calcEk(oS,i)
    #根据KTT条件，判断是否满足需要优化的条件，误差超过设置的误差值极限值tol时，选择新的j
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
        ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] >0)):
        #依据误差最大原则和选择最大步长原则，确定新的j和相对应的误差值
        j,Ej = selectJ(i,oS,Ei)
        #复制当前的alphas[i]和alphas[j]值，确保不会因为参数传递而发生变化
        alphasIold = oS.alphas[i].copy();alphasJold = oS.alphas[j].copy();
        #根据y值是否相等，分别计算不同边界值L，H，用于将alphas调整至0至C之间
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C,oS.alphas[j] + oS.alphas[i])
        #如果L等于H，则不需要调整alphas，返回0
        if L==H: print("L==H");return 0
        #计算alphas[j]的最佳修改值
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T -\
        oS.X[i,:]*oS.X[j,:].T
        #eta>=0，暂不优化j，返回0
        if eta >= 0:print("eta>=0");return 0
        #在eta<0的时候，对计算新的alphas值
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        #使用chipAlpha对alphas[j]进行修正，确保其在区间[L,H]
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        #在更新alphas[j]之后，更新误差缓存值
        updateEk(oS,j)
        #比较新的alphas[j]和alphasold,如果变化很小，返回0
        if (abs(oS.alphas[j]-alphasJold)<0.00001):
            print("j not moving enough");return 0
        #当alpha[j]发生足够大变化时，同时改变alphas[i]，但是改变的方向相反
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*\
        (alphasJold-oS.alphas[j])
        #在更新alphas[i]之后，更新误差缓存值
        updateEk(oS,i)
        #根据Ei和oS.X[i,:]计算b1
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphasIold)*\
        oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*\
        (oS.alphas[j]-alphasJold)*oS.X[i,:]*oS.X[j,:].T
        #根据Ej和oS.X[j,:]计算b1
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphasIold)*\
        oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*\
        (oS.alphas[j]-alphasJold)*oS.X[j,:]*oS.X[j,:].T
        #判断不同情形下，oS.b的取值
        if (0<oS.alphas[i]) and (oS.C > oS.alphas[i]):oS.b = b1
        elif (0<oS.alphas[j]) and (oS.C > oS.alphas[j]):oS.b = b2
        else:oS.b = (b1+b2)/2.0
        return 1
    else:return 0

#外部循环函数
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    #调用optStruct,初始化必要的参数
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    #记录迭代次数
    iter = 0
    #设置entireSet和记录alpha改变的次数
    entireSet = True;alphaPairsChanged = 0
    #构造循环执行的条件，当小于迭代次数并且alphas改变次数大于0，或者entireSet=True
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            #遍历每一行数据
            for i in range(oS.m):
                #调用innerL函数，计算最优的alphas[j]，同时更新alphas[i]，计算当前的b值
                alphaPairsChanged += innerL(i,oS)
                #记录alphas改变的次数
                print("fullSet,iter:%d i:%d,pairs changed %d"%\
                (iter,i,alphaPairsChanged))
            #记录遍历所有数据集的次数
            iter +=1
        else:
            #遍历一次之后,alphas就会有改变，不全为零，取出不全为零的点
            nonBoundIs = nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            #遍历不为零对应所在行数据
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound,iter:%d i:%d,pairs changed %d"%\
                (iter,i,alphaPairsChanged))
            iter +=1
        if entireSet:entireSet = False
        elif (alphaPairsChanged==0):entireSet = True
        print("iteration number:%d"%iter)
    return oS.b,oS.alphas
#计算w参数
def calcWs(alphas,dataArr,classLabels):
    X =mat(dataArr);labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

# kernel version of smo

def kernelTrans(X,A,kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin':K = X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:raise NameError('BEIJING We have a Problem --\
        That kernel is not recognized')
    return K

#=================================================================================

#kernel version innerL

# new opt Struct
class optStructk:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        #保存训练数据集
        self.X = dataMatIn
        #保存数据标签类别
        self.labelMat = classLabels
        #设置松弛变量C，用来允许一定程度上的错误分类
        self.C = C
        #设置每一次长度
        self.tol = toler
        #获取数据集的函数
        self.m = shape(dataMatIn)[0]
        #构造一个列向量alphas
        self.alphas = mat(zeros((self.m,1)))
        #初始化b值
        self.b = 0
        #缓存误差，包含两列，第一列是标记误差是否有效，第二列保存误差值
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)


def calcEkk(oS,k):
    #根据当前alphas值，计算第k条数据预测的classlabel值
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    #计算误差值
    Ek = fXk -float(oS.labelMat[k])
    return Ek
def selectJk(i,oS,Ei):
    #初始化参数
    maxK = -1;maxDeltaE = 0;Ej = 0
    #更新缓存误差值，并设置有效值为1
    oS.eCache[i] = [1,Ei]
    #获取有效的误差值
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    #判断有效误差值的个数
    if (len(validEcacheList))>1:
        #遍历所有的有效误差值，选择误差值deltaE最大时对应的Ek,保证此次优化获得最大步长
        for k in validEcacheList:
            #跳过k=i的条件
            if k==i:continue
            #计算误差
            Ek = calcEkk(oS,k)
            #获取和Ei误差相对值
            deltaE = abs(Ei-Ek)
            #确保误差最大
            if (deltaE>maxDeltaE):
                maxK = k;maxDeltaE = deltaE;Ej=Ek
        return maxK,Ej
    #一般第一次，随机选择j
    else:
        #选择j，是的i<>j,同时j在区间（0，m)之间
        j = selectJrand(i,oS.m)
        #计算行数为j对应的误差
        Ej = calcEkk(oS,j)
    return j,Ej

def updateEkk(oS,k):
    #计算误差值
    Ek = calcEkk(oS,k)
    #更新误差值，标记有效性
    oS.eCache[k]=[1,Ek]

#kernel version innerLk
def innerLk(i,oS):
    #计算误差值
    Ei = calcEkk(oS,i)
    #根据KTT条件，判断是否满足需要优化的条件，误差超过设置的误差值极限值tol时，选择新的j
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
        ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] >0)):
        #依据误差最大原则和选择最大步长原则，确定新的j和相对应的误差值
        j,Ej = selectJk(i,oS,Ei)
        #复制当前的alphas[i]和alphas[j]值，确保不会因为参数传递而发生变化
        alphasIold = oS.alphas[i].copy();alphasJold = oS.alphas[j].copy();
        #根据y值是否相等，分别计算不同边界值L，H，用于将alphas调整至0至C之间
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C,oS.alphas[j] + oS.alphas[i])
        #如果L等于H，则不需要调整alphas，返回0
        if L==H: print("L==H");return 0
        #计算alphas[j]的最佳修改值
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        #eta>=0，暂不优化j，返回0
        if eta >= 0:print("eta>=0");return 0
        #在eta<0的时候，对计算新的alphas值
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        #使用chipAlpha对alphas[j]进行修正，确保其在区间[L,H]
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        #在更新alphas[j]之后，更新误差缓存值
        updateEkk(oS,j)
        #比较新的alphas[j]和alphasold,如果变化很小，返回0
        if (abs(oS.alphas[j]-alphasJold)<0.00001):
            print("j not moving enough");return 0
        #当alpha[j]发生足够大变化时，同时改变alphas[i]，但是改变的方向相反
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*\
        (alphasJold-oS.alphas[j])
        #在更新alphas[i]之后，更新误差缓存值
        updateEkk(oS,i)
        #根据Ei和oS.X[i,:]计算b1
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphasIold)*oS.K[i,i]\
         - oS.labelMat[j]*(oS.alphas[j]-alphasJold)*oS.K[i,j]
        #根据Ej和oS.X[j,:]计算b1
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphasIold)*oS.K[i,j]\
         - oS.labelMat[j]*(oS.alphas[j]-alphasJold)*oS.K[j,j]
        #判断不同情形下，oS.b的取值
        if (0<oS.alphas[i]) and (oS.C > oS.alphas[i]):oS.b = b1
        elif (0<oS.alphas[j]) and (oS.C > oS.alphas[j]):oS.b = b2
        else:oS.b = (b1+b2)/2.0
        return 1
    else:return 0



#外部循环函数
def smoPk(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    #调用optStruct,初始化必要的参数
    oS = optStructk(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    #记录迭代次数
    iter = 0
    #设置entireSet和记录alpha改变的次数
    entireSet = True;alphaPairsChanged = 0
    #构造循环执行的条件，当小于迭代次数并且alphas改变次数大于0，或者entireSet=True
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            #遍历每一行数据
            for i in range(oS.m):
                #调用innerL函数，计算最优的alphas[j]，同时更新alphas[i]，计算当前的b值
                alphaPairsChanged += innerLk(i,oS)
                #记录alphas改变的次数
                print("fullSet,iter:%d i:%d,pairs changed %d"%\
                (iter,i,alphaPairsChanged))
            #记录遍历所有数据集的次数
            iter +=1
        else:
            #遍历一次之后,alphas就会有改变，不全为零，取出不全为零的点
            nonBoundIs = nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            #遍历不为零对应所在行数据
            for i in nonBoundIs:
                alphaPairsChanged += innerLk(i,oS)
                print("non-bound,iter:%d i:%d,pairs changed %d"%\
                (iter,i,alphaPairsChanged))
            iter +=1
        if entireSet:entireSet = False
        elif (alphaPairsChanged==0):entireSet = True
        print("iteration number:%d"%iter)
    return oS.b,oS.alphas




def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoPk(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print ("there are %d Support Vectors"% shape(sVs)[0])
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]):errorCount += 1
    print ("the training error rate is:%f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd])+b
        if sign(predict)!= sign(labelArr[i]):errorCount+=1
    print("the test error rate is :%f" %(float(errorCount)/m))


#===========================================================================

#example of digit recognition

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoPk(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print ("there are %d Support Vectors"% shape(sVs)[0])
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]):errorCount += 1
    print ("the training error rate is:%f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!= sign(labelArr[i]):errorCount+=1
    print("the test error rate is :%f" %(float(errorCount)/m))

