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
                b1 = b - Ei -labelMat[i]*(alphas[i]-alphasIold)*dataMatrix[i,:]*dataMatrix[i,:].T-\
                labelMat[j]*(alphas[j]-alphasIold)*dataMatrix[i,:]*dataMatrix[j,:].T
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

# full version of SMO
#构造对象，保存重要的参数
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        #缓存误差
        self.eCache = mat(zeros((self.m,2)))

#计算误差
def calcEk(oS,k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + os.b
    Ek = fXk -float(oS.labelMat[k])
    return Ek

#根据下标i和误差Ei来计算第二个alphas的值，用来保证每次优化中可以获得最大的步长
def selectJ(i,oS,Ei):
    maxK = -1;maxDeltaE = 0;Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei-Ek)
            if (deltaE>maxDeltaE):
                maxK = k;maxDeltaE = deltaE;Ej=Ek
        return maxK,Ej
    else:
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)
    return j,Ej

#更新误差值到eCache中
def updateEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k]=[1,Ek]

#内部循环函数
def innerL(i,oS):
    Ei = calcEk(oS,i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < os.C)) or \
    ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] >0)):
        j,Ej = selectJ(i,oS,Ei)
        alphasIold = oS.alphas[i].copy();alphasJold = oS.alphas[j].copy();
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(os.C,oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C,oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H");return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:].T -\
        oS.X[i,:]*oS.X[j,:].T
        if eta >= 0:print("eta>=0");return 0
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if (abs(oS.alphas[j]-alphasJold)<0.00001):
            print("j not moving enough");return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*\
        (alphasJold-os.alphas[j])
        updateEk(oS,i)
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphasIold)*\
        os.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*\
        (oS.alphas[j]-alphasJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphasIold)*\
        os.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*\
        (oS.alphas[j]-alphasJold)*oS.X[j,:]*oS.X[j,:].T
        if (0<oS.alphas[i]) and (oS.C > oS.alphas[i]):oS.b = b1
        elif (0<oS.alphas[j]) and (oS.C > oS.alphas[j]):oS.b = b2
        else:oS.b = (b1+b2)/2.0
        return 1
    else:return 0

#外部循环函数
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS = optStruct(mat(dataMat),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True;alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS,m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet,iter:%d i:%d,pairs changed %d"%\
                (iter,i,alphaPairsChanged))
            iter +=1
        else:
            nonBoundIs = nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound,iter:%d i:%d,pairs changed %d"%\
                (iter,i,alphaPairsChanged))
            iter +=1
        if entireSet:entireSet = False
        elif (alphaPairsChanged==0):entireSet = True
        print("iteration number:%d"%iter)
    return oS.b,oS.alphas





















        

























