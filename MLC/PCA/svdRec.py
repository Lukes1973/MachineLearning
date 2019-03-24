def loadExData():
    return [[4, 4, 0, 2, 2],
        [4, 0, 0, 3, 3],
        [4, 0, 0, 1, 1],
        [1, 1, 1, 2, 0],
        [2, 2, 2, 0, 0],
        [1, 1, 1, 0, 0],
        [5, 5, 5, 0, 0]]

def loadExData2():
	return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
			[0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
			[0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
			[3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
			[5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
			[0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
			[4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
			[0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
			[0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
			[0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
			[1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


from numpy import *
from numpy import linalg as la

#欧式距离相似度计算法
def ecludSim(inA,inB):
	return 1.0/(1.0+la.norm(inA-inB))
#皮尔逊相关系数相似度计算法
def pearsSim(inA,inB):
	if len(inA)<3: return 1.0
	return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]

#余弦相似度计算法
def cosSim(inA,inB):
	num = float(inA.T*inB)
	denom = la.norm(inA)*la.norm(inB)
	return 0.5+0.5*(num/denom)



#通过SVD降维处理相似度函数

def svdEst(dataMat,user,simMeas,item):
	# 得到商品的数量
	n = shape(dataMat)[1]
	#初始化评分参数
	simTotal = 0.0;ratSimTotal = 0.0
	#使用svd方法对dataMat进行svd分解
	U,Sigma,VT = la.svd(dataMat)
	Sig4 = mat(eye(4)*Sigma[:4])
	#对原矩阵进行降维，注意此处对dataMat先做转置处理，我的理解是此处应用的基于物品的相识度算法
	xformedItems = dataMat.T*U[:,:4]*Sig4.I
	#遍历每个商品
	for j in range(n):
		userRating = dataMat[user,j]
		#排除评价为0商品，不参与相似度计算
		if userRating == 0 or j==item:continue
		#余弦相似度计算
		similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
		print('the %d and %d similarity is :%f' %(item,j,similarity))
		#相似度汇总
		simTotal+=similarity
		#相似度*评分汇总
		ratSimTotal+=similarity*userRating
	if simTotal == 0:return 0
	#计算最终缺失商品相似度
	else:return ratSimTotal/simTotal

#不做降维处理，直接计算相似度函数
def standEst(dataMat,user,simMeas,item):
	#获取商品数量
	n = shape(dataMat)[1]
	#初始化基本参数
	simTotal = 0.0 ;ratSimTotal = 0.0
	#遍历每个商品
	for j in range(n):
		#过滤商品评价为0,不参与相似度计算
		userRating = dataMat[user,j]
		if userRating == 0: continue
		#获取特定两列中所在行都大于0的部分
		overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        #如果不存在两列对应所在行都为零，则直接认为相似度为0
		if len(overLap) ==0: similarity = 0
		#计算相似度
		else:similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
		print('the %d and %d similarity is:%f' % (item,j,similarity))
		simTotal +=similarity
		ratSimTotal+=similarity*userRating
	if simTotal ==0:return 0
	else:return ratSimTotal/simTotal


#推荐函数
def recommend(dataMat,user,N=3,simMeas = cosSim,estMethod  = standEst):
	#获取特定user商品评价为0的列
	unrateItems = nonzero(dataMat[user,:].A==0)[1]
	if len(unrateItems) == 0:return 'you rated everything'
	itemScores = []
	#对每个评价缺失的商品，调用svdEst或者standEst进行相似度计算
	for item in unrateItems:
		#计算获得评分
		estimatedScore = estMethod(dataMat,user,simMeas,item)
		#商品，评分对应保存
		itemScores.append((item,estimatedScore))
	#返回相识度排名前三的未评价商品以及对应计算出来的评分
	return sorted(itemScores,key= lambda jj:jj[1],reverse=True)[:N]

# 使用SVD 来压缩图片

#构建打印矩阵函数
def printMat(inMat,thresh = 0.8):
	for i in range(32):
		for k in range(32):
			if float(inMat[i,k])>thresh:
				print(1,end="")
			else:print(0,end="")
		print('')

#图像像素压缩
def imgCompress(numSV=3,thresh=0.8):
	myl = []
	for line in open('MLC/PCA/0_5.txt').readlines():
		newRow = []
		for i in range(32):
			newRow.append(int(line[i]))
		myl.append(newRow)
	myMat = mat(myl)
	print("****original matrix****")
	printMat(myMat,thresh)
	#先试用SVD分解
	U,Sigma,VT = la.svd(myMat)
	#构造3*3 0值矩阵
	SigRecon = mat(zeros((numSV,numSV)))
	for k in range(numSV):
		SigRecon[k,k] = Sigma[k]
	#利用降维后的矩阵，还原原矩阵，从另一个角度来看，通过分解矩阵达到压缩存储空间的目的
	reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
	print("****reconstructed matrix using %d singular values******"%numSV)
	printMat(reconMat,thresh)








