# -*- coding: utf-8 -*-
import operator
from math import log


'''创建数据集   '''
def createData():  
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    label=['no surfacing','flippers']
    return dataSet,label


'''计算数据集的信息熵 （信息熵即指类别标签的混乱程度，值越小越好）'''
def calcshan(dataSet):   
    lenDataSet=len(dataSet)
    p={}
    H=0.0
    for data in dataSet:
        currentLabel=data[-1]  #获取类别标签
        if currentLabel not in p.keys():  #若字典中不存在该类别标签，即创建
            p[currentLabel]=0
        p[currentLabel]+=1    #递增类别标签的值
    for key in p:
        px=float(p[key])/float(lenDataSet)  #计算某个标签的概率
        H-=px*log(px,2)  #计算信息熵
    return H
        
            
'''根据某一特征分类数据集'''
def spiltData(dataSet,axis,value):    #dataSet为要划分的数据集,axis为给定的特征，value为给定特征的具体值
    subDataSet=[]
    for data in dataSet:
        subData=[]
        if data[axis]==value:
            subData=data[:axis]  #取出data中第0到axis-1个数进subData;
            subData.extend(data[axis+1:])  #取出data中第axis+1到最后一个数进subData;这两行代码相当于把第axis个数从数据集中剔除掉
            subDataSet.append(subData) #此处要注意expend和append的区别
    return subDataSet


'''遍历所有特征，选择信息熵最小的特征，即为最好的分类特征'''      
def chooseBestFeature(dataSet):  
    lenFeature=len(dataSet[0])-1    #计算特征维度时要把类别标签那一列去掉
    shanInit=calcshan(dataSet)      #计算原始数据集的信息熵
    feature=[]
    inValue=0.0
    bestFeature=0
    for i in range(lenFeature):
        shanCarry=0.0
        feature=[example[i] for example in dataSet]  #提取第i个特征的所有数据
        feature=set(feature)  #得到第i个特征所有的分类值，如'0'和'1'
        for feat in feature:  
            subData=spiltData(dataSet,i,feat)  #先对数据集按照分类值分类
            prob=float(len(subData))/float(len(dataSet))
            shanCarry+=prob*calcshan(subData)  #计算第i个特征的信息熵
        outValue=shanInit-shanCarry  #原始数据信息熵与循环中的信息熵的差
        if (outValue>inValue):
            inValue=outValue  #将信息熵与原始熵相减后的值赋给inValue，方便下一个循环的信息熵差值与其比较
            bestFeature=i
    return bestFeature

  
''' 如果数据集已经处理了所有属性，但是类标签依然不是唯一时使用，采用多数表决的方法定义该节点的分类'''
def majorCount(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():  #若字典中不存在该类别标签，即创建
            classCount[vote]=0
        classCount[vote]+=1                #递增类别标签的值
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)  #对各类标签所出现的频率进行降序排序
    return sortedClassCount[0][0]   #返回出现频率最高的标签值


'''创建我们所要分类的决策树'''
def createTree(dataSet,label):    
    classList=[example[-1] for example in dataSet]   #classList是指当前数据集的类别标签
    if classList.count(classList[0])==len(classList): #计算classList中某个类别标签的数量，若只有一类，则数量与它的数据长度相等
        return classList[0]
    if len(dataSet[0])==1:   #当处理完所有特征而类别标签还不唯一时起作用
        return majorCount(classList)
    featBest=chooseBestFeature(dataSet)  #选择最好的分类特征
    feature=[example[featBest] for example in dataSet]  #接下来使用该分类特征进行分类
    featValue=set(feature)  #得到该特征所有的分类值，如'0'和'1'
    newLabel=label[featBest]
    del(label[featBest])
    Tree={newLabel:{}}  #创建一个多重字典，存储决策树分类结果
    for value in featValue:
        subLabel=label[:]
        Tree[newLabel][value]=createTree(spiltData(dataSet,featBest,value),subLabel) #递归函数使得Tree不断创建分支，直到分类结束
    return Tree


'''使用决策树执行分类，返回分类结果'''
def classify(tree,label,testVec):       #tree为createTree()函数返回的决策树；label为特征的标签值；testVec为测试数据，即所有特征的具体值构成的向量
    firstFeat=list(tree.keys())[0]            #取出tree的第一个键
    secondDict=tree[firstFeat]          #取出tree第一个键的值，即tree的第二个字典（包含关系）
    labelIndex=label.index(firstFeat)   #得到第一个特征firstFeat在标签label中的索引
    for key in secondDict.keys():       #遍历第二个字典的键
        if testVec[labelIndex]==key:    #如果第一个特征的测试值与第二个字典的键相等时
            if type(secondDict[key]).__name__=='dict':  #如果第二个字典的值还是一个字典，说明分类还没结束，递归执行classify函数
                classLabel=classify(secondDict[key],label,testVec)  #递归函数中只有输入的第一个参数不同，不断向字典内层渗入
            else:
                classLabel=secondDict[key]  #最后将得到的分类值赋给classLabel输出
    return classLabel


'''使用pickle模块存储决策树'''
def storeTree(tree,filename):  
    import pickle
    fw=open(filename,'wb')
    pickle.dump(tree,fw)
    fw.close()

'''打开文件取出决策树'''
def loadTree(filename):         
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)

    
'''主函数'''
if __name__=='__main__':
    dataSet,label=createData()
    labelTree=label[:]  #在createTree函数中会改变label的值，所以这里先将其赋值给labelTree，防止label的值改变
    tree=createTree(dataSet,labelTree) #创建决策树
    print(tree)
    print(list(tree.keys()))
    classLabel=classify(tree,label,[1,0])  #分类函数，输出分类结果
    print (classLabel)
    storeTree(tree,'dataTree.txt')
    print (loadTree('dataTree.txt'))
