__author__ = 'Administrator'

import csv
from numpy import  *
import operator


def toInt(arr):
    arr=mat(arr)
    m,n=shape(arr)
    newArr=zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            newArr[i,j]=int(arr[i,j])
    return newArr

def nomalizing(arr):
    m,n=shape(arr)
    for i in xrange(m):
        for j in xrange(n):
            if arr[i,j]!=0:
                arr[i,j]=1
    return arr

def loadTrainData():
    l=[]
    with open("train.csv") as fil:
        lines=csv.reader(fil)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l=array(l)
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)),toInt(label)

def loadTestData():
    l=[]
    with open("test.csv") as fil:
        lines=csv.reader(fil)
        for line in lines:
            l.append(line)
        l.remove(l[0])
        data=array(l)
        return nomalizing(toInt(data))

def classify(inX,dataset,labels,k):
    print 'loading...'
    inX=mat(inX)
    dataset=mat(dataset)
    labels=mat(labels)
    dataSetSize=dataset.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataset
    sqDiffMat=array(diffMat)**2
    sqDistances=sqDiffMat.sum(axis=1)
    distance=sqDistances**0.5
    sortedDistIndicies=distance.argsort()
    classCount={}
    for i in range(k):
        voteiLabel=labels[sortedDistIndicies[i]]
        classCount[voteiLabel]=classCount.get(voteiLabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def saveResult(result):
    with open('result.csv','wb') as myfile:
        mywriter=csv.writer(myfile)
        for i in result:
            temp= []
            temp.append(i)
            mywriter.writerow(temp)

def saveLabel():
    with open('label_result.csv','wb') as myfile:
        mywriter=csv.writer(myfile)
        for i in range(28001):
            temp= [i]
            mywriter.writerow(temp)


def handwritingClassTest():
    print "loading..."
    trainData,trainLabel=loadTrainData()
    testData=loadTestData()
    m,n=shape(testData)
    resultList=[]
    for i in range(1000):
        classifierResult=classify(testData[i],trainData[0:5000],trainLabel[0:5000],5)
        resultList.append(classifierResult)
    saveResult(resultList)

#handwritingClassTest()

saveLabel()


