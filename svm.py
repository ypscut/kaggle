__author__ = 'Administrator'
import csv
from numpy import  *
import operator
from sklearn.ensemble import RandomForestClassifier




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
    return nomalizing(toInt(data)), toInt(label)

def loadTestData():
    l=[]
    with open("test.csv") as fil:
        lines=csv.reader(fil)
        for line in lines:
            l.append(line)
        l.remove(l[0])
        data=array(l)
        return nomalizing(toInt(data))
def saveResult(result):
    with open('svm_result.csv','wb') as myfile:
        mywriter=csv.writer(myfile)
        for i in result:
            temp= [i]
            mywriter.writerow(temp)

def handwritingClassTest():
    print "loading..."
    trainData,trainLabel=loadTrainData()
    testData=loadTestData()
    m,n=shape(testData)
    resultList=[]
    clf=
    clf.fit(trainData,trainLabel)
    for i in range(m):
        classifierResult=clf.predict(testData[i])
        resultList.append(int(classifierResult[0]))
    saveResult(resultList)

handwritingClassTest()




############


import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

t_row=train.shape[0]
t_col=train.shape[1]

x_train=train[,1:t_col]
y_train=train[,0]




alg = AdaBoostClassifier()
#alg =model = RandomForestClassifier(n_estimators=100)

predictions = alg.predict(test)

result = pd.DataFrame({
        "ImageId": test["ImageId"],
        "Label": predictions
    })

submission.to_csv('result.csv', index=False)










