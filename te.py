__author__ = 'Administrator'
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


X=[[0,0],[1,1]]
y=[0,1]
#clf=MultinomialNB()
#clf=GradientBoostingClassifier()
clf=LogisticRegression()
clf.fit(X,y)
print clf.predict_proba([[2,2]])