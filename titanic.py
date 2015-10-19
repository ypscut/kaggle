from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import  matplotlib.pyplot as plt

train = pd.read_csv("titanic_train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("titanic_test.csv", dtype={"Age": np.float64}, )

print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#train.to_csv('copy_of_the_training_data.csv', index=False)

train["Age"]=train["Age"].fillna(-1)
test["Age"]=test["Age"].fillna(-1)

train.loc[train["Sex"]=="male","Sex"]=0
test.loc[test["Sex"]=="male","Sex"]=0
train.loc[train["Sex"]=="female","Sex"]=1
test.loc[test["Sex"]=="female","Sex"]=1

print(train["Embarked"].unique())
train["Embarked"]=train["Embarked"].fillna("S")
test["Embarked"]=test["Embarked"].fillna("S")

train.loc[train["Embarked"]=="S","Embarked"]=0
train.loc[train["Embarked"]=="C","Embarked"]=1
train.loc[train["Embarked"]=="Q","Embarked"]=2


test.loc[test["Embarked"]=="S","Embarked"]=0
test.loc[test["Embarked"]=="C","Embarked"]=1
test.loc[test["Embarked"]=="Q","Embarked"]=2

train["Fare"]=train["Fare"].fillna(train["Fare"].median())
test["Fare"]=test["Fare"].fillna(test["Fare"].median())

#Generating a familysize column 
train["FamilySize"]=train["SibSp"]+train["Parch"]
test["FamilySize"]=train["SibSp"]+test["Parch"]

train["NameLength"]=train["Name"].apply(lambda x:len(x))
test["NameLength"]=test["Name"].apply(lambda x:len(x))

import re 

def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
train_titles = train["Name"].apply(get_title)
test_titles=test["Name"].apply(get_title)

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2,"Dona":9}
for k,v in title_mapping.items():
    train_titles[train_titles == k] =v
    test_titles[test_titles==k]=v



# Add in the title column.
train["Title"] = train_titles
test["Title"]= test_titles

#print (test["Title"])



import operator

# A dictionary mapping family name to id
family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

# Get the family ids with the apply method
train_family_ids = train.apply(get_family_id, axis=1)
test_family_ids = test.apply(get_family_id,axis=1)

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
train_family_ids[train["FamilySize"] < 3] = -1
test_family_ids[test["FamilySize"]<3]=-1


train["FamilyId"] = train_family_ids
test["FamilyId"]=test_family_ids





alg = AdaBoostClassifier()
#alg=RandomForestClassifier(random_state=1,n_estimators=150,min_samples_split=4,min_samples_leaf=2)
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]


# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(train[predictors], train["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

print("#######")
predictors = ["Pclass", "Sex", "Fare","Title"]


x_train = train[predictors]
y_train = train['Survived']
x_test=test[predictors]
alg.fit(x_train,y_train)
predictions = alg.predict(x_test)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv('submission.csv', index=False)










