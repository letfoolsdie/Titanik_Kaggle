# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 12:32:53 2014

@author: Nikolay_Semyachkin
"""


import numpy as np
import pandas as pd
import pylab as P
import csv
from sklearn.ensemble import RandomForestClassifier 

df = pd.read_csv("train.csv", header = 0)
testdf = pd.read_csv("test.csv", header = 0)
#for i in range(1,4):
#    print(i, len(df[(df.Sex=='male')&(df.Pclass==i)]))

#df['Age'].dropna().hist(bins=24)
#P.show()

median_ages = np.zeros((2,3))

df['Gender'] = df.Sex.map({'female':0, 'male':1})
df['AgeFill'] = 0

for i in range(2):
    for j in range(1,4):
        median_ages[i][j-1] = df['Age'][(df.Gender == i) & (df.Pclass == j)].median()
        
for i in range(2):
    for j in range(1,4):
        df.AgeFill[df.Age.isnull() == False] = df.Age
        df.AgeFill[df.Age.isnull() & (df.Gender == i) & (df.Pclass == j)] = median_ages[i][j-1]
        
        
df['AgeIsNull'] = df.Age.isnull().astype(int)
df['FamilySize'] = df['SibSp']+df['Parch']
df['Age*Class'] = df['AgeFill']*df['Pclass']

df = df.drop(['Name', 'Age', 'Ticket', 'Cabin', 'Embarked', 'Sex', 'PassengerId'], axis = 1)

train_data = df.values


# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

median_ages = np.zeros((2,3))

testdf['Gender'] = testdf.Sex.map({'female':0, 'male':1})
testdf['AgeFill'] = 0

for i in range(2):
    for j in range(1,4):
        median_ages[i][j-1] = testdf['Age'][(testdf.Gender == i) & (testdf.Pclass == j)].median()
        
for i in range(2):
    for j in range(1,4):
        testdf.AgeFill[testdf.Age.isnull() == False] = testdf.Age
        testdf.AgeFill[testdf.Age.isnull() & (testdf.Gender == i) & (testdf.Pclass == j)] = median_ages[i][j-1]
        
        
testdf['AgeIsNull'] = testdf.Age.isnull().astype(int)
testdf['FamilySize'] = testdf['SibSp']+testdf['Parch']
testdf['Age*Class'] = testdf['AgeFill']*testdf['Pclass']
testdf[testdf.Fare.isnull()] = testdf.Fare.median()
testdf = testdf.drop(['Name', 'Age', 'Ticket', 'Cabin', 'Embarked', 'Sex', 'PassengerId'], axis = 1)
test_data = testdf.values



# Take the same decision trees and run it on the test data
output = forest.predict(test_data)

prediction_file = open('randomforest_py.csv', 'w', newline = '')
prediction_file_csv = csv.writer(prediction_file)
prediction_file_csv.writerow(["PassengerId", "Survived"])
for i in range(len(testdf)):
    prediction_file_csv.writerow([i+892, output[i]])