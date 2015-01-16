# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 17:39:32 2015

@author: Nikolay_Semyachkin
"""

import numpy as np
import pandas as pd
import pylab as P
from numpy import linalg
import csv

def evaluatehypofunction(w, vector):
    y = []    
    for i in vector:
        if np.dot(w, i) > 0.5:
            y.append(1)
        else:
            y.append(0)
    return np.array(y)
    
    
def calcerror(actual, target):
    err = 0
    for i in range(len(actual)):
        if actual[i] != target[i]:
            err += 1
    return err/len(actual)
    
    
df = pd.read_csv("train.csv", header = 0)
#df = df.drop(['Name'], axis=1)
#print(len(df[df.Sex.isnull()]))
traindf = df.drop(['Name', 'Age', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis = 1)

#map male:1; female:0
traindf.Sex = traindf.Sex.map({'male':1, 'female':0})
train_array = traindf.values

##from available data sepatate 100 entries for hypothesys testing:
#data = train_array[:(len(train_array)-100),2:]
#target_value = train_array[:(len(train_array)-100),1]
#test_data = train_array[100:,2:]
#test_value = train_array[100:,1]

#OK, now train on full dataset:
data = train_array[:,2:]
target_value = train_array[:,1]

#So I tested on available data, let's not test on provided test data (with no 'Survived' column) to upload solution to Kaggle:
testdf = pd.read_csv("test.csv", header = 0)
testdf = testdf.drop(['Name', 'Age', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
testdf.Sex = testdf.Sex.map({'male':1, 'female':0})
test_array = testdf.values
test_data = test_array[:,1:]

#create vectors and calculate w:
vectors = [[1.0, data[i][0], data[i][1], data[i][2], data[i][3]] for i in range(len(data))]
dagger = np.dot(linalg.inv(np.dot(np.transpose(vectors), vectors)), np.transpose(vectors))
w = np.dot(dagger, target_value)
hypov = evaluatehypofunction(w, vectors)

#calculate in sample error:
print(calcerror(hypov, target_value))
#
outvectors = [[1.0, test_data[i][0], test_data[i][1], test_data[i][2], test_data[i][3]] for i in range(len(test_data))]
hypov = evaluatehypofunction(w, outvectors)
#
#
##calculate out of sample error:
#print(calcerror(hypov, test_value))

prediction_file = open('LR_w_class_sex_fare_sibsp_py.csv', 'w', newline = '')
prediction_file_csv = csv.writer(prediction_file)
prediction_file_csv.writerow(["PassengerId", "Survived"])
for i in range(len(hypov)):
    prediction_file_csv.writerow([i+892, int(hypov[i])])
prediction_file.close()