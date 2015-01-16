# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 12:20:57 2015

@author: Nikolay_Semyachkin
"""

#Yet another solution to Kaggle Titanic Challenge.
#Features to be used:
#    -Sex
#    -Class
#    -Number of relatives
#    -Ticket fare
#Suggested solution: group all passengers by choosed features (above) and calculate statistical probability of survival.
#Use this probability to estimate chance of survival for test data set: 0.5+ = survived, else = not.


import numpy as np
import pandas as pd
import pylab as P
from numpy import linalg
import csv

def calcerror(actual, target):
    err = 0
    for i in range(len(actual)):
        if actual[i] != target[i]:
            err += 1
    return err/len(actual)
    
    
df = pd.read_csv("train.csv", header = 0)
#df = df.drop(['Name'], axis=1)
#print(len(df[df.Sex.isnull()]))

df['Relatives'] = df['SibSp'] + df['Parch']

traindf = df.drop(['Name', 'Age', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch'], axis = 1)

#map male:1; female:0
traindf.Sex = traindf.Sex.map({'male':1, 'female':0})
train_array = traindf.values

targetv = train_array[0:,1]
# In order to analyse the price column I need to bin up that data
# here are my binning parameters, the problem we face is some of the fares are very large
# So we can either have a lot of bins with nothing in them or we can just lose some
# information by just considering that anythng over 39 is simply in the last bin.
# So we add a ceiling
fare_ceiling = 40
fare_bracket_size = 10

num_of_classes = len(np.unique(train_array[0:,2]))
num_of_relatives = 5
num_of_brackets = 4

survival_table = np.zeros([2,num_of_classes, num_of_relatives, num_of_brackets],float)
# then modify the data in the Fare column to = 39, if it is greater or equal to the ceiling
#So the fare bins I'm going to make are: 0-9; 10-19; 19-29; 29-39; 39+
train_array[train_array[0:,4] >= fare_ceiling, 4] = fare_ceiling - 1.

#I'm taking into account number of relaives. For simplicity I'm going to separate passengers in groups with 0, 1, 2 and 2+ relatives.
# So I'm changing all values in Relatives column above 2 to 3:
train_array[train_array[0:,5] > 3, 5] = 4

for i in range(num_of_classes):
    for j in range(num_of_relatives):
        for m in range(num_of_brackets):
            women_stats = train_array[(train_array[0:,3] == 0) & (train_array[0:, 2] == i+1) \
            & (train_array[0:, 5] == j) & (train_array[0:,4] >= (m*fare_bracket_size)) & (train_array[0:,4] < ((m+1)*fare_bracket_size)),1]
           
            men_stats = train_array[(train_array[0:,3] == 1) & (train_array[0:, 2] == i+1) \
            & (train_array[0:, 5] == j) & (train_array[0:,4] >= (m*fare_bracket_size)) & (train_array[0:,4] < ((m+1)*fare_bracket_size)),1]
            
            survival_table[0,i,j,m] = sum(women_stats) / len(women_stats) if len(women_stats)>0 else 0
            survival_table[1,i,j,m] = sum(men_stats) / len(men_stats) if len(men_stats)>0 else 0
            
            survival_table[ survival_table < 0.5 ] = 0
            survival_table[ survival_table >= 0.5 ] = 1
#hypov = np.array([])
#for i in train_array:
##    np.append(hypov, survival_table[i[3]] [i[2]] [i[5]] [int(i[4]/10)])
##    np.append(hypov, survival_table[int(i[3])] [int(i[2])] [int(i[5])] [int(i[4]/10)])
#    hypov = np.append(hypov, survival_table[i[3]] [i[2]-1] [i[5]] [int(i[4]/10)])
#


#print(calcerror(hypov, targetv))

testdf = pd.read_csv("test.csv", header = 0)
testdf['Relatives'] = testdf['SibSp'] + testdf['Parch']
testdf = testdf.drop(['Name', 'Age', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch'], axis = 1)
#map male:1; female:0
testdf.Sex = testdf.Sex.map({'male':1, 'female':0})

#testdf[testdf.Fare != testdf.Fare, 3] = 10



test_array = testdf.values

test_array[test_array[0:,3] >= fare_ceiling, 3] = fare_ceiling - 1.
test_array[test_array[0:,4] > 2, 4] = 3

#EXPLAIN YOURSELF:
test_array[test_array[0:,3] != test_array[0:,3],3] = 10

hypov = np.array([])
for i in test_array:
#    np.append(hypov, survival_table[i[3]] [i[2]] [i[5]] [int(i[4]/10)])
#    np.append(hypov, survival_table[int(i[3])] [int(i[2])] [int(i[5])] [int(i[4]/10)])
    hypov = np.append(hypov, survival_table[i[2]] [i[1]-1] [i[4]] [int(i[3]/10)])


prediction_file = open('group_aver_gend_rltv_fare_class.csv', 'w', newline = '')
prediction_file_csv = csv.writer(prediction_file)
prediction_file_csv.writerow(["PassengerId", "Survived"])


for i in range(len(hypov)):
    prediction_file_csv.writerow([i+892, int(hypov[i])])
prediction_file.close()