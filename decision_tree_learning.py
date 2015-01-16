# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:06:18 2015

@author: Nikolay_Semyachkin
"""
import math
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


def entropy(pos, neg):
    return -pos*math.log(pos, 2) - neg*math.log(neg, 2)
    
    
def decision_tree(data, feature, col, prev_entropy):
    entropy_gain = 0
    for i in feature:
        selected = data[data[0:,col] == i]
        pos = sum(selected[0:,1])/len(selected) #% of survived passengers
        neg = 1 - pos #% of passengers who died
        if pos == 0 or pos == 1:
            ent = 0
        else:
            ent = entropy(pos, neg) * len(selected)/len(data)
        entropy_gain += ent
    return prev_entropy - entropy_gain
    
df = pd.read_csv("train.csv", header = 0)
#df = df.drop(['Name'], axis=1)
#print(len(df[df.Sex.isnull()]))

traindf = df.drop(['Name', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis = 1)

#map male:1; female:0
#traindf.Sex = traindf.Sex.map({'male':1, 'female':0})
train_array = traindf.values

fare_ceiling = 40
fare_bracket_size = [0, 1, 2, 3, 4]
#train_array[train_array[0:,4] >= fare_ceiling, 4] = fare_ceiling - 1.

targetv = train_array[0:,1]

num_of_survived = sum(targetv)
num_of_passeng = len(targetv)
start_entropy = entropy(num_of_survived / num_of_passeng, 1 - num_of_survived / num_of_passeng)

pclass = list(set(train_array[0:,2])) #equal to np.unique(train_array[0:,2])
sex = list(set(train_array[0:,3]))
sibsp = list(set(train_array[0:,4]))
parch = list(set(train_array[0:,5]))

