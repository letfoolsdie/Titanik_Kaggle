# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 16:14:34 2014

@author: Nikolay_Semyachkin
"""

import csv
import numpy as np

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('train.csv')) 
header = next(csv_file_object)  # The next() command just skips the 
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = np.array(data) 	         # Then convert from a list to an array
			         # Be aware that each item is currently
                                 # a string in this format

num_passengers = len(data[0:,0])
num_survived = sum(data[0:,1].astype(np.float))
proportion_survivors = num_survived / num_passengers

women_only_stats = data[data[0:,4] == 'female']
men_only_stats = data[data[0:,4] != 'female']

women_onboard = len(women_only_stats)
women_survived = sum(women_only_stats[0:,1].astype(np.float))
men_survived = sum(men_only_stats[0:,1].astype(np.float))
men_onboard = len(men_only_stats)

proportion_women_survived = women_survived / women_onboard
proportion_men_survived = men_survived / men_onboard

print('Proportion of women who survived is', proportion_women_survived)
print('Proportion of men who survived is', proportion_men_survived)



#prediction_file = open('genderprediction_py.csv', 'w', newline = '')
#prediction_file_csv = csv.writer(prediction_file)
#prediction_file_csv.writerow(["PassengerId", "Survived"])
#for row in test_file:
#    if row[3] == 'female':
#        prediction_file_csv.writerow([row[0], '1'])
#    else:
#        prediction_file_csv.writerow([row[0], '0'])
##test_file.close()
#prediction_file.close()

result = np.zeros([2, 3, 4])
fare_limit = 40
num_of_brackets = 4
num_of_classes = 3
bracket_step = fare_limit / num_of_brackets

#we'll group fare in 4 groups (can change #of groups later): <10; 10<20; 20<30; 30+. As far as we need 
#bins to be of the same size, we need to change all fares more than 40 to 39. 
#Thus, the following line is written:
data[data[0:,9].astype(float) > 40, 9] = fare_limit - 1

for i in range(num_of_classes):
    for j in range(num_of_brackets):
        #go through the loop, searching for males which are from class i and paid for ticket more
#    than j*bracket_size and less than (j+1)*bracket_size
        men_stats = data[(data[0:,4] != 'female') & (data[0:,2].astype(float) == i+1) \
        & (data[0:,9].astype(float) >= j*bracket_step) & (data[0:,9].astype(float) < (j+1)*bracket_step)]
        
        women_stats = data[(data[0:,4] == 'female') & (data[0:,2].astype(float) == i+1) \
        & (data[0:,9].astype(float) >= j*bracket_step) & (data[0:,9].astype(float) < (j+1)*bracket_step)]
        
        result[1,i,j] = np.mean(men_stats[0:,1].astype(float))
        result[0,i,j] = np.mean(women_stats[0:,1].astype(float))
        
result[result != result] = 0



test_file = csv.reader(open('test.csv'))
header = next(test_file)

prediction_file = open('genderclassprediction_py.csv', 'w', newline = '')
prediction_file_csv = csv.writer(prediction_file)
prediction_file_csv.writerow(["PassengerId", "Survived"])
for row in test_file:
    for j in range(num_of_brackets):
        try:
            fare = float(row[8])
        except:
            bin_fare = 3 - float(row[1])
            break
        if float(fare) >= fare_limit:
            bin_fare = num_of_brackets - 1
            break
        else:
            if (fare >= j*bracket_step) and (fare < (j+1)*bracket_step):
                bin_fare = j
                break
#    survived = 0
    
    
    if row[3] == 'female':
#        survived = 1 if float(result[0,float(row[1])-1,bin_fare])>=0.5 else 0
        prediction_file_csv.writerow([row[0], (1 if float(result[0,float(row[1])-1,bin_fare])>=0.5 else 0)])
    else:
#        survived = 1 if float(result[1,float(row[1])-1,bin_fare])>=0.5 else survived = 0
        prediction_file_csv.writerow([row[0], (1 if float(result[1,float(row[1])-1,bin_fare])>=0.5 else 0)])

prediction_file.close()