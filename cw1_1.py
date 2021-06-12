# cw1_1.py
# data exploration using downloaded data to make decsision tree
# author: Ryusei Sashida
# created: 16 Feb 2021

import sys
import csv
import math
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as model_select
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import tree
from collections import Counter
from statistics import mode 

DATA_DIR  = 'data/'
DATA_FILE = 'adult.csv'
#PLOT_DIR  = '../plots/'

#get data from a file
try:
    f = open( DATA_DIR + DATA_FILE, 'rt', encoding="utf8", errors='ignore')
    rawdata0 = csv.reader( f )
#parse data in csv format and slice fnlwgt
    rawdata = [rec[:2]+rec[3:] for rec in rawdata0]

    rawdataWithoutClass = [rec[:2]+rec[3:len(rec)-1] for rec in rawdata0]
#handle exceptions:
except IOError as iox:
    print('there was an I/O error trying to open the data file: ' + str( iox ))
    sys.exit()
except Exception as x:
    print('there was an error: ' + str( x ))
    sys.exit()

#save header and delete from rest of data array
header = rawdata[0]
del rawdata[0]

print("-----------1.1--------------")
missingCount=0
missingInst=0
for rec in rawdata:
    err = 0
    #slice class
    #count for instance with at least one missing value
    if "" in rec[:-1]:
        missingInst+=1
    #slice class
    #count for missing values
    for data in rec[:-1]:
        if data=="":
            missingCount+=1

print("number of instances", len(rawdata))
print("number of missing values", missingCount)
print("fraction of missing values over all attribute values", missingCount/(len(rawdata)*13))
print("number of instances with missing values", missingInst)
print("fraction of instances with missing values over all instances", missingInst/len(rawdata))



#nominal (1.2)
print()
print("-----------1.2--------------")
#Initialize with the number of fields dimenstional array to store data by each attribute
dataByAttAll=[[] for j in range(len(header))]
#Initialize an array to store labeled data by each attribute
dataByAttAllId=[]

#store data by each attribute
for rec in rawdata:
    for i in range(len(rec[:len(rec)])):
        dataByAttAll[i].append(rec[i])
        
#store labled data by each attribute
for i in range(len(dataByAttAll)):
    le = preprocessing.LabelEncoder()
    le.fit(dataByAttAll[i])
    datas_id = le.transform(dataByAttAll[i]).tolist()
    dataByAttAllId.append(datas_id)
    print(header[i],":" ,set(datas_id))
    #print(header[i],":" ,le.inverse_transform(list(set(datas_id))))



#################tree (1.3)
print()
print("-----------1.3--------------")
#initialize with the number of fields dimenstional array to store data by each attribute without instance that has missing value 
dataByAtt=[[] for j in range(len(header))]
#initialize an array to store labeled data by each attribute without instances that has missing value 
dataByAttId=[]

#initialize with the number of fields dimenstional array to store data by each attribute without instance that has missing value for traininf
dataByAttTrain=[[] for j in range(len(header))]
#initialize an array to store labeled data for training
dataByAttIdTrain=[]

#duplicate the rawdata
rawDataCopy=copy.deepcopy(rawdata)
rawData=[]
rawTarget=[]

#store data by each attribute and drop all instances with missing values
for rec in rawdata:
    if "" not in rec[:len(rec)-1]:
        for i in range(len(rec[:len(rec)])):
            dataByAtt[i].append(rec[i])
        
#store labled data by each attribute
for i in range(len(dataByAtt)):
    le = preprocessing.LabelEncoder()
    le.fit(dataByAtt[i])
    datas_id = le.transform(dataByAtt[i]).tolist()
    dataByAttId.append(datas_id)

dataByAttIdT=copy.deepcopy(np.array(dataByAttId).T)

#store data by each attribute without intances that have missing values
for rec in dataByAttIdT:
    rawData.append(rec[:-1])
    rawTarget.append(rec[-1])

clf = tree.DecisionTreeClassifier()
#4-fold cross-validation
scores = cross_val_score(clf, rawData, rawTarget, cv = 4)
#score for each
print('4-Fold Cross-Validation scores: {}'.format(scores))
#average score
print('Average score: {}'.format(np.mean(scores)))
#average error rate
print('Average error rate: {}'.format(1-np.mean(scores)))


###################tree (1.4)
print()
print("-----------1.4--------------")
D_dash=[]
D_dashTarget=[]
#duplicate the rawdata
rawDataCopy=copy.deepcopy(rawdata)
rawDataTrim=[]
rawTargetTrim=[]

for rec in rawDataCopy:
    #slice class
    #store instances that have missing value to D_dash and D_dashtarget
    if "" in rec[:len(rec)-1]:
        D_dash.append(rec[:-1])
        D_dashTarget.append(rec[-1])
    #store instances that have no missing value to rawDataTrim and rawTargetTrim
    else:
        rawDataTrim.append(rec[:-1])
        rawTargetTrim.append(rec[-1])

#randomly select instances from instances that have no missing value
#The train size is the same number as instances that have missing value
#The test size is all the remaining instances from D
testSize=len(rawDataTrim)-missingInst
x_train, x_test, y_train, y_test = model_select.train_test_split(rawDataTrim, rawTargetTrim, train_size=missingInst, test_size=testSize, random_state=0)

#Before handling missing value
D_dash=copy.deepcopy(D_dash)+x_train
D_dashTarget=D_dashTarget+y_train
D1=copy.deepcopy(D_dash)
D1Target=copy.deepcopy(D_dashTarget)
D2=copy.deepcopy(D_dash)
D2Target=copy.deepcopy(D_dashTarget)

#Replace mising value to "missing" for D1
for s in range(len(D1)):
    for i in range(len(D1[s])):
        if D1[s][i]=='':
            D1[s][i]='missing'

D2ByAtt=[[] for j in range(len(header))]
for rec in D2:
    for i in range(len(rec)):
        D2ByAtt[i].append(rec[i])

D2ByAtt[13]=copy.deepcopy(y_train)
        
#Find most common value in each attribute
commons=[]            
for i in range(len(D2ByAtt[:-1])):
    elem_to_count = (data for data in D2ByAtt[i] if data[:1].isupper())
    c = Counter(elem_to_count)
    #if the data is not numeric
    if c.most_common(1) != []:
        commons.append(c.most_common(1)[0][0])
    #if the data is numeric
    else:
        commons.append(mode(D2ByAtt[i]))

#Set missing value to most common value in each attribute for D2
for rec in D2:
    for i in range(len(rec)):
        if rec[i]=='':
            rec[i]=commons[i]


#transpose train and test data
D1t = []
for i in range(len(D1[0])):
    tr_row = []
    for vector in D1:
        tr_row.append(vector[i])
    D1t.append(tr_row)

D2t = []
for i in range(len(D2[0])):
    tr_row = []
    for vector in D2:
        tr_row.append(vector[i])
    D2t.append(tr_row)

x_test_t = []
for i in range(len(x_test[0])):
    tr_row = []
    for vector in x_test:
        tr_row.append(vector[i])
    x_test_t.append(tr_row)

#After handling missing vlues
D1Target=np.array(D1Target).T.tolist()
D2Target=np.array(D2Target).T.tolist()
y_test=np.array(y_test).T.tolist()
D1data=D1t+[D1Target]
D2data=D2t+[D2Target]
testData=x_test_t+[y_test]

labeledD1=[]
labeledD2=[]
labeled_test_x_d1=[]
labeled_test_x_d2=[]

#The array to store original data by attribute and replace missing value to "missing"
dataByAttM=[]
#Label encodeing for original data with rplacing missing value, D1data, and test data for D1
for i in range(len(dataByAttAll)):
    dataByAttM.append(["missing" if t=="" else t for t in dataByAttAll[i]])
    leD1 = preprocessing.LabelEncoder()
    leD1.fit(dataByAttM[i])
    datas_id_d1 = leD1.transform(D1data[i]).tolist()
    labeledD1.append(datas_id_d1)
    test_id_d1 = leD1.transform(testData[i]).tolist()
    labeled_test_x_d1.append(test_id_d1)

#Label encodeing for original data, D2data, and test data for D2
for i in range(len(dataByAttAll)):
    le = preprocessing.LabelEncoder()
    le.fit(dataByAttAll[i])
    datas_id = le.transform(dataByAttAll[i]).tolist()
    dataByAttId.append(datas_id)
    datas_id_d2 = le.transform(D2data[i]).tolist()
    labeledD2.append(datas_id_d2)
    test_id_d2 = le.transform(testData[i]).tolist()
    labeled_test_x_d2.append(test_id_d2)


#Labled D1data and D1target
labeledD1Target=labeledD1[-1]
labeledD1Data=np.array(labeledD1[:-1]).T

#Labled D2data and D2target
labeledD2Target=labeledD2[-1]
labeledD2Data=np.array(labeledD2[:-1]).T

#Labled testdata for D1 and D2
labeled_test_y_d1=labeled_test_x_d1[-1]
labeled_test_y_d2=labeled_test_x_d2[-1]
labeled_test_x_d1=np.array(labeled_test_x_d1[:-1]).T
labeled_test_x_d2=np.array(labeled_test_x_d2[:-1]).T

#Creat descision tree for D1
clfD1 = tree.DecisionTreeClassifier()
clfD1 = clfD1.fit(labeledD1Data, labeledD1Target)
predictedD1 = list(clfD1.predict(labeled_test_x_d1))
errorCountD1=0

#Compute error rate for D1
for i in range(len(predictedD1)):
    if predictedD1[i] != labeled_test_y_d1[i]:
        errorCountD1+=1
print("D1 error rate:", errorCountD1/len(predictedD1))

#Creat descision tree for D2
clfD2 = tree.DecisionTreeClassifier()
clfD2 = clfD2.fit(labeledD2Data, labeledD2Target)
predictedD2 = list(clfD2.predict(labeled_test_x_d2))
errorCountD2=0

#Compute error rate for D1
for i in range(len(predictedD2)):
    if predictedD2[i] != labeled_test_y_d2[i]:
        errorCountD2+=1
print("D2 error rate:", errorCountD2/len(predictedD2))
