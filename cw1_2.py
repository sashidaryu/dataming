# cw1_2.py
# data exploration using downloaded data to run k-means
# author: Ryusei Sashida
# created: 16 Feb 2021

import sys
import csv
import math
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import itertools

DATA_DIR  = 'data/'
DATA_FILE = 'wholesale_customers.csv'
#PLOT_DIR  = '../plots/'



#--
# dataByAttribute
# data formed by attribute
# inputs:
#  data = multi dimetional array
# output:
#  dataByAttAll multi dimetional array
#--
def dataByAttribute(data):
    #Initialize with the number fields dimenstional array to store data by each attribute
    dataByAttAll=[[] for j in range(len(data[0]))]
    #store data by each attribute
    for rec in rawdata:
        for i in range(len(rec[:len(rec)])):
            dataByAttAll[i].append(rec[i])     
    return dataByAttAll

#get data from a file
try:
#open data file in csv format
    f = open( DATA_DIR + DATA_FILE, 'rt', encoding="utf8", errors='ignore')
#read contents of data file into "rawdata" list
    rawdata0 = csv.reader( f )
#parse data in csv format and unncessary field
    rawdata = [rec[2:] for rec in rawdata0]
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

##########2.1
print("---------2.1---------")
#store data by each attribute
dataByAttr=dataByAttribute(rawdata)

tableData=[]
intData=[]

#compute mean and range
for i in range(len(header)):
    intDataByAttr=[int(raw) for raw in dataByAttr[i]]
    intData.append(intDataByAttr)
    mean=sum(intDataByAttr)/len(intDataByAttr)
    minimum=min(intDataByAttr)
    maximum=max(intDataByAttr)
    print(header[i], mean, "["+str(minimum)+", "+str(maximum)+"]")



print()
print("---------2.2---------")
print("Plots will be shown")
#All data converted into Int
intrawData=[]
for i in range(len(rawdata)):
    intRaw=[int(raw) for raw in rawdata[i]]
    intrawData.append(intRaw)

n_clusters = 3
colors = ['red', 'blue', 'green']
#Comibination list of fields
combos=list(itertools.combinations(header, 2))

#Transpose
intrawDataT=np.array(copy.deepcopy(intrawData)).T

#run k-means
model = KMeans(n_clusters=n_clusters).fit(intrawData)
results = model.labels_
centers = model.cluster_centers_

for i in range(len(combos)):
    #Initialize plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    #Indexes of each field
    x_col_index=header.index(combos[i][0])
    y_col_index=header.index(combos[i][1])

    #Data for each field of combination
    x=np.array(intData[x_col_index])
    y=np.array(intData[y_col_index])

    #Field names
    xlabel=combos[i][0]
    ylabel=combos[i][1]

    for t in range(n_clusters):
        ax.scatter(intrawDataT[x_col_index][results==t], intrawDataT[y_col_index][results==t], color=colors[t], alpha=0.5)
        ax.scatter(centers[t, 0], centers[t, 1], marker='x', color=colors[t], s=300)
   
    ax.set_title('k-means', size=16)
    ax.set_xlabel(xlabel, size=14)
    ax.set_ylabel(ylabel, size=14)
    #plt.savefig( PLOT_DIR +"2-2-"+str(i)+'.png' )
    plt.show()
    plt.close()
print("All plots were shown")

##########2.3
print()
print("---------2.3---------")
#K sets
n_clusters = [3,5,10]

#Combination of fields
combos=list(itertools.combinations(header, 2))

#15 colors
colors = ['red', 'blue', 'green', 'black', 'pink', 'purple', 'yellow', 'coral', 'aqua', 'azure', 'brown', 'darkblue', 'gold', 'grey', 'khaki']

BC = np.zeros( 11 ) # between cluster
WC = np.zeros( 11 ) # within cluster
# set number of instances
M = len( rawdata )

for K in range(len(n_clusters)):
    print("K=",n_clusters[K])
    km = KMeans(n_clusters=n_clusters[K]).fit(intrawData)
    results = km.labels_
    centers = km.cluster_centers_
    
    #print(centers)
    for i in range(len(combos)):
        #Initialize plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        #Indexes of each field
        x_col_index=header.index(combos[i][0])
        y_col_index=header.index(combos[i][1])

        #Data for each field of combination
        x=np.array(intData[x_col_index])
        y=np.array(intData[y_col_index])

        #Field names
        xlabel=combos[i][0]
        ylabel=combos[i][1]
        
        for t in range(n_clusters[K]):
            ax.scatter(intrawDataT[x_col_index][results==t], intrawDataT[y_col_index][results==t], color=colors[t], alpha=0.5)
            ax.scatter(centers[t, 0], centers[t, 1], marker='x', color=colors[t], s=300)

        ax.set_title('k-means', size=16)
        ax.set_xlabel(xlabel, size=14)
        ax.set_ylabel(ylabel, size=14)
        #plt.savefig( PLOT_DIR +"2-3-"+str(i)+'.png' )
        #plt.show()
        plt.close()
        
    #members of each cluster
    members = [[] for i in range( n_clusters[K] )] # lists of members of each cluster
    for j in range( M ): # loop through instances
        members[ km.labels_[j] ].append( j ) # add this instance to cluster returned by scikit function

    #compute the within-cluster score
    within = np.zeros(( n_clusters[K] ))
    for i in range( n_clusters[K] ): # loop through all clusters
        within[i] = 0.0
        for j in members[i]: # loop through members of this cluster
            # tally the distance to this cluster centre from each of its members
            within[i] += (np.square( intrawData[j][0]-km.cluster_centers_[i][0]) + np.square(intrawData[j][1]-km.cluster_centers_[i][1])
                          + np.square(intrawData[j][2]-km.cluster_centers_[i][2]) + np.square(intrawData[j][3]-km.cluster_centers_[i][3])
                          + np.square(intrawData[j][4]-km.cluster_centers_[i][4]) + + np.square(intrawData[j][5]-km.cluster_centers_[i][5]))
    WC[n_clusters[K]] = np.sum( within )
    
    #print("inertia:", km.inertia_)

    #compute the between-cluster score
    between = np.zeros(( n_clusters[K] ))
    for i in range( n_clusters[K] ): # loop through all clusters
        between[i] = 0.0
        for l in range( i+1, n_clusters[K] ): # loop through remaining clusters
            # tally the distance from this cluster centre to the centres of the remaining clusters
            between[i] += (np.square( km.cluster_centers_[i][0]-km.cluster_centers_[l][0]) + np.square(km.cluster_centers_[i][1]-km.cluster_centers_[l][1])
                           + np.square(km.cluster_centers_[i][2]-km.cluster_centers_[l][2]) + np.square(km.cluster_centers_[i][3]-km.cluster_centers_[l][3])
                           + np.square(km.cluster_centers_[i][4]-km.cluster_centers_[l][4]) + np.square(km.cluster_centers_[i][5]-km.cluster_centers_[l][5]))
    BC[n_clusters[K]] = np.sum( between )
    
    print("BC:",BC[n_clusters[K]])
    print("WC:",WC[n_clusters[K]])
    #-compute overall clustering score
    score = (BC[n_clusters[K]] / WC[n_clusters[K]])
    print("score:",score)
