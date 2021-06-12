# text_mining.py
# text mining applying several mthods to make the data meaningful
# author: Ryusei Sashida
# created: 26 Mrach 2021

import math
import numpy as np
from collections import Counter
import pandas as pd
from itertools import chain 
import re
import requests
import time
import warnings
#To igonore the warning for countVectorizer(), which is about the notification of the actual words
#such as "couldn" "wouldn". This is the due to od tokenization of first part.
warnings.simplefilter('ignore')

#To observe the exutin time
init_time=time.time()

#data directory and file 
DATA_DIR='data/text_data/'
DATA_FILE='Corona_NLP_train.csv'

#read data
data=pd.read_csv(DATA_DIR+DATA_FILE, encoding="latin-1")

#make the data into dataframe
df=pd.DataFrame(data)
#the number of documenst
docNum=df.shape[0]

#-----part1----
print("----part1----")
#print all sentiment possible value
print("The possible sentiments:", (df["Sentiment"].unique()))

#print the secin most popular sentiment
print("The second most popular sentiment:")
print(df["Sentiment"].value_counts()[1:2])
    
#Find the date when the Extremely Positive tweets are posted at the most 
ext=df.loc[df['Sentiment']=='Extremely Positive']
tweetAtCounter = ext['TweetAt'].value_counts().index.tolist()
print("The most popular date of Extremely Positive tweet : ", tweetAtCounter[0] )

#make OriginalTweet lower case
df['OriginalTweet']=df['OriginalTweet'].str.lower()
#replace non alphabetic words white-space
df['OriginalTweet']=df['OriginalTweet'].str.replace('[^a-zA-Z]', ' ',regex=True)
# ensure that the words of a message are separated by a single whitespace
df['OriginalTweet']=df['OriginalTweet'].str.strip().replace({' +':' '},regex=True)
print()


#----part2---
print("----part2----")
#make tweet to 2d array, that each message separeted by whit space.
words=df['OriginalTweet'].str.split(" ").to_list()
#make 2darry to 1d
wordList=list(chain.from_iterable(words))
#wordList = list(np.concatenate(words).flat)
#make the array to series
wordSeries=pd.Series(wordList)

#outputs before removing
print("the total number of all words:", len(wordSeries))
print("the number of all distinct words:", len(set(wordSeries)))
print("the 10 most frequent words:")
print(wordSeries.value_counts()[:10])

#load stopwords
stopwords=requests.get( "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt" ).content.decode('utf-8').split( "\n" )

#remove stopwords and words whose length is less tha 2 characters
tweetsSeries=wordSeries[(~wordSeries.isin(stopwords))&(wordSeries.apply(len) > 2)]

#output after removing
print("Remove stop words --  the number of all words:", tweetsSeries.size)
print("Remove stop words --  the number of all distinct words:", tweetsSeries.nunique())
print("Remove stop words --  the 10 most frequent words:")
print(tweetsSeries.value_counts()[:10])
print()


print("----part3----")
#To avoid the words that appered in more than one time, make the set of each message list
tweetSet=pd.Series(words).apply(lambda x : set(x)).to_list()
#make 2d to 1d
tweetSetList=list(chain.from_iterable(tweetSet))
#convert list to Series
tweetSetSeries=pd.Series(tweetSetList)
#remove  stopwords and words whose length is less tha 2 characters
tweetSetSeriesClean=tweetSetSeries[(~tweetSetSeries.isin(stopwords))&(tweetSetSeries.apply(len) > 2)]
#count the frequency of each term and make it to datframe
docCount=tweetSetSeriesClean.value_counts().to_frame()
#set the column name
docCount.columns=["Rate"]
#devided the number of documents that the term appear by the number of ducumets
docCount["Rate"]=docCount["Rate"]/docNum

#plot the fraction of frequent words
plot=docCount.plot.line(rot=15, title="The fraction of frequent words", x_compat=True, xlabel="Term", ylabel="Frequency")
#By bbox_inches="tight", it automatically rescale to show the x label in the saved figure
plot.figure.savefig('outputs/plot.png', bbox_inches="tight")
plot.figure.show()

#plot the logarithm of fraction of frequent words 
docCount["Rate"]=np.log(docCount["Rate"])
plotLog=docCount.plot.line(rot=15, title="The logarithm of fraction of frequent words", x_compat=True, xlabel="Term", ylabel="Frequency")
plotLog.legend(["Log of Rate"])
#By bbox_inches="tight", it automatically rescale to show the x label in the saved figure
plotLog.figure.savefig('outputs/plotLog.png', bbox_inches="tight")
#plot the logarithm of fraction of frequent words after 10000th term
plotLogEx=docCount[10000:].plot.line(rot=15, title="The logarithm of fraction of frequent words(after 10000th terms)", x_compat=True, xlabel="Term", ylabel="Frequency")
plotLogEx.legend(["Log of Rate"])
#By bbox_inches="tight", it automatically rescale to show the x label in the saved figure
plotLogEx.figure.savefig('outputs/plotLogEx.png', bbox_inches="tight")
plotLog.figure.show()
plotLogEx.figure.show()
print("The plots will be shown")

#As explained in report, this is how to compute the numbe of tweets for the constant part
#The value of constant part
minVal=docCount["Rate"][-1]
#The numbe of tweets for the constant part
tweetNum=int(docNum*2**minVal)
print("The lowest value,", minVal, ", means the word appeard in", tweetNum, "tweets(documents)")
print("The numnber of documents:", docNum)
print("The numnber of terms appeared in more than", tweetNum, "tweets:", docCount[docCount["Rate"] > minVal].size) 
print()


print("----part4----")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#convert tweest to numpy
corpus=df["OriginalTweet"].to_numpy()
#convert target to numpy
Y=df["Sentiment"].to_numpy()

#Multinomial Naive Bayse model
model = MultinomialNB()

#CountVectorizer with removing stopw and trivial words.
vectorizer=CountVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b\w\w\w+\b")
#fit and trnasfrom teh corpus for vectorizer
X=vectorizer.fit_transform(corpus)

#train and evaluate the score for the model
model.fit(X, Y)
score=model.score(X, Y)
print("The accuracy:", score)
print("Training error:", 1-score)
print()

#print out the exution time
end_time = time.time()
print("excution time:", end_time-init_time)
