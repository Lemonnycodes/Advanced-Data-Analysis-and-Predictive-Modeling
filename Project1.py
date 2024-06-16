# -*- coding: utf-8 -*-
"""Hw1_DharmaSankaran.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BhVV5cmsb4FtOtwOrjc4tjmPZ8vxv-gQ
"""

from google.colab import drive
drive.mount('/content/gdrive')

#import libraries
import numpy as np
import pandas as pd

import scipy as sp
from scipy import spatial

import matplotlib.pyplot as plt

from nltk.stem.lancaster import LancasterStemmer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
import heapq
import string
import re

from nltk.corpus import stopwords
from collections import defaultdict


vectorizer = CountVectorizer()
st = LancasterStemmer()

#read data

with open("gdrive/My Drive/train_file (hw1).csv", "r") as sw:
    linesOfTrainData = sw.readlines()
print(len(linesOfTrainData))


with open("gdrive/My Drive/format_file (hw1).csv", "r") as sw:
    linesOfFormat = sw.readlines()
print(len(linesOfFormat))

with open('gdrive/My Drive/test_file (hw1).csv', "r") as sw:
    linesOfTestData = sw.readlines()
print(len(linesOfTestData))

df=pd.read_csv('gdrive/My Drive/train_file.csv',header=None)
df.head(10)

!pip3 install contractions

import contractions
from tqdm import tqdm

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#donwloading the stopwords of english language
stopwords=stopwords.words('english')
#Removing stopwords 'no','nor' and 'not'
print('not' in stopwords)
stopwords.remove('no')
stopwords.remove('nor')
stopwords.remove('not')
print('not' in stopwords)

nltk_revs=[]
for i in tqdm(df[1]):
    i=re.sub('(<[\w\s]*/?>)',"",i)
    i=contractions.fix(i)
    i=re.sub('[^a-zA-Z0-9\s]+',"",i)
    i=re.sub('\d+',"",i)
    nltk_revs.append(" ".join([j.lower() for j in i.split() if j not in stopwords and len(j)>=3]))

new_df=pd.DataFrame({'review':nltk_revs,'sentiment':list(df[0])})

new_df

new_df['sentiment'].value_counts()[1]

X=new_df['review']
Y=new_df['sentiment'] #labels were assigned

def print_shape(a,b):
    """
    Function that prints the shape of the numpy arrays passed as arguments
    """
    print("Size of Training Samples")
    print("="*30)
    print(a.shape)
    print("Size of Testing Samples")
    print("="*30)
    print(b.shape)
print_shape(x_train,x_test)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,stratify=Y,test_size=0.33)

df.groupby('')['FLIGHT_NUMBER'].count().plot.pie(figsize=(10,10),rot=45)
plt.ylabel('',fontsize=30)
plt.title("Pie chart for number of flights for each airline",fontsize=20)

vectorizer = CountVectorizer(analyzer='word', lowercase = True, stop_words='english')

linesOfTrainData_Transformed = vectorizer.fit_transform(linesOfTrainData)

linesOfTestData_Transformed = vectorizer.transform(linesOfTestData)

feature_names = vectorizer.get_feature_names() 
len(feature_names)

#cosine similarity is determined
def CalculateCosine(vt,vs):
        cosineSimilarityValue = cosine_similarity(vt,vs)
        return cosineSimilarityValue

cosineSimilarityValue = CalculateCosine(linesOfTestData_Transformed,linesOfTrainData_Transformed)

print(len(cosineSimilarityValue))

#KNN ALGORITHM 
f = open('gdrive/My Drive/format_file (hw1) .csv', 'w')
count = 0
for row in cosineSimilarityValue:
    k=72
    partitioned_row_byindex = np.argpartition(-row, k)  
    similar_index = partitioned_row_byindex[:k]
    
    #print(similar_index)
    
    neighbourReviewTypeList = []
    neighbourReviewTypeNegative = 0
    neighbourReviewTypePositive = 0

    for index in similar_index:

        if linesOfTrainData[index].strip()[0] == '-':
            #neighbourReviewTypeList.append("-1")
            neighbourReviewTypeNegative+=1
        elif linesOfTrainData[index].strip()[0] == '+':
            #neighbourReviewTypeList.append("+1")
            neighbourReviewTypePositive+=1
            
    
    if neighbourReviewTypeNegative > neighbourReviewTypePositive:
        f.write('-1\n')
        count+=1
    else:
        f.write('+1\n')
        count+=1
        print("Value: ",count)
print("End of command")

"""CONFUSION MATRIX """

from sklearn.metrics import confusion_matrix,precision_score
print('Precision: %.3f' % precision_score(y_test, labels))
import seaborn as sns
sns.heatmap(confusion_matrix(y_test,labels),annot=True)

def processToVectorize(df2,x_train,y_train):
    nltk_revs=[]
    for i in tqdm(df2[0]):
        i=re.sub('(<[\w\s]*/?>)',"",i)
        i=contractions.fix(i)
        i=re.sub('[^a-zA-Z0-9\s]+',"",i)
        i=re.sub('\d+',"",i)
        nltk_revs.append(" ".join([j.lower() for j in i.split() if j not in stopwords and len(j)>=3]))
    vectorizer=CountVectorizer()
    x_train_vec=vectorizer.fit_transform(x_train)
    x_test_vec=vectorizer.transform(nltk_revs)
    print_shape(x_train_vec,x_test_vec)
    cosinevals = cosine_calculate(x_test_vec,x_train_vec)
    labels = predictKNN2(cosinevals,y_train,70)
    ansdf = pd.DataFrame(labels)
    ansdf.to_csv('answerfile.csv',header=False, index=False)

