#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string


# In[2]:


# moviereview_v2.csv was cleaned in hw 1

csv_file = 'moviereview_v2.csv'

df_movie = pd.read_csv(csv_file)
print(df_movie.shape)
df_movie = df_movie.dropna()
print(df_movie.shape)


# In[3]:


df_movie.head()


# In[4]:


reviews = df_movie["reviewclass"] # keep sentiment

df_movie.drop(["reviewclass"], axis=1, inplace=True) # drop sentiment for vectorization
df_movie.head()


# In[5]:


'''
my_list=[] # list to store results
shortword = re.compile(r'\W*\b\w{1,3}\b') # filtering out all words with len <= 3
for i in range(len(df_movie)):

    my_list.append(shortword.sub('', df_movie.iloc[i,0]))
'''


# In[6]:


my_list=[] # list to store results
shortword = re.compile(r'\W*\b\w{1,3}\b') # filtering out all words with len <= 3

my_list = [shortword.sub('', df_movie.iloc[i,0]) for i in range(len(df_movie))]


# In[7]:


my_list[5]


# In[18]:


count_vec = CountVectorizer(input="content")
cv = count_vec.fit_transform(my_list)

# stolen from Dr. Gates
MyColumnNames=count_vec.get_feature_names()
VectorizedDF_Text=pd.DataFrame(cv.toarray(),columns=MyColumnNames)

NEW_Labels = reviews.to_frame()  

NEW_Labels.index =NEW_Labels.index-1

LabeledCLEAN_DF=VectorizedDF_Text
LabeledCLEAN_DF["LABEL"]=NEW_Labels


# In[19]:


print(VectorizedDF_Text.shape) # check size of vecorization before

threshold = 1000 # threshold for drop
to_drop = [] # store indexes needed to be dropped

# id all indexes that don't meet threshold

'''
for i in range (len(VectorizedDF_Text.columns)-1):
    if sum(VectorizedDF_Text.iloc[:,i]) < threshold:
        to_drop.append(i)
'''        
to_drop =  [i for i in range (len(VectorizedDF_Text.columns)-1) if sum(VectorizedDF_Text.iloc[:,i]) < threshold]  

# drop columns that don't meet threshold
VectorizedDF_Text.drop(VectorizedDF_Text.columns[to_drop],axis=1,inplace=True)

print(VectorizedDF_Text.shape)  # check size of vecorization after


# In[20]:


VectorizedDF_Text.head()


# In[35]:


VectorizedDF_Text.iloc[:,0:82].apply(lambda x : x/len(VectorizedDF_Text)) # frequency of words


# In[ ]:




