#!/usr/bin/env python
# coding: utf-8

# In[49]:


import nltk
import pandas as pd
import re  
import os
import numpy as np
import seaborn as sns; sns.set()

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


# In[50]:



file_path = "C:\\Users\\dvjr2\\Google Drive\\Documents\\Syracuse\\IST_736_TextMining\\HW\\003\\FedPapersCorpus"
count_vectorizer = CountVectorizer(input = 'filename', analyzer = 'word', stop_words = 'english', lowercase = True)
count_vectorizer_02 = CountVectorizer(input = 'filename', analyzer = 'word', lowercase = True)


# In[51]:


file_list = []
essay_list = []
for item in os.listdir(file_path):
    
    file_list.append(file_path + "\\" + item)  
    essay_list.append(item)
    
print(file_list[0])
print(essay_list[0])


# In[52]:


CV = count_vectorizer.fit_transform(file_list)
CV2 = count_vectorizer_02.fit_transform(file_list)

columns = count_vectorizer.get_feature_names()
fed_df = pd.DataFrame(CV.toarray(), columns = columns)

columns = count_vectorizer_02.get_feature_names()
fed_df_02 = pd.DataFrame(CV2.toarray(), columns = columns)
#fed_df['Essay'] = essay_list

print(fed_df.shape)
print(fed_df_02.shape)


# In[53]:


fed_df.head()


# In[54]:


fed_df_02.head()


# In[55]:


def toDrop(df, threshold):

    fed_df = df
    #threshold = 100 # threshold for drop
    to_drop = [] # store indexes needed to be dropped
    to_drop =  [i for i in range (len(fed_df.columns)-1) if sum(fed_df.iloc[:,i]) < threshold]  
    
    return to_drop


# In[56]:


to_drop = toDrop(fed_df, 100)
to_drop_02 = toDrop(fed_df_02, 200)

print(fed_df.shape[1] - len(to_drop))
print(fed_df_02.shape[1] - len(to_drop_02))


# In[57]:


# drop columns that don't meet threshold
fed_df.drop(fed_df.columns[to_drop],axis=1,inplace=True)
fed_df_02.drop(fed_df_02.columns[to_drop_02],axis=1,inplace=True)

print(fed_df.shape)  # check size of vecorization after
print(fed_df_02.shape)


# In[58]:


final_df = fed_df.iloc[:,0:].apply(lambda x : x/len(fed_df)) # frequency of words
final_df['Essay'] = essay_list
final_df.head()


# In[59]:


final_df_02 = fed_df_02.iloc[:,0:].apply(lambda x : x/len(fed_df_02)) # frequency of words
final_df_02['Essay'] = essay_list
final_df_02.head()


# In[60]:


def mostWords(df):
    
    final_df = df
    most_words = final_df.sum(axis=0)
    most_words = most_words[0:len(most_words)-1]
    pd.to_numeric(most_words)
    mw = most_words.sort_values(ascending=False)
    return (mw)


# In[61]:


mostWords(final_df).head()


# In[62]:


mostWords(final_df_02).head()


# In[63]:


mw1 = mostWords(final_df)
mw1_words = mw1.index.values[0:10]
final_df.loc[:,mw1_words]

plt.figure(figsize=(20,10))
sns.heatmap(final_df.loc[:,mw1_words])
plt.savefig('heat_01.png')


# In[64]:


mw2 = mostWords(final_df_02)
mw2_words = mw2.index.values[0:10]
final_df_02.loc[:,mw2_words]

plt.figure(figsize=(20,10))
sns.heatmap(final_df_02.loc[:,mw2_words])
plt.savefig('heat_02.png')


# In[65]:


mw3_words = mw2.index.values[25:40]
final_df_02.loc[:,mw3_words]

plt.figure(figsize=(20,10))
sns.heatmap(final_df_02.loc[:,mw3_words])
plt.savefig('heat_03.png')


# In[66]:


mw1.index.values


# In[67]:


mw2.index.values


# In[68]:


count = 0
for word in mw1.index.values:
    if word in mw2.index.values:
        count = count+1

print(count/len(mw1.index.values))


# In[ ]:




