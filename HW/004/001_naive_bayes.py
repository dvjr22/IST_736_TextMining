#!/usr/bin/env python
# coding: utf-8

# In[37]:


import nltk
import pandas as pd
import numpy as np
import sklearn
import csv
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


# In[6]:


# moviereview_v2.csv was cleaned in hw 1

file = open('deception_data_converted_final_v2.csv', 'w')

with open('deception_data_converted_final.csv') as csvfile:
    
    reader = csv.reader(csvfile, delimiter = ',')
    file.write(','.join(next(reader)) +'\n') # header
    for row in reader:
        
        file.write(row[0] + ',' + row[1] + ',' + re.sub('\W+',' ', ''.join(row[2:])) +'\n') #data stripped and formatted

file.close()


# In[7]:


csv_file = 'deception_data_converted_final_v2.csv'

df_review = pd.read_csv(csv_file, encoding = "ISO-8859-1")
df_review.dropna(inplace=True)
print(df_review.shape)


# In[8]:


df_review.head()


# In[9]:


lieCol = df_review['lie']
sentCol = df_review['sentiment']
df_review.drop(['lie','sentiment'], axis=1, inplace=True)
df_review.head()


# In[156]:


my_list = []
shortword = re.compile(r'\W*\b\w{1,3}\b')
my_list = [shortword.sub('', df_review.iloc[i,0]) for i in range(len(df_review))]


# In[157]:


my_list[9]


# In[172]:


count_vec = CountVectorizer(input="content", analyzer = 'word', stop_words = 'english', lowercase = True)
cv = count_vec.fit_transform(my_list)

# stolen from Dr. Gates
MyColumnNames=count_vec.get_feature_names()
VectorizedDF_Text=pd.DataFrame(cv.toarray(),columns=MyColumnNames)


# In[173]:


print(VectorizedDF_Text.shape) # check size of vecorization before

threshold = 0 # threshold for drop
to_drop = [] # store indexes needed to be dropped

# id all indexes that don't meet threshold       
to_drop =  [i for i in range (len(VectorizedDF_Text.columns)) if sum(VectorizedDF_Text.iloc[:,i]) < threshold]  

# drop columns that don't meet threshold
VectorizedDF_Text.drop(VectorizedDF_Text.columns[to_drop],axis=1,inplace=True)

print(VectorizedDF_Text.shape)  # check size of vecorization after


# In[174]:


VectorizedDF_Text.head()


# In[175]:


final_vector_df = VectorizedDF_Text.iloc[:,0:].apply(lambda x : x/len(VectorizedDF_Text)) # frequency of words

final_vector_df.insert(0, 'Lie', lieCol.to_frame())
final_vector_df.insert(1, 'Sentiment', sentCol.to_frame())

final_vector_df.head()


# In[190]:


train_df, test_df = train_test_split(final_vector_df, test_size=0.3)
print(train_df.shape)
print(test_df.shape)


# In[191]:


train_df.insert(2, 'SL', train_df[['Sentiment', 'Lie']].apply(lambda x: ''.join(x), axis=1))
test_df.insert(2, 'SL', test_df[['Sentiment', 'Lie']].apply(lambda x: ''.join(x), axis=1))


nb_model_s = MultinomialNB()
nb_model_l = MultinomialNB()
nb_model_ls = MultinomialNB()
sentiment_l = train_df['Sentiment']
lie_l = train_df['Lie']
lieSent = train_df['SL']


train_NB = train_df.drop(['Lie','Sentiment','SL'],axis=1)

nb_model_s.fit(train_NB, sentiment_l)
nb_model_l.fit(train_NB, lie_l)
nb_model_ls.fit(train_NB, lieSent)


# In[192]:


actual_l = test_df['Lie']
actual_s = test_df['Sentiment']
actual_ls = test_df['SL']
test_NB = test_df.drop(['Lie','Sentiment','SL'],axis=1)

prediction_s = nb_model_s.predict(test_NB)
prediction_l = nb_model_l.predict(test_NB)
prediction_ls = nb_model_ls.predict(test_NB)


# In[180]:


print(prediction_l)
print(actual_l)


# In[181]:


print(prediction_s)
print(actual_s)


# In[193]:


print(prediction_ls)
print(actual_ls)


# In[194]:


cm_s = confusion_matrix(actual_l, prediction_l)
cm_l = confusion_matrix(actual_s, prediction_s)
cm_sl = confusion_matrix(actual_ls, prediction_ls)


# In[196]:


print(cm_s)
print()
print(cm_l)
print()
print(cm_sl)


# In[184]:


np.round(nb_model_l.predict_proba(test_NB))


# In[ ]:




