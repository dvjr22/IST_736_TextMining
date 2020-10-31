#!/usr/bin/env python
# coding: utf-8

# In[41]:


import re
import csv
import random

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


# In[ ]:


# process initial file to convert to two rows (review, sentiment)
file = open('moviereview_v2.csv', 'w')

with open('moviereview.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for row in reader:
        sentiment = row[-1]
        row.pop()  
        file.write(re.sub('\W+',' ', ''.join(row)) + ',' + sentiment+'\n')
        
file.close()


# In[35]:


# format for sentiment analysis
docs = []
docs_v =[]

with open('moviereview_v2.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for row in reader:
        
        docs.append((row[0].split(), row[1]))
        docs_v.append(row[0])


# In[45]:


len(docs)
docs = docs[1:] # get rid of header

# shuffle so not in order
random.shuffle(docs)
random.shuffle(docs)
random.shuffle(docs)

# create training and test docs
train_docs = docs[:1800]
test_docs = docs[1800:]


# # Sentiment Analyzer

# In[29]:


sentim_analyzer = SentimentAnalyzer()

all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in train_docs])

unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)

len(unigram_feats)


# In[33]:


sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
training_set = sentim_analyzer.apply_features(train_docs)
test_set = sentim_analyzer.apply_features(test_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)


# In[34]:


for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))


# # Vader

# In[48]:


random.shuffle(docs_v)
random.shuffle(docs_v)
random.shuffle(docs_v)


# In[46]:


docs_v = docs_v[1:]


# In[53]:


sid = SentimentIntensityAnalyzer()

for s in docs_v:
    #print(s)
    ss = sid.polarity_scores(s)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()
    


# In[ ]:




