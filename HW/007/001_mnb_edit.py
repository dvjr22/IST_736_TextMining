#!/usr/bin/env python
# coding: utf-8

# # Tutorial - build MNB with sklearn

# This tutorial demonstrates how to use the Sci-kit Learn (sklearn) package to build Multinomial Naive Bayes model, rank features, and use the model for prediction. 
# 
# The data from the Kaggle Sentiment Analysis on Movie Review Competition are used in this tutorial. Check out the details of the data and the competition on Kaggle.
# https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
# 
# The tutorial also includes sample code to prepare your prediction result for submission to Kaggle. Although the competition is over, you can still submit your prediction to get an evaluation score.

# # Step 1: Read in data

# In[62]:


# read in the training data

# the data set includes four columns: PhraseId, SentenceId, Phrase, Sentiment
# In this data set a sentence is further split into phrases 
# in order to build a sentiment classification model
# that can not only predict sentiment of sentences but also shorter phrases

# A data example:
# PhraseId SentenceId Phrase Sentiment
# 1 1 A series of escapades demonstrating the adage that what is good for the goose is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story .1

# the Phrase column includes the training examples
# the Sentiment column includes the training labels
# "0" for very negative
# "1" for negative
# "2" for neutral
# "3" for positive
# "4" for very positive

import numpy as np
import pandas as p


train=p.read_csv("train.tsv", delimiter='\t')
y=train['Sentiment'].values
X=train['Phrase'].values


# In[63]:


# borrowed from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
# made a few modifications

import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# # Step 2: Split train/test data for hold-out test

# In[64]:


# check the sklearn documentation for train_test_split
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# "test_size" : float, int, None, optional
# If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
# If int, represents the absolute number of test samples. 
# If None, the value is set to the complement of the train size. 
# By default, the value is set to 0.25. The default will change in version 0.21. It will remain 0.25 only if train_size is unspecified, otherwise it will complement the specified train_size.    

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train[0])
print(y_train[0])
print(X_test[0])
print(y_test[0])


# Sample output from the code above:
# 
# (93636,) (93636,) (62424,) (62424,)
# almost in a class with that of Wilde
# 3
# escape movie
# 2

# # Step 2.1 Data Checking

# In[65]:


# Check how many training examples in each category
# this is important to see whether the data set is balanced or skewed

unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)))


# The sample output shows that the data set is skewed with 47718/93636=51% "neutral" examples. All other categories are smaller.
# 
# {0, 1, 2, 3, 4}
# [[    0  4141]
#  [    1 16449]
#  [    2 47718]
#  [    3 19859]
#  [    4  5469]]

# # Exercise A

# In[66]:


# Print out the category distribution in the test data set. 
#Is the test data set's category distribution similar to the training data set's?

# Your code starts here
training_labels = set(y_train)
print(training_labels)
from scipy.stats import itemfreq
#training_category_dist = itemfreq(y_train) # deprecated
training_category_dist = np.unique(y_train, return_counts=True)
print(training_category_dist)
# Your code ends here


# # Step 3: Vectorization

# In[67]:


# sklearn contains two vectorizers

# CountVectorizer can give you Boolean or TF vectors
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# TfidfVectorizer can give you TF or TFIDF vectors
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

# Read the sklearn documentation to understand all vectorization options

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# several commonly used vectorizer setting

#  unigram boolean vectorizer, set minimum document frequency to 5
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=5, stop_words='english')

#  unigram term frequency vectorizer, set minimum document frequency to 5
unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=5, stop_words='english')

#  unigram and bigram term frequency vectorizer, set minimum document frequency to 5
gram12_count_vectorizer = CountVectorizer(encoding='latin-1', ngram_range=(1,2), min_df=5, stop_words='english')

#  unigram tfidf vectorizer, set minimum document frequency to 5
unigram_tfidf_vectorizer = TfidfVectorizer(encoding='latin-1', use_idf=True, min_df=5, stop_words='english')


# ## Step 3.1: Vectorize the training data

# In[68]:


# The vectorizer can do "fit" and "transform"
# fit is a process to collect unique tokens into the vocabulary
# transform is a process to convert each document to vector based on the vocabulary
# These two processes can be done together using fit_transform(), or used individually: fit() or transform()

# fit vocabulary in training documents and transform the training documents into vectors
X_train_vec = unigram_count_vectorizer.fit_transform(X_train)
#X_train_vec = unigram_tfidf_vectorizer.fit_transform(X_train)

# check the content of a document vector
print(X_train_vec.shape)
print(X_train_vec[0].toarray())

# check the size of the constructed vocabulary
print(len(unigram_count_vectorizer.vocabulary_))

# print out the first 10 items in the vocabulary
print(list(unigram_count_vectorizer.vocabulary_.items())[:10])

# check word index in vocabulary
print(unigram_count_vectorizer.vocabulary_.get('imaginative'))


# Sample output:
# 
# (93636, 11967)
# [[0 0 0 ..., 0 0 0]]
# 11967
# [('imaginative', 5224), ('tom', 10809), ('smiling', 9708), ('easy', 3310), ('diversity', 3060), ('impossibly', 5279), ('buy', 1458), ('sentiments', 9305), ('households', 5095), ('deteriorates', 2843)]
# 5224

# ## Step 3.2: Vectorize the test data

# In[69]:


# use the vocabulary constructed from the training data to vectorize the test data. 
# Therefore, use "transform" only, not "fit_transform", 
# otherwise "fit" would generate a new vocabulary from the test data

X_test_vec = unigram_count_vectorizer.transform(X_test)
#X_test_vec = unigram_tfidf_vectorizer.transform(X_test)

# print out #examples and #features in the test set
print(X_test_vec.shape)


# Sample output:
# 
# (62424, 14324)

# # Exercise B

# In[70]:


# In the above sample code, the term-frequency vectors were generated for training and test data.

# Some people argue that 
# because the MultinomialNB algorithm is based on word frequency, 
# we should not use boolean representation for MultinomialNB.
# While in theory it is true, you might see people use boolean representation for MultinomialNB
# especially when the chosen tool, e.g. Weka, does not provide the BernoulliNB algorithm.

# sklearn does provide both MultinomialNB and BernoulliNB algorithms.
# http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
# You will practice that later

# In this exercise you will vectorize the training and test data using boolean representation
# You can decide on other options like ngrams, stopwords, etc.

# Your code starts here

# Your code ends here


# # Step 4: Train a MNB classifier

# In[71]:


# import the MNB module
from sklearn.naive_bayes import MultinomialNB

# initialize the MNB model
nb_clf= MultinomialNB()

# use the training data to train the MNB model
nb_clf.fit(X_train_vec,y_train)


# # Step 4.1 Interpret a trained MNB model

# In[72]:


## interpreting naive Bayes models
## by consulting the sklearn documentation you can also find out feature_log_prob_, 
## which are the conditional probabilities
## http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

# the code below will print out the conditional prob of the word "worthless" in each category
# sample output
# -8.98942647599 -> logP('worthless'|very negative')
# -11.1864401922 -> logP('worthless'|negative')
# -12.3637684625 -> logP('worthless'|neutral')
# -11.9886066961 -> logP('worthless'|positive')
# -11.0504454621 -> logP('worthless'|very positive')
# the above output means the word feature "worthless" is indicating "very negative" 
# because P('worthless'|very negative) is the greatest among all conditional probs

unigram_count_vectorizer.vocabulary_.get('worthless')
for i in range(0,5):
  print(nb_clf.feature_log_prob_[i][unigram_count_vectorizer.vocabulary_.get('worthless')])


# Sample output:
# 
# -8.5389826392
# -10.6436375867
# -11.8419845779
# -11.4778370023
# -10.6297551464

# In[73]:


# sort the conditional probability for category 0 "very negative"
# print the words with highest conditional probs
# these can be words popular in the "very negative" category alone, or words popular in all cateogires

feature_ranks = sorted(zip(nb_clf.feature_log_prob_[0], unigram_count_vectorizer.get_feature_names()))
very_negative_features = feature_ranks[-10:]
print(very_negative_features)
print()
feature_ranks = sorted(zip(nb_clf.feature_log_prob_[4], unigram_count_vectorizer.get_feature_names()))
very_positive_features = feature_ranks[-10:]
print(very_positive_features)


# In[74]:


for i in very_positive_features:
    print(str(i[1])+','+str(i[0])) 


# Sample output for print(log_ratios[0])
# 
# -0.838009538739

# # Step 5: Test the MNB classifier

# In[76]:


# test the classifier on the test data set, print accuracy score

nb_clf.score(X_test_vec,y_test)


# In[77]:


# print confusion matrix (row: ground truth; col: prediction)

from sklearn.metrics import confusion_matrix
y_pred = nb_clf.fit(X_train_vec, y_train).predict(X_test_vec)
cm=confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
print(cm)

plot_confusion_matrix(y_test, y_pred)


# In[78]:


# print classification report

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))

from sklearn.metrics import classification_report
target_names = ['0','1','2','3','4']
print(classification_report(y_test, y_pred, target_names=target_names))


# # Step 5.1 Interpret the prediction result

# In[79]:


## find the calculated posterior probability
posterior_probs = nb_clf.predict_proba(X_test_vec)

## find the posterior probabilities for the first test example
print(posterior_probs[0])

# find the category prediction for the first test example
y_pred = nb_clf.predict(X_test_vec)
print(y_pred[0])

# check the actual label for the first test example
print(y_test[0])


# sample output array([ 0.06434628  0.34275846  0.50433091  0.07276319  0.01580115]
# 
# Because the posterior probability for category 2 (neutral) is the greatest, 0.50, the prediction should be "2". Because the actual label is also "2", this is a correct prediction
# 

# # Step 5.2 Error Analysis

# In[80]:


# print out specific type of error for further analysis

# print out the very positive examples that are mistakenly predicted as negative
# according to the confusion matrix, there should be 53 such examples
# note if you use a different vectorizer option, your result might be different

err_cnt = 0
for i in range(0, len(y_test)):
    if(y_test[i]==4 and y_pred[i]==1):
        print(X_test[i])
        err_cnt = err_cnt+1
print("errors:", err_cnt)


# # Exercise D

# In[ ]:


########## submit to Kaggle submission

# we are still using the model trained on 60% of the training data
# you can re-train the model on the entire data set 
#   and use the new model to predict the Kaggle test data
# below is sample code for using a trained model to predict Kaggle test data 
#    and format the prediction output for Kaggle submission

# read in the test data
kaggle_test=p.read_csv("/Users/byu/Desktop/data/kaggle/test.tsv", delimiter='\t') 

# preserve the id column of the test examples
kaggle_ids=kaggle_test['PhraseId'].values

# read in the text content of the examples
kaggle_X_test=kaggle_test['Phrase'].values

# vectorize the test examples using the vocabulary fitted from the 60% training data
kaggle_X_test_vec=unigram_count_vectorizer.transform(kaggle_X_test)

# predict using the NB classifier that we built
kaggle_pred=nb_clf.fit(X_train_vec, y_train).predict(kaggle_X_test_vec)

# combine the test example ids with their predictions
kaggle_submission=zip(kaggle_ids, kaggle_pred)

# prepare output file
outf=open('/Users/byu/Desktop/data/kaggle/kaggle_submission.csv', 'w')

# write header
outf.write('PhraseId,Sentiment\n')

# write predictions with ids to the output file
for x, value in enumerate(kaggle_submission): outf.write(str(value[0]) + ',' + str(value[1]) + '\n')

# close the output file
outf.close()


# # Exercise E

# In[ ]:


# generate your Kaggle submissions with boolean representation and TF representation
# submit to Kaggle
# report your scores here
# which model gave better performance in the hold-out test
# which model gave better performance in the Kaggle test


# Sample output:
# 
# (93636, 9968)
# [[0 0 0 ..., 0 0 0]]
# 9968
# [('disloc', 2484), ('surgeon', 8554), ('camaraderi', 1341), ('sketchiest', 7943), ('dedic', 2244), ('impud', 4376), ('adopt', 245), ('worker', 9850), ('buy', 1298), ('systemat', 8623)]
# 245

# # BernoulliNB

# In[ ]:


from sklearn.naive_bayes import BernoulliNB
X_train_vec_bool = unigram_bool_vectorizer.fit_transform(X_train)
bernoulliNB_clf = BernoulliNB(X_train_vec_bool, y_train)


# # Cross Validation

# In[ ]:


# cross validation

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
nb_clf_pipe = Pipeline([('vect', CountVectorizer(encoding='latin-1', binary=False)),('nb', MultinomialNB())])
scores = cross_val_score(nb_clf_pipe, X, y, cv=3)
avg=sum(scores)/len(scores)
print(avg)


# # Exercise F

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

stem_vectorizer = StemmedCountVectorizer(min_df=3, analyzer="word")
X_train_stem_vec = stem_vectorizer.fit_transform(X_train)


# In[ ]:


# check the content of a document vector
print(X_train_stem_vec.shape)
print(X_train_stem_vec[0].toarray())

# check the size of the constructed vocabulary
print(len(stem_vectorizer.vocabulary_))

# print out the first 10 items in the vocabulary
print(list(stem_vectorizer.vocabulary_.items())[:10])

# check word index in vocabulary
print(stem_vectorizer.vocabulary_.get('adopt'))


# In[ ]:





# In[ ]:




