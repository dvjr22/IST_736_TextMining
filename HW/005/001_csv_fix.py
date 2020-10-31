#!/usr/bin/env python
# coding: utf-8

# In[50]:


import csv
import re
import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score


# In[49]:


#read_file = 'Batch_3847752_batch_results.csv'
#file = open('Batch_3847752_batch_results_v2.csv', 'w')
line_count = 1
new_line = 3

with open(read_file) as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for row in reader:
        
        #print(row)
        if line_count%new_line == 0:
            line = line + ', ' + row[-1] + '\n'
            file.write(line)
        elif line_count%new_line == 2:
            line = line + ', ' + row[-1]
        else:
            line = row[0] + ',' + row[-1]
        
        line_count = line_count+1
        
file.close()


# In[70]:


csv_file = 'Batch_3847752_batch_results_v2.csv'

df_movie = pd.read_csv(csv_file)
print(df_movie.shape)
df_movie = df_movie.dropna()
print(df_movie.shape)


# In[76]:


df_movie.head(5)


# In[72]:



print(cohen_kappa_score(df_movie['worker1'].astype('category'),df_movie['worker2'].astype('category')))
print(cohen_kappa_score(df_movie['worker1'].astype('category'),df_movie['worker3'].astype('category')))
print(cohen_kappa_score(df_movie['worker2'].astype('category'),df_movie['worker3'].astype('category')))


# In[73]:


df_movie['orginal']= np.where((df_movie.orginal == 'neg'), 'Negative', df_movie.orginal)
df_movie['orginal']= np.where((df_movie.orginal == 'pos'), 'Positive', df_movie.orginal)


# In[77]:


print(cohen_kappa_score(df_movie['worker1'].astype('category'),df_movie['orginal'].astype('category')))
print(cohen_kappa_score(df_movie['worker2'].astype('category'),df_movie['orginal'].astype('category')))
print(cohen_kappa_score(df_movie['worker3'].astype('category'),df_movie['orginal'].astype('category')))


# In[ ]:




