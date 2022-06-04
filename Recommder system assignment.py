#!/usr/bin/env python
# coding: utf-8

# Q-Problem statement.
# Build a recommender system by using cosine simillarties score.

# In[211]:


#import library
import pandas as pd
import numpy as np


# In[220]:


books_df = pd.read_csv('E:\\DATA SCIENCE\\LMS\\ASSIGNMENT\\MY ASSIGNMENT\\RecommderSystem\\book.csv', encoding ='unicode_escape')
books_df = books_df.drop(['Unnamed: 0'], axis=1)
books_df.head()


# In[221]:


books_df.info()


# In[234]:


# drop duplicated and rename columns
books_df1=books_df.drop_duplicates(['User.ID'])
books_df1=books_df1.rename(columns={"User.ID": "userid", "Book.Title": "Booktitle","Book.Rating":"Bookrating"})
books_df1


# In[255]:


#change visulization of DataFrame using pivot
books_df2=books_df1.pivot(index='userid',columns='Booktitle',values='Bookrating').reset_index(drop=True)
books_df2.head()


# In[256]:


books_df2.index=books_df1.userid.unique()
books_df2


# In[257]:


#books_df3=books_df2.fillna(0,inplace=True)
books_df2.fillna(0,inplace=True)


# In[258]:


books_df2


# In[259]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[261]:


#Build a recommender system by using cosine simillarties score
user1=1-pairwise_distances(books_df2.values,metric='cosine')
user1


# In[263]:


user1_df=pd.DataFrame(user1)
user1_df


# In[267]:


user1_df.index=books_df1.userid.unique()
user1_df.columns=books_df1.userid.unique()


# In[271]:


np.fill_diagonal(user1,0)


# In[281]:


user2


# In[277]:


user2.idxmax(axis=1)

