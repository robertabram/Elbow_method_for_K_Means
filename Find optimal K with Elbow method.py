#!/usr/bin/env python
# coding: utf-8

# In[11]:


# import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# read data

df = pd.read_csv('Wholesale customers data.csv')

df.head(10)

df.dtypes


# In[3]:


# split categorical and continuos columns

cat_col = ['Channel','Region']
con_col = ['Fresh  Milk','Grocery','Frozen','Detergents_Paper Delicassen']

df.describe().transpose()


# In[4]:


# describe

df.describe().transpose()

sns.distplot(df['Fresh'], hist = True)


# In[5]:


# convert categorical data into binary

for col in cat_col:
    dummies = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df,dummies],axis = 1)
    df.drop(col, axis = 1, inplace = True)

df.head()


# In[9]:


# convert into binary continuos data

scale = MinMaxScaler()
scale.fit(df)
scaled_data = scale.transform(df)

print(scaled_data)


# In[7]:


# find the sum of squared distance by inertia

ssd = []

K = range(1,15)

for k in K:
    km = KMeans(n_clusters = k)
    km = km.fit(scaled_data)
    ssd.append(km.inertia_)


# In[15]:


# plot elbow method for Sum of Squared Distance and K
plt.plot(K, ssd, 'rx-')
plt.xlabel('K labels')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow method for optimal K')
plt.show()

