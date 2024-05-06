#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# In[44]:


mall=pd.read_csv(r"C:\Users\rahul\Downloads\Mall_Customers.csv",index_col=0,header=0)
mall.head()


# In[45]:


print(mall.shape)
print(mall.info())


# In[46]:


mall.describe(include="all")


# In[47]:


mall.isnull().sum()


# In[48]:


X=mall.values[:,[2,3]]
X


# In[49]:


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
 
wsse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i,  random_state = 10)
    kmeans.fit(X)
    wsse.append(kmeans.inertia_)
plt.plot(range(1, 11), wsse)
plt.scatter(range(1, 11),wsse)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WSSE')
plt.show()


# In[50]:


print(wsse)


# In[51]:


# Fitting K-means to the dataset
kmeans=KMeans(n_clusters=5,random_state=10)
Y_pred=kmeans.fit_predict(X)

#kmeans.fit(X)---> training
#Y_pred=kmeans.predict(X)---->predicting


# In[52]:


Y_pred


# In[55]:


kmeans.inertia_


# In[57]:


kmeans.n_iter_


# In[69]:


mall["Clusters"]=Y_pred
mall.head()


# In[70]:


sns.lmplot( data=mall, x='Annual Income (k$)', y='Spending Score (1-100)',
           fit_reg=False, # No regression line
           hue='Clusters',palette="Set1")  
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
     s = 300, c = 'black')
plt.show()


# In[71]:


kmeans.cluster_centers_


# In[61]:


mall["Clusters"]=mall.Clusters.replace({0:"Standard",1:"Target",2:"Sensible",3:"Careless",4:"Careful"})


# In[62]:


mall.head()


# In[63]:


new_df=mall[mall["Clusters"]=="Target"]


# In[64]:


new_df.shape


# In[65]:


new_df


# In[66]:


new_df.to_excel(r"TargetCustomers.xlsx",index=True)


# In[ ]:




