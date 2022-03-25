#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_excel("CC GENERAL (1).xlsx")
df.head()


# In[2]:


df.shape


# In[3]:


df = df.drop(["CUST_ID"],axis=1)


# In[4]:


df.isnull().sum()


# In[5]:


df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean(skipna=True), inplace=True)


# In[6]:


df.isnull().sum()


# In[7]:


df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean(skipna=True), inplace=True)


# In[8]:


df.isnull().sum()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


from sklearn.cluster import AgglomerativeClustering 
model=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
clust_labels=model.fit_predict(df) 


# In[12]:


import pandas as pd
agglomerative=pd.DataFrame(clust_labels)
agglomerative


# In[13]:


import matplotlib.pyplot as plt
fig =plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter (df['MINIMUM_PAYMENTS'] , df["PAYMENTS"] , c= agglomerative[0], s=50)
ax.set_title("Agglomerative Clutering")
ax.set_xlabel("MINIMUM_PAYMENTS")
ax.set_ylabel("PAYMENTS")
plt.colorbar(scatter)


# In[ ]:





# In[14]:


import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,7))
plt.title(" Dendrograms")
dend=shc.dendrogram(shc.linkage(df, method="complete"))


# In[15]:


from sklearn.cluster import KMeans  
kmeans=KMeans(n_clusters=3, random_state=0) 
kmeans.fit(df)


# In[16]:


labels=pd.DataFrame(kmeans.labels_)
labels


# In[17]:


kmeans.predict(df)
print(kmeans.cluster_centers_) 


# In[18]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
plt.scatter(df["BALANCE"][df.label == 0],          
            df["PURCHASES"][df.label == 0],s=80,c='magenta',label='Careful')
plt.scatter(df["BALANCE"][df.label == 1],
           df["PURCHASES"][df.label == 1],s=80,c='yellow',label='Standard')
plt.scatter(df["BALANCE"][df.label == 2],
           df["PURCHASES"][df.label == 2],s=80,c='green',label='Target')
plt.scatter(df["BALANCE"][df.label == 3], 
           df["PURCHASES"][df.label == 3],s=80,c='cyan',label='Careless')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label= 'Centroids')
plt.title('Clusters of Marketing')
plt.xlabel('BALANCE')
plt.ylabel('PURCHASES')
plt.legend()
plt.show()


# In[ ]:





# In[24]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[22]:


sum_of_squared_distance =[]
k=range(1,15)
for k in k:
    km= KMeans(n_clusters=k)
    km= km.fit(df)
    sum_of_squared_distance.append(km.inertia_)


# In[25]:


plt.plot(k,sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('sum_of_squared_distances')
plt.title('Elbow Methode For optimal k')
plr.show()


# In[ ]:


.# K-Means may be computationally faster than hierarchical clustering andproduce tighter clusters than hierarchical clustering 



