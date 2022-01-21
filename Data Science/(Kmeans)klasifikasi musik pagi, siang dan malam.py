#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pada project kali ini membuat prediksi dan juga rekomendasi musik yang enak di dengarkan sesuai dengan waktu pagi, siang,
# ataupun malam,dengan menggunakan pemodelan Kmeans data menggunakan datasets dari kaggle pada februari 2021


# In[2]:


#liblary yang dipakai pada project ini
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing


# In[3]:


#membuka data
df = pd.read_csv('Documents/Portofolio/datamentah/dataspotify.csv')
data = df[['year','popularity','acousticness', 'danceability', 'energy', 'liveness', 'loudness','valence', 'name', 'artists', 'tempo']]
#memeriksa kekosongan data
data.isna().sum()


# In[4]:


#memeriksa nilai data
data.describe()


# In[5]:


#menyeleksi data yang akan digunakan
data2 = data[data['year'] > 2015]
data3= data2[data2['popularity'] > 50]
data4 = data3.drop(['year', 'popularity','artists', 'name'], axis = 1)


# In[6]:


data3.describe()


# In[7]:


minmax = preprocessing.MinMaxScaler().fit_transform(data4)


# In[8]:


data5 = pd.DataFrame(minmax, index = data4.index, columns = data4.columns)


# In[9]:


data5


# In[10]:


#mencari nilai klastering yang baik
scr=[]
for i in range (1,20):
    score = KMeans(n_clusters = i).fit(data5).score(data5)
    print(score)
    scr.append(score)


# In[11]:


#Untuk menentukan jumlah klaster yang baik adalah dengan mengambil nilai klaster pada lengkungan grafik
plt.plot(scr)


# In[12]:


kmeans = KMeans(n_clusters = 3)
kmeans.fit(data5)


# In[13]:


data5['clusters'] = kmeans.labels_
data5


# In[14]:


#melihat jumlah lagu pada masing-masing kluster data, musik pada kluster 0 dan 1 lebih banyak dari pada kluster 2
plt.hist(data5['clusters'])


# In[15]:


#melihat persebaran data melalui grafik, musik pada pagi hari menggunakan klaster 2, siang hari menggunakan klaster 1, dan pada
#malam hari menggunakan klaster 0 karena pada malam hari orang-orang lebih mendengarkan musik yang tenang untuk membantu tidur
sns.pairplot(data5, hue = 'clusters' )


# In[16]:


#menampilkan  judul lagu dan mengurutkan sesuai popularitas lagu
data5[['artists','name', 'popularity']] = data2[['artists','name', 'popularity']]
data5 = data5.sort_values(by = 'popularity', ascending = False)
data5


# In[17]:


#Menampilkan 100 Rekomendasi musik malam
dp = data5[data5['clusters'] == 0]
dp = dp[['artists', 'name', 'popularity']]
dp[:100]


# In[18]:


#Menampilkan 100 rekomendasi musik siang
ds = data5[data5['clusters'] == 1]
ds[:100]
ds = ds[['artists', 'name', 'popularity']]
ds[:100]


# In[19]:


#Menampilkan 100 rekomendasi musik pagi
dm = data5[data5['clusters'] == 2]
dm[:100]
dm = dm[['artists', 'name', 'popularity']]
dm[:100]

