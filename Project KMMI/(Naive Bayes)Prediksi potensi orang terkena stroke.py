#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Memprediksi pasien yang rawan terkena penyakit stroke dengan menggunakan datasets yang sudah ada sebelumnya yang diambil dari kaggle


# In[2]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# In[3]:


# Membuka datasest csv kemudian menghapus kolom id dan juga kolom pernikahan karena rata-rata sudah menikah
df = pd.read_csv('Documents/Portofolio/datamentah/stroke.csv', delimiter = ',', header = 0)
df = df.drop(columns = ['id', 'ever_married'])
df


# In[4]:


#melihat kekosongan data
df.isna().sum()


# In[5]:


#membersihkan kekosongan data dengan menghapus nilai Nan
df.dropna(inplace = True)
df.isna().sum()


# In[6]:


#mengubah data stirng menjadi bilangan integer agar pemodelan bisa dilakukan dengan mudah
df['gender'] = pd.factorize(df.gender)[0]
df['Residence_type'] = pd.factorize(df.Residence_type)[0]
df['smoking_status'] = pd.factorize(df.smoking_status)[0]
df['work_type'] = pd.factorize(df.work_type)[0]
df


# In[7]:


#melihat persebaran data 
df.describe()


# In[8]:


#memisahkan variabel bebas dan juga terikat dan membagi data tersebut untuk training model dan juga test mode;
X = df.drop('stroke', axis = 1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2,random_state = 1)


# In[9]:


#menggunakan pemodelan Gaussian Naive Bayes untuk mengolah data
model = GaussianNB()
model.fit(X_train, y_train)


# In[10]:


#memprediksi hasil dari pemodelan
y_pred = model.predict(X_test)


# In[11]:


y_pred


# In[12]:


#melihat apakah sama antara data luaran aslinya dengan data luaran dari pemodelan
y_test = y_pred


# In[13]:


#melihat persentase keakuratan luaran dari data yang sudah di prediksi
r2_score(y_test, y_pred)


# In[14]:


#memasukkan data pasien yang baru
newdata = np.array([[input('gender:'),
                     input('age:'),
                     input('hypertension:'),
                     input('heart disease:'),
                     input('work type:'),
                     input('Residence type'),
                     input('avg_glucose_level'),
                     input('bmi:'),
                     input('smoking_status')]],dtype=float)


# In[15]:


data2 = pd.DataFrame(newdata, columns=['gender', 'age', 'hypertension', 'heart_disease', 'work_type','Residence_type','avg_glucose_level', 'bmi', 'smoking_status'])


# In[16]:


#memprediksi data dari pasien baru
#jika 0 maka memiliki potensi stroke rendah, sedangkan 1 memiliki potensi stroke yang tinggi
pred = model.predict(data2)
pred


# In[ ]:


Halo semuanya perkenalkan nama saya Muhammad Ajriel Rahayu, saya merupakan salah satu Mahasiswa di Universitas pendidikan Indonesia dengan program studi Fisik

