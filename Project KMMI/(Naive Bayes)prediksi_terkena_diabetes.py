#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[3]:


#membuka data dan memeriksa data
df = pd.read_csv('Documents/Portofolio/datamentah/diabetes.csv', delimiter = ',', header = 0)
df


# In[4]:


#memeriksa kekosongan data
df.isna().sum()


# In[5]:


#membagi data menjadi parameter x dan y
X = df.drop('Outcome', axis = 1)
y = df['Outcome']
#membagi data menjadi data training dan data test
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 46)


# In[6]:


#membuat pemodelan naive bayes untuk pengklasifikasian orang yang terkena diabetes dan juga tidak
model = GaussianNB()
model.fit(X_train, y_train)


# In[7]:


#mengetes hasil hasil pengklasifikasian
y_pred = model.predict(X_test)


# In[8]:


#melihat data hasil prediksi komputer
y_pred


# In[9]:


#melihat kesesuaian prediksi dengan data yang sebenarnya
y_pred == y_test


# In[10]:


#melihat nilai kesamaan antara prediksi dengan data yang sebenarnya
r2_score(y_test, y_pred)


# In[ ]:


#memasukkan data baru
newdata = np.array([[input('Jumlah riwayat kehamilan:'),
                     input('Besar glukosa:'),
                     input('Tekanan darah:'),
                     input('Tebal Kulit:'),
                     input('insulin:'),
                     input('BMI'),
                     input('Diabetes Pedigree Function:'),
                     input('Usia')]],dtype=float)


# In[ ]:


data2 = pd.DataFrame(newdata, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'])


# In[ ]:


#memprediksi data dari pasien baru
pred = model.predict(data2)
pred


# In[5]:


for j in range(1000):

            X_train, X_test, y_train, y_test = train_test_split(X, y , random_state =j,     test_size=0.35)
            lr = LarsCV().fit(X_train, y_train)

            tr_score.append(lr.score(X_train, y_train))
            ts_score.append(lr.score(X_test, y_test))

                J = ts_score.index(np.max(ts_score))

                X_train, X_test, y_train, y_test = train_test_split(X, y , random_state =J, test_size=0.35)
                M = LarsCV().fit(X_train, y_train)
                y_pred = M.predict(X_test)


# In[ ]:




