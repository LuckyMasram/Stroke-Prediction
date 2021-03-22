#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('stroke-data.csv')
df.head()


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df[df['gender']=='Other']


# In[7]:


sns.countplot(x='gender', hue='hypertension', data=df)


# In[8]:


sns.countplot(x='gender', hue='heart_disease', data=df)


# In[9]:


sns.countplot(x='gender', hue='smoking_status', data=df)


# In[10]:


sns.countplot(x='gender', hue='stroke', data=df)


# In[11]:


sns.countplot(x='gender', hue='Residence_type', data=df)


# In[12]:


sns.countplot(x='work_type', hue='Residence_type', data=df)


# In[13]:


plt.figure(figsize=(12,8))
sns.countplot(x='gender', hue='work_type', data=df)


# In[14]:


df['bmi'].loc[df['bmi']>50].count()


# In[15]:


df.isnull().sum()


# In[16]:


df['bmi']=np.where(df['bmi'].isnull(),df['bmi'].median(),df['bmi'])


# In[17]:


sns.boxplot(x='bmi', data=df)


# In[18]:


sns.boxplot(x='avg_glucose_level', data=df)


# In[19]:


df.describe()


# In[20]:


def remove_outliers(data):
    arr = []
    q1 = np.percentile(data,25)
    q3 = np.percentile(data,75)
    iqr = q3 - q1
    minm = q1 - (1.5*iqr)
    maxm = q3 + (1.5*iqr)
    
    for i in list(data):
        if i < minm:
            i = minm
            arr.append(i)
        elif i > maxm:
            i = maxm
            arr.append(i)
        else:
            arr.append(i)
    #print(max(arr))
    return arr


# In[21]:


df['bmi']=remove_outliers(df['bmi'])
df['avg_glucose_level']=remove_outliers(df['avg_glucose_level'])


# In[22]:


sns.boxplot(x='bmi', data=df)


# In[23]:


sns.boxplot(x='avg_glucose_level', data=df)


# In[24]:


sns.heatmap(df.corr(), annot=True, fmt='.2f')


# In[25]:


df.head()


# In[26]:


print(df['gender'].value_counts())
print('_'*50)
print(df['work_type'].value_counts())
print('_'*50)
df['smoking_status'].value_counts()


# In[27]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
df['gender'] = enc.fit_transform(df['gender'])
df['ever_married'] = enc.fit_transform(df['ever_married'])
df['work_type'] = enc.fit_transform(df['work_type'])
df['Residence_type'] = enc.fit_transform(df['Residence_type'])
df['smoking_status'] = enc.fit_transform(df['smoking_status'])


# In[28]:


df.head()


# In[29]:


df = df.drop('id', axis=1)


# In[30]:


print(df['gender'].value_counts())
print('_'*50)
print(df['work_type'].value_counts())
print('_'*50)
df['smoking_status'].value_counts()


# In[31]:


print(df.shape)


# In[32]:


x = df.drop('stroke', axis=1)
y = df['stroke']


# In[33]:


from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
x_os,y_os=oversample.fit_resample(x, y)


# In[34]:


print(x_os.shape)
print(y_os.shape)


# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(x_os,y_os, test_size=0.3, random_state=10)


# In[37]:


path = RandomForestClassifier.get_params


# In[38]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train)
X = scaler.transform(x_train)
X_test = scaler.transform(x_test)


# In[39]:


rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X,y_train)
rf_pred = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, rf_pred)
accuracy_rf


# In[40]:


from xgboost import XGBClassifier
xgb = XGBClassifier(use_label_encoder=False)
xgb.fit(X,y_train)
xgb_pred = xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, xgb_pred)
accuracy_xgb


# In[41]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X,y_train)
svm_pred = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, svm_pred)
accuracy_svm


# In[42]:


lr = LogisticRegression(max_iter=800)
lr.fit(X,y_train)
lr_pred = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, lr_pred)
accuracy_lr


# In[43]:


xgb_cm = pd.DataFrame(data=confusion_matrix(y_test,xgb_pred),columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
sns.heatmap(xgb_cm, annot=True,fmt='d',cmap="Blues")


# In[44]:


print(classification_report(y_test,xgb_pred))


# In[45]:


rf_cm = pd.DataFrame(data=confusion_matrix(y_test,rf_pred),columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
sns.heatmap(rf_cm, annot=True,fmt='d',cmap="Blues")


# In[46]:


print(classification_report(y_test,rf_pred))


# In[47]:


lr_cm = pd.DataFrame(data=confusion_matrix(y_test,lr_pred),columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
sns.heatmap(lr_cm, annot=True,fmt='d',cmap="Blues")
print(classification_report(y_test,lr_pred))


# In[48]:


from sklearn.model_selection import cross_val_score
print(cross_val_score(lr, x_test, y_test, cv=6))


# In[49]:


print(cross_val_score(lr, x_train, y_train, cv=6))


# In[50]:


print(cross_val_score(lr, x, y, cv=6))


# In[51]:


print(cross_val_score(xgb, x, y, cv=6))


# In[52]:


pickle.dump(xgb, open('model.pkl','wb'))


# In[ ]:




