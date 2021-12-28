#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the required lib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn import metrics
from sklearn import preprocessing


# In[2]:


df=pd.read_csv(r"D:\DATASET\heart.csv")
df.head()


# In[4]:


# check information about the variables
df.info()


# In[5]:


# check number off null values
df.isnull().sum()


# In[6]:


# check the shape of the dataset
df.shape


# In[7]:


# check the dtype of the dataset
print(df.dtypes)


# In[10]:


df.describe()


# In[28]:


# fetch non-numeric and categorical columns
numerical=df.select_dtypes(include=[np.float64,np.int64])
print(numerical.columns)


# In[29]:


# check and treat for outliers in numeric columns
plt.figure(figsize=(15, 10))
sns.boxplot(x="variable", y="value", data=pd.melt(numerical))
sns.stripplot(x="variable", y="value", data=pd.melt(numerical), color="orange", jitter=0.2, size=2.5)
plt.title("Outliers", loc="left")
plt.grid()


# In[72]:


def remove_outlier(col):
    global lower_range,upper_range
    Q1,Q3=df[col].quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    res=df[df[col].between(lower_range,upper_range)]
    a=res[col].mean()
    df.loc[df[col]<lower_range,col]=a
    df.loc[df[col]>upper_range,col]=a
    return


# In[74]:


plt.figure(figsize=(15, 10))
sns.boxplot(x="variable", y="value", data=pd.melt(numerical))
sns.stripplot(x="variable", y="value", data=pd.melt(numerical), color="orange", jitter=0.2, size=2.5)
plt.title("Outliers", loc="left")
plt.grid()


# In[30]:


def bar_plot(variable):
    var = df[variable]
    varValue = var.value_counts()
    plt.figure(figsize=(15,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()


# In[31]:


bar_plot('Age')


# In[32]:


bar_plot('ChestPainType')


# In[33]:


bar_plot('RestingBP')


# In[34]:


fig, ax = plt.subplots()
ax.barh(df['MaxHR'],df['ChestPainType'])


# In[35]:


label_encoder = preprocessing.LabelEncoder()
change=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
for i in change:
    df[i]= label_encoder.fit_transform(df[i])


# In[36]:


df


# In[37]:


plt.figure(figsize =(15,10))
sns.heatmap(df.corr(),robust=True,fmt='.1g',linewidths=1.3,linecolor='gold',annot=True);


# In[39]:


# Check whether the data set is balanced or not
heart_0=df['HeartDisease'][df['HeartDisease']==0].count()
heart_1=df['HeartDisease'][df['HeartDisease']==1].count()
print('Heart Disease: 0 : ',heart_0)
print('Heart Disease: 1 : ',heart_1)
print('Heart Disease: 0 (%): ',(heart_0/(heart_0+heart_1))*100)
print('Heart Disease: 1 (%): ',(heart_1/(heart_0+heart_1))*100)


# In[40]:


df.to_csv('processed.csv')


# >>KNN CLASSIFICATION 

# In[63]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[64]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)


# In[65]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)


# In[66]:


# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[68]:


# Predicting a new result
print(classifier.predict(sc.transform([[40,1,1,140,289,0,1,172,0,0.0,2]])))


# In[69]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[70]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

