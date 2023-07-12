#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imported necessary modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[2]:


#reading data from csv file
df=pd.read_csv('Downloads/NNDL_Code and Data/Salary_Data.csv')


# In[4]:


#checking data
df


# In[6]:


#splitting data into feature and target set
X=df[['YearsExperience']];


# In[7]:


#splitting data into feature and target set
y=df[['Salary']];


# In[8]:


#splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=1)


# In[9]:


#using linear Regression model on training set
from sklearn.linear_model import LinearRegression
r=LinearRegression()
r.fit(X_train,y_train)


# In[10]:


#predicting based on test data
y_pred=r.predict(X_test)


# In[11]:


#calculating mean_squared_error
mean_squared_error=metrics.mean_squared_error(y_test,y_pred)


# In[12]:


#printing mean_squared_error
print(mean_squared_error)


# In[13]:


plt.scatter(X_train, y_train, color='Blue')
plt.scatter(X_test, y_test, color='Green')
plt.plot(X_train, r.predict(X_train), color='Red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:





# In[ ]:




