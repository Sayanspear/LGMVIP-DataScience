#!/usr/bin/env python
# coding: utf-8

# # LetsGrowMore Virtual Internship Programme
# ## Data Science Internship
# ## Task 1: Iris Flower Classification ML Project
# 
# ### Author: Sayan Das
# 

# ## Importing Some Necessary Libraries

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


iris_data=pd.read_csv("Downloads\iris.csv")


# ## Data Wrangling

# In[6]:


iris_data.shape


# In[24]:


iris_data.head()


# In[3]:


iris_data.info()


# So from the info function we can see that there are 150 entries and 150 non-null counts for each column means no missing value is present. 
# 
# The data type of each coloumn except Species is numerical and Species is categorical.

# In[19]:


iris_data[["Species"]].value_counts()


# In[7]:


iris_data.describe().T


# Here, we can see the mean values, standard deviation, minimum & maximum value and the quartiles of the numerical columns.
# 
# Now let us check for outliers.

# In[15]:


sns.boxplot(iris_data['SepalLengthCm'])


# In[16]:


sns.boxplot(iris_data['SepalWidthCm'])


# In[17]:


sns.boxplot(iris_data['PetalLengthCm'])


# In[18]:


sns.boxplot(iris_data['PetalWidthCm'])


#  So only for SpalWidthCm, 2 values are above upper quartile and 1 value is below lower quartile. Since they are not too far, we are not considering them as outlier. 

# In[30]:


x=iris_data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
x[0:5]


# In[28]:


y=iris_data['Species']
y[0:5]


# ## Train-Test-Split 

# In[31]:


from sklearn.model_selection import train_test_split


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)


# In[34]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ## Modeling 

# In[35]:


from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree


# We will first create an instance of the DecisionTreeClassifier called irisTree.
# Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node. 

# In[36]:


irisTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
irisTree


# In[37]:


irisTree.fit(x_train,y_train)


# ## Prediction 

# In[38]:


predTree=irisTree.predict(x_test)
print(predTree[0:5])
print(y_test[0:5])


# ## Evaluation 

# Let's import metrics from sklearn and check the accuracy of our model. 

# In[39]:


from sklearn import metrics
print("Accuracy of the Decision Tree:", metrics.accuracy_score(y_test,predTree))


# ## Visualization
# 
# Let us now visualizde the decision tree.

# In[41]:


plt.figure(figsize=(7,7))
tree.plot_tree(irisTree)
plt.show()


# # You can now feed any new/test data to this classifer and it would be able to predict the right class accordingly.
