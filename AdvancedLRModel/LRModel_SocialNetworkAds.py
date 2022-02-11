#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# In[2]:


dataset = pd.read_csv("snads_dataset.csv")
dataset.head()


# In[3]:


dataset.info()


# In[4]:


dataset.describe()


# In[5]:


dataset.isnull().all()


# In[6]:


# Creating 4 plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# Age Boxplot
sns.boxplot(ax=axes[0,0],x= dataset['Age'], palette = "Set1")
axes[0,0].set_title('Age Of People')
# Age Histogram
sns.histplot(ax=axes[0,1],x='Age',data=dataset,color="g")
axes[0,1].set_title('Distribution Of Ages')

# EstimatedSalary Boxplot
sns.boxplot(ax=axes[1,0],data = dataset['EstimatedSalary'])
axes[1,0].set_title('Estimated Salary Of People')
# EstimatedSalary Histogram
sns.histplot(ax=axes[1,1],x='EstimatedSalary',data=dataset,color="y")
axes[1,1].set_title('Distribution Of Estimated Salary')


# In[7]:


fig, axes = plt.subplots(1, 2, figsize=(15,5))
sns.boxplot(ax=axes[0],x=dataset['Gender'], y=dataset['EstimatedSalary'], palette="PRGn")
axes[0].set_title('Estimated Salary By Gender')

sns.boxplot(ax=axes[1],x=dataset['Gender'], y=dataset['Age'], palette="pink")
axes[1].set_title('Ages By Gender')
plt.show()


# In[8]:


fig ,axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(ax=axes[0],x='Purchased',data=dataset)
axes[0].set_title('Number Of People Purchased')
sns.countplot(ax=axes[1],x='Purchased',hue='Gender',data=dataset,palette="magma")
axes[1].set_title('Number Of People Purchased By Gender')
plt.show()


# In[9]:


dataset.corr()


# In[10]:


f,ax = plt.subplots(figsize=(6, 5))
#Heatmap for labels
sns.heatmap(dataset.corr(), annot=True, linewidths=0.5,linecolor="black", fmt= '.2f',cmap='viridis',ax=ax)
plt.show()


# In[11]:


#preparing data
# Removing User ID column and its data
dataset.drop('User ID',axis = 1, inplace = True)
label = {'Male': 0 ,"Female" : 1}
dataset['Gender'].replace(label, inplace= True)


# In[21]:


# set inputs and outputs
#Dataset without correct dependent variable values
X = dataset.drop('Purchased',axis = 1)     
#Dependent variable values
y = dataset['Purchased']
X, y


# In[27]:


# we have to scale the data for better result

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)
data_scaled1 = pd.DataFrame(data_scaled)
data_scaled1.head()


# In[28]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(data_scaled,y,test_size=0.20,random_state=42)


# In[46]:


from sklearn.linear_model import LogisticRegression

# LR model
model = LogisticRegression(C=0.1,max_iter = 500)
# LR model training
model.fit(X_train,y_train)

# LR testing/prediction
y_pred = model.predict(X_test)


# In[47]:


# y = B + W*x1...

print(f'Weight Coefficient : {model.coef_}')
print(f'Bias : {model.intercept_}')


# In[54]:


# Accuracy: The amount of correct classifications / the total amount of classifications.
# The train accuracy: The accuracy of a model on examples it was constructed on.
# The test accuracy is the accuracy of a model on examples it hasn't seen.
print(f'Test accuracy: {model.score(X_test,y_test)}')
print(f'Train accuracy: {model.score(X_train,y_train)}')


# In[55]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[56]:


from sklearn.metrics import confusion_matrix

df = pd.DataFrame(confusion_matrix(y_test, y_pred), columns = ['Predicted Positive', 'Predicted Negative'], 
                  index=['Actual Positive', 'Actual Negative'])
df


# In[61]:


# We can visualize the confusion matrix
import scikitplot.metrics as splt

splt.plot_confusion_matrix(y_test,y_pred,figsize=(7,7))
plt.show()


# In[57]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_test,y_pred))
print("Precision:", precision_score(y_test, y_pred, ))
print("Recall:", recall_score(y_test,y_pred))
print("F1 Score:", f1_score(y_test,y_pred))


# In[63]:


# Area Under Curve - AUC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds.
model_roc_auc = roc_auc_score(y_test, model.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % model_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim(([0.0, 1.0]))
plt.ylim(([0.0, 1.05]))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()


# In[78]:


from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

clf = LogisticRegression()
model_res = clf.fit(X_train_res, y_train_res)


# In[72]:


print(f'Test accuracy {model_res.score(X_test,y_test)}')


# In[74]:


print(f'Original: {X_train.shape}')
print(f'With SMOTE: {X_train_res.shape}')

