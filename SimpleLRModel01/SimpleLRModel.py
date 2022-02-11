#!/usr/bin/env python
# coding: utf-8

# In[73]:


from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


# In[74]:


data = pd.read_csv("ageinsurance.csv")
x = data.iloc[:, 0].values 
y = data.iloc[:, 1].values
print(x,y)
data.head()


# In[75]:


plt.scatter(x, y, marker="+", c=y, cmap='rainbow')
plt.title('Scatter Plot of Logistic Regression')
plt.show()


# In[76]:


x_train, x_test, y_train, y_test = train_test_split(data[["age"]], data.has_insurance, test_size= 0.25, random_state=0)


# In[77]:


model = LogisticRegression()


# In[78]:


model.fit(x_train,y_train)


# In[79]:


prediction = model.predict(x_test)
x_test_array= x_test.iloc[:, 0].values 
data_pred= {'x_test': x_test_array, 'y_test':y_test, 'insurance_prediction': prediction}
df = pd.DataFrame(data_pred)  
print(df)


# In[80]:


model.score(x_test,y_test)


# In[81]:


model.predict_proba(x_test)

