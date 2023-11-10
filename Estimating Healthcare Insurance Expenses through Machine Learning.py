#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#Data Preprocessing


# In[3]:


df=pd.read_csv('insurance.csv')
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[32]:


descriptive_stats= df.describe()
print("Descriptive Statistics for Insurance\n","\n", descriptive_stats)


# In[6]:


duplicates=df.duplicated().sum()
duplicates


# In[7]:


df.drop_duplicates(inplace=True)
df.shape


# In[42]:


descriptive_stats= df.describe()
print("Descriptive Statistics for Insurance\n","\n", descriptive_stats)
print("--------------------------------------------------------------")
mode = df.mode().iloc[0]
print("Mode is\n",mode)


# In[45]:


correlation_matrix = df.corr()
correlation_matrix['charges'].sort_values(ascending=False)


# In[47]:


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')

plt.show()


# In[54]:


# Comparative Analysis: Insurance costs across different groups

# Smokers vs Non-smokers
smoker_costs = df.groupby('smoker')['charges'].mean()

# Costs by Region
region_costs = df.groupby('region')['charges'].mean()


# Display the results
print(smoker_costs)
print("---------------")
print(region_costs)


# In[8]:


null_values=df.isnull().sum()
null_values


# In[9]:


lab_encode = LabelEncoder()

for column in ['sex', 'smoker']:
    df[column] = lab_encode.fit_transform(df[column])


df[['sex', 'smoker']].head()


# In[10]:


one_hot_encode = pd.get_dummies(df['region'])


# In[11]:


df1 = pd.concat([df, one_hot_encode], axis=1)


# In[12]:


df1.head()


# In[13]:


df1.drop('region',axis=1, inplace=True)
df1.head()


# In[14]:


X = df1.drop('charges', axis=1)
y = df1['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


rand_forest_model = RandomForestRegressor(n_estimators=50, n_jobs=2, random_state=42)
cv_scores = cross_val_score(rand_forest_model, X_train, y_train,scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-cv_scores)
performance = np.mean(rmse_scores)

std = np.std(rmse_scores)


# In[26]:



rand_forest_model.fit(X_train, y_train)
predictions = rand_forest_model.predict(X_test)
rounded_predictions = np.round(predictions, 2)
rounded_actuals = np.round(y_test, 2)

y_test_reset = y_test.reset_index(drop=True)
rounded_actuals = np.round(y_test_reset, 2)
mae = mean_absolute_error(y_test_reset, predictions)
mse = mean_squared_error(y_test_reset, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reset, predictions)

compare = pd.DataFrame({'Actual Charges': rounded_actuals.head(10), 
                        'Predicted Charges': rounded_predictions[:10]})


print("RMSE score is",rmse)
print("r2 score is", r2)


# In[24]:


compare


# In[ ]:




