#!/usr/bin/env python
# coding: utf-8

# # Task 3 - Modeling
# 
# This notebook will get you started by helping you to load the data, but then it'll be up to you to complete the task! If you need help, refer to the `modeling_walkthrough.ipynb` notebook.
# 
# 
# ## Section 1 - Setup
# 
# First, we need to mount this notebook to our Google Drive folder, in order to access the CSV data file. If you haven't already, watch this video https://www.youtube.com/watch?v=woHxvbBLarQ to help you mount your Google Drive folder.

# We want to use dataframes once again to store and manipulate the data.

# In[1]:


import pandas as pd


# ---
# 
# ## Section 2 - Data loading
# 
# Similar to before, let's load our data from Google Drive for the 3 datasets provided. Be sure to upload the datasets into Google Drive, so that you can access them here.

# In[2]:


path = 'C://Users//Admin//Downloads//cognizant internship//'

sales_df = pd.read_csv(f"{path}sales.csv")
sales_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
sales_df.head()


# In[3]:


stock_df = pd.read_csv(f"{path}sensor_stock_levels.csv")
stock_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
stock_df.head()


# In[4]:


temp_df = pd.read_csv(f"{path}sensor_storage_temperature.csv")
temp_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
temp_df.head()


# In[5]:


def convert_to_datetime(data: pd.DataFrame = None, column: str = None):

  dummy = data.copy()
  dummy[column] = pd.to_datetime(dummy[column], format='%Y-%m-%d %H:%M:%S')
  return dummy


# In[6]:


sales_df = convert_to_datetime(sales_df, 'timestamp')
sales_df.info()


# In[7]:


stock_df = convert_to_datetime(stock_df, 'timestamp')
stock_df.info()


# In[8]:


temp_df = convert_to_datetime(temp_df, 'timestamp')
temp_df.info()


# In[9]:


from datetime import datetime

def convert_timestamp_to_hourly(data: pd.DataFrame = None, column: str = None):
  dummy = data.copy()
  new_ts = dummy[column].tolist()
  new_ts = [i.strftime('%Y-%m-%d %H:00:00') for i in new_ts]
  new_ts = [datetime.strptime(i, '%Y-%m-%d %H:00:00') for i in new_ts]
  dummy[column] = new_ts
  return dummy


# In[10]:


sales_df = convert_timestamp_to_hourly(sales_df, 'timestamp')
sales_df.head()


# In[11]:


stock_df = convert_timestamp_to_hourly(stock_df, 'timestamp')
stock_df.head()


# In[12]:


temp_df = convert_timestamp_to_hourly(temp_df, 'timestamp')
temp_df.head()


# Now it's up to you, refer back to the steps in your strategic plan to complete this task. Good luck!

# In[13]:


sales_agg = sales_df.groupby(['timestamp', 'product_id']).agg({'quantity': 'sum'}).reset_index()
sales_agg.head()


# In[14]:


stock_agg = stock_df.groupby(['timestamp', 'product_id']).agg({'estimated_stock_pct': 'mean'}).reset_index()
stock_agg.head()


# In[15]:


temp_agg = temp_df.groupby(['timestamp']).agg({'temperature': 'mean'}).reset_index()
temp_agg.head()


# In[16]:


merged_df = stock_agg.merge(sales_agg, on=['timestamp', 'product_id'], how='left')
merged_df.head()


# In[17]:


merged_df = merged_df.merge(temp_agg, on='timestamp', how='left')
merged_df.head()


# In[18]:


merged_df['quantity'] = merged_df['quantity'].fillna(0)
merged_df.info()


# In[19]:


product_categories = sales_df[['product_id', 'category']]
product_categories = product_categories.drop_duplicates()

product_price = sales_df[['product_id', 'unit_price']]
product_price = product_price.drop_duplicates()


# In[20]:


merged_df = merged_df.merge(product_categories, on="product_id", how="left")
merged_df.head()


# In[21]:


merged_df = merged_df.merge(product_price, on="product_id", how="left")
merged_df.head()


# In[22]:


merged_df['timestamp_day_of_month'] = merged_df['timestamp'].dt.day
merged_df['timestamp_day_of_week'] = merged_df['timestamp'].dt.dayofweek
merged_df['timestamp_hour'] = merged_df['timestamp'].dt.hour
merged_df.drop(columns=['timestamp'], inplace=True)
merged_df.head()


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt
#To visaulize the correlation between the columns
sns.heatmap(merged_df.corr(),annot=True,cmap='winter')


# In[24]:


category=merged_df.groupby('category')[['unit_price']].mean().sort_values(by='unit_price',ascending=False)
category.head()


# In[31]:


#covert the categorical columns to numerical
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
#Covert the categorical column to numerical columns
label=LabelEncoder()
merged_df['category']=label.fit_transform(merged_df['category'])
merged_df['product_id']=label.fit_transform(merged_df['product_id'])
#Divided the data set for the training and testing
X=merged_df.drop(['estimated_stock_pct'],axis=1)
y=merged_df['estimated_stock_pct']
standrd=StandardScaler()
X=standrd.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=120)


# In[32]:


#Modelbuliding 
import numpy as np
def model_buliding(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    model_pred=model.predict(X_test)
    print(f'The {model} test accuracy_score is {model.score(X_test,y_test)*100:.2f}')
    print(f'The {model} test accuracy_score is {model.score(X_train,y_train)*100:.2f}')
    rmse=mean_absolute_error(y_test,model_pred)
    print('The root_mean_absolute_error',np.sqrt(rmse))
linear=LinearRegression()
model_buliding(linear,X_train,X_test,y_train,y_test)


# In[33]:


#import the randomforestRegressor
from sklearn.ensemble import RandomForestRegressor
random=RandomForestRegressor()
model_buliding(random,X_train,X_test,y_train,y_test)


# In[34]:


#import the decisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor()
model_buliding(tree,X_train,X_test,y_train,y_test)


# In[ ]:




