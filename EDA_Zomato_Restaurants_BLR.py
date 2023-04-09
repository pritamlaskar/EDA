#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("C:/Users/Pritam Laskar/Documents/EDA/Zomato Bangalore Restaurants/zomato.csv")


# In[3]:


df


# In[4]:


# Check null values:

df.isnull().sum()


# In[5]:


# Check the percentage of null values in each column: 

(df.isnull().sum()/len(df))*100


# In[6]:


# Check duplicates:
df.duplicated()


# In[7]:


# Remove duplicates:

df.drop_duplicates(inplace = True)


# In[8]:


# Check shape after removing duplicates:

df.shape

# No duplicates found as shape is same before and after remove_duplicate function


# In[9]:


# Check data types of columns:

df.dtypes


# In[10]:


# # Data Cleaning # #


# In[11]:


# Cleaning in 'rate' columns:


# In[12]:


# Check unique values in 'rate':
df['rate'].unique()


# In[13]:


# Import numpy:

import numpy as np


# In[14]:


# Replace 'NEW' and '-' with nan by calling np:

df['rate'].replace(['NEW', '-'], np.nan, inplace = True)


# In[15]:


# Check percentage of null values after making changes in 'rate' column above:

(df.isnull().sum()/len(df))*100


# In[16]:


# Now check unique after making changes in 'rate' column:

df['rate'].unique()


# In[17]:


# Remove '/5' from rate by creating two different columns using split:

df[['new rate 1', 'new rate 2']] = df['rate'].str.split('/', expand = True)


# In[18]:


# Check the new columns:

df.head()


# In[19]:


# Drop 'new rate 2' column:

df.drop('new rate 2', axis = 1, inplace = True)


# In[20]:


# Rename 'new rate 1' to 'rate1':

df = df.rename(columns = {'new rate 1':'rate1'})


# In[21]:


# Change dtype of 'rate1' from object to float:

df['rate1'] = df['rate1'].astype('float')


# In[22]:


# Check dtype of 'rate1': 

df['rate1'].dtype


# In[23]:


# Replace null values of 'rate1' with mean of 'rate1'

df['rate1'] = df['rate1'].fillna(df['rate1'].mean())


# In[24]:


(df.isnull().sum()/len(df))*100


# In[25]:


# Check mode of 'rest_type':

df['rest_type'].mode()


# In[26]:


# Fill null values of 'rest_type' with mode of 'rest_type':

df['rest_type'] = df['rest_type'].fillna(df['rest_type'].mode())


# In[27]:


# Find mode of approx cost of two people:

df['approx_cost(for two people)'].mode()


# In[28]:


# Fill null values of approx cost column with mode:

df['approx_cost(for two people)'] = df['approx_cost(for two people)'].fillna(df['approx_cost(for two people)'].mode())


# In[29]:


# Find unique values in 'approx cost for two people':

df['approx_cost(for two people)'].unique()


# In[30]:


# Replace ',' in values of 'qpprox cost of two people':

df['approx_cost(for two people)'] = df['approx_cost(for two people)'].str.replace(',', '', regex = True)


# In[31]:


# Drop null values in 'approx cost of two people'

df.dropna(subset=['approx_cost(for two people)'], inplace = True)


# In[32]:


# Change dtype of 'approx cost for two people':

df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(int)


# In[33]:


# Check dtype of 'approx cost for two people' after changing dtype:

df['approx_cost(for two people)'].dtype


# In[34]:


(df.isnull().sum()/len(df))*100


# In[35]:


# Replace null values of 'cuisine' with its mode:

df['cuisines'] = df['cuisines'].fillna(df['cuisines'].mode().iloc[0])

# Note: .mode() returns a series and not a single value. That is why we need to use iloc[0] to extract the first and the only value in the mode series. Then using 'fillna()' we can replace the null values with this mode.


# In[36]:


df['cuisines'].isnull().sum()


# In[37]:


df.sample(50)


# In[38]:


# Drop unrequired columns:

df.drop(['url', 'address', 'phone', 'dish_liked', 'reviews_list', 'menu_item'], axis = 1, inplace = True)


# In[39]:


df.drop(['rate'], axis = 1, inplace = True)


# In[40]:


# Find unique values in 'listed_in(city)':

df['listed_in(city)'].unique()


# In[41]:


# Fill null values of 'rest_type' with its mode:

df['rest_type'] = df['rest_type'].fillna(df['rest_type'].mode().iloc[0])


# In[42]:


df = df.rename(columns = {'rate1':'rate'})


# In[43]:


df['rate'].dtype


# In[44]:


(df.isnull().sum()/len(df))*100


# In[45]:


df['approx_cost(for two people)'].dtype


# In[46]:


df.info()


# In[47]:


df = df.rename(columns = {'approx_cost(for two people)':'Cost2Plates'})


# In[48]:


## Checking distribution and removing outliers ##


# In[49]:


# Fetching boxplot for 'rate':

sns.boxplot(data = df['rate'])


# In[50]:


# Finding outliers of 'rate':
Q1 = df['rate'].quantile(.25)
Q3 = df['rate'].quantile(.75)
IQR = Q3 - Q1
UL = Q3 + 1.5*(IQR)
LL = Q1 - 1.5*(IQR)
print(UL)
print(LL)


# In[51]:


# Eliminating outliers by calling nympy:

df['rate'] = np.where(df['rate'] > 4.5, 4, np.where(df['rate'] < 3, 3, df['rate']))

# Note: When we set the lower value in 2.9, we could still see outliers, hence we took 3 and this time all outliers got removed.


# In[52]:


# boxplot after removing outliers:

sns.boxplot(data = df['rate'])


# In[53]:


df.shape


# In[54]:


df['rate'].nunique()


# In[55]:


# Finding outliers for votes: 

sns.boxplot(data = df['votes'])


# In[56]:


Q1 = df['votes'].quantile(.25)
Q3 = df['votes'].quantile(.75)
IQR = Q3 - Q1
UL = Q3 + 1.5*(IQR)
LL = Q1 - 1.5*(IQR)
print(UL)
print(LL)


# In[57]:


# Removing outliers in vote column:

df['votes'] = np.where(df['votes'] < -279.5, -270, np.where(df['votes'] > 484.4, 484, df['votes']))


# In[58]:


# Boxplot after removing outliers:

sns.boxplot(data = df['votes'])


# In[59]:


df.info()


# In[60]:


df = df.rename(columns = {'Cost2Plates': 'cost2plates'})


# In[61]:


# Checking outlier for 'cost2plates':

sns.boxplot(data = df['cost2plates'])


# In[62]:


# IQR for 'cost2plates':

Q1 = df['cost2plates'].quantile(.25)
Q3 = df['cost2plates'].quantile(.75)
IQR = Q3-Q1
UL = Q3 + 1.5*(IQR)
LL = Q1 - 1.5*(IQR)
print(UL)
print(LL)


# In[63]:


# Removing outlier using numpy:

df['cost2plates'] = np.where(df['cost2plates'] < -225, -225, np.where(df['cost2plates'] > 1175, 1175, df['cost2plates']))


# In[64]:


# Boxplot after removing outliers:

sns.boxplot(data = df['cost2plates'])


# In[65]:


df.shape


# In[66]:


df.isnull().sum()


# In[67]:


# Renaming some columns:

df = df.rename(columns = {'listed_in(type)': 'type', 'listed_in(city)': 'city'})


# In[68]:


# Find no. of unique values in 'location':

df['location'].nunique()


# In[69]:


# Drop 'city' column:

df = df.drop(['city'], axis = 1)


# In[70]:


df.info()


# In[71]:


# Find unique values in 'rest_type':

df['rest_type'].value_counts(ascending = False)


# In[72]:


df['rest_type'].unique()


# In[73]:


# To minimise the no. of unique values in 'rest_type', we will group all rest_type with values lower than 1000:
# (1) We save the count of 'rest_type' in 'rest_types'.

rest_types = df['rest_type'].value_counts(ascending = False)
rest_types


# In[74]:


# (2) We create another value 'rest_types_lessthan1000' where we save only 'rest_type' < 1000 from 'rest_types'.

rest_types_lessthan1000 = rest_types[rest_types < 1000]
rest_types_lessthan1000


# In[75]:


# (3) We define a new function 'handle_rest_type' where we assign 'rest_types_lessthan1000' as 'Others'.
# 

def handle_rest_type(value):
    if(value in rest_types_lessthan1000):
        return 'Others'
    else:
        return value
        
df['rest_type'] = df['rest_type'].apply(handle_rest_type)
df['rest_type'].value_counts()


# In[76]:


# We will proceed with same approach with 'location', 'cuisines', and 'type'
# Location below

df['location'].value_counts(ascending = False)


# In[77]:


locations = df['location'].value_counts(ascending = False)
locations_under200 = locations[locations<200]


# In[78]:


def handle_location(value):
    if(value in locations_under200):
        return 'Others'
    else:
        return value

df['location'] = df['location'].apply(handle_location)
df['location'].value_counts()


# In[79]:


# Cuisines below:

df['cuisines'].value_counts(ascending = False)


# In[80]:


cuisine = df['cuisines'].value_counts(ascending = False)
cuisine_under100 = cuisine[cuisine<100]


# In[81]:


def handle_cuisines(value):
    if(value in cuisine_under100):
        return 'Others'
    else:
        return value

df['cuisines'] = df['cuisines'].apply(handle_cuisines)


# In[82]:


df['cuisines'].value_counts()


# In[83]:


df.dtypes


# In[84]:


sns.boxplot(data = df['rate'])


# In[85]:


df.head()


# In[86]:


df.describe()


# In[87]:


## Data is cleaned,let's jump to visualisation ##


# In[88]:


# Heatmap of correlation:

a = sns.heatmap(df.corr(), annot = True, vmin = -1, vmax = 1)


# In[89]:


# Countplot of 'location'

location_groupby = df.groupby(['location'])['name'].count()
sorted_location = location_groupby.sort_values(ascending=False).index

plt.figure(figsize=(5, 10))
sns.countplot(data=df, y='location', order=sorted_location)
plt.show()


# In[90]:


# Countplot of 'online_order':

df['online_order'].value_counts().plot(kind = 'pie', autopct = '%1.1f%%', explode = [0.0, 0.1])


# In[91]:


# Barplot of 'cost2plates' and 'location'

sorted_cost2plates = df.groupby(['location'])['cost2plates'].mean().sort_values(ascending = False).index

plt.figure(figsize = (15,10))
sns.barplot(data = df, y = 'cost2plates', x = 'location', ci = None, order = sorted_cost2plates)
plt.xticks(rotation = 90);

# Use semi-colon(;) after 'xticks' to avoid arrays


# In[92]:


# Boxplot: rate - types

plt.figure(figsize = (15,5))
sns.boxplot(data = df, x = 'type', y = 'rate')


# In[93]:


# Barplot: Rates by Location

sorted_votes = df.groupby(['location'])['rate'].mean().sort_values(ascending = False).head(10).index

sns.barplot(data = df, y = 'location', x = 'rate', ci = None, order = sorted_votes)


# In[94]:


# Countplot: Book Table

df['book_table'].value_counts().plot(kind = 'pie', autopct = '%1.1f%%', explode = [0.0, 0.1])


# In[95]:


# Boxplot: Online order vs rate

sns.boxplot(data = df, y = 'rate', x = 'online_order')


# In[96]:


# Boxplot: book_table vs rate:

sns.boxplot(data = df, y = 'book_table', x = 'rate')


# In[97]:


# Countplot: Location vs online_order

plt.figure(figsize = (5,15))
sns.countplot(data = df, y = 'location', hue = 'online_order')


# In[98]:


# Countplot: location vs book_table

plt.figure(figsize = (15,5))
sns.countplot(data = df, x = 'location', hue = 'book_table');
plt.xticks(rotation = 90);


# In[99]:


# Pivot Table : Type vs Location

pivot = pd.pivot_table(df, values='name', index='type', columns='location', aggfunc=len, fill_value=0)
pivot = pivot.loc[:, pivot.sum().sort_values(ascending=False).head(10).index]

pivot
# Note: .head() here gives top 10 columns, not top 10 values.
# To show top 10 locations by each column, we need to create a stack (see next code).


# In[100]:


# Groupby: Names by votes

df.groupby(['name'])['votes'].sum().sort_values(ascending = False).head(10)


# In[101]:


# Barplot: Votes by location

sorted_location = df.groupby(['location'])['votes'].sum().sort_values(ascending = False).head(10).index

sns.barplot(data = df, y = 'location', x = df['votes'], estimator = sum, ci = None, order = sorted_location)


# In[102]:


# Barplot: Cuisines vs votes

a = df.groupby(['cuisines'])['votes'].sum().sort_values(ascending = False).head(10).iloc[1:]
plt.figure(figsize = (10,5))
sns.barplot(x = a.index, y = a.values)
plt.xticks(rotation = 90);

# iloc[1:] removes the first value from barplot. Without iloc[1:], the first value would be 'Others'.

