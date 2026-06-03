import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data= pd.read_csv("/content/ecommerce_data.csv")#to load data set
data.head()#for display 1st 5 row

data.dtypes

data.shape

data.isna().sum()

"""Eliminating NULL values"""

#first_order_date eliminated null values using mode operation (Due to datatype object)
data['first_order_date']=data['first_order_date'].fillna(data['first_order_date'].mode()[0])

#last_order_date eliminated null values using mode operation (Due to datatype object)
data['last_order_date']=data['last_order_date'].fillna(data['last_order_date'].mode()[0])

#preferred_payment eliminated null values using mode operation (Due to datatype object)
data['preferred_payment']=data['preferred_payment'].fillna(data['preferred_payment'].mode()[0])

#preferred_device_ord eliminated null values using mode operation (Due to datatype object)
data['preferred_device_ord']=data['preferred_device_ord'].fillna(data['preferred_device_ord'].mode()[0])

#preferred_source eliminated null values using mode operation (Due to datatype object)
data['preferred_source']=data['preferred_source'].fillna(data['preferred_source'].mode()[0])

#top_category_bought eliminated null values using mode operation (Due to datatype object)
data['top_category_bought']=data['top_category_bought'].fillna(data['top_category_bought'].mode()[0])

#avg_rating_given plootting histogram to find whether to take the mode,median or mean(due to the dtype is float)
plt.hist(data['avg_rating_given'])
plt.title('Distribution of avg_rating_given')

plt.show()

#due to left skew we take median here
data['avg_rating_given']=data['avg_rating_given'].fillna(data['avg_rating_given'].median())

#preferred_device_sess eliminated null values using mode operation (Due to datatype object)
data['preferred_device_sess']=data['preferred_device_sess'].fillna(data['preferred_device_sess'].mode()[0])

#preferred_source_sess eliminated null values using mode operation (Due to datatype object)
data['preferred_source_sess']=data['preferred_source_sess'].fillna(data['preferred_source_sess'].mode()[0])

#first_session_date eliminated null values using mode operation (Due to datatype object)
data['first_session_date']=data['first_session_date'].fillna(data['first_session_date'].mode()[0])

#last_session_date eliminated null values using mode operation (Due to datatype object)
data['last_session_date']=data['last_session_date'].fillna(data['last_session_date'].mode()[0])

data.isna().sum()

data.dtypes

"""Checking Outliers"""

sns.boxplot(data)

plt.boxplot(data['customer_id'])
plt.title('box plot of customer_id')
plt.show()

plt.boxplot(data['age'])
plt.title('box plot of age')
plt.show()

plt.boxplot(data['total_orders'])
plt.title('box plot of total_orders')
plt.show()

#outliers in total orders we need to remove it
Q1 = data['total_orders'].quantile(0.25)
Q3 = data['total_orders'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

data = data[(data['total_orders'] >= lower) & (data['total_orders'] <= upper)]

plt.boxplot(data['total_orders'])
plt.title('box plot of total_orders')
plt.show()

plt.boxplot(data['total_spend_usd'])
plt.title('box plot of total_spend_usd')
plt.show()

#outliers in total_spend_usd we need to remove it
Q1 = data['total_spend_usd'].quantile(0.25)
Q3 = data['total_spend_usd'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

data = data[(data['total_spend_usd'] >= lower) & (data['total_spend_usd'] <= upper)]

plt.boxplot(data['total_spend_usd'])
plt.title('box plot of total_spend_usd')
plt.show()

plt.boxplot(data['avg_order_value'])
plt.title('box plot of avg_order_value')
plt.show()

#outliers in avg_order_value we need to remove it
Q1 = data['avg_order_value'].quantile(0.25)
Q3 = data['avg_order_value'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

data = data[(data['avg_order_value'] >= lower) & (data['avg_order_value'] <= upper)]

plt.boxplot(data['avg_discount_pct'])
plt.title('box plot of avg_discount_pct')
plt.show()

plt.boxplot(data['avg_rating_given'])
plt.title('box plot of avg_rating_given')
plt.show()

#outliers in avg_rating_given we need to remove it
Q1 = data['avg_rating_given'].quantile(0.25)
Q3 = data['avg_rating_given'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

data = data[(data['avg_rating_given'] >= lower) & (data['avg_rating_given'] <= upper)]

plt.boxplot(data['total_sessions'])
plt.title('box plot of total_sessions')
plt.show()

#outliers in total_sessions we need to remove it
Q1 = data['total_sessions'].quantile(0.25)
Q3 = data['total_sessions'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

data = data[(data['total_sessions'] >= lower) & (data['total_sessions'] <= upper)]

plt.boxplot(data['has_abandoned_cart'])
plt.title('box plot of has_abandoned_cart')
plt.show()

plt.boxplot(data['is_repeat_customer'])
plt.title('box plot of is_repeat_customer')
plt.show()

"""Encoding"""

# encoding number_of_projects using label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data.dtypes

data['name'].nunique()

#encoding name with label
data['name']=le.fit_transform(data['name'])
data.head()

data['email'].nunique()

#encoding email with label
data['email']=le.fit_transform(data['email'])
data.head()

data['country'].nunique()

#encoding country with label
data['country']=le.fit_transform(data['country'])
data.head()

data['age_group'].nunique()

#encoding age_group with label
data['age_group']=le.fit_transform(data['age_group'])
data.head()

data['signup_date'].nunique()

#marketing is bool so binary encoding
data['marketing_opt_in'] = data['marketing_opt_in'].map({True: 1, False: 0})
data.head()

data['first_order_date'].nunique()

#encoding first_order_date with label
data['first_order_date']=le.fit_transform(data['first_order_date'])
data.head()

data['last_order_date'].nunique()

#encoding last_order_date with label
data['last_order_date']=le.fit_transform(data['last_order_date'])
data.head()

data['preferred_payment'].nunique()

#use one hot code for preferred_payment For Preferred Payment Method (4 unique values), One-Hot Encoding is usually better.
data = pd.get_dummies(
    data,
    columns=['preferred_payment'],
    drop_first=True,
    dtype=int
)

data.head()

data['preferred_device_ord'].nunique()

#encoding preferred_device_ord with label
data['preferred_device_ord']=le.fit_transform(data['preferred_device_ord'])
data.head()

data['preferred_source'].nunique()

#encoding preferred_source with label
data['preferred_source']=le.fit_transform(data['preferred_source'])
data.head()

data['top_category_bought'].nunique()

#encoding top_category_bought with label
data['top_category_bought']=le.fit_transform(data['top_category_bought'])
data.head()

data['preferred_device_sess'].nunique()

#use one hot code for preferred_device_sess For Preferred device  (3 unique values), One-Hot Encoding is usually better.
data = pd.get_dummies(data, columns=['preferred_device_sess'], drop_first=True,dtype=int)
data.head()

data['preferred_source_sess'].nunique()

#one hot for preferred source
data = pd.get_dummies(
    data,
    columns=['preferred_source_sess'],
    drop_first=True,
    dtype=int
)

data.head()

data['first_session_date'].nunique()

#encoding first_session_date with label
data['first_session_date']=le.fit_transform(data['first_session_date'])
data.head()

data['last_session_date'].nunique()

#encoding last_session_date with label
data['last_session_date']=le.fit_transform(data['last_session_date'])
data.head()

data['clv_tier'].nunique()

#encoding clv_tier with label
data['clv_tier']=le.fit_transform(data['clv_tier'])
data.head()

data['signup_date'].nunique()

#encoding signup_date with label
data['signup_date']=le.fit_transform(data['signup_date'])
data.head()

data.dtypes

