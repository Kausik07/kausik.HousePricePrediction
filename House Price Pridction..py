#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


from sklearn.model_selection import train_test_split 


# In[3]:


from sklearn.linear_model import LinearRegression


# In[4]:


from sklearn.metrics import mean_squared_error,r2_score


# In[5]:


data = pd.read_csv(r"C:\\Users\\Thamaiyanthi\\Documents\\Arathi-Projects\\ML Project\\HousePricePrediction.csv")

print(data)


# In[6]:


data.shape


# In[13]:


numeric_data = data.select_dtypes(include=['float64', 'int64'])


# In[14]:


import matplotlib.pyplot as plt


# In[20]:


numeric_data = data.select_dtypes(include=['float64', 'int64'])


# In[21]:


plt.figure(figsize=(12, 6))
sns.heatmap(numeric_dataset.corr(),
            cmap='BrBG',
            fmt='.2f',
            linewidths=2,
            annot=True)






# In[22]:


obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (data.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (data.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))


# In[23]:


unique_values = []
for col in object_cols:
    unique_values.append(data[col].unique().size)

plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values)


# In[25]:


unique_values = []
for col in object_cols:
    unique_values.append(data[col].unique().size)

plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values)


# In[26]:


data.drop(['Id'],
axis=1,
inplace=True)


# In[27]:


data['SalePrice'] = data['SalePrice'].fillna(
data['SalePrice'].mean())


# In[28]:


new_data = data.dropna()


# In[29]:


new_data.isnull().sum()


# In[30]:


from sklearn.preprocessing import OneHotEncoder

s = (new_data.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', 
len(object_cols))


# In[31]:


OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_data[object_cols]))
OH_cols.index = new_data.index


feature_names = OH_encoder.get_feature_names_out(object_cols)
OH_cols.columns = feature_names

df_final = new_data.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)



# In[35]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split the training set into 
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(
X, Y, train_size=0.8, test_size=0.2, random_state=0)


# In[34]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))


# In[36]:


from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)

mean_absolute_percentage_error(Y_valid, Y_pred)


# In[37]:


X = data[['MSSubClass', 'MSZoning', 'LotArea', 'LotConfig', 'BldgType', 
          'OverallCond', 'YearBuilt', 'YearRemodAdd', 'Exterior1st', 
          'BsmtFinSF2', 'TotalBsmtSF']]

# Define target variable (y)
y = data['SalePrice']


# In[38]:


from sklearn.linear_model import LinearRegression

model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))


# In[39]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# In[45]:


if 'FeatureName' in data.columns:
    data.drop(['FeatureName'], axis=1, inplace=True)

# One-hot encode categorical variables
data = pd.get_dummies(data)

# Define features (X) and target variable (y)
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']


# In[41]:


pip install catboost


# In[54]:


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Assuming X and y are your features and target variable
# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate CatBoostClassifier model
cb_classifier = CatBoostClassifier()

# Train the model
cb_classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = cb_classifier.predict(X_valid)

# Evaluate the model
accuracy = accuracy_score(y_valid, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print(classification_report(y_valid, y_pred))


# In[55]:


from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

# Instantiate the CatBoostRegressor model
cb_model = CatBoostRegressor()

# Fit the model to the training data
cb_model.fit(X_train, y_train)

# Make predictions on the validation set
preds = cb_model.predict(X_valid) 

# Calculate the R2 score
cb_r2_score = r2_score(y_valid, preds)
print("R2 Score:", cb_r2_score)




# In[56]:


preds = cb_model.predict(X_valid) 
cb_r2_score = r2_score(y_valid, preds)
print("R2 Score:", cb_r2_score)


# In[57]:





# In[58]:


print("R2 Score:", cb_r2_score)


# In[59]:


import matplotlib.pyplot as plt

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_valid, preds, color='blue', alpha=0.5)
plt.plot([min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted values')
plt.show()


# In[ ]:




