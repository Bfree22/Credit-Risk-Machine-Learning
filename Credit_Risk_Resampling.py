#!/usr/bin/env python
# coding: utf-8

# ## Credit Risk Resampling Techniques
# ____________
# 

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[10]:


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ## Read the CSV into DataFrame
# _______

# In[4]:


loans_data = pd.read_csv("LoansData.csv")
loans_data.head()


# ## Split the Data into Training and Testing
# ___________
# 

# In[5]:


# Create our features
X = pd.get_dummies(loans_data.drop('loan_status', axis=1))


# Create our target
y = loans_data.loc[:, 'loan_status']


# In[6]:


X.describe() 


# In[8]:


# Check the balance of our target values
y.value_counts()


# In[11]:


# Create X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)


# ## Data Pre-Processing
# ________
# 
# #### Scale the training and testing data using the StandardScaler from sklearn. Remember that when scaling the data, you only scale the features data (X_train and X_testing).

# In[12]:


# Create the StandardScaler instance
scaler = StandardScaler()


# In[13]:


# Fit the Standard Scaler with the training data
X_scaler = scaler.fit(X_train)


# In[14]:


# Scale the training and testing data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# ## Simple Logistic Regression
# ___________

# In[15]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_train, y_train)


# In[16]:


# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)


# In[17]:


# Display the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[18]:


# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))


# ## Oversampling
# ____________
# 
# #### In this section, you will compare two oversampling algorithms to determine which algorithm results in the best performance. You will oversample the data using the naive random oversampling algorithm and the SMOTE algorithm. For each algorithm, be sure to complete the folliowing steps:
# ____________
# 
# ### 1. View the count of the target classes using Counter from the collections library.
# #### 2. Use the resampled data to train a logistic regression model.
# #### 3. Calculate the balanced accuracy score from sklearn.metrics.
# #### 4. Print the confusion matrix from sklearn.metrics.
# #### 5. Generate a classication report using the imbalanced_classification_report from imbalanced-learn.
# 
# #### Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests

# ## Naive Random Oversampling
# ________

# In[20]:


# Resample the training data with the RandomOversampler
from imblearn.over_sampling import RandomOverSampler

# View the count of target classes with Counter
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)
Counter(y_resampled)


# In[27]:


# Train the Logistic Regression model using the resampled data
from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression(solver='lbfgs', random_state=1)
lg_model.fit(X_resampled, y_resampled)


# In[28]:


# Calculate the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score

predictions = lg_model.predict(X_test_scaled)

balanced_accuracy_score(y_test, predictions)


# In[29]:


# Display the confusion matrix
confusion_matrix(y_test, predictions)


# In[30]:


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, predictions))


# ## SMOTE Oversampling
# ________

# In[31]:


# Resample the training data with SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=1, sampling_strategy=1.0)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# View the count of target classes with Counter
Counter(y_resampled)


# In[32]:


# Train the Logistic Regression model using the resampled data
smote_model = LogisticRegression(solver='lbfgs', random_state=1)
smote_model.fit(X_resampled, y_resampled)


# In[33]:


# Calculated the balanced accuracy score
predictions = smote_model.predict(X_test_scaled)

balanced_accuracy_score(y_test, predictions)


# In[34]:


# Display the confusion matrix
confusion_matrix(y_test, predictions)


# In[35]:


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, predictions))


# ## Undersampling
# __________
# 
# In this section, you will test an undersampling algorithm to determine which algorithm results in the best performance compared to the oversampling algorithms above. You will undersample the data using the Cluster Centroids algorithm and complete the folliowing steps:
# 
# 1. View the count of the target classes using Counter from the collections library. 
# 2. Use the resampled data to train a logistic regression model.
# 3. Calculate the balanced accuracy score from sklearn.metrics.
# 4. Display the confusion matrix from sklearn.metrics.
# 5. Generate a classication report using the imbalanced_classification_report from imbalanced-learn.
# 
# Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests

# In[36]:


# Resample the data using the ClusterCentroids resampler
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train_scaled, y_train)

# View the count of target classes with Counter
Counter(y_resampled)


# In[37]:


# Train the Logistic Regression model using the resampled data
cluster_model = LogisticRegression(solver='lbfgs', random_state=1)
cluster_model.fit(X_resampled, y_resampled)


# In[38]:


# Calculate the balanced accuracy score
predictions = cluster_model.predict(X_test_scaled)
balanced_accuracy_score(y_test, predictions)


# In[39]:


# Display the confusion matrix
confusion_matrix(y_test, predictions)


# In[40]:


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, predictions))


# ## Combination (Over and Under) Sampling
# ___________
# 
# In this section, you will test a combination over- and under-sampling algorithm to determine if the algorithm results in the best performance compared to the other sampling algorithms above. You will resample the data using the SMOTEENN algorithm and complete the folliowing steps:
#     
# 1. View the count of the target classes using Counter from the collections library. 
# 2. Use the resampled data to train a logistic regression model.
# 3. Calculate the balanced accuracy score from sklearn.metrics.
# 4. Display the confusion matrix from sklearn.metrics.
# 5. Generate a classication report using the imbalanced_classification_report from imbalanced-learn.
# 
# Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests

# In[41]:


# Resample the training data with SMOTEENN
from imblearn.combine import SMOTEENN

sm = SMOTEENN(random_state=1)
X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)

# View the count of target classes with Counter
Counter(y_resampled)


# In[42]:


# Train the Logistic Regression model using the resampled data
smoteenn_model = LogisticRegression(solver='lbfgs', random_state=1)
smoteenn_model.fit(X_resampled, y_resampled)


# In[43]:


# Calculate the balanced accuracy score
predictions = smoteenn_model.predict(X_test_scaled)
balanced_accuracy_score(y_test, predictions)


# In[44]:


# Display the confusion matrix
confusion_matrix(y_test, predictions)


# In[45]:


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, predictions))


# ## Final Questions
# ______
# 
# #### 1. Which model had the best balanced accuracy score?
# The SMOTTEEN Combination sampling and SMOTE Oversmapling had a very similar balanced accuracy score of 99.5%
# 
# #### 2. Which model had the best recall score? 
# The Logistical Regression using the SMOTE oversampler had the highest score of 99%.
# 
# #### 3. Which model had the best geometric mean score? 
# The SMOTTEEN combination sampling had the highest geometric mean score of 99%.

# In[ ]:




