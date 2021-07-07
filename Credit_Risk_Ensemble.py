#!/usr/bin/env python
# coding: utf-8

# # Ensemble Learning
# ___________
# ### Initial Imports
# ________

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter 


# In[3]:


from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced


# ### Read the CSV and Perform Basic Data Cleaning
# __________

# In[4]:


# Load the data
loan_stats = pd.read_csv("LoanStats.csv")

# Preview the data
loan_stats.head()


# ## Split the Data into Training and Testing
# ___________

# In[5]:


# Create our features
X = pd.get_dummies(loan_stats.drop('loan_status', axis=1))

# Create our target
y = loan_stats['loan_status'].tolist()


# In[6]:


X.describe()


# In[7]:


# Check the balance of our target values
loan_stats["loan_status"].value_counts()


# In[8]:


# Split the X and y into X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=1, stratify=y)
X_train.shape


# ## Data Pre-Processing
# ___________
# 
# ### Scale the training and testing data using the StandardScaler from sklearn. Remember that when scaling the data, you only scale the features data (X_train and X_testing).

# In[9]:


# Create the StandardScaler instance
scaler = StandardScaler()


# In[10]:


# Fit the Standard Scaler with the training data
X_scaler = scaler.fit(X_train)


# In[11]:


# Scale the training and testing data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# ## Ensemble Learners
# ________
# #### In this section, you will compare two ensemble algorithms to determine which algorithm results in the best performance. You will train a Balanced Random Forest Classifier and an Easy Ensemble classifier . For each algorithm, be sure to complete the folliowing steps:
# 
# #### 1. Train the model using the training data. 
# #### 2. Calculate the balanced accuracy score from sklearn.metrics.
# #### 3. Display the confusion matrix from sklearn.metrics.
# #### 4. Generate a classication report using the imbalanced_classification_report from imbalanced-learn. 
# #### 5. For the Balanced Random Forest Classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score

# ## Balanced Random Forest Classifier

# In[12]:


# Resample the training data with the BalancedRandomForestClassifier

from imblearn.ensemble import BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=1)
brf.fit(X_train, y_train)


# In[18]:


# Calculated the balanced accuracy score
predictions = brf.predict(X_test)

balanced_accuracy_score(y_test, predictions)


# In[19]:


# Display the confusion matrix
confusion_matrix(y_test, predictions)


# In[20]:


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, predictions))


# In[21]:


# List the features sorted in descending order by feature importance
importances = pd.DataFrame(brf.feature_importances_, index = X_train.columns, columns=['Importance']).sort_values('Importance', ascending=False)
importances.loc[:,:]


# ## Easy Ensemble Classifier

# In[22]:


# Train the Classifier
from imblearn.ensemble import EasyEnsembleClassifier

ensemble_model = EasyEnsembleClassifier(n_estimators=100, random_state=1,)

ensemble_model.fit(X_train_scaled, y_train)


# In[23]:


# Calculated the balanced accuracy score
predictions = ensemble_model.predict(X_test_scaled)

balanced_accuracy_score(y_test, predictions)


# In[24]:


# Display the confusion matrix
confusion_matrix(y_test, predictions)


# In[25]:


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, predictions))


# ### Final Questions
# 
# #### 1. Which model had the best balanced accuracy score?
# 
# The Ensemble model had the best accuracy score of 93%, while the brf model was a 78%.

# #### 2. Which model had the best recall score?
# 
# The Ensemble model had the best recall score of 94%, while the brf model had a 91% score.

# #### 3. Which model had the best geometric mean score?
# 
# The Ensemble model had the best geometric mean score of 93%, while the brf model had a 78% score.

# #### 4. What are the top three features?
# 
# The top three features are total_rec_prncp (7.3%), total_rec_int (6.3%) and total_pymnt_int (6.1%).
