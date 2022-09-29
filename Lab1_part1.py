# Lab 1 in the course TNM108 - Machine Learning for Social Media at Linköpings University 2022
# Anna Jonsson and Amanda Bigelius

# Questions to answer:
# 1. What are the relevant features of the Titanic dataset? Why are they relevant?
    # Age and Survival rate
    # The data is relevant because it shows the survival rate of the passengers on the Titanic

# 2. Can you find a parameter configuration to get a validation score greater than 62%?
# 3. What are the advantages/ disadvantages of K-Means clustering?
# 4. How can you address the weaknesses?

# --- Dependencies ---
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load the train and test datasets to create two DataFrames ---
train_url = 'http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv' 
train = pd.read_csv(train_url) 
test_url = 'http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv'
test = pd.read_csv(test_url)

# --- Preview data, printing the loaded datasets ---
# print("\n")
# print("***** Train_Set *****") 
# print(train.head()) 

# # Initial statistics from both DataFrames, using the method 'descibe' from panda
# print(train.describe()) 

# print("\n") 
# print("***** Test_Set *****") 
# print(test.head())
# print("\n") 

# # --- Feature names ---
# print(train.columns.values) 
# print("\n") 

# --- Missing values in data ---

# For the train set 
# train.isna().head()

# # For the test set
# test.isna().head()

# Total number of missing values in both datasets
# print("*****In the train set*****")
# print(train.isna().sum())
# print("\n")
# print("*****In the test set*****")
# print(test.isna().sum())
# print("\n")	

# --- Handle missing values with Mean Imputation ---

# Fill missing values with mean column values in the train set 
train.fillna(train.mean(numeric_only=True), inplace=True) 
# Fill missing values with mean column values in the test set 
test.fillna(test.mean(numeric_only=True), inplace=True)

# numeric_only=True, to only get the numeric columns in train.mean() and test.mean(). FutureWarning

# Check if there are still missing values in the train set
#print(train.isna().sum())
#print("\n")	

# Check if there are still missing values in the test set
#print(test.isna().sum())
#print("\n")	

# Survival count with respect to Pclass
train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)

# Survival count with respect to sex
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Survival count with respect to SibSp
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# --- Plot graphs ---

# Age vs. Survived
# g = sns.FacetGrid(train, col='Survived')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()

# Relations between Pclass and Survived
# grid = sns.FacetGrid(train, col='Survived', row='Pclass', aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()

# --- Build KMeans model ---

# train.info()

# Feature engineering, drop insignificant features from the the datasets
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1) 
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

# Convert from non-numeric to numeric with label encoding
labelEncoder = LabelEncoder() 
labelEncoder.fit(train['Sex']) 
labelEncoder.fit(test['Sex']) 
train['Sex'] = labelEncoder.transform(train['Sex']) 
test['Sex'] = labelEncoder.transform(test['Sex'])

# train.info()
# test.info()

# Drop the survival column from the data
x = np.array(train.drop(['Survived'], 1).astype(float)) 
y = np.array(train['Survived'])

# train.info() # Still shows survived as a column

# Cluster the passenger records into 2 clusters: Survived and Not Survived
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(x)
# KMeans(algorithm = 'lloyd', copy_x = True, init = 'k-means++', max_iter = 300, n_clusters = 2, n_init = 10,  random_state = None, tol = 0.0001, verbose = 0)

# # View percentage of passenger records that were correctly clustered
# correct = 0
# for i in range(len(x)):
#     predict_me = np.array(x[i].astype(float))
#     predict_me = predict_me.reshape(-1, len(predict_me))
#     prediction = kmeans.predict(predict_me)
#     if prediction[0] == y[i]:
#         correct += 1

# print("\n")
# print(correct/len(x))
# print("\n")

# #Följande kod borde få annorlunda värden mot det ovanför... Något är knas.
# kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'lloyd')
# kmeans.fit(x)
# KMeans(algorithm = 'lloyd', copy_x = True, init = 'k-means++', max_iter = 600, n_clusters = 2, n_init = 10,  random_state = None, tol = 0.0001, verbose = 0)
# correct = 0
# for i in range(len(x)):
#     predict_me = np.array(x[i].astype(float))
#     predict_me = predict_me.reshape(-1, len(predict_me))
#     prediction = kmeans.predict(predict_me)
#     if prediction[0] == y[i]:
#         correct += 1

# print("\n")
# print(correct/len(x))
# print("\n")

# # --- Scale the values of the features to the same range ---
kmeans = KMeans(n_clusters=2)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
kmeans.fit(x_scaled)
KMeans(algorithm = 'lloyd', copy_x = True, init = 'k-means++', max_iter = 600, n_clusters = 2, n_init = 10,  random_state = None, tol = 0.0001, verbose = 0)
correct = 0

for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("\n")
print(correct/len(x))
print("\n")

# WRONG VALUES