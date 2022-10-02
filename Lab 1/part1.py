# Lab 1 in the course TNM108 - Machine Learning for Social Media at Linköpings University 2022
# Anna Jonsson and Amanda Bigelius

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

# Initial statistics from both DataFrames, using the method 'descibe' from panda
# print("\n")
# print("***** Statistics *****") 
# print(train.describe()) 

# print("\n") 
# print("***** Test_Set *****") 
# print(test.head())

# print("\n")
# print("***** Dataset features *****") 
# print(train.columns.values) 


# --- Missing values in data ---
# print("\n")
# print("***** Missing values in train set *****") 
# train.isna().head()

# print("\n")
# print("***** Missing values in test set *****") 
# test.isna().head()

# print("*****Total number of missing values in the train set*****")
# print(train.isna().sum())
# print("\n")

# print("*****Total number of missing values in the test set*****")
# print(test.isna().sum())
# print("\n")	


# --- Handle missing values with Mean Imputation ---
# Fill missing values with mean column values in the train set 
train.fillna(train.mean(numeric_only=True), inplace=True) 
# Fill missing values with mean column values in the test set 
test.fillna(test.mean(numeric_only=True), inplace=True)

# numeric_only=True, to only get the numeric columns in train.mean() and test.mean(). FutureWarning

# print("\n")
# print("***** Check for any reminding missing values in train set *****")
# print(train.isna().sum())

# print("\n")
# print("***** Check for any reminding missing values in test set *****")
# print(test.isna().sum())


# --- Handle categorical data ---
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

# Pclass vs. Survived
# grid = sns.FacetGrid(train, col='Survived', row='Pclass', aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()


# --- Build KMeans model ---
# print("\n")
# print("***** Data types of different features in train set *****")
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

# print("\n")
# print("***** Sex should now be numeric in both sets *****")
# print("\n")
# print("***** Train set *****")
# train.info()
# print("\n")
# print("***** Test set *****")
# test.info()

# Drop the survival column from the data
x = np.array(train.drop(['Survived'], 1).astype(float)) 
y = np.array(train['Survived'])

# print("\n")
# print("***** Survived column should be dropped from train set *****") # Doesn't work
# train.info()

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
# print("***** Validation score 1: *****")
# print(correct/len(x))

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
# print("***** Validation score 2: *****")
# print(correct/len(x))

# # --- Scale the values of the features to the same range ---
kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'lloyd')
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
print("***** Validation score 3: *****")
print(correct/len(x))
print("\n")

# WRONG VALUES