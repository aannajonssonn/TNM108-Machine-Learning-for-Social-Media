# Lab 3 in the course TNM108 - Machine Learning for Social Media at Linköpings University 2022
# Anna Jonsson and Amanda Bigelius
# ---- PART 3 ----

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Load the data into a pandas dataframe
df = pd.read_csv('C:/Users/aanna/Desktop/TNM108/TNM108-Labs/Lab 3/data_cars.csv', header=None)
for i in range(len(df.columns)):
    df[i] = df[i].astype('category')
df.head()

# Map into numbers to be used in the classification algorithm (categories to numbers)
map0 = dict(zip(df[0].cat.categories, range(len(df[0].cat.categories))))
map1 = dict(zip(df[1].cat.categories, range(len(df[1].cat.categories))))
map2 = dict(zip(df[2].cat.categories, range(len(df[2].cat.categories))))
map3 = dict(zip(df[3].cat.categories, range(len(df[3].cat.categories))))
map4 = dict(zip(df[4].cat.categories, range(len(df[4].cat.categories))))
map5 = dict(zip(df[5].cat.categories, range(len(df[5].cat.categories))))
map6 = dict(zip(df[6].cat.categories, range(len(df[6].cat.categories))))

cat_cols = df.select_dtypes(['category']).columns
df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)
df = df.iloc[np.random.permutation(len(df))]

print(df.head())

# Divide the label vector Y from the features X
df_f1 = pd.DataFrame(columns = ['method'] + sorted(map6, key = map6.get))
df_precision = pd.DataFrame(columns = ['method'] + sorted(map6, key = map6.get))
df_recall = pd.DataFrame(columns = ['method'] + sorted(map6, key = map6.get))

def CalcMeasures(method, y_pred, y_true, df_f1 = df_f1, df_precision = df_precision, df_recall = df_recall):
    df_f1.loc[len(df_f1)] = [method] + list(f1_score(y_true, y_pred, average = None))
    df_precision.loc[len(df_precision)] = [method] + list(precision_score(y_true, y_pred, average = None))
    df_recall.loc[len(df_recall)] = [method] + list(recall_score(y_true, y_pred, average = None))

x = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

# 10 fold cross validation
cv  = 10
# cv = KFold(n_splits=10, shuffle=True)

method = 'linear support vector machine'
clf = svm.SVC(kernel='linear', C=50)
y_pred = cross_val_predict(clf, x, y, cv=cv)
CalcMeasures(method, y_pred, y)

method = 'naive bayes'
clf = MultinomialNB()
y_pred = cross_val_predict(clf, x, y, cv=cv)
CalcMeasures(method, y_pred, y)

method = 'logistic regression'
clf = LogisticRegression()
y_pred = cross_val_predict(clf, x, y, cv=cv)
CalcMeasures(method, y_pred, y)

method = 'k-nearest neighbors'
clf = KNeighborsClassifier(weights = 'distance', n_neighbors=5)
y_pred = cross_val_predict(clf, x, y, cv=cv)
CalcMeasures(method, y_pred, y)

method = 'random forest'
clf = RandomForestClassifier(n_estimators=90, max_depth=None, min_samples_split=2, random_state=0)
y_pred = cross_val_predict(clf, x, y, cv=cv)
CalcMeasures(method, y_pred, y)

method = 'decision tree'
clf = DecisionTreeRegressor(random_state=0)
y_pred = cross_val_predict(clf, x,y, cv=cv)
CalcMeasures(method,y_pred,y)

method = 'rbf support vector machine'
clf = svm.SVC(kernel='rbf',C=50, gamma="auto")
y_pred = cross_val_predict(clf, x,y, cv=cv)
CalcMeasures(method,y_pred,y)


# Measure values are stored in the data frames

# Calculate the number of samples in each class
labels_counts = df[6].value_counts()
pd.Series(map6).map(labels_counts)

print(df_f1)
