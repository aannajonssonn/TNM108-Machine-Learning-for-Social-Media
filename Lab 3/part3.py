# Lab 3 in the course TNM108 - Machine Learning for Social Media at Linköpings University 2022
# Anna Jonsson and Amanda Bigelius
# ---- PART 3 ----

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE

# Housing data (APPARENTLY UNETHICAL TO USE THIS DATA)
# 10 fold cross validation to evaluate the performance of several estimators
from sklearn.datasets import load_boston
boston = load_boston()  # Load the data
x = boston.data  # Features
y = boston.target  # Target variable
# cv = 10  # Number of folds

# Shuffle and split the data to randomize the cross validation folds
cv = KFold(n_splits=10, shuffle=True)

# Linear regression
print('\nLinear regression')
lin = LinearRegression()
scores = cross_val_score(lin, x, y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(lin, x, y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# Ridge regression
print('\nRidge regression')
ridge = Ridge(alpha = 1.0)
scores = cross_val_score(ridge, x, y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(ridge, x, y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# Lasso regression
print('\nLasso regression')
lasso = Lasso(alpha = 0.1)
scores = cross_val_score(lasso, x, y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(lasso, x, y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# Decision Tree regression
print('\nDecision Tree regression')
tree = DecisionTreeRegressor(random_state=0)
scores = cross_val_score(tree, x, y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(tree, x, y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# Random Forest regression
print('\nRandom Forest regression')
forest = RandomForestRegressor(n_estimators=50, max_depth = None, min_samples_split = 2, random_state=0)
scores = cross_val_score(forest, x, y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(forest, x, y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# Linear Support Vector Machine
print('\nLinear Support Vector Machine')
svm_lin = svm.SVR(epsilon = 0.2, kernel='linear', C = 1)
scores = cross_val_score(svm_lin, x, y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(svm_lin, x, y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# Support Vector Machine with RBF kernel
print('\nSupport Vector Machine with RBF kernel')
clf = svm.SVR(epsilon = 0.2, kernel='rbf', C = 1.)
scores = cross_val_score(clf, x, y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(clf, x, y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# K-Nearest Neighbors
print('\nK-Nearest Neighbors')
knn = KNeighborsRegressor()
scores = cross_val_score(knn, x, y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(knn, x, y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# ---- RECURSIVE FEATURE ELIMINATION ----
# RFE, select best 4 features of the for linear regression
bestFeatures = 4

rfe_lin = RFE(estimator=lin, n_features_to_select=bestFeatures).fit(x,y)
supported_features=rfe_lin.get_support(indices=True) 
for i in range(0, 4): 
    z=supported_features[i] 
    print(i+1,boston.feature_names[z])

# Feature Selection on Linear regression
print('\nFeature Selection on Linear regression')
rfe_lin = RFE(estimator=lin, n_features_to_select=bestFeatures).fit(x,y)
mask = np.array(rfe_lin.support_)
scores = cross_val_score(rfe_lin, x[:, mask], y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(rfe_lin, x[:, mask], y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# Feature Selection on Ridge regression
print('\nFeature Selection on Ridge regression')
rfe_ridge = RFE(estimator=ridge, n_features_to_select=bestFeatures).fit(x,y)
mask = np.array(rfe_ridge.support_)
scores = cross_val_score(rfe_ridge, x[:, mask], y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(rfe_ridge, x[:, mask], y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# Feature Selection on Lasso regression
print('\nFeature Selection on Lasso regression')
rfe_lasso = RFE(estimator=lasso, n_features_to_select=bestFeatures).fit(x,y)
mask = np.array(rfe_lasso.support_)
scores = cross_val_score(rfe_lasso, x[:, mask], y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(rfe_lasso, x[:, mask], y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# Feature Selection on Decision Tree regression
print('\nFeature Selection on Decision Tree regression')
rfe_tree = RFE(estimator=tree, n_features_to_select=bestFeatures).fit(x,y)
mask = np.array(rfe_tree.support_)
scores = cross_val_score(rfe_tree, x[:, mask], y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(rfe_tree, x[:, mask], y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# Feature Selection on Random Forest regression
print('\nFeature Selection on Random Forest regression')
rfe_forest = RFE(estimator=forest, n_features_to_select=bestFeatures).fit(x,y)
mask = np.array(rfe_forest.support_)
scores = cross_val_score(rfe_forest, x[:, mask], y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(rfe_forest, x[:, mask], y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# Linear Support Vector Machine
print('\nFeature Selection on Linear Support Vector Machine')
rfe_svm_lin = RFE(estimator=svm_lin, n_features_to_select=bestFeatures).fit(x,y)
scores = cross_val_score(rfe_svm_lin, x[:, mask], y, cv=cv)
print('Mean R2: %0.2f (+/- %0.2f) ' % (scores.mean(), scores.std()*2))
predicted = cross_val_predict(rfe_svm_lin, x[:, mask], y, cv=cv)
print('MSE: %0.2f' % mean_squared_error(y, predicted))

# K-Nearest Neighbors does not provide weights on the features, so RFE cannot be used.