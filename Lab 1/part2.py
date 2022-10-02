# Lab 1 in the course TNM108 - Machine Learning for Social Media at Linköpings University 2022
# Anna Jonsson and Amanda Bigelius

# Dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import dendrogram, linkage

# Import shopping dataset
customer_data = pd.read_csv('C:/Users/aanna/Desktop/TNM108/TNM108-Labs/Lab 1/shopping_data.csv')

# Check number of records and attributes
print('\n')
print('Number of records and attributes: ')
print(customer_data.shape)

# Check 2 last columns annual income ($k) and spending score (1-100)
data = customer_data.iloc[:, 3:5].values
print('\n')
print('Last two columns: ')
print(data)

# Plot dendrogram
linked = linkage(data, 'ward')
labelList = range(1, len(data)+1)
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)
plt.show()

# Create clusters
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()
