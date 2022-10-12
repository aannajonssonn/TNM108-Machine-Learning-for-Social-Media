# Lab 2 in the course TNM108 - Machine Learning for Social Media at Linköpings University 2022
# Anna Jonsson and Amanda Bigelius

# ---- PART 2 ----

# Dependencies
from sklearn import preprocessing # Import labelEncoder
from sklearn.neighbors import KNeighborsClassifier # Import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Assigning features and label variables 
# First Feature 
weather = ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 'Rainy','Sunny','Overcast','Overcast','Rainy']

# Second Feature
temp = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'] 

# Label or target variable 
play = ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
weather_encoded = le.fit_transform(weather)
print ("Weather: ", weather_encoded)

temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)

# Combining weather and temp into single listof tuples
features = list(zip(weather_encoded,temp_encoded))

# Build the KNN classifier model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
model.fit(features,label)

# Predict Output
predicted = model.predict([[0,2]]) # 0:Overcast, 2:Mild
print ("\nPrediction: ", predicted)

# Load dataset
wine = datasets.load_wine()

# Print name of the features
print("\nFeatures: ", wine.feature_names)

# Print the top 5 records of the wine data
print("\nTop 5 records: ", wine.data[0:5])

# Print the label type of wine ( 0: class_0, 1: class_1, 2: class_2)
print("\nLabels: ", wine.target)

# Check shape of dataset
print("\nShape of dataset: ", wine.data.shape)
print("\nShape of target: ", wine.target.shape)

# Split dataset into training set and test set, 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) 

# k = 5
# Create KNN Classifier for k = 5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training sets
knn.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = knn.predict(X_test)

# Check accuracy of our model on the test data
print("\nAccuracy for k=5: ", metrics.accuracy_score(y_test, y_pred))

# k = 7
# Create KNN Classifier for k = 7
knn = KNeighborsClassifier(n_neighbors=7)

# Train the model using the training sets
knn.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = knn.predict(X_test)

# Check accuracy of our model on the test data
print("\nAccuracy for k=7: ", metrics.accuracy_score(y_test, y_pred))