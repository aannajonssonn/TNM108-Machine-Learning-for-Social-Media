# Lab 4 in the course TNM108 - Machine Learning for Social Media at Linköpings University 2022
# Anna Jonsson and Amanda Bigelius
# ---- PART 1 ----

# Dependencies
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import math

# Vector space conversion
d1 = "The sky is blue." 
d2 = "The sun is bright." 
d3 = "The sun in the sky is bright." 
d4 = "We can see the shining sun, the bright sun." 
Z = (d1,d2,d3,d4)

# Create new instance of CountVectorizer (term-frequency)
vectorizer = CountVectorizer()
print(vectorizer)

# Create vocabulary and stop words set
my_stop_words = {'the', 'is'}
my_vocabulary = {'blue','bright', 'sky'}
vectorizer = CountVectorizer(stop_words=my_stop_words, vocabulary=my_vocabulary)

# Print stop words list and vocabulary
print(vectorizer.vocabulary)
print(vectorizer.stop_words)

# Create sparse matrix of the document set
smatrix = vectorizer.transform(Z)
# Spicy sparse matrix, elements are stored in a coordinate format
print(smatrix)

# Convert sparse matrix to dense matrix
smatrix = smatrix.todense()
print(smatrix)

# Calculate the TF_IDF values
tfidf_transformer = TfidfTransformer(norm = 'l2')
tfidf_transformer.fit(smatrix)

# print idf values
feature_names = vectorizer.get_feature_names()
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=feature_names, columns=["idf_weights"])

# Sort Ascending
df_idf.sort_values(by=['idf_weights'])

# Compute TF-IDF values
tf_idf_vector = tfidf_transformer.transform(smatrix)

# Get tdifd vector for first document
first_document_vector = tf_idf_vector[0] # first document "The sky is blue."

# Print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)

# Cosine similarity
print('\n COSINE SIMILARITY \n')

# Transform the document set into a count-vectorized form
tfidf_vectorizer = TfidfVectorizer()

# Fit the transformer into a tf-idf matrix with the supplied data
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)

# Print the tf-idf matrix shape
print(tfidf_matrix.shape)

# Take the cos similarity of the third document (cos similarity = 0.52)
angle_in_radians = math.acos(cosine_similarity(tfidf_matrix[0], tfidf_matrix[2]))

# print the angle between the first and the thrird document
print(math.degrees(angle_in_radians))

# Classifying text
print('\n CLASSIFYING TEXT \n')

# Download dataset
data = fetch_20newsgroups()

# Look at the target names
data.target_names

# Consider only 4 categories
my_categories = ['rec.sport.baseball','rec.motorcycles','sci.space','comp.graphics']

# Fetch the training and testing data set
train = fetch_20newsgroups(subset='train', categories=my_categories)
test = fetch_20newsgroups(subset='test', categories=my_categories)

# Convert the content of each string into a vector of numbers
cv = CountVectorizer()
X_train_counts = cv.fit_transform(train.data)

# Compute the TF-IDF matrix
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Pass the TF_IDF matrix to the multinomial Naive Bayes classifier to create a predictive model
model = MultinomialNB().fit(X_train_tfidf, train.target)

# Apply the model
docs_new = ['Pierangelo is a really good baseball player','Maria rides her motorcycle', 'OpenGL on the GPU is fast', 'Pierangelo rides his motorcycle and goes to play football since he is a good football player too.']
X_new_counts = cv.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = model.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, train.target_names[category]))

print('\n THE END \n')