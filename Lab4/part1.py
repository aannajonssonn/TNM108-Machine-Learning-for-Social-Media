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
my_vocabulary = {'blue': 0,'bright': 1, 'sky' : 3}
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
tfidf_transformer = TfidfTransformer(norm = '12')
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
