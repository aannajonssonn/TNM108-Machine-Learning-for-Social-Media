# Lab 4 in the course TNM108 - Machine Learning for Social Media at Linköpings University 2022
# Anna Jonsson and Amanda Bigelius
# ---- PART 2 ----

# Importing libraries
import numpy as np
from sklearn import metrics

# Load movie_reviews corpus data
import sklearn
from sklearn.datasets import load_files
moviedir = 'C:/Users/aanna/Desktop/TNM108/Lab4/movie_reviews'

# import CountVectorizer, nltk
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# For the Parameter tuning using grid search
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split


# loading all files
movie = load_files(moviedir, shuffle=True)

# length of data
len(movie.data)

# target names ("classes") are automatically generated from subfolder names
movie.target_names

# First file seems to be about a Schwarzenegger movie. 
movie.data[0][:500]

# first file is in "neg" folder
movie.filenames[0]

# first file is a negative review and is mapped to 0 index 'neg' in target_names
movie.target[0]

## Try CountVectorizer and TF-IDF

# Turn off pretty printing of jupyter notebook... it generates long lines
# %pprint

# Three tiny "documents"
docs = ['A rose is a rose is a rose is a rose.',
        'Oh, what a fine day it is.',
        "A day ain't over till it's truly over."]

# Initialize a CountVectorizer to use NLTK's tokenizer instead of its 
#    default one (which ignores punctuation and stopwords). 
# Minimum document frequency set to 1. 
fooVzer = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)

# .fit_transform does two things:
# (1) fit: adapts fooVzer to the supplied text data (rounds up top words into vector space) 
# (2) transform: creates and returns a count-vectorized output of docs
docs_counts = fooVzer.fit_transform(docs)

# fooVzer now contains vocab dictionary which maps unique words to indexes
fooVzer.vocabulary_

# docs_counts has a dimension of 3 (document count) by 16 (# of unique words)
docs_counts.shape

# this vector is small enough to view in a full, non-sparse form! 
docs_counts.toarray()

# Convert raw frequency counts into TF-IDF (Term Frequency -- Inverse Document Frequency) values
fooTfmer = TfidfTransformer()

# Again, fit and transform
docs_tfidf = fooTfmer.fit_transform(docs_counts)

# TF-IDF values
# raw counts have been normalized against document length, 
# terms that are found across many docs are weighted down ('a' vs. 'rose')
docs_tfidf.toarray()

# A list of new documents
newdocs = ["I have a rose and a lily.", "What a beautiful day."]

# This time, no fitting needed: transform the new docs into count-vectorized form
# Unseen words ('lily', 'beautiful', 'have', etc.) are ignored
newdocs_counts = fooVzer.transform(newdocs)
newdocs_counts.toarray()

# Again, transform using tfidf to sort data
newdocs_tfidf = fooTfmer.transform(newdocs_counts)
newdocs_tfidf.toarray()

# Use on movie reviews
# Split data into training and test sets
docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, test_size = 0.20, random_state = 12)

# initialize CountVectorizer
movieVzer= CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000) # use top 3000 words only. 78.25% acc.
# movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)         # use all 25K words. Higher accuracy

# fit and tranform using training text 
docs_train_counts = movieVzer.fit_transform(docs_train)

# 'screen' is found in the corpus, mapped to index 2290
movieVzer.vocabulary_.get('screen')

# Likewise, Mr. Steven Seagal is present...
movieVzer.vocabulary_.get('seagal')

# huge dimensions! 1,600 documents, 3K unique terms. 
docs_train_counts.shape

# Convert raw frequency counts into TF-IDF values
movieTfmer = TfidfTransformer()
docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)

# Same dimensions, now with tf-idf values instead of raw frequency counts
docs_train_tfidf.shape

# Using the fitted vectorizer and transformer, tranform the test data
docs_test_counts = movieVzer.transform(docs_test)
docs_test_tfidf = movieTfmer.transform(docs_test_counts)

# Now ready to build a classifier. 
# We will use Multinominal Naive Bayes as our model
from sklearn.naive_bayes import MultinomialNB

# Train a Multimoda Naive Bayes classifier. Again, we call it "fitting"
clf = MultinomialNB()
clf.fit(docs_train_tfidf, y_train)

# Predict the Test set results, find accuracy
y_pred = clf.predict(docs_test_tfidf)
sklearn.metrics.accuracy_score(y_test, y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Trying the classifier on fake movie reviews
# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride', 
            'Steven Seagal was terrible', 'Steven Seagal shone through.', 
              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through', 
              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough', 
              'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

reviews_new_counts = movieVzer.transform(reviews_new)         # turn text into count vector
reviews_new_tfidf = movieTfmer.transform(reviews_new_counts)  # turn into tfidf vector

# have classifier make a prediction
pred = clf.predict(reviews_new_tfidf)

# print out results
for review, category in zip(reviews_new, pred):
    print('\n %r => %s' % (review, movie.target_names[category]))

# Mr. Seagal simply cannot win! 

# Final notes:
# In practice, you should use TfidVectorizer, which is CountVectorizer and TfidfTransformer convieniently combined.
# from sklearn.feature_extraction.text import TfidfVectorizer
# You can also use a pipeline to chain together multiple steps, including vectorization and classification.
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())

### Parameter Tuning Using Grid Search
categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)
twenty_test = fetch_20newsgroups(subset='test',categories=categories,shuffle=True,random_state=42)

# Building a pipeline
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
# The names vect, tfidf, and clf are arbitrary. We'll use them to perfom grid search for suitable hyperparameters.

# Train the model
text_clf.fit(twenty_train.data, twenty_train.target)

# Evaluating the predictive accuracy of the model
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print('\n multinomialBC accuracy:', np.mean(predicted == twenty_test.target))

# Testing accuracy using a linear SVM

# Training SVM classifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)),])
text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted = text_clf_svm.predict(docs_test)
print('\n SVM accuracy:', np.mean(predicted == twenty_test.target))

# Use scikit-learn metrics for more detailed analysis
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted)) 

# Parameter tuning using grid search
# We try out all classifiers on wither words or bigrams, with or without idf, and a penatly parameter of either 0.01 or 0.00. for the linear SVM

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3),}

# Grid search will detect how many CPU cores we have at our disposal n_jobs = (-1)
gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

# Let’s perform the search on a smaller subset of the training data to speed up the computation
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

# Calling fit on a GridSearchCV 
print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])

# Best mean score
print(gs_clf.best_score_)

# And its parameter settings
for param_name in sorted(parameters.keys()):
  print("\n %s: %r" % (param_name, gs_clf.best_params_[param_name]))