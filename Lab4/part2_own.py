# Lab 4 in the course TNM108 - Machine Learning for Social Media at Linköpings University 2022
# Anna Jonsson and Amanda Bigelius
# ---- PART 2 ----
# Create a pipeline and use grid search

# Dependencies
import numpy as np
from sklearn import metrics
import sklearn
from sklearn.datasets import load_files
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

print('\n LETS START PART 2 \n')

# Load movie_reviews data
moviedir = 'C:/Users/aanna/Desktop/TNM108/Lab4/movie_reviews'
movie = load_files(moviedir, shuffle=True)

# Split data into training and test sets
movie_data_train, movie_data_test, movie_target_train, movie_target_test = train_test_split(movie.data, movie.target, test_size = 0.20, random_state = 12)

# Create SVM classifier with a pipeline, using Naive Bayes as classifier
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

# Create SVM classifier with a pipeline, using SGD as classifier
# text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss = 'hinge', penalty = 'l2', alpha=1e-3, random_state=42, max_iter=5, tol=None), )])
# hinge gives a linear SVM, l2 is the regularisation parameter, alpha is the learning rate, max_iter is the number of iterations, tol is the stopping criterion

# !!!!! SGD gave the best mean, but MultinomialNB gave the best classification.

# The names vect, tfidf, and clf are arbitrary. We'll use them to perfom grid search for suitable hyperparameters.

# Train the model
text_clf.fit(movie_data_train, movie_target_train)

# Predict the test set
predicted = text_clf.predict(movie_data_test)
print('\n Accuracy SVM: ', np.mean(predicted == movie_target_test))

# Scikit metrics to get more detailed analysis
print(metrics.classification_report(movie_target_test, predicted, target_names = movie.target_names))

# Confusion matrix
print('\nConfusion matrix: ', metrics.confusion_matrix(movie_target_test, predicted))

# Parameter tuning using grid search
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3),}
# parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-1, 1e-3),}
# parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-1, 1e-2),}

# Grid search will detect how many CPU cores we have at our disposal n_jobs = (-1)
gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

# Let’s perform the search on a smaller subset of the training data to speed up the computation
gs_clf = gs_clf.fit(movie_data_train[:400], movie_target_train[:400])

# Create fake reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride', 
            'Steven Seagal was terrible', 'Steven Seagal shone through.', 
            'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through', 
            "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough', 
            'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

# Best mean score
print('\n Best mean score: ', gs_clf.best_score_)

# And its parameter settings
for param_name in sorted(parameters.keys()):
  print(" %s: %r" % (param_name, gs_clf.best_params_[param_name]))

print('\n')
# Predict the new reviews
predicted = gs_clf.predict(reviews_new)

# Print the predicted results
for review, category in zip(reviews_new, predicted):
    print('\n %r => %s' % (review, movie.target_names[category]))

print('\n THE END OF PART 2\n')

