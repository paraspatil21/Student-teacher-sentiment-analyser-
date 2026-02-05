import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
import pickle

# Download stopwords
# The data will be downloaded to C:\Users\paras\AppData\Roaming\nltk_data
nltk.download("stopwords")

# Load dataset
# Note: Ensure 'reviews.txt' is present in the project folder
dataset = pd.read_csv('reviews.txt', sep='\t', names=['Reviews', 'Comments'])

# Set up stopwords
stopset = set(stopwords.words('english'))

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=list(stopset))

# Transform the comments and extract labels
X = vectorizer.fit_transform(dataset.Comments)
y = dataset.Reviews

# Save the vectorizer for future use
pickle.dump(vectorizer, open('tranform.pkl', 'wb'))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize and train the Multinomial Naive Bayes classifier
clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, clf.predict(X_test)) * 100
print(f"Initial Test Accuracy: {accuracy}%")

# Re-train the model on the entire dataset
clf = naive_bayes.MultinomialNB()
clf.fit(X, y)

# Final accuracy check
final_accuracy = accuracy_score(y_test, clf.predict(X_test)) * 100
print(f"Final Model Accuracy: {final_accuracy}%")

# Save the trained model
filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))
print(f"Model saved as {filename}")
