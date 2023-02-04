from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import pandas as pd
from preprocessing import preprocess_text
from feature_extraction import extract_features
import numpy as np
# Assume that the features and labels have already been extracted
# features is a sparse matrix and labels is a list of binary labels
c1 = time.time()
tokens = []
data = pd.read_csv("dataset/csv/train.csv",encoding= 'unicode_escape')
for i in data['text']:
    tokens.append(preprocess_text(str(i)))

features = extract_features(tokens)
data['sentiment'] = np.where(data['sentiment'] == "neutral",0,data['sentiment'])
data['sentiment'] = np.where(data['sentiment'] == "negative",0,data['sentiment'])
data['sentiment'] = np.where(data['sentiment'] == "positive",0,data['sentiment'])
#data.loc[str(data['sentiment']) == "negative",'sentiment'] = 0
#data.loc[str(data['sentiment']) == "positive",'sentiment'] = 1
labels = data['sentiment']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the classifier
clf = LogisticRegression()

# Train the classifier
clf.fit(X_train, y_train)
c2 = time.time()

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ",acc)