import numpy as np
import pandas as pd
import re
import nltk
from sklearn.metrics import confusion_matrix, accuracy_score
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

def tokens(dataset):
    final_set = []
    for i in range(0, len(dataset)):
        temp_set = re.sub('[^a-zA-Z]', ' ', str(dataset['text'][i]))
        temp_set = temp_set.lower()
        temp_set = temp_set.split()
        ps = PorterStemmer()
        stopwords_new = stopwords.words('english')
        stopwords_new.remove('not')
        temp_set = [ps.stem(word) for word in temp_set if not word in set(stopwords_new)]
        temp_set = ' '.join(temp_set)
        final_set.append(temp_set)
    return final_set

def convert(dataset):
    dataset['sentiment'] = np.where(dataset['sentiment'] == "neutral", 0, dataset['sentiment'])
    dataset['sentiment'] = np.where(dataset['sentiment'] == "negative", 0, dataset['sentiment'])
    dataset['sentiment'] = np.where(dataset['sentiment'] == "positive", 1, dataset['sentiment'])
    return dataset

dataset_train = pd.read_csv('dataset/csv/train.csv',encoding='unicode_escape')
dataset_test = pd.read_csv('dataset/csv/test.csv',encoding='unicode_escape')
dataset_test = dataset_test.dropna()
li_train = tokens(dataset_train)
li_test = tokens(dataset_test)


cv_train = CountVectorizer()
x_train=cv_train.fit_transform(li_train).toarray()

cv_test = CountVectorizer(


)
x_test=cv_test.fit_transform(li_test).toarray()

y_train=convert(dataset_train)['sentiment'].astype('int')
y_test=convert(dataset_test)['sentiment'].astype('int')


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
print("Navibayes : ",accuracy_score(y_predict, y_test),"\n")

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
print("RandomForest : ",accuracy_score(y_predict, y_test),"\n")