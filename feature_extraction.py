from sklearn.feature_extraction.text import CountVectorizer


def extract_features(tokens):
    # Create a bag of words representation
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(tokens)
    return features



print("features")