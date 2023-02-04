import requests
import json
from sklearn.metrics import accuracy_score

# Assume that the deployed model is running on a server at the following URL
MODEL_URL = "http://localhost:8000/predict"

# Assume that the test data is stored in a file called test_data.json
with open("test_data.json", "r") as f:
    test_data = json.load(f)

# Extract the text and labels from the test data
texts = [data["text"] for data in test_data]
labels = [data["label"] for data in test_data]

# Use the deployed model to make predictions on the test data
predictions = []
for text in texts:
    response = requests.post(MODEL_URL, json={"text": text})
    predictions.append(response.json()["prediction"])

# Calculate the accuracy of the model on the test data
acc = accuracy_score(labels, predictions)
print("Accuracy: ", acc)