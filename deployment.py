import joblib
from flask import Flask, request
import json

# Load the trained model
clf = joblib.load("sentiment_model.pkl")

app = Flask(_name_)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the data from the request
    data = request.get_json()
    text = data["text"]

    # Extract features and make predictions
    features = extract_features([text])
    prediction = clf.predict(features)[0]

    # Return the prediction as a JSON object
    return json.dumps({"prediction": prediction})

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=8000)