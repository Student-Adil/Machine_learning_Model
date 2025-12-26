from flask import Flask, request, jsonify,render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# 🔥 LOAD SAVED OBJECTS
model = pickle.load(open("model.pkl", "rb"))
ct = pickle.load(open("encoder.pkl", "rb"))
le = pickle.load(open("label.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')  

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    major = data["major"]
    minor = data["minor"]
    cgpa = float(data["cgpa"])

    # Convert input to DataFrame (VERY IMPORTANT)
    new = pd.DataFrame(
        [[major, minor, cgpa]],
        columns=["Major", "Minor", "Cgpa"]
    )

    # Encode input
    new_encoded = ct.transform(new)

    # Predict
    pred = model.predict(new_encoded)
    pred_label = le.inverse_transform(pred)[0]

    return jsonify({
        "prediction": f"Predicted Course is {pred_label}"
    })

if __name__ == "__main__":
    app.run(debug=True)
