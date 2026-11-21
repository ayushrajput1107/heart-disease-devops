import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE, "model"))

app = Flask(__name__)
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    metrics = json.load(f)

# feature name -> (min, max) accepted range
FEATURES = {
    "age": (1, 120), "sex": (0, 1), "cp": (0, 3), "trestbps": (50, 260),
    "chol": (50, 700), "fbs": (0, 1), "restecg": (0, 2), "thalach": (40, 250),
    "exang": (0, 1), "oldpeak": (0, 10), "slope": (0, 2), "ca": (0, 4), "thal": (0, 3),
}


@app.route("/")
def home():
    return render_template("index.html", metrics=metrics)


@app.route("/health")
def health():
    return jsonify(status="ok", model_accuracy=metrics["accuracy"]), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    row = {}
    for name, (lo, hi) in FEATURES.items():
        if name not in data:
            return jsonify(error=f"missing field: {name}"), 400
        try:
            value = float(data[name])
        except (TypeError, ValueError):
            return jsonify(error=f"{name} must be a number"), 400
        if not lo <= value <= hi:
            return jsonify(error=f"{name} must be between {lo} and {hi}"), 400
        row[name] = value

    X = pd.DataFrame([row], columns=list(FEATURES))
    prob = float(model.predict_proba(X)[0][1])
    return jsonify(
        prediction=int(prob >= 0.5),
        label="Heart disease likely" if prob >= 0.5 else "Heart disease unlikely",
        probability=round(prob, 3),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
