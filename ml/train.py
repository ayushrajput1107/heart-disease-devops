"""Train the heart disease classifier and save it to model/model.pkl."""
import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "heart.csv")
OUT = os.path.join(BASE, "model")

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    # In this version of the UCI file, target=1 means NO disease, so we flip it:
    # y = 1 means heart disease present.
    y = 1 - df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    os.makedirs(OUT, exist_ok=True)
    joblib.dump(model, os.path.join(OUT, "model.pkl"))
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Model saved.", metrics)


if __name__ == "__main__":
    main()
