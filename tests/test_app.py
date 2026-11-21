import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "app"))
from app import app  # noqa: E402  (needs model/model.pkl -> run ml/train.py first)

SAMPLE = {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
          "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
          "ca": 0, "thal": 1}


def client():
    return app.test_client()


def test_health():
    r = client().get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_home_page():
    assert client().get("/").status_code == 200


def test_predict_valid():
    r = client().post("/predict", json=SAMPLE)
    d = r.get_json()
    assert r.status_code == 200
    assert d["prediction"] in (0, 1) and 0 <= d["probability"] <= 1


def test_predict_missing_field():
    bad = dict(SAMPLE)
    bad.pop("age")
    assert client().post("/predict", json=bad).status_code == 400


def test_predict_out_of_range():
    bad = dict(SAMPLE, age=500)
    assert client().post("/predict", json=bad).status_code == 400


def test_model_quality_gate():
    """CI fails if the retrained model is worse than 75% accuracy."""
    with open(os.path.join(BASE, "model", "metrics.json")) as f:
        assert json.load(f)["accuracy"] >= 0.75
