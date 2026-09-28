"""
Tests for the churn prediction API.

Runs the FastAPI app in-process with TestClient — no server, no network,
no real dataset needed (the fixture trains a tiny model on synthetic data
so these tests work even without data/streamflix_customers.csv).

Run:  pytest -q   (from the repository root, with API_TOKEN set)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

VALID_PAYLOAD = {
    "customer_id": 999, "age": 35, "country": "ES", "device": "mobile",
    "plan_type": "standard", "tenure_months": 24, "num_streams_per_week": 10,
    "avg_watch_time_hours": 3.5, "support_tickets_last_6m": 1,
    "auto_renew": 1, "payment_method": "card",
}


@pytest.fixture(scope="session", autouse=True)
def trained_model(tmp_path_factory):
    """Trains a throwaway model on synthetic data so tests don't depend on
    the (unpublished) real dataset, then points the API at it."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "country": rng.choice(["ES", "IT", "FR", "DE", "PT"], n),
        "device": rng.choice(["web", "mobile", "smart_tv"], n),
        "plan_type": rng.choice(["basic", "standard", "premium"], n),
        "tenure_months": rng.integers(0, 60, n),
        "num_streams_per_week": rng.integers(0, 30, n),
        "avg_watch_time_hours": rng.uniform(0.2, 6, n),
        "support_tickets_last_6m": rng.integers(0, 5, n),
        "auto_renew": rng.integers(0, 2, n),
        "payment_method": rng.choice(["card", "paypal", "bank_transfer"], n),
        "churn": rng.integers(0, 2, n),
    })
    tmp_dir = tmp_path_factory.mktemp("model")
    data_path = tmp_dir / "streamflix_customers.csv"
    df.to_csv(data_path, index=False)

    from train_model import add_engagement, build_preprocessor
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    import joblib

    df_eng, bounds = add_engagement(df)
    X = df_eng[["age", "tenure_months", "num_streams_per_week", "avg_watch_time_hours",
               "support_tickets_last_6m", "auto_renew", "engagement",
               "country", "device", "plan_type", "payment_method"]]
    pipe = Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression(max_iter=500))])
    pipe.fit(X, df["churn"])

    model_path = tmp_dir / "modelo_fuga.joblib"
    with open(model_path, "wb") as f:
        joblib.dump(pipe, f)
    bounds_path = tmp_dir / "engagement_bounds.json"
    bounds_path.write_text(json.dumps(bounds), encoding="utf-8")

    os.environ["API_TOKEN"] = "test-token"
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["BOUNDS_PATH"] = str(bounds_path)
    yield


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import api as api_module
    with TestClient(api_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_requires_token(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code in (401, 403)


def test_predict_rejects_wrong_token(client):
    r = client.post("/predict", json=VALID_PAYLOAD, headers={"Authorization": "Bearer wrong-token"})
    assert r.status_code == 401


def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD, headers={"Authorization": "Bearer test-token"})
    assert r.status_code == 200
    body = r.json()
    assert body["churn_pred"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


@pytest.mark.parametrize("field,bad_value", [
    ("age", 10),               # below minimum (18)
    ("country", "XX"),          # not in allowed set
    ("device", "tablet"),       # not in allowed set
    ("plan_type", "gold"),      # not in allowed set
    ("payment_method", "cash"),  # not in allowed set
    ("avg_watch_time_hours", 0),  # must be > 0
])
def test_predict_validates_input(client, field, bad_value):
    payload = {**VALID_PAYLOAD, field: bad_value}
    r = client.post("/predict", json=payload, headers={"Authorization": "Bearer test-token"})
    assert r.status_code == 422


def test_predict_missing_field(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    r = client.post("/predict", json=payload, headers={"Authorization": "Bearer test-token"})
    assert r.status_code == 422


def test_api_requires_token_env_var(monkeypatch):
    """Importing the module without API_TOKEN set must fail loudly, not
    fall back to a hardcoded default (the original exam's bug)."""
    monkeypatch.delenv("API_TOKEN", raising=False)
    for mod in ("api",):
        sys.modules.pop(mod, None)
    with pytest.raises(RuntimeError, match="API_TOKEN"):
        import api  # noqa: F401
