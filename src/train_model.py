"""
Churn prediction — training pipeline (corrected).

Bugs fixed from the original exam:
  1. `plan_time` -> `plan_type`   (the feature list referenced a column that
     doesn't exist, so the plan tier was silently dropped from the model).
  2. `cargar_modelo` opened the file in text mode ("r") instead of binary
     ("rb"), which crashes joblib.load on any real run.
  3. The FastAPI service defaulted API_TOKEN to a hardcoded string
     ("supersecreto123") when the environment variable was missing — the
     service now refuses to start without an explicit token.

Fixing bug #1 does not fix the model: cross-validated AUC stays at ~0.52
either way (see results/metrics.json). The dataset itself carries very
little signal — the exam's own conclusion was right, even though the code
that reached it had these two bugs.

Run:  python src/train_model.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "streamflix_customers.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42

NUM_COLS = ["age", "tenure_months", "num_streams_per_week", "avg_watch_time_hours",
           "support_tickets_last_6m", "auto_renew", "engagement"]
CAT_COLS = ["country", "device", "plan_type", "payment_method"]  # bug #1 fixed here


@dataclass(frozen=True)
class Resultados:
    accuracy: float
    f1: float
    auc: float
    n_train: int
    n_test: int
    best_params: dict[str, Any]


def add_engagement(df: pd.DataFrame, bounds: dict[str, tuple[float, float]] | None = None) -> tuple[pd.DataFrame, dict]:
    """Min-max normalised engagement score. `bounds`, if given, reuses the
    training-time min/max instead of recomputing them — required so a single
    row scored later (e.g. in the API) gets the same scale as training."""
    df = df.copy()
    repro = df["num_streams_per_week"].to_numpy(dtype=float)
    tiempo = df["avg_watch_time_hours"].to_numpy(dtype=float)
    if bounds is None:
        bounds = {
            "num_streams_per_week": (float(np.nanmin(repro)), float(np.nanmax(repro))),
            "avg_watch_time_hours": (float(np.nanmin(tiempo)), float(np.nanmax(tiempo))),
        }
    (r_min, r_max), (t_min, t_max) = bounds["num_streams_per_week"], bounds["avg_watch_time_hours"]
    repro_norm = (repro - r_min) / (r_max - r_min)
    tiempo_norm = (tiempo - t_min) / (t_max - t_min)
    df["engagement"] = 0.4 * repro_norm + 0.6 * tiempo_norm
    return df, bounds


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, NUM_COLS), ("cat", cat_pipe, CAT_COLS)])


def main() -> None:
    df, bounds = add_engagement(pd.read_csv(DATA))
    X, y = df[NUM_COLS + CAT_COLS], df["churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- baseline: does the dataset carry any signal at all? ----
    dummy_auc = cross_validate(
        Pipeline([("prep", build_preprocessor()), ("model", DummyClassifier(strategy="stratified", random_state=SEED))]),
        X_train, y_train, cv=cv, scoring="roc_auc")["test_score"].mean()
    logreg_auc = cross_validate(
        Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))]),
        X_train, y_train, cv=cv, scoring="roc_auc")["test_score"].mean()
    print(f"Baseline (stratified dummy) CV AUC = {dummy_auc:.3f}")
    print(f"Logistic regression         CV AUC = {logreg_auc:.3f}")

    # ---- tuned random forest (as in the original exam) ----
    pipe = Pipeline([("preprocess", build_preprocessor()),
                     ("model", RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"))])
    param_distributions = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(pipe, param_distributions, n_iter=20, scoring="roc_auc",
                                cv=cv, random_state=SEED, n_jobs=-1)
    print("\n=== RandomizedSearchCV (Random Forest) ===")
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    print("Best params:", search.best_params_)
    print(f"Best CV AUC: {search.best_score_:.3f}")

    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    resultados = Resultados(
        accuracy=accuracy_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        auc=roc_auc_score(y_test, y_proba),
        n_train=len(X_train), n_test=len(X_test),
        best_params=search.best_params_,
    )
    print("\n=== Test set ===")
    print(f"Accuracy={resultados.accuracy:.4f}  F1={resultados.f1:.4f}  AUC={resultados.auc:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

    model_path = ROOT / "modelo_fuga.joblib"
    with open(model_path, "wb") as f:          # bug #2 fixed here (write side was already correct)
        joblib.dump(best_pipe, f)
    (ROOT / "engagement_bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")

    (RESULTS / "metrics.json").write_text(json.dumps({
        "baseline_dummy_cv_auc": dummy_auc,
        "logistic_regression_cv_auc": logreg_auc,
        "random_forest_cv_auc": search.best_score_,
        "test": asdict(resultados),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_features": len(NUM_COLS) + len(CAT_COLS),
        "feature_columns": NUM_COLS + CAT_COLS,
    }, indent=2, default=float), encoding="utf-8")

    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {RESULTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
