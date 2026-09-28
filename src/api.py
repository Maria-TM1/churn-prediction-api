"""
Churn Prediction Service — FastAPI (corrected).

Serves the model trained by `train_model.py`. Computes the same derived
`engagement` feature, validates every input field with Pydantic, and
requires a bearer token.

Bug fixed here: the original exam hardcoded a fallback API token
("supersecreto123") when the API_TOKEN environment variable was missing.
That means anyone reading the source code — or the public GitHub repo it
might get pushed to — would know the default credential of every
deployment that forgot to set it. The service now refuses to start
without an explicit token.

Run:
    export API_TOKEN=choose-a-real-secret     # PowerShell: $env:API_TOKEN = "..."
    uvicorn src.api:app --reload
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

ROOT = Path(__file__).resolve().parent.parent

API_TOKEN = os.environ.get("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError(
        "API_TOKEN is not set. Refusing to start with a default credential — "
        "export API_TOKEN before launching the service."
    )
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "modelo_fuga.joblib"))
BOUNDS_PATH = Path(os.environ.get("BOUNDS_PATH", ROOT / "engagement_bounds.json"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
RATE_LIMIT = os.environ.get("RATE_LIMIT", "10/minute")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Streamflix Churn Prediction Service",
    description=("FastAPI service exposing the trained churn model for Streamflix customers. "
                "Receives customer data (age, country, device, plan, etc.) and returns the churn probability."),
    version=MODEL_VERSION,
    lifespan=lifespan,
)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

model = None
engagement_bounds: dict | None = None
security = HTTPBearer()


class CustomerInput(BaseModel):
    customer_id: Optional[int] = Field(None, ge=1, description="Optional customer identifier.", examples=[123])
    age: int = Field(..., ge=18, le=100, description="Customer age.", examples=[35])
    country: str = Field(..., description="Country (ES, IT, FR, DE, PT).", examples=["ES"])
    device: str = Field(..., description="Device (web, mobile, smart_tv).", examples=["mobile"])
    plan_type: str = Field(..., description="Plan tier (basic, standard, premium).", examples=["standard"])
    tenure_months: int = Field(..., ge=0, le=120, description="Months as a customer.", examples=[24])
    num_streams_per_week: int = Field(..., ge=0, le=100, description="Streams per week.", examples=[10])
    avg_watch_time_hours: float = Field(..., gt=0, le=24, description="Average watch time per session (hours).", examples=[3.5])
    support_tickets_last_6m: int = Field(..., ge=0, le=50, description="Support tickets in the last 6 months.", examples=[1])
    auto_renew: int = Field(..., ge=0, le=1, description="1 if auto-renew is on, 0 otherwise.", examples=[1])
    payment_method: str = Field(..., description="Payment method (card, paypal, bank_transfer).", examples=["card"])

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        allowed = {"ES", "IT", "FR", "DE", "PT"}
        if v not in allowed:
            raise ValueError(f"country must be one of {sorted(allowed)}")
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"web", "mobile", "smart_tv"}
        if v not in allowed:
            raise ValueError(f"device must be one of {sorted(allowed)}")
        return v

    @field_validator("plan_type")
    @classmethod
    def validate_plan(cls, v: str) -> str:
        allowed = {"basic", "standard", "premium"}
        if v not in allowed:
            raise ValueError(f"plan_type must be one of {sorted(allowed)}")
        return v

    @field_validator("payment_method")
    @classmethod
    def validate_payment(cls, v: str) -> str:
        allowed = {"card", "paypal", "bank_transfer"}
        if v not in allowed:
            raise ValueError(f"payment_method must be one of {sorted(allowed)}")
        return v


class ChurnPrediction(BaseModel):
    churn_pred: int
    probability: float


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> None:
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API token")


def load_model() -> None:
    global model, engagement_bounds
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. Run src/train_model.py first.")
    with open(MODEL_PATH, "rb") as f:          # bug fixed: was opened in text mode ("r") in the original exam
        model = joblib.load(f)
    if not BOUNDS_PATH.exists():
        raise RuntimeError(f"Engagement bounds file not found: {BOUNDS_PATH}. Run src/train_model.py first.")
    engagement_bounds = json.loads(BOUNDS_PATH.read_text(encoding="utf-8"))


def build_features(payload: CustomerInput) -> pd.DataFrame:
    """Reconstructs the same `engagement` feature used at training time, by
    re-applying the min/max bounds saved by train_model.py — a single request
    has no distribution of its own to normalise against."""
    row = payload.model_dump(exclude={"customer_id"})
    (r_min, r_max) = engagement_bounds["num_streams_per_week"]
    (t_min, t_max) = engagement_bounds["avg_watch_time_hours"]
    repro_norm = (row["num_streams_per_week"] - r_min) / (r_max - r_min)
    tiempo_norm = (row["avg_watch_time_hours"] - t_min) / (t_max - t_min)
    row["engagement"] = float(np.clip(0.4 * repro_norm + 0.6 * tiempo_norm, 0.0, 1.0))
    return pd.DataFrame([row])


@app.post("/predict", response_model=ChurnPrediction, dependencies=[Depends(verify_token)])
@limiter.limit(RATE_LIMIT)
def predict(request: Request, payload: CustomerInput) -> ChurnPrediction:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = build_features(payload)
    proba = float(model.predict_proba(X)[:, 1][0])
    return ChurnPrediction(churn_pred=int(proba >= 0.5), probability=round(proba, 4))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None, "version": MODEL_VERSION}


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return HTTPException(status_code=429, detail="Rate limit exceeded")


def main() -> None:
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
