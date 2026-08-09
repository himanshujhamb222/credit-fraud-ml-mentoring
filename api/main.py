"""FastAPI inference service for the selected credit-fraud model."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from credit_fraud import config
from credit_fraud.io_utils import load_json, load_model
from credit_fraud.modeling import predict_proba_positive

STATE: dict[str, Any] = {
    "model": None,
    "feature_builder": None,
    "feature_names": [],
    "raw_feature_names": [],
    "metrics": None,
}


def _load_artifacts() -> None:
    model_path = config.MODEL_PATH
    builder_path = config.ARTIFACTS_DIR / "feature_builder.joblib"
    contract_path = config.ARTIFACTS_DIR / "feature_contract.json"

    if not model_path.exists():
        STATE["model"] = None
        return

    STATE["model"] = load_model(model_path)
    if builder_path.exists():
        STATE["feature_builder"] = load_model(builder_path)
    if contract_path.exists():
        contract = load_json(contract_path)
        STATE["feature_names"] = contract.get("feature_names", [])
        STATE["raw_feature_names"] = contract.get("raw_feature_names", [])
    if config.METRICS_PATH.exists():
        STATE["metrics"] = load_json(config.METRICS_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="Credit Fraud Mentoring API",
    version="0.1.0",
    description="Score transactions with the selected RF/XGBoost model.",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """Either pass engineered features, or raw columns (Amount/Time/V1...)."""

    model_config = ConfigDict(extra="allow")

    features: dict[str, float] | None = Field(
        default=None,
        description="Flat feature map. Raw columns are engineered via FeatureBuilder when present.",
    )


class PredictResponse(BaseModel):
    fraud_probability: float
    predicted_label: int
    model_name: str | None = None
    threshold: float = 0.5


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = STATE["model"] is not None
    return {
        "status": "ok" if loaded else "model_missing",
        "model_loaded": loaded,
        "model_path": str(config.MODEL_PATH),
        "best_model": (STATE["metrics"] or {}).get("best_model"),
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    if STATE["metrics"] is None:
        if config.METRICS_PATH.exists():
            STATE["metrics"] = load_json(config.METRICS_PATH)
        else:
            raise HTTPException(status_code=503, detail="metrics.json not found. Run training first.")
    return STATE["metrics"]


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if STATE["model"] is None:
        _load_artifacts()
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    raw_map = payload.features if payload.features is not None else payload.model_dump(exclude_none=True)
    if "features" in raw_map:
        raw_map = raw_map["features"] or {}
    if not raw_map:
        raise HTTPException(status_code=422, detail="Provide a non-empty feature map.")

    row = pd.DataFrame([raw_map])
    builder = STATE["feature_builder"]
    feature_names: list[str] = STATE["feature_names"] or list(row.columns)

    try:
        if builder is not None:
            # Ensure expected raw columns exist for transform
            for col in STATE["raw_feature_names"]:
                if col not in row.columns:
                    row[col] = 0.0
            row = row[STATE["raw_feature_names"]] if STATE["raw_feature_names"] else row
            engineered = builder.transform(row)
        else:
            engineered = row

        # Align to training columns
        for col in feature_names:
            if col not in engineered.columns:
                engineered[col] = 0.0
        engineered = engineered[feature_names]
        proba = float(predict_proba_positive(STATE["model"], engineered)[0])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Failed to score row: {exc}") from exc

    threshold = 0.5
    return PredictResponse(
        fraud_probability=proba,
        predicted_label=int(proba >= threshold),
        model_name=(STATE["metrics"] or {}).get("best_model"),
        threshold=threshold,
    )


@app.post("/reload")
def reload_artifacts() -> dict[str, Any]:
    """Reload model artifacts without restarting the process."""
    _load_artifacts()
    return health()


# Allow `uvicorn api.main:app` from repo root
def create_app() -> FastAPI:
    return app
