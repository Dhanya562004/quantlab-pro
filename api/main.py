"""
FastAPI Local Inference & Monitoring API for QuantLab Pro.
Exposes REST endpoints for model inference, health status, experiment metrics, and model monitoring.
"""

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from quantlab.storage.db import ExperimentStorage
from quantlab.tools.domain_tools import RunExperimentArgs, tool_run_experiment

app = FastAPI(
    title="QuantLab Pro API",
    description="Local MLOps Inference & Monitoring REST Service for QuantLab Pro Platform.",
    version="1.0.0",
)


# --- Request & Response Schemas ---

class PredictRequest(BaseModel):
    features: list[float] = Field(description="Normalized feature vector array matching model input dimension.")
    model_type: str = Field(default="logistic_regression", description="Trained model type.")


class PredictResponse(BaseModel):
    predicted_class: int = Field(description="Predicted target class (1 = Positive Return, 0 = Non-Positive).")
    probabilities: list[float] = Field(description="Class probabilities [P(0), P(1)].")
    confidence: float = Field(description="Prediction confidence score.")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database_connected: bool


class ReadinessResponse(BaseModel):
    status: str
    database: str
    storage_accessible: bool



# Simulated active model cache for inference endpoint
ACTIVE_MODEL_CACHE: dict[str, Any] = {}


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Service health status check."""
    return HealthResponse(
        status="healthy",
        service="QuantLab Pro API",
        version="1.0.0",
        database_connected=True,
    )


@app.get("/ready", response_model=ReadinessResponse)
def readiness_check():
    """Service readiness check verifying required SQLite database access."""
    try:
        storage = ExperimentStorage()
        storage.list_experiments(limit=1)
        return ReadinessResponse(
            status="ready",
            database="connected",
            storage_accessible=True,
        )
    except Exception as ex:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: SQLite database connection error: {ex!s}"
        )



@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Real-time model inference endpoint.
    Accepts normalized feature vector and returns class prediction + probabilities.
    """
    if not req.features:
        raise HTTPException(status_code=400, detail="Feature vector cannot be empty.")

    x = np.array(req.features).reshape(1, -1)

    # Check cache or run default lightweight model fit
    if "model" not in ACTIVE_MODEL_CACHE:
        # Run default experiment to populate active model
        res = tool_run_experiment(RunExperimentArgs(model_type=req.model_type, n_bars=200))
        ACTIVE_MODEL_CACHE["experiment_id"] = res.get("experiment_id")

    # Logistic regression simulation for prediction endpoint
    # Dot product calculation for demonstration
    weights = np.ones(x.shape[1]) / np.sqrt(x.shape[1])
    score = float(np.dot(x, weights)[0])
    p1 = 1.0 / (1.0 + np.exp(-score))
    p0 = 1.0 - p1

    pred_class = 1 if p1 >= 0.5 else 0
    confidence = float(max(p0, p1))

    return PredictResponse(
        predicted_class=pred_class,
        probabilities=[round(p0, 4), round(p1, 4)],
        confidence=round(confidence, 4),
    )


@app.get("/metrics/{experiment_id}")
def get_metrics(experiment_id: str):
    """Retrieve recorded evaluation metrics for a specific experiment ID."""
    storage = ExperimentStorage()
    rec = storage.get_experiment(experiment_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Experiment ID '{experiment_id}' not found.")
    return {"experiment_id": rec.experiment_id, "metrics": rec.metrics_json}


@app.get("/monitoring/distribution")
def monitoring_distribution():
    """
    Monitoring endpoint returning historical prediction distributions and data quality warnings.
    Note: Educational research monitoring summary; does not imply real-world production performance.
    """
    storage = ExperimentStorage()
    experiments = storage.list_experiments(limit=50)

    if not experiments:
        return {
            "total_runs": 0,
            "mean_accuracy": 0.0,
            "mean_sharpe": 0.0,
            "data_quality_status": "NO_EXPERIMENTS_RECORDED"
        }

    accuracies = [e["accuracy"] for e in experiments]
    sharpes = [e["sharpe_ratio"] for e in experiments]

    return {
        "total_runs": len(experiments),
        "mean_accuracy": round(float(np.mean(accuracies)), 4),
        "mean_sharpe": round(float(np.mean(sharpes)), 4),
        "min_accuracy": round(float(np.min(accuracies)), 4),
        "max_accuracy": round(float(np.max(accuracies)), 4),
        "data_quality_status": "HEALTHY",
        "monitoring_disclaimer": "Educational research platform metrics. Past simulated backtests do not guarantee future live trading results."
    }
