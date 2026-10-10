"""
Unit Tests for FastAPI REST Inference & Monitoring Endpoints (QuantLab Pro).
"""

from fastapi.testclient import TestClient

from api.main import app
from quantlab.tools.domain_tools import RunExperimentArgs, tool_run_experiment

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "QuantLab Pro API"


def test_api_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["database"] == "connected"


def test_api_predict_endpoint():
    features = [0.1, -0.05, 0.02, 1.2, 0.4, -0.1, 0.05, 52.0, 0.01, 0.005, 0.005, 0.0, 0.0, 0.0]
    response = client.post("/predict", json={"features": features, "model_type": "logistic_regression"})

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]
    assert len(data["probabilities"]) == 2
    assert 0.0 <= data["confidence"] <= 1.0


def test_api_predict_empty_features_error():
    response = client.post("/predict", json={"features": [], "model_type": "logistic_regression"})
    assert response.status_code == 400


def test_api_metrics_endpoint_and_404():
    # Run a real experiment to populate DB
    res = tool_run_experiment(RunExperimentArgs(symbol="API_TEST", n_bars=200))
    exp_id = res["experiment_id"]

    # Test valid experiment ID
    response = client.get(f"/metrics/{exp_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == exp_id
    assert "accuracy" in data["metrics"]

    # Test invalid experiment ID
    response_404 = client.get("/metrics/NON_EXISTENT_EXP_999")
    assert response_404.status_code == 404


def test_api_monitoring_endpoint():
    response = client.get("/monitoring/distribution")
    assert response.status_code == 200
    data = response.json()
    assert "data_quality_status" in data
