"""
Unit Tests for FastAPI REST Inference & Monitoring Endpoints (QuantLab Pro).
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["service"] == "QuantLab Pro API"


def test_api_predict_endpoint():
    features = [0.1, -0.05, 0.02, 1.2, 0.4, -0.1, 0.05, 52.0, 0.01, 0.005, 0.005, 0.0, 0.0, 0.0]
    response = client.post("/predict", json={"features": features, "model_type": "logistic_regression"})

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]
    assert len(data["probabilities"]) == 2
    assert 0.0 <= data["confidence"] <= 1.0


def test_api_monitoring_endpoint():
    response = client.get("/monitoring/distribution")
    assert response.status_code == 200
    data = response.json()
    assert "data_quality_status" in data
