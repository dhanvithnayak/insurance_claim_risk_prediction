"""
Automated unit & integration test suite for the Car Insurance Claim API.
"""

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    """Test health check probe returns 200 and reports healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "auto-insurance-claim-risk-prediction"
    assert data["model_loaded"] is True


def test_web_dashboard_renders(client):
    """Test root endpoint returns 200 OK and serves HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Auto Insurance Claim Risk Engine" in response.text
    assert "Score Policy Risk" in response.text


def test_predict_standard_applicant(client):
    """Test prediction on standard experienced, low-risk customer."""
    payload = {
        "age": 2,
        "gender": 1,
        "driving_experience": "20-29y",
        "education": "university",
        "income": "upper class",
        "credit_score": 0.65,
        "vehicle_ownership": 1.0,
        "vehicle_year": "after 2015",
        "married": 1.0,
        "children": 1.0,
        "postal_code": 10238,
        "annual_mileage": 12000.0,
        "vehicle_type": "sedan",
        "speeding_violations": 0,
        "duis": 0,
        "past_accidents": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "claim_probability" in data
    assert 0.0 <= data["claim_probability"] <= 1.0
    assert data["predicted_claim"] == 0
    assert data["risk_tier"] == "Low"
    assert data["missing_fields_imputed"] == []


def test_predict_missing_values(client):
    """Test that missing credit_score and annual_mileage trigger median imputation without error."""
    payload = {
        "age": 0,
        "gender": 0,
        "driving_experience": "0-9y",
        "education": "none",
        "income": "poverty",
        "credit_score": None,       # Missing
        "vehicle_ownership": 0.0,
        "vehicle_year": "before 2015",
        "married": 0.0,
        "children": 0.0,
        "postal_code": 21217,
        "annual_mileage": None,     # Missing
        "vehicle_type": "sedan",
        "speeding_violations": 2,
        "duis": 1,
        "past_accidents": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["claim_probability"] > 0.8
    assert data["predicted_claim"] == 1
    assert data["risk_tier"] == "High"
    assert set(data["missing_fields_imputed"]) == {"credit_score", "annual_mileage"}


def test_predict_unseen_categories(client):
    """Test fault tolerance on unseen categorical values (graceful fallback to zeros)."""
    payload = {
        "age": 1,
        "gender": 1,
        "driving_experience": "10-19y",
        "education": "doctorate",       # Unseen
        "income": "middle class",
        "credit_score": 0.5,
        "vehicle_ownership": 1.0,
        "vehicle_year": "unknown_year", # Unseen
        "married": 1.0,
        "children": 0.0,
        "postal_code": 99999,           # Unseen
        "annual_mileage": 10000.0,
        "vehicle_type": "truck",        # Unseen
        "speeding_violations": 0,
        "duis": 0,
        "past_accidents": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["claim_probability"] <= 1.0
    assert data["predicted_claim"] in (0, 1)


def test_predict_schema_validation(client):
    """Test that invalid values (e.g. negative accidents) fail fast with HTTP 422."""
    payload = {
        "age": 2,
        "gender": 1,
        "driving_experience": "20-29y",
        "education": "university",
        "income": "upper class",
        "credit_score": 1.5,            # Invalid > 1.0
        "vehicle_ownership": 1.0,
        "vehicle_year": "after 2015",
        "married": 1.0,
        "children": 1.0,
        "postal_code": 10238,
        "annual_mileage": 12000.0,
        "vehicle_type": "sedan",
        "speeding_violations": -1,      # Invalid < 0
        "duis": 0,
        "past_accidents": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
