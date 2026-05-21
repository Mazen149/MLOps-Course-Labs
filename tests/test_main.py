"""Tests for the Churn Prediction API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""

import pytest
from litestar.testing import TestClient

from app.model_utils import predict_churn
from main import app


# ---------------------------------------------------------------------------
# Function Tests
# ---------------------------------------------------------------------------


# TODO 1: Write a test that calls predict_churn() directly with sample features
#         and asserts the result is 0 or 1
def test_predict_churn_direct_valid():
    """Validates the core prediction function with standard sample data."""
    # Order: [CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography, Gender]
    sample_features = [619, 42, 2, 0.0, 1, 1, 1, 101348.88, "France", "Female"]

    prediction = predict_churn(sample_features)

    assert isinstance(prediction, int)
    assert prediction in (0, 1)


# TODO 2 (bonus): Write another function test with edge-case inputs
def test_predict_churn_direct_edge_cases():
    """Tests the model logic against extreme baseline boundaries."""
    # Low score, older age, maximized products/balances, alternative categories
    edge_features = [350, 85, 0, 250000.0, 4, 0, 0, 0.0, "Germany", "Male"]

    prediction = predict_churn(edge_features)

    assert isinstance(prediction, int)
    assert prediction in (0, 1)


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------


# TODO 3: Write a test that POSTs to /predict with valid JSON
#         and checks the status code and response body
def test_post_predict_endpoint_success():
    """Validates HTTP 201 Created and response format for valid payloads."""
    payload = {
        "credit_score": 600,
        "age": 40,
        "tenure": 3,
        "balance": 60000.0,
        "num_of_products": 2,
        "has_cr_card": 1,
        "is_active_member": 1,
        "estimated_salary": 50000.0,
        "geography": "Spain",
        "gender": "Male",
    }

    with TestClient(app=app) as client:
        response = client.post("/predict", json=payload)

        # Litestar returns 201 Created by default for successful POST operations
        assert response.status_code == 201

        # Check response body fields
        data = response.json()
        assert "churn_prediction" in data
        assert data["churn_prediction"] in (0, 1)


# TODO 4: Write a test for GET /health
def test_get_health_endpoint():
    """Validates that health check route returns operational status."""
    with TestClient(app=app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


# TODO 5: Write a test for GET /
def test_get_index_endpoint():
    """Validates that welcome root route renders successfully."""
    with TestClient(app=app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.text == "Welcome There!"


# TODO 6 (bonus): Test that invalid input returns status 400
@pytest.mark.parametrize(
    "bad_payload",
    [
        # Missing required parameter "age"
        {
            "credit_score": 600,
            "tenure": 3,
            "balance": 0.0,
            "num_of_products": 1,
            "has_cr_card": 1,
            "is_active_member": 1,
            "estimated_salary": 50000.0,
            "geography": "France",
            "gender": "Female",
        },
        # Invalid categorical string for "geography"
        {
            "credit_score": 600,
            "age": 40,
            "tenure": 3,
            "balance": 0.0,
            "num_of_products": 1,
            "has_cr_card": 1,
            "is_active_member": 1,
            "estimated_salary": 50000.0,
            "geography": "Egypt",  # Not allowed by Literal validation
            "gender": "Female",
        },
    ],
)
def test_post_predict_validation_failure(bad_payload):
    """Ensures Pydantic errors catch malformed schemas with an HTTP 400."""
    with TestClient(app=app) as client:
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 400
