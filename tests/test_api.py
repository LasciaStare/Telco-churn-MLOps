"""
Unit tests for FastAPI endpoints
"""
import pytest
from fastapi.testclient import TestClient
from app.api import app


@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)


@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing"""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }


@pytest.fixture
def sample_batch_data():
    """Sample batch data for testing"""
    return {
        "customers": [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            },
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 34,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 56.95,
                "TotalCharges": 1889.5
            }
        ]
    }


class TestRootEndpoint:
    """Tests for the root endpoint"""

    def test_root_endpoint(self, client):
        """Test that root endpoint returns welcome message"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Telco Churn Prediction API" in data["message"]


class TestHealthEndpoint:
    """Tests for the health check endpoint"""

    def test_health_check(self, client):
        """Test that health endpoint returns healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Tests for the single prediction endpoint"""

    def test_predict_endpoint_success(self, client, sample_customer_data):
        """Test successful prediction with valid data"""
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "churn_prediction" in data
        assert "churn_probability" in data
        
        # Check data types and ranges
        assert isinstance(data["churn_prediction"], str)
        assert data["churn_prediction"] in ["Yes", "No"]
        assert isinstance(data["churn_probability"], float)
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_endpoint_invalid_gender(self, client, sample_customer_data):
        """Test prediction with invalid gender value"""
        sample_customer_data["gender"] = "Invalid"
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_invalid_senior_citizen(self, client, sample_customer_data):
        """Test prediction with invalid SeniorCitizen value"""
        sample_customer_data["SeniorCitizen"] = 2  # Should be 0 or 1
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 422

    def test_predict_endpoint_negative_tenure(self, client, sample_customer_data):
        """Test prediction with negative tenure"""
        sample_customer_data["tenure"] = -1
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 422

    def test_predict_endpoint_negative_charges(self, client, sample_customer_data):
        """Test prediction with negative monthly charges"""
        sample_customer_data["MonthlyCharges"] = -10.0
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 422

    def test_predict_endpoint_missing_field(self, client, sample_customer_data):
        """Test prediction with missing required field"""
        del sample_customer_data["gender"]
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 422

    def test_predict_endpoint_extra_field(self, client, sample_customer_data):
        """Test prediction with extra field (should be ignored)"""
        sample_customer_data["extra_field"] = "ignored"
        response = client.post("/predict", json=sample_customer_data)
        # Pydantic by default ignores extra fields
        assert response.status_code == 200


class TestPredictBatchEndpoint:
    """Tests for the batch prediction endpoint"""

    def test_batch_prediction_success(self, client, sample_batch_data):
        """Test successful batch prediction"""
        response = client.post("/predict-batch", json=sample_batch_data)
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        
        # Check each prediction
        for prediction in data["predictions"]:
            assert "churn_prediction" in prediction
            assert "churn_probability" in prediction
            assert prediction["churn_prediction"] in ["Yes", "No"]
            assert 0.0 <= prediction["churn_probability"] <= 1.0

    def test_batch_prediction_empty_list(self, client):
        """Test batch prediction with empty customer list"""
        response = client.post("/predict-batch", json={"customers": []})
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"] == []

    def test_batch_prediction_invalid_customer(self, client, sample_batch_data):
        """Test batch prediction with one invalid customer"""
        sample_batch_data["customers"][0]["gender"] = "Invalid"
        response = client.post("/predict-batch", json=sample_batch_data)
        assert response.status_code == 422

    def test_batch_prediction_single_customer(self, client, sample_customer_data):
        """Test batch prediction with single customer"""
        response = client.post("/predict-batch", json={"customers": [sample_customer_data]})
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1


class TestModelIntegration:
    """Integration tests for model predictions"""

    def test_consistent_predictions(self, client, sample_customer_data):
        """Test that same input gives consistent predictions"""
        response1 = client.post("/predict", json=sample_customer_data)
        response2 = client.post("/predict", json=sample_customer_data)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Predictions should be identical
        assert response1.json() == response2.json()

    def test_batch_vs_single_consistency(self, client, sample_customer_data):
        """Test that batch and single predictions are consistent"""
        # Single prediction
        single_response = client.post("/predict", json=sample_customer_data)
        single_data = single_response.json()
        
        # Batch prediction with same customer
        batch_response = client.post("/predict-batch", json={"customers": [sample_customer_data]})
        batch_data = batch_response.json()
        
        assert single_response.status_code == 200
        assert batch_response.status_code == 200
        
        # Compare predictions
        assert single_data["churn_prediction"] == batch_data["predictions"][0]["churn_prediction"]
        assert abs(single_data["churn_probability"] - batch_data["predictions"][0]["churn_probability"]) < 1e-6


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_zero_tenure(self, client, sample_customer_data):
        """Test prediction with zero tenure"""
        sample_customer_data["tenure"] = 0
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 200

    def test_high_tenure(self, client, sample_customer_data):
        """Test prediction with very high tenure"""
        sample_customer_data["tenure"] = 72  # Max tenure in dataset
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 200

    def test_zero_charges(self, client, sample_customer_data):
        """Test prediction with zero charges"""
        sample_customer_data["MonthlyCharges"] = 0.0
        sample_customer_data["TotalCharges"] = 0.0
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 200

    def test_no_internet_service(self, client, sample_customer_data):
        """Test prediction with no internet service"""
        sample_customer_data["InternetService"] = "No"
        sample_customer_data["OnlineSecurity"] = "No internet service"
        sample_customer_data["OnlineBackup"] = "No internet service"
        sample_customer_data["DeviceProtection"] = "No internet service"
        sample_customer_data["TechSupport"] = "No internet service"
        sample_customer_data["StreamingTV"] = "No internet service"
        sample_customer_data["StreamingMovies"] = "No internet service"
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 200
