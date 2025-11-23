"""
Unit tests for model loading and prediction functionality
"""
import pytest
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def model_path():
    """Path to the trained model"""
    return Path("c:/Users/Jose/Documents/GitHub/Telco-churn-MLOps/app/model.joblib")


@pytest.fixture
def model(model_path):
    """Load the trained model"""
    if not model_path.exists():
        pytest.skip(f"Model file not found at {model_path}. Run notebook 2 first.")
    return joblib.load(model_path)


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with customer data"""
    return pd.DataFrame({
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'tenure': [1],
        'PhoneService': ['No'],
        'MultipleLines': ['No phone service'],
        'InternetService': ['DSL'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [29.85],
        'TotalCharges': [29.85]
    })


@pytest.fixture
def sample_batch_dataframe():
    """Sample DataFrame with multiple customers"""
    return pd.DataFrame({
        'gender': ['Female', 'Male'],
        'SeniorCitizen': [0, 0],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'No'],
        'tenure': [1, 34],
        'PhoneService': ['No', 'Yes'],
        'MultipleLines': ['No phone service', 'No'],
        'InternetService': ['DSL', 'DSL'],
        'OnlineSecurity': ['No', 'Yes'],
        'OnlineBackup': ['Yes', 'No'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport': ['No', 'No'],
        'StreamingTV': ['No', 'No'],
        'StreamingMovies': ['No', 'No'],
        'Contract': ['Month-to-month', 'One year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check'],
        'MonthlyCharges': [29.85, 56.95],
        'TotalCharges': [29.85, 1889.5]
    })


class TestModelLoading:
    """Tests for model loading functionality"""

    def test_model_file_exists(self, model_path):
        """Test that model file exists"""
        assert model_path.exists(), f"Model file not found at {model_path}"

    def test_model_loads_successfully(self, model):
        """Test that model can be loaded"""
        assert model is not None

    def test_model_is_pipeline(self, model):
        """Test that loaded model is a scikit-learn Pipeline"""
        from sklearn.pipeline import Pipeline
        assert isinstance(model, Pipeline), "Model should be a scikit-learn Pipeline"

    def test_model_has_required_methods(self, model):
        """Test that model has required prediction methods"""
        assert hasattr(model, 'predict'), "Model should have predict method"
        assert hasattr(model, 'predict_proba'), "Model should have predict_proba method"
        assert callable(model.predict), "predict should be callable"
        assert callable(model.predict_proba), "predict_proba should be callable"


class TestModelPrediction:
    """Tests for model prediction functionality"""

    def test_predict_single_sample(self, model, sample_dataframe):
        """Test prediction on single sample"""
        prediction = model.predict(sample_dataframe)
        
        assert prediction is not None
        assert len(prediction) == 1
        assert prediction[0] in [0, 1], "Prediction should be binary (0 or 1)"

    def test_predict_proba_single_sample(self, model, sample_dataframe):
        """Test probability prediction on single sample"""
        probabilities = model.predict_proba(sample_dataframe)
        
        assert probabilities is not None
        assert probabilities.shape == (1, 2), "Should return probabilities for both classes"
        assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1), "Probabilities should be between 0 and 1"

    def test_predict_batch(self, model, sample_batch_dataframe):
        """Test prediction on multiple samples"""
        predictions = model.predict(sample_batch_dataframe)
        
        assert predictions is not None
        assert len(predictions) == 2
        assert all(pred in [0, 1] for pred in predictions), "All predictions should be binary"

    def test_predict_proba_batch(self, model, sample_batch_dataframe):
        """Test probability prediction on multiple samples"""
        probabilities = model.predict_proba(sample_batch_dataframe)
        
        assert probabilities is not None
        assert probabilities.shape == (2, 2), "Should return probabilities for both samples"
        assert np.allclose(probabilities.sum(axis=1), 1.0), "Each row should sum to 1"


class TestPredictionConsistency:
    """Tests for prediction consistency"""

    def test_prediction_deterministic(self, model, sample_dataframe):
        """Test that predictions are deterministic"""
        pred1 = model.predict(sample_dataframe)
        pred2 = model.predict(sample_dataframe)
        
        assert np.array_equal(pred1, pred2), "Same input should give same prediction"

    def test_probability_deterministic(self, model, sample_dataframe):
        """Test that probability predictions are deterministic"""
        prob1 = model.predict_proba(sample_dataframe)
        prob2 = model.predict_proba(sample_dataframe)
        
        assert np.allclose(prob1, prob2), "Same input should give same probabilities"

    def test_prediction_probability_consistency(self, model, sample_dataframe):
        """Test that predictions are consistent with probabilities"""
        prediction = model.predict(sample_dataframe)[0]
        probabilities = model.predict_proba(sample_dataframe)[0]
        
        # If prediction is 1, probability for class 1 should be > 0.5
        if prediction == 1:
            assert probabilities[1] > 0.5, "Prediction 1 should have probability > 0.5"
        else:
            assert probabilities[0] > 0.5, "Prediction 0 should have probability > 0.5"


class TestInputValidation:
    """Tests for input validation and edge cases"""

    def test_missing_column_raises_error(self, model, sample_dataframe):
        """Test that missing column raises error"""
        incomplete_df = sample_dataframe.drop('gender', axis=1)
        
        with pytest.raises(Exception):
            model.predict(incomplete_df)

    def test_extra_column_ignored(self, model, sample_dataframe):
        """Test that extra columns are handled appropriately"""
        sample_dataframe['extra_column'] = 'ignored'
        
        # Should not raise error (pipeline should handle it)
        try:
            prediction = model.predict(sample_dataframe)
            assert prediction is not None
        except Exception as e:
            pytest.fail(f"Model should handle extra columns: {e}")

    def test_zero_values(self, model, sample_dataframe):
        """Test prediction with zero values"""
        sample_dataframe['tenure'] = 0
        sample_dataframe['MonthlyCharges'] = 0.0
        sample_dataframe['TotalCharges'] = 0.0
        
        prediction = model.predict(sample_dataframe)
        assert prediction is not None

    def test_high_values(self, model, sample_dataframe):
        """Test prediction with high values"""
        sample_dataframe['tenure'] = 72
        sample_dataframe['MonthlyCharges'] = 118.75
        sample_dataframe['TotalCharges'] = 8684.8
        
        prediction = model.predict(sample_dataframe)
        assert prediction is not None


class TestDifferentScenarios:
    """Tests for different customer scenarios"""

    def test_no_internet_service(self, model):
        """Test prediction for customer with no internet service"""
        df = pd.DataFrame({
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['Yes'],
            'Dependents': ['Yes'],
            'tenure': [12],
            'PhoneService': ['Yes'],
            'MultipleLines': ['Yes'],
            'InternetService': ['No'],
            'OnlineSecurity': ['No internet service'],
            'OnlineBackup': ['No internet service'],
            'DeviceProtection': ['No internet service'],
            'TechSupport': ['No internet service'],
            'StreamingTV': ['No internet service'],
            'StreamingMovies': ['No internet service'],
            'Contract': ['One year'],
            'PaperlessBilling': ['No'],
            'PaymentMethod': ['Bank transfer (automatic)'],
            'MonthlyCharges': [20.0],
            'TotalCharges': [240.0]
        })
        
        prediction = model.predict(df)
        probabilities = model.predict_proba(df)
        
        assert prediction is not None
        assert probabilities is not None

    def test_senior_citizen(self, model):
        """Test prediction for senior citizen"""
        df = pd.DataFrame({
            'gender': ['Female'],
            'SeniorCitizen': [1],
            'Partner': ['No'],
            'Dependents': ['No'],
            'tenure': 36,
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['Fiber optic'],
            'OnlineSecurity': ['Yes'],
            'OnlineBackup': ['Yes'],
            'DeviceProtection': ['Yes'],
            'TechSupport': ['Yes'],
            'StreamingTV': ['Yes'],
            'StreamingMovies': ['Yes'],
            'Contract': ['Two year'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Credit card (automatic)'],
            'MonthlyCharges': [105.0],
            'TotalCharges': [3780.0]
        })
        
        prediction = model.predict(df)
        assert prediction is not None

    def test_long_tenure_customer(self, model):
        """Test prediction for long-tenure customer"""
        df = pd.DataFrame({
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['Yes'],
            'Dependents': ['Yes'],
            'tenure': [72],
            'PhoneService': ['Yes'],
            'MultipleLines': ['Yes'],
            'InternetService': ['Fiber optic'],
            'OnlineSecurity': ['Yes'],
            'OnlineBackup': ['Yes'],
            'DeviceProtection': ['Yes'],
            'TechSupport': ['Yes'],
            'StreamingTV': ['Yes'],
            'StreamingMovies': ['Yes'],
            'Contract': ['Two year'],
            'PaperlessBilling': ['No'],
            'PaymentMethod': ['Bank transfer (automatic)'],
            'MonthlyCharges': [110.0],
            'TotalCharges': [7920.0]
        })
        
        prediction = model.predict(df)
        probabilities = model.predict_proba(df)
        
        assert prediction is not None
        # Long tenure customers should typically have lower churn probability
        assert probabilities is not None


class TestModelPerformance:
    """Tests for basic model performance characteristics"""

    def test_probability_ranges(self, model, sample_batch_dataframe):
        """Test that all probabilities are in valid range"""
        probabilities = model.predict_proba(sample_batch_dataframe)
        
        assert np.all(probabilities >= 0.0), "All probabilities should be >= 0"
        assert np.all(probabilities <= 1.0), "All probabilities should be <= 1"

    def test_probability_sum(self, model, sample_batch_dataframe):
        """Test that probabilities sum to 1"""
        probabilities = model.predict_proba(sample_batch_dataframe)
        row_sums = probabilities.sum(axis=1)
        
        assert np.allclose(row_sums, 1.0), "Probabilities should sum to 1 for each sample"

    def test_prediction_speed(self, model, sample_batch_dataframe):
        """Test that predictions are reasonably fast"""
        import time
        
        start_time = time.time()
        _ = model.predict(sample_batch_dataframe)
        elapsed_time = time.time() - start_time
        
        # Should predict 2 samples in less than 1 second
        assert elapsed_time < 1.0, f"Prediction took too long: {elapsed_time:.3f}s"
