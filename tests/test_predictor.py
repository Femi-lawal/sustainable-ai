"""
Unit Tests for the Energy Prediction Module (estimator.py).
Tests supervised ML energy prediction.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestEnergyPredictor:
    """Test EnergyPredictor class."""
    
    def test_predictor_initialization(self, energy_predictor):
        """Test predictor initializes correctly."""
        assert energy_predictor is not None
        assert energy_predictor.model_type == "random_forest"
    
    def test_create_random_forest_model(self, temp_model_dir):
        """Test creating a Random Forest model."""
        from prediction.estimator import EnergyPredictor
        predictor = EnergyPredictor(model_type="random_forest")
        predictor.model_path = temp_model_dir / "rf_model.joblib"
        predictor._create_model()
        assert predictor.model is not None
        assert predictor.scaler is not None
    
    def test_create_gradient_boost_model(self, temp_model_dir):
        """Test creating a Gradient Boosting model."""
        from prediction.estimator import EnergyPredictor
        predictor = EnergyPredictor(model_type="gradient_boost")
        predictor.model_path = temp_model_dir / "gb_model.joblib"
        predictor._create_model()
        assert predictor.model is not None
    
    def test_invalid_model_type_raises_error(self, temp_model_dir):
        """Test that invalid model type raises error."""
        from prediction.estimator import EnergyPredictor
        predictor = EnergyPredictor(model_type="invalid")
        predictor.model_path = temp_model_dir / "invalid_model.joblib"
        with pytest.raises(ValueError):
            predictor._create_model()
    
    def test_extract_features(self, energy_predictor, sample_prompts):
        """Test feature extraction from prompt."""
        features = energy_predictor.extract_features(sample_prompts["medium"])
        assert isinstance(features, dict)
        assert "token_count" in features
        assert "complexity_score" in features
        assert "char_count" in features
    
    def test_extract_features_with_model_params(self, energy_predictor, sample_prompts):
        """Test feature extraction with model parameters."""
        features = energy_predictor.extract_features(
            sample_prompts["medium"],
            num_layers=48,
            training_hours=24.0,
            flops_per_hour=1e12
        )
        # Calibrated model uses only 6 core features, original uses more
        if getattr(energy_predictor, 'is_calibrated', False):
            # Calibrated model: only core NLP features
            assert "token_count" in features
            assert "word_count" in features
            assert "complexity_score" in features
            # Model params not included in calibrated features
        else:
            # Original model: includes model parameters
            assert features["num_layers"] == 48
            assert features["training_hours"] == 24.0
            assert features["flops_per_hour"] == 1e12
    
    def test_predict_returns_energy_prediction(self, energy_predictor, sample_prompts):
        """Test that predict returns EnergyPrediction object."""
        result = energy_predictor.predict(sample_prompts["simple"])
        assert hasattr(result, "energy_kwh")
        assert hasattr(result, "carbon_footprint_kg")
        assert hasattr(result, "energy_level")
        assert hasattr(result, "confidence_score")
    
    def test_predict_energy_positive(self, energy_predictor, sample_prompts):
        """Test that predicted energy is positive."""
        result = energy_predictor.predict(sample_prompts["medium"])
        assert result.energy_kwh > 0
    
    def test_predict_carbon_footprint_calculated(self, energy_predictor, sample_prompts):
        """Test carbon footprint calculation."""
        result = energy_predictor.predict(sample_prompts["medium"])
        assert result.carbon_footprint_kg > 0
        assert result.carbon_footprint_kg == pytest.approx(
            result.energy_kwh * 0.21,  # California carbon intensity
            rel=0.5
        )
    
    def test_predict_water_usage_calculated(self, energy_predictor, sample_prompts):
        """Test water usage calculation."""
        result = energy_predictor.predict(sample_prompts["medium"])
        assert result.water_usage_liters > 0
    
    def test_predict_electricity_cost_calculated(self, energy_predictor, sample_prompts):
        """Test electricity cost calculation."""
        result = energy_predictor.predict(sample_prompts["medium"])
        assert result.electricity_cost_usd >= 0
    
    def test_energy_level_classification(self, energy_predictor, sample_prompts):
        """Test energy level classification."""
        result = energy_predictor.predict(sample_prompts["medium"])
        assert result.energy_level in ["low", "medium", "high", "very_high"]
    
    def test_comparison_to_average(self, energy_predictor, sample_prompts):
        """Test comparison to average calculation."""
        result = energy_predictor.predict(sample_prompts["medium"])
        assert isinstance(result.comparison_to_average, float)
    
    def test_prediction_to_dict(self, energy_predictor, sample_prompts):
        """Test EnergyPrediction.to_dict() method."""
        result = energy_predictor.predict(sample_prompts["medium"])
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "energy_kwh" in result_dict
        assert "carbon_footprint_kg" in result_dict
    
    def test_predict_batch(self, energy_predictor, sample_prompts):
        """Test batch prediction."""
        prompts = [sample_prompts["simple"], sample_prompts["medium"]]
        results = energy_predictor.predict_batch(prompts)
        assert len(results) == 2
        assert all(hasattr(r, "energy_kwh") for r in results)
    
    def test_formula_based_estimation(self, energy_predictor, sample_energy_features):
        """Test fallback formula-based estimation."""
        energy = energy_predictor._estimate_energy_formula(sample_energy_features)
        assert energy > 0
    
    def test_different_regions(self, energy_predictor, sample_prompts):
        """Test predictions for different regions."""
        regions = ["california", "texas", "eu_average"]
        results = {}
        for region in regions:
            results[region] = energy_predictor.predict(
                sample_prompts["medium"],
                region=region
            )
        
        # Carbon footprint should differ by region
        ca_carbon = results["california"].carbon_footprint_kg
        tx_carbon = results["texas"].carbon_footprint_kg
        assert ca_carbon != tx_carbon  # Different carbon intensities


class TestEnergyPredictorTraining:
    """Test EnergyPredictor training functionality."""
    
    def test_train_model(self, energy_predictor, sample_training_data):
        """Test model training."""
        metrics = energy_predictor.train(sample_training_data, target_column="energy_kwh")
        assert "test_rmse" in metrics
        assert "test_mae" in metrics
        assert "test_r2" in metrics
        assert energy_predictor.is_trained
    
    def test_train_saves_model(self, energy_predictor, sample_training_data):
        """Test that training saves the model."""
        energy_predictor.train(sample_training_data, target_column="energy_kwh")
        assert energy_predictor.model_path.exists()
        assert energy_predictor.scaler_path.exists()
    
    def test_train_cross_validation(self, energy_predictor, sample_training_data):
        """Test training with cross-validation."""
        metrics = energy_predictor.train(
            sample_training_data,
            target_column="energy_kwh",
            validate=True
        )
        assert "cv_rmse_mean" in metrics
    
    def test_get_feature_importance(self, energy_predictor, sample_training_data):
        """Test getting feature importance after training."""
        energy_predictor.train(sample_training_data, target_column="energy_kwh")
        importance = energy_predictor.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0
    
    def test_save_and_load_model(self, energy_predictor, sample_training_data, temp_model_dir):
        """Test model saving and loading."""
        # Train and save
        energy_predictor.train(sample_training_data, target_column="energy_kwh")
        
        # Create new predictor and load
        from prediction.estimator import EnergyPredictor
        new_predictor = EnergyPredictor(model_type="random_forest")
        new_predictor.model_path = energy_predictor.model_path
        new_predictor.scaler_path = energy_predictor.scaler_path
        loaded = new_predictor._load_model()
        
        assert loaded
        assert new_predictor.is_trained


class TestEnergyPredictorEdgeCases:
    """Test edge cases for EnergyPredictor."""
    
    def test_empty_prompt(self, energy_predictor, sample_prompts):
        """Test prediction for empty prompt."""
        result = energy_predictor.predict(sample_prompts["empty"])
        assert result.energy_kwh >= 0
    
    def test_very_long_prompt(self, energy_predictor, sample_prompts):
        """Test prediction for very long prompt."""
        result = energy_predictor.predict(sample_prompts["long"])
        assert result.energy_kwh > 0
        # Long prompts should have higher energy
    
    def test_unicode_prompt(self, energy_predictor, sample_prompts):
        """Test prediction for unicode prompt."""
        result = energy_predictor.predict(sample_prompts["unicode"])
        assert result.energy_kwh > 0
    
    def test_extreme_model_params(self, energy_predictor, sample_prompts):
        """Test prediction with extreme model parameters."""
        result = energy_predictor.predict(
            sample_prompts["simple"],
            num_layers=1000,
            training_hours=1000.0,
            flops_per_hour=1e15
        )
        assert result.energy_kwh > 0


class TestEnergyPredictorConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_predictor_function(self):
        """Test create_predictor() function."""
        from prediction.estimator import create_predictor
        predictor = create_predictor(model_type="random_forest")
        assert predictor is not None
    
    def test_predict_energy_function(self, sample_prompts):
        """Test predict_energy() convenience function."""
        from prediction.estimator import predict_energy
        result = predict_energy(sample_prompts["medium"])
        assert isinstance(result, dict)
        assert "energy_kwh" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
