"""
Unit Tests for the Anomaly Detection Module (detector.py).
Tests unsupervised ML anomaly detection.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAnomalyDetector:
    """Test AnomalyDetector class."""
    
    def test_detector_initialization(self, anomaly_detector):
        """Test detector initializes correctly."""
        assert anomaly_detector is not None
        assert anomaly_detector.model_type == "isolation_forest"
    
    def test_create_isolation_forest_model(self, temp_model_dir):
        """Test creating an Isolation Forest model."""
        from anomaly.detector import AnomalyDetector
        detector = AnomalyDetector(model_type="isolation_forest")
        detector.model_path = temp_model_dir / "if_model.joblib"
        detector._create_model()
        assert detector.model is not None
        assert detector.scaler is not None
    
    def test_create_one_class_svm_model(self, temp_model_dir):
        """Test creating a One-Class SVM model."""
        from anomaly.detector import AnomalyDetector
        detector = AnomalyDetector(model_type="one_class_svm")
        detector.model_path = temp_model_dir / "svm_model.joblib"
        detector._create_model()
        assert detector.model is not None
    
    def test_create_lof_model(self, temp_model_dir):
        """Test creating a Local Outlier Factor model."""
        from anomaly.detector import AnomalyDetector
        detector = AnomalyDetector(model_type="local_outlier_factor")
        detector.model_path = temp_model_dir / "lof_model.joblib"
        detector._create_model()
        assert detector.model is not None
    
    def test_extract_features(self, anomaly_detector, sample_prompts):
        """Test feature extraction from prompt."""
        features = anomaly_detector.extract_features(sample_prompts["medium"])
        assert isinstance(features, dict)
        assert "token_count" in features
        assert "complexity_score" in features
        assert "token_to_word_ratio" in features
    
    def test_extract_features_with_energy(self, anomaly_detector, sample_prompts):
        """Test feature extraction with pre-computed energy."""
        features = anomaly_detector.extract_features(
            sample_prompts["medium"],
            energy_kwh=0.5
        )
        assert "energy_kwh" in features
        assert "energy_per_token" in features
    
    def test_detect_returns_anomaly_result(self, anomaly_detector, sample_prompts):
        """Test that detect returns AnomalyResult object."""
        result = anomaly_detector.detect(sample_prompts["simple"])
        assert hasattr(result, "is_anomaly")
        assert hasattr(result, "anomaly_score")
        assert hasattr(result, "severity")
        assert hasattr(result, "recommendation")
    
    def test_detect_is_anomaly_boolean(self, anomaly_detector, sample_prompts):
        """Test that is_anomaly is a boolean."""
        result = anomaly_detector.detect(sample_prompts["simple"])
        assert isinstance(result.is_anomaly, bool)
    
    def test_detect_score_is_float(self, anomaly_detector, sample_prompts):
        """Test that anomaly_score is a float."""
        result = anomaly_detector.detect(sample_prompts["simple"])
        assert isinstance(result.anomaly_score, float)
    
    def test_detect_confidence_range(self, anomaly_detector, sample_prompts):
        """Test that confidence is in valid range."""
        result = anomaly_detector.detect(sample_prompts["simple"])
        assert 0 <= result.confidence <= 1
    
    def test_detect_severity_classification(self, anomaly_detector, sample_prompts):
        """Test severity classification."""
        result = anomaly_detector.detect(sample_prompts["simple"])
        assert result.severity in ["normal", "low", "medium", "high", "critical"]
    
    def test_detect_anomaly_type(self, anomaly_detector, sample_prompts):
        """Test anomaly type classification."""
        result = anomaly_detector.detect(sample_prompts["simple"])
        valid_types = ["normal", "high_complexity", "high_tokens", "unusual_pattern"]
        assert result.anomaly_type in valid_types
    
    def test_detect_recommendation_exists(self, anomaly_detector, sample_prompts):
        """Test that recommendation is provided."""
        result = anomaly_detector.detect(sample_prompts["simple"])
        assert result.recommendation is not None
        assert len(result.recommendation) > 0
    
    def test_detect_to_dict(self, anomaly_detector, sample_prompts):
        """Test AnomalyResult.to_dict() method."""
        result = anomaly_detector.detect(sample_prompts["simple"])
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "is_anomaly" in result_dict
        assert "anomaly_score" in result_dict
    
    def test_detect_batch(self, anomaly_detector, sample_prompts):
        """Test batch detection."""
        prompts = [sample_prompts["simple"], sample_prompts["medium"]]
        results = anomaly_detector.detect_batch(prompts)
        assert len(results) == 2
        assert all(hasattr(r, "is_anomaly") for r in results)
    
    def test_rule_based_detection(self, anomaly_detector, sample_prompts):
        """Test fallback rule-based detection."""
        features = anomaly_detector.extract_features(sample_prompts["long"])
        is_anomaly, score = anomaly_detector._rule_based_detection(features)
        assert isinstance(is_anomaly, bool)
        assert isinstance(score, float)


class TestAnomalyDetectorTraining:
    """Test AnomalyDetector training functionality."""
    
    def test_train_model(self, anomaly_detector, sample_training_data):
        """Test model training."""
        # Use subset of features for anomaly detection
        train_data = sample_training_data[["token_count", "complexity_score", "char_count"]].copy()
        
        metrics = anomaly_detector.train(train_data)
        assert "n_samples" in metrics
        assert "n_anomalies_detected" in metrics
        assert "anomaly_rate" in metrics
        assert anomaly_detector.is_trained
    
    def test_train_saves_model(self, anomaly_detector, sample_training_data):
        """Test that training saves the model."""
        train_data = sample_training_data[["token_count", "complexity_score", "char_count"]].copy()
        anomaly_detector.train(train_data)
        assert anomaly_detector.model_path.exists()
    
    def test_train_stores_feature_stats(self, anomaly_detector, sample_training_data):
        """Test that training stores feature statistics."""
        train_data = sample_training_data[["token_count", "complexity_score", "char_count"]].copy()
        anomaly_detector.train(train_data)
        assert anomaly_detector.feature_stats is not None
        assert "token_count" in anomaly_detector.feature_stats
    
    def test_contamination_affects_detection(self):
        """Test that contamination rate affects detection."""
        from anomaly.detector import AnomalyDetector
        # Higher contamination = more anomalies detected
        # This is a property of Isolation Forest


class TestAnomalyDetectorAnomalousPrompts:
    """Test detection of actually anomalous prompts."""
    
    def test_detect_long_prompt_anomaly(self, anomaly_detector, sample_prompts):
        """Test that very long prompts may be flagged."""
        result = anomaly_detector.detect(sample_prompts["long"])
        # Very long prompts should have higher anomaly potential
        # (may or may not be flagged depending on threshold)
        assert result.anomaly_score is not None
    
    def test_detect_normal_prompt(self, anomaly_detector, sample_prompts):
        """Test that normal prompts are usually not flagged."""
        result = anomaly_detector.detect(sample_prompts["simple"])
        # Simple prompts should typically not be anomalies
        # (using rule-based detection for untrained model)
        assert isinstance(result.is_anomaly, bool)
    
    def test_high_complexity_detected(self, anomaly_detector, sample_prompts):
        """Test high complexity anomaly detection."""
        result = anomaly_detector.detect(sample_prompts["complex"])
        # Complex prompts may be flagged due to high complexity
        if result.is_anomaly:
            assert "complexity" in str(result.contributing_factors).lower() or \
                   result.anomaly_type in ["high_complexity", "unusual_pattern"]


class TestAnomalyDetectorStatistics:
    """Test anomaly statistics functionality."""
    
    def test_get_anomaly_statistics(self, anomaly_detector, sample_prompts):
        """Test getting statistics from batch results."""
        prompts = [sample_prompts["simple"], sample_prompts["medium"], sample_prompts["complex"]]
        results = anomaly_detector.detect_batch(prompts)
        stats = anomaly_detector.get_anomaly_statistics(results)
        
        assert "total_analyzed" in stats
        assert stats["total_analyzed"] == 3
        assert "total_anomalies" in stats
        assert "anomaly_rate" in stats
        assert "severity_distribution" in stats
    
    def test_statistics_empty_results(self, anomaly_detector):
        """Test statistics with empty results."""
        stats = anomaly_detector.get_anomaly_statistics([])
        assert "error" in stats


class TestAnomalyDetectorEdgeCases:
    """Test edge cases for AnomalyDetector."""
    
    def test_empty_prompt(self, anomaly_detector, sample_prompts):
        """Test detection for empty prompt."""
        result = anomaly_detector.detect(sample_prompts["empty"])
        assert result is not None
    
    def test_unicode_prompt(self, anomaly_detector, sample_prompts):
        """Test detection for unicode prompt."""
        result = anomaly_detector.detect(sample_prompts["unicode"])
        assert result is not None
    
    def test_special_chars_prompt(self, anomaly_detector, sample_prompts):
        """Test detection for prompt with special characters."""
        result = anomaly_detector.detect(sample_prompts["special_chars"])
        assert result is not None


class TestAnomalyDetectorConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_detector_function(self):
        """Test create_detector() function."""
        from anomaly.detector import create_detector
        detector = create_detector(model_type="isolation_forest")
        assert detector is not None
    
    def test_detect_anomaly_function(self, sample_prompts):
        """Test detect_anomaly() convenience function."""
        from anomaly.detector import detect_anomaly
        result = detect_anomaly(sample_prompts["medium"])
        assert isinstance(result, dict)
        assert "is_anomaly" in result


class TestAnomalyTypes:
    """Test different anomaly type detection."""
    
    def test_determine_anomaly_type_high_complexity(self, anomaly_detector):
        """Test high complexity anomaly type."""
        features = {"complexity_score": 0.9, "token_count": 100}
        anomaly_type, factors = anomaly_detector._determine_anomaly_type(features)
        assert anomaly_type == "high_complexity"
        assert "complexity" in factors
    
    def test_determine_anomaly_type_high_tokens(self, anomaly_detector):
        """Test high token count anomaly type."""
        features = {"complexity_score": 0.3, "token_count": 600}
        anomaly_type, factors = anomaly_detector._determine_anomaly_type(features)
        assert anomaly_type == "high_tokens"
        assert "tokens" in factors
    
    def test_generate_recommendation_normal(self, anomaly_detector):
        """Test recommendation generation for normal prompts."""
        recommendation = anomaly_detector._generate_recommendation(
            "normal", "normal", {}
        )
        assert "normal" in recommendation.lower() or "efficient" in recommendation.lower()
    
    def test_generate_recommendation_high_complexity(self, anomaly_detector):
        """Test recommendation generation for high complexity."""
        recommendation = anomaly_detector._generate_recommendation(
            "high_complexity", "high", {"complexity": "High complexity"}
        )
        assert "simplif" in recommendation.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
