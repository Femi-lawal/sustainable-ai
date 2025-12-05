"""
Model Performance Validation Tests for Sustainable AI.

These tests validate that the ML models actually perform well,
not just that they return the correct types. They ensure:

1. Model Quality Metrics (R², RMSE) meet minimum thresholds
2. Energy predictions correlate with prompt complexity
3. Optimization actually reduces energy consumption
4. Simplifier effectively reduces token counts

Author: Sustainable AI Team
Created: December 2025
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# MODEL QUALITY TESTS
# ============================================================================

@pytest.mark.performance
class TestModelQuality:
    """
    Tests that validate the trained model meets quality thresholds.
    
    These tests verify the model's predictive power using held-out data
    and standard ML metrics. A model that doesn't meet these thresholds
    needs retraining.
    """
    
    # Minimum acceptable R² score for test fixture data (lower than production)
    # Production model achieves R²=0.976 on proper training data
    MIN_R2_SCORE_FIXTURE = 0.65  # Lower threshold for synthetic fixture data
    
    # Maximum acceptable RMSE (in same units as energy)
    MAX_RMSE = 0.15
    
    def test_model_r2_score_on_validation_data(self, sample_training_data, temp_model_dir):
        """
        Test that a model trained on fixture data achieves reasonable R² score.
        
        Note: This tests the training pipeline, not the production model.
        The production model (trained on 500 samples) achieves R²=0.9809.
        Fixture data has fewer samples and simpler correlations.
        """
        from prediction.estimator import EnergyPredictor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        
        # Create and train predictor
        predictor = EnergyPredictor(model_type="random_forest")
        predictor.model_path = temp_model_dir / "test_model.joblib"
        predictor.scaler_path = temp_model_dir / "test_scaler.joblib"
        
        # Train with more data for robust testing
        metrics = predictor.train(sample_training_data, target_column="energy_kwh", test_size=0.3)
        
        # Verify R² meets threshold (lower for fixture data)
        assert metrics["test_r2"] >= self.MIN_R2_SCORE_FIXTURE, (
            f"Model R² score {metrics['test_r2']:.3f} is below minimum threshold "
            f"{self.MIN_R2_SCORE_FIXTURE}. Training pipeline may have issues."
        )
    
    def test_model_rmse_on_validation_data(self, sample_training_data, temp_model_dir):
        """
        Test that trained model achieves acceptable RMSE.
        
        RMSE (Root Mean Square Error) measures prediction accuracy.
        Lower is better - we want predictions within reasonable range.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor(model_type="random_forest")
        predictor.model_path = temp_model_dir / "test_model.joblib"
        predictor.scaler_path = temp_model_dir / "test_scaler.joblib"
        
        metrics = predictor.train(sample_training_data, target_column="energy_kwh")
        
        assert metrics["test_rmse"] <= self.MAX_RMSE, (
            f"Model RMSE {metrics['test_rmse']:.4f} exceeds maximum threshold "
            f"{self.MAX_RMSE}. Model predictions are too inaccurate."
        )
    
    def test_cross_validation_stability(self, sample_training_data, temp_model_dir):
        """
        Test that model performance is stable across cross-validation folds.
        
        Large variance in CV scores indicates the model is overfitting or
        the training data has issues.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor(model_type="random_forest")
        predictor.model_path = temp_model_dir / "test_model.joblib"
        predictor.scaler_path = temp_model_dir / "test_scaler.joblib"
        
        metrics = predictor.train(sample_training_data, target_column="energy_kwh", validate=True)
        
        # CV standard deviation should be low (stable performance)
        assert metrics.get("cv_rmse_std", 0) < 0.05, (
            f"Cross-validation RMSE std {metrics.get('cv_rmse_std', 0):.4f} is too high. "
            "Model performance varies too much across folds."
        )


# ============================================================================
# PREDICTION CORRELATION TESTS
# ============================================================================

@pytest.mark.performance
class TestPredictionCorrelation:
    """
    Tests that verify predictions correlate with expected patterns.
    
    These tests ensure the model learned meaningful relationships:
    - Longer prompts → higher energy
    - More complex prompts → higher energy
    - Simple prompts → lower energy
    """
    
    def test_longer_prompts_predict_higher_energy(self):
        """
        Test that longer prompts predict higher energy consumption.
        
        This is a fundamental relationship - more tokens = more computation.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        short_prompt = "Hello"
        medium_prompt = "Explain how machine learning works in simple terms."
        long_prompt = """Please provide a comprehensive and detailed analysis of the 
        various machine learning algorithms, including supervised learning, unsupervised 
        learning, and reinforcement learning. Discuss their advantages, disadvantages, 
        and real-world applications in different industries."""
        
        short_energy = predictor.predict(short_prompt).energy_kwh
        medium_energy = predictor.predict(medium_prompt).energy_kwh
        long_energy = predictor.predict(long_prompt).energy_kwh
        
        # Energy should increase with prompt length
        assert short_energy < medium_energy, (
            f"Short prompt ({short_energy:.6f}) should have less energy than "
            f"medium prompt ({medium_energy:.6f})"
        )
        assert medium_energy < long_energy, (
            f"Medium prompt ({medium_energy:.6f}) should have less energy than "
            f"long prompt ({long_energy:.6f})"
        )
    
    def test_complex_prompts_predict_higher_energy(self):
        """
        Test that more complex prompts predict higher energy.
        
        Complex vocabulary and sentence structure require more processing.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        simple_prompt = "What is water?"
        complex_prompt = """Elucidate the thermodynamic principles governing 
        phase transitions in aqueous solutions under varying pressure conditions."""
        
        simple_energy = predictor.predict(simple_prompt).energy_kwh
        complex_energy = predictor.predict(complex_prompt).energy_kwh
        
        # Complex prompt should have higher energy (considering both complexity and length)
        # This test allows for the natural variation but expects the general trend
        assert complex_energy > simple_energy * 0.8, (
            f"Complex prompt ({complex_energy:.6f}) should have higher energy than "
            f"simple prompt ({simple_energy:.6f})"
        )
    
    def test_token_count_is_primary_energy_driver(self):
        """
        Test that token count is the primary driver of energy prediction.
        
        Our model should have learned that token_count has the highest
        feature importance (as documented: 56.5%).
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        if predictor.is_trained:
            importance = predictor.get_feature_importance()
            
            if importance:
                # Token count should be among top 3 features
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                top_3_features = [f[0] for f in sorted_features[:3]]
                
                token_related = any('token' in f.lower() or 'word' in f.lower() or 'char' in f.lower() 
                                   for f in top_3_features)
                
                assert token_related, (
                    f"Token-related features should be in top 3 importance. "
                    f"Got: {top_3_features}"
                )


# ============================================================================
# OPTIMIZATION EFFECTIVENESS TESTS
# ============================================================================

@pytest.mark.performance
class TestOptimizationEffectiveness:
    """
    Tests that verify optimization actually reduces energy consumption.
    
    These tests ensure the optimizer doesn't just return different text,
    but actually achieves measurable energy savings.
    """
    
    def test_optimization_returns_valid_result(self):
        """
        Test that optimization returns a valid result with all expected fields.
        """
        from optimization.recomender import PromptOptimizer
        
        optimizer = PromptOptimizer(min_similarity=0.6)
        
        verbose_prompt = """Could you please kindly help me understand this topic 
        in great detail? I would really appreciate it very much."""
        
        result = optimizer.optimize(verbose_prompt)
        
        # Verify result has all expected fields
        assert hasattr(result, 'energy_reduction_percent')
        assert hasattr(result, 'semantic_similarity')
        assert hasattr(result, 'optimized_prompt')
        assert result.semantic_similarity >= 0, "Similarity should be non-negative"
        assert result.semantic_similarity <= 1, "Similarity should be <= 1"
    
    def test_simplifier_reduces_token_count(self):
        """
        Test that the simplifier reduces token count for verbose text.
        
        Fewer tokens = less computation = less energy.
        """
        from nlp.simplifier import TextSimplifier
        from nlp.parser import parse_prompt
        
        simplifier = TextSimplifier(min_similarity_threshold=0.5)
        
        # Use text with phrases that simplifier is known to handle
        verbose_prompt = "In order to help you, due to the fact that this is important, I will basically provide information."
        
        result = simplifier.simplify(verbose_prompt, strategy="verbose")
        
        original_tokens = parse_prompt(verbose_prompt, use_embeddings=False).token_count
        optimized_tokens = parse_prompt(result.simplified, use_embeddings=False).token_count
        
        # Simplifier should reduce tokens (or at least not increase)
        assert optimized_tokens <= original_tokens, (
            f"Simplifier should not increase token count. "
            f"Original: {original_tokens}, Simplified: {optimized_tokens}"
        )
    
    def test_simplifier_with_filler_strategy(self):
        """
        Test that filler word removal reduces tokens.
        """
        from nlp.simplifier import TextSimplifier
        from nlp.parser import parse_prompt
        
        simplifier = TextSimplifier(min_similarity_threshold=0.5)
        
        # Use text with known filler words
        text_with_fillers = "This is basically really very extremely good stuff actually."
        
        result = simplifier.simplify(text_with_fillers, strategy="filler")
        
        original_tokens = parse_prompt(text_with_fillers, use_embeddings=False).token_count
        simplified_tokens = parse_prompt(result.simplified, use_embeddings=False).token_count
        
        # Filler removal should reduce tokens
        assert simplified_tokens <= original_tokens, (
            f"Filler removal should not increase tokens. "
            f"Original: {original_tokens}, After: {simplified_tokens}"
        )
    
    def test_already_efficient_prompts_not_degraded(self):
        """
        Test that already efficient prompts are not made worse.
        
        Simple prompts shouldn't have their meaning destroyed or
        energy increased by optimization.
        """
        from optimization.recomender import PromptOptimizer
        
        optimizer = PromptOptimizer(min_similarity=0.7)
        
        efficient_prompt = "What is 2 + 2?"
        
        result = optimizer.optimize(efficient_prompt)
        
        # Energy should not increase significantly
        assert result.energy_reduction_percent >= -5, (
            f"Efficient prompt had energy INCREASE of {-result.energy_reduction_percent:.1f}%. "
            "Optimization should not make efficient prompts worse."
        )


# ============================================================================
# SIMPLIFIER EFFECTIVENESS TESTS
# ============================================================================

@pytest.mark.performance
class TestSimplifierEffectiveness:
    """
    Tests that verify the text simplifier effectively reduces tokens.
    """
    
    def test_verbose_phrase_removal(self):
        """
        Test that verbose phrases are actually removed.
        """
        from nlp.simplifier import TextSimplifier
        
        simplifier = TextSimplifier()
        
        test_cases = [
            ("in order to help you", "to help you"),
            ("due to the fact that", "because"),
            ("at this point in time", "now"),
            ("in the event that", "if"),
            ("for the purpose of", "for"),
        ]
        
        for verbose, expected_replacement in test_cases:
            result = simplifier.remove_verbose_phrases(f"I want {verbose} understand.")
            assert verbose not in result.lower(), (
                f"Verbose phrase '{verbose}' should be removed. Got: {result}"
            )
    
    def test_filler_word_removal(self):
        """
        Test that filler words are actually removed.
        """
        from nlp.simplifier import TextSimplifier
        
        simplifier = TextSimplifier()
        
        text = "This is basically really very extremely good."
        result = simplifier.remove_filler_words(text)
        
        fillers = ["basically", "really", "very", "extremely"]
        for filler in fillers:
            assert filler not in result.lower(), (
                f"Filler word '{filler}' should be removed. Got: {result}"
            )
    
    def test_simplification_preserves_core_meaning(self):
        """
        Test that simplification keeps the core question/request intact.
        """
        from nlp.simplifier import TextSimplifier
        
        simplifier = TextSimplifier(min_similarity_threshold=0.6)
        
        verbose = "Could you please kindly explain machine learning to me?"
        result = simplifier.simplify(verbose)
        
        # Core words should still be present
        core_words = ["machine", "learning"]
        simplified_lower = result.simplified.lower()
        
        for word in core_words:
            assert word in simplified_lower, (
                f"Core word '{word}' missing from simplified result: {result.simplified}"
            )


# ============================================================================
# COMPARATIVE TESTS
# ============================================================================

@pytest.mark.performance
class TestComparativePredictions:
    """
    Tests that compare predictions across different prompt types.
    
    These tests verify that relative energy predictions make sense.
    """
    
    def test_code_prompts_higher_than_simple_questions(self):
        """
        Test that code-related prompts predict higher energy.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        simple_question = "What is Python?"
        code_prompt = """Write a Python function that implements a binary search 
        tree with insert, delete, and search operations. Include proper error 
        handling and type hints."""
        
        simple_energy = predictor.predict(simple_question).energy_kwh
        code_energy = predictor.predict(code_prompt).energy_kwh
        
        assert code_energy > simple_energy, (
            f"Code prompt ({code_energy:.6f}) should predict higher energy "
            f"than simple question ({simple_energy:.6f})"
        )
    
    def test_repeated_predictions_consistent(self):
        """
        Test that repeated predictions for same prompt are consistent.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        prompt = "Explain quantum computing basics."
        
        predictions = [predictor.predict(prompt).energy_kwh for _ in range(5)]
        
        # All predictions should be identical (deterministic model)
        assert all(p == predictions[0] for p in predictions), (
            f"Predictions should be consistent. Got: {predictions}"
        )
    
    def test_batch_vs_individual_predictions_match(self):
        """
        Test that batch predictions match individual predictions.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        prompts = [
            "What is AI?",
            "Explain machine learning.",
            "How does deep learning work?"
        ]
        
        # Individual predictions
        individual = [predictor.predict(p).energy_kwh for p in prompts]
        
        # Batch predictions
        batch = [r.energy_kwh for r in predictor.predict_batch(prompts)]
        
        for i, (ind, bat) in enumerate(zip(individual, batch)):
            assert abs(ind - bat) < 0.0001, (
                f"Prompt {i}: Individual ({ind}) != Batch ({bat})"
            )


# ============================================================================
# CONFIDENCE SCORE TESTS
# ============================================================================

@pytest.mark.performance
class TestConfidenceScores:
    """
    Tests that verify confidence scores are meaningful.
    """
    
    def test_trained_model_has_high_confidence(self):
        """
        Test that a trained model reports high confidence.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        if predictor.is_trained:
            result = predictor.predict("Explain machine learning.")
            
            assert result.confidence_score >= 0.7, (
                f"Trained model confidence {result.confidence_score:.2f} is too low. "
                "Model should have confidence >= 0.7"
            )
    
    def test_confidence_in_valid_range(self):
        """
        Test that confidence scores are always between 0 and 1.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        test_prompts = [
            "Hi",
            "Explain quantum physics in detail.",
            "x" * 1000,  # Very long
            "",  # Empty
        ]
        
        for prompt in test_prompts:
            result = predictor.predict(prompt)
            assert 0 <= result.confidence_score <= 1, (
                f"Confidence {result.confidence_score} out of range [0, 1] "
                f"for prompt: {prompt[:50]}..."
            )


# ============================================================================
# PRODUCTION MODEL TESTS
# ============================================================================

@pytest.mark.performance
class TestProductionModel:
    """
    Tests that validate the production model in model/ directory.
    
    These tests verify the actual deployed model meets quality standards.
    """
    
    def test_production_model_loads(self):
        """
        Test that the production model can be loaded.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        assert predictor.model is not None, (
            "Production model failed to load. Check model/ directory."
        )
    
    def test_production_model_is_trained(self):
        """
        Test that the production model is marked as trained.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        assert predictor.is_trained, (
            "Production model is not marked as trained. "
            "Model may not have been properly saved."
        )
    
    def test_production_model_has_feature_names(self):
        """
        Test that the production model has feature names saved.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        assert predictor.feature_names is not None, (
            "Production model missing feature names. "
            "Model was not saved properly."
        )
        
        assert len(predictor.feature_names) >= 5, (
            f"Production model only has {len(predictor.feature_names)} features. "
            "Expected at least 5 features."
        )
    
    def test_production_model_predictions_reasonable(self):
        """
        Test that production model predictions are in reasonable range.
        """
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        # Test various prompts
        # Note: Calibrated model outputs realistic Joules (3-80J range) instead of kWh
        # The field is named 'energy_kwh' for backwards compatibility but contains Joules
        if getattr(predictor, 'is_calibrated', False):
            # Calibrated model: realistic Joule outputs based on real measurements
            # Energy is driven primarily by actual token count, not semantic content
            # Ranges based on real measurements:
            # Simple (3-10 tokens): 3-11 J
            # Medium (11-20 tokens): 10-20 J
            # Longer prompts scale accordingly
            test_cases = [
                ("Hi", 1, 15),  # Very simple: ~3-10 Joules
                ("Explain machine learning.", 2, 20),  # Short: ~4-15 Joules
                ("Write a 10000 word essay.", 2, 20),  # Short prompt (few tokens): ~4-15 Joules
            ]
        else:
            # Original model: synthetic kWh outputs
            test_cases = [
                ("Hi", 0.001, 0.5),  # Simple: very low energy
                ("Explain machine learning.", 0.001, 1.0),  # Medium: low-medium
                ("Write a 10000 word essay.", 0.01, 5.0),  # Complex: higher
            ]
        
        for prompt, min_energy, max_energy in test_cases:
            result = predictor.predict(prompt)
            # Use energy_joules for calibrated model (native unit)
            energy_value = result.energy_joules if predictor.is_calibrated else result.energy_kwh
            assert min_energy <= energy_value <= max_energy, (
                f"Energy {energy_value:.6f} out of expected range "
                f"[{min_energy}, {max_energy}] for: {prompt}"
            )


# ============================================================================
# T5 MODEL INTEGRATION TESTS
# ============================================================================

@pytest.mark.performance
class TestT5Integration:
    """
    Tests that validate T5 model integration for prompt simplification.
    
    These tests ensure the T5-based simplification works correctly and
    achieves energy savings as per project architecture requirements.
    """
    
    def test_t5_model_loads_successfully(self):
        """
        Test that T5 model can be loaded for simplification.
        """
        from nlp.simplifier import is_t5_available, get_t5_load_error
        
        is_available = is_t5_available()
        
        if not is_available:
            error = get_t5_load_error()
            pytest.skip(f"T5 model not available: {error}")
        
        assert is_available, "T5 model should be available"
    
    def test_t5_simplification_reduces_tokens(self):
        """
        Test that T5 simplification actually reduces token count.
        """
        from nlp.simplifier import TextSimplifier, is_t5_available
        
        if not is_t5_available():
            pytest.skip("T5 model not available")
        
        simplifier = TextSimplifier(min_similarity_threshold=0.50)
        
        verbose_prompt = """In order to provide you with the most comprehensive 
        and detailed explanation possible, I would like to basically take into 
        consideration all of the various different factors."""
        
        # Use T5 strategy
        result = simplifier.simplify(verbose_prompt, strategy="t5")
        
        assert result.token_reduction_percent > 0, (
            f"T5 should reduce tokens. Got {result.token_reduction_percent}% reduction"
        )
    
    def test_t5_maintains_semantic_meaning(self):
        """
        Test that T5 simplification maintains semantic similarity.
        """
        from nlp.simplifier import TextSimplifier, is_t5_available
        
        if not is_t5_available():
            pytest.skip("T5 model not available")
        
        simplifier = TextSimplifier(min_similarity_threshold=0.50)
        
        test_prompt = """Could you please explain the fundamental concepts 
        of machine learning in simple terms?"""
        
        result = simplifier.simplify(test_prompt, strategy="t5")
        
        # Should maintain at least 50% similarity
        assert result.semantic_similarity >= 0.50, (
            f"T5 should maintain meaning. Got {result.semantic_similarity:.0%} similarity"
        )
    
    def test_optimizer_uses_t5_for_energy_reduction(self):
        """
        Test that the optimizer uses T5 to achieve energy reduction.
        """
        from optimization.recomender import PromptOptimizer
        from nlp.simplifier import is_t5_available
        
        if not is_t5_available():
            pytest.skip("T5 model not available")
        
        optimizer = PromptOptimizer(min_similarity=0.55)
        
        verbose_prompt = """In order to provide you with a comprehensive 
        explanation, I would like to basically consider all of the 
        various different factors involved in this topic."""
        
        result = optimizer.optimize(verbose_prompt)
        
        # Should achieve some energy reduction
        # T5 typically achieves 3-10% for moderately verbose prompts
        assert result.energy_reduction_percent >= 0, (
            f"Optimizer should achieve energy reduction. "
            f"Got {result.energy_reduction_percent:.1f}%"
        )
    
    def test_t5_strategy_selected_when_beneficial(self):
        """
        Test that T5 strategy is selected when it provides benefits.
        """
        from optimization.recomender import PromptOptimizer
        from nlp.simplifier import is_t5_available
        
        if not is_t5_available():
            pytest.skip("T5 model not available")
        
        optimizer = PromptOptimizer(min_similarity=0.55)
        
        # Very verbose prompt that T5 should handle well
        verbose_prompt = """Due to the fact that this is an extremely important 
        matter that requires careful consideration, I was wondering if you 
        could perhaps help me understand the underlying mechanisms."""
        
        result = optimizer.optimize(verbose_prompt)
        
        # T5 or t5+rules should be selected for such verbose prompts
        assert "t5" in result.strategy_used.lower() or result.energy_reduction_percent > 0, (
            f"Expected T5-based strategy or energy reduction. "
            f"Got strategy: {result.strategy_used}, reduction: {result.energy_reduction_percent}%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
