"""
Integration Tests for Sustainable AI Application.
Tests interactions between multiple modules.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.integration
class TestFullAnalysisPipeline:
    """Test the full analysis pipeline."""
    
    def test_analyze_prompt_full_pipeline(self, sample_prompts):
        """Test complete prompt analysis pipeline."""
        from nlp.parser import parse_prompt
        from nlp.complexity_score import compute_complexity
        from prediction.estimator import EnergyPredictor
        from anomaly.detector import AnomalyDetector
        
        prompt = sample_prompts["medium"]
        
        # Step 1: Parse prompt
        parsed = parse_prompt(prompt, use_embeddings=False)
        assert parsed.token_count > 0
        
        # Step 2: Calculate complexity
        complexity = compute_complexity(prompt)
        assert 0 <= complexity <= 1
        
        # Step 3: Predict energy
        predictor = EnergyPredictor()
        energy_result = predictor.predict(prompt)
        assert energy_result.energy_kwh > 0
        
        # Step 4: Detect anomalies
        detector = AnomalyDetector()
        anomaly_result = detector.detect(prompt, energy_kwh=energy_result.energy_kwh)
        assert anomaly_result.anomaly_score is not None
    
    def test_analyze_and_optimize_pipeline(self, sample_prompts):
        """Test analysis and optimization pipeline."""
        from nlp.parser import parse_prompt
        from nlp.complexity_score import compute_complexity
        from optimization.recomender import PromptOptimizer
        
        prompt = sample_prompts["verbose"]
        
        # Step 1: Analyze
        original_parsed = parse_prompt(prompt, use_embeddings=False)
        original_complexity = compute_complexity(prompt)
        
        # Step 2: Optimize
        optimizer = PromptOptimizer(min_similarity=0.6)
        result = optimizer.optimize(prompt)
        
        # Step 3: Analyze optimized
        if result.optimized_prompt != prompt:
            optimized_parsed = parse_prompt(result.optimized_prompt, use_embeddings=False)
            optimized_complexity = compute_complexity(result.optimized_prompt)
            
            # Optimized should generally be simpler or shorter
            assert result.semantic_similarity >= 0.6


@pytest.mark.integration
class TestDatabaseIntegration:
    """Test database integration with other modules."""
    
    def test_log_analysis_results(self, database_manager, sample_prompts):
        """Test logging analysis results to database."""
        from nlp.parser import parse_prompt
        from nlp.complexity_score import compute_complexity
        from prediction.estimator import EnergyPredictor
        from anomaly.detector import AnomalyDetector
        
        prompt = sample_prompts["medium"]
        
        # Analyze
        parsed = parse_prompt(prompt, use_embeddings=False)
        complexity = compute_complexity(prompt)
        predictor = EnergyPredictor()
        energy_result = predictor.predict(prompt)
        detector = AnomalyDetector()
        anomaly_result = detector.detect(prompt, energy_kwh=energy_result.energy_kwh)
        
        # Log to database
        log_id = database_manager.log_energy(
            prompt=prompt,
            token_count=parsed.token_count,
            complexity_score=complexity,
            energy_kwh=energy_result.energy_kwh,
            carbon_kg=energy_result.carbon_footprint_kg,
            water_liters=energy_result.water_usage_liters,
            is_anomaly=anomaly_result.is_anomaly,
            model_params={"num_layers": 24}
        )
        
        assert log_id > 0
        
        # Verify log exists
        logs = database_manager.get_logs(limit=1)
        assert len(logs) >= 1
        assert logs[0]["token_count"] == parsed.token_count
    
    def test_log_optimization_results(self, database_manager, sample_prompts):
        """Test logging optimization results."""
        from nlp.parser import parse_prompt
        from nlp.complexity_score import compute_complexity
        from optimization.recomender import PromptOptimizer
        
        prompt = sample_prompts["verbose"]
        
        # Optimize
        optimizer = PromptOptimizer(min_similarity=0.6)
        result = optimizer.optimize(prompt)
        
        # Log original
        original_parsed = parse_prompt(prompt, use_embeddings=False)
        database_manager.log_energy(
            prompt=prompt,
            token_count=original_parsed.token_count,
            complexity_score=compute_complexity(prompt),
            energy_kwh=result.original_energy_kwh
        )
        
        # Log optimized
        optimized_parsed = parse_prompt(result.optimized_prompt, use_embeddings=False)
        database_manager.log_energy(
            prompt=result.optimized_prompt,
            token_count=optimized_parsed.token_count,
            complexity_score=compute_complexity(result.optimized_prompt),
            energy_kwh=result.optimized_energy_kwh,
            was_optimized=True,
            energy_saved_kwh=result.energy_saved_kwh
        )
        
        # Verify both logged
        logs = database_manager.get_logs(limit=10)
        assert len(logs) >= 2


@pytest.mark.integration
class TestReportingIntegration:
    """Test reporting integration."""
    
    def test_generate_report_after_analysis(self, database_manager, sample_prompts, date_range):
        """Test generating report after analysis."""
        from nlp.parser import parse_prompt
        from nlp.complexity_score import compute_complexity
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        # Analyze multiple prompts
        for prompt_name in ["simple", "medium", "complex"]:
            prompt = sample_prompts[prompt_name]
            parsed = parse_prompt(prompt, use_embeddings=False)
            complexity = compute_complexity(prompt)
            energy = predictor.predict(prompt)
            
            database_manager.log_energy(
                prompt=prompt,
                token_count=parsed.token_count,
                complexity_score=complexity,
                energy_kwh=energy.energy_kwh,
                carbon_kg=energy.carbon_footprint_kg,
                water_liters=energy.water_usage_liters
            )
        
        # Generate report
        report = database_manager.generate_transparency_report(
            date_range["start"],
            date_range["end"]
        )
        
        assert report["summary"]["total_prompts"] >= 3
        assert report["summary"]["total_energy_kwh"] > 0


@pytest.mark.integration
class TestNLPPipelineIntegration:
    """Test NLP pipeline integration."""
    
    def test_parser_complexity_integration(self, sample_prompts):
        """Test parser and complexity scorer integration."""
        from nlp.parser import parse_prompt, PromptParser
        from nlp.complexity_score import ComplexityScorer
        
        prompt = sample_prompts["technical"]
        
        # Parse
        parser = PromptParser(use_embeddings=False)
        parsed = parser.parse(prompt)
        
        # Score complexity
        scorer = ComplexityScorer()
        complexity = scorer.calculate(prompt)
        
        # Vocabulary richness from parser should correlate with vocabulary complexity
        assert parsed.vocabulary_richness is not None
        assert complexity.vocabulary_complexity is not None
    
    def test_parser_simplifier_integration(self, sample_prompts):
        """Test parser and simplifier integration."""
        from nlp.parser import parse_prompt
        from nlp.simplifier import TextSimplifier
        
        prompt = sample_prompts["verbose"]
        
        # Parse original
        original_parsed = parse_prompt(prompt, use_embeddings=False)
        
        # Simplify
        simplifier = TextSimplifier()
        result = simplifier.simplify(prompt)
        
        # Parse simplified
        simplified_parsed = parse_prompt(result.simplified, use_embeddings=False)
        
        # Token count should generally decrease
        assert result.simplified != "" or original_parsed.token_count == 0


@pytest.mark.integration
class TestMLPipelineIntegration:
    """Test ML pipeline integration."""
    
    def test_predictor_detector_integration(self, sample_prompts):
        """Test energy predictor and anomaly detector integration."""
        from prediction.estimator import EnergyPredictor
        from anomaly.detector import AnomalyDetector
        
        predictor = EnergyPredictor()
        detector = AnomalyDetector()
        
        for prompt_name, prompt in sample_prompts.items():
            if prompt.strip():  # Skip empty prompts
                energy = predictor.predict(prompt)
                anomaly = detector.detect(prompt, energy_kwh=energy.energy_kwh)
                
                # Both should return valid results
                assert energy.energy_kwh > 0
                assert anomaly.anomaly_score is not None
    
    def test_training_and_prediction_integration(self, sample_training_data, temp_model_dir):
        """Test training followed by prediction."""
        from prediction.estimator import EnergyPredictor
        
        # Create and train predictor
        predictor = EnergyPredictor(model_type="random_forest")
        predictor.model_path = temp_model_dir / "trained_model.joblib"
        predictor.scaler_path = temp_model_dir / "trained_scaler.joblib"
        
        metrics = predictor.train(sample_training_data, target_column="energy_kwh")
        assert predictor.is_trained
        
        # Now predict
        test_prompt = "Explain machine learning concepts."
        result = predictor.predict(test_prompt)
        
        assert result.energy_kwh > 0
        assert result.confidence_score > 0


@pytest.mark.integration
class TestOptimizationIntegration:
    """Test optimization integration."""
    
    def test_optimizer_uses_all_components(self, sample_prompts):
        """Test that optimizer correctly uses all components."""
        from optimization.recomender import PromptOptimizer
        
        optimizer = PromptOptimizer(min_similarity=0.6)
        
        # These components should be initialized
        assert optimizer.parser is not None
        assert optimizer.simplifier is not None
        assert optimizer.energy_predictor is not None
        
        # Optimize and verify all components used
        result = optimizer.optimize(sample_prompts["verbose"])
        
        # Result should have all expected fields
        assert result.original_energy_kwh is not None  # From energy predictor
        assert result.semantic_similarity is not None  # From parser
        assert result.optimized_prompt is not None  # From simplifier
    
    def test_alternatives_correctly_scored(self, sample_prompts):
        """Test that alternatives are correctly scored."""
        from optimization.recomender import PromptOptimizer
        
        optimizer = PromptOptimizer(min_similarity=0.5)
        alternatives = optimizer.generate_alternatives(
            sample_prompts["verbose"],
            max_alternatives=5
        )
        
        for alt in alternatives:
            # Each alternative should have energy estimated (in Joules)
            assert "energy_joules" in alt
            # And similarity calculated
            assert "similarity" in alt
            # And optimization score
            assert "optimization_score" in alt


@pytest.mark.integration
class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""
    
    def test_complete_user_workflow(self, database_manager, sample_prompts, date_range):
        """Test a complete user workflow."""
        from nlp.parser import parse_prompt
        from nlp.complexity_score import compute_complexity
        from prediction.estimator import EnergyPredictor
        from anomaly.detector import AnomalyDetector
        from optimization.recomender import PromptOptimizer
        
        # User enters a verbose prompt
        prompt = sample_prompts["verbose"]
        
        # Step 1: Analyze the prompt
        parsed = parse_prompt(prompt, use_embeddings=False)
        complexity = compute_complexity(prompt)
        predictor = EnergyPredictor()
        energy = predictor.predict(prompt)
        detector = AnomalyDetector()
        anomaly = detector.detect(prompt, energy_kwh=energy.energy_kwh)
        
        # Step 2: Log the analysis
        database_manager.log_energy(
            prompt=prompt,
            token_count=parsed.token_count,
            complexity_score=complexity,
            energy_kwh=energy.energy_kwh,
            carbon_kg=energy.carbon_footprint_kg,
            water_liters=energy.water_usage_liters,
            is_anomaly=anomaly.is_anomaly
        )
        
        # Step 3: Get optimization suggestions
        optimizer = PromptOptimizer(min_similarity=0.6)
        optimization = optimizer.optimize(prompt)
        suggestions = optimizer.get_improvement_suggestions(prompt)
        
        # Step 4: If user accepts optimization, log it
        if optimization.energy_saved_kwh > 0:
            optimized_parsed = parse_prompt(optimization.optimized_prompt, use_embeddings=False)
            database_manager.log_energy(
                prompt=optimization.optimized_prompt,
                token_count=optimized_parsed.token_count,
                complexity_score=compute_complexity(optimization.optimized_prompt),
                energy_kwh=optimization.optimized_energy_kwh,
                was_optimized=True,
                energy_saved_kwh=optimization.energy_saved_kwh
            )
        
        # Step 5: Generate report
        report = database_manager.generate_transparency_report(
            date_range["start"],
            date_range["end"]
        )
        
        # Verify complete workflow
        assert report["summary"]["total_prompts"] >= 1
        assert len(suggestions) > 0
    
    def test_batch_processing_workflow(self, database_manager, sample_prompts, date_range):
        """Test batch processing of multiple prompts."""
        from prediction.estimator import EnergyPredictor
        from anomaly.detector import AnomalyDetector
        
        predictor = EnergyPredictor()
        detector = AnomalyDetector()
        
        prompts_to_process = [
            sample_prompts["simple"],
            sample_prompts["medium"],
            sample_prompts["complex"],
            sample_prompts["technical"]
        ]
        
        # Batch predict
        energy_results = predictor.predict_batch(prompts_to_process)
        
        # Batch detect
        anomaly_results = detector.detect_batch(
            prompts_to_process,
            [r.energy_kwh for r in energy_results]
        )
        
        # Log all
        for prompt, energy, anomaly in zip(prompts_to_process, energy_results, anomaly_results):
            database_manager.log_energy(
                prompt=prompt,
                token_count=50,  # Simplified for test
                complexity_score=0.5,
                energy_kwh=energy.energy_kwh,
                is_anomaly=anomaly.is_anomaly
            )
        
        # Generate report
        report = database_manager.generate_transparency_report(
            date_range["start"],
            date_range["end"]
        )
        
        assert report["summary"]["total_prompts"] >= 4


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling across modules."""
    
    def test_pipeline_handles_empty_prompt(self, database_manager, sample_prompts):
        """Test that pipeline handles empty prompt gracefully."""
        from nlp.parser import parse_prompt
        from nlp.complexity_score import compute_complexity
        from prediction.estimator import EnergyPredictor
        from anomaly.detector import AnomalyDetector
        
        prompt = sample_prompts["empty"]
        
        # All should handle empty prompt
        parsed = parse_prompt(prompt, use_embeddings=False)
        complexity = compute_complexity(prompt)
        
        predictor = EnergyPredictor()
        energy = predictor.predict(prompt)
        
        detector = AnomalyDetector()
        anomaly = detector.detect(prompt)
        
        # Should all return valid (possibly zero/default) results
        assert parsed is not None
        assert complexity is not None
        assert energy is not None
        assert anomaly is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
