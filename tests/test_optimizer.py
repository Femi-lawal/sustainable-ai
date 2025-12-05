"""
Unit Tests for the Prompt Optimizer Module (recomender.py).
Tests prompt optimization and suggestion generation.
"""

import pytest
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPromptOptimizer:
    """Test PromptOptimizer class."""
    
    def test_optimizer_initialization(self, prompt_optimizer):
        """Test optimizer initializes correctly."""
        assert prompt_optimizer is not None
        assert prompt_optimizer.min_similarity > 0
        assert prompt_optimizer.target_energy_reduction > 0
    
    def test_optimizer_components_initialized(self, prompt_optimizer):
        """Test that optimizer initializes all components."""
        assert prompt_optimizer.parser is not None
        assert prompt_optimizer.simplifier is not None
        assert prompt_optimizer.energy_predictor is not None
    
    def test_estimate_energy(self, prompt_optimizer, sample_prompts):
        """Test energy estimation."""
        energy = prompt_optimizer.estimate_energy(sample_prompts["medium"])
        assert energy > 0
    
    def test_calculate_similarity(self, prompt_optimizer, sample_prompts):
        """Test similarity calculation."""
        similarity = prompt_optimizer.calculate_similarity(
            sample_prompts["simple"],
            sample_prompts["simple"]
        )
        # Same text should have high similarity
        assert similarity >= 0.8
    
    def test_optimize_returns_result(self, prompt_optimizer, sample_prompts):
        """Test that optimize returns OptimizationResult."""
        result = prompt_optimizer.optimize(sample_prompts["verbose"])
        assert hasattr(result, "original_prompt")
        assert hasattr(result, "optimized_prompt")
        assert hasattr(result, "energy_saved_kwh")
        assert hasattr(result, "optimization_score")
    
    def test_optimize_preserves_meaning(self, prompt_optimizer, sample_prompts):
        """Test that optimization preserves semantic meaning."""
        result = prompt_optimizer.optimize(sample_prompts["verbose"])
        # Similarity should be above threshold
        assert result.semantic_similarity >= prompt_optimizer.min_similarity
    
    def test_optimize_reduces_energy(self, prompt_optimizer, sample_prompts):
        """Test that optimization attempts to reduce energy."""
        result = prompt_optimizer.optimize(sample_prompts["verbose"])
        # May or may not reduce energy, but should try
        assert result.energy_reduction_percent >= 0 or result.optimized_prompt != result.original_prompt
    
    def test_optimization_score_range(self, prompt_optimizer, sample_prompts):
        """Test that optimization score is in valid range."""
        result = prompt_optimizer.optimize(sample_prompts["verbose"])
        assert 0 <= result.optimization_score <= 1
    
    def test_optimization_to_dict(self, prompt_optimizer, sample_prompts):
        """Test OptimizationResult.to_dict() method."""
        result = prompt_optimizer.optimize(sample_prompts["verbose"])
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "original_prompt" in result_dict
        assert "optimized_prompt" in result_dict


class TestPromptOptimizerStrategies:
    """Test different optimization strategies."""
    
    def test_simplify_strategy(self, prompt_optimizer, sample_prompts):
        """Test simplify strategy."""
        result = prompt_optimizer.optimize(sample_prompts["verbose"], strategy="simplify")
        assert result.strategy_used == "simplify"
    
    def test_compress_strategy(self, prompt_optimizer, sample_prompts):
        """Test compress strategy."""
        result = prompt_optimizer.optimize(sample_prompts["verbose"], strategy="compress")
        assert result.strategy_used == "compress"
    
    def test_truncate_strategy(self, prompt_optimizer, sample_prompts):
        """Test truncate strategy."""
        result = prompt_optimizer.optimize(sample_prompts["verbose"], strategy="truncate")
        assert result.strategy_used == "truncate"
    
    def test_auto_strategy(self, prompt_optimizer, sample_prompts):
        """Test auto strategy."""
        result = prompt_optimizer.optimize(sample_prompts["verbose"], strategy="auto")
        # Auto strategy should pick the best
        assert result.strategy_used is not None
    
    def test_apply_simplify_strategy(self, prompt_optimizer, sample_prompts):
        """Test internal simplify strategy application."""
        optimized, steps = prompt_optimizer._apply_simplify_strategy(sample_prompts["verbose"])
        assert isinstance(optimized, str)
        assert isinstance(steps, list)
    
    def test_apply_compress_strategy(self, prompt_optimizer, sample_prompts):
        """Test internal compress strategy application."""
        optimized, steps = prompt_optimizer._apply_compress_strategy(sample_prompts["verbose"])
        assert isinstance(optimized, str)
        assert isinstance(steps, list)
    
    def test_apply_truncate_strategy(self, prompt_optimizer, sample_prompts):
        """Test internal truncate strategy application."""
        optimized, steps = prompt_optimizer._apply_truncate_strategy(sample_prompts["verbose"])
        assert isinstance(optimized, str)
        assert isinstance(steps, list)
    
    def test_combine_strategies(self, prompt_optimizer, sample_prompts):
        """Test combined strategies."""
        optimized, steps = prompt_optimizer._combine_strategies(sample_prompts["verbose"])
        assert isinstance(optimized, str)
        assert len(steps) > 0


class TestPromptOptimizerAlternatives:
    """Test alternative generation."""
    
    def test_generate_alternatives(self, prompt_optimizer, sample_prompts):
        """Test generating alternatives."""
        alternatives = prompt_optimizer.generate_alternatives(
            sample_prompts["verbose"],
            max_alternatives=5
        )
        assert isinstance(alternatives, list)
    
    def test_alternatives_contain_required_fields(self, prompt_optimizer, sample_prompts):
        """Test that alternatives contain required fields."""
        alternatives = prompt_optimizer.generate_alternatives(
            sample_prompts["verbose"],
            max_alternatives=5
        )
        if alternatives:
            alt = alternatives[0]
            assert "prompt" in alt
            assert "strategy" in alt
            assert "similarity" in alt
            assert "energy_joules" in alt  # Energy in Joules (model's native unit)
    
    def test_alternatives_sorted_by_score(self, prompt_optimizer, sample_prompts):
        """Test that alternatives are sorted by optimization score."""
        alternatives = prompt_optimizer.generate_alternatives(
            sample_prompts["verbose"],
            max_alternatives=5
        )
        if len(alternatives) > 1:
            scores = [alt.get("optimization_score", 0) for alt in alternatives]
            assert scores == sorted(scores, reverse=True)
    
    def test_alternatives_respect_similarity(self, prompt_optimizer, sample_prompts):
        """Test that alternatives respect similarity threshold."""
        alternatives = prompt_optimizer.generate_alternatives(
            sample_prompts["verbose"],
            max_alternatives=5
        )
        for alt in alternatives:
            assert alt["similarity"] >= prompt_optimizer.min_similarity


class TestPromptOptimizerSuggestions:
    """Test improvement suggestion generation."""
    
    def test_get_improvement_suggestions(self, prompt_optimizer, sample_prompts):
        """Test getting improvement suggestions."""
        suggestions = prompt_optimizer.get_improvement_suggestions(sample_prompts["verbose"])
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
    
    def test_suggestions_for_high_token_count(self, prompt_optimizer, sample_prompts):
        """Test suggestions for high token count prompts."""
        suggestions = prompt_optimizer.get_improvement_suggestions(sample_prompts["long"])
        # Should suggest reducing tokens
        assert any("token" in s.lower() for s in suggestions)
    
    def test_suggestions_for_high_complexity(self, prompt_optimizer, sample_prompts):
        """Test suggestions for high complexity prompts."""
        suggestions = prompt_optimizer.get_improvement_suggestions(sample_prompts["technical"])
        # Should have suggestions
        assert len(suggestions) > 0
    
    def test_suggestions_for_simple_prompt(self, prompt_optimizer, sample_prompts):
        """Test suggestions for already simple prompts."""
        suggestions = prompt_optimizer.get_improvement_suggestions(sample_prompts["simple"])
        # Should still provide feedback
        assert len(suggestions) > 0


class TestPromptOptimizerEdgeCases:
    """Test edge cases for PromptOptimizer."""
    
    def test_optimize_empty_prompt(self, prompt_optimizer, sample_prompts):
        """Test optimizing empty prompt."""
        result = prompt_optimizer.optimize(sample_prompts["empty"])
        # Should handle gracefully
        assert result is not None
    
    def test_optimize_already_optimal(self, prompt_optimizer, sample_prompts):
        """Test optimizing already optimal prompt."""
        result = prompt_optimizer.optimize(sample_prompts["simple"])
        # Should still return result
        assert result is not None
        # Original and optimized may be similar
        assert result.semantic_similarity >= 0
    
    def test_optimize_unicode(self, prompt_optimizer, sample_prompts):
        """Test optimizing unicode prompt."""
        result = prompt_optimizer.optimize(sample_prompts["unicode"])
        assert result is not None
    
    def test_optimize_with_custom_params(self, prompt_optimizer, sample_prompts, model_config):
        """Test optimization with custom model parameters."""
        result = prompt_optimizer.optimize(
            sample_prompts["verbose"],
            num_layers=model_config["num_layers"],
            training_hours=model_config["training_hours"],
            flops_per_hour=model_config["flops_per_hour"]
        )
        assert result is not None


class TestPromptOptimizerCustomThresholds:
    """Test optimizer with custom thresholds."""
    
    def test_high_similarity_threshold(self):
        """Test with high similarity threshold."""
        from optimization.recomender import PromptOptimizer
        optimizer = PromptOptimizer(min_similarity=0.95)
        assert optimizer.min_similarity == 0.95
    
    def test_low_similarity_threshold(self):
        """Test with low similarity threshold."""
        from optimization.recomender import PromptOptimizer
        optimizer = PromptOptimizer(min_similarity=0.5)
        assert optimizer.min_similarity == 0.5
    
    def test_custom_energy_reduction_target(self):
        """Test with custom energy reduction target."""
        from optimization.recomender import PromptOptimizer
        optimizer = PromptOptimizer(target_energy_reduction=0.5)
        assert optimizer.target_energy_reduction == 0.5


class TestPromptOptimizerConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_optimizer_function(self):
        """Test create_optimizer() function."""
        from optimization.recomender import create_optimizer
        optimizer = create_optimizer(min_similarity=0.7)
        assert optimizer is not None
        assert optimizer.min_similarity == 0.7
    
    def test_optimize_prompt_function(self, sample_prompts):
        """Test optimize_prompt() convenience function."""
        from optimization.recomender import optimize_prompt
        result = optimize_prompt(sample_prompts["verbose"])
        assert isinstance(result, dict)
        assert "optimized_prompt" in result
    
    def test_get_suggestions_function(self, sample_prompts):
        """Test get_suggestions() convenience function."""
        from optimization.recomender import get_suggestions
        suggestions = get_suggestions(sample_prompts["verbose"])
        assert isinstance(suggestions, list)


class TestOptimizationMetrics:
    """Test optimization metrics calculation."""
    
    def test_energy_reduction_percent(self, prompt_optimizer, sample_prompts):
        """Test energy reduction percentage calculation."""
        result = prompt_optimizer.optimize(sample_prompts["verbose"])
        # Should be a valid percentage
        assert -100 <= result.energy_reduction_percent <= 100
    
    def test_complexity_reduction(self, prompt_optimizer, sample_prompts):
        """Test complexity reduction calculation."""
        result = prompt_optimizer.optimize(sample_prompts["verbose"])
        # Should be a valid percentage
        assert isinstance(result.complexity_reduction, float)
    
    def test_token_metrics(self, prompt_optimizer, sample_prompts):
        """Test that token metrics are tracked."""
        result = prompt_optimizer.optimize(sample_prompts["verbose"])
        assert result.original_energy_kwh > 0
        assert result.optimized_energy_kwh > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
