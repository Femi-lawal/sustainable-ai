"""
Prompt Optimization and Recommendation Engine.
Suggests energy-efficient alternatives while preserving semantic meaning.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from project
try:
    from utils.config import PROMPT_OPTIMIZER_CONFIG
    from nlp.parser import PromptParser, parse_prompt
    from nlp.complexity_score import compute_complexity, get_complexity_breakdown
    from nlp.simplifier import TextSimplifier, simplify_prompt
    from prediction.estimator import EnergyPredictor
except ImportError:
    from src.utils.config import PROMPT_OPTIMIZER_CONFIG
    from src.nlp.parser import PromptParser, parse_prompt
    from src.nlp.complexity_score import compute_complexity, get_complexity_breakdown
    from src.nlp.simplifier import TextSimplifier, simplify_prompt
    from src.prediction.estimator import EnergyPredictor


@dataclass
class OptimizationResult:
    """
    Result of prompt optimization.
    
    UNIT CONVENTION: Energy values are in Joules (model's native unit)
    """
    original_prompt: str
    optimized_prompt: str
    
    # Energy metrics (in Joules - model's native unit)
    original_energy_kwh: float  # Kept for backward compatibility, but values are in Joules
    optimized_energy_kwh: float  # Kept for backward compatibility, but values are in Joules  
    energy_saved_kwh: float      # Actually Joules saved
    energy_reduction_percent: float
    
    # Quality metrics
    semantic_similarity: float
    complexity_reduction: float
    
    # Strategy info
    strategy_used: str
    optimization_steps: List[str]
    
    # Alternative suggestions
    alternatives: List[Dict[str, Any]]
    
    # Overall score (0-1, higher is better optimization)
    optimization_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_prompt": self.original_prompt[:100] + "..." if len(self.original_prompt) > 100 else self.original_prompt,
            "optimized_prompt": self.optimized_prompt[:100] + "..." if len(self.optimized_prompt) > 100 else self.optimized_prompt,
            "original_energy_kwh": self.original_energy_kwh,
            "optimized_energy_kwh": self.optimized_energy_kwh,
            "energy_saved_kwh": self.energy_saved_kwh,
            "energy_reduction_percent": self.energy_reduction_percent,
            "semantic_similarity": self.semantic_similarity,
            "complexity_reduction": self.complexity_reduction,
            "strategy_used": self.strategy_used,
            "optimization_steps": self.optimization_steps,
            "alternatives": self.alternatives[:3],  # Top 3 alternatives
            "optimization_score": self.optimization_score
        }


class PromptOptimizer:
    """
    Optimizes prompts for energy efficiency while maintaining semantic meaning.
    
    Uses multiple strategies:
    1. T5 Model - AI-based text simplification (primary)
    2. Simplification - Reduce verbose language  
    3. Compression - Remove redundant information
    4. Truncation - Focus on essential content
    
    Combines T5 model inference with energy prediction to find optimal prompts.
    """
    
    def __init__(self, 
                 min_similarity: float = 0.65,
                 target_energy_reduction: float = 0.30):
        """
        Initialize the prompt optimizer.
        
        Args:
            min_similarity: Minimum semantic similarity to accept (default 0.65 for T5 compatibility)
            target_energy_reduction: Target energy reduction (0-1)
        """
        self.min_similarity = min_similarity
        self.target_energy_reduction = target_energy_reduction
        
        # Initialize components
        self.parser = PromptParser(use_embeddings=True)
        # Use slightly lower threshold for simplifier to allow T5 outputs
        self.simplifier = TextSimplifier(min_similarity_threshold=max(0.55, min_similarity - 0.10))
        self.energy_predictor = EnergyPredictor()
        
        # Optimization strategies in order of preference
        self.strategies = PROMPT_OPTIMIZER_CONFIG.strategies
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two prompts.
        
        Args:
            text1: First prompt
            text2: Second prompt
        
        Returns:
            Similarity score (0-1)
        """
        return self.parser.compute_similarity(text1, text2)
    
    def estimate_energy(self, prompt: str, num_layers: int = 24,
                       training_hours: float = 8.0,
                       flops_per_hour: float = 1e11) -> float:
        """
        Estimate energy consumption for a prompt.
        
        Args:
            prompt: Input prompt
            num_layers: LLM layers
            training_hours: Training time
            flops_per_hour: Compute operations
        
        Returns:
            Estimated energy in Joules (model's native unit)
        """
        prediction = self.energy_predictor.predict(
            prompt, num_layers, training_hours, flops_per_hour
        )
        return prediction.energy_joules
    
    def _apply_simplify_strategy(self, prompt: str) -> Tuple[str, List[str]]:
        """Apply simplification strategy (T5-based as default)."""
        result = self.simplifier.simplify(prompt, "auto")
        steps = [
            f"Applied {result.strategy_used} strategy",
            "T5-based text simplification" if "t5" in result.strategy_used.lower() else "Rule-based optimization",
            f"Achieved {result.token_reduction_percent:.0f}% token reduction"
        ]
        return result.simplified, steps
    
    def _apply_t5_strategy(self, prompt: str) -> Tuple[str, List[str]]:
        """Apply pure T5 model-based simplification."""
        result = self.simplifier.simplify(prompt, "t5")
        steps = [
            "T5 model paraphrasing",
            "Selected most efficient variant",
            f"Achieved {result.token_reduction_percent:.0f}% token reduction"
        ]
        return result.simplified, steps
    
    def _apply_compress_strategy(self, prompt: str) -> Tuple[str, List[str]]:
        """Apply compression strategy."""
        result = self.simplifier.simplify(prompt, "compress")
        steps = [
            "Compressed sentence structure",
            "Removed parenthetical expressions"
        ]
        return result.simplified, steps
    
    def _apply_truncate_strategy(self, prompt: str) -> Tuple[str, List[str]]:
        """Apply truncation strategy."""
        result = self.simplifier.simplify(prompt, "truncate")
        steps = [
            "Truncated to essential content",
            "Focused on key sentences"
        ]
        return result.simplified, steps
    
    def _apply_paraphrase_strategy(self, prompt: str) -> Tuple[str, List[str]]:
        """Apply paraphrasing strategy."""
        result = self.simplifier.simplify(prompt, "paraphrase")
        steps = [
            "Generated alternative phrasing",
            "Selected most efficient version"
        ]
        return result.simplified, steps
    
    def _combine_strategies(self, prompt: str) -> Tuple[str, List[str]]:
        """Apply multiple strategies in sequence."""
        current = prompt
        all_steps = []
        
        # Apply strategies in sequence
        for strategy in self.strategies[:3]:  # Use top 3 strategies
            if strategy == "simplify":
                current, steps = self._apply_simplify_strategy(current)
            elif strategy == "compress":
                current, steps = self._apply_compress_strategy(current)
            elif strategy == "truncate":
                current, steps = self._apply_truncate_strategy(current)
            elif strategy == "paraphrase":
                current, steps = self._apply_paraphrase_strategy(current)
            
            all_steps.extend(steps)
            
            # Check if still maintains similarity
            similarity = self.calculate_similarity(prompt, current)
            if similarity < self.min_similarity:
                # Revert to previous version
                break
        
        return current, all_steps
    
    def generate_alternatives(self, prompt: str, max_alternatives: int = 5,
                             **energy_params) -> List[Dict[str, Any]]:
        """
        Generate multiple optimized alternatives.
        
        Args:
            prompt: Original prompt
            max_alternatives: Maximum number of alternatives
            **energy_params: Parameters for energy estimation
        
        Returns:
            List of alternative dictionaries
        """
        alternatives = []
        original_energy = self.estimate_energy(prompt, **energy_params)
        original_complexity = compute_complexity(prompt)
        
        # Try each strategy - T5 first as primary per project requirements
        strategies_to_try = [
            ("t5", self._apply_t5_strategy),
            ("simplify", self._apply_simplify_strategy),
            ("compress", self._apply_compress_strategy),
            ("truncate", self._apply_truncate_strategy),
            ("combined", self._combine_strategies)
        ]
        
        for strategy_name, strategy_func in strategies_to_try:
            try:
                optimized, steps = strategy_func(prompt)
                
                # Skip if same as original
                if optimized.strip() == prompt.strip():
                    continue
                
                # Calculate metrics
                similarity = self.calculate_similarity(prompt, optimized)
                if similarity < self.min_similarity:
                    continue
                
                opt_energy = self.estimate_energy(optimized, **energy_params)
                opt_complexity = compute_complexity(optimized)
                
                energy_reduction = ((original_energy - opt_energy) / original_energy * 100) if original_energy > 0 else 0
                complexity_reduction = ((original_complexity - opt_complexity) / original_complexity * 100) if original_complexity > 0 else 0
                
                # Calculate optimization score
                # Prioritize energy reduction while ensuring minimum similarity
                # Score only counts if there's actual improvement
                if energy_reduction > 0:
                    score = (
                        0.50 * (energy_reduction / 50) +  # Normalize to ~50% max reduction
                        0.25 * similarity +
                        0.15 * max(0, complexity_reduction / 50) +
                        0.10 * max(0, 1 - len(optimized) / len(prompt))
                    )
                else:
                    # No energy savings = very low score
                    score = 0.1 * similarity
                
                alternatives.append({
                    "prompt": optimized,
                    "strategy": strategy_name,
                    "similarity": round(similarity, 4),
                    "energy_joules": round(opt_energy, 2),  # Energy in Joules (model's native unit)
                    "energy_reduction_percent": round(energy_reduction, 2),
                    "complexity_reduction_percent": round(complexity_reduction, 2),
                    "optimization_score": round(max(0, min(1, score)), 4)
                })
                
            except Exception as e:
                continue
        
        # Sort by optimization score
        alternatives.sort(key=lambda x: x["optimization_score"], reverse=True)
        
        return alternatives[:max_alternatives]
    
    def optimize(self, prompt: str, num_layers: int = 24,
                 training_hours: float = 8.0,
                 flops_per_hour: float = 1e11,
                 strategy: str = "auto") -> OptimizationResult:
        """
        Optimize a prompt for energy efficiency.
        
        Args:
            prompt: Original prompt text
            num_layers: Number of LLM layers
            training_hours: Training time in hours
            flops_per_hour: Compute operations per hour
            strategy: Optimization strategy ("auto", "simplify", "compress", "truncate", "paraphrase")
        
        Returns:
            OptimizationResult with detailed analysis
        """
        energy_params = {
            "num_layers": num_layers,
            "training_hours": training_hours,
            "flops_per_hour": flops_per_hour
        }
        
        # Calculate original metrics
        original_energy = self.estimate_energy(prompt, **energy_params)
        original_complexity = compute_complexity(prompt)
        
        # Apply optimization strategy
        if strategy == "auto":
            # Generate all alternatives and pick the best
            alternatives = self.generate_alternatives(prompt, 5, **energy_params)
            
            if alternatives:
                best = alternatives[0]
                optimized = best["prompt"]
                strategy_used = best["strategy"]
                optimization_steps = [f"Applied {strategy_used} strategy"]
            else:
                # Fallback to combined strategy
                optimized, optimization_steps = self._combine_strategies(prompt)
                strategy_used = "combined"
        else:
            # Apply specific strategy
            if strategy == "simplify":
                optimized, optimization_steps = self._apply_simplify_strategy(prompt)
            elif strategy == "compress":
                optimized, optimization_steps = self._apply_compress_strategy(prompt)
            elif strategy == "truncate":
                optimized, optimization_steps = self._apply_truncate_strategy(prompt)
            elif strategy == "paraphrase":
                optimized, optimization_steps = self._apply_paraphrase_strategy(prompt)
            else:
                optimized, optimization_steps = self._combine_strategies(prompt)
            
            strategy_used = strategy
            alternatives = self.generate_alternatives(prompt, 3, **energy_params)
        
        # Calculate optimized metrics
        optimized_energy = self.estimate_energy(optimized, **energy_params)
        optimized_complexity = compute_complexity(optimized)
        
        # Calculate improvements
        energy_saved = original_energy - optimized_energy
        energy_reduction_percent = (energy_saved / original_energy * 100) if original_energy > 0 else 0
        complexity_reduction = ((original_complexity - optimized_complexity) / original_complexity * 100) if original_complexity > 0 else 0
        
        # Semantic similarity
        similarity = self.calculate_similarity(prompt, optimized)
        
        # Overall optimization score
        optimization_score = (
            0.35 * max(0, energy_reduction_percent / 50) +  # Up to 50% reduction = full score
            0.35 * similarity +
            0.20 * max(0, complexity_reduction / 50) +
            0.10 * (1 if len(optimized) < len(prompt) else 0.5)
        )
        optimization_score = max(0, min(1, optimization_score))
        
        # Get alternatives if not already generated
        if strategy == "auto":
            pass  # Already generated
        else:
            alternatives = self.generate_alternatives(prompt, 3, **energy_params)
        
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            original_energy_kwh=round(original_energy, 6),
            optimized_energy_kwh=round(optimized_energy, 6),
            energy_saved_kwh=round(max(0, energy_saved), 6),
            energy_reduction_percent=round(max(0, energy_reduction_percent), 2),
            semantic_similarity=round(similarity, 4),
            complexity_reduction=round(max(0, complexity_reduction), 2),
            strategy_used=strategy_used,
            optimization_steps=optimization_steps,
            alternatives=alternatives,
            optimization_score=round(optimization_score, 4)
        )
    
    def get_improvement_suggestions(self, prompt: str) -> List[str]:
        """
        Get specific improvement suggestions for a prompt.
        
        Args:
            prompt: Input prompt
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Analyze the prompt
        parsed = parse_prompt(prompt, use_embeddings=False)
        complexity = get_complexity_breakdown(prompt)
        
        # Token count suggestions
        if parsed.token_count > 200:
            suggestions.append(
                f"Your prompt has {parsed.token_count} tokens. Consider reducing to under 200 tokens for better efficiency."
            )
        
        # Complexity suggestions
        if complexity["overall_score"] > 0.6:
            suggestions.append(
                "High complexity detected. Try using simpler vocabulary and shorter sentences."
            )
        
        if complexity["vocabulary_complexity"] > 0.5:
            suggestions.append(
                "Consider replacing technical jargon with simpler terms where possible."
            )
        
        if complexity["sentence_complexity"] > 0.5:
            suggestions.append(
                "Break long sentences into shorter ones for clearer communication."
            )
        
        # Redundancy suggestions
        if parsed.stopword_ratio > 0.5:
            suggestions.append(
                "High ratio of filler words detected. Remove unnecessary words like 'basically', 'actually', 'really'."
            )
        
        # Structure suggestions
        if parsed.sentence_count == 1 and parsed.word_count > 50:
            suggestions.append(
                "Single long sentence detected. Break into multiple focused questions or statements."
            )
        
        # General suggestions
        if not suggestions:
            suggestions.append(
                "Your prompt appears reasonably efficient. Minor optimizations may still be possible."
            )
        
        return suggestions


# Factory function
def create_optimizer(min_similarity: float = 0.75) -> PromptOptimizer:
    """
    Create a prompt optimizer instance.
    
    Args:
        min_similarity: Minimum semantic similarity threshold
    
    Returns:
        PromptOptimizer instance
    """
    return PromptOptimizer(min_similarity=min_similarity)


# Convenience functions
def optimize_prompt(prompt: str, strategy: str = "auto") -> Dict[str, Any]:
    """
    Quick function to optimize a prompt.
    
    Args:
        prompt: Input prompt text
        strategy: Optimization strategy
    
    Returns:
        Dictionary with optimization results
    """
    optimizer = PromptOptimizer()
    result = optimizer.optimize(prompt, strategy=strategy)
    return result.to_dict()


def get_suggestions(prompt: str) -> List[str]:
    """
    Get improvement suggestions for a prompt.
    
    Args:
        prompt: Input prompt text
    
    Returns:
        List of suggestions
    """
    optimizer = PromptOptimizer()
    return optimizer.get_improvement_suggestions(prompt)


if __name__ == "__main__":
    # Test the optimizer
    test_prompts = [
        """In order to provide you with the most comprehensive and detailed 
        explanation possible, I would like to basically take into consideration 
        all of the various different factors that are involved in machine 
        learning, including the fundamental concepts, the various types of 
        algorithms that are commonly used, and the practical applications.""",
        
        """Can you explain how transformers work in natural language processing, 
        including the attention mechanism and self-attention?"""
    ]
    
    optimizer = PromptOptimizer()
    
    print("Prompt Optimization Results")
    print("=" * 70)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Prompt {i} ---")
        print(f"Original ({len(prompt)} chars):")
        print(f"  {prompt[:100]}...")
        
        result = optimizer.optimize(prompt)
        
        print(f"\nOptimized ({len(result.optimized_prompt)} chars):")
        print(f"  {result.optimized_prompt[:100]}...")
        
        print(f"\nMetrics:")
        print(f"  Energy: {result.original_energy_kwh:.6f} -> {result.optimized_energy_kwh:.6f} kWh")
        print(f"  Reduction: {result.energy_reduction_percent:.1f}%")
        print(f"  Similarity: {result.semantic_similarity:.2%}")
        print(f"  Strategy: {result.strategy_used}")
        print(f"  Score: {result.optimization_score:.2f}")
        
        print("\nSuggestions:")
        for suggestion in optimizer.get_improvement_suggestions(prompt)[:3]:
            print(f"  • {suggestion[:80]}...")
