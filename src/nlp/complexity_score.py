"""
Complexity Score Calculator for prompts.
Computes a multi-dimensional complexity score that correlates with computational requirements.
"""

import math
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


@dataclass
class ComplexityBreakdown:
    """
    Detailed breakdown of complexity components.
    """
    # Individual scores (0-1 range)
    sentence_complexity: float
    vocabulary_complexity: float
    syntactic_complexity: float
    semantic_density: float
    structural_complexity: float
    
    # Overall composite score
    overall_score: float
    
    # Interpretation
    level: str  # "low", "medium", "high", "very_high"
    energy_impact: str  # Description of expected energy impact
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sentence_complexity": self.sentence_complexity,
            "vocabulary_complexity": self.vocabulary_complexity,
            "syntactic_complexity": self.syntactic_complexity,
            "semantic_density": self.semantic_density,
            "structural_complexity": self.structural_complexity,
            "overall_score": self.overall_score,
            "level": self.level,
            "energy_impact": self.energy_impact
        }


class ComplexityScorer:
    """
    Calculates multi-dimensional complexity scores for prompts.
    
    The complexity score is designed to correlate with the computational
    resources needed to process a prompt by an LLM.
    """
    
    # Complexity thresholds
    LOW_THRESHOLD = 0.25
    MEDIUM_THRESHOLD = 0.50
    HIGH_THRESHOLD = 0.75
    
    # Weights for composite score
    DEFAULT_WEIGHTS = {
        "sentence_complexity": 0.20,
        "vocabulary_complexity": 0.25,
        "syntactic_complexity": 0.20,
        "semantic_density": 0.20,
        "structural_complexity": 0.15
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the complexity scorer.
        
        Args:
            weights: Custom weights for complexity components
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # Common technical/domain terms that increase complexity
        self.technical_indicators = {
            'algorithm', 'neural', 'network', 'machine', 'learning',
            'optimization', 'gradient', 'architecture', 'transformer',
            'attention', 'embedding', 'vector', 'matrix', 'tensor',
            'inference', 'training', 'model', 'layer', 'activation',
            'backpropagation', 'convolution', 'recurrent', 'parameter',
            'hyperparameter', 'regularization', 'normalization', 'batch',
            'epoch', 'loss', 'function', 'derivative', 'stochastic',
            'analysis', 'statistical', 'probability', 'distribution',
            'hypothesis', 'correlation', 'regression', 'classification'
        }
        
        # Question words that indicate complexity of reasoning
        self.reasoning_indicators = {
            'why', 'how', 'explain', 'analyze', 'compare', 'contrast',
            'evaluate', 'assess', 'justify', 'elaborate', 'distinguish',
            'synthesize', 'critique', 'hypothesize', 'deduce', 'infer'
        }
    
    def calculate_sentence_complexity(self, text: str) -> float:
        """
        Calculate complexity based on sentence structure.
        
        Args:
            text: Input text
        
        Returns:
            Complexity score (0-1)
        """
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = text.split('.')
        
        if not sentences:
            return 0.0
        
        # Metrics
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentences)
        
        # Variance in sentence length (more variety = more complexity)
        if len(sentence_lengths) > 1:
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            length_variance = min(math.sqrt(variance) / 10, 1.0)
        else:
            length_variance = 0.0
        
        # Normalize average length (typical sentence: 15-20 words)
        normalized_avg = min(avg_length / 25, 1.0)
        
        # Clause indicators (commas, semicolons, conjunctions per sentence)
        clause_count = text.count(',') + text.count(';') + text.lower().count(' and ') + text.lower().count(' but ')
        clause_density = min(clause_count / (len(sentences) * 3), 1.0)
        
        return 0.4 * normalized_avg + 0.3 * length_variance + 0.3 * clause_density
    
    def calculate_vocabulary_complexity(self, text: str) -> float:
        """
        Calculate complexity based on vocabulary usage.
        
        Args:
            text: Input text
        
        Returns:
            Complexity score (0-1)
        """
        try:
            words = word_tokenize(text.lower())
        except Exception:
            words = text.lower().split()
        
        word_tokens = [w for w in words if w.isalnum()]
        
        if not word_tokens:
            return 0.0
        
        # Type-token ratio (vocabulary richness)
        unique_words = set(word_tokens)
        ttr = len(unique_words) / len(word_tokens)
        
        # Average word length (longer words often more complex)
        avg_word_length = sum(len(w) for w in word_tokens) / len(word_tokens)
        normalized_word_length = min(avg_word_length / 10, 1.0)
        
        # Technical term density
        technical_count = sum(1 for w in word_tokens if w in self.technical_indicators)
        technical_density = min(technical_count / max(len(word_tokens), 1) * 10, 1.0)
        
        # Long word ratio (words > 8 characters)
        long_words = sum(1 for w in word_tokens if len(w) > 8)
        long_word_ratio = min(long_words / max(len(word_tokens), 1) * 3, 1.0)
        
        return 0.25 * ttr + 0.25 * normalized_word_length + 0.30 * technical_density + 0.20 * long_word_ratio
    
    def calculate_syntactic_complexity(self, text: str) -> float:
        """
        Calculate complexity based on syntactic structure.
        
        Args:
            text: Input text
        
        Returns:
            Complexity score (0-1)
        """
        # Count nested structures
        nested_parens = text.count('(') + text.count('[') + text.count('{')
        nested_score = min(nested_parens / 5, 1.0)
        
        # Question complexity
        question_marks = text.count('?')
        question_words = sum(1 for word in self.reasoning_indicators if word in text.lower())
        question_score = min((question_marks + question_words) / 5, 1.0)
        
        # Conditional structures
        conditionals = sum(1 for pattern in ['if ', 'when ', 'unless ', 'while ', 'although '] 
                         if pattern in text.lower())
        conditional_score = min(conditionals / 3, 1.0)
        
        # Lists and enumerations
        list_indicators = text.count(':') + len(re.findall(r'\d+\.', text)) + text.count('•') + text.count('-')
        list_score = min(list_indicators / 5, 1.0)
        
        return 0.25 * nested_score + 0.30 * question_score + 0.25 * conditional_score + 0.20 * list_score
    
    def calculate_semantic_density(self, text: str) -> float:
        """
        Calculate the semantic information density.
        
        Args:
            text: Input text
        
        Returns:
            Complexity score (0-1)
        """
        try:
            words = word_tokenize(text.lower())
        except Exception:
            words = text.lower().split()
        
        word_tokens = [w for w in words if w.isalnum()]
        char_count = len(text.replace(' ', ''))
        
        if not word_tokens or char_count == 0:
            return 0.0
        
        # Information per character (concepts per character)
        concepts = len(set(word_tokens))
        info_density = min(concepts / char_count * 50, 1.0)
        
        # Named entity density (approximate by capitalized words)
        capitalized = sum(1 for w in text.split() if w and w[0].isupper() and w[1:].islower())
        entity_density = min(capitalized / max(len(word_tokens), 1) * 5, 1.0)
        
        # Number density (numbers often indicate specific information)
        numbers = len(re.findall(r'\d+', text))
        number_density = min(numbers / max(len(word_tokens), 1) * 10, 1.0)
        
        # Domain specificity (based on technical terms)
        technical_count = sum(1 for w in word_tokens if w in self.technical_indicators)
        domain_density = min(technical_count / max(len(word_tokens), 1) * 8, 1.0)
        
        return 0.30 * info_density + 0.20 * entity_density + 0.20 * number_density + 0.30 * domain_density
    
    def calculate_structural_complexity(self, text: str) -> float:
        """
        Calculate complexity based on overall structure.
        
        Args:
            text: Input text
        
        Returns:
            Complexity score (0-1)
        """
        # Length-based complexity
        char_count = len(text)
        length_score = min(char_count / 2000, 1.0)
        
        # Paragraph complexity (line breaks)
        paragraphs = text.count('\n\n') + 1
        para_score = min(paragraphs / 5, 1.0)
        
        # Format diversity (code blocks, quotes, etc.)
        format_indicators = text.count('```') + text.count('`') + text.count('"') + text.count("'")
        format_score = min(format_indicators / 10, 1.0)
        
        # Multi-part structure (numbered lists, sections)
        sections = len(re.findall(r'#+\s|^\d+\.|^[a-z]\)', text, re.MULTILINE))
        section_score = min(sections / 5, 1.0)
        
        return 0.35 * length_score + 0.20 * para_score + 0.20 * format_score + 0.25 * section_score
    
    def calculate(self, text: str) -> ComplexityBreakdown:
        """
        Calculate comprehensive complexity score.
        
        Args:
            text: Input prompt text
        
        Returns:
            ComplexityBreakdown with all scores and interpretation
        """
        # Calculate individual components
        sentence = self.calculate_sentence_complexity(text)
        vocabulary = self.calculate_vocabulary_complexity(text)
        syntactic = self.calculate_syntactic_complexity(text)
        semantic = self.calculate_semantic_density(text)
        structural = self.calculate_structural_complexity(text)
        
        # Calculate weighted composite score
        overall = (
            self.weights["sentence_complexity"] * sentence +
            self.weights["vocabulary_complexity"] * vocabulary +
            self.weights["syntactic_complexity"] * syntactic +
            self.weights["semantic_density"] * semantic +
            self.weights["structural_complexity"] * structural
        )
        
        # Determine level and impact
        if overall < self.LOW_THRESHOLD:
            level = "low"
            energy_impact = "Minimal computational resources required. Low energy consumption expected."
        elif overall < self.MEDIUM_THRESHOLD:
            level = "medium"
            energy_impact = "Moderate computational resources needed. Standard energy consumption."
        elif overall < self.HIGH_THRESHOLD:
            level = "high"
            energy_impact = "Significant computational resources required. Above-average energy consumption."
        else:
            level = "very_high"
            energy_impact = "Substantial computational resources needed. High energy consumption expected."
        
        return ComplexityBreakdown(
            sentence_complexity=round(sentence, 4),
            vocabulary_complexity=round(vocabulary, 4),
            syntactic_complexity=round(syntactic, 4),
            semantic_density=round(semantic, 4),
            structural_complexity=round(structural, 4),
            overall_score=round(overall, 4),
            level=level,
            energy_impact=energy_impact
        )
    
    def get_score(self, text: str) -> float:
        """
        Get just the overall complexity score.
        
        Args:
            text: Input prompt text
        
        Returns:
            Overall complexity score (0-1)
        """
        return self.calculate(text).overall_score


# Convenience functions
def compute_complexity(text: str) -> float:
    """
    Quick function to compute complexity score.
    
    Args:
        text: Input prompt text
    
    Returns:
        Complexity score (0-1)
    """
    scorer = ComplexityScorer()
    return scorer.get_score(text)


def get_complexity_breakdown(text: str) -> Dict[str, Any]:
    """
    Get detailed complexity breakdown as dictionary.
    
    Args:
        text: Input prompt text
    
    Returns:
        Dictionary with complexity breakdown
    """
    scorer = ComplexityScorer()
    return scorer.calculate(text).to_dict()


if __name__ == "__main__":
    # Test examples
    test_prompts = [
        "Hello, how are you?",
        "Explain the concept of machine learning.",
        """Analyze the multi-layer perceptron architecture, discussing how 
        backpropagation enables gradient-based optimization of neural network 
        parameters through the chain rule of calculus. Include the mathematical 
        formulation of the cross-entropy loss function and its partial derivatives.""",
        """Design a comprehensive system architecture for a distributed machine 
        learning pipeline that handles:
        1. Real-time data ingestion from multiple sources
        2. Feature engineering with automated transformations
        3. Model training with hyperparameter optimization
        4. A/B testing framework for model deployment
        5. Monitoring and alerting for model drift detection
        
        Provide code examples in Python and explain the trade-offs between 
        different architectural choices."""
    ]
    
    scorer = ComplexityScorer()
    
    print("Complexity Analysis Results")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        result = scorer.calculate(prompt)
        print(f"\nPrompt {i}: {prompt[:50]}...")
        print(f"  Overall Score: {result.overall_score:.4f} ({result.level})")
        print(f"  Sentence Complexity: {result.sentence_complexity:.4f}")
        print(f"  Vocabulary Complexity: {result.vocabulary_complexity:.4f}")
        print(f"  Syntactic Complexity: {result.syntactic_complexity:.4f}")
        print(f"  Semantic Density: {result.semantic_density:.4f}")
        print(f"  Structural Complexity: {result.structural_complexity:.4f}")
        print(f"  Energy Impact: {result.energy_impact}")
