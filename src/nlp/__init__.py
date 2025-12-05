"""
NLP module for Sustainable AI application.
Provides text parsing, complexity scoring, and simplification capabilities.
"""

from .parser import PromptParser, ParsedPrompt, parse_prompt, extract_features_dict
from .complexity_score import ComplexityScorer, ComplexityBreakdown, compute_complexity, get_complexity_breakdown
from .simplifier import TextSimplifier, SimplifiedPrompt, simplify_prompt, get_efficient_alternatives

__all__ = [
    # Parser
    'PromptParser',
    'ParsedPrompt',
    'parse_prompt',
    'extract_features_dict',
    
    # Complexity
    'ComplexityScorer',
    'ComplexityBreakdown',
    'compute_complexity',
    'get_complexity_breakdown',
    
    # Simplifier
    'TextSimplifier',
    'SimplifiedPrompt',
    'simplify_prompt',
    'get_efficient_alternatives'
]
