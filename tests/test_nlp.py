"""
Unit Tests for the NLP Module.
Tests parser.py, complexity_score.py, and simplifier.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# PARSER TESTS
# ============================================================================

class TestPromptParser:
    """Test PromptParser class."""
    
    def test_parser_initialization(self, parser):
        """Test parser initializes correctly."""
        assert parser is not None
        assert parser.language == "english"
        assert len(parser.stop_words) > 0
    
    def test_parse_simple_prompt(self, parser, sample_prompts):
        """Test parsing a simple prompt."""
        result = parser.parse(sample_prompts["simple"])
        assert result.token_count > 0
        assert result.word_count > 0
        assert result.char_count > 0
        assert result.sentence_count >= 1
    
    def test_parse_complex_prompt(self, parser, sample_prompts):
        """Test parsing a complex prompt."""
        result = parser.parse(sample_prompts["complex"])
        assert result.token_count > 30  # Token count depends on tokenizer
        assert result.word_count > 10
        assert result.sentence_count >= 1
    
    def test_parse_empty_prompt(self, parser, sample_prompts):
        """Test parsing an empty prompt."""
        result = parser.parse(sample_prompts["empty"])
        assert result.token_count >= 0
        assert result.word_count == 0
    
    def test_parse_whitespace_prompt(self, parser, sample_prompts):
        """Test parsing whitespace-only prompt."""
        result = parser.parse(sample_prompts["whitespace"])
        assert result.word_count == 0
    
    def test_parse_unicode_prompt(self, parser, sample_prompts):
        """Test parsing Unicode prompt."""
        result = parser.parse(sample_prompts["unicode"])
        assert result.char_count > 0
        # Should not raise any exceptions
    
    def test_parse_special_chars(self, parser, sample_prompts):
        """Test parsing prompt with special characters."""
        result = parser.parse(sample_prompts["special_chars"])
        assert result.punct_ratio > 0
    
    def test_parsed_prompt_to_dict(self, parser, sample_prompts):
        """Test ParsedPrompt.to_dict() method."""
        result = parser.parse(sample_prompts["simple"])
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "token_count" in result_dict
        assert "word_count" in result_dict
        assert "char_count" in result_dict
    
    def test_parsed_prompt_feature_vector(self, parser, sample_prompts):
        """Test ParsedPrompt.get_feature_vector() method."""
        result = parser.parse(sample_prompts["simple"])
        features = result.get_feature_vector()
        assert isinstance(features, list)
        assert all(isinstance(f, (int, float)) for f in features)
    
    def test_clean_text_removes_whitespace(self, parser):
        """Test that clean_text removes extra whitespace."""
        dirty_text = "  Hello   world  \n\t "
        cleaned = parser.clean_text(dirty_text)
        assert cleaned == "Hello world"
    
    def test_get_token_count(self, parser):
        """Test transformer tokenizer token count."""
        text = "Hello, how are you today?"
        token_count = parser.get_token_count(text)
        assert token_count > 0
        assert token_count >= 5  # At least 5 tokens expected
    
    def test_get_pos_distribution(self, parser):
        """Test POS tag distribution."""
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        noun_ratio, verb_ratio, adj_ratio, adv_ratio = parser.get_pos_distribution(words)
        # All ratios should be between 0 and 1
        for ratio in [noun_ratio, verb_ratio, adj_ratio, adv_ratio]:
            assert 0 <= ratio <= 1
    
    def test_parse_batch(self, parser, sample_prompts):
        """Test batch parsing."""
        prompts = [sample_prompts["simple"], sample_prompts["medium"]]
        results = parser.parse_batch(prompts)
        assert len(results) == 2
        assert all(hasattr(r, "token_count") for r in results)
    
    def test_stopword_ratio_calculated(self, parser, sample_prompts):
        """Test that stopword ratio is calculated."""
        result = parser.parse(sample_prompts["medium"])
        assert 0 <= result.stopword_ratio <= 1
    
    def test_vocabulary_richness_calculated(self, parser, sample_prompts):
        """Test vocabulary richness (TTR)."""
        result = parser.parse(sample_prompts["medium"])
        assert 0 <= result.vocabulary_richness <= 1
    
    def test_lexical_density_calculated(self, parser, sample_prompts):
        """Test lexical density calculation."""
        result = parser.parse(sample_prompts["medium"])
        assert 0 <= result.lexical_density <= 1


class TestParserEmbeddings:
    """Test parser with embeddings."""
    
    @pytest.mark.slow
    def test_embedding_generation(self, parser_with_embeddings, sample_prompts):
        """Test that embeddings are generated when enabled."""
        result = parser_with_embeddings.parse(sample_prompts["simple"])
        if result.embedding is not None:
            assert isinstance(result.embedding, np.ndarray)
            assert result.embedding.shape[0] > 0
    
    @pytest.mark.slow
    def test_compute_similarity(self, parser_with_embeddings, sample_prompts):
        """Test semantic similarity computation."""
        similarity = parser_with_embeddings.compute_similarity(
            sample_prompts["simple"],
            sample_prompts["simple"]
        )
        assert 0.9 <= similarity <= 1.0  # Same text should have high similarity
    
    @pytest.mark.slow
    def test_similarity_different_texts(self, parser_with_embeddings, sample_prompts):
        """Test similarity for different texts."""
        similarity = parser_with_embeddings.compute_similarity(
            sample_prompts["simple"],
            sample_prompts["technical"]
        )
        assert 0 <= similarity < 0.9  # Different texts should have lower similarity


class TestParserConvenienceFunctions:
    """Test convenience functions in parser module."""
    
    def test_parse_prompt_function(self, sample_prompts):
        """Test parse_prompt() convenience function."""
        from nlp.parser import parse_prompt
        result = parse_prompt(sample_prompts["simple"])
        assert result.token_count > 0
    
    def test_extract_features_dict(self, sample_prompts):
        """Test extract_features_dict() convenience function."""
        from nlp.parser import extract_features_dict
        features = extract_features_dict(sample_prompts["simple"])
        assert isinstance(features, dict)
        assert "token_count" in features


# ============================================================================
# COMPLEXITY SCORE TESTS
# ============================================================================

class TestComplexityScorer:
    """Test ComplexityScorer class."""
    
    def test_scorer_initialization(self, complexity_scorer):
        """Test scorer initializes correctly."""
        assert complexity_scorer is not None
        assert len(complexity_scorer.weights) > 0
    
    def test_calculate_returns_breakdown(self, complexity_scorer, sample_prompts):
        """Test that calculate returns ComplexityBreakdown."""
        result = complexity_scorer.calculate(sample_prompts["medium"])
        assert hasattr(result, "overall_score")
        assert hasattr(result, "sentence_complexity")
        assert hasattr(result, "vocabulary_complexity")
        assert hasattr(result, "level")
    
    def test_score_in_valid_range(self, complexity_scorer, sample_prompts):
        """Test that scores are in valid range [0, 1]."""
        for prompt_name, prompt in sample_prompts.items():
            if prompt.strip():  # Skip empty prompts
                result = complexity_scorer.calculate(prompt)
                assert 0 <= result.overall_score <= 1, f"Failed for {prompt_name}"
    
    def test_simple_prompt_lower_score(self, complexity_scorer, sample_prompts):
        """Test that simple prompts have lower scores than complex ones."""
        simple_score = complexity_scorer.get_score(sample_prompts["simple"])
        complex_score = complexity_scorer.get_score(sample_prompts["complex"])
        assert simple_score < complex_score
    
    def test_complexity_levels(self, complexity_scorer, sample_prompts):
        """Test complexity level classification."""
        result = complexity_scorer.calculate(sample_prompts["simple"])
        assert result.level in ["low", "medium", "high", "very_high"]
    
    def test_energy_impact_description(self, complexity_scorer, sample_prompts):
        """Test energy impact description."""
        result = complexity_scorer.calculate(sample_prompts["medium"])
        assert result.energy_impact is not None
        assert len(result.energy_impact) > 0
    
    def test_sentence_complexity_component(self, complexity_scorer):
        """Test sentence complexity calculation."""
        # Long sentences should have higher complexity
        short = "Hello world."
        long = "This is a very long sentence that contains many different clauses, including subordinate clauses, and it goes on for quite a while."
        
        short_score = complexity_scorer.calculate_sentence_complexity(short)
        long_score = complexity_scorer.calculate_sentence_complexity(long)
        assert short_score < long_score
    
    def test_vocabulary_complexity_component(self, complexity_scorer):
        """Test vocabulary complexity calculation."""
        simple = "The cat sat on the mat."
        technical = "The neural network architecture utilizes backpropagation."
        
        simple_score = complexity_scorer.calculate_vocabulary_complexity(simple)
        technical_score = complexity_scorer.calculate_vocabulary_complexity(technical)
        assert simple_score < technical_score
    
    def test_breakdown_to_dict(self, complexity_scorer, sample_prompts):
        """Test ComplexityBreakdown.to_dict() method."""
        result = complexity_scorer.calculate(sample_prompts["medium"])
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "overall_score" in result_dict


class TestComplexityConvenienceFunctions:
    """Test convenience functions in complexity_score module."""
    
    def test_compute_complexity_function(self, sample_prompts):
        """Test compute_complexity() convenience function."""
        from nlp.complexity_score import compute_complexity
        score = compute_complexity(sample_prompts["medium"])
        assert 0 <= score <= 1
    
    def test_get_complexity_breakdown_function(self, sample_prompts):
        """Test get_complexity_breakdown() convenience function."""
        from nlp.complexity_score import get_complexity_breakdown
        breakdown = get_complexity_breakdown(sample_prompts["medium"])
        assert isinstance(breakdown, dict)
        assert "overall_score" in breakdown


# ============================================================================
# SIMPLIFIER TESTS
# ============================================================================

class TestTextSimplifier:
    """Test TextSimplifier class."""
    
    def test_simplifier_initialization(self, text_simplifier):
        """Test simplifier initializes correctly."""
        assert text_simplifier is not None
        assert text_simplifier.min_similarity > 0
    
    def test_simplify_verbose_text(self, text_simplifier, sample_prompts):
        """Test simplification of verbose text."""
        result = text_simplifier.simplify(sample_prompts["verbose"], strategy="auto")
        assert len(result.simplified) <= len(sample_prompts["verbose"])
    
    def test_simplify_returns_simplified_prompt(self, text_simplifier, sample_prompts):
        """Test that simplify returns SimplifiedPrompt."""
        result = text_simplifier.simplify(sample_prompts["verbose"])
        assert hasattr(result, "simplified")
        assert hasattr(result, "original")
        assert hasattr(result, "token_reduction_percent")
    
    def test_remove_verbose_phrases(self, text_simplifier):
        """Test verbose phrase removal."""
        text = "In order to help you, due to the fact that you asked."
        result = text_simplifier.remove_verbose_phrases(text)
        assert "in order to" not in result.lower()
        assert "due to the fact that" not in result.lower()
    
    def test_remove_filler_words(self, text_simplifier):
        """Test filler word removal."""
        text = "This is basically really very extremely good."
        result = text_simplifier.remove_filler_words(text)
        assert "basically" not in result.lower()
        assert "extremely" not in result.lower()
    
    def test_remove_redundancy(self, text_simplifier):
        """Test redundancy removal."""
        text = "The past history shows the end result."
        result = text_simplifier.remove_redundancy(text)
        assert "past history" not in result.lower()
    
    def test_compress_sentences(self, text_simplifier):
        """Test sentence compression."""
        text = "Hello (this is a parenthetical), how are you?"
        result = text_simplifier.compress_sentences(text)
        assert "parenthetical" not in result
    
    def test_truncate_to_essential(self, text_simplifier):
        """Test truncation."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        result = text_simplifier.truncate_to_essential(text, max_sentences=2)
        # Should reduce the number of sentences
        assert len(result) < len(text)
    
    def test_simplification_strategies(self, text_simplifier, sample_prompts):
        """Test different simplification strategies."""
        strategies = ["verbose", "filler", "compress", "truncate"]
        for strategy in strategies:
            result = text_simplifier.simplify(sample_prompts["verbose"], strategy=strategy)
            assert result.strategy_used == strategy
    
    def test_token_reduction_calculated(self, text_simplifier, sample_prompts):
        """Test token reduction calculation."""
        result = text_simplifier.simplify(sample_prompts["verbose"])
        assert result.token_reduction_percent >= 0
    
    def test_energy_reduction_estimated(self, text_simplifier, sample_prompts):
        """Test energy reduction estimation."""
        result = text_simplifier.simplify(sample_prompts["verbose"])
        assert result.estimated_energy_reduction_percent >= 0
    
    def test_semantic_similarity_calculated(self, text_simplifier, sample_prompts):
        """Test semantic similarity calculation."""
        result = text_simplifier.simplify(sample_prompts["verbose"])
        assert 0 <= result.semantic_similarity <= 1
    
    def test_get_all_alternatives(self, text_simplifier, sample_prompts):
        """Test getting all alternatives."""
        alternatives = text_simplifier.get_all_alternatives(sample_prompts["verbose"])
        assert isinstance(alternatives, list)
        # May be empty if no valid alternatives found
    
    def test_simplify_to_dict(self, text_simplifier, sample_prompts):
        """Test SimplifiedPrompt.to_dict() method."""
        result = text_simplifier.simplify(sample_prompts["verbose"])
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "simplified" in result_dict


class TestSimplifierConvenienceFunctions:
    """Test convenience functions in simplifier module."""
    
    def test_simplify_prompt_function(self, sample_prompts):
        """Test simplify_prompt() convenience function."""
        from nlp.simplifier import simplify_prompt
        result = simplify_prompt(sample_prompts["verbose"])
        assert isinstance(result, str)
    
    def test_get_efficient_alternatives_function(self, sample_prompts):
        """Test get_efficient_alternatives() convenience function."""
        from nlp.simplifier import get_efficient_alternatives
        alternatives = get_efficient_alternatives(sample_prompts["verbose"])
        assert isinstance(alternatives, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
