"""
NLP Parser module for extracting linguistic features from prompts.
This module handles tokenization, feature extraction, and semantic embedding generation.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Lazy imports for heavy libraries
_tokenizer = None
_sentence_transformer = None


def get_tokenizer():
    """Lazy load the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return _tokenizer


def get_sentence_transformer():
    """Lazy load the sentence transformer for embeddings."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            _sentence_transformer = None
    return _sentence_transformer


@dataclass
class ParsedPrompt:
    """
    Data class containing all parsed features from a prompt.
    """
    original_text: str
    cleaned_text: str
    
    # Basic counts
    token_count: int
    word_count: int
    char_count: int
    sentence_count: int
    
    # Linguistic features
    avg_word_length: float
    avg_sentence_length: float
    punct_ratio: float
    stopword_ratio: float
    unique_word_ratio: float
    
    # POS tag distribution
    noun_ratio: float
    verb_ratio: float
    adj_ratio: float
    adv_ratio: float
    
    # Advanced features
    vocabulary_richness: float  # Type-Token Ratio
    lexical_density: float
    
    # Embedding (optional)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for processing."""
        return {
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "token_count": self.token_count,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "sentence_count": self.sentence_count,
            "avg_word_length": self.avg_word_length,
            "avg_sentence_length": self.avg_sentence_length,
            "punct_ratio": self.punct_ratio,
            "stopword_ratio": self.stopword_ratio,
            "unique_word_ratio": self.unique_word_ratio,
            "noun_ratio": self.noun_ratio,
            "verb_ratio": self.verb_ratio,
            "adj_ratio": self.adj_ratio,
            "adv_ratio": self.adv_ratio,
            "vocabulary_richness": self.vocabulary_richness,
            "lexical_density": self.lexical_density
        }
    
    def get_feature_vector(self) -> List[float]:
        """Get numeric feature vector for ML models."""
        return [
            self.token_count,
            self.word_count,
            self.char_count,
            self.sentence_count,
            self.avg_word_length,
            self.avg_sentence_length,
            self.punct_ratio,
            self.stopword_ratio,
            self.unique_word_ratio,
            self.noun_ratio,
            self.verb_ratio,
            self.adj_ratio,
            self.adv_ratio,
            self.vocabulary_richness,
            self.lexical_density
        ]


class PromptParser:
    """
    Main parser class for extracting features from text prompts.
    """
    
    def __init__(self, language: str = "english", use_embeddings: bool = True):
        """
        Initialize the prompt parser.
        
        Args:
            language: Language for stopwords
            use_embeddings: Whether to generate semantic embeddings
        """
        self.language = language
        self.use_embeddings = use_embeddings
        self.stop_words = set(stopwords.words(language))
        
        # Punctuation characters
        self.punctuation = set(".,!?;:\"'()-[]{}...")
        
        # POS tag mappings
        self.noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
        self.verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        self.adj_tags = {'JJ', 'JJR', 'JJS'}
        self.adv_tags = {'RB', 'RBR', 'RBS'}
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text.
        
        Args:
            text: Raw input text
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    def get_token_count(self, text: str) -> int:
        """
        Get token count using transformer tokenizer.
        
        Args:
            text: Input text
        
        Returns:
            Number of tokens
        """
        try:
            tokenizer = get_tokenizer()
            return len(tokenizer.encode(text, truncation=True, max_length=512))
        except Exception:
            # Fallback to word count
            return len(text.split())
    
    def get_pos_distribution(self, words: List[str]) -> Tuple[float, float, float, float]:
        """
        Get POS tag distribution.
        
        Args:
            words: List of words
        
        Returns:
            Tuple of (noun_ratio, verb_ratio, adj_ratio, adv_ratio)
        """
        if not words:
            return (0.0, 0.0, 0.0, 0.0)
        
        try:
            tagged = pos_tag(words)
            total = len(tagged)
            
            nouns = sum(1 for _, tag in tagged if tag in self.noun_tags)
            verbs = sum(1 for _, tag in tagged if tag in self.verb_tags)
            adjs = sum(1 for _, tag in tagged if tag in self.adj_tags)
            advs = sum(1 for _, tag in tagged if tag in self.adv_tags)
            
            return (
                nouns / total,
                verbs / total,
                adjs / total,
                advs / total
            )
        except Exception:
            return (0.0, 0.0, 0.0, 0.0)
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate semantic embedding for the text.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector or None
        """
        if not self.use_embeddings:
            return None
        
        try:
            model = get_sentence_transformer()
            if model is not None:
                return model.encode(text)
        except Exception:
            pass
        
        return None
    
    def parse(self, text: str) -> ParsedPrompt:
        """
        Parse a prompt and extract all features.
        
        Args:
            text: Input prompt text
        
        Returns:
            ParsedPrompt object with all extracted features
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Basic tokenization
        try:
            words = word_tokenize(cleaned.lower())
            sentences = sent_tokenize(cleaned)
        except Exception:
            words = cleaned.lower().split()
            sentences = [cleaned]
        
        # Filter to just words (remove punctuation tokens)
        word_tokens = [w for w in words if w.isalnum()]
        
        # Basic counts
        char_count = len(cleaned)
        word_count = len(word_tokens)
        sentence_count = max(len(sentences), 1)
        token_count = self.get_token_count(cleaned)
        
        # Calculate averages
        avg_word_length = (
            sum(len(w) for w in word_tokens) / word_count 
            if word_count > 0 else 0
        )
        avg_sentence_length = word_count / sentence_count
        
        # Ratios
        punct_count = sum(1 for c in cleaned if c in self.punctuation)
        punct_ratio = punct_count / char_count if char_count > 0 else 0
        
        stopword_count = sum(1 for w in word_tokens if w in self.stop_words)
        stopword_ratio = stopword_count / word_count if word_count > 0 else 0
        
        unique_words = set(word_tokens)
        unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0
        
        # POS distribution
        noun_ratio, verb_ratio, adj_ratio, adv_ratio = self.get_pos_distribution(word_tokens)
        
        # Vocabulary richness (Type-Token Ratio)
        vocabulary_richness = len(unique_words) / word_count if word_count > 0 else 0
        
        # Lexical density (content words / total words)
        content_words = sum(1 for w in word_tokens if w not in self.stop_words)
        lexical_density = content_words / word_count if word_count > 0 else 0
        
        # Get embedding
        embedding = self.get_embedding(cleaned) if self.use_embeddings else None
        
        return ParsedPrompt(
            original_text=text,
            cleaned_text=cleaned,
            token_count=token_count,
            word_count=word_count,
            char_count=char_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            punct_ratio=punct_ratio,
            stopword_ratio=stopword_ratio,
            unique_word_ratio=unique_word_ratio,
            noun_ratio=noun_ratio,
            verb_ratio=verb_ratio,
            adj_ratio=adj_ratio,
            adv_ratio=adv_ratio,
            vocabulary_richness=vocabulary_richness,
            lexical_density=lexical_density,
            embedding=embedding
        )
    
    def parse_batch(self, texts: List[str]) -> List[ParsedPrompt]:
        """
        Parse multiple prompts.
        
        Args:
            texts: List of prompt texts
        
        Returns:
            List of ParsedPrompt objects
        """
        return [self.parse(text) for text in texts]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            # Fallback to Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 > 0 and norm2 > 0:
            return float(dot_product / (norm1 * norm2))
        return 0.0


# Convenience function for quick parsing
def parse_prompt(text: str, use_embeddings: bool = False) -> ParsedPrompt:
    """
    Quick function to parse a single prompt.
    
    Args:
        text: Input prompt text
        use_embeddings: Whether to generate embeddings
    
    Returns:
        ParsedPrompt object
    """
    parser = PromptParser(use_embeddings=use_embeddings)
    return parser.parse(text)


def extract_features_dict(text: str) -> Dict[str, Any]:
    """
    Extract features as a dictionary (for ML pipelines).
    
    Args:
        text: Input prompt text
    
    Returns:
        Dictionary of features
    """
    parsed = parse_prompt(text, use_embeddings=False)
    return parsed.to_dict()


if __name__ == "__main__":
    # Test the parser
    test_prompt = """
    Explain the concept of machine learning and how neural networks 
    can be used to solve complex pattern recognition problems in 
    medical imaging applications.
    """
    
    parser = PromptParser(use_embeddings=False)
    result = parser.parse(test_prompt)
    
    print("Parsed Prompt Features:")
    print("-" * 50)
    for key, value in result.to_dict().items():
        if key not in ['original_text', 'cleaned_text']:
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
