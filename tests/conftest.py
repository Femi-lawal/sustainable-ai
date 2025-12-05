"""
Pytest Configuration and Fixtures for Sustainable AI Tests.
This file contains shared fixtures and configuration for all tests.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import shutil

# Add source directories to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_prompts():
    """Collection of sample prompts for testing."""
    return {
        "simple": "Hello, how are you?",
        "medium": "Explain the concept of machine learning and how it can be used in real-world applications.",
        "complex": """Design a comprehensive distributed system architecture for processing 
        real-time data streams at scale, including considerations for fault tolerance, 
        horizontal scaling, and data consistency guarantees. Include code examples in 
        Python and explain the trade-offs between different architectural choices.""",
        "verbose": """In order to provide you with the most comprehensive and detailed 
        explanation possible, I would like to basically take into consideration all 
        of the various different factors that are involved in this matter.""",
        "technical": """Analyze the multi-layer perceptron architecture, discussing how 
        backpropagation enables gradient-based optimization of neural network parameters 
        through the chain rule of calculus.""",
        "empty": "",
        "whitespace": "   ",
        "single_word": "Hello",
        "long": "word " * 500,  # Very long prompt
        "unicode": "Hello, 你好, مرحبا, Привет!",
        "special_chars": "Hello! @#$%^&*() How are you?"
    }


@pytest.fixture
def sample_parsed_features():
    """Sample parsed features for testing ML models."""
    return {
        "token_count": 50,
        "word_count": 45,
        "char_count": 250,
        "sentence_count": 3,
        "avg_word_length": 5.5,
        "avg_sentence_length": 15.0,
        "punct_ratio": 0.04,
        "stopword_ratio": 0.35,
        "unique_word_ratio": 0.8,
        "vocabulary_richness": 0.8,
        "lexical_density": 0.65,
        "noun_ratio": 0.25,
        "verb_ratio": 0.15,
        "adj_ratio": 0.10,
        "adv_ratio": 0.05
    }


@pytest.fixture
def sample_energy_features():
    """Sample features for energy prediction."""
    return {
        "token_count": 100,
        "char_count": 500,
        "punct_ratio": 0.04,
        "avg_word_length": 5.0,
        "stopword_ratio": 0.3,
        "num_layers": 24,
        "training_hours": 8.0,
        "flops_per_hour": 1e11,
        "flops_per_layer": 1e11 / 24,
        "training_efficiency": 8.0 / 24,
        "complexity_score": 0.5
    }


@pytest.fixture
def sample_training_data():
    """
    Sample training data for ML models.
    
    Generates realistic data with proper correlations between features
    and energy consumption. This mimics the actual training data structure.
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_samples = 200  # Enough samples for meaningful training
    
    # Generate features with realistic distributions
    token_count = np.random.lognormal(mean=3.5, sigma=0.8, size=n_samples).astype(int)
    token_count = np.clip(token_count, 5, 1000)
    
    word_count = (token_count * 0.85 + np.random.normal(0, 5, n_samples)).astype(int)
    word_count = np.clip(word_count, 3, 900)
    
    char_count = word_count * np.random.uniform(4.5, 6.5, n_samples)
    char_count = char_count.astype(int)
    
    complexity_score = np.random.beta(2, 5, n_samples)  # Right-skewed
    
    avg_word_length = np.random.normal(5.0, 1.0, n_samples)
    avg_word_length = np.clip(avg_word_length, 3, 10)
    
    avg_sentence_length = np.random.normal(15, 5, n_samples)
    avg_sentence_length = np.clip(avg_sentence_length, 5, 50)
    
    # Model parameters
    num_layers = np.random.choice([12, 24, 48, 96], n_samples)
    training_hours = np.random.uniform(1, 24, n_samples)
    flops_per_hour = np.random.uniform(1e10, 1e12, n_samples)
    
    # Generate energy with STRONG correlation to features
    # This mimics realistic energy consumption patterns
    base_energy = 0.05
    energy_kwh = (
        base_energy +
        0.002 * token_count +           # Primary driver (strong)
        0.001 * word_count +            # Secondary driver
        0.0001 * char_count +           # Tertiary driver
        0.3 * complexity_score +        # Complexity adds energy
        0.005 * avg_sentence_length +   # Longer sentences = more processing
        0.0005 * num_layers             # More layers = more compute
    )
    
    # Add small noise (but keep strong correlation)
    noise = np.random.normal(0, 0.02, n_samples)
    energy_kwh = energy_kwh + noise
    energy_kwh = np.clip(energy_kwh, 0.01, 5.0)
    
    data = {
        "token_count": token_count,
        "word_count": word_count,
        "char_count": char_count,
        "complexity_score": complexity_score,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "punct_ratio": np.random.uniform(0.01, 0.1, n_samples),
        "stopword_ratio": np.random.uniform(0.2, 0.5, n_samples),
        "num_layers": num_layers,
        "training_hours": training_hours,
        "flops_per_hour": flops_per_hour,
        "energy_kwh": energy_kwh
    }
    
    return pd.DataFrame(data)


# ============================================================================
# TEMPORARY DIRECTORY FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def temp_db_path(temp_dir):
    """Create a temporary database path."""
    return temp_dir / "test_energy_logs.db"


@pytest.fixture
def temp_model_dir(temp_dir):
    """Create a temporary model directory."""
    model_dir = temp_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


# ============================================================================
# MODULE FIXTURES
# ============================================================================

@pytest.fixture
def parser():
    """Create a PromptParser instance."""
    from nlp.parser import PromptParser
    return PromptParser(use_embeddings=False)


@pytest.fixture
def parser_with_embeddings():
    """Create a PromptParser with embeddings enabled."""
    from nlp.parser import PromptParser
    return PromptParser(use_embeddings=True)


@pytest.fixture
def complexity_scorer():
    """Create a ComplexityScorer instance."""
    from nlp.complexity_score import ComplexityScorer
    return ComplexityScorer()


@pytest.fixture
def text_simplifier():
    """Create a TextSimplifier instance."""
    from nlp.simplifier import TextSimplifier
    return TextSimplifier(min_similarity_threshold=0.6)


@pytest.fixture
def energy_predictor(temp_model_dir):
    """Create an EnergyPredictor instance with temp storage."""
    from prediction.estimator import EnergyPredictor
    predictor = EnergyPredictor(model_type="random_forest")
    predictor.model_path = temp_model_dir / "test_model.joblib"
    predictor.scaler_path = temp_model_dir / "test_scaler.joblib"
    return predictor


@pytest.fixture
def anomaly_detector(temp_model_dir):
    """Create an AnomalyDetector instance with temp storage."""
    from anomaly.detector import AnomalyDetector
    detector = AnomalyDetector(model_type="isolation_forest")
    detector.model_path = temp_model_dir / "test_anomaly_model.joblib"
    return detector


@pytest.fixture
def prompt_optimizer():
    """Create a PromptOptimizer instance."""
    from optimization.recomender import PromptOptimizer
    return PromptOptimizer(min_similarity=0.6)


@pytest.fixture
def database_manager(temp_db_path):
    """Create a DatabaseManager with temp database."""
    from utils.database import DatabaseManager
    return DatabaseManager(db_path=temp_db_path)


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def model_config():
    """Standard model configuration for tests."""
    return {
        "num_layers": 24,
        "training_hours": 8.0,
        "flops_per_hour": 1e11,
        "region": "california"
    }


@pytest.fixture
def date_range():
    """Standard date range for report tests."""
    return {
        "start": datetime.now() - timedelta(days=7),
        "end": datetime.now() + timedelta(hours=1)  # Add buffer for logged data
    }


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests that validate model performance metrics"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark slow tests."""
    for item in items:
        # Mark tests with 'slow' in the name
        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        # Mark e2e tests
        if "e2e" in item.nodeid.lower():
            item.add_marker(pytest.mark.e2e)
