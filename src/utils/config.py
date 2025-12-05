"""
Configuration settings for the Sustainable AI Energy-Efficient Prompt Engineering Application.
This module contains all application-wide settings, paths, and constants.

Updated (December 2025):
- Model type changed from random_forest to gradient_boost (R²=0.976)
- Model paths updated to model/energy_predictor.pkl, feature_scaler.pkl
- Added feature_names_path for loading trained feature order

Key Configuration Classes:
- EnergyPredictorConfig: ML model settings (Gradient Boosting, 12 features)
- AnomalyDetectorConfig: Isolation Forest for outlier detection
- NLPConfig: Tokenizer and embedding model settings
- PromptOptimizerConfig: Simplification strategies and thresholds
- AppConfig: Streamlit GUI settings
- DataCenterConfig: Energy cost and carbon intensity by region
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============================================================================
# BASE PATHS
# ============================================================================

# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model subdirectories
ENERGY_PREDICTOR_DIR = MODEL_DIR / "energy_predictor"
ANOMALY_DETECTOR_DIR = MODEL_DIR / "anomaly_detector"
NLP_TRANSFORMER_DIR = MODEL_DIR / "nlp_transformer"
PROMPT_OPTIMIZER_DIR = MODEL_DIR / "prompt_optimizer"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, SYNTHETIC_DATA_DIR, PROCESSED_DATA_DIR,
                 ENERGY_PREDICTOR_DIR, ANOMALY_DETECTOR_DIR, 
                 NLP_TRANSFORMER_DIR, PROMPT_OPTIMIZER_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

@dataclass
class DatabaseConfig:
    """Configuration for SQLite database used for logging and transparency reports."""
    db_path: Path = DATA_DIR / "energy_logs.db"
    log_table: str = "energy_logs"
    report_table: str = "transparency_reports"
    benchmark_table: str = "benchmark_results"

DATABASE_CONFIG = DatabaseConfig()

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class EnergyPredictorConfig:
    """Configuration for the energy prediction model (Supervised Learning)."""
    model_type: str = "gradient_boost"  # Options: "random_forest", "neural_network", "gradient_boost"
    model_path: Path = MODEL_DIR / "energy_predictor.pkl"
    scaler_path: Path = MODEL_DIR / "feature_scaler.pkl"
    feature_names_path: Path = MODEL_DIR / "feature_names.pkl"
    
    # Random Forest hyperparameters
    n_estimators: int = 100
    max_depth: int = 15
    min_samples_split: int = 5
    random_state: int = 42
    
    # Neural Network hyperparameters (if using NN)
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    
    # Feature columns used for prediction
    feature_columns: List[str] = field(default_factory=lambda: [
        "token_count", "char_count", "punct_ratio", "avg_word_length",
        "stopword_ratio", "num_layers", "training_hours", "flops_per_hour",
        "flops_per_layer", "training_efficiency", "complexity_score"
    ])
    target_column: str = "energy_kwh"

ENERGY_PREDICTOR_CONFIG = EnergyPredictorConfig()

@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detection model (Unsupervised Learning)."""
    model_type: str = "isolation_forest"  # Options: "isolation_forest", "one_class_svm", "local_outlier_factor"
    model_path: Path = ANOMALY_DETECTOR_DIR / "anomaly_model.joblib"
    
    # Isolation Forest hyperparameters
    contamination: float = 0.1  # Expected proportion of outliers
    n_estimators: int = 100
    max_samples: str = "auto"
    random_state: int = 42
    
    # Anomaly threshold (prompts with score below this are flagged)
    anomaly_threshold: float = -0.5
    
    # Features used for anomaly detection
    feature_columns: List[str] = field(default_factory=lambda: [
        "token_count", "complexity_score", "flops_per_hour", "energy_kwh"
    ])

ANOMALY_DETECTOR_CONFIG = AnomalyDetectorConfig()

@dataclass
class NLPConfig:
    """Configuration for NLP processing and embeddings."""
    tokenizer_model: str = "distilbert-base-uncased"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_token_length: int = 512
    
    # Complexity scoring weights
    complexity_weights: Dict[str, float] = field(default_factory=lambda: {
        "sentence_length": 0.25,
        "vocabulary_richness": 0.25,
        "syntactic_complexity": 0.20,
        "semantic_density": 0.30
    })
    
    # Stopwords language
    stopwords_language: str = "english"

NLP_CONFIG = NLPConfig()

@dataclass
class PromptOptimizerConfig:
    """Configuration for prompt optimization and recommendation."""
    paraphrase_model: str = "t5-small"  # Options: "t5-small", "t5-base", "facebook/bart-large-paraphrase"
    max_alternatives: int = 5
    min_similarity_threshold: float = 0.75  # Semantic similarity threshold
    max_energy_reduction_target: float = 0.30  # Target 30% energy reduction
    
    # Optimization strategies
    strategies: List[str] = field(default_factory=lambda: [
        "simplify",      # Reduce complexity while maintaining meaning
        "truncate",      # Remove unnecessary verbose parts
        "paraphrase",    # Generate alternative phrasings
        "compress"       # Compress information density
    ])

PROMPT_OPTIMIZER_CONFIG = PromptOptimizerConfig()

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

@dataclass
class AppConfig:
    """Main application configuration."""
    app_name: str = "Sustainable AI - Energy Efficient Prompt Engineering"
    version: str = "1.0.0"
    
    # Streamlit page config
    page_title: str = "🌱 Sustainable AI Energy Monitor"
    page_icon: str = "🌱"
    layout: str = "wide"
    
    # Default LLM parameters
    default_num_layers: int = 24
    default_training_hours: float = 8.0
    default_flops_per_hour: float = 1e11
    
    # Energy reporting thresholds (kWh)
    energy_low_threshold: float = 0.5
    energy_medium_threshold: float = 1.0
    energy_high_threshold: float = 2.0
    
    # Carbon footprint conversion (kg CO2 per kWh)
    # Based on average US grid emission factor
    carbon_per_kwh: float = 0.42
    
    # Water usage conversion (liters per kWh for cooling)
    water_per_kwh: float = 1.8
    
    # EU Reporting compliance date
    eu_reporting_deadline: str = "2026-08-01"

APP_CONFIG = AppConfig()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

@dataclass
class LoggingConfig:
    """Configuration for application logging."""
    log_dir: Path = PROJECT_ROOT / "logs"
    log_file: str = "sustainable_ai.log"
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_log_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    
    def __post_init__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)

LOGGING_CONFIG = LoggingConfig()

# ============================================================================
# DATA CENTER SIMULATION CONFIGURATION
# ============================================================================

@dataclass
class DataCenterConfig:
    """Configuration for data center energy simulation."""
    # Power Usage Effectiveness (PUE) - typical range 1.1-2.0
    pue: float = 1.4
    
    # GPU power consumption (Watts)
    gpu_power_watts: float = 400  # NVIDIA A100 typical
    
    # Memory power (Watts per GB)
    memory_power_per_gb: float = 0.3
    
    # Network power (Watts)
    network_power_watts: float = 50
    
    # Cooling efficiency factor
    cooling_efficiency: float = 0.85
    
    # Regional electricity costs ($/kWh)
    electricity_costs: Dict[str, float] = field(default_factory=lambda: {
        "california": 0.22,
        "texas": 0.12,
        "virginia": 0.11,
        "eu_average": 0.25,
        "canada_average": 0.10
    })
    
    # Carbon intensity by region (kg CO2/kWh)
    carbon_intensity: Dict[str, float] = field(default_factory=lambda: {
        "california": 0.21,
        "texas": 0.45,
        "virginia": 0.35,
        "eu_average": 0.28,
        "canada_average": 0.12
    })

DATA_CENTER_CONFIG = DataCenterConfig()

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

def get_all_configs() -> Dict:
    """Return all configurations as a dictionary."""
    return {
        "app": APP_CONFIG,
        "database": DATABASE_CONFIG,
        "energy_predictor": ENERGY_PREDICTOR_CONFIG,
        "anomaly_detector": ANOMALY_DETECTOR_CONFIG,
        "nlp": NLP_CONFIG,
        "prompt_optimizer": PROMPT_OPTIMIZER_CONFIG,
        "logging": LOGGING_CONFIG,
        "data_center": DATA_CENTER_CONFIG
    }

def get_paths() -> Dict[str, Path]:
    """Return all important paths as a dictionary."""
    return {
        "project_root": PROJECT_ROOT,
        "src": SRC_DIR,
        "data": DATA_DIR,
        "model": MODEL_DIR,
        "reports": REPORTS_DIR,
        "raw_data": RAW_DATA_DIR,
        "synthetic_data": SYNTHETIC_DATA_DIR,
        "processed_data": PROCESSED_DATA_DIR
    }
