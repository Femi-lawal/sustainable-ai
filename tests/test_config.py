"""
Unit Tests for the Configuration Module.
Tests all config dataclasses, paths, and configuration functions.
"""

import pytest
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPaths:
    """Test path configurations."""
    
    def test_project_root_exists(self):
        """Test that PROJECT_ROOT is valid."""
        from utils.config import PROJECT_ROOT
        assert PROJECT_ROOT.exists(), "PROJECT_ROOT should exist"
        assert PROJECT_ROOT.is_dir(), "PROJECT_ROOT should be a directory"
    
    def test_src_dir_exists(self):
        """Test that SRC_DIR is valid."""
        from utils.config import SRC_DIR
        assert SRC_DIR.exists(), "SRC_DIR should exist"
        assert (SRC_DIR / "nlp").exists(), "SRC_DIR should contain nlp module"
    
    def test_data_dir_creation(self):
        """Test that data directories are created."""
        from utils.config import DATA_DIR, RAW_DATA_DIR, SYNTHETIC_DATA_DIR
        assert DATA_DIR.exists(), "DATA_DIR should be created"
        assert RAW_DATA_DIR.exists(), "RAW_DATA_DIR should be created"
        assert SYNTHETIC_DATA_DIR.exists(), "SYNTHETIC_DATA_DIR should be created"
    
    def test_model_dir_creation(self):
        """Test that model directories are created."""
        from utils.config import MODEL_DIR, ENERGY_PREDICTOR_DIR, ANOMALY_DETECTOR_DIR
        assert MODEL_DIR.exists(), "MODEL_DIR should be created"
        assert ENERGY_PREDICTOR_DIR.exists(), "ENERGY_PREDICTOR_DIR should be created"
        assert ANOMALY_DETECTOR_DIR.exists(), "ANOMALY_DETECTOR_DIR should be created"


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""
    
    def test_database_config_defaults(self):
        """Test default DatabaseConfig values."""
        from utils.config import DATABASE_CONFIG
        assert DATABASE_CONFIG.db_path is not None
        assert DATABASE_CONFIG.log_table == "energy_logs"
        assert DATABASE_CONFIG.report_table == "transparency_reports"
    
    def test_database_path_is_path(self):
        """Test that db_path is a Path object."""
        from utils.config import DATABASE_CONFIG
        assert isinstance(DATABASE_CONFIG.db_path, Path)


class TestEnergyPredictorConfig:
    """Test EnergyPredictorConfig dataclass."""
    
    def test_energy_predictor_config_defaults(self):
        """Test default EnergyPredictorConfig values."""
        from utils.config import ENERGY_PREDICTOR_CONFIG
        assert ENERGY_PREDICTOR_CONFIG.model_type == "random_forest"
        assert ENERGY_PREDICTOR_CONFIG.n_estimators == 200
        assert ENERGY_PREDICTOR_CONFIG.random_state == 42
    
    def test_energy_predictor_feature_columns(self):
        """Test feature columns list (5 core features)."""
        from utils.config import ENERGY_PREDICTOR_CONFIG
        assert len(ENERGY_PREDICTOR_CONFIG.feature_columns) == 5
        assert "token_count" in ENERGY_PREDICTOR_CONFIG.feature_columns
        assert "word_count" in ENERGY_PREDICTOR_CONFIG.feature_columns
        assert "char_count" in ENERGY_PREDICTOR_CONFIG.feature_columns
    
    def test_energy_predictor_target_column(self):
        """Test target column."""
        from utils.config import ENERGY_PREDICTOR_CONFIG
        assert ENERGY_PREDICTOR_CONFIG.target_column == "energy_joules"
    
    def test_hidden_layers_is_list(self):
        """Test that hidden_layers is a list."""
        from utils.config import ENERGY_PREDICTOR_CONFIG
        assert isinstance(ENERGY_PREDICTOR_CONFIG.hidden_layers, list)
        assert len(ENERGY_PREDICTOR_CONFIG.hidden_layers) > 0


class TestAnomalyDetectorConfig:
    """Test AnomalyDetectorConfig dataclass."""
    
    def test_anomaly_detector_config_defaults(self):
        """Test default AnomalyDetectorConfig values."""
        from utils.config import ANOMALY_DETECTOR_CONFIG
        assert ANOMALY_DETECTOR_CONFIG.model_type == "isolation_forest"
        assert 0 < ANOMALY_DETECTOR_CONFIG.contamination < 1
        assert ANOMALY_DETECTOR_CONFIG.n_estimators > 0
    
    def test_anomaly_threshold(self):
        """Test anomaly threshold value."""
        from utils.config import ANOMALY_DETECTOR_CONFIG
        assert ANOMALY_DETECTOR_CONFIG.anomaly_threshold < 0


class TestNLPConfig:
    """Test NLPConfig dataclass."""
    
    def test_nlp_config_defaults(self):
        """Test default NLPConfig values."""
        from utils.config import NLP_CONFIG
        assert NLP_CONFIG.tokenizer_model is not None
        assert NLP_CONFIG.max_token_length == 512
        assert NLP_CONFIG.stopwords_language == "english"
    
    def test_complexity_weights_sum(self):
        """Test that complexity weights sum to approximately 1."""
        from utils.config import NLP_CONFIG
        total = sum(NLP_CONFIG.complexity_weights.values())
        assert 0.95 <= total <= 1.05, f"Complexity weights should sum to ~1, got {total}"


class TestPromptOptimizerConfig:
    """Test PromptOptimizerConfig dataclass."""
    
    def test_prompt_optimizer_config_defaults(self):
        """Test default PromptOptimizerConfig values."""
        from utils.config import PROMPT_OPTIMIZER_CONFIG
        assert PROMPT_OPTIMIZER_CONFIG.max_alternatives > 0
        assert 0 < PROMPT_OPTIMIZER_CONFIG.min_similarity_threshold < 1
        assert 0 < PROMPT_OPTIMIZER_CONFIG.max_energy_reduction_target < 1
    
    def test_optimization_strategies(self):
        """Test optimization strategies list."""
        from utils.config import PROMPT_OPTIMIZER_CONFIG
        strategies = PROMPT_OPTIMIZER_CONFIG.strategies
        assert len(strategies) > 0
        assert "simplify" in strategies


class TestAppConfig:
    """Test AppConfig dataclass."""
    
    def test_app_config_basics(self):
        """Test basic AppConfig values."""
        from utils.config import APP_CONFIG
        assert APP_CONFIG.app_name is not None
        assert APP_CONFIG.version is not None
        assert APP_CONFIG.page_title is not None
    
    def test_energy_thresholds(self):
        """Test energy threshold values."""
        from utils.config import APP_CONFIG
        assert APP_CONFIG.energy_low_threshold < APP_CONFIG.energy_medium_threshold
        assert APP_CONFIG.energy_medium_threshold < APP_CONFIG.energy_high_threshold
    
    def test_carbon_conversion_factor(self):
        """Test carbon conversion factor is reasonable."""
        from utils.config import APP_CONFIG
        assert 0.1 < APP_CONFIG.carbon_per_kwh < 1.0
    
    def test_eu_reporting_deadline(self):
        """Test EU reporting deadline format."""
        from utils.config import APP_CONFIG
        from datetime import datetime
        # Should be parseable as a date
        deadline = datetime.strptime(APP_CONFIG.eu_reporting_deadline, "%Y-%m-%d")
        assert deadline is not None


class TestLoggingConfig:
    """Test LoggingConfig dataclass."""
    
    def test_logging_config_defaults(self):
        """Test default LoggingConfig values."""
        from utils.config import LOGGING_CONFIG
        assert LOGGING_CONFIG.log_file is not None
        assert LOGGING_CONFIG.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert LOGGING_CONFIG.max_log_size > 0
    
    def test_log_dir_created(self):
        """Test that log directory is created."""
        from utils.config import LOGGING_CONFIG
        assert LOGGING_CONFIG.log_dir.exists()


class TestDataCenterConfig:
    """Test DataCenterConfig dataclass."""
    
    def test_data_center_config_defaults(self):
        """Test default DataCenterConfig values."""
        from utils.config import DATA_CENTER_CONFIG
        assert 1.0 < DATA_CENTER_CONFIG.pue < 3.0  # Typical PUE range
        assert DATA_CENTER_CONFIG.gpu_power_watts > 0
    
    def test_electricity_costs(self):
        """Test electricity costs dictionary."""
        from utils.config import DATA_CENTER_CONFIG
        assert "california" in DATA_CENTER_CONFIG.electricity_costs
        assert all(cost > 0 for cost in DATA_CENTER_CONFIG.electricity_costs.values())
    
    def test_carbon_intensity(self):
        """Test carbon intensity dictionary."""
        from utils.config import DATA_CENTER_CONFIG
        assert "california" in DATA_CENTER_CONFIG.carbon_intensity
        assert all(0 < intensity < 2 for intensity in DATA_CENTER_CONFIG.carbon_intensity.values())


class TestConfigFunctions:
    """Test configuration helper functions."""
    
    def test_get_all_configs(self):
        """Test get_all_configs function."""
        from utils.config import get_all_configs
        configs = get_all_configs()
        assert "app" in configs
        assert "database" in configs
        assert "energy_predictor" in configs
        assert "anomaly_detector" in configs
    
    def test_get_paths(self):
        """Test get_paths function."""
        from utils.config import get_paths
        paths = get_paths()
        assert "project_root" in paths
        assert "src" in paths
        assert "data" in paths
        assert "model" in paths
        assert all(isinstance(p, Path) for p in paths.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
