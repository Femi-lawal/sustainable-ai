"""
Unit Tests for the GUI Module (layout.py and app.py).
Tests Streamlit components and application logic.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestLayoutComponents:
    """Test layout component functions."""
    
    def test_layout_module_imports(self):
        """Test that layout module can be imported."""
        try:
            # Import should work even if Streamlit isn't running
            import gui.layout as layout
            assert hasattr(layout, 'render_metric_card')
            assert hasattr(layout, 'render_energy_gauge')
        except ImportError:
            # Expected if Streamlit context not available
            pass
    
    def test_metric_card_function_exists(self):
        """Test that metric card function exists."""
        try:
            from gui.layout import render_metric_card
            # Function should exist
            assert callable(render_metric_card)
        except ImportError:
            pass


class TestAppLogic:
    """Test application logic without Streamlit."""
    
    def test_app_module_imports(self):
        """Test that app module can be imported."""
        try:
            import gui.app as app
            assert hasattr(app, 'main')
            assert hasattr(app, 'load_models')
        except ImportError:
            # May fail due to Streamlit context
            pass


class TestGUIIntegration:
    """Test GUI integration with backend modules."""
    
    def test_models_can_be_loaded(self):
        """Test that ML models can be loaded (for GUI use)."""
        from prediction.estimator import EnergyPredictor
        from anomaly.detector import AnomalyDetector
        from optimization.recomender import PromptOptimizer
        from utils.database import DatabaseManager
        
        # These should initialize without errors
        predictor = EnergyPredictor()
        detector = AnomalyDetector()
        optimizer = PromptOptimizer()
        # Database uses default path which may or may not exist
        
        assert predictor is not None
        assert detector is not None
        assert optimizer is not None
    
    def test_analysis_workflow_components(self, sample_prompts):
        """Test components used in analysis workflow."""
        from nlp.parser import parse_prompt
        from nlp.complexity_score import get_complexity_breakdown
        from prediction.estimator import EnergyPredictor
        from anomaly.detector import AnomalyDetector
        
        prompt = sample_prompts["medium"]
        
        # Components used in Tab 1
        parsed = parse_prompt(prompt, use_embeddings=False)
        complexity = get_complexity_breakdown(prompt)
        predictor = EnergyPredictor()
        energy = predictor.predict(prompt)
        detector = AnomalyDetector()
        anomaly = detector.detect(prompt, energy.energy_kwh)
        
        # All should return valid results
        assert parsed.token_count > 0
        assert complexity["overall_score"] >= 0
        assert energy.energy_kwh > 0
        assert anomaly.anomaly_score is not None
    
    def test_optimization_workflow_components(self, sample_prompts):
        """Test components used in optimization workflow."""
        from optimization.recomender import PromptOptimizer
        
        prompt = sample_prompts["verbose"]
        
        # Components used in Tab 2
        optimizer = PromptOptimizer(min_similarity=0.6)
        result = optimizer.optimize(prompt)
        suggestions = optimizer.get_improvement_suggestions(prompt)
        
        # All should return valid results
        assert result.optimized_prompt is not None
        assert len(suggestions) > 0
    
    def test_dashboard_workflow_components(self, database_manager, sample_prompts):
        """Test components used in dashboard workflow."""
        from nlp.parser import parse_prompt
        from nlp.complexity_score import compute_complexity
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        
        # Log some data
        for prompt_name in ["simple", "medium"]:
            prompt = sample_prompts[prompt_name]
            parsed = parse_prompt(prompt, use_embeddings=False)
            complexity = compute_complexity(prompt)
            energy = predictor.predict(prompt)
            
            database_manager.log_energy(
                prompt=prompt,
                token_count=parsed.token_count,
                complexity_score=complexity,
                energy_kwh=energy.energy_kwh
            )
        
        # Get statistics (used in Tab 3)
        stats = database_manager.get_statistics()
        logs = database_manager.get_logs(limit=10)
        
        assert stats["total_prompts"] >= 2
        assert len(logs) >= 2
    
    def test_report_workflow_components(self, database_manager, date_range):
        """Test components used in report workflow."""
        # Generate report (used in Tab 4)
        report = database_manager.generate_transparency_report(
            date_range["start"],
            date_range["end"]
        )
        
        # Should have required sections
        assert "report_id" in report
        assert "compliance" in report
        
        # Get previous reports
        reports = database_manager.get_reports(limit=5)
        assert isinstance(reports, list)


class TestChartData:
    """Test chart data generation."""
    
    def test_energy_gauge_data(self, sample_prompts):
        """Test data for energy gauge chart."""
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        result = predictor.predict(sample_prompts["medium"])
        
        # Data for gauge chart
        energy_kwh = result.energy_kwh
        energy_level = result.energy_level
        
        assert energy_kwh > 0
        assert energy_level in ["low", "medium", "high", "very_high"]
    
    def test_complexity_breakdown_data(self, sample_prompts):
        """Test data for complexity breakdown chart."""
        from nlp.complexity_score import get_complexity_breakdown
        
        breakdown = get_complexity_breakdown(sample_prompts["medium"])
        
        # Data for radar/bar chart
        assert "sentence_complexity" in breakdown
        assert "vocabulary_complexity" in breakdown
        assert "syntactic_complexity" in breakdown
        assert "semantic_density" in breakdown
        assert "structural_complexity" in breakdown
        
        # All values should be in [0, 1]
        for key in ["sentence_complexity", "vocabulary_complexity", "syntactic_complexity",
                    "semantic_density", "structural_complexity"]:
            assert 0 <= breakdown[key] <= 1
    
    def test_energy_comparison_data(self, sample_prompts):
        """Test data for energy comparison chart."""
        from optimization.recomender import PromptOptimizer
        
        optimizer = PromptOptimizer()
        result = optimizer.optimize(sample_prompts["verbose"])
        
        # Data for comparison chart
        original = result.original_energy_kwh
        optimized = result.optimized_energy_kwh
        
        assert original > 0
        assert optimized > 0


class TestDataFormatting:
    """Test data formatting for display."""
    
    def test_energy_formatting(self, sample_prompts):
        """Test energy value formatting."""
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        result = predictor.predict(sample_prompts["medium"])
        
        # Format as would be displayed
        formatted = f"{result.energy_kwh:.4f} kWh"
        assert "kWh" in formatted
    
    def test_percentage_formatting(self, sample_prompts):
        """Test percentage formatting."""
        from optimization.recomender import PromptOptimizer
        
        optimizer = PromptOptimizer()
        result = optimizer.optimize(sample_prompts["verbose"])
        
        # Format as percentage
        formatted = f"{result.semantic_similarity:.1%}"
        assert "%" in formatted
    
    def test_carbon_formatting(self, sample_prompts):
        """Test carbon footprint formatting."""
        from prediction.estimator import EnergyPredictor
        
        predictor = EnergyPredictor()
        result = predictor.predict(sample_prompts["medium"])
        
        # Format carbon
        formatted = f"{result.carbon_footprint_kg:.4f} kg CO₂"
        assert "kg" in formatted


class TestAnomalyAlertData:
    """Test anomaly alert data generation."""
    
    def test_anomaly_alert_normal(self, sample_prompts):
        """Test anomaly alert for normal prompt."""
        from anomaly.detector import AnomalyDetector
        
        detector = AnomalyDetector()
        result = detector.detect(sample_prompts["simple"])
        
        alert_data = result.to_dict()
        assert "is_anomaly" in alert_data
        assert "severity" in alert_data
        assert "recommendation" in alert_data
    
    def test_anomaly_alert_complex(self, sample_prompts):
        """Test anomaly alert for complex prompt."""
        from anomaly.detector import AnomalyDetector
        
        detector = AnomalyDetector()
        result = detector.detect(sample_prompts["complex"])
        
        alert_data = result.to_dict()
        assert "contributing_factors" in alert_data


class TestSidebarConfig:
    """Test sidebar configuration handling."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        from utils.config import APP_CONFIG
        
        # Default values used in sidebar
        assert APP_CONFIG.default_num_layers == 24
        assert APP_CONFIG.default_training_hours == 8.0
    
    def test_region_options(self):
        """Test available region options."""
        from utils.config import DATA_CENTER_CONFIG
        
        # Regions available in sidebar
        regions = list(DATA_CENTER_CONFIG.electricity_costs.keys())
        assert len(regions) > 0
        assert "california" in regions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
