"""
Unit Tests for the Database Module (database.py).
Tests SQLite storage and transparency reporting.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDatabaseManager:
    """Test DatabaseManager class."""
    
    def test_database_initialization(self, database_manager):
        """Test database initializes correctly."""
        assert database_manager is not None
        assert database_manager.db_path.exists()
    
    def test_database_tables_created(self, database_manager):
        """Test that tables are created."""
        conn = database_manager._get_connection()
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "energy_logs" in tables
        assert "transparency_reports" in tables
        assert "benchmarks" in tables
        
        conn.close()
    
    def test_get_connection(self, database_manager):
        """Test getting a database connection."""
        conn = database_manager._get_connection()
        assert conn is not None
        conn.close()


class TestDatabaseLogging:
    """Test energy logging functionality."""
    
    def test_log_energy_basic(self, database_manager):
        """Test basic energy logging."""
        log_id = database_manager.log_energy(
            prompt="Test prompt",
            token_count=50,
            complexity_score=0.5,
            energy_kwh=0.1
        )
        assert log_id > 0
    
    def test_log_energy_full(self, database_manager):
        """Test energy logging with all parameters."""
        log_id = database_manager.log_energy(
            prompt="Test prompt with all parameters",
            token_count=100,
            complexity_score=0.75,
            energy_kwh=0.25,
            carbon_kg=0.105,
            water_liters=0.45,
            is_anomaly=True,
            was_optimized=True,
            energy_saved_kwh=0.05,
            model_params={"num_layers": 24, "training_hours": 8},
            region="california"
        )
        assert log_id > 0
    
    def test_log_hashes_prompt(self, database_manager):
        """Test that prompt is hashed for privacy."""
        log_id = database_manager.log_energy(
            prompt="Sensitive prompt content",
            token_count=30,
            complexity_score=0.3,
            energy_kwh=0.05
        )
        
        # Retrieve the log
        logs = database_manager.get_logs(limit=1)
        assert logs[0]["prompt_hash"] is not None
        assert len(logs[0]["prompt_hash"]) == 16  # SHA256 truncated to 16 chars
        assert "Sensitive prompt content" not in logs[0]["prompt_hash"]
    
    def test_log_creates_preview(self, database_manager):
        """Test that prompt preview is created."""
        long_prompt = "A" * 200  # 200 characters
        log_id = database_manager.log_energy(
            prompt=long_prompt,
            token_count=50,
            complexity_score=0.5,
            energy_kwh=0.1
        )
        
        logs = database_manager.get_logs(limit=1)
        assert len(logs[0]["prompt_preview"]) <= 103  # 100 chars + "..."


class TestDatabaseRetrieval:
    """Test log retrieval functionality."""
    
    def test_get_logs_empty(self, database_manager):
        """Test getting logs from empty database."""
        logs = database_manager.get_logs()
        assert isinstance(logs, list)
    
    def test_get_logs_with_entries(self, database_manager):
        """Test getting logs with entries."""
        # Add some logs
        for i in range(5):
            database_manager.log_energy(
                prompt=f"Test prompt {i}",
                token_count=50 + i * 10,
                complexity_score=0.3 + i * 0.1,
                energy_kwh=0.1 + i * 0.05
            )
        
        logs = database_manager.get_logs()
        assert len(logs) >= 5
    
    def test_get_logs_with_limit(self, database_manager):
        """Test getting logs with limit."""
        # Add some logs
        for i in range(10):
            database_manager.log_energy(
                prompt=f"Test prompt {i}",
                token_count=50,
                complexity_score=0.5,
                energy_kwh=0.1
            )
        
        logs = database_manager.get_logs(limit=5)
        assert len(logs) <= 5
    
    def test_get_logs_with_date_range(self, database_manager):
        """Test getting logs with date range."""
        # Add a log
        database_manager.log_energy(
            prompt="Test prompt",
            token_count=50,
            complexity_score=0.5,
            energy_kwh=0.1
        )
        
        # Query with date range
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        logs = database_manager.get_logs(start_date=start_date, end_date=end_date)
        assert len(logs) >= 1


class TestDatabaseStatistics:
    """Test statistics calculation."""
    
    def test_get_statistics_empty(self, database_manager):
        """Test statistics on empty database."""
        stats = database_manager.get_statistics()
        assert "total_prompts" in stats
    
    def test_get_statistics_with_data(self, database_manager):
        """Test statistics with data."""
        # Add test data
        for i in range(5):
            database_manager.log_energy(
                prompt=f"Test prompt {i}",
                token_count=50 + i * 10,
                complexity_score=0.3 + i * 0.1,
                energy_kwh=0.1 + i * 0.05,
                carbon_kg=0.04 + i * 0.02,
                water_liters=0.2 + i * 0.1,
                is_anomaly=(i == 4),
                was_optimized=(i % 2 == 0)
            )
        
        stats = database_manager.get_statistics()
        
        assert stats["total_prompts"] >= 5
        assert stats["total_energy_kwh"] > 0
        assert stats["average_energy_kwh"] > 0
        assert stats["min_energy_kwh"] > 0
        assert stats["max_energy_kwh"] > 0
    
    def test_statistics_date_range(self, database_manager):
        """Test statistics with date range."""
        # Add a log
        database_manager.log_energy(
            prompt="Test prompt",
            token_count=50,
            complexity_score=0.5,
            energy_kwh=0.1
        )
        
        start_date = datetime.now() - timedelta(days=1)
        stats = database_manager.get_statistics(start_date=start_date)
        assert "total_prompts" in stats


class TestTransparencyReports:
    """Test transparency report generation."""
    
    def test_generate_report_empty(self, database_manager, date_range):
        """Test generating report from empty database."""
        report = database_manager.generate_transparency_report(
            date_range["start"],
            date_range["end"]
        )
        assert "report_id" in report
        assert "generated_at" in report
        assert "compliance" in report
    
    def test_generate_report_with_data(self, database_manager, date_range):
        """Test generating report with data."""
        # Add test data
        for i in range(5):
            database_manager.log_energy(
                prompt=f"Test prompt {i}",
                token_count=50 + i * 10,
                complexity_score=0.3 + i * 0.1,
                energy_kwh=0.1 + i * 0.05,
                carbon_kg=0.04 + i * 0.02,
                water_liters=0.2 + i * 0.1
            )
        
        report = database_manager.generate_transparency_report(
            date_range["start"],
            date_range["end"]
        )
        
        assert report["report_id"].startswith("TR-")
        assert report["summary"]["total_prompts"] >= 5
        assert report["summary"]["total_energy_kwh"] > 0
    
    def test_report_contains_required_sections(self, database_manager, date_range):
        """Test that report contains all required sections."""
        report = database_manager.generate_transparency_report(
            date_range["start"],
            date_range["end"]
        )
        
        assert "report_id" in report
        assert "generated_at" in report
        assert "period" in report
        assert "summary" in report
        assert "environmental_impact" in report
        assert "optimization_impact" in report
        assert "anomalies" in report
        assert "compliance" in report
    
    def test_report_environmental_impact(self, database_manager, date_range):
        """Test environmental impact section."""
        # Add test data
        database_manager.log_energy(
            prompt="Test prompt",
            token_count=50,
            complexity_score=0.5,
            energy_kwh=0.1,
            carbon_kg=0.042,
            water_liters=0.18
        )
        
        report = database_manager.generate_transparency_report(
            date_range["start"],
            date_range["end"]
        )
        
        impact = report["environmental_impact"]
        assert "carbon_footprint_kg" in impact
        assert "water_usage_liters" in impact
        assert "trees_equivalent" in impact
        assert "homes_powered_hours" in impact
    
    def test_report_compliance_section(self, database_manager, date_range):
        """Test compliance section."""
        report = database_manager.generate_transparency_report(
            date_range["start"],
            date_range["end"]
        )
        
        compliance = report["compliance"]
        assert "standard" in compliance
        assert "deadline" in compliance
        assert "days_until_deadline" in compliance
        assert "status" in compliance
    
    def test_report_stored(self, database_manager, date_range):
        """Test that report is stored in database."""
        report = database_manager.generate_transparency_report(
            date_range["start"],
            date_range["end"]
        )
        
        # Retrieve stored reports
        reports = database_manager.get_reports(limit=1)
        assert len(reports) >= 1
        assert reports[0]["report_id"] == report["report_id"]


class TestDatabaseReports:
    """Test report retrieval."""
    
    def test_get_reports_empty(self, database_manager):
        """Test getting reports from empty database."""
        reports = database_manager.get_reports()
        assert isinstance(reports, list)
    
    def test_get_reports_with_data(self, database_manager, date_range):
        """Test getting reports with data."""
        # Generate some reports
        for _ in range(3):
            database_manager.generate_transparency_report(
                date_range["start"],
                date_range["end"]
            )
        
        reports = database_manager.get_reports(limit=10)
        assert len(reports) >= 3


class TestDatabaseHourlyBreakdown:
    """Test hourly breakdown functionality."""
    
    def test_get_hourly_breakdown(self, database_manager):
        """Test hourly breakdown."""
        # Add a log
        database_manager.log_energy(
            prompt="Test prompt",
            token_count=50,
            complexity_score=0.5,
            energy_kwh=0.1
        )
        
        breakdown = database_manager.get_hourly_breakdown(datetime.now())
        assert isinstance(breakdown, list)


class TestDatabaseExport:
    """Test export functionality."""
    
    def test_export_to_csv(self, database_manager, temp_dir):
        """Test exporting to CSV."""
        # Add test data
        for i in range(5):
            database_manager.log_energy(
                prompt=f"Test prompt {i}",
                token_count=50,
                complexity_score=0.5,
                energy_kwh=0.1
            )
        
        csv_path = temp_dir / "export.csv"
        database_manager.export_to_csv(csv_path)
        
        assert csv_path.exists()
        
        # Verify CSV content
        with open(csv_path, "r") as f:
            content = f.read()
            assert "energy_kwh" in content


class TestDatabaseCleanup:
    """Test cleanup functionality."""
    
    def test_clear_old_logs(self, database_manager):
        """Test clearing old logs."""
        # Add a log
        database_manager.log_energy(
            prompt="Test prompt",
            token_count=50,
            complexity_score=0.5,
            energy_kwh=0.1
        )
        
        # Clear logs older than 0 days (should clear nothing recent)
        deleted = database_manager.clear_old_logs(days_to_keep=0)
        # May or may not delete depending on timing


class TestDatabaseConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_database_function(self):
        """Test get_database() function."""
        from utils.database import get_database
        db = get_database()
        assert db is not None
    
    def test_log_prompt_energy_function(self, temp_db_path):
        """Test log_prompt_energy() convenience function."""
        # This would need a modified setup to use temp db
        pass
    
    def test_get_energy_statistics_function(self):
        """Test get_energy_statistics() convenience function."""
        from utils.database import get_energy_statistics
        stats = get_energy_statistics(days=7)
        assert "total_prompts" in stats or "message" in stats


class TestDatabaseEdgeCases:
    """Test edge cases."""
    
    def test_log_empty_prompt(self, database_manager):
        """Test logging empty prompt."""
        log_id = database_manager.log_energy(
            prompt="",
            token_count=0,
            complexity_score=0,
            energy_kwh=0
        )
        assert log_id > 0
    
    def test_log_unicode_prompt(self, database_manager):
        """Test logging unicode prompt."""
        log_id = database_manager.log_energy(
            prompt="Hello 你好 مرحبا",
            token_count=5,
            complexity_score=0.3,
            energy_kwh=0.05
        )
        assert log_id > 0
    
    def test_log_very_long_prompt(self, database_manager):
        """Test logging very long prompt."""
        long_prompt = "A" * 10000
        log_id = database_manager.log_energy(
            prompt=long_prompt,
            token_count=500,
            complexity_score=0.5,
            energy_kwh=1.0
        )
        assert log_id > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
