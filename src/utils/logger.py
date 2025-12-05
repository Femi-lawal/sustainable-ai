"""
Logging utilities for the Sustainable AI application.
Provides comprehensive logging for energy tracking, operations, and transparency reporting.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler
import json

# Import config (handle both direct run and module import)
try:
    from .config import LOGGING_CONFIG, PROJECT_ROOT
except ImportError:
    from config import LOGGING_CONFIG, PROJECT_ROOT


class EnergyLogger:
    """
    Custom logger for tracking energy consumption and operations.
    Supports both console and file logging with rotation.
    """
    
    def __init__(self, name: str = "sustainable_ai", level: Optional[str] = None):
        """
        Initialize the energy logger.
        
        Args:
            name: Logger name
            level: Optional override for log level
        """
        self.name = name
        self.level = getattr(logging, level or LOGGING_CONFIG.log_level)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Energy metrics storage for session
        self.session_metrics = {
            "start_time": datetime.now().isoformat(),
            "total_prompts": 0,
            "total_energy_kwh": 0.0,
            "anomalies_detected": 0,
            "optimizations_made": 0
        }
    
    def _setup_handlers(self):
        """Set up console and file handlers."""
        formatter = logging.Formatter(LOGGING_CONFIG.log_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = LOGGING_CONFIG.log_dir / LOGGING_CONFIG.log_file
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=LOGGING_CONFIG.max_log_size,
            backupCount=LOGGING_CONFIG.backup_count
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message(message, kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(self._format_message(message, kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(self._format_message(message, kwargs))
    
    def _format_message(self, message: str, extra: Dict[str, Any]) -> str:
        """Format message with extra context."""
        if extra:
            return f"{message} | {json.dumps(extra)}"
        return message
    
    # =========================================================================
    # ENERGY TRACKING METHODS
    # =========================================================================
    
    def log_energy_prediction(self, prompt: str, energy_kwh: float, 
                              features: Dict[str, Any], model_params: Dict[str, Any]):
        """
        Log an energy prediction event.
        
        Args:
            prompt: The input prompt (truncated for privacy)
            energy_kwh: Predicted energy consumption in kWh
            features: Features extracted from the prompt
            model_params: LLM model parameters
        """
        self.session_metrics["total_prompts"] += 1
        self.session_metrics["total_energy_kwh"] += energy_kwh
        
        truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        
        self.info(
            "Energy prediction completed",
            prompt_preview=truncated_prompt,
            energy_kwh=round(energy_kwh, 6),
            token_count=features.get("token_count", 0),
            complexity_score=round(features.get("complexity_score", 0), 4),
            num_layers=model_params.get("num_layers", 0)
        )
    
    def log_anomaly_detection(self, prompt: str, anomaly_score: float, 
                               is_anomaly: bool, reason: str = ""):
        """
        Log an anomaly detection event.
        
        Args:
            prompt: The input prompt (truncated)
            anomaly_score: The anomaly score from the detector
            is_anomaly: Whether the prompt was flagged as anomalous
            reason: Optional reason for the anomaly
        """
        if is_anomaly:
            self.session_metrics["anomalies_detected"] += 1
        
        truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        
        log_level = logging.WARNING if is_anomaly else logging.INFO
        self.logger.log(
            log_level,
            self._format_message(
                "Anomaly detection completed",
                {
                    "prompt_preview": truncated_prompt,
                    "anomaly_score": round(anomaly_score, 4),
                    "is_anomaly": is_anomaly,
                    "reason": reason
                }
            )
        )
    
    def log_optimization(self, original_prompt: str, optimized_prompt: str,
                         original_energy: float, optimized_energy: float,
                         strategy: str):
        """
        Log a prompt optimization event.
        
        Args:
            original_prompt: The original prompt
            optimized_prompt: The optimized prompt
            original_energy: Original energy estimate (kWh)
            optimized_energy: Optimized energy estimate (kWh)
            strategy: Optimization strategy used
        """
        self.session_metrics["optimizations_made"] += 1
        
        energy_reduction = original_energy - optimized_energy
        reduction_percent = (energy_reduction / original_energy * 100) if original_energy > 0 else 0
        
        self.info(
            "Prompt optimization completed",
            original_preview=original_prompt[:50] + "...",
            optimized_preview=optimized_prompt[:50] + "...",
            original_energy_kwh=round(original_energy, 6),
            optimized_energy_kwh=round(optimized_energy, 6),
            energy_reduction_kwh=round(energy_reduction, 6),
            reduction_percent=round(reduction_percent, 2),
            strategy=strategy
        )
    
    def log_session_summary(self) -> Dict[str, Any]:
        """
        Log and return session summary.
        
        Returns:
            Dictionary with session metrics
        """
        self.session_metrics["end_time"] = datetime.now().isoformat()
        
        self.info(
            "Session summary",
            **self.session_metrics
        )
        
        return self.session_metrics.copy()
    
    # =========================================================================
    # TRANSPARENCY REPORTING
    # =========================================================================
    
    def generate_transparency_report(self, period_start: datetime, 
                                      period_end: datetime,
                                      detailed_logs: list) -> Dict[str, Any]:
        """
        Generate a transparency report for energy usage.
        
        Args:
            period_start: Report period start date
            period_end: Report period end date
            detailed_logs: List of detailed log entries
        
        Returns:
            Dictionary containing the transparency report
        """
        total_energy = sum(log.get("energy_kwh", 0) for log in detailed_logs)
        total_prompts = len(detailed_logs)
        avg_energy = total_energy / total_prompts if total_prompts > 0 else 0
        
        # Calculate carbon footprint (using default 0.42 kg CO2/kWh)
        carbon_footprint = total_energy * 0.42
        
        # Calculate water usage (using default 1.8 L/kWh)
        water_usage = total_energy * 1.8
        
        report = {
            "report_id": f"TR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat()
            },
            "energy_metrics": {
                "total_energy_kwh": round(total_energy, 4),
                "average_energy_per_prompt_kwh": round(avg_energy, 6),
                "total_prompts_processed": total_prompts,
                "peak_energy_kwh": max((log.get("energy_kwh", 0) for log in detailed_logs), default=0)
            },
            "environmental_impact": {
                "carbon_footprint_kg_co2": round(carbon_footprint, 4),
                "water_usage_liters": round(water_usage, 4)
            },
            "anomalies": {
                "total_detected": sum(1 for log in detailed_logs if log.get("is_anomaly", False)),
                "percentage": round(
                    sum(1 for log in detailed_logs if log.get("is_anomaly", False)) / total_prompts * 100
                    if total_prompts > 0 else 0, 2
                )
            },
            "optimizations": {
                "total_optimizations": sum(1 for log in detailed_logs if log.get("was_optimized", False)),
                "total_energy_saved_kwh": round(
                    sum(log.get("energy_saved_kwh", 0) for log in detailed_logs), 4
                )
            },
            "eu_compliance": {
                "reporting_standard": "EU AI Act Energy Reporting (2026)",
                "data_completeness_percent": 100 if detailed_logs else 0
            }
        }
        
        self.info("Transparency report generated", report_id=report["report_id"])
        
        return report


# Global logger instance
_logger_instance = None

def get_logger(name: str = "sustainable_ai") -> EnergyLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        EnergyLogger instance
    """
    global _logger_instance
    if _logger_instance is None or _logger_instance.name != name:
        _logger_instance = EnergyLogger(name)
    return _logger_instance


# Convenience functions for quick logging
def log_info(message: str, **kwargs):
    """Quick info log."""
    get_logger().info(message, **kwargs)

def log_warning(message: str, **kwargs):
    """Quick warning log."""
    get_logger().warning(message, **kwargs)

def log_error(message: str, **kwargs):
    """Quick error log."""
    get_logger().error(message, **kwargs)
