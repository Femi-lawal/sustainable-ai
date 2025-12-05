"""
Anomaly module for Sustainable AI application.
Provides unsupervised anomaly detection for identifying high-resource prompts.
"""

from .detector import (
    AnomalyDetector,
    AnomalyResult,
    create_detector,
    detect_anomaly
)

__all__ = [
    'AnomalyDetector',
    'AnomalyResult',
    'create_detector',
    'detect_anomaly'
]
