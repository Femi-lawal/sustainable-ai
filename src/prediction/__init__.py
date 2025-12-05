"""
Prediction module for Sustainable AI application.
Provides energy consumption prediction using supervised machine learning.
"""

from .estimator import (
    EnergyPredictor,
    EnergyPrediction,
    create_predictor,
    predict_energy
)

__all__ = [
    'EnergyPredictor',
    'EnergyPrediction',
    'create_predictor',
    'predict_energy'
]
