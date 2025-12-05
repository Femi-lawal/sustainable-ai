"""
Anomaly Detection module using Unsupervised Machine Learning.
Identifies prompts with unusually high resource demands for transparency reporting.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Import from project
try:
    from utils.config import ANOMALY_DETECTOR_CONFIG, MODEL_DIR
    from nlp.parser import parse_prompt
    from nlp.complexity_score import compute_complexity
except ImportError:
    from src.utils.config import ANOMALY_DETECTOR_CONFIG, MODEL_DIR
    from src.nlp.parser import parse_prompt
    from src.nlp.complexity_score import compute_complexity


@dataclass
class AnomalyResult:
    """
    Result of anomaly detection analysis.
    """
    prompt: str
    is_anomaly: bool
    anomaly_score: float  # Lower is more anomalous (negative for outliers)
    confidence: float
    
    # Detailed analysis
    anomaly_type: str  # "normal", "high_complexity", "high_tokens", "unusual_pattern"
    severity: str  # "low", "medium", "high", "critical"
    
    # Contributing factors
    contributing_factors: Dict[str, str]
    
    # Recommendations
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "contributing_factors": self.contributing_factors,
            "recommendation": self.recommendation
        }


class AnomalyDetector:
    """
    Unsupervised learning model for detecting anomalous prompts.
    
    Identifies prompts that:
    - Require excessive computation relative to their length
    - Have unusual complexity patterns
    - Deviate significantly from normal usage patterns
    
    Supports multiple detection algorithms:
    - Isolation Forest (default, fast and effective)
    - One-Class SVM (better for high-dimensional data)
    - Local Outlier Factor (good for local anomalies)
    """
    
    # Severity thresholds
    SEVERITY_THRESHOLDS = {
        "low": -0.3,      # Slightly anomalous
        "medium": -0.5,   # Moderately anomalous
        "high": -0.7,     # Highly anomalous
        "critical": -0.9  # Critically anomalous
    }
    
    def __init__(self, model_type: str = "isolation_forest", 
                 model_path: Optional[Path] = None):
        """
        Initialize the anomaly detector.
        
        Args:
            model_type: Type of anomaly detection model
            model_path: Optional path to load a pre-trained model
        """
        self.model_type = model_type
        self.model_path = model_path or ANOMALY_DETECTOR_CONFIG.model_path
        
        self.model = None
        self.scaler = None
        self.feature_stats = None  # For rule-based detection
        self.is_trained = False
        
        # Try to load existing model
        if self.model_path.exists():
            self._load_model()
        else:
            self._create_model()
    
    def _create_model(self):
        """Create a new anomaly detection model."""
        if self.model_type == "isolation_forest":
            self.model = IsolationForest(
                contamination=ANOMALY_DETECTOR_CONFIG.contamination,
                n_estimators=ANOMALY_DETECTOR_CONFIG.n_estimators,
                max_samples=ANOMALY_DETECTOR_CONFIG.max_samples,
                random_state=ANOMALY_DETECTOR_CONFIG.random_state,
                n_jobs=-1
            )
        elif self.model_type == "one_class_svm":
            self.model = OneClassSVM(
                kernel='rbf',
                gamma='auto',
                nu=ANOMALY_DETECTOR_CONFIG.contamination
            )
        elif self.model_type == "local_outlier_factor":
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=ANOMALY_DETECTOR_CONFIG.contamination,
                novelty=True
            )
        else:
            # Default to Isolation Forest
            self.model = IsolationForest(
                contamination=ANOMALY_DETECTOR_CONFIG.contamination,
                random_state=ANOMALY_DETECTOR_CONFIG.random_state
            )
        
        self.scaler = StandardScaler()
    
    def _load_model(self) -> bool:
        """Load pre-trained model from disk."""
        try:
            saved = joblib.load(self.model_path)
            self.model = saved.get('model')
            self.scaler = saved.get('scaler', StandardScaler())
            self.feature_stats = saved.get('feature_stats', {})
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Could not load model: {e}")
            self._create_model()
            return False
    
    def _save_model(self):
        """Save trained model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_stats': self.feature_stats
        }, self.model_path)
    
    def extract_features(self, prompt: str, energy_kwh: Optional[float] = None) -> Dict[str, float]:
        """
        Extract features for anomaly detection.
        
        Args:
            prompt: Input prompt text
            energy_kwh: Optional pre-computed energy (if available)
        
        Returns:
            Dictionary of features
        """
        # Parse prompt
        parsed = parse_prompt(prompt, use_embeddings=False)
        
        # Get complexity score
        complexity = compute_complexity(prompt)
        
        # Calculate derived ratios that might indicate anomalies
        token_to_word_ratio = parsed.token_count / max(parsed.word_count, 1)
        complexity_to_length_ratio = complexity / max(parsed.char_count, 1) * 1000
        
        features = {
            "token_count": parsed.token_count,
            "complexity_score": complexity,
            "char_count": parsed.char_count,
            "avg_word_length": parsed.avg_word_length,
            "vocabulary_richness": parsed.vocabulary_richness,
            "lexical_density": parsed.lexical_density,
            "token_to_word_ratio": token_to_word_ratio,
            "complexity_to_length_ratio": complexity_to_length_ratio
        }
        
        if energy_kwh is not None:
            features["energy_kwh"] = energy_kwh
            # Energy efficiency metric
            features["energy_per_token"] = energy_kwh / max(parsed.token_count, 1)
        
        return features
    
    def train(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the anomaly detection model.
        
        Args:
            training_data: DataFrame with features (normal samples)
        
        Returns:
            Dictionary with training metrics
        """
        # Create model if needed
        if self.model is None:
            self._create_model()
        
        # Prepare features
        feature_cols = [col for col in training_data.columns 
                       if col not in ['prompt', 'is_anomaly', 'label']]
        X = training_data[feature_cols].values
        
        # Store feature statistics for rule-based detection
        self.feature_stats = {
            col: {
                'mean': training_data[col].mean(),
                'std': training_data[col].std(),
                'q1': training_data[col].quantile(0.25),
                'q3': training_data[col].quantile(0.75),
                'iqr': training_data[col].quantile(0.75) - training_data[col].quantile(0.25)
            }
            for col in feature_cols
        }
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Get training scores for metrics
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X_scaled)
        else:
            scores = self.model.score_samples(X_scaled)
        
        # Calculate metrics
        anomaly_predictions = self.model.predict(X_scaled)
        n_anomalies = (anomaly_predictions == -1).sum()
        
        metrics = {
            "n_samples": len(X),
            "n_anomalies_detected": n_anomalies,
            "anomaly_rate": n_anomalies / len(X),
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "min_score": scores.min(),
            "max_score": scores.max()
        }
        
        # Save model
        self._save_model()
        
        return metrics
    
    def _determine_anomaly_type(self, features: Dict[str, float]) -> Tuple[str, Dict[str, str]]:
        """
        Determine the type of anomaly based on features.
        
        Args:
            features: Extracted features
        
        Returns:
            Tuple of (anomaly_type, contributing_factors)
        """
        contributing_factors = {}
        anomaly_type = "unusual_pattern"
        
        # Check for specific anomaly patterns
        if features.get("complexity_score", 0) > 0.8:
            contributing_factors["complexity"] = "Very high complexity score"
            anomaly_type = "high_complexity"
        
        if features.get("token_count", 0) > 500:
            contributing_factors["tokens"] = "Excessive token count"
            anomaly_type = "high_tokens"
        
        if features.get("token_to_word_ratio", 0) > 2.0:
            contributing_factors["tokenization"] = "Unusual tokenization pattern"
        
        if features.get("energy_per_token", 0) > 0.01:
            contributing_factors["efficiency"] = "Low energy efficiency per token"
        
        if features.get("complexity_to_length_ratio", 0) > 1.0:
            contributing_factors["density"] = "High complexity density"
        
        if not contributing_factors:
            contributing_factors["general"] = "Unusual feature combination"
        
        return anomaly_type, contributing_factors
    
    def _generate_recommendation(self, anomaly_type: str, severity: str, 
                                  contributing_factors: Dict[str, str]) -> str:
        """
        Generate a recommendation based on the anomaly analysis.
        
        Args:
            anomaly_type: Type of anomaly detected
            severity: Severity level
            contributing_factors: Factors contributing to the anomaly
        
        Returns:
            Recommendation string
        """
        recommendations = {
            "high_complexity": "Consider simplifying the prompt by breaking it into smaller, more focused queries.",
            "high_tokens": "The prompt is very long. Try to be more concise or split into multiple requests.",
            "unusual_pattern": "The prompt has unusual characteristics. Review for efficiency.",
            "normal": "The prompt appears normal and should process efficiently."
        }
        
        base_recommendation = recommendations.get(anomaly_type, "Review prompt for optimization opportunities.")
        
        if severity in ["high", "critical"]:
            base_recommendation += " This prompt may significantly impact energy consumption."
        
        return base_recommendation
    
    def detect(self, prompt: str, energy_kwh: Optional[float] = None) -> AnomalyResult:
        """
        Detect if a prompt is anomalous.
        
        Args:
            prompt: Input prompt text
            energy_kwh: Optional pre-computed energy estimate
        
        Returns:
            AnomalyResult with detailed analysis
        """
        # Extract features
        features = self.extract_features(prompt, energy_kwh)
        
        # Prepare features for model
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Get anomaly score and prediction
        if self.is_trained and self.model is not None:
            # Scale features
            if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                feature_array = self.scaler.transform(feature_array)
            
            # Get prediction and score
            prediction = self.model.predict(feature_array)[0]
            
            if hasattr(self.model, 'decision_function'):
                score = float(self.model.decision_function(feature_array)[0])
            elif hasattr(self.model, 'score_samples'):
                score = float(self.model.score_samples(feature_array)[0])
            else:
                score = -0.5 if prediction == -1 else 0.5
            
            is_anomaly = prediction == -1
        else:
            # Rule-based detection fallback
            is_anomaly, score = self._rule_based_detection(features)
        
        # Calculate confidence
        confidence = min(abs(score), 1.0)
        
        # Determine severity
        if score < self.SEVERITY_THRESHOLDS["critical"]:
            severity = "critical"
        elif score < self.SEVERITY_THRESHOLDS["high"]:
            severity = "high"
        elif score < self.SEVERITY_THRESHOLDS["medium"]:
            severity = "medium"
        elif score < self.SEVERITY_THRESHOLDS["low"]:
            severity = "low"
        else:
            severity = "normal"
        
        # Determine anomaly type and contributing factors
        if is_anomaly:
            anomaly_type, contributing_factors = self._determine_anomaly_type(features)
        else:
            anomaly_type = "normal"
            contributing_factors = {}
        
        # Generate recommendation
        recommendation = self._generate_recommendation(anomaly_type, severity, contributing_factors)
        
        return AnomalyResult(
            prompt=prompt,
            is_anomaly=is_anomaly,
            anomaly_score=round(score, 4),
            confidence=round(confidence, 4),
            anomaly_type=anomaly_type,
            severity=severity if is_anomaly else "normal",
            contributing_factors=contributing_factors,
            recommendation=recommendation
        )
    
    def _rule_based_detection(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """
        Fallback rule-based anomaly detection.
        
        Args:
            features: Extracted features
        
        Returns:
            Tuple of (is_anomaly, score)
        """
        anomaly_score = 0.5  # Start with neutral score
        
        # Check each feature against thresholds
        if features.get("token_count", 0) > 400:
            anomaly_score -= 0.3
        
        if features.get("complexity_score", 0) > 0.7:
            anomaly_score -= 0.2
        
        if features.get("token_to_word_ratio", 0) > 1.8:
            anomaly_score -= 0.15
        
        if features.get("complexity_to_length_ratio", 0) > 0.8:
            anomaly_score -= 0.15
        
        is_anomaly = anomaly_score < ANOMALY_DETECTOR_CONFIG.anomaly_threshold
        
        return is_anomaly, anomaly_score
    
    def detect_batch(self, prompts: List[str], 
                     energy_values: Optional[List[float]] = None) -> List[AnomalyResult]:
        """
        Detect anomalies in multiple prompts.
        
        Args:
            prompts: List of prompt texts
            energy_values: Optional list of energy values
        
        Returns:
            List of AnomalyResult objects
        """
        if energy_values is None:
            energy_values = [None] * len(prompts)
        
        return [
            self.detect(prompt, energy) 
            for prompt, energy in zip(prompts, energy_values)
        ]
    
    def get_anomaly_statistics(self, results: List[AnomalyResult]) -> Dict[str, Any]:
        """
        Calculate statistics from a batch of anomaly detection results.
        
        Args:
            results: List of AnomalyResult objects
        
        Returns:
            Dictionary with statistics
        """
        total = len(results)
        if total == 0:
            return {"error": "No results to analyze"}
        
        anomalies = [r for r in results if r.is_anomaly]
        
        severity_counts = {
            "normal": 0, "low": 0, "medium": 0, "high": 0, "critical": 0
        }
        for r in results:
            sev = r.severity if r.is_anomaly else "normal"
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        type_counts = {}
        for r in results:
            type_counts[r.anomaly_type] = type_counts.get(r.anomaly_type, 0) + 1
        
        return {
            "total_analyzed": total,
            "total_anomalies": len(anomalies),
            "anomaly_rate": len(anomalies) / total,
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "average_anomaly_score": np.mean([r.anomaly_score for r in results]),
            "min_score": min(r.anomaly_score for r in results),
            "max_score": max(r.anomaly_score for r in results)
        }


# Factory function
def create_detector(model_type: str = "isolation_forest") -> AnomalyDetector:
    """
    Create an anomaly detector instance.
    
    Args:
        model_type: Type of anomaly detection model
    
    Returns:
        AnomalyDetector instance
    """
    return AnomalyDetector(model_type=model_type)


# Convenience function
def detect_anomaly(prompt: str, energy_kwh: Optional[float] = None) -> Dict[str, Any]:
    """
    Quick function to detect anomalies in a prompt.
    
    Args:
        prompt: Input prompt text
        energy_kwh: Optional energy estimate
    
    Returns:
        Dictionary with detection results
    """
    detector = AnomalyDetector()
    result = detector.detect(prompt, energy_kwh)
    return result.to_dict()


if __name__ == "__main__":
    # Test the detector
    test_prompts = [
        "Hello, how are you?",  # Normal
        "Explain quantum computing briefly.",  # Normal
        """Write a comprehensive 5000-word essay covering every aspect of 
        quantum computing including historical development, theoretical 
        foundations, current implementations, future applications, and 
        detailed mathematical formulations of quantum gates, with code 
        examples in Python, C++, and Qiskit.""",  # Potentially anomalous
        "a" * 1000  # Definitely anomalous
    ]
    
    detector = AnomalyDetector()
    
    print("Anomaly Detection Results")
    print("=" * 60)
    
    for prompt in test_prompts:
        result = detector.detect(prompt)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Is Anomaly: {result.is_anomaly}")
        print(f"  Score: {result.anomaly_score:.4f}")
        print(f"  Type: {result.anomaly_type}")
        print(f"  Severity: {result.severity}")
        print(f"  Factors: {result.contributing_factors}")
        print(f"  Recommendation: {result.recommendation[:80]}...")
