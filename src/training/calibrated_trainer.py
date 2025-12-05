"""
Calibrated Model Training with Real Measurement Grounding.

This script implements the advisor's recommended hybrid approach:
1. Uses real energy measurements as ground truth
2. Creates calibrated synthetic data that matches real hardware behavior
3. Trains model on calibrated data for accurate real-world predictions

Key Insight:
- Real measurements showed 18.5x higher energy than synthetic estimates
- Real correlation (tokens vs energy): 0.9333 (excellent!)
- Model correlation with real data: 0.9574 (model ranks correctly)

Calibration Approach:
- Scale synthetic energy values by calibration factor from real measurements
- Derive energy formula from actual real measurement regression
- Generate 2000-3000 calibrated samples for robust training
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from nlp.parser import parse_prompt
from nlp.complexity_score import compute_complexity
from utils.config import MODEL_DIR


class CalibratedModelTrainer:
    """
    Train energy prediction model calibrated to real measurements.
    
    This implements the advisor's recommended hybrid approach:
    - Real measurements provide ground truth calibration
    - Synthetic data provides volume for robust training
    - Combined approach gives accurate real-world predictions
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the calibrated trainer."""
        self.output_dir = output_dir or MODEL_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths
        self.real_measurements_path = Path(__file__).parent.parent.parent / "data" / "validation" / "real_measurements.csv"
        self.calibrated_data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "calibrated_training_data.csv"
        
        # Calibration parameters (to be derived from real measurements)
        self.calibration_factor = None
        self.energy_intercept = None
        self.energy_slope = None
        self.real_stats = {}
        
    def load_real_measurements(self) -> pd.DataFrame:
        """Load real measurements for calibration."""
        if not self.real_measurements_path.exists():
            raise FileNotFoundError(
                f"Real measurements not found at {self.real_measurements_path}. "
                "Run collect_real_measurements.py first."
            )
        
        df = pd.read_csv(self.real_measurements_path)
        print(f"Loaded {len(df)} real measurements")
        print(f"  Token range: {df['token_count'].min()} - {df['token_count'].max()}")
        print(f"  Energy range: {df['energy_joules'].min():.2f} - {df['energy_joules'].max():.2f} J")
        
        return df
    
    def derive_calibration(self, real_df: pd.DataFrame) -> Dict:
        """
        Derive calibration parameters from real measurements.
        
        Fits a linear regression to understand the real token-energy relationship,
        then uses this to scale synthetic data.
        
        Returns:
            Dictionary with calibration parameters
        """
        print("\n" + "="*60)
        print("DERIVING CALIBRATION FROM REAL MEASUREMENTS")
        print("="*60)
        
        # Fit linear regression: energy = intercept + slope * tokens
        X = real_df['token_count'].values.reshape(-1, 1)
        y = real_df['energy_joules'].values
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        self.energy_intercept = reg.intercept_
        self.energy_slope = reg.coef_[0]
        
        # Calculate R² of linear fit
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Calculate correlation
        correlation = np.corrcoef(real_df['token_count'], real_df['energy_joules'])[0, 1]
        
        # Store real stats for reference
        self.real_stats = {
            'min_tokens': real_df['token_count'].min(),
            'max_tokens': real_df['token_count'].max(),
            'min_energy': real_df['energy_joules'].min(),
            'max_energy': real_df['energy_joules'].max(),
            'mean_energy': real_df['energy_joules'].mean(),
            'energy_intercept': self.energy_intercept,
            'energy_slope': self.energy_slope,
            'linear_r2': r2,
            'correlation': correlation,
        }
        
        print(f"\nReal Measurement Statistics:")
        print(f"  Token range: {self.real_stats['min_tokens']} - {self.real_stats['max_tokens']}")
        print(f"  Energy range: {self.real_stats['min_energy']:.2f} - {self.real_stats['max_energy']:.2f} J")
        print(f"\nLinear Calibration Formula:")
        print(f"  Energy (J) = {self.energy_intercept:.4f} + {self.energy_slope:.4f} × tokens")
        print(f"  R² of linear fit: {r2:.4f}")
        print(f"  Correlation: {correlation:.4f}")
        
        # Compare with old synthetic formula
        # Old: E = 0.1 + 0.015 * tokens * multipliers
        # New: E = intercept + slope * tokens
        old_slope = 0.015  # From original synthetic formula
        self.calibration_factor = self.energy_slope / old_slope
        
        print(f"\nCalibration Factor: {self.calibration_factor:.2f}x")
        print(f"  (Real energy is ~{self.calibration_factor:.1f}x higher than original synthetic)")
        
        return self.real_stats
    
    def generate_calibrated_synthetic_data(
        self,
        num_samples: int = 2500,
        token_range: Tuple[int, int] = (5, 250)
    ) -> pd.DataFrame:
        """
        Generate synthetic training data calibrated to real measurements.
        
        Uses the derived calibration formula instead of arbitrary synthetic values.
        
        Args:
            num_samples: Number of samples to generate
            token_range: (min, max) token range
            
        Returns:
            DataFrame with calibrated training data
        """
        print("\n" + "="*60)
        print(f"GENERATING {num_samples} CALIBRATED SYNTHETIC SAMPLES")
        print("="*60)
        
        if self.energy_slope is None:
            raise ValueError("Run derive_calibration() first!")
        
        # Prompt templates for variety
        templates = {
            "simple": [
                "What is {topic}?",
                "Define {topic}.",
                "Explain {topic}.",
                "Tell me about {topic}.",
            ],
            "question": [
                "How can I improve {topic}? What are the best practices?",
                "What are the key considerations for {topic}?",
                "How do I get started with {topic} development?",
            ],
            "explanation": [
                "Explain how {topic} works in detail, including the key components and their interactions.",
                "Describe the complete process of {topic}, from start to finish with all important steps.",
                "Provide a comprehensive explanation of {topic} covering theory and practical applications.",
            ],
            "comparison": [
                "Compare and contrast {topic1} and {topic2} across multiple dimensions including performance, ease of use, and scalability.",
                "What are the key differences between {topic1} and {topic2}? When should I use each one?",
            ],
            "coding": [
                "Write a Python function that implements {task}. Include proper error handling, documentation, and example usage.",
                "Create a complete implementation of {task} with unit tests and performance considerations.",
            ],
            "complex": [
                "Design a comprehensive system for {task}. Include architecture diagrams, component interactions, scalability considerations, and deployment strategies.",
                "Analyze the trade-offs in implementing {task}. Consider factors like performance, maintainability, cost, and team expertise. Provide recommendations for different scenarios.",
                "Develop a detailed strategy for {task} covering initial requirements gathering, architecture design, implementation phases, testing methodology, deployment approach, and ongoing maintenance procedures.",
            ],
            "extra_long": [
                """Provide an exhaustive analysis of {topic1} compared to {topic2}, covering: 
                1) Historical context and evolution of both approaches
                2) Core architectural differences and design philosophies  
                3) Performance benchmarks across different use cases
                4) Scalability characteristics for small, medium, and large deployments
                5) Developer experience including learning curve, debugging tools, and community support
                6) Integration capabilities with existing systems and third-party tools
                7) Cost considerations including licensing, infrastructure, and maintenance
                8) Security implications and best practices
                9) Future roadmap and industry adoption trends
                10) Concrete recommendations based on project requirements and team expertise.""",
                """Create a comprehensive guide to implementing {task} that includes:
                - Problem definition and requirements analysis
                - Survey of existing approaches and their limitations  
                - Proposed solution architecture with justification
                - Detailed implementation plan broken into phases
                - Data pipeline design for training and inference
                - Model selection criteria and evaluation methodology
                - Hyperparameter tuning strategy
                - Deployment options (cloud, on-premise, edge)
                - Monitoring and alerting setup
                - CI/CD pipeline for model updates
                - A/B testing framework for production evaluation
                - Documentation requirements and knowledge transfer
                - Cost estimation and resource planning
                - Risk assessment and mitigation strategies"""
            ]
        }
        
        topics = [
            "machine learning", "neural networks", "deep learning", "Python programming",
            "data science", "natural language processing", "computer vision", "reinforcement learning",
            "API development", "database optimization", "cloud computing", "DevOps practices",
            "microservices architecture", "containerization", "Kubernetes", "CI/CD pipelines",
            "test automation", "code review", "agile methodology", "software design patterns",
            "data pipelines", "feature engineering", "model deployment", "MLOps",
            "distributed systems", "message queues", "caching strategies", "load balancing",
            "authentication systems", "API security", "encryption", "monitoring and observability"
        ]
        
        tasks = [
            "data preprocessing", "feature extraction", "model training", "hyperparameter tuning",
            "cross-validation", "performance optimization", "memory management", "parallel processing",
            "real-time prediction service", "batch processing pipeline", "data quality monitoring",
            "automated testing framework", "logging and error handling", "configuration management"
        ]
        
        samples = []
        categories = list(templates.keys())
        
        # Weight categories to ensure variety (more complex prompts = more tokens)
        category_weights = {
            "simple": 0.10,
            "question": 0.15,
            "explanation": 0.20,
            "comparison": 0.15,
            "coding": 0.15,
            "complex": 0.15,
            "extra_long": 0.10
        }
        
        for i in range(num_samples):
            # Sample category based on weights
            category = np.random.choice(
                categories, 
                p=[category_weights[c] for c in categories]
            )
            
            # Select template
            template = np.random.choice(templates[category])
            
            # Fill in template
            topic1 = np.random.choice(topics)
            topic2 = np.random.choice([t for t in topics if t != topic1])
            task = np.random.choice(tasks)
            
            prompt = template.format(
                topic=topic1,
                topic1=topic1,
                topic2=topic2,
                task=task
            )
            
            # Parse features
            try:
                parsed = parse_prompt(prompt, use_embeddings=False)
                complexity = compute_complexity(prompt)
                
                token_count = parsed.token_count
                word_count = parsed.word_count
                char_count = parsed.char_count
                avg_word_length = parsed.avg_word_length
                avg_sentence_length = parsed.avg_sentence_length
                unique_word_ratio = getattr(parsed, 'unique_word_ratio', 0.7)
            except Exception:
                # Fallback to simple calculation
                words = prompt.split()
                token_count = len(words) + 2
                word_count = len(words)
                char_count = len(prompt)
                avg_word_length = np.mean([len(w) for w in words]) if words else 4.5
                avg_sentence_length = len(words)
                unique_word_ratio = len(set(words)) / len(words) if words else 0.7
                complexity = 0.5
            
            # Calculate CALIBRATED energy using real measurement formula
            # Base energy from linear regression
            base_energy = self.energy_intercept + self.energy_slope * token_count
            
            # Add complexity effect (proportional, not multiplicative)
            complexity_effect = complexity * 2.0  # Complexity adds up to 2 Joules
            
            # Add category effect
            category_multipliers = {
                "simple": 0.9,
                "question": 1.0,
                "explanation": 1.1,
                "comparison": 1.15,
                "coding": 1.2,
                "complex": 1.25,
                "extra_long": 1.1
            }
            category_mult = category_multipliers.get(category, 1.0)
            
            # Calculate final energy
            energy = (base_energy + complexity_effect) * category_mult
            
            # Add realistic noise (±10%)
            noise = np.random.uniform(0.90, 1.10)
            energy *= noise
            
            # Ensure energy is positive and reasonable
            energy = max(0.5, energy)
            
            samples.append({
                'prompt_id': f'CAL{i+1:05d}',
                'prompt': prompt,
                'category': category,
                'token_count': token_count,
                'word_count': word_count,
                'char_count': char_count,
                'complexity_score': complexity,
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'unique_word_ratio': unique_word_ratio,
                'energy_joules': round(energy, 4),
            })
        
        df = pd.DataFrame(samples)
        
        # Save
        df.to_csv(self.calibrated_data_path, index=False)
        print(f"\nSaved calibrated data to: {self.calibrated_data_path}")
        
        # Print statistics
        print(f"\nCalibrated Dataset Statistics:")
        print(f"  Samples: {len(df)}")
        print(f"  Token range: {df['token_count'].min()} - {df['token_count'].max()}")
        print(f"  Energy range: {df['energy_joules'].min():.2f} - {df['energy_joules'].max():.2f} J")
        print(f"  Mean energy: {df['energy_joules'].mean():.2f} J")
        print(f"\n  Category distribution:")
        print(df['category'].value_counts().to_string())
        
        # Verify correlation
        corr = df['token_count'].corr(df['energy_joules'])
        print(f"\n  Token-Energy correlation: {corr:.4f}")
        
        return df
    
    def train_model(self, df: pd.DataFrame) -> Tuple[object, object, Dict]:
        """
        Train Gradient Boosting model on calibrated data.
        
        Args:
            df: Calibrated training data
            
        Returns:
            (model, scaler, metrics)
        """
        print("\n" + "="*60)
        print("TRAINING CALIBRATED MODEL")
        print("="*60)
        
        # Prepare features
        feature_cols = [
            'token_count', 'word_count', 'char_count',
            'complexity_score', 'avg_word_length', 'avg_sentence_length'
        ]
        
        X = df[feature_cols].values
        y = df['energy_joules'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Gradient Boosting model
        print("\nTraining Gradient Boosting Regressor...")
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        print(f"\nModel Performance:")
        print(f"  Training R²: {metrics['train_r2']:.4f}")
        print(f"  Test R²:     {metrics['test_r2']:.4f}")
        print(f"  CV R² Mean:  {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        print(f"  Test MAE:    {metrics['test_mae']:.4f} J")
        print(f"  Test RMSE:   {metrics['test_rmse']:.4f} J")
        
        # Feature importance
        print(f"\nFeature Importance:")
        importance = dict(zip(feature_cols, model.feature_importances_))
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {feat}: {imp:.4f}")
        
        return model, scaler, metrics
    
    def save_model(self, model, scaler, metrics: Dict):
        """Save the calibrated model and associated files."""
        print("\n" + "="*60)
        print("SAVING CALIBRATED MODEL")
        print("="*60)
        
        # Create energy_predictor subdirectory
        model_dir = self.output_dir / "energy_predictor"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model with energy_predictor naming
        model_path = model_dir / "energy_predictor.joblib"
        joblib.dump(model, model_path)
        print(f"  Model saved to: {model_path}")
        
        # Save scaler
        scaler_path = model_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved to: {scaler_path}")
        
        # Save calibration info
        calibration_info = {
            'calibration_factor': self.calibration_factor,
            'energy_intercept': self.energy_intercept,
            'energy_slope': self.energy_slope,
            **self.real_stats,
            **metrics
        }
        
        info_path = model_dir / "calibration_info.joblib"
        joblib.dump(calibration_info, info_path)
        print(f"  Calibration info saved to: {info_path}")
        
        # Also save as CSV for human readability
        info_df = pd.DataFrame([calibration_info])
        info_csv_path = model_dir / "calibration_info.csv"
        info_df.to_csv(info_csv_path, index=False)
        print(f"  Calibration CSV saved to: {info_csv_path}")
        
        return model_path, scaler_path
    
    def run_full_pipeline(self, num_samples: int = 2500) -> Dict:
        """
        Run the full calibrated training pipeline.
        
        Steps:
        1. Load real measurements
        2. Derive calibration parameters
        3. Generate calibrated synthetic data
        4. Train model
        5. Save model
        
        Args:
            num_samples: Number of synthetic samples to generate
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*70)
        print("CALIBRATED MODEL TRAINING PIPELINE")
        print("Hybrid approach: Real measurements + Calibrated synthetic data")
        print("="*70)
        
        # Step 1: Load real measurements
        real_df = self.load_real_measurements()
        
        # Step 2: Derive calibration
        self.derive_calibration(real_df)
        
        # Step 3: Generate calibrated synthetic data
        synth_df = self.generate_calibrated_synthetic_data(num_samples=num_samples)
        
        # Step 4: Train model
        model, scaler, metrics = self.train_model(synth_df)
        
        # Step 5: Save model
        model_path, scaler_path = self.save_model(model, scaler, metrics)
        
        print("\n" + "="*70)
        print("CALIBRATED TRAINING COMPLETE!")
        print("="*70)
        print(f"\nResults Summary:")
        print(f"  Calibration Factor: {self.calibration_factor:.2f}x")
        print(f"  Energy Formula: E = {self.energy_intercept:.2f} + {self.energy_slope:.4f} × tokens")
        print(f"  Training Samples: {num_samples}")
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        print(f"  Test MAE: {metrics['test_mae']:.2f} J")
        print(f"\nModel files saved to: {self.output_dir}")
        
        return {
            'calibration_factor': self.calibration_factor,
            'real_stats': self.real_stats,
            'metrics': metrics,
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
        }


def main():
    """Run calibrated model training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train calibrated energy model")
    parser.add_argument('--samples', type=int, default=2500,
                        help='Number of synthetic samples (default: 2500)')
    args = parser.parse_args()
    
    trainer = CalibratedModelTrainer()
    results = trainer.run_full_pipeline(num_samples=args.samples)
    
    return results


if __name__ == "__main__":
    main()
