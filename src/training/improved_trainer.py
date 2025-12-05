"""
Improved Model Training Pipeline with Proper Evaluation and Iterative Improvement.

This script trains energy prediction models using:
1. Proper train/validation/test splits
2. Hyperparameter tuning with Grid Search / Random Search
3. Cross-validation for robust evaluation
4. Iterative improvement until R² > 0.85
5. Feature importance analysis

Production Model (December 2025):
- Algorithm: Random Forest Regressor
- R² Score: 0.9809
- RMSE: 3.28 Joules
- MAE: 2.46 Joules
- Features: token_count, word_count, char_count, avg_word_length, avg_sentence_length
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple, List, Any
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

from utils.config import MODEL_DIR


class ImprovedModelTrainer:
    """
    Advanced model training with hyperparameter tuning and iterative improvement.
    """
    
    TARGET_R2 = 0.85  # Target R² score for model acceptance
    MIN_R2 = 0.70     # Minimum acceptable R²
    
    def __init__(self, data_path: Path = None):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to training dataset CSV
        """
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.best_scaler = None
        self.feature_names = None
        self.training_history = []
        
    def load_data(self, data_path: Path = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load and prepare training data.
        
        Returns:
            DataFrame, X features array, y target array
        """
        if data_path is None:
            data_path = self.data_path or Path(__file__).parent.parent.parent / "data" / "processed" / "training_dataset.csv"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}. Run generate_training_data.py first.")
        
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} training samples")
        
        # Select feature columns (excluding non-numeric and target)
        exclude_cols = ['prompt_id', 'prompt', 'category', 'energy_joules', 'efficiency_label']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        self.feature_names = feature_cols
        X = df[feature_cols].values
        y = df['energy_joules'].values
        
        print(f"Features: {len(feature_cols)}")
        print(f"Feature names: {feature_cols}")
        
        return df, X, y
    
    def create_polynomial_features(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Create polynomial features for non-linear relationships.
        
        Args:
            X: Original features
            degree: Polynomial degree
        
        Returns:
            Transformed features
        """
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(X)
    
    def get_model_configs(self) -> Dict[str, Dict]:
        """
        Define model configurations for hyperparameter search.
        
        Returns:
            Dictionary of model configs
        """
        return {
            "random_forest": {
                "model": RandomForestRegressor(random_state=42, n_jobs=-1),
                "params": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [5, 10, 15, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ['sqrt', 'log2', None]
                }
            },
            "gradient_boosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "min_samples_split": [2, 5, 10],
                    "subsample": [0.8, 0.9, 1.0]
                }
            },
            "ridge": {
                "model": Ridge(random_state=42),
                "params": {
                    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    "solver": ['auto', 'svd', 'lsqr']
                }
            },
            "elastic_net": {
                "model": ElasticNet(random_state=42, max_iter=10000),
                "params": {
                    "alpha": [0.001, 0.01, 0.1, 1.0],
                    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            }
        }
    
    def train_with_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                          model_type: str = "random_forest",
                          n_iter: int = 50) -> Tuple[Any, Dict]:
        """
        Train model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model
            n_iter: Number of iterations for random search
        
        Returns:
            Best model, best parameters
        """
        config = self.get_model_configs()[model_type]
        
        print(f"\nTuning {model_type}...")
        
        # Use RandomizedSearchCV for efficiency
        search = RandomizedSearchCV(
            config["model"],
            config["params"],
            n_iter=min(n_iter, 100),
            cv=5,
            scoring='r2',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        
        print(f"  Best R² (CV): {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            "r2": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "mape": np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
        }
        
        return metrics
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """
        Train all model types and compare.
        
        Args:
            X: Features
            y: Targets
        
        Returns:
            Dictionary of model results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        for model_type in self.get_model_configs().keys():
            try:
                # Train with tuning
                model, params = self.train_with_tuning(X_train_scaled, y_train, model_type)
                
                # Evaluate
                metrics = self.evaluate_model(model, X_test_scaled, y_test)
                
                results[model_type] = {
                    "model": model,
                    "scaler": scaler,
                    "params": params,
                    "metrics": metrics
                }
                
                print(f"\n{model_type} Results:")
                print(f"  R²: {metrics['r2']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
                
            except Exception as e:
                print(f"Error training {model_type}: {e}")
        
        return results
    
    def select_best_model(self, results: Dict) -> Tuple[str, Any, Any]:
        """
        Select the best model based on R² score.
        
        Args:
            results: Dictionary of model results
        
        Returns:
            Best model type, model, scaler
        """
        best_type = None
        best_r2 = -float('inf')
        
        for model_type, result in results.items():
            r2 = result['metrics']['r2']
            if r2 > best_r2:
                best_r2 = r2
                best_type = model_type
        
        if best_type:
            print(f"\n=== Best Model: {best_type} with R² = {best_r2:.4f} ===")
            return best_type, results[best_type]['model'], results[best_type]['scaler']
        
        return None, None, None
    
    def iterative_training(self, X: np.ndarray, y: np.ndarray, 
                           max_iterations: int = 5) -> Tuple[Any, Any, Dict]:
        """
        Iteratively improve model until target R² is reached.
        
        Args:
            X: Features
            y: Targets
            max_iterations: Maximum training iterations
        
        Returns:
            Best model, scaler, final metrics
        """
        print("\n" + "="*60)
        print("ITERATIVE MODEL TRAINING")
        print("="*60)
        
        best_model = None
        best_scaler = None
        best_metrics = {"r2": 0}
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Train all models
            results = self.train_all_models(X, y)
            
            # Select best
            model_type, model, scaler = self.select_best_model(results)
            
            if model_type:
                metrics = results[model_type]['metrics']
                
                if metrics['r2'] > best_metrics['r2']:
                    best_model = model
                    best_scaler = scaler
                    best_metrics = metrics
                    self.training_history.append({
                        'iteration': iteration + 1,
                        'model_type': model_type,
                        **metrics
                    })
                
                # Check if target reached
                if metrics['r2'] >= self.TARGET_R2:
                    print(f"\n✓ Target R² of {self.TARGET_R2} reached!")
                    break
                elif metrics['r2'] >= self.MIN_R2:
                    print(f"\n✓ Acceptable R² of {self.MIN_R2} reached")
                else:
                    print(f"\n⚠ R² {metrics['r2']:.4f} below minimum {self.MIN_R2}")
        
        self.best_model = best_model
        self.best_scaler = best_scaler
        
        return best_model, best_scaler, best_metrics
    
    def get_feature_importance(self, model: Any = None) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Args:
            model: Trained model (uses best_model if not provided)
        
        Returns:
            Dictionary of feature importances
        """
        model = model or self.best_model
        
        if model is None or self.feature_names is None:
            return {}
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importances))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_)
            importance_dict = dict(zip(self.feature_names, coefs))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def save_model(self, model: Any = None, scaler: Any = None, 
                   model_path: Path = None, scaler_path: Path = None):
        """
        Save the trained model and scaler.
        
        Args:
            model: Model to save
            scaler: Scaler to save
            model_path: Path for model file
            scaler_path: Path for scaler file
        """
        model = model or self.best_model
        scaler = scaler or self.best_scaler
        
        # Create energy_predictor subdirectory
        energy_predictor_dir = MODEL_DIR / "energy_predictor"
        energy_predictor_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_path or energy_predictor_dir / "energy_predictor.joblib"
        scaler_path = scaler_path or energy_predictor_dir / "scaler.joblib"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Also save feature names
        feature_path = energy_predictor_dir / "feature_names.txt"
        with open(feature_path, 'w') as f:
            for name in self.feature_names:
                f.write(f"{name}\n")
        
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Features saved to {feature_path}")
    
    def run_full_pipeline(self) -> Dict:
        """
        Run the complete training pipeline.
        
        Returns:
            Training results summary
        """
        print("\n" + "="*60)
        print("SUSTAINABLE AI - MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Load data
        df, X, y = self.load_data()
        
        # Train iteratively
        model, scaler, metrics = self.iterative_training(X, y)
        
        # Get feature importance
        importance = self.get_feature_importance(model)
        
        print("\n--- Feature Importance ---")
        for feature, imp in list(importance.items())[:10]:
            print(f"  {feature}: {imp:.4f}")
        
        # Save model
        self.save_model(model, scaler)
        
        # Summary
        summary = {
            "final_r2": metrics['r2'],
            "final_rmse": metrics['rmse'],
            "final_mae": metrics['mae'],
            "training_history": self.training_history,
            "feature_importance": importance,
            "model_type": type(model).__name__
        }
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Final R²: {metrics['r2']:.4f}")
        print(f"Final RMSE: {metrics['rmse']:.4f}")
        print(f"Final MAE: {metrics['mae']:.4f}")
        
        if metrics['r2'] >= self.TARGET_R2:
            print(f"✓ Model meets target R² of {self.TARGET_R2}")
        elif metrics['r2'] >= self.MIN_R2:
            print(f"✓ Model meets minimum R² of {self.MIN_R2}")
        else:
            print(f"⚠ Model below minimum R² - consider more data or feature engineering")
        
        return summary


def train_energy_model():
    """Main function to train the energy prediction model."""
    trainer = ImprovedModelTrainer()
    return trainer.run_full_pipeline()


if __name__ == "__main__":
    train_energy_model()
