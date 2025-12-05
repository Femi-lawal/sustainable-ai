"""
Script to create and save pre-trained model artifacts.
Generates models for energy prediction, anomaly detection, and NLP tasks.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

# Set up paths
MODEL_DIR = Path(__file__).parent

def create_energy_predictor_model():
    """Create and save a trained energy prediction model."""
    print("Creating energy predictor model...")
    
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 500
    
    # Features
    token_count = np.random.randint(5, 200, n_samples)
    char_count = token_count * np.random.uniform(4, 8, n_samples)
    word_count = token_count * np.random.uniform(0.6, 0.9, n_samples)
    sentence_count = np.maximum(1, word_count / np.random.uniform(8, 20, n_samples))
    avg_word_length = np.random.uniform(4, 8, n_samples)
    avg_sentence_length = word_count / sentence_count
    punct_ratio = np.random.uniform(0.01, 0.05, n_samples)
    stopword_ratio = np.random.uniform(0.15, 0.45, n_samples)
    unique_word_ratio = np.random.uniform(0.5, 0.95, n_samples)
    vocabulary_richness = unique_word_ratio * np.random.uniform(0.8, 1.0, n_samples)
    lexical_density = np.random.uniform(0.3, 0.7, n_samples)
    complexity_score = np.random.uniform(0.1, 0.9, n_samples)
    num_layers = np.random.choice([12, 24, 32, 48], n_samples)
    training_hours = np.random.uniform(1, 20, n_samples)
    flops_per_hour = np.random.uniform(1e9, 1e12, n_samples)
    flops_per_layer = flops_per_hour / num_layers
    training_efficiency = training_hours / num_layers
    
    # Create DataFrame
    X = pd.DataFrame({
        'token_count': token_count,
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'punct_ratio': punct_ratio,
        'stopword_ratio': stopword_ratio,
        'unique_word_ratio': unique_word_ratio,
        'vocabulary_richness': vocabulary_richness,
        'lexical_density': lexical_density,
        'complexity_score': complexity_score,
        'num_layers': num_layers,
        'training_hours': training_hours,
        'flops_per_hour': flops_per_hour,
        'flops_per_layer': flops_per_layer,
        'training_efficiency': training_efficiency
    })
    
    # Target: energy consumption (kWh)
    y = (0.1 + 
         token_count * 0.002 + 
         complexity_score * 0.5 + 
         num_layers * 0.01 + 
         np.log10(flops_per_hour + 1) * 0.05 +
         np.random.normal(0, 0.02, n_samples))
    y = np.maximum(0.01, y)
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    # Save model and scaler
    output_dir = MODEL_DIR / "energy_predictor"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, output_dir / "energy_predictor.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    
    # Save feature names
    with open(output_dir / "feature_names.txt", 'w') as f:
        f.write('\n'.join(X.columns.tolist()))
    
    # Remove placeholder
    placeholder = output_dir / "placeholder.txt"
    if placeholder.exists():
        placeholder.unlink()
    
    print(f"  Saved to {output_dir}")
    return model, scaler


def create_anomaly_detector_model():
    """Create and save a trained anomaly detection model."""
    print("Creating anomaly detector model...")
    
    # Create synthetic training data (normal samples)
    np.random.seed(42)
    n_samples = 500
    
    # Normal pattern features
    token_count = np.random.randint(5, 150, n_samples)
    complexity_score = np.random.uniform(0.1, 0.7, n_samples)
    char_count = token_count * np.random.uniform(4, 7, n_samples)
    avg_word_length = np.random.uniform(4, 7, n_samples)
    vocabulary_richness = np.random.uniform(0.5, 0.9, n_samples)
    lexical_density = np.random.uniform(0.3, 0.6, n_samples)
    token_to_word_ratio = np.random.uniform(1.1, 1.6, n_samples)
    complexity_to_length_ratio = complexity_score / np.maximum(char_count, 1) * 1000
    
    X = pd.DataFrame({
        'token_count': token_count,
        'complexity_score': complexity_score,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'vocabulary_richness': vocabulary_richness,
        'lexical_density': lexical_density,
        'token_to_word_ratio': token_to_word_ratio,
        'complexity_to_length_ratio': complexity_to_length_ratio
    })
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        contamination=0.1,
        n_estimators=100,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)
    
    # Calculate feature statistics for rule-based detection
    feature_stats = {
        col: {
            'mean': float(X[col].mean()),
            'std': float(X[col].std()),
            'q1': float(X[col].quantile(0.25)),
            'q3': float(X[col].quantile(0.75)),
            'iqr': float(X[col].quantile(0.75) - X[col].quantile(0.25))
        }
        for col in X.columns
    }
    
    # Save model
    output_dir = MODEL_DIR / "anomaly_detector"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_stats': feature_stats
    }, output_dir / "anomaly_detector.joblib")
    
    # Save feature names
    with open(output_dir / "feature_names.txt", 'w') as f:
        f.write('\n'.join(X.columns.tolist()))
    
    # Remove placeholder
    placeholder = output_dir / "placeholder.txt"
    if placeholder.exists():
        placeholder.unlink()
    
    print(f"  Saved to {output_dir}")
    return model, scaler


def create_nlp_transformer_config():
    """Create NLP transformer configuration files."""
    print("Creating NLP transformer configuration...")
    
    output_dir = MODEL_DIR / "nlp_transformer"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration for the T5/DistilBERT models used
    config = {
        "tokenizer": "distilbert-base-uncased",
        "simplification_model": "t5-small",
        "max_length": 512,
        "complexity_thresholds": {
            "simple": 0.3,
            "moderate": 0.6,
            "complex": 0.8
        },
        "simplification_strategies": [
            "reduce_sentence_length",
            "replace_complex_words",
            "split_compound_sentences",
            "remove_redundancy"
        ]
    }
    
    import json
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create vocabulary info
    vocab_info = {
        "base_vocab_size": 30522,
        "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        "max_position_embeddings": 512
    }
    
    with open(output_dir / "vocab_info.json", 'w') as f:
        json.dump(vocab_info, f, indent=2)
    
    # Remove placeholder
    placeholder = output_dir / "placeholder.txt"
    if placeholder.exists():
        placeholder.unlink()
    
    print(f"  Saved to {output_dir}")


def create_prompt_optimizer_config():
    """Create prompt optimizer configuration and rules."""
    print("Creating prompt optimizer configuration...")
    
    output_dir = MODEL_DIR / "prompt_optimizer"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    
    # Optimization rules
    optimization_rules = {
        "token_reduction": {
            "max_tokens": 150,
            "strategies": [
                "remove_filler_words",
                "consolidate_repeated_concepts",
                "use_concise_phrasing"
            ]
        },
        "complexity_reduction": {
            "target_complexity": 0.5,
            "strategies": [
                "simplify_vocabulary",
                "shorten_sentences",
                "reduce_nesting"
            ]
        },
        "energy_optimization": {
            "target_efficiency": 0.8,
            "strategies": [
                "batch_similar_requests",
                "cache_common_patterns",
                "use_smaller_models_when_possible"
            ]
        }
    }
    
    with open(output_dir / "optimization_rules.json", 'w') as f:
        json.dump(optimization_rules, f, indent=2)
    
    # Word replacement dictionary for simplification
    word_replacements = {
        "utilize": "use",
        "implement": "do",
        "facilitate": "help",
        "comprehensive": "full",
        "subsequently": "then",
        "approximately": "about",
        "demonstrate": "show",
        "accomplish": "do",
        "endeavor": "try",
        "ascertain": "find out"
    }
    
    with open(output_dir / "word_replacements.json", 'w') as f:
        json.dump(word_replacements, f, indent=2)
    
    # Energy-efficient prompt templates
    efficient_templates = {
        "question": "{topic}?",
        "explanation": "Explain {topic} briefly.",
        "comparison": "Compare {item1} and {item2}.",
        "summary": "Summarize: {content}",
        "list": "List 5 {items}."
    }
    
    with open(output_dir / "efficient_templates.json", 'w') as f:
        json.dump(efficient_templates, f, indent=2)
    
    # Remove placeholder
    placeholder = output_dir / "placeholder.txt"
    if placeholder.exists():
        placeholder.unlink()
    
    print(f"  Saved to {output_dir}")


def main():
    """Create all model artifacts."""
    print("=" * 60)
    print("Creating Model Artifacts for Sustainable AI")
    print("=" * 60)
    
    create_energy_predictor_model()
    create_anomaly_detector_model()
    create_nlp_transformer_config()
    create_prompt_optimizer_config()
    
    print("\n" + "=" * 60)
    print("All model artifacts created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
