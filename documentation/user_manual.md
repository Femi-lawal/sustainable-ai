# Sustainable AI Energy Monitor - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Features](#features)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

The **Sustainable AI Energy Monitor** is a comprehensive tool for measuring, predicting, and optimizing the environmental impact of Large Language Model (LLM) prompts. It provides real-time energy consumption estimates, carbon footprint calculations, and suggestions for more efficient prompt design.

### Key Features

- 🔋 **Energy Prediction**: Calibrated ML model (R²=0.9813, MAPE=6.8%)
- 🌱 **Carbon Footprint**: Calculate CO2 emissions and water usage
- 🔍 **Anomaly Detection**: Identify unusually resource-intensive prompts
- ✏️ **Prompt Optimization**: Suggestions for 8-43% energy savings
- 📊 **EU Compliance**: Generate transparency reports for AI Act compliance
- 📈 **Dashboard**: Interactive Streamlit visualization
- ✅ **Professional Validation**: Tested with 100 real energy measurements

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)

### Step-by-Step Installation

1. **Clone or Download the Project**

   ```bash
   git clone <repository-url>
   cd Sustainable_AI_G3--main
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "from src.prediction.estimator import EnergyPredictor; print('Installation successful!')"
   ```

---

## Quick Start

### Using the GUI (Recommended)

1. **Start the Streamlit Application**

   ```bash
   # Windows
   .\start.bat
   # or
   streamlit run src/gui/app.py
   ```

2. **Open in Browser**
   Navigate to `http://localhost:8501`

3. **Enter Your Prompt**
   Type or paste your LLM prompt in the text area

4. **Analyze**
   Click "Analyze Prompt" to get energy predictions

### Using the Command Line

```python
from src.prediction.estimator import predict_energy

result = predict_energy("What is the capital of France?")
print(f"Energy: {result['energy_kwh']:.6f} kWh")
print(f"Carbon: {result['carbon_footprint_kg']:.6f} kg CO2")
```

---

## Features

### 1. Energy Prediction

The energy predictor uses a **Calibrated Gradient Boosting model** trained on hybrid data (synthetic + 100 real measurements):

**Professional Validation Results:**

- **R² Score**: 0.9813 (98.1% variance explained)
- **MAPE**: 6.8% (professional standard: <25%)
- **Prediction Bias**: 0.9988 (target: 0.90-1.10)
- **Within 20%**: 94% of predictions

```python
from src.prediction.estimator import EnergyPredictor

predictor = EnergyPredictor()
result = predictor.predict(
    prompt="Explain quantum computing",
    num_layers=24,        # LLM model layers
    training_hours=8.0,   # Training time
    flops_per_hour=1e11   # Compute operations
)

print(f"Energy: {result.energy_kwh} kWh")
print(f"Level: {result.energy_level}")
print(f"Confidence: {result.confidence_score:.2%}")
```

### 2. Anomaly Detection

Identify prompts that consume unusual amounts of resources:

```python
from src.anomaly.detector import AnomalyDetector

detector = AnomalyDetector()
result = detector.detect("Your very long and complex prompt here...")

if result.is_anomaly:
    print(f"⚠️ Anomaly detected!")
    print(f"Type: {result.anomaly_type}")
    print(f"Severity: {result.severity}")
    print(f"Recommendation: {result.recommendation}")
```

### 3. Prompt Optimization

Get suggestions for more efficient prompts:

```python
from src.optimization.recomender import PromptOptimizer

optimizer = PromptOptimizer()
suggestions = optimizer.optimize(
    "Please write a very comprehensive and detailed explanation..."
)

for suggestion in suggestions:
    print(f"- {suggestion.description}")
    print(f"  Potential savings: {suggestion.energy_savings:.1%}")
```

### 4. Text Simplification

Automatically simplify complex prompts:

```python
from src.nlp.simplifier import TextSimplifier

simplifier = TextSimplifier()
simplified = simplifier.simplify("Subsequently, utilize the aforementioned methodology...")
print(f"Simplified: {simplified}")
# Output: "Then, use this method..."
```

### 5. Complexity Analysis

Measure prompt complexity:

```python
from src.nlp.complexity_score import compute_complexity

complexity = compute_complexity("Your prompt text here")
print(f"Complexity: {complexity:.2f}")  # 0.0 to 1.0
```

---

## Usage Guide

### Dashboard Overview

The Streamlit dashboard has several sections:

#### Main Analysis Panel

1. **Prompt Input**: Enter your text
2. **Model Configuration**: Set LLM parameters (layers, training hours)
3. **Analyze Button**: Run the analysis

#### Results Display

- **Energy Estimate**: kWh consumption with confidence
- **Environmental Impact**: Carbon footprint, water usage, cost
- **Complexity Score**: Visual gauge of prompt complexity
- **Optimization Suggestions**: Actionable recommendations

#### Reports Tab

- View historical analyses
- Generate EU compliance reports
- Export data as CSV/JSON

### Batch Processing

Process multiple prompts at once:

```python
from src.prediction.estimator import EnergyPredictor

predictor = EnergyPredictor()
prompts = [
    "What is AI?",
    "Explain machine learning in detail",
    "Write a comprehensive essay on deep learning"
]

results = predictor.predict_batch(prompts)
for r in results:
    print(f"{r.prompt[:30]}... -> {r.energy_kwh:.4f} kWh")
```

### Database Logging

All analyses are automatically logged to SQLite:

```python
from src.utils.database import EnergyDatabase

db = EnergyDatabase()

# Log a prediction
db.log_prediction(
    prompt="Your prompt",
    energy_kwh=0.125,
    carbon_kg=0.052,
    features={"token_count": 15, "complexity": 0.35}
)

# Query history
history = db.get_recent_predictions(limit=10)
```

---

## API Reference

### EnergyPredictor

| Method                             | Description                        |
| ---------------------------------- | ---------------------------------- |
| `predict(prompt, **kwargs)`        | Predict energy for a single prompt |
| `predict_batch(prompts, **kwargs)` | Predict for multiple prompts       |
| `extract_features(prompt)`         | Get feature dictionary             |
| `train(data, target_column)`       | Train/retrain the model            |
| `get_feature_importance()`         | Get feature importance scores      |

### AnomalyDetector

| Method                            | Description                   |
| --------------------------------- | ----------------------------- |
| `detect(prompt, energy_kwh)`      | Detect if prompt is anomalous |
| `detect_batch(prompts, energies)` | Batch anomaly detection       |
| `train(training_data)`            | Train on normal samples       |
| `get_anomaly_statistics(results)` | Get batch statistics          |

### TextSimplifier

| Method                                        | Description                    |
| --------------------------------------------- | ------------------------------ |
| `simplify(text)`                              | Simplify complex text          |
| `get_suggestions(text)`                       | Get simplification suggestions |
| `calculate_improvement(original, simplified)` | Measure improvement            |

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Data center region (affects carbon calculations)
DATA_CENTER_REGION=california

# Model parameters
MODEL_LAYERS=24
DEFAULT_TRAINING_HOURS=8.0
DEFAULT_FLOPS=1e11

# Database path
DATABASE_PATH=data/energy_logs.db

# Logging level
LOG_LEVEL=INFO
```

### Config File

Edit `src/utils/config.py` for advanced settings:

```python
# Energy prediction model
ENERGY_PREDICTOR_CONFIG = EnergyPredictorConfig(
    n_estimators=100,
    max_depth=10,
    model_path=MODEL_DIR / "energy_predictor" / "energy_predictor.joblib"
)

# Anomaly detection
ANOMALY_DETECTOR_CONFIG = AnomalyDetectorConfig(
    contamination=0.1,  # Expected anomaly rate
    n_estimators=100
)

# Carbon intensity by region (kg CO2 per kWh)
DATA_CENTER_CONFIG = DataCenterConfig(
    carbon_intensity={
        "california": 0.42,
        "texas": 0.75,
        "virginia": 0.62,
        "europe": 0.35
    }
)
```

---

## Troubleshooting

### Common Issues

#### 1. Model Not Found Error

```
Error: Could not load model: [Errno 2] No such file or directory
```

**Solution**: Run the model creation script:

```bash
python model/create_models.py
```

#### 2. Import Errors

```
ModuleNotFoundError: No module named 'transformers'
```

**Solution**: Install missing dependencies:

```bash
pip install -r requirements.txt
```

#### 3. Streamlit Port Already in Use

```
Port 8501 is already in use
```

**Solution**: Use a different port:

```bash
streamlit run src/gui/app.py --server.port 8502
```

#### 4. Database Locked

```
sqlite3.OperationalError: database is locked
```

**Solution**: Close other connections to the database or restart the application.

### Performance Tips

1. **Use Batch Processing**: Process multiple prompts together for better throughput
2. **Enable Caching**: Set `use_cache=True` in predictor initialization
3. **Reduce Model Complexity**: Use `model_type="random_forest"` for faster predictions

### Getting Help

- Check the [README.md](../README.md) for quick start
- Review the [Architecture Documentation](architecture.md) for system design
- Open an issue on GitHub for bugs or feature requests

---

## Appendix

### Supported Regions for Carbon Calculations

| Region           | Carbon Intensity (kg CO2/kWh) |
| ---------------- | ----------------------------- |
| California       | 0.42                          |
| Texas            | 0.75                          |
| Virginia         | 0.62                          |
| Oregon           | 0.31                          |
| Europe (Average) | 0.35                          |
| Nordic Countries | 0.15                          |
| China            | 0.92                          |
| India            | 0.82                          |

### Energy Level Thresholds (Science-Backed)

**Based on 100 real energy measurements using CodeCarbon:**

| Level         | Joules Range | kWh Equivalent    | Description           | Typical Prompts                    |
| ------------- | ------------ | ----------------- | --------------------- | ---------------------------------- |
| **Low**       | ≤ 10 J       | ≤ 2.78e-6 kWh     | Efficient prompt      | Definitions, simple questions      |
| **Medium**    | 10 - 25 J    | 2.78e-6 - 6.94e-6 | Normal consumption    | Explanations, short analysis       |
| **High**      | 25 - 50 J    | 6.94e-6 - 1.39e-5 | Above average         | Detailed explanations              |
| **Very High** | > 50 J       | > 1.39e-5 kWh     | Consider optimization | Comprehensive analysis, multi-part |

**Methodology**: Thresholds derived from real measurement data:

- Simple prompts (3.4-10.6 J, mean 5.7 J)
- Medium prompts (10.3-20.1 J, mean 13.8 J)
- Long prompts (25.5-36.1 J, mean 29.6 J)
- Very long prompts (42.9-73.2 J, mean 51.9 J)
- Extra long prompts (56.0-79.3 J, mean 67.3 J)

### Carbon Display Units

Carbon footprint is displayed in **milligrams (mg)** for per-prompt values because:

- Per-prompt carbon values are extremely small (e.g., 0.5-2.0 mg CO₂)
- "1.03e-06 kg" is confusing; "1.03 mg" is human-readable
- Milligrams provide meaningful comparison between prompts

---

_Document Version: 2.0_
_Last Updated: December 2025_
_For CSCN8010 Final Project - Sustainable AI Energy Monitor_
_Model Performance: R²=0.9813, MAPE=6.8%, 283 Tests Passing_
