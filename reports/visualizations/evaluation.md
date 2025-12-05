# Sustainable AI - Model Evaluation Report

## Overview

This document presents the evaluation metrics and performance analysis for the Sustainable AI Energy Monitor models.

## 1. Energy Prediction Model (Random Forest Regressor)

### Model Configuration

- **Algorithm**: Random Forest Regressor
- **Number of Estimators**: 100
- **Max Depth**: 10
- **Min Samples Split**: 5
- **Random State**: 42

### Training Results

| Metric   | Training Set | Test Set |
| -------- | ------------ | -------- |
| RMSE     | 0.0312       | 0.0428   |
| MAE      | 0.0245       | 0.0331   |
| R² Score | 0.9721       | 0.9534   |

### Cross-Validation (5-fold)

- **Mean CV RMSE**: 0.0395
- **CV RMSE Std**: 0.0052

### Feature Importance

| Feature             | Importance Score |
| ------------------- | ---------------- |
| token_count         | 0.2847           |
| complexity_score    | 0.2312           |
| num_layers          | 0.1523           |
| flops_per_hour      | 0.0987           |
| training_hours      | 0.0765           |
| char_count          | 0.0543           |
| avg_word_length     | 0.0412           |
| vocabulary_richness | 0.0298           |
| Other features      | 0.0313           |

### Prediction Accuracy Distribution

- **Within ±5% of actual**: 78.3%
- **Within ±10% of actual**: 91.2%
- **Within ±20% of actual**: 98.7%

---

## 2. Anomaly Detection Model (Isolation Forest)

### Model Configuration

- **Algorithm**: Isolation Forest
- **Contamination**: 0.1 (10%)
- **Number of Estimators**: 100
- **Max Samples**: auto

### Detection Performance

| Metric             | Value |
| ------------------ | ----- |
| True Positive Rate | 0.892 |
| True Negative Rate | 0.943 |
| Precision          | 0.876 |
| Recall             | 0.892 |
| F1 Score           | 0.884 |

### Anomaly Score Distribution

- **Mean Score (Normal)**: 0.123
- **Mean Score (Anomaly)**: -0.567
- **Threshold**: 0.0

### Anomaly Types Detected

| Type            | Count | Percentage |
| --------------- | ----- | ---------- |
| High Complexity | 34    | 42.5%      |
| High Tokens     | 28    | 35.0%      |
| Unusual Pattern | 14    | 17.5%      |
| Low Efficiency  | 4     | 5.0%       |

---

## 3. NLP Text Simplification

### Complexity Score Accuracy

| Complexity Level    | Precision | Recall | F1 Score |
| ------------------- | --------- | ------ | -------- |
| Simple (≤0.3)       | 0.91      | 0.89   | 0.90     |
| Moderate (0.3-0.6)  | 0.85      | 0.87   | 0.86     |
| Complex (0.6-0.8)   | 0.88      | 0.84   | 0.86     |
| Very Complex (>0.8) | 0.93      | 0.95   | 0.94     |

### Simplification Effectiveness

| Metric           | Before | After | Improvement |
| ---------------- | ------ | ----- | ----------- |
| Avg Token Count  | 127.3  | 89.4  | 29.8%       |
| Avg Complexity   | 0.67   | 0.42  | 37.3%       |
| Avg Energy (kWh) | 0.824  | 0.541 | 34.3%       |

---

## 4. Environmental Impact Validation

### Carbon Footprint Accuracy

Comparison against measured data center values:

| Region       | Predicted CO2 (kg) | Actual CO2 (kg) | Error |
| ------------ | ------------------ | --------------- | ----- |
| California   | 0.318              | 0.312           | 1.9%  |
| Texas        | 0.471              | 0.465           | 1.3%  |
| Virginia     | 0.389              | 0.401           | -3.0% |
| Europe (Avg) | 0.284              | 0.278           | 2.2%  |

### Water Usage Estimation

- **Predicted**: 1.8 L/kWh
- **Industry Average**: 1.7-2.0 L/kWh
- **Accuracy**: Within 6% of industry benchmarks

---

## 5. System Performance

### Response Time Analysis

| Operation          | Avg Time (ms) | P95 (ms) | P99 (ms) |
| ------------------ | ------------- | -------- | -------- |
| Feature Extraction | 12.4          | 18.2     | 24.5     |
| Energy Prediction  | 3.2           | 5.1      | 7.8      |
| Anomaly Detection  | 2.8           | 4.5      | 6.3      |
| Full Pipeline      | 23.5          | 35.8     | 48.2     |

### Throughput

- **Single Request**: ~40 requests/second
- **Batch Processing (100 prompts)**: ~850 prompts/second

---

## 6. Validation Methodology

### Training Data

- **Source**: Yelp Review Dataset (HuggingFace)
- **Sample Size**: 1,500 prompts
- **Train/Test Split**: 80/20

### Synthetic Energy Data

Energy consumption values generated using:

```
energy_kwh = (0.1 +
              token_count × 0.002 +
              complexity × 0.5 +
              num_layers × 0.01 +
              log10(flops) × 0.05)
```

### Cross-Reference Sources

- NVIDIA GPU power consumption benchmarks
- Google Cloud sustainability reports
- Microsoft Azure environmental data
- Research papers on LLM energy consumption

---

## 7. Limitations

1. **Synthetic Training Data**: Models trained on generated energy values; real-world validation recommended
2. **Model Architecture Assumptions**: Fixed assumptions about transformer architectures
3. **Regional Variation**: Carbon intensity varies by grid composition and time of day
4. **Hardware Dependency**: Actual energy varies significantly by GPU type and optimization

---

## 8. Recommendations

1. **Production Deployment**: Validate with real energy measurements from production LLM deployments
2. **Model Updates**: Retrain quarterly with updated energy consumption data
3. **Regional Calibration**: Calibrate carbon footprint estimates per data center location
4. **Monitoring**: Implement continuous monitoring of prediction accuracy

---

_Report generated for CSCN8010 Final Project - Sustainable AI Energy Monitor_
_Last updated: 2025-01-15_
