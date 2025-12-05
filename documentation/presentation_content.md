# Sustainable AI - Presentation Content

## Energy-Efficient Prompt Engineering with Machine Learning

---

# SLIDE 1: Title Slide

## Sustainable AI

### Transparency and Energy-Efficient Prompt/Context Engineering with Machine Learning

**Course:** CSCN8010 - Applied Machine Learning

**Team Members:**

- Jarius Bedward (#8841640)
- Mostafa Allahmoradi
- Oluwafemi Lawal
- Jatinder Pal Singh

**Date:** December 2025

---

# SLIDE 2: Problem Statement

## The Growing Environmental Impact of AI

### The Challenge

- **AI data centers are consuming massive amounts of energy**
- Training GPT-4 alone estimated to use **50 GWh** of electricity
- A single ChatGPT query uses **10x more energy** than a Google search

### Regulatory Pressure

- **EU Energy Efficiency Directive** requires energy usage reporting by **August 2026**
- Over **1,240 new data centers** built globally in 2025
- Growing demand for **transparency** in AI's environmental footprint

### The Gap

- Users have **no visibility** into how their prompts consume energy
- No tools exist to **optimize prompts** for energy efficiency
- Need for **real-time feedback** on environmental impact

---

# SLIDE 3: Project Objectives

## What We Set Out to Build

### Three Core Objectives

| #   | Objective                                     | ML Approach           |
| --- | --------------------------------------------- | --------------------- |
| 1   | **Predict** energy consumption of LLM prompts | Supervised Learning   |
| 2   | **Detect** anomalous or wasteful prompts      | Unsupervised Learning |
| 3   | **Optimize** prompts for energy efficiency    | NLP + Rule-based      |

### Success Criteria

- Model accuracy **R² > 0.80**
- Feature correlation **> 0.85**
- Energy savings **> 10%**
- Semantic similarity preserved

---

# SLIDE 4: Solution Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (Streamlit)                   │
│  • Prompt Input  • Energy Dashboard  • Optimization Suggestions │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌───────────────┐       ┌─────────────────┐       ┌───────────────┐
│  NLP Module   │       │ Energy Predictor │       │   Anomaly     │
│               │       │                  │       │   Detector    │
│ • Tokenizer   │──────►│ Random Forest    │       │               │
│ • Parser      │       │ R² = 0.9809      │       │ Isolation     │
│ • 5 Features  │       │ 5 Features       │       │ Forest        │
└───────────────┘       └─────────────────┘       └───────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │   Prompt Optimizer      │
                    │ • 7 Strategies          │
                    │ • 8-43% Energy Savings  │
                    └─────────────────────────┘
```

### Four Integrated Modules

1. **NLP Module** - Extract features from prompts
2. **Energy Predictor** - Random Forest model for energy estimation
3. **Anomaly Detector** - Isolation Forest for outlier detection
4. **Prompt Optimizer** - Suggest energy-efficient alternatives

---

# SLIDE 5: Methodology - Data Collection

## Real Energy Measurements with CodeCarbon

### Data Collection Process

1. **Tool:** CodeCarbon library (tracks actual CPU/GPU energy)
2. **Model:** T5-small transformer for inference
3. **Measurement:** Real Joules consumed per prompt

### Dataset Statistics

| Category  | Samples | Energy Range  | Mean Energy |
| --------- | ------- | ------------- | ----------- |
| Simple    | 25      | 3.4 - 10.6 J  | 5.7 J       |
| Medium    | 25      | 10.3 - 20.1 J | 13.8 J      |
| Long      | 25      | 25.5 - 36.1 J | 29.6 J      |
| Very Long | 25      | 42.9 - 73.2 J | 51.9 J      |
| **Total** | **100** | 3.4 - 73.2 J  | 25.3 J      |

### Why This Matters

- **Real measurements** - not synthetic/estimated data
- **Reproducible** - CodeCarbon is open-source
- **Calibrated model** - predictions validated against actual consumption

---

# SLIDE 6: Methodology - Feature Engineering

## From Raw Text to ML Features

### 5 Core Features Selected

| Feature                 | Description        | Correlation with Energy |
| ----------------------- | ------------------ | ----------------------- |
| **token_count**         | Number of tokens   | 0.933 (very strong)     |
| **word_count**          | Number of words    | 0.922 (very strong)     |
| **char_count**          | Total characters   | 0.927 (very strong)     |
| **avg_sentence_length** | Words per sentence | 0.922 (very strong)     |
| **avg_word_length**     | Chars per word     | -0.194 (weak)           |

### Feature Importance (Random Forest)

| Feature             | Importance |
| ------------------- | ---------- |
| avg_sentence_length | 26.4%      |
| token_count         | 23.8%      |
| word_count          | 22.8%      |
| char_count          | 22.1%      |
| avg_word_length     | 4.8%       |

### Key Insight

**Token count and sentence length are the primary drivers of energy consumption**

---

# SLIDE 7: Model Comparison

## Assignment Requirement: Compare 3 Model Types

### Models Implemented

| Model                | Type          | Complexity |
| -------------------- | ------------- | ---------- |
| Linear Regression    | Baseline      | Low        |
| **Random Forest**    | Ensemble      | Medium     |
| Neural Network (MLP) | Deep Learning | High       |

### Performance Comparison

| Model                | Test R²    | Test RMSE  | Test MAE   |
| -------------------- | ---------- | ---------- | ---------- |
| Linear Regression    | 0.8696     | 8.58 J     | 6.55 J     |
| **Random Forest** ⭐ | **0.9809** | **3.28 J** | **2.46 J** |
| Neural Network       | ~0.95      | ~4.5 J     | ~3.5 J     |

### Why Random Forest Was Selected

1. ✅ **Best R² Score** - 98.09% variance explained
2. ✅ **Lowest Error** - RMSE of 3.28 J
3. ✅ **Interpretable** - Feature importance available
4. ✅ **No GPU Required** - Fast inference
5. ✅ **Assignment Compliant** - Listed as recommended option

---

# SLIDE 8: Results - Model Performance

## Professional-Grade Accuracy Achieved

### Key Metrics

| Metric          | Target | Achieved   | Status       |
| --------------- | ------ | ---------- | ------------ |
| **R² Score**    | > 0.80 | **0.9809** | ✅ Exceeded  |
| **RMSE**        | -      | **3.28 J** | ✅ Low error |
| **MAE**         | -      | **2.46 J** | ✅ Low error |
| **Correlation** | > 0.85 | **0.93**   | ✅ Exceeded  |
| **Within 20%**  | > 70%  | **94.0%**  | ✅ Exceeded  |

### What These Metrics Mean

- **R² = 0.9809** → Model explains **98.09%** of variance in energy consumption
- **RMSE = 3.28 J** → Average prediction error of ~3 Joules
- **94% within 20%** → Nearly all predictions are close to actual values

### Validation Approach

- **Training:** 80 samples (80%)
- **Testing:** 20 samples (20%)
- **Cross-validation:** 5-fold CV for hyperparameter tuning

---

# SLIDE 9: Results - Energy Optimization

## Practical Energy Savings

### Optimization Examples

| Original Prompt                                         | Optimized                       | Energy Saved |
| ------------------------------------------------------- | ------------------------------- | ------------ |
| "Due to the fact that I need assistance..."             | "Because I need help..."        | **35.1%**    |
| "I was wondering if you could perhaps maybe tell me..." | "Please tell me..."             | **42.6%**    |
| "In order to understand the fundamental concepts..."    | "To understand the concepts..." | **13.4%**    |
| "At this point in time, I would like to..."             | "Now, I want to..."             | **28.7%**    |

### 7 Simplification Strategies

| Strategy       | Description                 | Typical Savings |
| -------------- | --------------------------- | --------------- |
| **aggressive** | All strategies combined     | 30-50%          |
| **verbose**    | Remove 100+ verbose phrases | 20-40%          |
| **filler**     | Remove 60+ filler words     | 10-20%          |
| **compress**   | Merge redundant sentences   | 15-25%          |
| **truncate**   | Keep essential content      | 20-35%          |
| **core**       | Extract main question       | 25-40%          |
| **dedup**      | Remove repeated content     | 10-30%          |

### Key Achievement

**8-43% energy reduction while maintaining semantic similarity (>60%)**

---

# SLIDE 10: Live Demo / Screenshots

## Streamlit GUI Application

### Main Interface

```
┌─────────────────────────────────────────────────────────────────┐
│  🌱 Sustainable AI - Energy Monitor                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📝 Enter Your Prompt:                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Explain the concept of machine learning...               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ⚡ Analyze Energy                                               │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📊 RESULTS                                                      │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ Energy: 15.4 J   │  │ Carbon: 1.2 mg   │                     │
│  │ Level: MEDIUM    │  │ Cost: $0.0001    │                     │
│  └──────────────────┘  └──────────────────┘                     │
│                                                                  │
│  💡 OPTIMIZED VERSION                                            │
│  "Explain ML concepts" → Energy: 8.7 J (43% saved!)             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Features Demonstrated

1. **Prompt Input** - Enter any text prompt
2. **Energy Prediction** - Instant energy estimate in Joules
3. **Carbon Footprint** - CO₂ emissions in milligrams
4. **Optimization** - Suggested efficient alternative
5. **Comparison** - Side-by-side energy comparison

---

# SLIDE 11: Anomaly Detection

## Identifying Unusual Prompts

### Isolation Forest Algorithm

- **Type:** Unsupervised Learning
- **Purpose:** Detect outliers without labeled data
- **Method:** Isolate anomalies by random partitioning

### What Gets Flagged

| Anomaly Type     | Example                | Why Flagged                    |
| ---------------- | ---------------------- | ------------------------------ |
| Excessive length | 10,000+ tokens         | Unusually high resource demand |
| Repeated content | Copy-pasted text       | Inefficient duplication        |
| High complexity  | Deeply nested requests | Disproportionate compute cost  |

### Benefits

- **Transparency:** Users see when prompts are wasteful
- **Alerting:** Automatic notification of outliers
- **Logging:** All anomalies recorded for analysis

---

# SLIDE 12: Technical Implementation

## Technology Stack

### Core Technologies

| Component           | Technology                  |
| ------------------- | --------------------------- |
| **ML Framework**    | scikit-learn                |
| **NLP**             | Sentence-Transformers, NLTK |
| **GUI**             | Streamlit                   |
| **Database**        | SQLite                      |
| **Energy Tracking** | CodeCarbon                  |
| **Deep Learning**   | PyTorch (for NN comparison) |

### Code Quality

| Metric            | Value                               |
| ----------------- | ----------------------------------- |
| **Unit Tests**    | 283 tests                           |
| **Pass Rate**     | 100%                                |
| **Modules**       | 12 Python modules                   |
| **Documentation** | Full docstrings + architecture docs |

### Project Structure

```
Sustainable_AI_G3/
├── src/
│   ├── nlp/          # Text parsing & simplification
│   ├── prediction/   # Energy prediction (Random Forest)
│   ├── anomaly/      # Outlier detection
│   ├── optimization/ # Prompt optimization
│   └── gui/          # Streamlit application
├── model/
│   └── energy_predictor/  # Trained RF model
├── data/
│   └── validation/   # Real measurements
└── tests/            # 283 unit tests
```

---

# SLIDE 13: Key Takeaways

## What We Achieved

### ✅ Professional-Grade ML Model

- **R² = 0.9809** (98% accuracy)
- Validated against **100 real energy measurements**
- **94%** of predictions within 20% of actual values

### ✅ Practical Energy Savings

- **8-43% energy reduction** per prompt
- **7 optimization strategies** available
- Semantic similarity preserved (>60%)

### ✅ Complete End-to-End Solution

- NLP feature extraction
- Supervised learning (Random Forest)
- Unsupervised learning (Isolation Forest)
- User-friendly Streamlit GUI

### ✅ Assignment Requirements Met

- 3 model types compared (Linear, RF, NN)
- Supervised + Unsupervised learning
- Real data collection and validation

---

# SLIDE 14: Future Work

## Potential Improvements

### Short-Term Enhancements

1. **Multi-model support** - Extend to GPT-4, Claude, Llama
2. **Real-time monitoring** - Live energy tracking dashboard
3. **API integration** - REST API for third-party apps

### Long-Term Vision

1. **Cloud deployment** - AWS/Azure hosted service
2. **Browser extension** - Optimize prompts before sending
3. **Enterprise features** - Team analytics, budgets, alerts
4. **Carbon offsetting** - Integration with offset providers

### Research Directions

- Fine-tuned models per LLM architecture
- Prompt-response energy correlation
- Multi-query optimization batching

---

# SLIDE 15: Q&A

## Thank You!

### Questions?

We welcome any questions about:

- 🔬 **Methodology** - Data collection, feature engineering
- 📊 **Results** - Model performance, optimization
- 🛠️ **Implementation** - Code, architecture, testing
- 🔮 **Future Work** - Extensions, improvements

### Contact

**Team:** Sustainable AI Group 3

**Course:** CSCN8010 - Applied Machine Learning

**Repository:** Available upon request

---

## Appendix: Additional Technical Details

### Model Hyperparameters (Random Forest)

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

### Calibration Formula

Derived from linear regression on real measurements:

```
Energy (Joules) = 9.27 + 0.331 × token_count
```

(R² = 0.871 on calibration data)

### Energy Level Thresholds

| Level     | Threshold | Typical Use Case             |
| --------- | --------- | ---------------------------- |
| Low       | ≤ 10 J    | Short questions, definitions |
| Medium    | ≤ 25 J    | Explanations, summaries      |
| High      | ≤ 50 J    | Detailed analysis            |
| Very High | > 50 J    | Complex multi-part queries   |
