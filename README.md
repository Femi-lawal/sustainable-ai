# Sustainable AI - Transparency and Energy-Efficient Prompt/Context Engineering with Machine Learning (CSCN8010)

## Project Members:

    1. Jarius Bedward #8841640
    2. Mostafa Allahmoradi
    3. Oluwafemi Lawal
    4. Jatinder Pal Singh

## Project Summary:

This project addresses the critical environmental impact of the rapid expansion of AI data centers. With EU regulations requiring energy usage reporting by August 2026 and over 1,240 new data centers built in 2025 alone, there is an urgent need for transparency. This project builds a Machine Learning framework to predict the energy consumption of Large Language Model (LLM) prompts and recommends semantically equivalent, energy-efficient alternatives. It combines supervised learning for energy estimation and unsupervised learning for anomaly detection to optimize dynamic workloads.

## 🎯 Key Results (December 2025) - Professional Standards Achieved! ✅

| Metric              | Target    | Achieved                               | Status |
| ------------------- | --------- | -------------------------------------- | ------ |
| **Model R²**        | > 0.80    | **0.9813** (98.1% variance explained!) | ✅     |
| **MAPE**            | < 25%     | **6.8%**                               | ✅     |
| **Prediction Bias** | 0.90-1.10 | **0.9988**                             | ✅     |
| **Correlation**     | > 0.85    | **0.9906**                             | ✅     |
| **Within 20%**      | > 70%     | **94.0%**                              | ✅     |
| Training Samples    | -         | 2,600 (hybrid: synthetic + real)       | ✅     |
| Validation Samples  | -         | 100 real measurements                  | ✅     |
| Energy Savings      | -         | **8-43%** (prompt optimization)        | ✅     |
| Test Coverage       | -         | 283 tests (100% pass)                  | ✅     |

### Energy Reduction Examples

| Original Prompt                                         | Optimized                | Energy Saved |
| ------------------------------------------------------- | ------------------------ | ------------ |
| "Due to the fact that I need assistance..."             | "Because I need help..." | **35.1%**    |
| "I was wondering if you could perhaps maybe tell me..." | "Please tell me..."      | **42.6%**    |
| "In order to understand..."                             | "To understand..."       | **13.4%**    |

## Project Setup:

## **Key Features:**

- **User Interface (GUI):**
  - Built with Streamlit to accept user prompts and model parameters (Layers, Training Time, FLOPs).
  - Visualizes energy costs and recommended improvements side-by-side.
- **NLP Module:**
  - Parses input prompts to extract 12 features (token count, complexity score, lexical density, etc.).
  - Uses sentence embeddings (Sentence-Transformers) to understand semantic context.
  - Enhanced simplifier with 100+ verbose phrase replacements and aggressive optimization strategies.
- **Energy Prediction Engine:**
  - A Supervised Learning model (**Gradient Boosting, R²=0.9813**) predicts energy consumption.
  - **Hybrid Training**: 2,600 samples (2,000 synthetic + 100 real measurements + 500 augmented)
  - **Calibrated to Real Measurements**: Model validated against actual energy measurements
  - **Professional Metrics**: MAPE=6.8%, Bias=0.9988, 94% predictions within 20% of actual
  - Primary features: token_count, word_count, char_count, complexity_score
  - **Calibration Formula**: Energy (J) = 9.27 + 0.331 × tokens
  - **Token Scaling**: For prompts longer than 25 tokens (beyond training range), energy is scaled using √(token_ratio) to accurately predict consumption for real-world prompts.
  - **Deduplication Detection**: Automatically detects and handles repeated/copy-pasted content in prompts.
- **Anomaly Detection:**
  - An Unsupervised Learning module (Isolation Forest) flags prompts with unusually high resource demands.
  - Identifies outliers in usage patterns for transparency.
- **Prompt Optimization:**
  - A recommendation engine suggests alternative prompts that yield similar outputs but require less computational power.
  - Multiple strategies: aggressive, verbose, filler, compress, truncate, core extraction.
  - Achieves **8-43% energy reduction** while maintaining semantic similarity.

## Requirements:

    - pip install -r requirements.txt

## 🎯 How to Run:

1. Clone this repo (git clone <repo-url> cd <repo-folder>)
2. Install Required Dependencies: "pip install -r requirements.txt"
3. Navigate to the source directory: `cd src/gui`
4. Run the application: `streamlit run app.py` (or `python app.py`)
5. Input a prompt

## Code Explanation/Workflow:

1. **User Input & Configuration**

   - The user submits a text prompt via the GUI.
   - User provides LLM architecture details: Number of Layers, Known Training Time, and Expected FLOPs/hour.

2. **NLP Preprocessing**

   - The system calculates the token count and complexity score of the input text.
   - Vector embeddings are generated to capture the semantic meaning for the optimization engine.

3. **Energy Prediction (Supervised)**

   - The extracted features are passed to the Energy Prediction Model.
   - The model estimates the specific energy cost (kWh) for processing that prompt.

4. **Anomaly Detection (Unsupervised)**

   - The input metrics are cross-referenced with normal usage patterns using the Anomaly Detection Module.
   - Outliers are flagged (e.g., if a prompt requires excessive computation relative to its length).

5. **Optimization & Recommendation**

   - The Prompt Optimizer searches for or generates a more efficient version of the prompt.
   - It targets a lower token count or complexity while maintaining the original intent.

6. **Output & Logging**
   - The estimated energy and the optimized prompt are displayed to the user.
   - Data is logged to the backend (SQLite/PostgreSQL) for future benchmarking and transparency reports.

### Final Conclusion:

    - This project demonstrates a proof-of-concept for "Sustainable AI" by linking NLP inputs directly to physical energy estimates.
    - The energy prediction model achieves R² = 0.9813 (98.1% accuracy) calibrated against real energy measurements.
    - The model meets all professional evaluation standards: MAPE = 6.8%, Prediction Bias = 0.9988, 94% predictions within 20%.
    - The optimization pipeline reduces energy consumption by 8-43% while maintaining semantic meaning.
    - The application successfully combines supervised learning (energy prediction), unsupervised learning (anomaly detection), and NLP (prompt simplification) into a unified framework.
    - By integrating anomaly detection and prompt optimization, users can make informed decisions about their AI usage and reduce environmental impact.

### 📊 Technical Details

**Model Training (Hybrid Approach):**

- 2,600 training samples (synthetic + real measurements + augmented)
- Calibration derived from 100 real energy measurements
- Hyperparameter tuning via RandomizedSearchCV with 5-fold cross-validation
- Model validated against real measurements with R² = 0.9813
- Gradient Boosting selected as best model

**Calibration Formula:**

```
Energy (Joules) = 9.27 + 0.331 × token_count
```

(R² = 0.871 derived from real measurements)

### 📏 Unit Convention & Thresholds

**Primary Unit: Joules (J)** - Appropriate for per-prompt energy consumption

| Conversion | Value            |
| ---------- | ---------------- |
| 1 kWh      | 3,600,000 Joules |
| 1 Joule    | 2.78e-7 kWh      |

**Energy Level Thresholds (Science-Backed):**

Derived from 100 real energy measurements using CodeCarbon on T5-small model:

| Level         | Threshold | Source Data                          | Typical Prompts              |
| ------------- | --------- | ------------------------------------ | ---------------------------- |
| **Low**       | ≤ 10 J    | Simple: 3.4-10.6 J (mean 5.7 J)      | Definitions, short questions |
| **Medium**    | ≤ 25 J    | Medium: 10.3-20.1 J (mean 13.8 J)    | Explanations, analysis       |
| **High**      | ≤ 50 J    | Long: 25.5-36.1 J (mean 29.6 J)      | Detailed explanations        |
| **Very High** | > 50 J    | Very long: 42.9-73.2 J (mean 51.9 J) | Comprehensive, multi-part    |

**Carbon Display:** Shown in **milligrams (mg CO₂)** for human readability (e.g., "1.03 mg" instead of "1.03e-06 kg")

**Professional Evaluation Metrics:**
| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| R² Score | > 0.80 | 0.9813 | ✅ |
| MAPE | < 25% | 6.8% | ✅ |
| Prediction Bias | 0.90-1.10 | 0.9988 | ✅ |
| Correlation | > 0.85 | 0.9906 | ✅ |
| Within 20% | > 70% | 94.0% | ✅ |

**Simplification Strategies:**

- `aggressive`: All strategies combined (30-50% reduction)
- `verbose`: Remove 100+ verbose phrases
- `filler`: Remove 60+ filler words
- `core`: Extract essential question
- `truncate`: Keep important sentences

### 🤝 Contributing

This is a Final Project Protocol developed for CSCN8010. If any questions arise do not hesitate to contact the project member.

### References:
