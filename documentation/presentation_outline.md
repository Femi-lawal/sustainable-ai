# Sustainable AI - Presentation Outline (10 Minutes)

## Slide Structure Overview

| Slide # | Title                            | Duration | Content Summary                                                 |
| ------- | -------------------------------- | -------- | --------------------------------------------------------------- |
| 1       | Title Slide                      | 15 sec   | Project name, team members, course                              |
| 2       | Problem Statement                | 45 sec   | AI environmental impact, EU regulations, 1,240 new data centers |
| 3       | Project Objectives               | 30 sec   | 3 main goals - predict, detect, optimize                        |
| 4       | Solution Architecture            | 45 sec   | High-level system diagram, 4 modules                            |
| 5       | Methodology: Data Collection     | 45 sec   | CodeCarbon measurements, 100 real samples                       |
| 6       | Methodology: Feature Engineering | 30 sec   | 5 core features, correlation analysis                           |
| 7       | Model Comparison                 | 45 sec   | Linear Regression vs Random Forest vs Neural Network            |
| 8       | Results: Model Performance       | 45 sec   | R²=0.9809, RMSE=3.28J, key metrics table                        |
| 9       | Results: Energy Optimization     | 45 sec   | 8-43% energy savings, before/after examples                     |
| 10      | Live Demo / Screenshots          | 60 sec   | Streamlit GUI demonstration                                     |
| 11      | Anomaly Detection                | 30 sec   | Isolation Forest, outlier flagging                              |
| 12      | Technical Implementation         | 30 sec   | Tech stack, test coverage, folder structure                     |
| 13      | Key Takeaways                    | 30 sec   | 3-4 bullet points of achievements                               |
| 14      | Future Work                      | 30 sec   | Potential improvements and extensions                           |
| 15      | Q&A                              | 30 sec   | Questions slide, thank you, contact                             |

**Total: ~10 minutes**

---

## Detailed Slide Breakdown

### Slide 1: Title

- Project title
- Team member names with student IDs
- Course: CSCN8010
- Date: December 2025

### Slide 2: Problem Statement

- AI data centers consume massive energy
- EU regulations require energy reporting by August 2026
- 1,240+ new data centers built in 2025
- Need for transparency and optimization

### Slide 3: Project Objectives

1. **Predict** energy consumption of LLM prompts (Supervised ML)
2. **Detect** anomalous/wasteful prompts (Unsupervised ML)
3. **Optimize** prompts for energy efficiency (NLP)

### Slide 4: Solution Architecture

- System diagram showing 4 modules
- User Interface (Streamlit)
- NLP Module (feature extraction)
- Energy Prediction Engine (Random Forest)
- Anomaly Detection (Isolation Forest)

### Slide 5: Data Collection Methodology

- CodeCarbon library for real measurements
- T5-small model inference
- 100 real energy measurements
- Categories: simple, medium, long, very long

### Slide 6: Feature Engineering

- 5 core features selected
- Correlation analysis results
- Feature importance from Random Forest

### Slide 7: Model Comparison

- Assignment requirement: 3 model types
- Linear Regression (baseline): R²=0.87
- Random Forest (selected): R²=0.98
- Neural Network (MLP): R²~0.95

### Slide 8: Model Performance Results

- Key metrics table
- R²=0.9809 (98% variance explained)
- RMSE=3.28 J, MAE=2.46 J
- 94% predictions within 20% of actual

### Slide 9: Energy Optimization Results

- Before/after prompt examples
- 8-43% energy reduction
- Semantic similarity preserved
- 7 simplification strategies

### Slide 10: Live Demo

- Streamlit GUI screenshots
- Input prompt → Energy prediction
- Optimization recommendations
- Dashboard visualization

### Slide 11: Anomaly Detection

- Isolation Forest algorithm
- Identifies unusual prompts
- Flags resource-intensive outliers
- Transparency reporting

### Slide 12: Technical Implementation

- Python, scikit-learn, Streamlit
- 283 unit tests (100% pass)
- Modular architecture
- SQLite logging

### Slide 13: Key Takeaways

- Professional-grade ML model (R²=0.98)
- Real energy measurement validation
- Practical energy savings (8-43%)
- Complete end-to-end solution

### Slide 14: Future Work

- Integration with more LLM providers
- Real-time energy monitoring
- Expanded prompt optimization strategies
- Cloud deployment

### Slide 15: Q&A

- Thank you
- Questions welcome
- Team contact information
