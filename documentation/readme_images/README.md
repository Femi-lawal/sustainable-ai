# Readme Images Directory

This directory contains images used in the project documentation.

## Expected Images

The following images should be added for complete documentation:

1. **architecture_diagram.png** - System architecture overview
2. **dashboard_screenshot.png** - Streamlit dashboard UI
3. **energy_prediction_flow.png** - Energy prediction pipeline
4. **anomaly_detection_flow.png** - Anomaly detection process
5. **results_example.png** - Example analysis results

## Image Guidelines

- Format: PNG or SVG preferred
- Resolution: Minimum 800px width
- File size: Under 500KB each
- Alt text: Always provide descriptive alt text in Markdown

## Generating Diagrams

The architecture diagrams can be generated using Mermaid.js from `architecture.md`:

```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Generate PNG from Mermaid
mmdc -i ../architecture.md -o architecture_diagram.png
```

## Note

If images are missing, the documentation will still be readable as text descriptions are provided alongside image references.
