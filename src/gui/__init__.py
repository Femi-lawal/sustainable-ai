"""
GUI module for Sustainable AI application.
Provides Streamlit-based user interface components.
"""

from .layout import (
    setup_page_config,
    apply_custom_css,
    render_header,
    render_sidebar,
    render_metric_card,
    render_energy_gauge,
    render_complexity_breakdown,
    render_energy_comparison,
    render_alternatives_table,
    render_transparency_report,
    render_suggestions,
    render_anomaly_alert
)

__all__ = [
    'setup_page_config',
    'apply_custom_css',
    'render_header',
    'render_sidebar',
    'render_metric_card',
    'render_energy_gauge',
    'render_complexity_breakdown',
    'render_energy_comparison',
    'render_alternatives_table',
    'render_transparency_report',
    'render_suggestions',
    'render_anomaly_alert'
]
