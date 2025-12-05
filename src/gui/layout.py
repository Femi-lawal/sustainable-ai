"""
GUI Layout components for the Sustainable AI application.
Provides reusable Streamlit components for building the interface.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd


def setup_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="🌱 Sustainable AI Energy Monitor",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/sustainable-ai',
            'Report a bug': 'https://github.com/sustainable-ai/issues',
            'About': """
            # Sustainable AI - Energy Efficient Prompt Engineering
            
            This application helps you understand and reduce the energy 
            consumption of AI prompts, supporting transparency and 
            EU compliance requirements.
            
            **Version:** 1.0.0
            """
        }
    )


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            padding: 1rem;
        }
        
        /* Card styling */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1rem;
        }
        
        .metric-card h3 {
            margin: 0;
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .metric-card .value {
            font-size: 2rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        /* Energy level indicators */
        .energy-low {
            color: #10B981;
            background-color: #D1FAE5;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
        }
        
        .energy-medium {
            color: #F59E0B;
            background-color: #FEF3C7;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
        }
        
        .energy-high {
            color: #EF4444;
            background-color: #FEE2E2;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
        }
        
        /* Anomaly indicator */
        .anomaly-badge {
            background-color: #FEE2E2;
            color: #DC2626;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            border-left: 4px solid #DC2626;
        }
        
        /* Success indicator */
        .success-badge {
            background-color: #D1FAE5;
            color: #059669;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            border-left: 4px solid #059669;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            padding-top: 2rem;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            border-radius: 5px;
            font-weight: 600;
        }
        
        /* Info box */
        .info-box {
            background-color: #EFF6FF;
            border-left: 4px solid #3B82F6;
            padding: 1rem;
            border-radius: 0 5px 5px 0;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the application header."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🌱 Sustainable AI Energy Monitor")
        st.markdown("""
            *Transparency and Energy-Efficient Prompt Engineering with Machine Learning*
        """)
    
    with col2:
        st.markdown(f"""
            <div style="text-align: right; padding-top: 1rem;">
                <small>EU Reporting Deadline: <strong>Aug 2026</strong></small><br>
                <small>Version 1.0.0</small>
            </div>
        """, unsafe_allow_html=True)


def render_sidebar(default_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Render the sidebar with model configuration.
    
    Returns:
        Dictionary with user-configured parameters
    """
    default_params = default_params or {}
    
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        st.markdown("---")
        
        # LLM Architecture settings
        st.subheader("LLM Architecture")
        
        num_layers = st.slider(
            "Number of Layers",
            min_value=4,
            max_value=96,
            value=default_params.get("num_layers", 24),
            step=2,
            help="Number of transformer layers in the model"
        )
        
        training_hours = st.slider(
            "Training Time (hours)",
            min_value=0.5,
            max_value=100.0,
            value=default_params.get("training_hours", 8.0),
            step=0.5,
            help="Approximate training time for the model"
        )
        
        flops_exp = st.slider(
            "FLOPs/hour (10^x)",
            min_value=9,
            max_value=15,
            value=default_params.get("flops_exp", 11),
            help="Computational operations per hour (exponent)"
        )
        flops_per_hour = 10 ** flops_exp
        
        st.markdown("---")
        
        # Region settings
        st.subheader("🌍 Region")
        
        region = st.selectbox(
            "Data Center Region",
            options=["california", "texas", "virginia", "eu_average", "canada_average"],
            index=0,
            help="Region affects electricity costs and carbon intensity"
        )
        
        st.markdown("---")
        
        # Display info
        st.subheader("📊 Regional Factors")
        
        region_info = {
            "california": {"cost": "$0.22/kWh", "carbon": "0.21 kg/kWh"},
            "texas": {"cost": "$0.12/kWh", "carbon": "0.45 kg/kWh"},
            "virginia": {"cost": "$0.11/kWh", "carbon": "0.35 kg/kWh"},
            "eu_average": {"cost": "$0.25/kWh", "carbon": "0.28 kg/kWh"},
            "canada_average": {"cost": "$0.10/kWh", "carbon": "0.12 kg/kWh"}
        }
        
        info = region_info.get(region, {})
        st.markdown(f"""
            - **Electricity Cost:** {info.get('cost', 'N/A')}
            - **Carbon Intensity:** {info.get('carbon', 'N/A')}
        """)
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
                This tool helps you:
                - **Estimate** energy consumption of AI prompts
                - **Detect** anomalous high-resource requests
                - **Optimize** prompts for efficiency
                - **Generate** transparency reports for EU compliance
                
                Built with 🌱 for a sustainable AI future.
            """)
    
    return {
        "num_layers": num_layers,
        "training_hours": training_hours,
        "flops_per_hour": flops_per_hour,
        "region": region
    }


def render_metric_card(title: str, value: str, delta: str = None, 
                       color: str = "default") -> None:
    """Render a metric card with optional delta."""
    colors = {
        "default": "#667eea",
        "green": "#10B981",
        "yellow": "#F59E0B",
        "red": "#EF4444",
        "blue": "#3B82F6"
    }
    
    bg_color = colors.get(color, colors["default"])
    
    delta_html = ""
    if delta:
        delta_color = "#10B981" if "+" not in delta or float(delta.replace('%', '').replace('+', '')) < 0 else "#EF4444"
        delta_html = f'<span style="color: {delta_color}; font-size: 0.9rem;">{delta}</span>'
    
    st.markdown(f"""
        <div style="background: {bg_color}; padding: 1.25rem; border-radius: 10px; color: white;">
            <h4 style="margin: 0; opacity: 0.85; font-size: 0.9rem;">{title}</h4>
            <p style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{value}</p>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def render_energy_gauge(energy_joules: float, max_joules: float = 80.0) -> None:
    """Render an energy consumption gauge chart.
    
    Args:
        energy_joules: Energy consumption in Joules (model's native unit)
        max_joules: Maximum value for gauge scale (default 80 J for typical prompts)
    
    Thresholds based on real measurements (100 samples with CodeCarbon):
    - Low: ≤10 J (simple prompts: definitions, short questions)
    - Medium: ≤25 J (medium prompts: explanations, analysis)
    - High: ≤50 J (complex prompts: detailed explanations)
    - Very High: >50 J (comprehensive analysis, multi-part prompts)
    """
    # Determine color based on energy level (science-backed thresholds)
    if energy_joules < 10:
        color = "green"
    elif energy_joules < 25:
        color = "yellow"
    elif energy_joules < 50:
        color = "orange"
    else:
        color = "red"
    
    # Convert to kWh for display
    energy_kwh = energy_joules / 3_600_000
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=energy_joules,
        number={'suffix': ' J', 'valueformat': '.1f'},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Energy: {energy_joules:.1f} J ({energy_kwh:.2e} kWh)", 'font': {'size': 14}},
        delta={'reference': 33, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},  # Mean from real data
        gauge={
            'axis': {'range': [0, max_joules], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10], 'color': '#D1FAE5'},       # Low: ≤10 J (green)
                {'range': [10, 25], 'color': '#FEF3C7'},      # Medium: ≤25 J (yellow)
                {'range': [25, 50], 'color': '#FED7AA'},      # High: ≤50 J (orange)
                {'range': [50, max_joules], 'color': '#FEE2E2'}  # Very high: >50 J (red)
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_complexity_breakdown(breakdown: Dict[str, float]) -> None:
    """Render a radar chart of complexity breakdown."""
    categories = [
        'Sentence', 'Vocabulary', 'Syntactic', 
        'Semantic', 'Structural'
    ]
    
    values = [
        breakdown.get('sentence_complexity', 0),
        breakdown.get('vocabulary_complexity', 0),
        breakdown.get('syntactic_complexity', 0),
        breakdown.get('semantic_density', 0),
        breakdown.get('structural_complexity', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name='Complexity',
        line_color='#667eea'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_energy_comparison(original_joules: float, optimized_joules: float) -> None:
    """Render a comparison bar chart for energy before/after optimization.
    
    Args:
        original_joules: Original energy consumption in Joules
        optimized_joules: Optimized energy consumption in Joules
    """
    fig = go.Figure()
    
    # Convert to kWh for secondary display
    original_kwh = original_joules / 3_600_000
    optimized_kwh = optimized_joules / 3_600_000
    
    fig.add_trace(go.Bar(
        name=f'Original ({original_joules:.2f} J)',
        x=['Energy (Joules)'],
        y=[original_joules],
        marker_color='#EF4444',
        hovertemplate=f'Original<br>{original_joules:.2f} J<br>({original_kwh:.2e} kWh)<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name=f'Optimized ({optimized_joules:.2f} J)',
        x=['Energy (Joules)'],
        y=[optimized_joules],
        marker_color='#10B981',
        hovertemplate=f'Optimized<br>{optimized_joules:.2f} J<br>({optimized_kwh:.2e} kWh)<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='group',
        height=250,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_alternatives_table(alternatives: List[Dict[str, Any]]) -> None:
    """Render a table of optimization alternatives."""
    if not alternatives:
        st.info("No alternatives generated.")
        return
    
    df = pd.DataFrame(alternatives)
    
    # Format columns
    if 'prompt' in df.columns:
        df['prompt'] = df['prompt'].apply(lambda x: x[:60] + "..." if len(x) > 60 else x)
    
    if 'similarity' in df.columns:
        df['similarity'] = df['similarity'].apply(lambda x: f"{x:.1%}")
    
    if 'energy_reduction_percent' in df.columns:
        df['energy_reduction_percent'] = df['energy_reduction_percent'].apply(lambda x: f"{x:.1f}%")
    
    if 'optimization_score' in df.columns:
        df['optimization_score'] = df['optimization_score'].apply(lambda x: f"{x:.2f}")
    
    # Format energy column if present
    if 'energy_joules' in df.columns:
        df['energy_joules'] = df['energy_joules'].apply(lambda x: f"{x:.1f} J")
    
    # Rename columns for display
    df = df.rename(columns={
        'prompt': 'Alternative Prompt',
        'strategy': 'Strategy',
        'similarity': 'Similarity',
        'energy_joules': 'Energy (J)',
        'energy_reduction_percent': 'Energy Saved',
        'optimization_score': 'Score'
    })
    
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_transparency_report(report: Dict[str, Any]) -> None:
    """Render a transparency report summary."""
    st.subheader(f"📋 Report: {report.get('report_id', 'N/A')}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        summary = report.get('summary', {})
        st.metric("Total Prompts", summary.get('total_prompts', 0))
        st.metric("Total Energy", f"{summary.get('total_energy_kwh', 0):.4f} kWh")
    
    with col2:
        env_impact = report.get('environmental_impact', {})
        # Convert kg to mg for more readable per-prompt values
        carbon_kg = env_impact.get('carbon_footprint_kg', 0)
        carbon_mg = carbon_kg * 1_000_000
        st.metric("Carbon Footprint", f"{carbon_mg:.2f} mg CO₂")
        st.metric("Water Usage", f"{env_impact.get('water_usage_liters', 0):.2f} L")
    
    with col3:
        compliance = report.get('compliance', {})
        status = compliance.get('status', 'UNKNOWN')
        status_color = "🟢" if status == "COMPLIANT" else "🟡" if status == "NO_DATA" else "🔴"
        st.metric("Compliance Status", f"{status_color} {status}")
        st.metric("Days to Deadline", compliance.get('days_until_deadline', 'N/A'))
    
    # Optimization impact
    st.markdown("---")
    opt_impact = report.get('optimization_impact', {})
    
    col1, col2 = st.columns(2)
    with col1:
        # Convert to more readable units
        carbon_avoided_kg = opt_impact.get('carbon_avoided_kg', 0)
        carbon_avoided_mg = carbon_avoided_kg * 1_000_000
        st.markdown(f"""
            **Optimization Impact:**
            - Total Optimizations: {opt_impact.get('total_optimizations', 0)}
            - Energy Saved: {opt_impact.get('energy_saved_kwh', 0):.4f} kWh
            - Carbon Avoided: {carbon_avoided_mg:.2f} mg CO₂
        """)
    
    with col2:
        anomalies = report.get('anomalies', {})
        st.markdown(f"""
            **Anomaly Detection:**
            - Total Detected: {anomalies.get('total_detected', 0)}
            - Anomaly Rate: {anomalies.get('anomaly_rate_percent', 0):.2f}%
        """)


def render_time_series_chart(data: List[Dict[str, Any]], 
                              x_col: str, y_col: str, 
                              title: str = "Energy Over Time") -> None:
    """Render a time series chart."""
    if not data:
        st.info("No data available for visualization.")
        return
    
    df = pd.DataFrame(data)
    
    fig = px.line(df, x=x_col, y=y_col, title=title)
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_pie_chart(data: Dict[str, float], title: str = "Distribution") -> None:
    """Render a pie chart."""
    fig = px.pie(
        values=list(data.values()),
        names=list(data.keys()),
        title=title
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_suggestions(suggestions: List[str]) -> None:
    """Render improvement suggestions."""
    st.subheader("💡 Improvement Suggestions")
    
    for i, suggestion in enumerate(suggestions, 1):
        st.markdown(f"""
            <div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; 
                        padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0 5px 5px 0;">
                <strong>{i}.</strong> {suggestion}
            </div>
        """, unsafe_allow_html=True)


def render_anomaly_alert(result: Dict[str, Any]) -> None:
    """Render an anomaly detection alert."""
    if result.get('is_anomaly', False):
        severity = result.get('severity', 'unknown')
        severity_colors = {
            'low': '#FEF3C7',
            'medium': '#FED7AA',
            'high': '#FEE2E2',
            'critical': '#FECACA'
        }
        bg_color = severity_colors.get(severity, '#FEE2E2')
        
        st.markdown(f"""
            <div style="background-color: {bg_color}; border-left: 4px solid #DC2626; 
                        padding: 1rem; border-radius: 0 5px 5px 0; margin: 1rem 0;">
                <h4 style="color: #DC2626; margin: 0 0 0.5rem 0;">⚠️ Anomaly Detected</h4>
                <p><strong>Severity:</strong> {severity.upper()}</p>
                <p><strong>Type:</strong> {result.get('anomaly_type', 'Unknown')}</p>
                <p><strong>Score:</strong> {result.get('anomaly_score', 0):.4f}</p>
                <p style="margin-bottom: 0;"><strong>Recommendation:</strong> {result.get('recommendation', 'N/A')}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ No anomalies detected. The prompt appears normal.")
