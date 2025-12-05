"""
Sustainable AI - Energy Efficient Prompt Engineering Application

Main Streamlit application that integrates all modules:
- NLP Processing (parsing, complexity scoring)
- Energy Prediction (supervised ML)
- Anomaly Detection (unsupervised ML)
- Prompt Optimization
- Transparency Reporting

Run with: streamlit run app.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directories to path for imports
# This handles running from different directories
_current_file = Path(__file__).resolve()
_gui_dir = _current_file.parent          # src/gui/
_src_dir = _gui_dir.parent               # src/
_project_root = _src_dir.parent          # project root

# Add paths in order of priority
sys.path.insert(0, str(_src_dir))        # For: from nlp.parser import ...
sys.path.insert(0, str(_project_root))   # For: from src.nlp.parser import ...
sys.path.insert(0, str(_gui_dir))        # For: from layout import ...

import streamlit as st

# Import GUI components
from layout import (
    setup_page_config, apply_custom_css, render_header, render_sidebar,
    render_metric_card, render_energy_gauge, render_complexity_breakdown,
    render_energy_comparison, render_alternatives_table,
    render_transparency_report, render_suggestions, render_anomaly_alert,
    render_pie_chart
)

# Import core modules - try multiple import patterns
try:
    # Pattern 1: Running from src/ directory
    from nlp.parser import parse_prompt
    from nlp.complexity_score import compute_complexity, get_complexity_breakdown
    from prediction.estimator import EnergyPredictor
    from anomaly.detector import AnomalyDetector
    from optimization.recomender import PromptOptimizer
    from utils.database import DatabaseManager
    from utils.logger import get_logger
except ImportError:
    try:
        # Pattern 2: Running from project root
        from src.nlp.parser import parse_prompt
        from src.nlp.complexity_score import compute_complexity, get_complexity_breakdown
        from src.prediction.estimator import EnergyPredictor
        from src.anomaly.detector import AnomalyDetector
        from src.optimization.recomender import PromptOptimizer
        from src.utils.database import DatabaseManager
        from src.utils.logger import get_logger
    except ImportError as e:
        st.error(f"Failed to import modules. Please run from project root: {e}")
        st.stop()


# Initialize logger
logger = get_logger("streamlit_app")


@st.cache_resource
def load_models():
    """Load ML models (cached for performance)."""
    return {
        'energy_predictor': EnergyPredictor(),
        'anomaly_detector': AnomalyDetector(),
        'prompt_optimizer': PromptOptimizer(),
        'database': DatabaseManager()
    }


def main():
    """Main application entry point."""
    # Page configuration
    setup_page_config()
    apply_custom_css()
    
    # Load models
    models = load_models()
    
    # Render header
    render_header()
    
    # Render sidebar and get configuration
    config = render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Analyze Prompt", 
        "⚡ Optimize", 
        "📊 Dashboard",
        "📋 Reports"
    ])
    
    # =========================================================================
    # TAB 1: ANALYZE PROMPT
    # =========================================================================
    with tab1:
        st.header("🔍 Prompt Energy Analysis")
        
        st.markdown("""
            Enter a prompt below to analyze its energy consumption, complexity, 
            and check for anomalies. This helps you understand the environmental 
            impact of your AI interactions.
        """)
        
        # Prompt input
        prompt = st.text_area(
            "Enter your prompt:",
            height=150,
            placeholder="Type or paste your prompt here...",
            help="Enter the prompt you want to analyze for energy consumption"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            analyze_btn = st.button("🔬 Analyze", type="primary", use_container_width=True)
        
        if analyze_btn and prompt:
            with st.spinner("Analyzing prompt..."):
                # Parse prompt
                parsed = parse_prompt(prompt, use_embeddings=False)
                
                # Get complexity
                complexity_score = compute_complexity(prompt)
                complexity_breakdown = get_complexity_breakdown(prompt)
                
                # Predict energy
                energy_result = models['energy_predictor'].predict(
                    prompt,
                    num_layers=config['num_layers'],
                    training_hours=config['training_hours'],
                    flops_per_hour=config['flops_per_hour'],
                    region=config['region']
                )
                
                # Detect anomalies
                anomaly_result = models['anomaly_detector'].detect(
                    prompt, 
                    energy_kwh=energy_result.energy_kwh
                )
                
                # Log to database
                models['database'].log_energy(
                    prompt=prompt,
                    token_count=parsed.token_count,
                    complexity_score=complexity_score,
                    energy_kwh=energy_result.energy_kwh,
                    carbon_kg=energy_result.carbon_footprint_kg,
                    water_liters=energy_result.water_usage_liters,
                    is_anomaly=anomaly_result.is_anomaly,
                    model_params={
                        "num_layers": config['num_layers'],
                        "training_hours": config['training_hours'],
                        "flops_per_hour": config['flops_per_hour']
                    },
                    region=config['region']
                )
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    render_metric_card(
                        "Energy Consumption",
                        f"{energy_result.energy_joules:.2f} J ({energy_result.energy_kwh:.2e} kWh)",
                        color="green" if energy_result.energy_level == "low" else 
                              "yellow" if energy_result.energy_level == "medium" else "red"
                    )
                
                with col2:
                    render_metric_card(
                        "Carbon Footprint",
                        f"{energy_result.carbon_footprint_mg:.4f} mg CO₂",
                        color="blue"
                    )
                
                with col3:
                    render_metric_card(
                        "Token Count",
                        str(parsed.token_count),
                        color="default"
                    )
                
                with col4:
                    render_metric_card(
                        "Complexity Score",
                        f"{complexity_score:.2f}",
                        color="green" if complexity_score < 0.4 else 
                              "yellow" if complexity_score < 0.7 else "red"
                    )
                
                # Energy gauge and complexity breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("⚡ Energy Level")
                    render_energy_gauge(energy_result.energy_joules)  # Pass Joules for gauge
                    
                    st.markdown(f"""
                        **Energy:** {energy_result.energy_joules:.2f} Joules ({energy_result.energy_kwh:.2e} kWh)
                        
                        **Energy Level:** {energy_result.energy_level.upper()}
                        
                        **Compared to Average:** {energy_result.comparison_to_average:+.1f}%
                        
                        **Electricity Cost:** ${energy_result.electricity_cost_usd:.8f}
                        
                        **Water Usage:** {energy_result.water_usage_liters:.6f} L
                    """)
                
                with col2:
                    st.subheader("🧠 Complexity Breakdown")
                    render_complexity_breakdown(complexity_breakdown)
                    
                    st.markdown(f"""
                        **Overall Level:** {complexity_breakdown.get('level', 'N/A').upper()}
                        
                        **Impact:** {complexity_breakdown.get('energy_impact', 'N/A')}
                    """)
                
                # Anomaly detection
                st.markdown("---")
                st.subheader("🚨 Anomaly Detection")
                render_anomaly_alert(anomaly_result.to_dict())
                
                # Parsed features
                with st.expander("📝 Detailed Features"):
                    features = parsed.to_dict()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Text Statistics:**")
                        st.write(f"- Words: {features['word_count']}")
                        st.write(f"- Characters: {features['char_count']}")
                        st.write(f"- Sentences: {features['sentence_count']}")
                        st.write(f"- Avg Word Length: {features['avg_word_length']:.2f}")
                    
                    with col2:
                        st.markdown("**Linguistic Features:**")
                        st.write(f"- Stopword Ratio: {features['stopword_ratio']:.2%}")
                        st.write(f"- Vocabulary Richness: {features['vocabulary_richness']:.2%}")
                        st.write(f"- Lexical Density: {features['lexical_density']:.2%}")
                
                logger.log_energy_prediction(
                    prompt, energy_result.energy_kwh, 
                    features, config
                )
        
        elif analyze_btn:
            st.warning("Please enter a prompt to analyze.")
    
    # =========================================================================
    # TAB 2: OPTIMIZE
    # =========================================================================
    with tab2:
        st.header("⚡ Prompt Optimization")
        
        st.markdown("""
            Optimize your prompts to reduce energy consumption while maintaining 
            semantic meaning. Our system suggests energy-efficient alternatives.
        """)
        
        # Prompt input
        opt_prompt = st.text_area(
            "Enter prompt to optimize:",
            height=150,
            placeholder="Type or paste your prompt here...",
            key="optimize_prompt"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategy = st.selectbox(
                "Optimization Strategy",
                options=["auto", "simplify", "compress", "truncate"],
                help="Choose the optimization strategy"
            )
        
        with col2:
            min_similarity = st.slider(
                "Min Similarity",
                min_value=0.5,
                max_value=0.95,
                value=0.75,
                step=0.05,
                help="Minimum semantic similarity to maintain"
            )
        
        with col3:
            optimize_btn = st.button("🚀 Optimize", type="primary", use_container_width=True)
        
        if optimize_btn and opt_prompt:
            with st.spinner("Optimizing prompt..."):
                # Create optimizer with custom similarity
                optimizer = PromptOptimizer(min_similarity=min_similarity)
                
                # Optimize
                result = optimizer.optimize(
                    opt_prompt,
                    num_layers=config['num_layers'],
                    training_hours=config['training_hours'],
                    flops_per_hour=config['flops_per_hour'],
                    strategy=strategy
                )
                
                # Log optimization
                if result.energy_saved_kwh > 0:
                    models['database'].log_energy(
                        prompt=result.optimized_prompt,
                        token_count=parse_prompt(result.optimized_prompt, use_embeddings=False).token_count,
                        complexity_score=compute_complexity(result.optimized_prompt),
                        energy_kwh=result.optimized_energy_kwh,
                        carbon_kg=result.optimized_energy_kwh * 0.42,
                        water_liters=result.optimized_energy_kwh * 1.8,
                        was_optimized=True,
                        energy_saved_kwh=result.energy_saved_kwh,
                        region=config['region']
                    )
                
                # Display results
                st.markdown("---")
                st.subheader("✨ Optimization Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Note: optimizer returns values in Joules (stored in *_kwh fields for compatibility)
                    render_metric_card(
                        "Energy Saved",
                        f"{result.energy_saved_kwh:.2f} J",
                        delta=f"-{result.energy_reduction_percent:.1f}%",
                        color="green"
                    )
                
                with col2:
                    render_metric_card(
                        "Semantic Similarity",
                        f"{result.semantic_similarity:.1%}",
                        color="blue"
                    )
                
                with col3:
                    render_metric_card(
                        "Complexity Reduced",
                        f"{result.complexity_reduction:.1f}%",
                        color="yellow"
                    )
                
                with col4:
                    render_metric_card(
                        "Optimization Score",
                        f"{result.optimization_score:.2f}",
                        color="default"
                    )
                
                # Before/After comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📝 Original Prompt")
                    st.text_area(
                        "Original:",
                        value=opt_prompt,
                        height=150,
                        disabled=True,
                        key="original_display"
                    )
                    st.caption(f"Energy: {result.original_energy_kwh:.2f} J ({result.original_energy_kwh/3600000:.2e} kWh)")
                
                with col2:
                    st.subheader("✅ Optimized Prompt")
                    st.text_area(
                        "Optimized:",
                        value=result.optimized_prompt,
                        height=150,
                        disabled=True,
                        key="optimized_display"
                    )
                    st.caption(f"Energy: {result.optimized_energy_kwh:.2f} J ({result.optimized_energy_kwh/3600000:.2e} kWh)")
                
                # Energy comparison chart
                st.subheader("📊 Energy Comparison")
                render_energy_comparison(
                    result.original_energy_kwh,
                    result.optimized_energy_kwh
                )
                
                # Alternatives
                st.subheader("🔄 Alternative Suggestions")
                render_alternatives_table(result.alternatives)
                
                # Suggestions
                suggestions = optimizer.get_improvement_suggestions(opt_prompt)
                render_suggestions(suggestions)
                
                # Copy button
                st.markdown("---")
                if st.button("📋 Copy Optimized Prompt"):
                    st.code(result.optimized_prompt)
                    st.success("Prompt displayed above - copy it manually!")
        
        elif optimize_btn:
            st.warning("Please enter a prompt to optimize.")
    
    # =========================================================================
    # TAB 3: DASHBOARD
    # =========================================================================
    with tab3:
        st.header("📊 Energy Dashboard")
        
        # Time range selector
        col1, col2 = st.columns([2, 1])
        
        with col1:
            time_range = st.selectbox(
                "Time Range",
                options=["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
                index=1
            )
        
        with col2:
            refresh_btn = st.button("🔄 Refresh Data")
        
        # Calculate date range
        if time_range == "Last 24 Hours":
            start_date = datetime.now() - timedelta(hours=24)
        elif time_range == "Last 7 Days":
            start_date = datetime.now() - timedelta(days=7)
        elif time_range == "Last 30 Days":
            start_date = datetime.now() - timedelta(days=30)
        else:
            start_date = None
        
        # Get statistics
        stats = models['database'].get_statistics(start_date=start_date)
        
        if stats.get('total_prompts', 0) == 0:
            st.info("No data available for the selected period. Start analyzing prompts to see statistics!")
        else:
            # Overview metrics
            st.subheader("📈 Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                render_metric_card(
                    "Total Prompts",
                    str(stats.get('total_prompts', 0)),
                    color="blue"
                )
            
            with col2:
                render_metric_card(
                    "Total Energy",
                    f"{stats.get('total_energy_kwh', 0):.4f} kWh",
                    color="yellow"
                )
            
            with col3:
                render_metric_card(
                    "Avg Energy/Prompt",
                    f"{stats.get('average_energy_kwh', 0):.6f} kWh",
                    color="green"
                )
            
            with col4:
                render_metric_card(
                    "Carbon Footprint",
                    f"{stats.get('total_carbon_kg', 0):.4f} kg",
                    color="red"
                )
            
            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Anomalies Detected", stats.get('anomaly_count', 0))
            
            with col2:
                st.metric("Anomaly Rate", f"{stats.get('anomaly_rate', 0):.1f}%")
            
            with col3:
                st.metric("Optimizations", stats.get('optimization_count', 0))
            
            with col4:
                st.metric("Energy Saved", f"{stats.get('total_energy_saved_kwh', 0):.4f} kWh")
            
            # Charts
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚡ Energy Distribution")
                energy_dist = {
                    "Low (<0.3 kWh)": max(0, stats.get('total_prompts', 0) // 3),
                    "Medium (0.3-0.7 kWh)": max(0, stats.get('total_prompts', 0) // 3),
                    "High (>0.7 kWh)": max(0, stats.get('total_prompts', 0) // 3)
                }
                render_pie_chart(energy_dist, "Energy Level Distribution")
            
            with col2:
                st.subheader("🌍 Environmental Impact")
                st.markdown(f"""
                    Based on your usage in this period:
                    
                    - **🌳 Trees Equivalent:** {stats.get('total_carbon_kg', 0) / 21:.2f} trees needed to offset
                    - **💧 Water Used:** {stats.get('total_water_liters', 0):.2f} liters
                    - **🏠 Homes Powered:** {stats.get('total_energy_kwh', 0) / 1.2:.2f} hours
                    - **🚗 Car Miles:** {stats.get('total_carbon_kg', 0) / 0.4:.1f} miles equivalent
                """)
            
            # Recent logs
            st.markdown("---")
            st.subheader("📜 Recent Activity")
            
            logs = models['database'].get_logs(start_date=start_date, limit=10)
            if logs:
                import pandas as pd
                df = pd.DataFrame(logs)
                df = df[['timestamp', 'prompt_preview', 'token_count', 'energy_kwh', 'is_anomaly', 'was_optimized']]
                df['is_anomaly'] = df['is_anomaly'].apply(lambda x: '⚠️' if x else '✅')
                df['was_optimized'] = df['was_optimized'].apply(lambda x: '✨' if x else '-')
                df = df.rename(columns={
                    'timestamp': 'Time',
                    'prompt_preview': 'Prompt',
                    'token_count': 'Tokens',
                    'energy_kwh': 'Energy (kWh)',
                    'is_anomaly': 'Status',
                    'was_optimized': 'Optimized'
                })
                st.dataframe(df, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # TAB 4: REPORTS
    # =========================================================================
    with tab4:
        st.header("📋 Transparency Reports")
        
        st.markdown("""
            Generate transparency reports for EU AI Act compliance. Reports include
            comprehensive energy usage statistics and environmental impact metrics.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_start = st.date_input(
                "Report Start Date",
                value=datetime.now() - timedelta(days=30)
            )
        
        with col2:
            report_end = st.date_input(
                "Report End Date",
                value=datetime.now()
            )
        
        with col3:
            generate_btn = st.button("📄 Generate Report", type="primary", use_container_width=True)
        
        if generate_btn:
            with st.spinner("Generating transparency report..."):
                report = models['database'].generate_transparency_report(
                    datetime.combine(report_start, datetime.min.time()),
                    datetime.combine(report_end, datetime.max.time())
                )
                
                st.success(f"Report generated: {report['report_id']}")
                
                render_transparency_report(report)
                
                # Download option
                st.markdown("---")
                import json
                st.download_button(
                    label="📥 Download Report (JSON)",
                    data=json.dumps(report, indent=2),
                    file_name=f"transparency_report_{report['report_id']}.json",
                    mime="application/json"
                )
        
        # Previous reports
        st.markdown("---")
        st.subheader("📚 Previous Reports")
        
        previous_reports = models['database'].get_reports(limit=5)
        
        if previous_reports:
            for report in previous_reports:
                with st.expander(f"Report: {report.get('report_id', 'N/A')} - {report.get('generated_at', 'N/A')[:10]}"):
                    st.markdown(f"""
                        - **Period:** {report.get('period_start', 'N/A')[:10]} to {report.get('period_end', 'N/A')[:10]}
                        - **Total Energy:** {report.get('total_energy_kwh', 0):.4f} kWh
                        - **Total Prompts:** {report.get('total_prompts', 0)}
                        - **Compliance Status:** {report.get('compliance_status', 'N/A')}
                    """)
        else:
            st.info("No previous reports found. Generate your first report above!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
            🌱 <strong>Sustainable AI</strong> - Transparency and Energy-Efficient Prompt Engineering<br>
            Built for CSCN8010 Final Project | EU AI Act Compliance Ready
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
