# pages/1_🏥_Healthcare_Equity.py
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.governance_logic import run_simple_simulation
from utils.config import simulation_config, settings

# ───────────────────────────────────────────────
# Page Configuration
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Equity • GAGS",
    layout="wide",
    page_icon=""
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .bias-tag {
        display: inline-block;
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Header Section
# ───────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; color:white;"> Healthcare Equity Simulation</h1>
    <p style="margin:0; opacity:0.9; font-size:1.1rem;">
        Explore how different types of bias and data poisoning affect AI fairness and accuracy in healthcare
    </p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Sidebar – Simulation Controls
# ───────────────────────────────────────────────
with st.sidebar:
    # Logo/Title
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h2 style="color:#667eea; margin:0;"> Simulation Controls</h2>
        <p style="color:#666; font-size:0.9rem;">Configure your simulation parameters</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Bias Configuration
    st.subheader(" Bias Configuration")

    selected_biases = st.multiselect(
        "**Bias Types to Inject**",
        options=simulation_config.BIAS_TYPES,
        default=["demographic", "historical", "selection"],
        help="Select which bias mechanisms to simulate in your model",
        placeholder="Choose bias types..."
    )

    # Display selected biases as tags
    if selected_biases:
        tags_html = "".join([f'<span class="bias-tag">{bias}</span>' for bias in selected_biases])
        st.markdown(f"**Selected biases:**<br>{tags_html}", unsafe_allow_html=True)

    bias_intensity = st.slider(
        "**Bias Intensity**",
        min_value=0.0,
        max_value=simulation_config.MAX_BIAS_FACTOR,
        value=0.30,
        step=0.05,
        format="%.2f",
        help="Higher values indicate stronger bias injection"
    )

    # Visual indicator for bias intensity
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Low")
    with col2:
        st.progress(bias_intensity / simulation_config.MAX_BIAS_FACTOR)
    with col3:
        st.caption("High")

    st.divider()

    # Attack Configuration
    st.subheader("Attack Configuration")

    poison_rate = st.slider(
        "**Data Poisoning Rate**",
        min_value=simulation_config.ATTACK_TYPES["data_poisoning"]["min_rate"],
        max_value=simulation_config.ATTACK_TYPES["data_poisoning"]["max_rate"],
        value=simulation_config.ATTACK_TYPES["data_poisoning"]["default"],
        step=0.01,
        format="%.1f%%",
        help="Percentage of training data to poison"
    )

    # Data Configuration
    st.subheader(" Data Configuration")

    sample_size = st.slider(
        "**Sample Size**",
        min_value=1000,
        max_value=settings.DEFAULT_N_SAMPLES * 2,
        value=settings.DEFAULT_N_SAMPLES,
        step=1000,
        help="Number of data samples for simulation"
    )

    st.divider()

    # Run Button
    run_button = st.button(
        " **RUN SIMULATION**",
        type="primary",
        use_container_width=True,
        help="Click to execute simulation with current parameters"
    )

# ───────────────────────────────────────────────
# Main Content Area
# ───────────────────────────────────────────────
if run_button:
    with st.spinner("️ Running simulation... This may take a moment."):
        # Run simulation
        result = run_simple_simulation(
            bias_types=selected_biases,
            bias_factor=bias_intensity,
            poison_rate=poison_rate,
            n_samples=sample_size,
            n_features=10
        )

    # ── Performance Metrics Section ───────────────────────────────
    st.markdown("###  Performance Dashboard")

    # Main Metrics Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Model Accuracy",
            value=f"{result['accuracy']:.2%}",
            delta=f"{(result['accuracy'] - simulation_config.ACCURACY_TARGET):+.1%}" if result['accuracy'] else None,
            delta_color="inverse" if result['accuracy'] < simulation_config.ACCURACY_TARGET else "normal",
            help="Overall model prediction accuracy"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Fairness Score",
            value=f"{result['fairness_score']:.3f}",
            delta=f"{(result['fairness_score'] - simulation_config.FAIRNESS_TARGET):+.3f}" if result[
                'fairness_score'] else None,
            delta_color="inverse" if result['fairness_score'] < simulation_config.FAIRNESS_TARGET else "normal",
            help="1.0 = perfect fairness between demographic groups"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Data Integrity",
            value=f"{100 - (result['poisoned_samples'] / result['sample_size_after_bias'] * 100):.1f}%",
            delta=None,
            help="Percentage of clean data after poisoning"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Effective Samples",
            value=f"{result['sample_size_after_bias']:,}",
            delta=f"{result['sample_size_after_bias'] - sample_size:+,}" if result['sample_size_after_bias'] else None,
            help="Final dataset size after bias application"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Detailed Analysis Section ───────────────────────────────
    st.divider()
    st.markdown("###  Detailed Analysis")

    tab1, tab2, tab3 = st.tabs(["Performance Gauges", " Impact Analysis", " Simulation Details"])

    with tab1:
        # Create dual gauge chart
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=("Accuracy Gauge", "Fairness Gauge")
        )

        # Accuracy Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=result['accuracy'] * 100,
            title={'text': "Accuracy", 'font': {'size': 20}},
            delta={'reference': simulation_config.ACCURACY_TARGET * 100},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "royalblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 85], 'color': "gray"},
                    {'range': [85, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': simulation_config.ACCURACY_TARGET * 100
                }
            }
        ), row=1, col=1)

        # Fairness Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=result['fairness_score'] * 100,
            title={'text': "Fairness", 'font': {'size': 20}},
            delta={'reference': simulation_config.FAIRNESS_TARGET * 100},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 85], 'color': "gray"},
                    {'range': [85, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': simulation_config.FAIRNESS_TARGET * 100
                }
            }
        ), row=1, col=2)

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            # Create impact visualization
            fig = px.bar(
                x=["Bias Impact", "Poisoning Impact", "Data Quality"],
                y=[bias_intensity * 100, poison_rate * 100,
                   (result['sample_size_after_bias'] / sample_size) * 100],
                title="Impact Factors Comparison",
                labels={"x": "Factor", "y": "Impact (%)"},
                color_discrete_sequence=["#FF6B6B", "#4ECDC4", "#45B7D1"]
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("###  Impact Summary")
            st.markdown(f"""
            <div class="{'warning-box' if bias_intensity > 0.5 else 'success-box'}">
                <strong>Bias Impact:</strong> {' High' if bias_intensity > 0.5 else ' Moderate'} bias injection
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="{'warning-box' if poison_rate > 0.1 else 'success-box'}">
                <strong>Poisoning Impact:</strong> {' Critical' if poison_rate > 0.1 else ' Controlled'} poisoning level
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="{'warning-box' if result['accuracy'] < simulation_config.ACCURACY_TARGET else 'success-box'}">
                <strong>Accuracy Status:</strong> {' Below target' if result['accuracy'] < simulation_config.ACCURACY_TARGET else ' Above target'}
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        # Simulation details in expandable sections
        with st.expander(" Simulation Configuration", expanded=True):
            config_col1, config_col2 = st.columns(2)
            with config_col1:
                st.write("**Bias Types Applied:**")
                for bias in result.get('applied_biases', selected_biases):
                    st.code(bias)
            with config_col2:
                st.write("**Parameters:**")
                st.write(f"- Bias Intensity: {bias_intensity}")
                st.write(f"- Poison Rate: {poison_rate:.1%}")
                st.write(f"- Initial Samples: {sample_size:,}")

        with st.expander(" Detailed Metrics", expanded=False):
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.write("**Performance Metrics:**")
                st.write(f"- Accuracy: {result['accuracy']:.2%}")
                st.write(f"- Fairness Score: {result['fairness_score']:.3f}")
                st.write(f"- Poisoned Samples: {result['poisoned_samples']:,}")
            with metrics_col2:
                st.write("**Target Comparison:**")
                st.write(f"- Target Accuracy: {simulation_config.ACCURACY_TARGET:.0%}")
                st.write(f"- Target Fairness: {simulation_config.FAIRNESS_TARGET:.2f}")
                st.write(f"- Bias Types Applied: {result.get('applied_biases', len(selected_biases))}")

    # ── Recommendations Section ───────────────────────────────
    st.divider()
    st.markdown("###  Recommendations")

    if result['accuracy'] < simulation_config.ACCURACY_TARGET or result[
        'fairness_score'] < simulation_config.FAIRNESS_TARGET:
        st.warning("""
        ** Model Performance Alert:**

        Your simulation shows degraded performance. Consider:
        1. **Reduce bias intensity** to improve fairness
        2. **Implement data validation** to detect poisoning
        3. **Use fairness-aware algorithms** for better equity
        4. **Increase dataset diversity** to reduce bias impact
        """)
    else:
        st.success("""
        ** Good Performance Achieved:**

        Your model meets target objectives. To maintain performance:
        1. **Continue monitoring** bias and fairness metrics
        2. **Regularly audit** data quality
        3. **Implement continuous validation** pipelines
        4. **Document all bias mitigation strategies**
        """)

    st.caption("*Note: This is a simulation. Real-world models require additional validation and ethical review.*")

else:
    # Welcome State
    st.markdown("""
    ##  Welcome to Healthcare Equity Simulation

    This tool helps you understand how bias and data poisoning affect AI models in healthcare settings.

    ###  What You Can Explore:

    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="text-align:center; padding:1rem; background:#f8f9fa; border-radius:10px;">
            <h3> Bias Types</h3>
            <p>Simulate demographic, historical, and selection biases</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align:center; padding:1rem; background:#f8f9fa; border-radius:10px;">
            <h3> Data Attacks</h3>
            <p>Test resilience against data poisoning attacks</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align:center; padding:1rem; background:#f8f9fa; border-radius:10px;">
            <h3> Performance</h3>
            <p>Monitor accuracy and fairness metrics</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Quick Start Guide
    st.markdown("""
    ###  Quick Start Guide:

    1. **Configure** simulation parameters in the sidebar
    2. **Select** which bias types to inject
    3. **Adjust** intensity and poisoning levels
    4. **Click** "RUN SIMULATION" to see results

    """)

    # Example Scenarios
    with st.expander(" Try These Example Scenarios"):
        scenario_col1, scenario_col2 = st.columns(2)

        with scenario_col1:
            st.write("**High Bias, Low Poisoning**")
            st.caption("Bias Intensity: 0.8 | Poison Rate: 0.02")
            st.write("*Tests fairness impact*")

        with scenario_col2:
            st.write("**Low Bias, High Poisoning**")
            st.caption("Bias Intensity: 0.1 | Poison Rate: 0.15")
            st.write("*Tests resilience to attacks*")

    st.info(
        " **Tip:** Start with moderate settings and gradually increase complexity to understand interactions between different factors.")

# Footer
st.divider()
st.caption("Healthcare Equity Simulation • GAGS Framework • v1.0 • [Learn More](https://example.com)")