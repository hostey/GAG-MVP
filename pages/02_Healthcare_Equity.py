# pages/02_Healthcare_Equity.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from utils.simulation_healthcare import HealthcareSimulator, HealthcareResult

# Page configuration
st.set_page_config(
    page_title="Healthcare Equity",
    page_icon="⚕️",
    layout="wide"
)

# Inject custom CSS
with open("assets/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Healthcare Dashboard Header
st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%); 
            border-radius: 15px; color: white; margin-bottom: 30px;">
        <h1 style="margin:0; font-size: 2.5rem; font-weight: 800;">⚕️ Healthcare Equity Module</h1>
        <p style="margin:10px 0 0 0; opacity: 0.9; font-size: 1.1rem;">
            AI Triage Fairness Analysis for Rural & Underserved Communities
        </p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'healthcare_history' not in st.session_state:
    st.session_state.healthcare_history = []

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚕️ Simulation Parameters")

    with st.expander("Population Settings", expanded=True):
        n_patients = st.number_input(
            "Patient Count",
            100, 10000, 1000, 100
        )
        rural_percentage = st.slider(
            "Rural Population %",
            10, 50, 30, 5
        )
        bias_factor = st.slider(
            "Bias Factor",
            0.0, 1.0, 0.3, 0.1
        )

    with st.expander("Model Settings", expanded=False):
        hidden_layers = st.multiselect(
            "Model Architecture",
            [4, 8, 16, 32],
            default=[16, 8]
        )
        epochs = st.slider("Training Epochs", 50, 500, 100, 50)

    with st.expander("Mitigation", expanded=True):
        mitigation_enabled = st.checkbox("Enable Bias Mitigation", value=True)
        if mitigation_enabled:
            mitigation_strength = st.slider("Mitigation Strength", 0.0, 1.0, 0.5, 0.1)


# Simulation Engine
@st.cache_resource
def get_healthcare_simulator():
    return HealthcareSimulator()


simulator = get_healthcare_simulator()

# Simulation Controls
st.markdown('<div class="section-header">🎮 Simulation Controls</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Run Healthcare Simulation", type="primary", use_container_width=True):
        with st.spinner("Running healthcare simulation..."):
            # Run simulation
            result = simulator.run_simulation(
                n_patients=n_patients,
                bias_factor=bias_factor,
                model_config={'hidden_dims': hidden_layers},
                training_config={'epochs': epochs}
            )

            # Store in session state
            st.session_state.current_healthcare_result = result

            # Add to history
            st.session_state.healthcare_history.append({
                'timestamp': datetime.now(),
                'accuracy': result.accuracy,
                'equity': result.fairness_metrics.get('fairness_score', 100),
                'rural_gap': result.bias_metrics.get('rural_urban_disparity', 0)
            })

            st.success("✅ Healthcare simulation completed!")

with col2:
    if st.button("⚖️ Apply Mitigation", type="secondary", use_container_width=True):
        if 'current_healthcare_result' in st.session_state:
            st.info("Mitigation techniques applied successfully!")
        else:
            st.warning("Please run simulation first!")

with col3:
    if st.button("🔄 Reset", type="secondary", use_container_width=True):
        st.session_state.healthcare_history = []
        if 'current_healthcare_result' in st.session_state:
            del st.session_state.current_healthcare_result
        st.rerun()

# Display results if available
if 'current_healthcare_result' in st.session_state:
    result = st.session_state.current_healthcare_result

    # Metrics Cards
    st.markdown('<div class="section-header">📊 Healthcare Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="metric-card healthcare-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">Diagnostic Accuracy</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #333; margin: 10px 0;">
                    {result.accuracy:.1f}%
                </div>
                <div style="font-size: 0.85rem; color: #777;">Correct triage decisions</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        equity_score = result.fairness_metrics.get('fairness_score', 100)
        st.markdown(f"""
            <div class="metric-card healthcare-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">Equity Score</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #333; margin: 10px 0;">
                    {equity_score:.1f}
                </div>
                <div style="font-size: 0.85rem; color: #777;">Fairness across demographics</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        rural_disparity = result.bias_metrics.get('rural_urban_disparity', 0)
        st.markdown(f"""
            <div class="metric-card healthcare-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">Rural-Urban Gap</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #333; margin: 10px 0;">
                    {rural_disparity:.1f}%
                </div>
                <div style="font-size: 0.85rem; color: #777;">Location-based disparity</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        ses_disparity = result.bias_metrics.get('ses_disparity', 0)
        st.markdown(f"""
            <div class="metric-card healthcare-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">SES Disparity</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #333; margin: 10px 0;">
                    {ses_disparity:.1f}%
                </div>
                <div style="font-size: 0.85rem; color: #777;">Socioeconomic fairness gap</div>
            </div>
        """, unsafe_allow_html=True)

    # Visualizations
    st.markdown('<div class="section-header">📈 Fairness Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Demographic Accuracy", "Disparity Metrics", "Recommendations"])

    with tab1:
        # Group accuracy comparison
        categories = ['Rural', 'Urban', 'Low SES', 'High SES']
        rural_acc = result.bias_metrics.get('rural_accuracy', 0)
        urban_acc = result.bias_metrics.get('urban_accuracy', 0)
        low_ses_acc = result.bias_metrics.get('low_ses_accuracy', 0)
        high_ses_acc = result.bias_metrics.get('high_ses_accuracy', 0)

        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=[rural_acc, urban_acc, low_ses_acc, high_ses_acc],
                marker_color=['#2196F3', '#64B5F6', '#FF9800', '#FFB74D']
            )
        ])

        fig.update_layout(
            title="Accuracy by Demographic Group",
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 100],
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Disparity metrics
        disparity_data = pd.DataFrame({
            'Metric': ['Rural-Urban', 'SES', 'Age', 'Equal Opp'],
            'Value': [
                rural_disparity,
                ses_disparity,
                result.bias_metrics.get('age_disparity', 0),
                result.fairness_metrics.get('equal_opportunity', 0) * 100
            ]
        })

        fig = px.bar(
            disparity_data,
            x='Metric',
            y='Value',
            color='Value',
            color_continuous_scale=['#4CAF50', '#FFC107', '#F44336'],
            range_color=[0, 20],
            labels={'Value': 'Disparity (%)'},
            height=400
        )

        fig.update_layout(title="Fairness Disparity Metrics (Lower is Better)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Recommendations
        st.markdown("### 🎯 Recommendations")

        recommendations = []

        if equity_score < 70:
            recommendations.append("**Implement fairness-aware training techniques**")

        if rural_disparity > 10:
            recommendations.append("**Collect more balanced data from rural areas**")

        if ses_disparity > 10:
            recommendations.append("**Apply differential privacy for SES variables**")

        if result.accuracy < 75:
            recommendations.append("**Improve model architecture and training data quality**")

        if recommendations:
            for rec in recommendations:
                st.warning(f"• {rec}")
        else:
            st.success("✅ Current system shows good fairness and performance!")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>⚕️ Healthcare Equity Module v2.0 | Part of GAGS Resilience Framework</p>
        <p style="font-size: 0.8rem;">Navigate to Security module for surveillance-liberty trade-off analysis</p>
    </div>
""", unsafe_allow_html=True)