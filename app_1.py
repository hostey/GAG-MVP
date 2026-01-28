# app.py - Main entry point (Security/Resilience module)
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from utils.simulation_resilience import SimulationEngine, SimulationResult, ThreatDetector

# Initialize session state
if 'security_history' not in st.session_state:
    st.session_state.security_history = []

# Page configuration
st.set_page_config(
    page_title="GAGS Security Resilience",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
with open("assets/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Dashboard Header
st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%); 
            border-radius: 15px; color: white; margin-bottom: 30px;">
        <h1 style="margin:0; font-size: 2.5rem; font-weight: 800;">🛡️ GAGS Security Resilience Module</h1>
        <p style="margin:10px 0 0 0; opacity: 0.9; font-size: 1.1rem;">
            Surveillance-Liberty Trade-off Analysis & Threat Simulation
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Security Parameters")

    with st.expander("Policy Configuration", expanded=True):
        surveillance_level = st.slider(
            "🔍 Surveillance Intensity",
            0, 10, 5,
            help="Level of monitoring and data collection"
        )
        liberty_threshold = st.slider(
            "🗽 Liberty Protection",
            0, 10, 7,
            help="Degree of privacy and autonomy protection"
        )

    with st.expander("Attack Configuration", expanded=True):
        poison_rate = st.slider(
            "🧪 Attack Severity",
            0.0, 0.5, 0.1, 0.01,
            help="Data poisoning attack strength"
        )

    with st.expander("Model Settings", expanded=False):
        hidden_layers = st.multiselect(
            "Model Architecture",
            [4, 8, 16, 32],
            default=[8, 4]
        )
        epochs = st.slider("Training Epochs", 20, 200, 50)


# Simulation Engine
@st.cache_resource
def get_simulation_engine():
    return SimulationEngine()


engine = get_simulation_engine()

# Simulation Controls
st.markdown('<div class="section-header">🎮 Simulation Controls</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Run Security Simulation", type="primary", use_container_width=True):
        with st.spinner("Running security simulation..."):
            # Run simulation
            result = engine.run_simulation(
                surveillance_level=surveillance_level,
                liberty_threshold=liberty_threshold,
                poison_rate=poison_rate,
                model_config={"hidden_dims": hidden_layers}
            )

            # Store in session state
            st.session_state.current_result = result

            # Add to history
            st.session_state.security_history.append({
                'timestamp': datetime.now(),
                'surveillance': surveillance_level,
                'liberty': liberty_threshold,
                'safety': result.safety_score,
                'liberty_score': result.liberty_score,
                'attack': poison_rate * 100
            })

            st.success("✅ Simulation completed!")

with col2:
    if st.button("🔥 Launch Attack", type="secondary", use_container_width=True):
        if 'current_result' in st.session_state:
            st.warning("Attack simulation launched! Check metrics for impact.")
        else:
            st.warning("Please run a simulation first!")

with col3:
    if st.button("🔄 Reset", type="secondary", use_container_width=True):
        st.session_state.security_history = []
        if 'current_result' in st.session_state:
            del st.session_state.current_result
        st.rerun()

# Display results if available
if 'current_result' in st.session_state:
    result = st.session_state.current_result

    # Metrics Cards
    st.markdown('<div class="section-header">📊 Security Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="metric-card security-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">Safety Score</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #333; margin: 10px 0;">
                    {result.safety_score:.1f}%
                </div>
                <div style="font-size: 0.85rem; color: #777;">System security & protection</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card security-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">Liberty Score</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #333; margin: 10px 0;">
                    {result.liberty_score:.1f}/100
                </div>
                <div style="font-size: 0.85rem; color: #777;">Privacy & freedom protection</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        resilience = result.metrics.get('resilience_score', 0)
        st.markdown(f"""
            <div class="metric-card security-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">Resilience Index</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #333; margin: 10px 0;">
                    {resilience:.1f}
                </div>
                <div style="font-size: 0.85rem; color: #777;">Overall system robustness</div>
            </div>
        """, unsafe_allow_html=True)

    # Additional Metrics
    col4, col5, col6 = st.columns(3)

    with col4:
        tradeoff = result.metrics.get('tradeoff_index', 0) * 100
        st.metric("Trade-off Index", f"{tradeoff:.1f}%",
                  "Lower is better" if tradeoff < 30 else "Higher trade-off")

    with col5:
        st.metric("Surveillance Impact", f"{result.metrics.get('surveillance_impact', 0):.1f}")

    with col6:
        st.metric("Attack Strength", f"{result.metrics.get('poison_rate', 0):.1f}%")

    # Visualizations
    st.markdown('<div class="section-header">📈 Analysis & Insights</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Performance Analysis", "Historical Trends", "Recommendations"])

    with tab1:
        # Performance radar chart
        categories = ['Safety', 'Liberty', 'Resilience', 'Stability', 'Accuracy']
        values = [
            result.safety_score,
            result.liberty_score,
            resilience,
            78.0,  # Placeholder
            85.0  # Placeholder
        ]

        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            line_color='#1E88E5'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=400,
            title="Security Performance Profile"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Historical trends
        if st.session_state.security_history:
            history_df = pd.DataFrame(st.session_state.security_history)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['safety'],
                mode='lines+markers',
                name='Safety',
                line=dict(color='#4CAF50', width=3)
            ))

            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['liberty_score'],
                mode='lines+markers',
                name='Liberty',
                line=dict(color='#2196F3', width=3)
            ))

            fig.update_layout(
                title="Security Performance History",
                xaxis_title="Time",
                yaxis_title="Score",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run more simulations to see historical trends")

    with tab3:
        # Recommendations
        st.markdown("### 🎯 Policy Recommendations")

        recommendations = []

        if result.safety_score < 70:
            recommendations.append("**Increase threat detection capabilities**")

        if result.liberty_score < 60:
            recommendations.append("**Strengthen privacy protection measures**")

        if result.metrics.get('tradeoff_index', 0) > 0.3:
            recommendations.append("**Balance surveillance with transparency**")

        if poison_rate > 0.2:
            recommendations.append("**Implement robust data validation pipelines**")

        if recommendations:
            for rec in recommendations:
                st.warning(f"• {rec}")
        else:
            st.success("✅ Current policies appear balanced and effective!")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🛡️ GAGS Security Resilience Module v2.0 | Part of Global AI Governance Sandbox</p>
        <p style="font-size: 0.8rem;">Navigate to Healthcare Equity module for medical triage fairness analysis</p>
    </div>
""", unsafe_allow_html=True)