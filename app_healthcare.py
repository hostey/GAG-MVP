# app_healthcare.py (new file for healthcare dashboard)
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils.simulation_healthcare import HealthcareSimulator, HealthcareResult


def create_healthcare_dashboard():
    """Create the healthcare-specific dashboard."""

    st.set_page_config(
        page_title="GAGS Healthcare Triage Simulator",
        page_icon="🏥",
        layout="wide"
    )

    # Inject custom CSS
    st.markdown("""
        <style>
        .healthcare-card {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            border-left: 5px solid #2196F3;
        }
        .disparity-bad { color: #f44336; }
        .disparity-good { color: #4caf50; }
        .severity-high { background-color: #ffebee; padding: 5px; border-radius: 5px; }
        .severity-low { background-color: #e8f5e9; padding: 5px; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #2196F3 0%, #0d47a1 100%); 
                    border-radius: 15px; color: white; margin-bottom: 30px;">
            <h1 style="margin:0">🏥 Healthcare Triage Fairness Simulator</h1>
            <p style="margin:10px 0 0 0; opacity: 0.9;">Analyzing demographic biases in AI-powered healthcare triage systems</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("🏥 Simulation Parameters")

        col1, col2 = st.columns(2)
        with col1:
            n_patients = st.number_input("Number of Patients", 100, 10000, 1000, 100)
            bias_factor = st.slider("Bias Factor", 0.0, 1.0, 0.3, 0.1)
        with col2:
            rural_percent = st.slider("Rural Population %", 10, 50, 30, 5)
            severity_dist = st.selectbox("Severity Distribution",
                                         ["exponential", "uniform", "normal"])

        st.header("⚙️ Model Configuration")
        hidden_layers = st.multiselect(
            "Hidden Layer Sizes",
            [4, 8, 16, 32, 64],
            default=[16, 8]
        )
        epochs = st.slider("Training Epochs", 50, 500, 100, 50)

    # Initialize simulator
    if 'healthcare_simulator' not in st.session_state:
        st.session_state.healthcare_simulator = HealthcareSimulator()

    if 'healthcare_results' not in st.session_state:
        st.session_state.healthcare_results = []

    # Run simulation button
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🏃 Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Simulating healthcare triage..."):
                result = st.session_state.healthcare_simulator.run_simulation(
                    n_patients=n_patients,
                    bias_factor=bias_factor,
                    model_config={'hidden_dims': hidden_layers},
                    training_config={'epochs': epochs}
                )
                st.session_state.current_healthcare_result = result
                st.session_state.healthcare_results.append(result)
                st.success("Simulation completed!")

    with col2:
        if st.button("📊 View History", use_container_width=True):
            st.session_state.show_history = True

    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.healthcare_results = []
            st.rerun()

    # Display current results if available
    if 'current_healthcare_result' in st.session_state:
        result = st.session_state.current_healthcare_result

        # Metrics cards
        st.markdown("### 📈 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{result.accuracy:.1f}%")
        with col2:
            equity = result.fairness_metrics.get('fairness_score', 100)
            st.metric("Equity Score", f"{equity:.1f}")
        with col3:
            disparity = result.bias_metrics.get('rural_urban_disparity', 0)
            st.metric("Rural-Urban Gap", f"{disparity:.1f}%")
        with col4:
            ses_disparity = result.bias_metrics.get('ses_disparity', 0)
            st.metric("SES Disparity", f"{ses_disparity:.1f}%")

        # Visualizations
        st.markdown("### 📊 Fairness Analysis")
        tab1, tab2, tab3 = st.tabs(["Demographic Parity", "Accuracy Breakdown", "Bias Analysis"])

        with tab1:
            # Demographic parity chart
            fig = go.Figure(data=[
                go.Bar(name='Rural', x=['Accuracy'], y=[result.bias_metrics.get('rural_accuracy', 0)]),
                go.Bar(name='Urban', x=['Accuracy'], y=[result.bias_metrics.get('urban_accuracy', 0)])
            ])
            fig.update_layout(barmode='group', title="Accuracy by Location")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # SES accuracy comparison
            fig = go.Figure(data=[
                go.Bar(name='Low SES', x=['Accuracy'], y=[result.bias_metrics.get('low_ses_accuracy', 0)]),
                go.Bar(name='High SES', x=['Accuracy'], y=[result.bias_metrics.get('high_ses_accuracy', 0)])
            ])
            fig.update_layout(barmode='group', title="Accuracy by Socioeconomic Status")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Radar chart for fairness dimensions
            categories = ['Accuracy', 'Equity', 'Equal Opportunity', 'Demographic Parity']
            values = [
                result.accuracy,
                equity,
                100 - (result.fairness_metrics.get('equal_opportunity', 0) * 100),
                100 - (result.fairness_metrics.get('demographic_parity', 0) * 100)
            ]

            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                              title="Fairness Dimensions")
            st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.markdown("### 🎯 Recommendations")
        if equity < 70:
            st.error("**High Bias Detected:** Implement fairness-aware training techniques")
        elif disparity > 15:
            st.warning("**Significant Rural-Urban Gap:** Consider location-aware algorithms")
        else:
            st.success("**Good Fairness Performance:** Continue monitoring for bias")

    # Historical data
    if st.session_state.get('show_history', False) and st.session_state.healthcare_results:
        st.markdown("### 📜 Simulation History")

        history_df = pd.DataFrame([{
            'Run': i + 1,
            'Accuracy': r.accuracy,
            'Equity Score': r.fairness_metrics.get('fairness_score', 0),
            'Rural-Urban Gap': r.bias_metrics.get('rural_urban_disparity', 0),
            'Bias Factor': r.parameters.get('bias_factor', 0)
        } for i, r in enumerate(st.session_state.healthcare_results)])

        st.dataframe(history_df)

        # History chart
        fig = px.line(history_df, x='Run', y=['Accuracy', 'Equity Score'],
                      title="Performance Over Time")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    create_healthcare_dashboard()