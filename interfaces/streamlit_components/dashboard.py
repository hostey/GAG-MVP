# interfaces/streamlit_components/dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import asyncio
import json

from config.settings import settings
from config.constants import COLOR_SCHEMES, THRESHOLDS, DEFAULT_PARAMETERS, ThreatLevel
from core.engine.simulation_engine import SimulationOrchestrator, SimulationResult
from interfaces.streamlit_components.visualizations import (
    create_metric_cards,
    create_performance_radar,
    create_trend_chart,
    create_confusion_matrix,
    create_feature_importance
)
from interfaces.streamlit_components.widgets import (
    create_parameter_slider,
    create_model_selector,
    create_strategy_selector,
    create_export_buttons
)
from services.cache_service import CacheService
from services.export_service import ExportService


class SecurityDashboard:
    """Professional Security Resilience Dashboard."""

    def __init__(self):
        self.orchestrator = SimulationOrchestrator(max_concurrent=2)
        self.cache_service = CacheService()
        self.export_service = ExportService()
        self.color_scheme = COLOR_SCHEMES['security']

        # Initialize session state
        self._init_session_state()

        # Setup page
        self._setup_page()

    def _init_session_state(self):
        """Initialize Streamlit session state."""
        if 'simulation_history' not in st.session_state:
            st.session_state.simulation_history = []

        if 'current_result' not in st.session_state:
            st.session_state.current_result = None

        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = []

        if 'dashboard_tab' not in st.session_state:
            st.session_state.dashboard_tab = "simulation"

    def _setup_page(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title=f"{settings.APP_NAME} - Security",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/docs',
                'Report a bug': 'https://github.com/your-repo/issues',
                'About': f'### {settings.APP_NAME} v{settings.APP_VERSION}\nSecurity Resilience Module'
            }
        )

        # Inject custom CSS
        self._inject_custom_css()

    def _inject_custom_css(self):
        """Inject professional CSS styling."""
        st.markdown(f"""
            <style>
            /* Dashboard Theme */
            :root {{
                --primary-color: {self.color_scheme['primary']};
                --secondary-color: {self.color_scheme['secondary']};
                --accent-color: {self.color_scheme['accent']};
                --success-color: {self.color_scheme['success']};
                --warning-color: {self.color_scheme['warning']};
                --danger-color: {self.color_scheme['danger']};
            }}

            .main {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}

            /* Professional Header */
            .dashboard-header {{
                background: linear-gradient(135deg, 
                    var(--primary-color) 0%, 
                    var(--secondary-color) 100%);
                border-radius: 16px;
                color: white;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            }}

            /* Metric Cards */
            .metric-card {{
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                border-left: 5px solid var(--primary-color);
                margin: 10px 0;
                transition: all 0.3s ease;
                height: 100%;
            }}

            .metric-card:hover {{
                transform: translateY(-3px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.12);
            }}

            .metric-card.danger {{
                border-left-color: var(--danger-color);
            }}

            .metric-card.warning {{
                border-left-color: var(--warning-color);
            }}

            .metric-card.success {{
                border-left-color: var(--success-color);
            }}

            /* Section Headers */
            .section-header {{
                padding: 16px 0 10px 0;
                margin: 24px 0 16px 0;
                border-bottom: 2px solid #e0e0e0;
                color: #333;
                font-weight: 700;
                font-size: 1.5rem;
            }}

            /* Button Styling */
            .stButton > button {{
                border-radius: 10px;
                padding: 12px 24px;
                font-weight: 600;
                transition: all 0.3s ease;
                border: none;
                width: 100%;
            }}

            .primary-button {{
                background: linear-gradient(135deg, 
                    var(--primary-color) 0%, 
                    var(--secondary-color) 100%);
                color: white;
            }}

            .secondary-button {{
                background: white;
                color: var(--primary-color);
                border: 2px solid var(--primary-color) !important;
            }}

            .danger-button {{
                background: linear-gradient(135deg, 
                    var(--danger-color) 0%, 
                    #c62828 100%);
                color: white;
            }}

            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 7px 20px rgba(0,0,0,0.2);
            }}

            /* Tab Styling */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
                background: transparent;
            }}

            .stTabs [data-baseweb="tab"] {{
                background: white;
                border-radius: 10px 10px 0 0;
                padding: 12px 24px;
                font-weight: 600;
                color: #666;
                border: 2px solid transparent;
                border-bottom: none;
                transition: all 0.3s ease;
            }}

            .stTabs [data-baseweb="tab"]:hover {{
                color: var(--primary-color);
                background: #f8f9fa;
            }}

            .stTabs [data-baseweb="tab"][aria-selected="true"] {{
                background: white;
                color: var(--primary-color);
                border-color: #e0e0e0;
                border-bottom-color: white;
                box-shadow: 0 -3px 10px rgba(0,0,0,0.05);
            }}

            /* Progress Bar */
            .stProgress > div > div > div > div {{
                background: linear-gradient(90deg, 
                    var(--primary-color) 0%, 
                    var(--accent-color) 100%);
            }}

            /* Custom Scrollbar */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}

            ::-webkit-scrollbar-track {{
                background: #f1f1f1;
                border-radius: 4px;
            }}

            ::-webkit-scrollbar-thumb {{
                background: var(--primary-color);
                border-radius: 4px;
            }}

            ::-webkit-scrollbar-thumb:hover {{
                background: var(--secondary-color);
            }}
            </style>
        """, unsafe_allow_html=True)

    def render_header(self):
        """Render the professional dashboard header."""
        st.markdown(f"""
            <div class="dashboard-header">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <h1 style="margin:0; font-size: 2.5rem; font-weight: 800;">
                            🛡️ Security Resilience Dashboard
                        </h1>
                        <p style="margin:8px 0 0 0; opacity: 0.9; font-size: 1.1rem;">
                            Advanced Threat Detection & Surveillance-Liberty Trade-off Analysis
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Quick stats bar
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Simulations", len(st.session_state.simulation_history))

        with col2:
            success_rate = self._calculate_success_rate()
            st.metric("Success Rate", f"{success_rate:.1f}%")

        with col3:
            avg_safety = self._calculate_average_metric('safety_score')
            st.metric("Avg Safety", f"{avg_safety:.1f}%")

        with col4:
            avg_liberty = self._calculate_average_metric('liberty_score')
            st.metric("Avg Liberty", f"{avg_liberty:.1f}%")

    def _calculate_success_rate(self) -> float:
        """Calculate simulation success rate."""
        if not st.session_state.simulation_history:
            return 0.0

        successful = sum(1 for r in st.session_state.simulation_history
                         if r.get('status') == 'completed')
        return (successful / len(st.session_state.simulation_history)) * 100

    def _calculate_average_metric(self, metric: str) -> float:
        """Calculate average of a specific metric."""
        if not st.session_state.simulation_history:
            return 0.0

        values = [r.get(metric, 0) for r in st.session_state.simulation_history
                  if isinstance(r.get(metric), (int, float))]

        return np.mean(values) if values else 0.0

    def render_sidebar(self) -> Dict[str, Any]:
        """Render the configuration sidebar."""
        with st.sidebar:
            # Simulation Strategy
            st.markdown("### Simulation Strategy")
            strategy = create_strategy_selector(key="sidebar_sim_strategy")

            # Model Configuration
            st.markdown("### Model Configuration")
            with st.expander("Architecture", expanded=True):
                hidden_layers = create_model_selector(key="sidebar_model_selector")

                dropout_rate = st.slider(
                    "Dropout Rate",
                    0.0, 0.5,
                    value=settings.DEFAULT_DROPOUT_RATE,
                    step=0.05,
                    help="Regularization to prevent overfitting",
                    key="slider_dropout_rate"  # stable key
                )

                activation = st.selectbox(
                    "Activation Function",
                    ["relu", "leaky_relu", "tanh", "selu"],
                    index=0,
                    key="activation_function"  # stable key
                )
            # Policy Parameters
            st.markdown("### Policy Parameters")
            surveillance_level = create_parameter_slider(
                "surveillance_level", " Surveillance Intensity",
                "Level of monitoring", key="slider_surveillance_level"
            )

            liberty_threshold = create_parameter_slider(
                "liberty_threshold", "🗽 Liberty Protection",
                "Degree of privacy protection", key="slider_liberty_threshold"
            )

            # Attack Configuration
            st.markdown("### ⚔️ Attack Configuration")
            with st.expander("Threat Scenarios", expanded=True):
                poison_rate = create_parameter_slider(
                    "poison_rate", " Data Poisoning",
                    "Percentage of malicious data",
                    0.0, 0.5, 0.1, 0.01,
                    key="slider_poison_rate"
                )
                noise_level = create_parameter_slider(
                    "noise_level", " Noise Level",
                    "Amount of noise",
                    0.0, 2.0, 1.0, 0.1,
                    key="slider_noise_level"
                )

            # Simulation Settings
            st.markdown("### ⚡ Simulation Settings")
            with st.expander("Advanced", expanded=False):
                n_samples = st.number_input(
                    "Sample Size",
                    min_value=1000,
                    max_value=100000,
                    value=settings.DEFAULT_N_SAMPLES,
                    step=1000,
                    help="Number of data samples to generate",
                    key="number_samples"
                )

                epochs = create_parameter_slider(
                    "epochs", "Training Epochs",
                    "Number of training epochs",
                    10, 500, 100, 10,
                    key="slider_epochs"
                )
                learning_rate = create_parameter_slider(
                    "learning_rate", "Learning Rate",
                    "Learning rate for training",
                    0.0001, 0.1, 0.01, 0.0001,
                    key="slider_learning_rate"
                )

            # Compile parameters
            parameters = {
                'strategy': strategy,
                'surveillance_level': surveillance_level,
                'liberty_threshold': liberty_threshold,
                'poison_rate': poison_rate,
                'noise_level': noise_level,
                'n_samples': n_samples,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'hidden_layers': hidden_layers,
                'dropout_rate': dropout_rate,
                'activation': activation
            }

            return parameters

    def render_simulation_controls(self, params: Dict[str, Any]):
        """Render simulation control buttons."""
        st.markdown('<div class="section-header"> Simulation Controls</div>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button(" Run Simulation", type="primary", use_container_width=True):
                self.run_simulation(params)

        with col2:
            if st.button(" Quick Run", type="secondary", use_container_width=True):
                self.run_quick_simulation()

        with col3:
            if st.button(" Compare", type="secondary", use_container_width=True):
                self.add_to_comparison()

        with col4:
            if st.button(" Clear All", type="secondary", use_container_width=True):
                self.clear_simulations()

    def run_simulation(self, parameters: Dict[str, Any]):
        try:
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0)

            with st.spinner("🚀 Initializing Simulation Engine..."):
                # Update progress visually
                for i in range(1, 40, 5):
                    progress_bar.progress(i / 100)
                    time.sleep(0.05)

                # Use a more robust way to run the async task in Streamlit
                import asyncio

                # Helper to run the async orchestrator
                async def execute():
                    return await self.orchestrator.run_simulation(
                        strategy=parameters['strategy'],
                        parameters=parameters
                    )

                # Create a new event loop for this thread to avoid conflicts with Streamlit
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(execute())
                finally:
                    new_loop.close()

                progress_bar.progress(1.0)

            if result:
                st.session_state.current_result = result
                # Convert to dict for history (JSON serializable)
                st.session_state.simulation_history.append(result.to_dict())
                st.success("✅ Simulation Completed!")
                time.sleep(1)  # Give user a moment to see success
                progress_placeholder.empty()
                st.rerun()

        except Exception as e:
            st.error(f"❌ Simulation Engine Critical Failure: {str(e)}")
            # Log the full error for debugging
            import traceback
            print(traceback.format_exc())
    def run_quick_simulation(self):
        """Run a quick simulation with default parameters."""
        # Implement quick simulation logic
        st.info("Quick simulation feature coming soon!")

    def add_to_comparison(self):
        """Add current result to comparison list."""
        if st.session_state.current_result:
            st.session_state.comparison_results.append(
                st.session_state.current_result
            )
            st.success("Added to comparison list!")
        else:
            st.warning("Run a simulation first!")

    def clear_simulations(self):
        """Clear all simulation data."""
        if st.button("Confirm Clear All", type="primary"):
            st.session_state.simulation_history = []
            st.session_state.current_result = None
            st.session_state.comparison_results = []
            self.orchestrator.clear_cache()
            st.success("All simulations cleared!")
            st.rerun()

    def render_results(self):
        """Render simulation results."""

        if not st.session_state.current_result:
            st.info(" Run a simulation to see results here!")
            return

        result = st.session_state.current_result
        if result.status.value == "failed":
            st.error("### 🛑 Simulation Engine Error")
            st.warning(f"Error Detail: {result.metrics.get('error', 'No error message captured')}")
            st.info("Check your terminal console for the full traceback.")
            return  # Stop rendering the rest if it failed
        # Main Metrics Cards
        st.markdown('<div class="section-header"> Performance Metrics</div>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            create_metric_cards(result, 'safety_score', 'Safety Score', '🛡️')

        with col2:
            create_metric_cards(result, 'liberty_score', 'Liberty Score', '🗽')

        with col3:
            create_metric_cards(result, 'resilience_score', 'Resilience Index', '💪')

        with col4:
            create_metric_cards(result, 'fairness_score', 'Fairness Score', '⚖️')

        # Threat Level Indicator
        threat_level = result.threat_level
        st.markdown(f"""
            <div style="background: {'#FFEBEE' if threat_level == ThreatLevel.CRITICAL else
        '#FFF3E0' if threat_level == ThreatLevel.HIGH else
        '#FFF8E1' if threat_level == ThreatLevel.MEDIUM else
        '#E8F5E9'};
                        border-left: 5px solid {threat_level.icon};
                        padding: 20px;
                        border-radius: 10px;
                        margin: 20px 0;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span style="font-size: 2rem;">{threat_level.icon}</span>
                    <div>
                        <h3 style="margin:0; color: #333;">Threat Level: {threat_level.name}</h3>
                        <p style="margin:5px 0 0 0; color: #666;">{threat_level.description}</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Detailed Analysis Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Performance Analysis",
            "Threat Detection",
            "Model Insights",
            "Recommendations"
        ])

        with tab1:
            self._render_performance_analysis(result)

        with tab2:
            self._render_threat_detection(result)

        with tab3:
            self._render_model_insights(result)

        with tab4:
            self._render_recommendations(result)

        # Export Section
        st.markdown("---")
        create_export_buttons(result)

    def _render_performance_analysis(self, result: SimulationResult):
        """Render performance analysis visualizations."""
        col1, col2 = st.columns(2)

        with col1:
            # Radar chart
            fig = create_performance_radar(result)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Training history
            if result.training_history:
                fig = create_trend_chart(result.training_history)
                st.plotly_chart(fig, use_container_width=True)

        # Confusion matrix
        if 'confusion_matrix' in result.metrics:
            fig = create_confusion_matrix(result.metrics['confusion_matrix'])
            st.plotly_chart(fig, use_container_width=True)

    def _render_threat_detection(self, result: SimulationResult):
        """Render threat detection analysis."""
        st.markdown("###  Detected Threats")

        if result.threats_detected:
            for threat in result.threats_detected:
                st.warning(f"• {threat}")
        else:
            st.success("No significant threats detected")

        # Threat timeline (if available)
        if len(st.session_state.simulation_history) > 1:
            st.markdown("### Threat Trend")
            fig = self._create_threat_trend_chart()
            st.plotly_chart(fig, use_container_width=True)

    def _create_threat_trend_chart(self):
        """Create threat trend chart from history."""
        history = st.session_state.simulation_history[-10:]  # Last 10 runs

        if len(history) < 2:
            return go.Figure()

        timestamps = [h.get('timestamp', i) for i, h in enumerate(history)]
        safety_scores = [h.get('safety_score', 0) for h in history]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=safety_scores,
            mode='lines+markers',
            name='Safety Score',
            line=dict(color=self.color_scheme['primary'], width=3)
        ))

        # Add threshold lines
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color=self.color_scheme['warning'],
            annotation_text="Warning Threshold"
        )

        fig.add_hline(
            y=60,
            line_dash="dash",
            line_color=self.color_scheme['danger'],
            annotation_text="Critical Threshold"
        )

        fig.update_layout(
            title="Safety Score Trend",
            xaxis_title="Simulation",
            yaxis_title="Safety Score",
            height=400,
            showlegend=True
        )

        return fig

    def _render_model_insights(self, result: SimulationResult):
        """Render model insights and explainability."""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("###  Model Architecture")
            st.json(result.model_architecture, expanded=False)

            st.markdown("### Performance")
            performance_data = {
                "Inference Latency": f"{result.inference_latency:.2f} ms",
                "Memory Usage": f"{result.memory_usage:.2f} MB",
                "Model Parameters": f"{result.model_parameters:,}",
                "Training Duration": f"{result.duration_ms:.0f} ms"
            }

            for key, value in performance_data.items():
                st.metric(key, value)

        with col2:
            st.markdown("###  Feature Importance")
            if result.feature_importance:
                fig = create_feature_importance(result.feature_importance)
                st.plotly_chart(fig, use_container_width=True)

            # Decision boundary visualization
            if result.decision_boundary is not None:
                st.markdown("###  Decision Boundary")
                # Would implement decision boundary plot here

    def _render_recommendations(self, result: SimulationResult):
        """Render AI-powered recommendations."""
        st.markdown("### Actionable Recommendations")

        recommendations = []

        # Safety-based recommendations
        if result.safety_score < 70:
            recommendations.append({
                "priority": "HIGH",
                "title": "Increase Threat Detection Capabilities",
                "description": "Safety score below threshold. Consider enhancing anomaly detection algorithms.",
                "actions": [
                    "Increase training data diversity",
                    "Implement ensemble methods",
                    "Add adversarial training"
                ]
            })

        # Liberty-based recommendations
        if result.liberty_score < 60:
            recommendations.append({
                "priority": "MEDIUM",
                "title": "Strengthen Privacy Protection",
                "description": "Liberty score indicates potential privacy concerns.",
                "actions": [
                    "Implement differential privacy",
                    "Add privacy-preserving techniques",
                    "Review data retention policies"
                ]
            })

        # Resilience-based recommendations
        if result.resilience_score < 65:
            recommendations.append({
                "priority": "HIGH",
                "title": "Improve System Resilience",
                "description": "System may be vulnerable to adversarial attacks.",
                "actions": [
                    "Add robustness training",
                    "Implement failover mechanisms",
                    "Conduct penetration testing"
                ]
            })

        # Fairness-based recommendations
        if result.fairness_score < 75:
            recommendations.append({
                "priority": "MEDIUM",
                "title": "Address Fairness Concerns",
                "description": "Potential bias detected in model decisions.",
                "actions": [
                    "Audit training data for bias",
                    "Implement fairness constraints",
                    "Add explainability features"
                ]
            })

        # Display recommendations
        if not recommendations:
            st.success(" Current configuration appears optimal!")
            return

        # Sort by priority
        priority_order = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recommendations.sort(key=lambda x: priority_order[x["priority"]])

        for rec in recommendations:
            with st.container():
                st.markdown(f"""
                    <div style="background: {'#FFEBEE' if rec['priority'] == 'HIGH' else
                '#FFF3E0' if rec['priority'] == 'MEDIUM' else
                '#E8F5E9'};
                                border-left: 5px solid {'#F44336' if rec['priority'] == 'HIGH' else
                '#FF9800' if rec['priority'] == 'MEDIUM' else
                '#4CAF50'};
                                padding: 20px;
                                border-radius: 10px;
                                margin: 10px 0;">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <h4 style="margin:0; color: #333;">{rec['title']}</h4>
                            <span style="background: {'#F44336' if rec['priority'] == 'HIGH' else
                '#FF9800' if rec['priority'] == 'MEDIUM' else
                '#4CAF50'};
                                      color: white;
                                      padding: 4px 12px;
                                      border-radius: 12px;
                                      font-size: 0.8rem;
                                      font-weight: 600;">
                                {rec['priority']} PRIORITY
                            </span>
                        </div>
                        <p style="margin:10px 0 15px 0; color: #666;">{rec['description']}</p>
                        <div>
                            <strong style="color: #333;">Recommended Actions:</strong>
                            <ul style="margin:10px 0 0 0; padding-left: 20px;">
                                {''.join(f'<li style="margin:5px 0;">{action}</li>' for action in rec['actions'])}
                            </ul>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    def render_comparison_view(self):
        """Render comparison between multiple simulations."""
        if len(st.session_state.comparison_results) < 2:
            st.info("Add at least 2 simulations to compare!")
            return

        st.markdown('<div class="section-header"> Comparative Analysis</div>',
                    unsafe_allow_html=True)

        # Create comparison DataFrame
        comparison_data = []
        for i, result in enumerate(st.session_state.comparison_results):
            comparison_data.append({
                "Simulation": f"Run {i + 1}",
                "Safety Score": result.safety_score,
                "Liberty Score": result.liberty_score,
                "Resilience Score": result.resilience_score,
                "Fairness Score": result.fairness_score,
                "Threat Level": result.threat_level.name,
                "Duration (ms)": result.duration_ms,
                "Model Params": result.model_parameters
            })

        df = pd.DataFrame(comparison_data)

        # Display comparison table
        st.dataframe(
            df.style.format({
                "Safety Score": "{:.1f}",
                "Liberty Score": "{:.1f}",
                "Resilience Score": "{:.1f}",
                "Fairness Score": "{:.1f}",
                "Duration (ms)": "{:.0f}",
                "Model Params": "{:,}"
            }),
            use_container_width=True
        )

        # Comparison chart
        fig = go.Figure()

        metrics = ["Safety Score", "Liberty Score", "Resilience Score", "Fairness Score"]
        for i, result in enumerate(st.session_state.comparison_results):
            values = [
                result.safety_score,
                result.liberty_score,
                result.resilience_score,
                result.fairness_score
            ]

            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics + [metrics[0]],
                name=f"Run {i + 1}",
                fill='toself',
                opacity=0.7
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            height=500,
            title="Comparative Performance Analysis"
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_export_section(self):
        """Render export options."""
        st.markdown('<div class="section-header"> Export & Share</div>',
                    unsafe_allow_html=True)

        if st.session_state.current_result:
            create_export_buttons(st.session_state.current_result)
        else:
            st.info("Run a simulation to enable export options")

    def run(self):
        """Main method to run the dashboard."""
        # Render header
        self.render_header()

        # Get parameters from sidebar (but don't run yet)

        params = self.render_sidebar()

        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs([
            "Simulation",
            "Results",
            "Comparison",
            "Analytics"
        ])

        with tab1:
            self.render_simulation_controls(params)

            # Quick stats
            st.markdown("###  Recent Simulations")
            if st.session_state.simulation_history:
                recent_df = pd.DataFrame(st.session_state.simulation_history[-5:])
                st.dataframe(
                    recent_df[['simulation_id', 'safety_score', 'liberty_score',
                               'duration_ms', 'status']].tail(),
                    use_container_width=True
                )
            else:
                st.info("No simulations yet. Run one to get started!")

        with tab2:
            self.render_results()

        with tab3:
            self.render_comparison_view()

        with tab4:
            self.render_analytics_dashboard()

        # Footer
        st.markdown("---")
        self.render_footer()

    def render_analytics_dashboard(self):
        """Render analytics dashboard with historical data."""
        if not st.session_state.simulation_history:
            st.info("Run simulations to see analytics")
            return

        st.markdown("### Historical Analytics")

        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.simulation_history)

        # Time series analysis
        if 'timestamp' in history_df.columns:
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values('timestamp')

            col1, col2 = st.columns(2)

            with col1:
                # Safety trend
                fig = px.line(
                    history_df,
                    x='timestamp',
                    y='safety_score',
                    title='Safety Score Trend',
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Liberty trend
                fig = px.line(
                    history_df,
                    x='timestamp',
                    y='liberty_score',
                    title='Liberty Score Trend',
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)

        # Distribution analysis
        st.markdown("### Performance Distributions")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                history_df,
                x='safety_score',
                title='Safety Score Distribution',
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                history_df,
                y=['safety_score', 'liberty_score'],
                title='Score Distributions'
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_footer(self):
        """Render dashboard footer."""
        st.markdown(
            f"""
                    <div style="text-align: center; color: #666; padding: 20px;">
                        <p style="margin:0; font-size: 0.9rem;">
                             <strong>GAGS Security Resilience Module</strong> | 
                            Version {settings.APP_VERSION} | 
                            Environment: {settings.ENVIRONMENT}
                        </p>
                        <p style="margin:5px 0 0 0; font-size: 0.8rem; opacity: 0.7;">
                            © {datetime.now().year} Global AI Governance Sandbox | 
                            <a href="https://example.com/privacy" style="color: #666;">Privacy</a> | 
                            <a href="https://example.com/terms" style="color: #666;">Terms</a> | 
                            <a href="https://example.com/docs" style="color: #666;">Documentation</a>
                        </p>
                    </div>
                    """,
            unsafe_allow_html=True
        )


# Main execution
if __name__ == "__main__":
    dashboard = SecurityDashboard()
    dashboard.run()