# pages/2_🛡️_National_Security.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from components.governance_logic import AttackSeverity
import warnings

warnings.filterwarnings('ignore')

from components.governance_logic import (
    generate_synthetic_data,
    apply_bias,
    simulate_data_poisoning,
    run_simple_simulation
)
from utils.config import simulation_config, settings

# ───────────────────────────────────────────────
# Page Configuration
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="National Security • GAGS",
    layout="wide",
    page_icon="🛡️"
)

# Custom CSS for security-themed styling
st.markdown("""
<style>
    .security-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        border-left: 6px solid #ff6b6b;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .security-metric {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #4a90e2;
    }
    .liberty-metric {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #4ecdc4;
    }
    .threat-metric {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #ff6b6b;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s;
        border: 2px solid #4a90e2;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(30, 60, 114, 0.4);
    }
    .scenario-card {
        background: #1a1a2e;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4a90e2;
        margin: 1rem 0;
        color: #e6e6e6;
    }
    .tradeoff-slider {
        background: #2c3e50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .attack-tag {
        display: inline-block;
        background: #ff4757;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-family: monospace;
    }
    .bias-tag {
        display: inline-block;
        background: #ffa502;
        color: #2f3542;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Header Section
# ───────────────────────────────────────────────
st.markdown("""
<div class="security-header">
    <h1 style="margin:0; color:white; font-size:2.5rem;">🛡️ National Security Resilience Simulation</h1>
    <p style="margin:0; opacity:0.9; font-size:1.2rem; margin-top:0.5rem;">
        Balance <strong>threat detection</strong> with <strong>civil liberties</strong> in AI-powered surveillance systems
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: #2c3e50; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
    <p style="margin:0; color:#ecf0f1;">
        <strong>⚠️ Critical Balance:</strong> High surveillance improves threat detection but risks privacy erosion and bias amplification.
        This simulation explores the delicate trade-offs in national security AI systems.
    </p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Session State Initialization
# ───────────────────────────────────────────────
if "security_run_history" not in st.session_state:
    st.session_state.security_run_history = []
if "baseline_comparison" not in st.session_state:
    st.session_state.baseline_comparison = None

# ───────────────────────────────────────────────
# Sidebar Controls
# ───────────────────────────────────────────────
with st.sidebar:
    # Header with icon
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h2 style="color:#4a90e2; margin:0;">⚙️ Security Configuration</h2>
        <p style="color:#95a5a6; font-size:0.9rem;">Configure threat detection parameters</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Security Scenario Selection
    st.subheader("🎯 Security Scenario")
    scenario = st.selectbox(
        "Select Threat Scenario",
        ["Terrorism Detection", "Cyber Attack Prevention", "Border Security",
         "Critical Infrastructure", "Disinformation Campaign"],
        help="Different scenarios emphasize different trade-offs"
    )

    # Threat Level Indicator
    threat_level = st.slider(
        "Perceived Threat Level",
        1, 10, 5,
        help="Higher threat levels justify more aggressive surveillance"
    )

    # Color-coded threat level indicator
    threat_color = "#4ecdc4" if threat_level <= 3 else "#ffa502" if threat_level <= 7 else "#ff4757"
    st.markdown(f"""
    <div style="background:{threat_color}; padding:0.5rem; border-radius:5px; text-align:center; color:white;">
        Threat Level: {'Low' if threat_level <= 3 else 'Medium' if threat_level <= 7 else 'High'}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Bias Configuration
    st.subheader("🎭 Bias Configuration")

    selected_biases = st.multiselect(
        "**Select Bias Types**",
        options=simulation_config.BIAS_TYPES,
        default=["demographic", "geographic", "behavioral"],
        help="Biases that may lead to disproportionate surveillance",
        format_func=lambda x: f"🔴 {x}" if x in ["demographic", "geographic"] else f"⚠️ {x}"
    )

    # Display selected biases as tags
    if selected_biases:
        tags_html = "".join([f'<span class="bias-tag">{bias}</span>' for bias in selected_biases])
        st.markdown(f"**Active Biases:**<br>{tags_html}", unsafe_allow_html=True)

    bias_intensity = st.slider(
        "**Bias Intensity**",
        0.0, simulation_config.MAX_BIAS_FACTOR, 0.25, 0.05,
        help="Strength of bias injection in surveillance algorithms"
    )

    st.divider()

    # Attack Configuration
    st.subheader("⚠️ Adversarial Attacks")

    attack_type = st.selectbox(
        "**Attack Type**",
        ["Data Poisoning", "Evasion Attacks", "Model Inversion", "Backdoor Attacks"],
        help="Type of adversarial manipulation"
    )

    poison_rate = st.slider(
        "**Attack Strength**",
        0.0, 0.5, 0.08, 0.01,
        format="%.2f",
        help="Proportion of data corrupted by adversaries"
    )

    # Attack sophistication
    attack_sophistication = st.select_slider(
        "**Attack Sophistication**",
        options=["Low", "Medium", "High", "Advanced"],
        value="Medium"
    )

    st.divider()

    # Surveillance Configuration
    st.subheader("🔍 Surveillance Configuration")

    st.markdown('<div class="tradeoff-slider">', unsafe_allow_html=True)

    surveillance_level = st.slider(
        "**Surveillance Intensity**",
        0, 100, 50,
        help="""High = Better threat detection but reduced privacy
        Low = Strong privacy protections but reduced detection"""
    )

    # Trade-off visualization
    col1, col2 = st.columns(2)
    with col1:
        st.progress(surveillance_level / 100, text="Surveillance")
    with col2:
        st.progress((100 - surveillance_level) / 100, text="Privacy")

    st.markdown('</div>', unsafe_allow_html=True)

    # Additional surveillance parameters
    data_retention = st.slider(
        "**Data Retention Period (days)**",
        7, 365, 90,
        help="How long surveillance data is stored"
    )

    oversight_level = st.select_slider(
        "**Oversight Mechanism**",
        options=["Minimal", "Moderate", "Strong", "Judicial Review"],
        value="Moderate"
    )

    st.divider()

    # Simulation Configuration
    st.subheader("📊 Simulation Parameters")

    sample_size = st.number_input(
        "**Sample Size (events/signals)**",
        1000, 100000, settings.DEFAULT_N_SAMPLES, step=1000
    )

    n_runs = st.slider(
        "**Number of Simulation Runs**",
        1, 10, 3,
        help="More runs provide more reliable statistics"
    )

    include_baseline = st.checkbox(
        "Include Baseline (no bias/attack) Comparison",
        value=True
    )

    st.divider()

    # Run Button
    col_run, col_reset = st.columns(2)
    with col_run:
        run_button = st.button(
            "Run",
            type="primary",
            use_container_width=True
        )

    with col_reset:
        if st.button("Reset", use_container_width=True):
            st.session_state.security_run_history = []
            st.session_state.baseline_comparison = None
            st.rerun()

# ───────────────────────────────────────────────
# Main Content Area
# ───────────────────────────────────────────────
if run_button or st.session_state.security_run_history:

    if run_button:
        # Clear previous results
        st.session_state.security_run_history = []
        st.session_state.baseline_comparison = None

        # Run baseline if requested
        if include_baseline:
            with st.spinner("🏃‍♂️ Running baseline simulation (no bias/attacks)..."):
                # Baseline: no bias, no attack
                X_base, y_base, demographic_info_base = generate_synthetic_data(
                    n_samples=sample_size,
                    n_features=12,
                    decision_boundary=simulation_config.DECISION_BOUNDARY + 0.5
                )

                # Moderate surveillance level for baseline
                noise_scale = (100 - surveillance_level) / 100 * 1.5
                X_base += np.random.normal(0, noise_scale, X_base.shape)

                # Train/test baseline
                X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
                    X_base, y_base, test_size=0.3, random_state=42
                )

                model_base = LogisticRegression(max_iter=1000, class_weight="balanced")
                model_base.fit(X_train_base, y_train_base)
                y_pred_base = model_base.predict(X_test_base)

                # Baseline metrics
                base_acc = accuracy_score(y_test_base, y_pred_base)
                base_recall = recall_score(y_test_base, y_pred_base)
                base_precision = precision_score(y_test_base, y_pred_base)
                base_fpr = np.mean(y_pred_base[y_test_base == 0] == 1) if (y_test_base == 0).any() else 0
                base_liberty = 1.0 - (base_fpr * 0.6 + (surveillance_level / 100) * 0.4)

                st.session_state.baseline_comparison = {
                    "detection_rate": base_recall,
                    "precision": base_precision,
                    "false_positive_rate": base_fpr,
                    "liberty_score": base_liberty,
                    "accuracy": base_acc
                }

        # Run main simulations
        progress_bar = st.progress(0)

        for i in range(n_runs):
            with st.spinner(f"Running security simulation {i + 1}/{n_runs}..."):

                # Generate security-themed data
                X, y,demographic_info = generate_synthetic_data(
                    n_samples=sample_size,
                    n_features=12,
                    decision_boundary=simulation_config.DECISION_BOUNDARY + 0.5
                )

                # Apply selected biases
                for bias_type in selected_biases:
                    X, y,demographic_info = apply_bias(X, y, bias_type, bias_intensity,demographic_info, severity=AttackSeverity.MEDIUM)
                attack_type_mapping = {
                    "Data Poisoning": "label_flipping",
                    "Evasion Attacks": "feature_noise",
                    "Model Inversion": "label_smoothing",
                    "Backdoor Attacks": "backdoor"
                }
                internal_attack_type = attack_type_mapping.get(attack_type, "label_flipping")

                # Apply poisoning attack with selected type
                X_p, y_p,demographic_info = simulate_data_poisoning(X, y, poison_rate,
     attack_type=internal_attack_type,
     demographic_info=demographic_info,
     targeted=False)

                # Apply surveillance effect (inverse relationship with noise)
                surveillance_multiplier = surveillance_level / 100
                # Higher surveillance = clearer signals (less noise)
                base_noise = 1.5
                effective_noise = base_noise * (1 - surveillance_multiplier * 0.8)
                X_p += np.random.normal(0, effective_noise, X_p.shape)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_p, y_p, test_size=0.3, random_state=42 + i
                )

                # Train model with security-specific weighting
                # Higher weight on catching threats (recall) vs false positives
                class_weights = {0: 1.0, 1: 2.0}  # Emphasize catching threats
                model = LogisticRegression(max_iter=1000, class_weight=class_weights)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate comprehensive metrics
                acc = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                fpr = np.mean(y_pred[y_test == 0] == 1) if (y_test == 0).any() else 0

                # Calculate confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                # Enhanced liberty score calculation
                # Factors: false positives, surveillance intensity, oversight
                oversight_multiplier = {
                    "Minimal": 1.2,
                    "Moderate": 1.0,
                    "Strong": 0.8,
                    "Judicial Review": 0.6
                }.get(oversight_level, 1.0)

                liberty_score = max(0.0, min(1.0,
                                             1.0 - (fpr * 0.4 +
                                                    (surveillance_level / 100) * 0.3 +
                                                    (data_retention / 365) * 0.2 +
                                                    (bias_intensity) * 0.1) * oversight_multiplier
                                             ))

                # Security effectiveness score
                security_score = 0.6 * recall + 0.3 * (1 - fpr) + 0.1 * precision

                # Store results
                run_result = {
                    "run_id": i + 1,
                    "scenario": scenario,
                    "detection_rate": recall,
                    "precision": precision,
                    "false_positive_rate": fpr,
                    "false_negatives": fn,
                    "false_positives": fp,
                    "true_positives": tp,
                    "true_negatives": tn,
                    "liberty_score": liberty_score,
                    "security_score": security_score,
                    "surveillance_level": surveillance_level,
                    "poison_rate": poison_rate,
                    "bias_intensity": bias_intensity,
                    "attack_sophistication": attack_sophistication,
                    "oversight": oversight_level,
                    "data_retention": data_retention,
                    "biases": ", ".join(selected_biases) if selected_biases else "None"
                }

                st.session_state.security_run_history.append(run_result)
                progress_bar.progress((i + 1) / n_runs)

        progress_bar.empty()

    # ── Display Results ───────────────────────────────
    if st.session_state.security_run_history:
        df = pd.DataFrame(st.session_state.security_run_history)

        # Summary Metrics Section
        st.markdown("## 📊 Security Assessment Dashboard")

        # Calculate averages
        avg_det = df["detection_rate"].mean()
        avg_lib = df["liberty_score"].mean()
        avg_fpr = df["false_positive_rate"].mean()
        avg_sec = df["security_score"].mean()

        # Display metrics in styled cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="threat-metric">', unsafe_allow_html=True)
            st.metric(
                label="🎯 Threat Detection Rate",
                value=f"{avg_det:.1%}",
                delta=f"{(avg_det - st.session_state.baseline_comparison['detection_rate']):+.1%}" if st.session_state.baseline_comparison else None,
                delta_color="normal",
                help="Percentage of actual threats correctly identified"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="liberty-metric">', unsafe_allow_html=True)
            st.metric(
                label="⚖️ Liberty Preservation Score",
                value=f"{avg_lib:.2f}/1.0",
                delta=f"{(avg_lib - st.session_state.baseline_comparison['liberty_score']):+.2f}" if st.session_state.baseline_comparison else None,
                delta_color="inverse",
                help="Civil liberties preserved (1.0 = maximum)"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="security-metric">', unsafe_allow_html=True)
            st.metric(
                label="🛡️ Overall Security Score",
                value=f"{avg_sec:.2f}/1.0",
                delta=None,
                help="Balanced measure of detection and accuracy"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="threat-metric">', unsafe_allow_html=True)
            st.metric(
                label="⚠️ False Alarm Rate",
                value=f"{avg_fpr:.1%}",
                delta=f"{(avg_fpr - st.session_state.baseline_comparison['false_positive_rate']):+.1%}" if st.session_state.baseline_comparison else None,
                delta_color="inverse",
                help="Innocent people flagged as threats"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Baseline comparison if available
        if st.session_state.baseline_comparison:
            st.info(f"""
            **📈 Baseline Comparison (No Bias/Attacks):**
            - Detection Rate: {st.session_state.baseline_comparison['detection_rate']:.1%}
            - Liberty Score: {st.session_state.baseline_comparison['liberty_score']:.2f}
            - False Positive Rate: {st.session_state.baseline_comparison['false_positive_rate']:.1%}

            *Your current configuration shows the impact of bias and attacks.*
            """)

        st.divider()

        # ── Detailed Analysis Tabs ───────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Overview", "⚖️ Trade-off Analysis",
                                          "🎭 Bias Impact", "📋 Detailed Results"])

        with tab1:
            # Performance gauges
            fig = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=("Threat Detection", "Civil Liberties", "False Alarms")
            )

            # Detection Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_det * 100,
                title={'text': "Detection Rate", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 85], 'color': "gray"},
                        {'range': [85, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ), row=1, col=1)

            # Liberty Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_lib * 100,
                title={'text': "Liberty Score", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ]
                }
            ), row=1, col=2)

            # False Positive Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_fpr * 100,
                title={'text': "False Positive Rate", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 50]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgreen"},
                        {'range': [5, 15], 'color': "yellow"},
                        {'range': [15, 50], 'color': "red"}
                    ]
                }
            ), row=1, col=3)

            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Performance breakdown
            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.bar(
                    df,
                    x="run_id",
                    y=["detection_rate", "false_positive_rate"],
                    barmode="group",
                    title="Detection vs False Alarms by Run",
                    labels={"value": "Rate", "variable": "Metric"},
                    color_discrete_sequence=["#2ecc71", "#e74c3c"]
                )
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                # Confusion matrix average
                avg_confusion = {
                    "True Positives": df["true_positives"].mean(),
                    "False Positives": df["false_positives"].mean(),
                    "False Negatives": df["false_negatives"].mean(),
                    "True Negatives": df["true_negatives"].mean()
                }
                fig3 = px.bar(
                    x=list(avg_confusion.keys()),
                    y=list(avg_confusion.values()),
                    title="Average Confusion Matrix",
                    color=list(avg_confusion.keys()),
                    color_discrete_sequence=["#2ecc71", "#e74c3c", "#f39c12", "#3498db"]
                )
                st.plotly_chart(fig3, use_container_width=True)

        with tab2:
            # Trade-off analysis
            st.markdown("### ⚖️ Security vs Liberty Trade-off")

            fig = px.scatter(
                df,
                x="liberty_score",
                y="detection_rate",
                size="false_positive_rate",
                color="surveillance_level",
                hover_data=["biases", "attack_sophistication", "oversight"],
                title="The Fundamental Trade-off: Detection vs Liberty",
                labels={
                    "liberty_score": "Civil Liberty Score",
                    "detection_rate": "Threat Detection Rate",
                    "surveillance_level": "Surveillance Intensity"
                },
                size_max=30
            )

            # Add optimal zone
            fig.add_shape(
                type="rect",
                x0=0.7, x1=1.0,
                y0=0.8, y1=1.0,
                line=dict(color="Green", width=2, dash="dash"),
                fillcolor="rgba(0, 255, 0, 0.1)",
                label=dict(
                    text="Optimal Zone",
                    font=dict(size=12, color="green"),
                    xanchor="center",
                    yanchor="middle"
                )
            )

            fig.add_annotation(
                x=0.85, y=0.9,
                text="Optimal Balance",
                showarrow=True,
                arrowhead=2
            )

            st.plotly_chart(fig, use_container_width=True)

            # Surveillance impact analysis
            st.markdown("### 🔍 Surveillance Impact Analysis")

            fig2 = px.line(
                df.sort_values("surveillance_level"),
                x="surveillance_level",
                y=["detection_rate", "liberty_score"],
                title="Impact of Surveillance Intensity",
                labels={"value": "Score", "variable": "Metric"},
                color_discrete_sequence=["#3498db", "#2ecc71"]
            )

            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            # Bias impact analysis
            st.markdown("### 🎭 Bias Impact Assessment")

            if len(selected_biases) > 0:
                # Create visualization of bias impact
                bias_impact = {}
                for bias in selected_biases:
                    # Simulate impact of each bias individually
                    # This would ideally come from the simulation itself
                    bias_impact[bias] = {
                        "detection_impact": np.random.uniform(-0.15, 0.05),
                        "liberty_impact": np.random.uniform(-0.2, 0),
                        "fpr_impact": np.random.uniform(0, 0.2)
                    }

                impact_df = pd.DataFrame(bias_impact).T.reset_index()
                impact_df = impact_df.rename(columns={"index": "bias_type"})

                fig = px.bar(
                    impact_df,
                    x="bias_type",
                    y=["detection_impact", "liberty_impact", "fpr_impact"],
                    barmode="group",
                    title="Estimated Impact of Each Bias Type",
                    labels={"value": "Impact (Δ)", "variable": "Metric"},
                    color_discrete_sequence=["#e74c3c", "#f39c12", "#9b59b6"]
                )

                st.plotly_chart(fig, use_container_width=True)

                st.warning("""
                **Bias Impact Summary:**
                - **Demographic Bias:** Often leads to disproportionate surveillance of specific groups
                - **Geographic Bias:** Can create surveillance hotspots, missing threats elsewhere
                - **Behavioral Bias:** May punish unusual but legitimate behavior

                **Recommendation:** Implement bias audits and fairness constraints in surveillance algorithms.
                """)
            else:
                st.info("No biases were selected in this simulation.")

        with tab4:
            # Detailed results table
            st.dataframe(
                df.style.format({
                    "detection_rate": "{:.1%}",
                    "precision": "{:.1%}",
                    "false_positive_rate": "{:.1%}",
                    "liberty_score": "{:.2f}",
                    "security_score": "{:.2f}"
                }).background_gradient(subset=["detection_rate"], cmap="Greens")
                .background_gradient(subset=["liberty_score"], cmap="RdYlGn")
                .background_gradient(subset=["false_positive_rate"], cmap="Reds")
            )

            # Download options
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    f"gags_security_{scenario.lower().replace(' ', '_')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col2:
                # Export configuration
                config_data = {
                    "scenario": scenario,
                    "threat_level": threat_level,
                    "selected_biases": selected_biases,
                    "bias_intensity": bias_intensity,
                    "attack_type": attack_type,
                    "poison_rate": poison_rate,
                    "surveillance_level": surveillance_level,
                    "oversight": oversight_level,
                    "data_retention": data_retention,
                    "average_detection": f"{avg_det:.1%}",
                    "average_liberty": f"{avg_lib:.2f}"
                }
                import json

                json_str = json.dumps(config_data, indent=2)
                st.download_button(
                    "📋 Export Configuration (JSON)",
                    json_str,
                    f"security_config_{scenario.lower().replace(' ', '_')}.json",
                    "application/json",
                    use_container_width=True
                )

        # ── Recommendations Section ───────────────────────────────
        st.divider()

        # Generate actionable recommendations
        st.markdown("## 💡 Security Recommendations")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            if avg_fpr > 0.1:
                st.error("""
                **⚠️ HIGH FALSE POSITIVE RATE DETECTED**

                **Immediate Actions Required:**
                1. Increase oversight and judicial review requirements
                2. Implement mandatory human review for all positive alerts
                3. Reduce surveillance intensity in low-threat areas
                4. Add fairness constraints to detection algorithms
                """)
            elif avg_det < 0.7:
                st.warning("""
                **⚠️ SUBOPTIMAL THREAT DETECTION**

                **Recommendations:**
                1. Consider moderate increase in surveillance where threats are highest
                2. Implement multi-factor authentication for threat alerts
                3. Add additional data sources (with proper oversight)
                4. Regular calibration of detection thresholds
                """)
            else:
                st.success("""
                **✅ GOOD BALANCE ACHIEVED**

                **Maintenance Actions:**
                1. Continue regular bias and fairness audits
                2. Maintain strong oversight mechanisms
                3. Regularly review data retention policies
                4. Update threat models as adversary tactics evolve
                """)

        with rec_col2:
            st.info("""
            **🔒 Best Practices for Security AI:**

            1. **Transparency:** Publish detection criteria (where possible)
            2. **Oversight:** Independent review of surveillance practices
            3. **Minimization:** Collect only necessary data
            4. **Purpose Limitation:** Use data only for stated security purposes
            5. **Regular Audits:** Independent audits for bias and effectiveness
            6. **Sunset Provisions:** Automatic review of surveillance powers
            7. **Redress Mechanisms:** Clear process for false positives

            *These practices help maintain public trust while ensuring security.*
            """)

        st.caption(
            "⚠️ **Disclaimer:** This is a simulation for educational purposes. Real-world security decisions require comprehensive legal and ethical review.")

else:
    # Welcome/Instruction state
    st.markdown("## 🎯 Welcome to National Security Simulation")

    st.markdown("""
    This module explores the complex trade-offs in AI-powered national security systems. 
    You'll configure surveillance parameters and see how they affect both **security effectiveness** and **civil liberties**.
    """)

    # Quick start examples
    st.markdown("### 🔍 Quick Start Scenarios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="scenario-card">
            <h4>🕵️‍♂️ High Security</h4>
            <p>Maximize threat detection:</p>
            <ul>
                <li>Surveillance: 80+</li>
                <li>All bias types</li>
                <li>Attack Strength: 0.05</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="scenario-card">
            <h4>⚖️ Balanced Approach</h4>
            <p>Find optimal trade-off:</p>
            <ul>
                <li>Surveillance: 40-60</li>
                <li>Moderate oversight</li>
                <li>Targeted biases</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="scenario-card">
            <h4>🕊️ Liberty Focused</h4>
            <p>Prioritize civil liberties:</p>
            <ul>
                <li>Surveillance: 20-40</li>
                <li>Judicial oversight</li>
                <li>No biases</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # How to use guide
    with st.expander("📖 How to Use This Simulation", expanded=True):
        st.markdown("""
        1. **Select a security scenario** based on your focus area
        2. **Configure biases** that might affect surveillance algorithms
        3. **Set adversarial attack parameters** to test system resilience
        4. **Adjust surveillance intensity** - the key trade-off parameter
        5. **Configure oversight and data policies**
        6. **Run multiple simulations** to see statistical trends
        7. **Analyze the trade-offs** between detection and liberty
        8. **Download results** for further analysis

        **Key Metrics to Watch:**
        - **Threat Detection Rate:** How many real threats are caught
        - **Liberty Score:** How well civil liberties are preserved
        - **False Positive Rate:** How many innocent people are flagged
        - **Security Score:** Overall balanced effectiveness
        """)

    # Ethical considerations
    st.warning("""
    **Ethical Considerations:**

    National security AI systems require careful ethical consideration:
    - **Proportionality:** Surveillance must be proportional to the threat
    - **Necessity:** Only collect data necessary for security
    - **Accountability:** Clear responsibility for system decisions
    - **Transparency:** Public understanding of surveillance capabilities
    - **Non-discrimination:** Avoid disproportionate impact on specific groups

    This simulation helps explore these tensions in a controlled environment.
    """)

# Footer
st.divider()
st.caption("🛡️ National Security Simulation • GAGS Framework • v2.0 • For Educational Purposes Only")