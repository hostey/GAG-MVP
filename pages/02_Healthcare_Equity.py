# pages/1_🏥_Healthcare_Equity.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    page_title="Healthcare Equity • GAGS",
    layout="wide",
    page_icon="🏥"
)

# Custom CSS for healthcare-themed styling
st.markdown("""
<style>
    .health-header {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 50%, #1c5d8a 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        border-left: 6px solid #e74c3c;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .accuracy-metric-health {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #3498db;
    }
    .fairness-metric-health {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #2ecc71;
    }
    .access-metric-health {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #9b59b6;
    }
    .outcome-metric-health {
        background: linear-gradient(135deg, #e67e22 0%, #d35400 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #e67e22;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s;
        border: 2px solid #3498db;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
    }
    .hospital-card {
        background: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        color: #2c3e50;
    }
    .bias-tag-health {
        display: inline-block;
        background: #e74c3c;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: bold;
    }
    .patient-group {
        display: inline-block;
        background: #2ecc71;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .health-badge {
        background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: #2c3e50;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .warning-banner-health {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        color: #856404;
        margin: 1rem 0;
    }
    .safety-alert {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        color: #721c24;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Header Section
# ───────────────────────────────────────────────
st.markdown("""
<div class="health-header">
    <h1 style="margin:0; color:white; font-size:2.5rem;">🏥 Healthcare Equity Simulation</h1>
    <p style="margin:0; opacity:0.9; font-size:1.2rem; margin-top:0.5rem;">
        Explore how <strong>bias, algorithmic fairness, and data quality</strong> impact healthcare decisions and patient outcomes
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #3498db;">
    <p style="margin:0; color:#2c3e50;">
        <strong>🩺 Healthcare Mission:</strong> AI in healthcare holds promise for improved diagnostics and treatment but risks perpetuating 
        health disparities. This simulation explores fairness in diagnosis, treatment recommendations, and resource allocation.
    </p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Session State Initialization
# ───────────────────────────────────────────────
if "health_run_history" not in st.session_state:
    st.session_state.health_run_history = []
if "patient_groups" not in st.session_state:
    st.session_state.patient_groups = {}
if "treatment_results" not in st.session_state:
    st.session_state.treatment_results = {}

# ───────────────────────────────────────────────
# Sidebar Controls
# ───────────────────────────────────────────────
with st.sidebar:
    # Header with icon
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h2 style="color:#3498db; margin:0;">⚙️ Healthcare Configuration</h2>
        <p style="color:#7f8c8d; font-size:0.9rem;">Configure your healthcare equity simulation</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Healthcare Context Selection
    st.subheader("🏥 Healthcare Context")
    healthcare_setting = st.selectbox(
        "Select Healthcare Setting",
        ["Hospital System", "Primary Care", "Specialty Care", "Telemedicine",
         "Public Health Screening", "Clinical Trials"],
        help="Different healthcare settings have different fairness challenges"
    )

    # Country/Region
    region = st.selectbox(
        "Region/Country",
        ["United States", "United Kingdom", "Canada", "Germany", "Japan",
         "India", "Nigeria", "Brazil", "Global South", "Global North"],
        help="Region affects baseline healthcare access and disparities"
    )

    st.divider()

    # Bias Configuration
    st.subheader("🎭 Bias Configuration")

    selected_biases = st.multiselect(
        "**Select Bias Types**",
        options=["demographic", "socioeconomic", "historical", "measurement",
                 "access_bias", "diagnostic_bias", "treatment_bias"],
        default=["demographic", "socioeconomic", "historical"],
        help="Biases affecting healthcare decisions",
        format_func=lambda x: f"👥 {x}" if x == "demographic" else
        f"💰 {x}" if x == "socioeconomic" else
        f"📜 {x}" if x == "historical" else
        f"📏 {x}" if x == "measurement" else
        f"🚑 {x}" if x == "access_bias" else
        f"🔍 {x}" if x == "diagnostic_bias" else
        f"💊 {x}" if x == "treatment_bias" else
        str(x)
    )

    # Display selected biases as tags
    if selected_biases:
        tags_html = "".join([f'<span class="bias-tag-health">{bias}</span>' for bias in selected_biases])
        st.markdown(f"**Active Biases:**<br>{tags_html}", unsafe_allow_html=True)

    bias_intensity = st.slider(
        "**Bias Intensity**",
        0.0, simulation_config.MAX_BIAS_FACTOR, 0.3, 0.05,
        help="Strength of bias in healthcare algorithms"
    )

    st.divider()

    # Attack Configuration
    st.subheader("⚠️ Adversarial Attacks")

    attack_type = st.selectbox(
        "**Attack Type**",
        ["Data Poisoning", "Model Evasion", "Synthetic Patient Injection",
         "Diagnosis Manipulation", "Treatment Recommendation Tampering"],
        help="Type of attack on healthcare data"
    )

    poison_rate = st.slider(
        "**Attack Strength**",
        0.0, 0.5, 0.05, 0.01,
        format="%.2f",
        help="Proportion of healthcare data corrupted"
    )

    st.divider()

    # Patient Demographics
    st.subheader("👨‍⚕️ Patient Demographics")

    # Socioeconomic diversity
    low_income_ratio = st.slider(
        "**Low-Income Patients**",
        0.0, 1.0, 0.4, 0.05,
        help="Percentage of patients from low-income backgrounds"
    )

    # Demographic groups
    demographic_diversity = st.slider(
        "**Demographic Diversity**",
        0.0, 1.0, 0.6, 0.05,
        help="Diversity across demographic groups (0 = homogeneous, 1 = highly diverse)"
    )

    # Insurance coverage
    uninsured_ratio = st.slider(
        "**Uninsured/Underinsured**",
        0.0, 1.0, 0.25, 0.05,
        help="Percentage of patients with limited/no insurance coverage"
    )

    st.divider()

    # Healthcare Access Factors
    st.subheader("🏥 Access & Quality Factors")

    access_inequality = st.slider(
        "**Access Inequality Factor**",
        0.0, 1.0, 0.4, 0.05,
        help="0 = Equal access for all, 1 = Highly unequal access"
    )

    quality_variation = st.select_slider(
        "**Healthcare Quality Variation**",
        options=["Uniform", "Minimal", "Moderate", "Significant", "Extreme"],
        value="Moderate"
    )

    rural_access = st.slider(
        "**Rural Access Equity**",
        0.0, 1.0, 0.6,
        help="0 = Poor rural access, 1 = Equal rural/urban access"
    )

    st.divider()

    # Simulation Configuration
    st.subheader("📊 Simulation Parameters")

    sample_size = st.number_input(
        "**Number of Patient Records**",
        1000, 100000, settings.DEFAULT_N_SAMPLES, step=1000
    )

    n_runs = st.slider(
        "**Number of Simulation Runs**",
        1, 10, 3,
        help="More runs provide more reliable statistics"
    )

    prediction_task = st.selectbox(
        "**Prediction Task**",
        ["Disease Diagnosis", "Treatment Response", "Readmission Risk",
         "Length of Stay", "Mortality Risk", "Treatment Recommendation"],
        help="What the AI system is predicting"
    )

    st.divider()

    # Run Button
    col_run, col_reset = st.columns(2)
    with col_run:
        run_button = st.button(
            "🏥 **Run**",
            type="primary",
            use_container_width=True
        )
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.health_run_history = []
            st.session_state.patient_groups = {}
            st.rerun()


# ───────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────
def generate_healthcare_data(n_samples, n_features=12, low_income_ratio=0.4, demographic_diversity=0.6):
    """Generate realistic healthcare data with patient demographics."""
    # Get base synthetic data
    X, y_base, demo_info = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features
    )

    # Add healthcare features
    # Feature 0: Age
    X[:, 0] = np.random.normal(55, 18, n_samples)
    X[:, 0] = np.clip(X[:, 0], 18, 100)

    # Feature 1: BMI (Body Mass Index)
    X[:, 1] = np.random.normal(27, 6, n_samples)
    X[:, 1] = np.clip(X[:, 1], 15, 50)

    # Feature 2: Blood Pressure (systolic)
    X[:, 2] = np.random.normal(130, 20, n_samples)
    X[:, 2] = np.clip(X[:, 2], 80, 200)

    # Feature 3: Cholesterol level
    X[:, 3] = np.random.normal(200, 40, n_samples)
    X[:, 3] = np.clip(X[:, 3], 100, 350)

    # Feature 4: Socioeconomic status (0-1, higher = wealthier)
    n_low_income = int(n_samples * low_income_ratio)
    socioeconomic = np.ones(n_samples)
    socioeconomic[:n_low_income] = np.random.uniform(0.1, 0.4, n_low_income)
    socioeconomic[n_low_income:] = np.random.uniform(0.6, 1.0, n_samples - n_low_income)
    X[:, 4] = socioeconomic

    # Feature 5: Access to healthcare resources
    resource_access = socioeconomic * 0.8 + np.random.uniform(0, 0.2, n_samples)
    X[:, 5] = np.clip(resource_access, 0, 1)

    # Feature 6: Number of chronic conditions
    X[:, 6] = np.random.poisson(1.5, n_samples)
    X[:, 6] = np.clip(X[:, 6], 0, 8)

    # Feature 7: Health literacy score (0-1)
    health_literacy = socioeconomic * 0.6 + np.random.uniform(0, 0.4, n_samples)
    X[:, 7] = np.clip(health_literacy, 0, 1)

    # Feature 8: Insurance coverage (0-1)
    insurance_coverage = socioeconomic * 0.7 + np.random.uniform(0, 0.3, n_samples)
    X[:, 8] = np.clip(insurance_coverage, 0, 1)

    # Feature 9: Previous hospitalizations
    X[:, 9] = np.random.poisson(0.8, n_samples)

    # Feature 10: Genetic risk factor (0-1)
    X[:, 10] = np.random.beta(2, 5, n_samples)

    # Feature 11: Lifestyle score (0-1, higher = healthier)
    lifestyle = socioeconomic * 0.5 + np.random.uniform(0, 0.5, n_samples)
    X[:, 11] = np.clip(lifestyle, 0, 1)

    # Generate health outcome probability
    # Base on health factors, but introduce bias factors
    age_risk = (X[:, 0] - 18) / 82 * 0.2  # Age contribution
    bmi_risk = np.maximum(0, (X[:, 1] - 25) / 25) * 0.15  # BMI risk
    bp_risk = np.maximum(0, (X[:, 2] - 120) / 80) * 0.15  # Blood pressure
    chronic_risk = X[:, 6] / 8 * 0.2  # Chronic conditions
    genetic_risk = X[:, 10] * 0.1  # Genetic factors
    lifestyle_benefit = X[:, 11] * -0.1  # Healthy lifestyle reduces risk

    # Bias factor based on socioeconomic status and access
    bias_factor = np.zeros(n_samples)
    bias_factor[socioeconomic < 0.5] -= bias_intensity * 0.3  # Penalize low-income
    bias_factor[X[:, 5] < 0.5] -= bias_intensity * 0.2  # Penalize poor access

    health_risk = age_risk + bmi_risk + bp_risk + chronic_risk + genetic_risk + lifestyle_benefit + bias_factor

    # Add noise
    health_risk += np.random.normal(0, 0.1, n_samples)

    # Create binary labels (1 = adverse outcome, 0 = good outcome)
    # For healthcare, we want to predict adverse outcomes
    risk_threshold = np.percentile(health_risk, 40)  # Top 60% most at risk
    y = (health_risk > risk_threshold).astype(int)

    # Create patient groups (0 = low-income, 1 = middle/high income)
    patient_groups = np.zeros(n_samples)
    patient_groups[n_low_income:] = 1

    # Add demographic diversity
    if demographic_diversity > 0:
        n_demographic_groups = max(2, int(demographic_diversity * 5))
        demographic_groups = np.random.randint(0, n_demographic_groups, n_samples)
    else:
        demographic_groups = np.zeros(n_samples)

    return X, y, patient_groups, demographic_groups


def calculate_healthcare_metrics(y_true, y_pred, patient_groups, demographic_groups, X_features=None):
    """Calculate comprehensive healthcare equity metrics."""
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn
    })

    # Sensitivity and specificity (important for healthcare)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics.update({
        "sensitivity": sensitivity,
        "specificity": specificity
    })

    # Group-based fairness metrics (income groups)
    group_metrics = {}
    unique_groups = np.unique(patient_groups)

    for group in unique_groups:
        mask = (patient_groups == group)
        if np.sum(mask) == 0:
            continue

        group_true = y_true[mask]
        group_pred = y_pred[mask]

        group_accuracy = accuracy_score(group_true, group_pred)
        group_precision = precision_score(group_true, group_pred, zero_division=0)
        group_recall = recall_score(group_true, group_pred, zero_division=0)

        # Detection rates (for adverse outcomes)
        detection_rate = np.mean(group_pred == 1)

        # False positive/negative rates
        group_fp = np.sum((group_pred == 1) & (group_true == 0))
        group_fn = np.sum((group_pred == 0) & (group_true == 1))
        group_tn = np.sum((group_pred == 0) & (group_true == 0))
        group_tp = np.sum((group_pred == 1) & (group_true == 1))

        group_fpr = group_fp / (group_fp + group_tn) if (group_fp + group_tn) > 0 else 0
        group_fnr = group_fn / (group_fn + group_tp) if (group_fn + group_tp) > 0 else 0

        group_metrics[group] = {
            "accuracy": group_accuracy,
            "precision": group_precision,
            "recall": group_recall,
            "detection_rate": detection_rate,
            "fpr": group_fpr,
            "fnr": group_fnr,
            "sample_size": np.sum(mask)
        }

    # Calculate fairness disparities
    if len(group_metrics) >= 2:
        # Demographic parity difference (detection rate disparity)
        detection_rates = [metrics["detection_rate"] for metrics in group_metrics.values()]
        demographic_parity_diff = max(detection_rates) - min(detection_rates)

        # Equal opportunity difference (recall disparity)
        recalls = [metrics["recall"] for metrics in group_metrics.values()]
        equal_opportunity_diff = max(recalls) - min(recalls)

        # Equalized odds difference (combined FPR and FNR)
        fprs = [metrics["fpr"] for metrics in group_metrics.values()]
        fnrs = [metrics["fnr"] for metrics in group_metrics.values()]
        equalized_odds_diff = max(max(fprs) - min(fprs), max(fnrs) - min(fnrs))

        # Overall equity score (0-1, higher is better)
        max_disparity = max(demographic_parity_diff, equal_opportunity_diff, equalized_odds_diff)
        equity_score = 1.0 - min(1.0, max_disparity * 3)

        metrics.update({
            "group_metrics": group_metrics,
            "demographic_parity_difference": demographic_parity_diff,
            "equal_opportunity_difference": equal_opportunity_diff,
            "equalized_odds_difference": equalized_odds_diff,
            "equity_score": equity_score,
            "max_disparity": max_disparity
        })

    # Health access correlation (if feature data available)
    if X_features is not None:
        # Calculate correlation between socioeconomic status and predictions
        socioeconomic_status = X_features[:, 4]
        correlation = np.corrcoef(socioeconomic_status, y_pred)[0, 1]
        metrics["socioeconomic_correlation"] = correlation

        # Calculate healthcare access correlation
        healthcare_access = X_features[:, 5]
        access_correlation = np.corrcoef(healthcare_access, y_pred)[0, 1]
        metrics["access_correlation"] = access_correlation

        # Calculate insurance coverage correlation
        insurance_coverage = X_features[:, 8]
        insurance_correlation = np.corrcoef(insurance_coverage, y_pred)[0, 1]
        metrics["insurance_correlation"] = insurance_correlation

    return metrics


# ───────────────────────────────────────────────
# Main Content Area
# ───────────────────────────────────────────────
if run_button or st.session_state.health_run_history:

    if run_button:
        # Clear previous results
        st.session_state.health_run_history = []

        # Run simulations
        progress_bar = st.progress(0)

        for i in range(n_runs):
            with st.spinner(f"Running healthcare simulation {i + 1}/{n_runs}..."):

                # Generate healthcare data
                X, y, patient_groups, demographic_groups = generate_healthcare_data(
                    n_samples=sample_size,
                    low_income_ratio=low_income_ratio,
                    demographic_diversity=demographic_diversity
                )

                # Apply selected biases
                for bias_type in selected_biases:
                    if bias_type == "socioeconomic":
                        # Amplify bias based on socioeconomic status
                        low_ses_mask = (X[:, 4] < 0.5)  # Low SES patients
                        # Add noise to low SES patients' features
                        noise_scale = bias_intensity * 0.5
                        X[low_ses_mask] += np.random.normal(0, noise_scale, (np.sum(low_ses_mask), X.shape[1]))

                    elif bias_type == "historical":
                        # Historical bias: under-predict outcomes for certain groups
                        for group in np.unique(demographic_groups):
                            if group > 0:  # Assume non-majority groups
                                group_mask = (demographic_groups == group)
                                if np.any(group_mask):
                                    # Reduce predicted risk for minority groups (historical under-diagnosis)
                                    bias_strength = bias_intensity * 0.3
                                    if np.random.rand() < bias_strength:
                                        high_risk_mask = group_mask & (y == 1)
                                        if len(np.where(high_risk_mask)[0]) > 0:
                                            flip_count = int(np.sum(high_risk_mask) * bias_strength)
                                            flip_indices = np.random.choice(np.where(high_risk_mask)[0], flip_count,
                                                                            replace=False)
                                            y[flip_indices] = 0

                # Apply access inequality
                if access_inequality > 0:
                    # Low-income patients get worse access features
                    low_income_mask = (patient_groups == 0)
                    access_factor = 1 - (access_inequality * 0.6)
                    X[low_income_mask, 5] *= access_factor  # Resource access
                    X[low_income_mask, 8] *= access_factor  # Insurance coverage

                # Apply rural access inequality
                if rural_access < 1.0:
                    access_gap = 1 - rural_access
                    # Simulate rural patients (random 30% of low-access)
                    rural_mask = (X[:, 5] < access_gap) & (np.random.rand(len(X)) < 0.3)
                    # Reduce features for rural patients
                    X[rural_mask, 7] *= 0.7  # Health literacy
                    X[rural_mask, 11] *= 0.8  # Lifestyle score

                # Apply poisoning attack
                X_p, y_p, patient_groups = simulate_data_poisoning(
                    X, y, poison_rate,
                    attack_type="label_flipping",
                    demographic_info=patient_groups,
                    targeted=True
                )

                # Split data
                X_train, X_test, y_train, y_test, groups_train, groups_test, demo_train, demo_test = train_test_split(
                    X_p, y_p, patient_groups, demographic_groups, test_size=0.3, random_state=42 + i
                )

                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train model (using class weights to handle imbalance)
                model = RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    random_state=42 + i
                )

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                # Calculate comprehensive metrics
                metrics = calculate_healthcare_metrics(
                    y_test, y_pred, groups_test, demo_test, X_test
                )

                # Calculate additional healthcare-specific metrics
                # Adverse outcome rate by income group
                low_income_mask = (groups_test == 0)
                high_income_mask = (groups_test == 1)

                adverse_rate_low = np.mean(y_test[low_income_mask]) if np.any(low_income_mask) else 0
                adverse_rate_high = np.mean(y_test[high_income_mask]) if np.any(high_income_mask) else 0
                adverse_gap = abs(adverse_rate_high - adverse_rate_low)

                # Detection accuracy by group
                accuracy_low = accuracy_score(y_test[low_income_mask], y_pred[low_income_mask]) if np.any(
                    low_income_mask) else 0
                accuracy_high = accuracy_score(y_test[high_income_mask], y_pred[high_income_mask]) if np.any(
                    high_income_mask) else 0
                accuracy_gap = abs(accuracy_high - accuracy_low)

                # Sensitivity (true positive rate) by group
                sensitivity_low = recall_score(y_test[low_income_mask], y_pred[low_income_mask],
                                               zero_division=0) if np.any(low_income_mask) else 0
                sensitivity_high = recall_score(y_test[high_income_mask], y_pred[high_income_mask],
                                                zero_division=0) if np.any(high_income_mask) else 0
                sensitivity_gap = abs(sensitivity_high - sensitivity_low)

                # Health equity score
                health_equity_score = 1.0 - min(1.0, adverse_gap + accuracy_gap + sensitivity_gap)

                # Store results
                run_result = {
                    "run_id": i + 1,
                    "healthcare_setting": healthcare_setting,
                    "region": region,
                    "prediction_task": prediction_task,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1_score": metrics.get("f1_score", 0),
                    "sensitivity": metrics.get("sensitivity", 0),
                    "specificity": metrics.get("specificity", 0),
                    "equity_score": metrics.get("equity_score", 0.5),
                    "health_equity_score": health_equity_score,
                    "adverse_rate_low": adverse_rate_low,
                    "adverse_rate_high": adverse_rate_high,
                    "accuracy_low": accuracy_low,
                    "accuracy_high": accuracy_high,
                    "sensitivity_low": sensitivity_low,
                    "sensitivity_high": sensitivity_high,
                    "demographic_parity": metrics.get("demographic_parity_difference", 0),
                    "equal_opportunity": metrics.get("equal_opportunity_difference", 0),
                    "bias_intensity": bias_intensity,
                    "poison_rate": poison_rate,
                    "access_inequality": access_inequality,
                    "rural_access": rural_access,
                    "low_income_ratio": low_income_ratio,
                    "uninsured_ratio": uninsured_ratio,
                    "biases": ", ".join(selected_biases) if selected_biases else "None",
                    "quality_variation": quality_variation,
                    "attack_type": attack_type
                }

                st.session_state.health_run_history.append(run_result)
                progress_bar.progress((i + 1) / n_runs)

        progress_bar.empty()

    # ── Display Results ───────────────────────────────
    if st.session_state.health_run_history:
        df = pd.DataFrame(st.session_state.health_run_history)

        # Summary Metrics Section
        st.markdown("## 📊 Healthcare Equity Dashboard")

        # Calculate averages
        avg_accuracy = df["accuracy"].mean()
        avg_equity = df["equity_score"].mean()
        avg_health_equity = df["health_equity_score"].mean()
        avg_adverse_gap = abs(df["adverse_rate_high"].mean() - df["adverse_rate_low"].mean())

        # Display metrics in styled cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="accuracy-metric-health">', unsafe_allow_html=True)
            st.metric(
                label="🎯 Prediction Accuracy",
                value=f"{avg_accuracy:.1%}",
                delta=None,
                help="Overall accuracy of the healthcare AI system"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="fairness-metric-health">', unsafe_allow_html=True)
            st.metric(
                label="⚖️ Algorithmic Equity",
                value=f"{avg_equity:.2f}/1.0",
                delta=None,
                help="Fairness across patient groups (1.0 = perfect equity)"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="access-metric-health">', unsafe_allow_html=True)
            st.metric(
                label="🏥 Health Equity Score",
                value=f"{avg_health_equity:.2f}/1.0",
                delta=None,
                help="Equity in healthcare outcomes across socioeconomic groups"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="outcome-metric-health">', unsafe_allow_html=True)
            st.metric(
                label="📈 Adverse Outcome Gap",
                value=f"{avg_adverse_gap:.1%}",
                delta_color="inverse",
                help="Difference in adverse outcome rates between income groups"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Performance comparison by income group
        st.markdown("### 👨‍⚕️ Performance by Patient Income Group")

        col1, col2 = st.columns(2)

        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                name="Low-Income Patients",
                x=df["run_id"],
                y=df["accuracy_low"] * 100,
                marker_color='#e74c3c'
            ))
            fig1.add_trace(go.Bar(
                name="Higher-Income Patients",
                x=df["run_id"],
                y=df["accuracy_high"] * 100,
                marker_color='#2ecc71'
            ))
            fig1.update_layout(
                title="Prediction Accuracy by Income Group",
                barmode='group',
                yaxis_title="Accuracy (%)",
                xaxis_title="Simulation Run"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="Low-Income Adverse Rate",
                x=df["run_id"],
                y=df["adverse_rate_low"] * 100,
                marker_color='#3498db'
            ))
            fig2.add_trace(go.Bar(
                name="Higher-Income Adverse Rate",
                x=df["run_id"],
                y=df["adverse_rate_high"] * 100,
                marker_color='#9b59b6'
            ))
            fig2.update_layout(
                title="Actual Adverse Outcome Rates by Income Group",
                barmode='group',
                yaxis_title="Adverse Outcome Rate (%)",
                xaxis_title="Simulation Run"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ── Detailed Analysis Tabs ───────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Overview", "⚖️ Equity Analysis",
                                          "🏥 Clinical Impact", "📋 Detailed Results"])

        with tab1:
            # Performance gauges
            fig = make_subplots(
                rows=1, cols=4,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'},
                        {'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=("Accuracy", "Sensitivity", "Specificity", "Equity")
            )

            # Accuracy Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_accuracy * 100,
                title={'text': "Accuracy", 'font': {'size': 14}},
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
                        'value': 80
                    }
                }
            ), row=1, col=1)

            # Sensitivity Gauge
            avg_sensitivity = df["sensitivity"].mean() if "sensitivity" in df.columns else 0.7
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_sensitivity * 100,
                title={'text': "Sensitivity", 'font': {'size': 14}},
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
                        'value': 75
                    }
                }
            ), row=1, col=2)

            # Specificity Gauge
            avg_specificity = df["specificity"].mean() if "specificity" in df.columns else 0.8
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_specificity * 100,
                title={'text': "Specificity", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkorange"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 85], 'color': "gray"},
                        {'range': [85, 100], 'color': "darkgray"}
                    ]
                }
            ), row=1, col=3)

            # Equity Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_equity * 100,
                title={'text': "Equity", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ]
                }
            ), row=1, col=4)

            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Trade-off scatter plot
            fig2 = px.scatter(
                df,
                x="equity_score",
                y="accuracy",
                size="health_equity_score",
                color="bias_intensity",
                hover_data=["biases", "access_inequality", "quality_variation"],
                title="The Healthcare Trilemma: Accuracy vs Equity vs Clinical Safety",
                labels={
                    "equity_score": "Algorithmic Equity",
                    "accuracy": "Prediction Accuracy",
                    "health_equity_score": "Health Equity",
                    "bias_intensity": "Bias Intensity"
                },
                size_max=30
            )

            # Add optimal zone
            fig2.add_shape(
                type="rect",
                x0=0.7, x1=1.0,
                y0=0.7, y1=1.0,
                line=dict(color="Green", width=2, dash="dash"),
                fillcolor="rgba(0, 255, 0, 0.1)",
                label=dict(
                    text="Optimal Zone",
                    font=dict(size=12, color="green"),
                    xanchor="center",
                    yanchor="middle")
            )

            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            # Equity analysis
            st.markdown("### ⚖️ Health Equity Gap Analysis")

            # Calculate equity gaps
            df["accuracy_gap"] = abs(df["accuracy_high"] - df["accuracy_low"])
            df["adverse_gap"] = abs(df["adverse_rate_high"] - df["adverse_rate_low"])
            df["sensitivity_gap"] = abs(df["sensitivity_high"] - df["sensitivity_low"])

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    df,
                    x="run_id",
                    y=["accuracy_gap", "adverse_gap", "sensitivity_gap"],
                    barmode="group",
                    title="Health Equity Gaps Across Simulation Runs",
                    labels={"value": "Gap Size", "variable": "Metric"},
                    color_discrete_sequence=["#e74c3c", "#3498db", "#9b59b6"]
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Equity impact factors
                factors = pd.DataFrame({
                    "Factor": ["Bias Intensity", "Access Inequality", "Rural Access", "Insurance Gap"],
                    "Impact on Equity": [
                        -df["bias_intensity"].mean() * 0.8,
                        -df["access_inequality"].mean() * 0.7,
                        df["rural_access"].mean() * 0.5,  # Positive impact
                        -df["uninsured_ratio"].mean() * 0.6
                    ]
                })

                fig2 = px.bar(
                    factors,
                    x="Factor",
                    y="Impact on Equity",
                    title="Factors Affecting Healthcare Equity",
                    color="Impact on Equity",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Safety alerts based on sensitivity gaps
            avg_sensitivity_gap = df["sensitivity_gap"].mean()
            if avg_sensitivity_gap > 0.2:
                st.markdown("""
                <div class="safety-alert">
                    <strong>⚠️ CRITICAL SAFETY CONCERN DETECTED</strong><br>
                    <strong>High Sensitivity Gap ({:.1%})</strong> - Missed diagnoses disproportionately affect certain groups.<br>
                    <strong>Immediate Actions Required:</strong>
                    <ul style="margin-bottom:0;">
                        <li>Audit model for differential performance by demographic group</li>
                        <li>Implement group-specific thresholds for critical conditions</li>
                        <li>Enhance training data representation for underserved groups</li>
                        <li>Establish human oversight for high-stakes predictions</li>
                    </ul>
                </div>
                """.format(avg_sensitivity_gap), unsafe_allow_html=True)
            elif avg_equity < 0.7:
                st.markdown("""
                <div class="warning-banner-health">
                    <strong>⚠️ SIGNIFICANT EQUITY GAPS DETECTED</strong><br>
                    <strong>Recommended Interventions:</strong>
                    <ul style="margin-bottom:0;">
                        <li><strong>Bias Mitigation:</strong> Implement fairness-aware algorithms</li>
                        <li><strong>Data Augmentation:</strong> Enhance representation of underserved groups</li>
                        <li><strong>Clinical Validation:</strong> Validate across diverse patient populations</li>
                        <li><strong>Transparent Reporting:</strong> Disclose performance by demographic group</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("""
                **✅ GOOD EQUITY ACHIEVED**

                **Maintenance Actions:**
                1. Continue monitoring performance by demographic groups
                2. Regular bias audits of healthcare algorithms
                3. Engage with patient advocacy groups for feedback
                4. Update models as clinical guidelines evolve
                """)

        with tab3:
            # Clinical impact analysis
            st.markdown("### 🏥 Clinical Impact Analysis")

            # Simulated patient outcomes for visualization
            np.random.seed(42)

            # Create sample data for different patient groups
            groups = ["Low-Income", "Uninsured", "Rural", "Minority", "General"]
            detection_rates = [
                np.random.uniform(0.6, 0.8) * (1 - bias_intensity * 0.3),
                np.random.uniform(0.5, 0.7) * (1 - bias_intensity * 0.4),
                np.random.uniform(0.55, 0.75) * (1 - bias_intensity * 0.35),
                np.random.uniform(0.6, 0.8) * (1 - bias_intensity * 0.3),
                np.random.uniform(0.8, 0.95)
            ]

            false_positive_rates = [
                np.random.uniform(0.15, 0.25) * (1 + bias_intensity * 0.2),
                np.random.uniform(0.2, 0.3) * (1 + bias_intensity * 0.3),
                np.random.uniform(0.18, 0.28) * (1 + bias_intensity * 0.25),
                np.random.uniform(0.16, 0.26) * (1 + bias_intensity * 0.2),
                np.random.uniform(0.05, 0.15)
            ]

            treatment_access = [
                np.random.uniform(0.4, 0.6) * (1 - access_inequality * 0.5),
                np.random.uniform(0.3, 0.5) * (1 - access_inequality * 0.6),
                np.random.uniform(0.35, 0.55) * (1 - access_inequality * 0.4),
                np.random.uniform(0.45, 0.65) * (1 - access_inequality * 0.3),
                np.random.uniform(0.7, 0.9)
            ]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Detection Rate",
                x=groups,
                y=detection_rates,
                marker_color='#3498db',
                text=[f"{rate:.1%}" for rate in detection_rates],
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name="False Positive Rate",
                x=groups,
                y=false_positive_rates,
                marker_color='#e74c3c',
                text=[f"{rate:.1%}" for rate in false_positive_rates],
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name="Treatment Access",
                x=groups,
                y=treatment_access,
                marker_color='#2ecc71',
                text=[f"{rate:.1%}" for rate in treatment_access],
                textposition='auto'
            ))

            fig.update_layout(
                title="Clinical Performance by Patient Group",
                yaxis_title="Rate",
                barmode='group',
                yaxis_range=[0, 1],
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Resource allocation visualization
            st.markdown("#### 🏥 Healthcare Resource Allocation")

            resources = ["Preventive Care", "Specialist Access", "Diagnostic Tests", "Medications", "Follow-up Care"]
            low_income_allocation = [0.3, 0.25, 0.4, 0.35, 0.3]
            high_income_allocation = [0.7, 0.8, 0.9, 0.85, 0.8]

            # Apply access inequality factor
            low_income_allocation = [x * (1 - access_inequality) for x in low_income_allocation]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="Low-Income Access",
                x=resources,
                y=low_income_allocation,
                marker_color='#e74c3c'
            ))
            fig2.add_trace(go.Bar(
                name="Higher-Income Access",
                x=resources,
                y=high_income_allocation,
                marker_color='#2ecc71'
            ))

            fig2.update_layout(
                title="Healthcare Resource Access by Income Group",
                yaxis_title="Access Level (0-1)",
                barmode='group'
            )

            st.plotly_chart(fig2, use_container_width=True)

        with tab4:
            # Detailed results table
            st.dataframe(
                df.style.format({
                    "accuracy": "{:.1%}",
                    "precision": "{:.1%}",
                    "recall": "{:.1%}",
                    "f1_score": "{:.2f}",
                    "sensitivity": "{:.1%}",
                    "specificity": "{:.1%}",
                    "equity_score": "{:.2f}",
                    "health_equity_score": "{:.2f}",
                    "adverse_rate_low": "{:.1%}",
                    "adverse_rate_high": "{:.1%}",
                    "accuracy_low": "{:.1%}",
                    "accuracy_high": "{:.1%}",
                    "sensitivity_low": "{:.1%}",
                    "sensitivity_high": "{:.1%}",
                    "demographic_parity": "{:.3f}",
                    "equal_opportunity": "{:.3f}"
                }).background_gradient(subset=["accuracy"], cmap="Blues")
                .background_gradient(subset=["equity_score"], cmap="RdYlGn")
                .background_gradient(subset=["sensitivity_low"], cmap="Greens")
            )

            # Download options
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    f"gags_healthcare_{healthcare_setting.lower().replace(' ', '_')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col2:
                # Export configuration
                config_data = {
                    "healthcare_setting": healthcare_setting,
                    "region": region,
                    "prediction_task": prediction_task,
                    "selected_biases": selected_biases,
                    "bias_intensity": bias_intensity,
                    "attack_type": attack_type,
                    "poison_rate": poison_rate,
                    "low_income_ratio": low_income_ratio,
                    "demographic_diversity": demographic_diversity,
                    "access_inequality": access_inequality,
                    "rural_access": rural_access,
                    "uninsured_ratio": uninsured_ratio,
                    "quality_variation": quality_variation,
                    "average_accuracy": f"{avg_accuracy:.1%}",
                    "average_equity": f"{avg_equity:.2f}",
                    "average_sensitivity": f"{avg_sensitivity:.1%}"
                }
                import json

                json_str = json.dumps(config_data, indent=2)
                st.download_button(
                    "📋 Export Configuration (JSON)",
                    json_str,
                    f"healthcare_config_{healthcare_setting.lower().replace(' ', '_')}.json",
                    "application/json",
                    use_container_width=True
                )

        # ── Policy Recommendations ───────────────────────────────
        st.divider()

        # Generate actionable recommendations
        st.markdown("## 💡 Healthcare Policy Recommendations")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.info("""
            **🏥 For Healthcare Providers:**

            1. **Transparent Algorithms:** Publish validation studies and performance metrics
            2. **Human Oversight:** Ensure clinician review of high-stakes AI recommendations
            3. **Bias Audits:** Regular third-party audits for fairness across demographic groups
            4. **Diverse Training Data:** Ensure representation of all patient populations
            5. **Patient Consent:** Obtain informed consent for AI-assisted decisions

            **👨‍⚕️ For Clinical Safety:**

            1. **Safety Monitoring:** Continuous monitoring for differential performance
            2. **Error Analysis:** Regular review of false positives/negatives by group
            3. **Quality Assurance:** Integration with clinical quality improvement programs
            4. **Emergency Override:** Clear protocols for overriding AI recommendations
            """)

        with rec_col2:
            st.success("""
            **🤖 For AI Development:**

            1. **Clinical Validation:** Rigorous testing across diverse patient populations
            2. **Explainable AI:** Provide interpretable explanations for clinical decisions
            3. **Continuous Monitoring:** Track performance across demographic groups
            4. **Adversarial Testing:** Test systems against various attack vectors
            5. **Stakeholder Involvement:** Include clinicians, patients, and ethicists in design

            **📊 For Regulators & Policymakers:**

            1. **Clinical Standards:** Develop standards for healthcare AI validation
            2. **Equity Requirements:** Mandate fairness testing for regulatory approval
            3. **Transparency Mandates:** Require disclosure of performance by demographic group
            4. **Post-Market Surveillance:** Continuous monitoring of real-world performance
            """)

        st.caption(
            "⚠️ **Disclaimer:** This simulation is for educational purposes. Real healthcare AI systems require extensive clinical validation, regulatory approval, and ethical review.")

else:
    # Welcome/Instruction state
    st.markdown("## 🏥 Welcome to Healthcare Equity Simulation")

    st.markdown("""
    This module explores how AI impacts healthcare equity across diagnosis, treatment, resource allocation, 
    and patient outcomes. You'll configure healthcare scenarios and analyze fairness across different patient groups.
    """)

    # Quick start examples
    st.markdown("### 🚀 Quick Start Scenarios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="hospital-card">
            <h4>🏥 Hospital Diagnostics</h4>
            <p>Explore fairness in diagnostic algorithms:</p>
            <ul>
                <li>High-stakes prediction tasks</li>
                <li>Socioeconomic bias focus</li>
                <li>Clinical safety considerations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="hospital-card">
            <h4>🩺 Primary Care Screening</h4>
            <p>Analyze equity in preventive care:</p>
            <ul>
                <li>Access inequality focus</li>
                <li>Early detection challenges</li>
                <li>Population health impact</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="hospital-card">
            <h4>💊 Treatment Recommendations</h4>
            <p>Study fairness in treatment algorithms:</p>
            <ul>
                <li>Resource allocation decisions</li>
                <li>Cost-effectiveness considerations</li>
                <li>Clinical guideline adherence</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # How to use guide
    with st.expander("📖 How to Use This Simulation", expanded=True):
        st.markdown("""
        1. **Select healthcare context** (setting and region)
        2. **Configure biases** affecting healthcare algorithms
        3. **Set patient demographics** (income levels, insurance status)
        4. **Adjust access factors** (inequality, rural access)
        5. **Configure adversarial attacks** on healthcare data
        6. **Run multiple simulations** to see statistical trends
        7. **Analyze trade-offs** between accuracy and equity
        8. **Explore clinical impact** on different patient groups
        9. **Review recommendations** for policy and practice

        **Key Metrics to Watch:**
        - **Accuracy:** Overall prediction performance
        - **Sensitivity:** True positive rate (critical for healthcare)
        - **Equity Score:** Fairness across patient groups (0-1)
        - **Health Equity Score:** Equity in healthcare outcomes (0-1)
        - **Adverse Outcome Gap:** Difference in outcomes between income groups
        - **Sensitivity Gap:** Difference in detection rates by demographic group
        """)

    # Real-world context
    st.warning("""
    **Real-World Context:**

    Healthcare AI faces unique ethical challenges:
    - **High-Stakes Decisions:** Diagnosis and treatment affect patient health and lives
    - **Clinical Safety:** False negatives can lead to missed diagnoses
    - **Access Disparities:** Existing healthcare inequalities can be amplified
    - **Data Quality:** Clinical data may reflect historical biases
    - **Regulatory Compliance:** Must meet clinical validation and safety standards

    Responsible AI in healthcare requires balancing innovation with patient safety, equity, and clinical effectiveness.
    """)

# Footer
st.divider()
st.caption("🏥 Healthcare Equity Simulation • GAGS Framework • v2.0 • Advancing Health Equity Through Fair AI")