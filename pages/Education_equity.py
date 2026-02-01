# pages/4_🎓_Education_Equity.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
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
    page_title="Education Equity • GAGS",
    layout="wide",
    page_icon="🎓"
)

# Custom CSS for education-themed styling
st.markdown("""
<style>
    .edu-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 50%, #2980b9 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        border-left: 6px solid #f1c40f;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .accuracy-metric {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #3498db;
    }
    .equity-metric {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #2ecc71;
    }
    .access-metric {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #9b59b6;
    }
    .resource-metric {
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
    .school-card {
        background: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        color: #2c3e50;
    }
    .bias-tag-edu {
        display: inline-block;
        background: #e74c3c;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: bold;
    }
    .student-group {
        display: inline-block;
        background: #2ecc71;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .achievement-badge {
        background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: #2c3e50;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .warning-banner {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        color: #856404;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Header Section
# ───────────────────────────────────────────────
st.markdown("""
<div class="edu-header">
    <h1 style="margin:0; color:white; font-size:2.5rem;">🎓 Education Equity Simulation</h1>
    <p style="margin:0; opacity:0.9; font-size:1.2rem; margin-top:0.5rem;">
        Explore how <strong>bias, algorithmic fairness, and resource allocation</strong> impact educational opportunities and outcomes
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #3498db;">
    <p style="margin:0; color:#2c3e50;">
        <strong>📚 Educational Mission:</strong> AI in education holds promise for personalized learning but risks perpetuating 
        existing inequalities. This simulation explores fairness in admissions, grading, resource allocation, and student support systems.
    </p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Session State Initialization
# ───────────────────────────────────────────────
if "edu_run_history" not in st.session_state:
    st.session_state.edu_run_history = []
if "student_groups" not in st.session_state:
    st.session_state.student_groups = {}
if "admission_results" not in st.session_state:
    st.session_state.admission_results = {}

# ───────────────────────────────────────────────
# Sidebar Controls
# ───────────────────────────────────────────────
with st.sidebar:
    # Header with icon
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h2 style="color:#3498db; margin:0;">⚙️ Education Configuration</h2>
        <p style="color:#7f8c8d; font-size:0.9rem;">Configure your educational equity simulation</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Education Level Selection
    st.subheader("🏫 Educational Context")
    education_level = st.selectbox(
        "Select Education Level",
        ["K-12 Schools", "Undergraduate Admissions", "Graduate Programs",
         "Vocational Training", "Online Learning Platforms"],
        help="Different education levels have different fairness challenges"
    )

    # Institution Type
    institution_type = st.selectbox(
        "Institution Type",
        ["Public School", "Private School", "Charter School", "Community College",
         "University", "Online Platform"],
        help="Type of educational institution"
    )

    st.divider()

    # Bias Configuration
    st.subheader("🎭 Bias Configuration")

    selected_biases = st.multiselect(
        "**Select Bias Types**",
        options=simulation_config.BIAS_TYPES,
        default=["demographic", "socioeconomic", "historical", "measurement"],
        help="Biases affecting educational equity",
        format_func=lambda x: f"👨‍👩‍👧‍👦 {x}" if x == "demographic" else
        f"💰 {x}" if x == "socioeconomic" else
        f"📜 {x}" if x == "historical" else
        f"📊 {x}" if x == "measurement" else x
    )

    # Display selected biases as tags
    if selected_biases:
        tags_html = "".join([f'<span class="bias-tag-edu">{bias}</span>' for bias in selected_biases])
        st.markdown(f"**Active Biases:**<br>{tags_html}", unsafe_allow_html=True)

    bias_intensity = st.slider(
        "**Bias Intensity**",
        0.0, simulation_config.MAX_BIAS_FACTOR, 0.3, 0.05,
        help="Strength of bias in educational algorithms"
    )

    st.divider()

    # Attack Configuration
    st.subheader("⚠️ Adversarial Attacks")

    attack_type = st.selectbox(
        "**Attack Type**",
        ["Grade Inflation", "Test Score Manipulation", "Recommendation Forgery",
         "Admission Fraud", "Data Poisoning"],
        help="Type of attack on educational data"
    )

    poison_rate = st.slider(
        "**Attack Strength**",
        0.0, 0.5, 0.05, 0.01,
        format="%.2f",
        help="Proportion of educational data corrupted"
    )

    st.divider()

    # Student Demographics
    st.subheader("👨‍🎓 Student Demographics")

    # Socioeconomic diversity
    low_income_ratio = st.slider(
        "**Low-Income Students**",
        0.0, 1.0, 0.4, 0.05,
        help="Percentage of students from low-income backgrounds"
    )

    # Demographic groups
    demographic_diversity = st.slider(
        "**Demographic Diversity**",
        0.0, 1.0, 0.6, 0.05,
        help="Diversity across demographic groups (0 = homogeneous, 1 = highly diverse)"
    )

    # First-generation students
    first_gen_ratio = st.slider(
        "**First-Generation Students**",
        0.0, 1.0, 0.3, 0.05,
        help="Percentage of first-generation college students"
    )

    st.divider()

    # Resource Allocation
    st.subheader("📚 Resource Allocation")

    resource_inequality = st.slider(
        "**Resource Inequality Factor**",
        0.0, 1.0, 0.4, 0.05,
        help="0 = Equal resources for all, 1 = Highly unequal resource distribution"
    )

    support_services = st.select_slider(
        "**Support Services Availability**",
        options=["Minimal", "Basic", "Adequate", "Comprehensive", "Excellent"],
        value="Adequate"
    )

    digital_access = st.slider(
        "**Digital Access Equity**",
        0.0, 1.0, 0.7,
        help="0 = Digital divide exists, 1 = Equal digital access for all"
    )

    st.divider()

    # Simulation Configuration
    st.subheader("📊 Simulation Parameters")

    sample_size = st.number_input(
        "**Number of Student Records**",
        1000, 100000, settings.DEFAULT_N_SAMPLES, step=1000
    )

    n_runs = st.slider(
        "**Number of Simulation Runs**",
        1, 10, 3,
        help="More runs provide more reliable statistics"
    )

    prediction_task = st.selectbox(
        "**Prediction Task**",
        ["Admission Decisions", "Course Success", "Dropout Risk", "Graduation Probability"],
        help="What the AI system is predicting"
    )

    st.divider()

    # Run Button
    col_run, col_reset = st.columns(2)
    with col_run:
        run_button = st.button(
            "🎓 **Run**",
            type="primary",
            use_container_width=True
        )
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.edu_run_history = []
            st.session_state.student_groups = {}
            st.rerun()


# ───────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────
def generate_education_data(n_samples, n_features=12, low_income_ratio=0.4, demographic_diversity=0.6):
    """Generate realistic educational data with student demographics."""
    # Get base synthetic data
    X, y_base, demo_info = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features
    )

    # Add educational features
    # Feature 0: Previous academic performance (GPA)
    X[:, 0] = np.random.normal(3.0, 0.5, n_samples)
    X[:, 0] = np.clip(X[:, 0], 1.0, 4.0)

    # Feature 1: Standardized test scores
    X[:, 1] = np.random.normal(500, 100, n_samples)
    X[:, 1] = np.clip(X[:, 1], 200, 800)

    # Feature 2: Extracurricular activities (0-10 scale)
    X[:, 2] = np.random.poisson(3, n_samples)
    X[:, 2] = np.clip(X[:, 2], 0, 10)

    # Feature 3: Socioeconomic status (0-1, higher = wealthier)
    n_low_income = int(n_samples * low_income_ratio)
    socioeconomic = np.ones(n_samples)
    socioeconomic[:n_low_income] = np.random.uniform(0.1, 0.4, n_low_income)
    socioeconomic[n_low_income:] = np.random.uniform(0.6, 1.0, n_samples - n_low_income)
    X[:, 3] = socioeconomic

    # Feature 4: Access to educational resources
    # Correlated with socioeconomic status but with some randomness
    resource_access = socioeconomic * 0.7 + np.random.uniform(0, 0.3, n_samples)
    X[:, 4] = np.clip(resource_access, 0, 1)

    # Feature 5: Quality of previous school (0-1)
    # Lower for low-income students, higher correlation
    school_quality = socioeconomic * 0.8 + np.random.uniform(0, 0.2, n_samples)
    X[:, 5] = np.clip(school_quality, 0, 1)

    # Feature 6: First-generation status (binary, based on ratio)
    first_gen_ratio = 0.3  # Can be made parameter
    n_first_gen = int(n_samples * first_gen_ratio)
    first_gen = np.zeros(n_samples)
    first_gen_indices = np.random.choice(n_samples, n_first_gen, replace=False)
    first_gen[first_gen_indices] = 1
    X[:, 6] = first_gen

    # Generate success probability (for classification)
    # Base on academic performance, but introduce bias factors
    academic_score = (X[:, 0] / 4.0) * 0.4  # GPA contribution
    test_score = (X[:, 1] - 200) / 600 * 0.3  # Test score contribution
    extracurricular = X[:, 2] / 10 * 0.1  # Extracurriculars
    bias_factor = X[:, 3] * 0.2  # Socioeconomic bias (favor wealthier)

    success_prob = academic_score + test_score + extracurricular + bias_factor

    # Add noise
    success_prob += np.random.normal(0, 0.1, n_samples)

    # Create binary labels (success/failure)
    y = (success_prob > np.median(success_prob)).astype(int)

    # Create student groups (0 = low-income, 1 = middle/high income)
    student_groups = np.zeros(n_samples)
    student_groups[n_low_income:] = 1

    # Add demographic diversity
    if demographic_diversity > 0:
        n_demographic_groups = max(2, int(demographic_diversity * 5))
        demographic_groups = np.random.randint(0, n_demographic_groups, n_samples)
    else:
        demographic_groups = np.zeros(n_samples)

    return X, y, student_groups, demographic_groups


def calculate_education_metrics(y_true, y_pred, student_groups, demographic_groups, X_features=None):
    """Calculate comprehensive education equity metrics."""
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

    # Group-based fairness metrics (income groups)
    group_metrics = {}
    unique_groups = np.unique(student_groups)

    for group in unique_groups:
        mask = (student_groups == group)
        if np.sum(mask) == 0:
            continue

        group_true = y_true[mask]
        group_pred = y_pred[mask]

        group_accuracy = accuracy_score(group_true, group_pred)
        group_precision = precision_score(group_true, group_pred, zero_division=0)
        group_recall = recall_score(group_true, group_pred, zero_division=0)

        # Acceptance rates (for admission scenarios)
        acceptance_rate = np.mean(group_pred == 1)

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
            "acceptance_rate": acceptance_rate,
            "fpr": group_fpr,
            "fnr": group_fnr,
            "sample_size": np.sum(mask)
        }

    # Calculate fairness disparities
    if len(group_metrics) >= 2:
        # Demographic parity difference (acceptance rate disparity)
        acceptance_rates = [metrics["acceptance_rate"] for metrics in group_metrics.values()]
        demographic_parity_diff = max(acceptance_rates) - min(acceptance_rates)

        # Equal opportunity difference (recall disparity)
        recalls = [metrics["recall"] for metrics in group_metrics.values()]
        equal_opportunity_diff = max(recalls) - min(recalls)

        # Predictive parity difference (precision disparity)
        precisions = [metrics["precision"] for metrics in group_metrics.values()]
        predictive_parity_diff = max(precisions) - min(precisions)

        # Overall equity score (0-1, higher is better)
        max_disparity = max(demographic_parity_diff, equal_opportunity_diff, predictive_parity_diff)
        equity_score = 1.0 - min(1.0, max_disparity * 2)  # Scale appropriately

        metrics.update({
            "group_metrics": group_metrics,
            "demographic_parity_difference": demographic_parity_diff,
            "equal_opportunity_difference": equal_opportunity_diff,
            "predictive_parity_difference": predictive_parity_diff,
            "equity_score": equity_score,
            "max_disparity": max_disparity
        })

    # Opportunity gap calculation (if feature data available)
    if X_features is not None:
        # Calculate correlation between socioeconomic status and predictions
        socioeconomic_status = X_features[:, 3]  # Assuming feature 3 is SES
        correlation = np.corrcoef(socioeconomic_status, y_pred)[0, 1]
        metrics["socioeconomic_correlation"] = correlation

        # Calculate resource access correlation
        resource_access = X_features[:, 4]
        resource_correlation = np.corrcoef(resource_access, y_pred)[0, 1]
        metrics["resource_correlation"] = resource_correlation

    return metrics


# ───────────────────────────────────────────────
# Main Content Area
# ───────────────────────────────────────────────
if run_button or st.session_state.edu_run_history:

    if run_button:
        # Clear previous results
        st.session_state.edu_run_history = []

        # Run simulations
        progress_bar = st.progress(0)

        for i in range(n_runs):
            with st.spinner(f"Running education simulation {i + 1}/{n_runs}..."):

                # Generate education data
                X, y, student_groups, demographic_groups = generate_education_data(
                    n_samples=sample_size,
                    low_income_ratio=low_income_ratio,
                    demographic_diversity=demographic_diversity
                )

                # Apply selected biases
                for bias_type in selected_biases:
                    if bias_type == "socioeconomic":
                        # Amplify bias based on socioeconomic status
                        ses_mask = (X[:, 3] < 0.5)  # Low SES students
                        # Add noise to low SES students' features
                        noise_scale = bias_intensity * 0.5
                        X[ses_mask] += np.random.normal(0, noise_scale, (np.sum(ses_mask), X.shape[1]))

                    elif bias_type == "historical":
                        # Historical bias: under-predict success for certain groups
                        # Based on demographic groups
                        for group in np.unique(demographic_groups):
                            if group > 0:  # Assume non-majority groups
                                group_mask = (demographic_groups == group)
                                if np.any(group_mask):
                                    # Reduce predicted success probability
                                    bias_strength = bias_intensity * 0.3
                                    if np.random.rand() < bias_strength:
                                        y_pred_adjust = np.where(group_mask & (y == 1))
                                        if len(y_pred_adjust[0]) > 0:
                                            flip_count = int(len(y_pred_adjust[0]) * bias_strength)
                                            flip_indices = np.random.choice(y_pred_adjust[0], flip_count, replace=False)
                                            y[flip_indices] = 0

                # Apply resource inequality
                if resource_inequality > 0:
                    # Low-income students get worse features
                    low_income_mask = (student_groups == 0)
                    resource_factor = 1 - (resource_inequality * 0.5)
                    X[low_income_mask, 4] *= resource_factor  # Resource access
                    X[low_income_mask, 5] *= resource_factor  # School quality

                # Apply digital access inequality
                if digital_access < 1.0:
                    access_gap = 1 - digital_access
                    low_access_mask = np.random.rand(len(X)) < access_gap
                    # Reduce features for students with poor digital access
                    X[low_access_mask, 2] *= 0.8  # Extracurriculars
                    X[low_access_mask, 4] *= 0.7  # Resource access

                # Apply poisoning attack
                X_p, y_p, student_groups = simulate_data_poisoning(
                    X, y, poison_rate,
                    attack_type="label_flipping",
                    demographic_info=student_groups,
                    targeted=True
                )

                # Split data
                X_train, X_test, y_train, y_test, groups_train, groups_test, demo_train, demo_test = train_test_split(
                    X_p, y_p, student_groups, demographic_groups, test_size=0.3, random_state=42 + i
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
                metrics = calculate_education_metrics(
                    y_test, y_pred, groups_test, demo_test, X_test
                )

                # Calculate additional education-specific metrics
                # Success rate by income group
                low_income_mask = (groups_test == 0)
                high_income_mask = (groups_test == 1)

                success_rate_low = np.mean(y_test[low_income_mask]) if np.any(low_income_mask) else 0
                success_rate_high = np.mean(y_test[high_income_mask]) if np.any(high_income_mask) else 0
                success_gap = abs(success_rate_high - success_rate_low)

                # Prediction accuracy by group
                accuracy_low = accuracy_score(y_test[low_income_mask], y_pred[low_income_mask]) if np.any(
                    low_income_mask) else 0
                accuracy_high = accuracy_score(y_test[high_income_mask], y_pred[high_income_mask]) if np.any(
                    high_income_mask) else 0
                accuracy_gap = abs(accuracy_high - accuracy_low)

                # Opportunity score (measures equal opportunity)
                opportunity_score = 1.0 - min(1.0, success_gap + accuracy_gap)

                # Store results
                run_result = {
                    "run_id": i + 1,
                    "education_level": education_level,
                    "institution_type": institution_type,
                    "prediction_task": prediction_task,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1_score": metrics.get("f1_score", 0),
                    "equity_score": metrics.get("equity_score", 0.5),
                    "opportunity_score": opportunity_score,
                    "success_rate_low": success_rate_low,
                    "success_rate_high": success_rate_high,
                    "accuracy_low": accuracy_low,
                    "accuracy_high": accuracy_high,
                    "demographic_parity": metrics.get("demographic_parity_difference", 0),
                    "equal_opportunity": metrics.get("equal_opportunity_difference", 0),
                    "bias_intensity": bias_intensity,
                    "poison_rate": poison_rate,
                    "resource_inequality": resource_inequality,
                    "digital_access": digital_access,
                    "low_income_ratio": low_income_ratio,
                    "biases": ", ".join(selected_biases) if selected_biases else "None",
                    "support_services": support_services,
                    "attack_type": attack_type
                }

                st.session_state.edu_run_history.append(run_result)
                progress_bar.progress((i + 1) / n_runs)

        progress_bar.empty()

    # ── Display Results ───────────────────────────────
    if st.session_state.edu_run_history:
        df = pd.DataFrame(st.session_state.edu_run_history)

        # Summary Metrics Section
        st.markdown("## 📊 Educational Equity Dashboard")

        # Calculate averages
        avg_accuracy = df["accuracy"].mean()
        avg_equity = df["equity_score"].mean()
        avg_opportunity = df["opportunity_score"].mean()
        avg_success_gap = abs(df["success_rate_high"].mean() - df["success_rate_low"].mean())

        # Display metrics in styled cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="accuracy-metric">', unsafe_allow_html=True)
            st.metric(
                label="🎯 Prediction Accuracy",
                value=f"{avg_accuracy:.1%}",
                delta=None,
                help="Overall accuracy of the AI system"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="equity-metric">', unsafe_allow_html=True)
            st.metric(
                label="⚖️ Equity Score",
                value=f"{avg_equity:.2f}/1.0",
                delta=None,
                help="Fairness across student groups (1.0 = perfect equity)"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="access-metric">', unsafe_allow_html=True)
            st.metric(
                label="🚀 Opportunity Score",
                value=f"{avg_opportunity:.2f}/1.0",
                delta=None,
                help="Equal opportunity across socioeconomic groups"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="resource-metric">', unsafe_allow_html=True)
            st.metric(
                label="📈 Success Gap",
                value=f"{avg_success_gap:.1%}",
                delta_color="inverse",
                help="Difference in success rates between income groups"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Performance comparison by income group
        st.markdown("### 👨‍🎓 Performance by Income Group")

        col1, col2 = st.columns(2)

        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                name="Low-Income Students",
                x=df["run_id"],
                y=df["accuracy_low"] * 100,
                marker_color='#e74c3c'
            ))
            fig1.add_trace(go.Bar(
                name="Higher-Income Students",
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
                name="Low-Income Success",
                x=df["run_id"],
                y=df["success_rate_low"] * 100,
                marker_color='#3498db'
            ))
            fig2.add_trace(go.Bar(
                name="Higher-Income Success",
                x=df["run_id"],
                y=df["success_rate_high"] * 100,
                marker_color='#9b59b6'
            ))
            fig2.update_layout(
                title="Actual Success Rates by Income Group",
                barmode='group',
                yaxis_title="Success Rate (%)",
                xaxis_title="Simulation Run"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ── Detailed Analysis Tabs ───────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Overview", "⚖️ Equity Analysis",
                                          "🎓 Student Outcomes", "📋 Detailed Results"])

        with tab1:
            # Performance gauges
            fig = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=("Prediction Accuracy", "Equity Score", "Opportunity Score")
            )

            # Accuracy Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_accuracy * 100,
                title={'text': "Accuracy", 'font': {'size': 16}},
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

            # Equity Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_equity * 100,
                title={'text': "Equity", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ), row=1, col=2)

            # Opportunity Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_opportunity * 100,
                title={'text': "Opportunity", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkorange"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "darkgray"}
                    ]
                }
            ), row=1, col=3)

            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Trade-off scatter plot
            fig2 = px.scatter(
                df,
                x="equity_score",
                y="accuracy",
                size="opportunity_score",
                color="bias_intensity",
                hover_data=["biases", "resource_inequality", "support_services"],
                title="The Education Trilemma: Accuracy vs Equity vs Opportunity",
                labels={
                    "equity_score": "Equity Score",
                    "accuracy": "Prediction Accuracy",
                    "opportunity_score": "Opportunity Score",
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
            st.markdown("### ⚖️ Equity Gap Analysis")

            # Calculate equity gaps
            df["accuracy_gap"] = abs(df["accuracy_high"] - df["accuracy_low"])
            df["success_gap"] = abs(df["success_rate_high"] - df["success_rate_low"])

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    df,
                    x="run_id",
                    y=["accuracy_gap", "success_gap", "demographic_parity"],
                    barmode="group",
                    title="Equity Gaps Across Simulation Runs",
                    labels={"value": "Gap Size", "variable": "Metric"},
                    color_discrete_sequence=["#e74c3c", "#3498db", "#9b59b6"]
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Equity impact factors
                factors = pd.DataFrame({
                    "Factor": ["Bias Intensity", "Resource Inequality", "Digital Access", "Support Services"],
                    "Impact on Equity": [
                        -df["bias_intensity"].mean() * 0.8,
                        -df["resource_inequality"].mean() * 0.7,
                        df["digital_access"].mean() * 0.5,  # Positive impact
                        0.3 if support_services in ["Comprehensive", "Excellent"] else
                        0.1 if support_services == "Adequate" else -0.2
                    ]
                })

                fig2 = px.bar(
                    factors,
                    x="Factor",
                    y="Impact on Equity",
                    title="Factors Affecting Educational Equity",
                    color="Impact on Equity",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Equity recommendations
            if avg_equity < 0.7:
                st.markdown("""
                <div class="warning-banner">
                    <strong>⚠️ SIGNIFICANT EQUITY GAPS DETECTED</strong><br>
                    <strong>Recommended Interventions:</strong>
                    <ul style="margin-bottom:0;">
                        <li><strong>Blind Admissions:</strong> Remove identifying information during initial review</li>
                        <li><strong>Contextual Review:</strong> Consider socioeconomic context in evaluations</li>
                        <li><strong>Bias Audits:</strong> Regular audits of algorithmic systems</li>
                        <li><strong>Diverse Training Data:</strong> Ensure representation in training data</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("""
                **✅ GOOD EQUITY ACHIEVED**

                **Maintenance Actions:**
                1. Continue monitoring performance by demographic groups
                2. Regular bias audits of educational algorithms
                3. Engage with student groups for feedback
                4. Update models as educational contexts evolve
                """)

        with tab3:
            # Student outcomes analysis
            st.markdown("### 🎓 Student Success Analysis")

            # Simulated student outcomes for visualization
            np.random.seed(42)

            # Create sample data for different student groups
            groups = ["Low-Income", "First-Generation", "Underrepresented", "General"]
            success_rates = [
                np.random.uniform(0.3, 0.6) * (1 - bias_intensity * 0.3),
                np.random.uniform(0.4, 0.7) * (1 - bias_intensity * 0.2),
                np.random.uniform(0.35, 0.65) * (1 - bias_intensity * 0.4),
                np.random.uniform(0.6, 0.9)
            ]

            prediction_accuracy = [
                np.random.uniform(0.65, 0.85) * (1 - bias_intensity * 0.2),
                np.random.uniform(0.7, 0.9) * (1 - bias_intensity * 0.1),
                np.random.uniform(0.6, 0.8) * (1 - bias_intensity * 0.3),
                np.random.uniform(0.8, 0.95)
            ]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Actual Success Rate",
                x=groups,
                y=success_rates,
                marker_color='#3498db',
                text=[f"{rate:.1%}" for rate in success_rates],
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name="Prediction Accuracy",
                x=groups,
                y=prediction_accuracy,
                marker_color='#2ecc71',
                text=[f"{acc:.1%}" for acc in prediction_accuracy],
                textposition='auto'
            ))

            fig.update_layout(
                title="Student Outcomes by Demographic Group",
                yaxis_title="Rate",
                barmode='group',
                yaxis_range=[0, 1],
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Resource allocation visualization
            st.markdown("#### 📚 Resource Allocation Impact")

            resources = ["Tutoring", "Tech Access", "Counseling", "Extracurriculars", "Test Prep"]
            low_income_allocation = [0.3, 0.4, 0.5, 0.2, 0.3]
            high_income_allocation = [0.7, 0.9, 0.6, 0.8, 0.9]

            # Apply resource inequality factor
            low_income_allocation = [x * (1 - resource_inequality) for x in low_income_allocation]

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
                title="Resource Access by Income Group",
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
                    "equity_score": "{:.2f}",
                    "opportunity_score": "{:.2f}",
                    "success_rate_low": "{:.1%}",
                    "success_rate_high": "{:.1%}",
                    "accuracy_low": "{:.1%}",
                    "accuracy_high": "{:.1%}",
                    "demographic_parity": "{:.3f}",
                    "equal_opportunity": "{:.3f}"
                }).background_gradient(subset=["accuracy"], cmap="Blues")
                .background_gradient(subset=["equity_score"], cmap="RdYlGn")
                .background_gradient(subset=["success_rate_low"], cmap="Greens")
            )

            # Download options
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    f"gags_education_{education_level.lower().replace(' ', '_')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col2:
                # Export configuration
                config_data = {
                    "education_level": education_level,
                    "institution_type": institution_type,
                    "prediction_task": prediction_task,
                    "selected_biases": selected_biases,
                    "bias_intensity": bias_intensity,
                    "attack_type": attack_type,
                    "poison_rate": poison_rate,
                    "low_income_ratio": low_income_ratio,
                    "demographic_diversity": demographic_diversity,
                    "resource_inequality": resource_inequality,
                    "digital_access": digital_access,
                    "support_services": support_services,
                    "average_accuracy": f"{avg_accuracy:.1%}",
                    "average_equity": f"{avg_equity:.2f}"
                }
                import json

                json_str = json.dumps(config_data, indent=2)
                st.download_button(
                    "📋 Export Configuration (JSON)",
                    json_str,
                    f"education_config_{education_level.lower().replace(' ', '_')}.json",
                    "application/json",
                    use_container_width=True
                )

        # ── Policy Recommendations ───────────────────────────────
        st.divider()

        # Generate actionable recommendations
        st.markdown("## 💡 Education Policy Recommendations")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.info("""
            **🏫 For Educational Institutions:**

            1. **Transparent Algorithms:** Publish algorithm criteria and validation studies
            2. **Human-in-the-Loop:** Ensure human review of algorithmic decisions
            3. **Bias Audits:** Regular third-party audits for fairness
            4. **Diverse Data:** Ensure training data represents all student groups
            5. **Student Consent:** Obtain informed consent for data use

            **👨‍🎓 For Student Support:**

            1. **Digital Equity:** Ensure all students have technology access
            2. **Bias Training:** Educate staff about algorithmic bias
            3. **Appeal Processes:** Clear process for challenging algorithmic decisions
            4. **Alternative Pathways:** Multiple ways to demonstrate potential
            """)

        with rec_col2:
            st.success("""
            **🤖 For AI Development:**

            1. **Fairness Constraints:** Build equity into algorithm design
            2. **Explainable AI:** Make decisions understandable to stakeholders
            3. **Continuous Monitoring:** Track performance across demographic groups
            4. **Adversarial Testing:** Test systems against various attack vectors
            5. **Stakeholder Involvement:** Include educators, students, and families in design

            **📊 For Policymakers:**

            1. **Regulation:** Develop standards for educational AI
            2. **Oversight:** Independent review boards for high-stakes systems
            3. **Funding:** Support research on AI fairness in education
            4. **Guidelines:** Best practices for ethical AI implementation
            """)

        st.caption(
            "⚠️ **Disclaimer:** This simulation is for educational purposes. Real educational AI systems require extensive testing and ethical review.")

else:
    # Welcome/Instruction state
    st.markdown("## 🎓 Welcome to Education Equity Simulation")

    st.markdown("""
    This module explores how AI impacts educational equity across admissions, grading, resource allocation, 
    and student support systems. You'll configure educational scenarios and analyze fairness across different student groups.
    """)

    # Quick start examples
    st.markdown("### 🚀 Quick Start Scenarios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="school-card">
            <h4>🏫 University Admissions</h4>
            <p>Explore fairness in selective admissions:</p>
            <ul>
                <li>High-stakes prediction task</li>
                <li>Socioeconomic bias focus</li>
                <li>Comprehensive evaluation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="school-card">
            <h4>📚 K-12 Resource Allocation</h4>
            <p>Analyze equity in school resources:</p>
            <ul>
                <li>Resource inequality focus</li>
                <li>Digital access considerations</li>
                <li>Support services impact</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="school-card">
            <h4>💻 Online Learning Platforms</h4>
            <p>Study fairness in edtech:</p>
            <ul>
                <li>Digital divide considerations</li>
                <li>Personalized learning algorithms</li>
                <li>Accessibility focus</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # How to use guide
    with st.expander("📖 How to Use This Simulation", expanded=True):
        st.markdown("""
        1. **Select educational context** (level and institution type)
        2. **Configure biases** affecting educational algorithms
        3. **Set student demographics** (income levels, diversity)
        4. **Adjust resource allocation** factors
        5. **Configure adversarial attacks** on educational data
        6. **Run multiple simulations** to see statistical trends
        7. **Analyze trade-offs** between accuracy and equity
        8. **Explore recommendations** for policy and practice

        **Key Metrics to Watch:**
        - **Accuracy:** Overall prediction performance
        - **Equity Score:** Fairness across student groups (0-1)
        - **Opportunity Score:** Equal opportunity across socioeconomic groups
        - **Success Gap:** Difference in outcomes between income groups
        - **Performance Gaps:** Differences in accuracy by demographic group
        """)

    # Real-world context
    st.warning("""
    **Real-World Context:**

    Educational AI faces unique ethical challenges:
    - **High-Stakes Decisions:** Admissions, scholarships, and tracking
    - **Data Privacy:** Sensitive student information
    - **Digital Divide:** Unequal access to technology
    - **Transparency:** Need for explainable decisions
    - **Long-term Impact:** Educational decisions affect lifelong opportunities

    Responsible AI in education requires careful consideration of these factors while ensuring benefits reach all students.
    """)

# Footer
st.divider()
st.caption("🎓 Education Equity Simulation • GAGS Framework • v2.0 • Empowering Every Learner Through Fair AI")