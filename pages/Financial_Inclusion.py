import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pandas.core.methods.selectn import SelectNSeries
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
    page_title="Financial Inclusion • GAGS",
    layout="wide",
    page_icon="💰"
)

# Custom CSS for financial-themed styling
st.markdown("""
<style>
    .finance-header {
        background: linear-gradient(135deg, #0a2e36 0%, #27ae60 50%, #219653 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        border-left: 6px solid #f1c40f;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .approval-metric {
        background: linear-gradient(135deg, #27ae60 0%, #219653 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #27ae60;
    }
    .fairness-metric {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #3498db;
    }
    .inclusion-metric {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #9b59b6;
    }
    .risk-metric {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #e74c3c;
    }
    .stButton>button {
        background: linear-gradient(135deg, #27ae60 0%, #219653 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s;
        border: 2px solid #27ae60;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(39, 174, 96, 0.4);
    }
    .bank-card {
        background: #e8f8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
        color: #0a2e36;
    }
    .bias-tag-finance {
        display: inline-block;
        background: #e74c3c;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: bold;
    }
    .customer-group {
        display: inline-block;
        background: #3498db;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .credit-badge {
        background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: #0a2e36;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .warning-banner-finance {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        color: #856404;
        margin: 1rem 0;
    }
    .regulatory-tag {
        display: inline-block;
        background: #34495e;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Header Section
# ───────────────────────────────────────────────
st.markdown("""
<div class="finance-header">
    <h1 style="margin:0; color:white; font-size:2.5rem;">💰 Financial Inclusion & Credit Scoring Fairness</h1>
    <p style="margin:0; opacity:0.9; font-size:1.2rem; margin-top:0.5rem;">
        Analyze how <strong>algorithmic bias, fairness interventions, and regulation</strong> impact access to financial services
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: #e8f8f0; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #27ae60;">
    <p style="margin:0; color:#0a2e36;">
        <strong>🏦 Mission:</strong> AI in financial services can expand access but risks perpetuating discrimination. 
        This simulation explores fairness in credit scoring, loan approvals, and access to banking services across demographic groups.
    </p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Session State Initialization
# ───────────────────────────────────────────────
if "finance_run_history" not in st.session_state:
    st.session_state.finance_run_history = []
if "customer_groups" not in st.session_state:
    st.session_state.customer_groups = {}
if "loan_results" not in st.session_state:
    st.session_state.loan_results = {}

# ───────────────────────────────────────────────
# Sidebar Controls
# ───────────────────────────────────────────────
with st.sidebar:
    # Header with icon
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h2 style="color:#27ae60; margin:0;">⚙️ Financial System Configuration</h2>
        <p style="color:#7f8c8d; font-size:0.9rem;">Configure your financial inclusion simulation</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Financial Context Selection
    st.subheader("🏦 Financial Context")
    country_income_level = st.selectbox(
        "Select Country Income Level",
        ["Low-Income Country", "Lower-Middle Income", "Upper-Middle Income", "High-Income Country"],
        help="Income level affects baseline financial inclusion"
    )

    # Country metadata based on income level
    country_metadata = {
        "Low-Income Country": {"financial_inclusion": 0.35, "regulatory_capacity": 0.3, "digital_infrastructure": 0.2},
        "Lower-Middle Income": {"financial_inclusion": 0.55, "regulatory_capacity": 0.5, "digital_infrastructure": 0.4},
        "Upper-Middle Income": {"financial_inclusion": 0.75, "regulatory_capacity": 0.7, "digital_infrastructure": 0.6},
        "High-Income Country": {"financial_inclusion": 0.90, "regulatory_capacity": 0.85, "digital_infrastructure": 0.9}
    }

    selected_metadata = country_metadata[country_income_level]

    # Institution Type
    institution_type = st.selectbox(
        "Financial Institution Type",
        ["Traditional Bank", "Digital Bank", "Microfinance", "Credit Union",
         "Fintech Platform", "Payday Lender"],
        help="Type of financial institution"
    )

    st.divider()

    # Bias Configuration
    st.subheader("🎭 Bias Configuration")

    selected_biases = st.multiselect(
        "**Select Bias Types**",
        options=["historical_redlining", "proxy_discrimination", "group_disparity",
                 "feature_correlation", "measurement_bias", "sample_selection"],
        default=["historical_redlining", "proxy_discrimination", "group_disparity"],
        help="Biases affecting credit decisions",
        format_func=lambda x: f"🏘️ {x}" if x == "historical_redlining" else
        f"🔍 {x}" if x == "proxy_discrimination" else
        f"👥 {x}" if x == "group_disparity" else
        f"📊 {x}" if x == "feature_correlation" else
        f"📏 {x}" if x == "measurement_bias" else
        f"🎯 {x}" if x == "sample_selection" else
        str(x)
    )

    # Display selected biases as tags
    if selected_biases:
        tags_html = "".join([f'<span class="bias-tag-finance">{bias}</span>' for bias in selected_biases])
        st.markdown(f"**Active Biases:**<br>{tags_html}", unsafe_allow_html=True)

    bias_intensity = st.slider(
        "**Bias Intensity**",
        0.0, simulation_config.MAX_BIAS_FACTOR, 0.3, 0.05,
        help="Strength of bias in credit algorithms"
    )

    st.divider()

    # Attack Configuration
    st.subheader("⚠️ Adversarial Attacks")

    attack_type = st.selectbox(
        "**Attack Type**",
        ["Income Inflation", "Employment Forgery", "Credit History Manipulation",
         "Identity Theft", "Data Poisoning", "Sybil Attacks"],
        help="Type of attack on financial data"
    )

    poison_rate = st.slider(
        "**Attack Strength**",
        0.0, 0.5, 0.05, 0.01,
        format="%.2f",
        help="Proportion of financial data corrupted"
    )

    st.divider()

    # Customer Demographics
    st.subheader("👨‍👩‍👧‍👦 Customer Demographics")

    # Income distribution
    low_income_ratio = st.slider(
        "**Low-Income Customers**",
        0.0, 1.0, 0.5, 0.05,
        help="Percentage of customers from low-income backgrounds"
    )

    # Demographic groups
    minority_ratio = st.slider(
        "**Minority Representation**",
        0.0, 1.0, 0.3, 0.05,
        help="Percentage of customers from minority groups"
    )

    # Rural/Urban distribution
    rural_ratio = st.slider(
        "**Rural Customers**",
        0.0, 1.0, 0.4, 0.05,
        help="Percentage of customers from rural areas"
    )

    st.divider()

    # Credit Scoring Factors
    st.subheader("💳 Credit Scoring Configuration")

    traditional_data_weight = st.slider(
        "**Traditional Data Weight**",
        0.0, 1.0, 0.7, 0.05,
        help="Weight given to traditional credit data vs alternative data"
    )

    alternative_data_use = st.select_slider(
        "**Alternative Data Usage**",
        options=["None", "Limited", "Moderate", "Extensive", "Primary"],
        value="Moderate"
    )

    regulatory_compliance = st.slider(
        "**Regulatory Compliance Strictness**",
        0.0, 1.0, 0.6,
        help="0 = Lax regulation, 1 = Strict anti-discrimination enforcement"
    )

    st.divider()

    # Simulation Configuration
    st.subheader("📊 Simulation Parameters")

    sample_size = st.number_input(
        "**Number of Customer Records**",
        1000, 100000, settings.DEFAULT_N_SAMPLES, step=1000
    )

    n_runs = st.slider(
        "**Number of Simulation Runs**",
        1, 10, 3,
        help="More runs provide more reliable statistics"
    )

    prediction_task = st.selectbox(
        "**Prediction Task**",
        ["Loan Default Risk", "Credit Approval", "Interest Rate Tier",
         "Credit Limit Assignment", "Financial Inclusion Likelihood"],
        help="What the AI system is predicting"
    )

    st.divider()

    # Run Button
    col_run, col_reset = st.columns(2)
    with col_run:
        run_button = st.button(
            "💰 **Run**",
            type="primary",
            use_container_width=True
        )
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.finance_run_history = []
            st.session_state.customer_groups = {}
            st.rerun()


# ───────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────
def generate_financial_data(n_samples, n_features=12, low_income_ratio=0.5, minority_ratio=0.3):
    """Generate realistic financial data with customer demographics."""
    # Get base synthetic data
    X, y_base, demo_info = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features
    )

    # Add financial features
    # Feature 0: Income (log-normal distribution)
    X[:, 0] = np.random.lognormal(mean=10.5, sigma=0.6, size=n_samples)

    # Feature 1: Credit score (FICO-like, 300-850)
    X[:, 1] = np.random.normal(650, 100, n_samples)
    X[:, 1] = np.clip(X[:, 1], 300, 850)

    # Feature 2: Debt-to-income ratio (percentage)
    X[:, 2] = np.random.exponential(0.3, n_samples) * 100
    X[:, 2] = np.clip(X[:, 2], 0, 100)

    # Feature 3: Employment length (years)
    X[:, 3] = np.random.exponential(5, n_samples)
    X[:, 3] = np.clip(X[:, 3], 0, 40)

    # Feature 4: Savings/Assets (log-normal)
    X[:, 4] = np.random.lognormal(mean=9, sigma=1.2, size=n_samples)

    # Feature 5: Age
    X[:, 5] = np.random.normal(45, 15, n_samples)
    X[:, 5] = np.clip(X[:, 5], 18, 85)

    # Feature 6: Geographic location (0-1, higher = urban)
    X[:, 6] = np.random.beta(2, 2, n_samples)

    # Feature 7: Previous banking relationship (years)
    X[:, 7] = np.random.exponential(3, n_samples)

    # Feature 8: Number of credit inquiries (last year)
    X[:, 8] = np.random.poisson(2, n_samples)

    # Feature 9: Payment delinquencies (count)
    X[:, 9] = np.random.poisson(0.5, n_samples)

    # Feature 10: Education level (0-1, higher = more education)
    X[:, 10] = np.random.beta(3, 2, n_samples)

    # Feature 11: Mobile phone usage (proxy for digital access)
    X[:, 11] = np.random.beta(5, 2, n_samples)

    # Create customer groups (0 = low-income, 1 = middle/high income)
    n_low_income = int(n_samples * low_income_ratio)
    customer_groups = np.zeros(n_samples)
    customer_groups[n_low_income:] = 1

    # Add minority status
    n_minority = int(n_samples * minority_ratio)
    minority_status = np.zeros(n_samples)
    minority_indices = np.random.choice(n_samples, n_minority, replace=False)
    minority_status[minority_indices] = 1

    # Generate default probability (for classification)
    # Base on financial factors, but introduce bias factors
    income_score = (np.log(X[:, 0]) - 10) * 0.3  # Log income contribution
    credit_score = (X[:, 1] - 300) / 550 * 0.4  # Credit score contribution
    debt_penalty = -X[:, 2] / 100 * 0.2  # Debt-to-income penalty
    employment_score = X[:, 3] / 40 * 0.1  # Employment history

    # Bias factor based on customer groups
    bias_factor = np.zeros(n_samples)
    bias_factor[customer_groups == 0] -= bias_intensity * 0.3  # Penalize low-income
    bias_factor[minority_status == 1] -= bias_intensity * 0.2  # Penalize minorities

    default_prob = income_score + credit_score + debt_penalty + employment_score + bias_factor

    # Add noise
    default_prob += np.random.normal(0, 0.15, n_samples)

    # Create binary labels (1 = will default, 0 = will not default)
    # For loan approval, we want to approve those who won't default
    # So y = 0 means good credit (approve), y = 1 means bad credit (deny)
    default_threshold = np.percentile(default_prob, 30)  # Top 30% most risky default
    y = (default_prob > default_threshold).astype(int)

    return X, y, customer_groups, minority_status


def calculate_financial_metrics(y_true, y_pred, customer_groups, minority_status, X_features=None):
    """Calculate comprehensive financial inclusion metrics."""
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

    # Approval rate metrics
    # In credit scoring: y_pred = 0 means approve, y_pred = 1 means deny
    approval_rate = np.mean(y_pred == 0)
    metrics["approval_rate"] = approval_rate

    # Group-based fairness metrics (income groups)
    group_metrics = {}
    unique_groups = np.unique(customer_groups)

    for group in unique_groups:
        mask = (customer_groups == group)
        if np.sum(mask) == 0:
            continue

        group_true = y_true[mask]
        group_pred = y_pred[mask]

        group_accuracy = accuracy_score(group_true, group_pred)
        group_precision = precision_score(group_true, group_pred, zero_division=0)
        group_recall = recall_score(group_true, group_pred, zero_division=0)

        # Approval rates (for credit scenarios)
        group_approval_rate = np.mean(group_pred == 0)

        # False positive/negative rates (from perspective of good credit)
        # FP: Good customer incorrectly denied (bad prediction)
        # FN: Bad customer incorrectly approved (bad prediction)
        group_fp = np.sum((group_pred == 1) & (group_true == 0))  # Good customers denied
        group_fn = np.sum((group_pred == 0) & (group_true == 1))  # Bad customers approved
        group_tn = np.sum((group_pred == 0) & (group_true == 0))  # Good customers approved
        group_tp = np.sum((group_pred == 1) & (group_true == 1))  # Bad customers denied

        group_fpr = group_fp / (group_fp + group_tn) if (group_fp + group_tn) > 0 else 0
        group_fnr = group_fn / (group_fn + group_tp) if (group_fn + group_tp) > 0 else 0

        group_metrics[group] = {
            "accuracy": group_accuracy,
            "precision": group_precision,
            "recall": group_recall,
            "approval_rate": group_approval_rate,
            "fpr": group_fpr,  # False denial rate (type I error)
            "fnr": group_fnr,  # False approval rate (type II error)
            "sample_size": np.sum(mask)
        }

    # Calculate fairness disparities
    if len(group_metrics) >= 2:
        # Demographic parity difference (approval rate disparity)
        approval_rates = [metrics["approval_rate"] for metrics in group_metrics.values()]
        demographic_parity_diff = max(approval_rates) - min(approval_rates)

        # Equal opportunity difference (FNR disparity - false approval rate)
        fnrs = [metrics["fnr"] for metrics in group_metrics.values()]
        equal_opportunity_diff = max(fnrs) - min(fnrs)

        # Equalized odds difference (combined FPR and FNR)
        fprs = [metrics["fpr"] for metrics in group_metrics.values()]
        equalized_odds_diff = max(max(fprs) - min(fprs), max(fnrs) - min(fnrs))

        # Overall fairness score (0-1, higher is better)
        max_disparity = max(demographic_parity_diff, equal_opportunity_diff, equalized_odds_diff)
        fairness_score = 1.0 - min(1.0, max_disparity * 3)  # Scale appropriately

        metrics.update({
            "group_metrics": group_metrics,
            "demographic_parity_difference": demographic_parity_diff,
            "equal_opportunity_difference": equal_opportunity_diff,
            "equalized_odds_difference": equalized_odds_diff,
            "fairness_score": fairness_score,
            "max_disparity": max_disparity
        })

    # Inclusion metrics (if feature data available)
    if X_features is not None:
        # Calculate correlation between income and approvals
        income = X_features[:, 0]
        approval_mask = (y_pred == 0)
        if np.any(approval_mask):
            approved_income = income[approval_mask]
            rejected_income = income[~approval_mask]
            income_disparity = np.mean(approved_income) / np.mean(rejected_income) if np.mean(
                rejected_income) > 0 else 10
            metrics["income_disparity_ratio"] = income_disparity

        # Calculate geographic access correlation
        geographic_access = X_features[:, 6]  # Urban/rural score
        if np.any(approval_mask):
            approved_access = geographic_access[approval_mask]
            rejected_access = geographic_access[~approval_mask]
            access_disparity = np.mean(approved_access) - np.mean(rejected_access)
            metrics["access_disparity"] = access_disparity

    return metrics


# ───────────────────────────────────────────────
# Main Content Area
# ───────────────────────────────────────────────
if run_button or st.session_state.finance_run_history:

    if run_button:
        # Clear previous results
        st.session_state.finance_run_history = []

        # Run simulations
        progress_bar = st.progress(0)

        for i in range(n_runs):
            with st.spinner(f"Running financial simulation {i + 1}/{n_runs}..."):

                # Generate financial data
                X, y, customer_groups, minority_status = generate_financial_data(
                    n_samples=sample_size,
                    low_income_ratio=low_income_ratio,
                    minority_ratio=minority_ratio
                )

                # Apply selected biases
                for bias_type in selected_biases:
                    if bias_type == "historical_redlining":
                        # Apply geographic bias (similar to historical redlining)
                        rural_mask = (X[:, 6] < 0.3)  # Rural customers
                        # Penalize rural customers' credit scores
                        penalty_strength = bias_intensity * 50  # Points off credit score
                        X[rural_mask, 1] = np.maximum(300, X[rural_mask, 1] - penalty_strength)

                    elif bias_type == "proxy_discrimination":
                        # Use zip code/neighborhood as proxy for race
                        minority_mask = (minority_status == 1)
                        if np.any(minority_mask):
                            # Add noise to minority customers' features
                            noise_scale = bias_intensity * 0.5
                            X[minority_mask] += np.random.normal(0, noise_scale, (np.sum(minority_mask), X.shape[1]))

                    elif bias_type == "group_disparity":
                        # Group-based discrimination
                        low_income_mask = (customer_groups == 0)
                        # Reduce predicted creditworthiness for low-income
                        bias_strength = bias_intensity * 0.3
                        if np.random.rand() < bias_strength:
                            # Flip some good credit decisions to bad
                            good_credit_mask = low_income_mask & (y == 0)
                            if np.any(good_credit_mask):
                                flip_count = int(np.sum(good_credit_mask) * bias_strength)
                                flip_indices = np.random.choice(np.where(good_credit_mask)[0], flip_count,
                                                                replace=False)
                                y[flip_indices] = 1  # Mark as default

                # Apply alternative data weighting
                alt_data_weight_map = {"None": 0.0, "Limited": 0.2, "Moderate": 0.5, "Extensive": 0.8, "Primary": 1.0}
                alt_weight = alt_data_weight_map.get(alternative_data_use, 0.5)

                # Weight features differently based on traditional vs alternative
                if alt_weight > 0:
                    # Alternative data features (mobile usage, etc.)
                    alt_features = [11]  # Mobile usage as alternative data
                    for feat_idx in alt_features:
                        X[:, feat_idx] *= (1 + alt_weight * 0.5)  # Boost alternative data

                # Apply regulatory compliance effect
                if regulatory_compliance > 0.7:
                    # Strict regulation reduces some biases
                    compliance_factor = 1 - (regulatory_compliance - 0.7) * 0.5
                    # Reduce bias impact
                    bias_intensity_adj = bias_intensity * compliance_factor

                # Apply poisoning attack
                X_p, y_p, customer_groups = simulate_data_poisoning(
                    X, y, poison_rate * (1 + (1 - selected_metadata["regulatory_capacity"])),
                    attack_type="label_flipping",
                    demographic_info=customer_groups,
                    targeted=True
                )

                # Split data
                X_train, X_test, y_train, y_test, groups_train, groups_test, minority_train, minority_test = train_test_split(
                    X_p, y_p, customer_groups, minority_status, test_size=0.3, random_state=42 + i
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
                metrics = calculate_financial_metrics(
                    y_test, y_pred, groups_test, minority_test, X_test
                )

                # Calculate additional financial-specific metrics
                # Approval rate by income group
                low_income_mask = (groups_test == 0)
                high_income_mask = (groups_test == 1)

                approval_rate_low = np.mean(y_pred[low_income_mask] == 0) if np.any(low_income_mask) else 0
                approval_rate_high = np.mean(y_pred[high_income_mask] == 0) if np.any(high_income_mask) else 0
                approval_gap = abs(approval_rate_high - approval_rate_low)

                # Default rate by group (actual risk)
                default_rate_low = np.mean(y_test[low_income_mask] == 1) if np.any(low_income_mask) else 0
                default_rate_high = np.mean(y_test[high_income_mask] == 1) if np.any(high_income_mask) else 0

                # Prediction accuracy by group
                accuracy_low = accuracy_score(y_test[low_income_mask], y_pred[low_income_mask]) if np.any(
                    low_income_mask) else 0
                accuracy_high = accuracy_score(y_test[high_income_mask], y_pred[high_income_mask]) if np.any(
                    high_income_mask) else 0
                accuracy_gap = abs(accuracy_high - accuracy_low)

                # Inclusion score (measures financial inclusion)
                inclusion_score = 1.0 - min(1.0, approval_gap + accuracy_gap)

                # Store results
                run_result = {
                    "run_id": i + 1,
                    "country_income": country_income_level,
                    "institution_type": institution_type,
                    "prediction_task": prediction_task,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1_score": metrics.get("f1_score", 0),
                    "approval_rate": metrics.get("approval_rate", 0),
                    "fairness_score": metrics.get("fairness_score", 0.5),
                    "inclusion_score": inclusion_score,
                    "approval_rate_low": approval_rate_low,
                    "approval_rate_high": approval_rate_high,
                    "default_rate_low": default_rate_low,
                    "default_rate_high": default_rate_high,
                    "accuracy_low": accuracy_low,
                    "accuracy_high": accuracy_high,
                    "demographic_parity": metrics.get("demographic_parity_difference", 0),
                    "equal_opportunity": metrics.get("equal_opportunity_difference", 0),
                    "bias_intensity": bias_intensity,
                    "poison_rate": poison_rate,
                    "regulatory_compliance": regulatory_compliance,
                    "traditional_data_weight": traditional_data_weight,
                    "alternative_data_use": alternative_data_use,
                    "low_income_ratio": low_income_ratio,
                    "minority_ratio": minority_ratio,
                    "biases": ", ".join(selected_biases) if selected_biases else "None",
                    "attack_type": attack_type,
                    "financial_inclusion": selected_metadata["financial_inclusion"]
                }

                st.session_state.finance_run_history.append(run_result)
                progress_bar.progress((i + 1) / n_runs)

        progress_bar.empty()

    # ── Display Results ───────────────────────────────
    if st.session_state.finance_run_history:
        df = pd.DataFrame(st.session_state.finance_run_history)

        # Summary Metrics Section
        st.markdown("## 📊 Financial Inclusion Dashboard")

        # Calculate averages
        avg_accuracy = df["accuracy"].mean()
        avg_fairness = df["fairness_score"].mean()
        avg_inclusion = df["inclusion_score"].mean()
        avg_approval_gap = abs(df["approval_rate_high"].mean() - df["approval_rate_low"].mean())

        # Display metrics in styled cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="approval-metric">', unsafe_allow_html=True)
            overall_approval = df["approval_rate"].mean()
            st.metric(
                label="✅ Overall Approval Rate",
                value=f"{overall_approval:.1%}",
                delta=None,
                help="Overall credit approval rate across all customers"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="fairness-metric">', unsafe_allow_html=True)
            st.metric(
                label="⚖️ Fairness Score",
                value=f"{avg_fairness:.2f}/1.0",
                delta=None,
                help="Algorithmic fairness across demographic groups"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="inclusion-metric">', unsafe_allow_html=True)
            st.metric(
                label="🌍 Inclusion Score",
                value=f"{avg_inclusion:.2f}/1.0",
                delta=None,
                help="Financial inclusion across income groups"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="risk-metric">', unsafe_allow_html=True)
            st.metric(
                label="📉 Approval Gap",
                value=f"{avg_approval_gap:.1%}",
                delta_color="inverse",
                help="Difference in approval rates between income groups"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Performance comparison by income group
        st.markdown("### 👥 Credit Access by Income Group")

        col1, col2 = st.columns(2)

        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                name="Low-Income Customers",
                x=df["run_id"],
                y=df["approval_rate_low"] * 100,
                marker_color='#e74c3c'
            ))
            fig1.add_trace(go.Bar(
                name="Higher-Income Customers",
                x=df["run_id"],
                y=df["approval_rate_high"] * 100,
                marker_color='#2ecc71'
            ))
            fig1.update_layout(
                title="Approval Rates by Income Group",
                barmode='group',
                yaxis_title="Approval Rate (%)",
                xaxis_title="Simulation Run"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="Low-Income Default Rate",
                x=df["run_id"],
                y=df["default_rate_low"] * 100,
                marker_color='#3498db'
            ))
            fig2.add_trace(go.Bar(
                name="Higher-Income Default Rate",
                x=df["run_id"],
                y=df["default_rate_high"] * 100,
                marker_color='#9b59b6'
            ))
            fig2.update_layout(
                title="Actual Default Rates by Income Group",
                barmode='group',
                yaxis_title="Default Rate (%)",
                xaxis_title="Simulation Run"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ── Detailed Analysis Tabs ───────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Overview", "⚖️ Fairness Analysis",
                                          "💰 Economic Impact", "📋 Detailed Results"])

        with tab1:
            # Performance gauges
            fig = make_subplots(
                rows=1, cols=4,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'},
                        {'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=("Accuracy", "Fairness", "Inclusion", "Approval Gap")
            )

            # Accuracy Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_accuracy * 100,
                title={'text': "Accuracy", 'font': {'size': 14}},
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
                        'value': 80
                    }
                }
            ), row=1, col=1)

            # Fairness Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_fairness * 100,
                title={'text': "Fairness", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
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

            # Inclusion Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_inclusion * 100,
                title={'text': "Inclusion", 'font': {'size': 14}},
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

            # Approval Gap Gauge (inverse - lower is better)
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=100 - (avg_approval_gap * 100),
                title={'text': "Parity", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "gray"},
                        {'range': [90, 100], 'color': "darkgray"}
                    ]
                }
            ), row=1, col=4)

            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Trade-off scatter plot
            fig2 = px.scatter(
                df,
                x="fairness_score",
                y="accuracy",
                size="inclusion_score",
                color="bias_intensity",
                hover_data=["biases", "regulatory_compliance", "alternative_data_use"],
                title="The Financial Inclusion Trilemma: Accuracy vs Fairness vs Inclusion",
                labels={
                    "fairness_score": "Fairness Score",
                    "accuracy": "Prediction Accuracy",
                    "inclusion_score": "Inclusion Score",
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
            # Fairness analysis
            st.markdown("### ⚖️ Fairness Gap Analysis")

            # Calculate fairness gaps
            df["approval_gap"] = abs(df["approval_rate_high"] - df["approval_rate_low"])
            df["accuracy_gap"] = abs(df["accuracy_high"] - df["accuracy_low"])

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    df,
                    x="run_id",
                    y=["approval_gap", "accuracy_gap", "demographic_parity"],
                    barmode="group",
                    title="Fairness Gaps Across Simulation Runs",
                    labels={"value": "Gap Size", "variable": "Metric"},
                    color_discrete_sequence=["#e74c3c", "#3498db", "#9b59b6"]
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Fairness impact factors
                factors = pd.DataFrame({
                    "Factor": ["Bias Intensity", "Regulatory Compliance", "Alternative Data", "Traditional Data"],
                    "Impact on Fairness": [
                        -df["bias_intensity"].mean() * 0.8,
                        df["regulatory_compliance"].mean() * 0.6,  # Positive impact
                        df["traditional_data_weight"].mean() * -0.5,  # Negative impact (if high)
                        0.3 if alternative_data_use in ["Extensive", "Primary"] else -0.2
                    ]
                })

                fig2 = px.bar(
                    factors,
                    x="Factor",
                    y="Impact on Fairness",
                    title="Factors Affecting Financial Fairness",
                    color="Impact on Fairness",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Regulatory compliance display
            st.markdown("#### 🏛️ Regulatory Impact Analysis")

            compliance_level = "Strict" if regulatory_compliance > 0.7 else \
                "Moderate" if regulatory_compliance > 0.4 else \
                    "Lax"

            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #34495e;">
                <h4 style="margin:0; color:#34495e;">Current Regulatory Setting: <span class="regulatory-tag">{compliance_level}</span></h4>
                <p style="margin:0.5rem 0; color:#2c3e50;">
                    <strong>Compliance Score:</strong> {regulatory_compliance:.1%}<br>
                    <strong>Impact on Bias:</strong> {1 - (regulatory_compliance / 2):.1%} reduction<br>
                    <strong>Financial Inclusion Boost:</strong> {regulatory_compliance * 0.3:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Fairness recommendations
            if avg_fairness < 0.7:
                st.markdown("""
                <div class="warning-banner-finance">
                    <strong>⚠️ SIGNIFICANT FAIRNESS GAPS DETECTED</strong><br>
                    <strong>Recommended Interventions:</strong>
                    <ul style="margin-bottom:0;">
                        <li><strong>Alternative Data:</strong> Incorporate utility payments, rental history</li>
                        <li><strong>Bias Audits:</strong> Regular third-party fairness assessments</li>
                        <li><strong>Explainable AI:</strong> Provide reasons for credit denials</li>
                        <li><strong>Redlining Prevention:</strong> Remove geographic proxies for race</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("""
                **✅ GOOD FAIRNESS ACHIEVED**

                **Maintenance Actions:**
                1. Continue monitoring approval rates by demographic groups
                2. Regular bias audits of credit algorithms
                3. Engage with community organizations for feedback
                4. Update models as economic conditions evolve
                """)

        with tab3:
            # Economic impact analysis
            st.markdown("### 💰 Economic Impact Analysis")

            # Simulate economic outcomes
            np.random.seed(42)

            # Create sample data for different customer segments
            segments = ["Low-Income", "Minority", "Rural", "Young Adults", "General"]
            current_access = [
                selected_metadata["financial_inclusion"] * 0.5,  # Low-income
                selected_metadata["financial_inclusion"] * 0.6,  # Minority
                selected_metadata["financial_inclusion"] * 0.4,  # Rural
                selected_metadata["financial_inclusion"] * 0.7,  # Young adults
                selected_metadata["financial_inclusion"]  # General
            ]

            potential_access = [
                min(1.0, current_access[0] + avg_inclusion * 0.3),
                min(1.0, current_access[1] + avg_inclusion * 0.25),
                min(1.0, current_access[2] + avg_inclusion * 0.35),
                min(1.0, current_access[3] + avg_inclusion * 0.2),
                min(1.0, current_access[4] + avg_inclusion * 0.1)
            ]

            # Economic impact per segment (in millions)
            population_share = [0.3, 0.2, 0.25, 0.15, 1.0]
            economic_impact = []
            for i in range(len(segments)):
                new_customers = (potential_access[i] - current_access[i]) * population_share[i] * 100  # In millions
                avg_loan_size = [1000, 2000, 1500, 3000, 5000][i]  # USD
                impact = new_customers * avg_loan_size * 2.5  # Multiplier effect
                economic_impact.append(impact)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Current Financial Inclusion",
                x=segments,
                y=current_access,
                marker_color='#3498db',
                text=[f"{rate:.0%}" for rate in current_access],
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name="Potential with Fair AI",
                x=segments,
                y=potential_access,
                marker_color='#2ecc71',
                text=[f"{rate:.0%}" for rate in potential_access],
                textposition='auto'
            ))

            fig.update_layout(
                title="Financial Inclusion by Customer Segment",
                yaxis_title="Inclusion Rate",
                barmode='group',
                yaxis_range=[0, 1],
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Economic impact visualization
            st.markdown("#### 📊 Estimated Economic Impact")

            fig2 = go.Figure(data=[
                go.Bar(
                    name='Economic Impact ($M)',
                    x=segments,
                    y=economic_impact,
                    marker_color='#f1c40f',
                    text=[f"${impact / 1000000:.1f}M" for impact in economic_impact],
                    textposition='auto'
                )
            ])

            fig2.update_layout(
                title="Estimated Economic Impact of Improved Financial Inclusion",
                yaxis_title="Economic Impact (USD Millions)",
                showlegend=False
            )

            st.plotly_chart(fig2, use_container_width=True)

            # Total economic impact
            total_impact = sum(economic_impact)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #27ae60 0%, #219653 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; margin: 1rem 0;">
                <h3 style="margin:0; color:white;">Total Economic Impact</h3>
                <p style="font-size: 2rem; margin:0.5rem 0; font-weight:bold;">
                    ${total_impact / 1000000:,.1f} Million
                </p>
                <p style="margin:0; opacity:0.9;">
                    Estimated annual economic boost from improved financial inclusion
                </p>
            </div>
            """, unsafe_allow_html=True)

        with tab4:
            # Detailed results table
            st.dataframe(
                df.style.format({
                    "accuracy": "{:.1%}",
                    "precision": "{:.1%}",
                    "recall": "{:.1%}",
                    "f1_score": "{:.2f}",
                    "approval_rate": "{:.1%}",
                    "fairness_score": "{:.2f}",
                    "inclusion_score": "{:.2f}",
                    "approval_rate_low": "{:.1%}",
                    "approval_rate_high": "{:.1%}",
                    "default_rate_low": "{:.1%}",
                    "default_rate_high": "{:.1%}",
                    "accuracy_low": "{:.1%}",
                    "accuracy_high": "{:.1%}",
                    "demographic_parity": "{:.3f}",
                    "equal_opportunity": "{:.3f}"
                }).background_gradient(subset=["accuracy"], cmap="Blues")
                .background_gradient(subset=["fairness_score"], cmap="RdYlGn")
                .background_gradient(subset=["approval_rate_low"], cmap="Greens")
            )

            # Download options
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    f"gags_finance_{country_income_level.lower().replace(' ', '_')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col2:
                # Export configuration
                config_data = {
                    "country_income_level": country_income_level,
                    "institution_type": institution_type,
                    "prediction_task": prediction_task,
                    "selected_biases": selected_biases,
                    "bias_intensity": bias_intensity,
                    "attack_type": attack_type,
                    "poison_rate": poison_rate,
                    "low_income_ratio": low_income_ratio,
                    "minority_ratio": minority_ratio,
                    "regulatory_compliance": regulatory_compliance,
                    "traditional_data_weight": traditional_data_weight,
                    "alternative_data_use": alternative_data_use,
                    "average_accuracy": f"{avg_accuracy:.1%}",
                    "average_fairness": f"{avg_fairness:.2f}",
                    "average_inclusion": f"{avg_inclusion:.2f}",
                    "financial_inclusion": selected_metadata["financial_inclusion"]
                }
                import json

                json_str = json.dumps(config_data, indent=2)
                st.download_button(
                    "📋 Export Configuration (JSON)",
                    json_str,
                    f"finance_config_{country_income_level.lower().replace(' ', '_')}.json",
                    "application/json",
                    use_container_width=True
                )

        # ── Policy Recommendations ───────────────────────────────
        st.divider()

        # Generate actionable recommendations
        st.markdown("## 💡 Financial Inclusion Policy Recommendations")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.info("""
            **🏦 For Financial Institutions:**

            1. **Transparent Algorithms:** Publish credit criteria and validation studies
            2. **Human Review:** Ensure human oversight of algorithmic denials
            3. **Bias Audits:** Regular third-party fairness assessments
            4. **Alternative Data:** Incorporate non-traditional data sources
            5. **Financial Education:** Provide resources for credit building

            **👥 For Customer Protection:**

            1. **Explainable Denials:** Provide clear reasons for credit denials
            2. **Appeal Processes:** Clear process for challenging decisions
            3. **Data Privacy:** Protect sensitive financial information
            4. **Financial Counseling:** Support for credit improvement
            """)

        with rec_col2:
            st.success("""
            **🤖 For AI Development:**

            1. **Fairness Constraints:** Build equity into algorithm design
            2. **Proxy Detection:** Identify and remove discriminatory proxies
            3. **Continuous Monitoring:** Track performance across demographic groups
            4. **Adversarial Testing:** Test against manipulation and fraud
            5. **Stakeholder Involvement:** Include diverse communities in design

            **🏛️ For Regulators:**

            1. **Anti-discrimination Enforcement:** Strict enforcement of fair lending laws
            2. **Algorithmic Audits:** Mandatory fairness assessments for high-risk systems
            3. **Transparency Requirements:** Mandate disclosure of key factors in decisions
            4. **Sandbox Approaches:** Allow innovation while protecting consumers
            """)

        st.caption(
            "⚠️ **Disclaimer:** This simulation is for educational purposes. Real financial AI systems require extensive testing, regulatory compliance, and ethical review.")

else:
    # Welcome/Instruction state
    st.markdown("## 💰 Welcome to Financial Inclusion Simulation")

    st.markdown("""
    This module explores how AI impacts access to financial services across credit scoring, loan approvals, 
    and banking services. You'll configure financial scenarios and analyze fairness across different customer groups.
    """)

    # Quick start examples
    st.markdown("### 🚀 Quick Start Scenarios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="bank-card">
            <h4>🏦 Traditional Banking</h4>
            <p>Explore fairness in established banks:</p>
            <ul>
                <li>Traditional credit scoring</li>
                <li>Historical bias challenges</li>
                <li>Regulatory compliance focus</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="bank-card">
            <h4>📱 Fintech Innovation</h4>
            <p>Analyze new financial technologies:</p>
            <ul>
                <li>Alternative data usage</li>
                <li>Digital access considerations</li>
                <li>Innovation vs regulation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="bank-card">
            <h4>🌍 Emerging Markets</h4>
            <p>Study financial inclusion in developing economies:</p>
            <ul>
                <li>Mobile money systems</li>
                <li>Limited traditional data</li>
                <li>Infrastructure challenges</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # How to use guide
    with st.expander("📖 How to Use This Simulation", expanded=True):
        st.markdown("""
        1. **Select financial context** (country income level and institution type)
        2. **Configure biases** affecting credit algorithms (redlining, proxy discrimination)
        3. **Set customer demographics** (income levels, minority representation)
        4. **Adjust credit scoring factors** (traditional vs alternative data)
        5. **Configure regulatory environment** (compliance strictness)
        6. **Run multiple simulations** to see statistical trends
        7. **Analyze trade-offs** between accuracy, fairness, and inclusion
        8. **Explore economic impact** of improved financial inclusion
        9. **Review recommendations** for policy and practice

        **Key Metrics to Watch:**
        - **Approval Rate:** Overall credit approval percentage
        - **Fairness Score:** Algorithmic fairness across demographic groups (0-1)
        - **Inclusion Score:** Financial inclusion across income groups (0-1)
        - **Approval Gap:** Difference in approval rates between income groups
        - **Default Rates:** Actual risk by demographic group
        """)

    # Real-world context
    st.warning("""
    **Real-World Context:**

    Financial AI faces unique ethical challenges:
    - **High-Stakes Decisions:** Credit access affects housing, education, and opportunities
    - **Historical Biases:** Redlining and discriminatory practices create legacy effects
    - **Proxy Discrimination:** Using zip codes, shopping patterns as race proxies
    - **Transparency:** Need for explainable decisions (especially for denials)
    - **Regulatory Compliance:** Must follow fair lending laws (ECOA, FHA)

    Responsible AI in finance requires balancing innovation with consumer protection and anti-discrimination efforts.
    """)

# Footer
st.divider()
st.caption("💰 Financial Inclusion Simulation • GAGS Framework • v2.0 • Expanding Access Through Fair Financial AI")