# pages/5_⚖️_International_Judicial_Systems.py
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
# International Legal System Database
# ───────────────────────────────────────────────
LEGAL_SYSTEMS = {
    "Common Law (US/UK/Canada/Australia)": {
        "type": "common",
        "adversarial": True,
        "presumption_of_innocence": 0.9,
        "judicial_discretion": 0.8,
        "transparency": 0.7,
        "appeal_rights": 0.9,
        "jury_trial": 0.8,
        "precedent_based": 0.9,
        "bias_factors": ["historical", "demographic", "socioeconomic"],
        "key_metrics": ["procedural_fairness", "equal_protection", "due_process"],
        "color": "#3498db"
    },
    "Civil Law (France/Germany/Japan/Brazil)": {
        "type": "civil",
        "adversarial": False,
        "presumption_of_innocence": 0.8,
        "judicial_discretion": 0.6,
        "transparency": 0.6,
        "appeal_rights": 0.8,
        "jury_trial": 0.3,
        "precedent_based": 0.3,
        "bias_factors": ["bureaucratic", "class", "geographic"],
        "key_metrics": ["legal_certainty", "codified_rights", "systematic_fairness"],
        "color": "#2ecc71"
    },
    "Islamic Law (Saudi Arabia/Iran/Pakistan)": {
        "type": "islamic",
        "adversarial": False,
        "presumption_of_innocence": 0.5,
        "judicial_discretion": 0.7,
        "transparency": 0.4,
        "appeal_rights": 0.6,
        "jury_trial": 0.1,
        "precedent_based": 0.4,
        "bias_factors": ["religious", "gender", "tribal"],
        "key_metrics": ["sharia_compliance", "community_justice", "religious_fairness"],
        "color": "#e74c3c"
    },
    "Socialist Law (China/Vietnam/Cuba)": {
        "type": "socialist",
        "adversarial": False,
        "presumption_of_innocence": 0.4,
        "judicial_discretion": 0.5,
        "transparency": 0.3,
        "appeal_rights": 0.5,
        "jury_trial": 0.2,
        "precedent_based": 0.2,
        "bias_factors": ["political", "ideological", "state_interest"],
        "key_metrics": ["social_stability", "state_security", "collective_justice"],
        "color": "#9b59b6"
    },
    "Customary/Traditional (Various Africa/Indigenous)": {
        "type": "customary",
        "adversarial": False,
        "presumption_of_innocence": 0.6,
        "judicial_discretion": 0.9,
        "transparency": 0.5,
        "appeal_rights": 0.4,
        "jury_trial": 0.1,
        "precedent_based": 0.7,
        "bias_factors": ["tribal", "elders", "community_norms"],
        "key_metrics": ["community_harmony", "restorative_justice", "cultural_appropriateness"],
        "color": "#f39c12"
    },
    "Mixed/Hybrid (South Africa/Israel/Philippines)": {
        "type": "hybrid",
        "adversarial": True,
        "presumption_of_innocence": 0.7,
        "judicial_discretion": 0.7,
        "transparency": 0.6,
        "appeal_rights": 0.7,
        "jury_trial": 0.5,
        "precedent_based": 0.6,
        "bias_factors": ["multiple_legal_traditions", "colonial", "linguistic"],
        "key_metrics": ["pluralistic_fairness", "legal_hybridity", "cross_cultural_justice"],
        "color": "#1abc9c"
    }
}

COUNTRIES = {
    "United States": {"system": "Common Law (US/UK/Canada/Australia)", "hdi": 0.926, "rule_of_law": 0.73,
                      "gdp_per_capita": 65000},
    "United Kingdom": {"system": "Common Law (US/UK/Canada/Australia)", "hdi": 0.932, "rule_of_law": 0.82,
                       "gdp_per_capita": 45000},
    "Germany": {"system": "Civil Law (France/Germany/Japan/Brazil)", "hdi": 0.947, "rule_of_law": 0.84,
                "gdp_per_capita": 52000},
    "Japan": {"system": "Civil Law (France/Germany/Japan/Brazil)", "hdi": 0.925, "rule_of_law": 0.79,
              "gdp_per_capita": 42000},
    "China": {"system": "Socialist Law (China/Vietnam/Cuba)", "hdi": 0.768, "rule_of_law": 0.48,
              "gdp_per_capita": 12000},
    "Saudi Arabia": {"system": "Islamic Law (Saudi Arabia/Iran/Pakistan)", "hdi": 0.854, "rule_of_law": 0.44,
                     "gdp_per_capita": 55000},
    "South Africa": {"system": "Mixed/Hybrid (South Africa/Israel/Philippines)", "hdi": 0.713, "rule_of_law": 0.58,
                     "gdp_per_capita": 13000},
    "India": {"system": "Common Law (US/UK/Canada/Australia)", "hdi": 0.645, "rule_of_law": 0.51,
              "gdp_per_capita": 7000},
    "Brazil": {"system": "Civil Law (France/Germany/Japan/Brazil)", "hdi": 0.765, "rule_of_law": 0.54,
               "gdp_per_capita": 15000},
    "Nigeria": {"system": "Mixed/Hybrid (South Africa/Israel/Philippines)", "hdi": 0.539, "rule_of_law": 0.42,
                "gdp_per_capita": 5000},
    "Russia": {"system": "Civil Law (France/Germany/Japan/Brazil)", "hdi": 0.824, "rule_of_law": 0.45,
               "gdp_per_capita": 28000},
    "Rwanda": {"system": "Customary/Traditional (Various Africa/Indigenous)", "hdi": 0.534, "rule_of_law": 0.55,
               "gdp_per_capita": 2300}
}

# ───────────────────────────────────────────────
# Page Configuration
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="International Judicial Systems • GAGS",
    layout="wide",
    page_icon="🌍⚖️"
)

# Custom CSS for international justice-themed styling
st.markdown("""
<style>
    .international-header {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 50%, #16213e 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        border-left: 6px solid #e94560;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    .global-metric {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #3498db;
    }
    .fairness-metric-intl {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #2ecc71;
    }
    .liberty-metric-intl {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #9b59b6;
    }
    .cultural-metric {
        background: linear-gradient(135deg, #e67e22 0%, #d35400 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #e67e22;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s;
        border: 2px solid #e94560;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.4);
    }
    .country-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--country-color);
        margin: 1rem 0;
        color: #2c3e50;
    }
    .legal-system-tag {
        display: inline-block;
        background: var(--system-color);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: bold;
    }
    .bias-tag-intl {
        display: inline-block;
        background: #e94560;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: bold;
    }
    .cultural-note {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f39c12;
        color: #856404;
        margin: 1rem 0;
    }
    .international-alert {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        color: #721c24;
        margin: 1rem 0;
    }
    .comparison-table {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Header Section
# ───────────────────────────────────────────────
st.markdown("""
<div class="international-header">
    <h1 style="margin:0; color:white; font-size:2.5rem;">🌍⚖️ International Judicial Systems Simulation</h1>
    <p style="margin:0; opacity:0.9; font-size:1.2rem; margin-top:0.5rem;">
        Compare how <strong>algorithmic fairness, legal traditions, and cultural contexts</strong> impact justice across countries
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cultural-note">
    <p style="margin:0; color:#856404; font-weight:bold;">
        🌐 <em>"Justice is contextual: what constitutes fairness varies across legal traditions, 
        cultures, and historical contexts. This simulation explores these variations."</em>
    </p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Session State Initialization
# ───────────────────────────────────────────────
if "intl_judicial_history" not in st.session_state:
    st.session_state.intl_judicial_history = []
if "country_comparisons" not in st.session_state:
    st.session_state.country_comparisons = {}
if "legal_tradition_analysis" not in st.session_state:
    st.session_state.legal_tradition_analysis = {}

# ───────────────────────────────────────────────
# Sidebar Controls
# ───────────────────────────────────────────────
with st.sidebar:
    # Header with icon
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h2 style="color:#0f3460; margin:0;">⚙️ Global Configuration</h2>
        <p style="color:#7f8c8d; font-size:0.9rem;">Configure your international judicial simulation</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Country Selection
    st.subheader("🌍 Country Selection")

    selected_countries = st.multiselect(
        "**Select Countries to Compare**",
        options=list(COUNTRIES.keys()),
        default=["United States", "Germany", "China", "South Africa", "India"],
        help="Select multiple countries for comparative analysis"
    )

    # Display selected countries with their legal systems
    if selected_countries:
        st.markdown("**Selected Countries:**")
        for country in selected_countries:
            system = COUNTRIES[country]["system"]
            color = LEGAL_SYSTEMS[system]["color"]
            st.markdown(f"""
            <div style="background:{color}20; padding:0.5rem; border-radius:8px; margin:0.2rem 0; border-left:3px solid {color};">
                <strong>{country}</strong> • <span style="font-size:0.9rem;">{system.split(' ')[0]}</span>
            </div>
            """, unsafe_allow_html=True)

    # Legal System Focus
    st.subheader("⚖️ Legal Tradition Focus")

    compare_systems = st.checkbox(
        "Compare Across Legal Traditions",
        value=True,
        help="Analyze differences between Common Law, Civil Law, Islamic Law, etc."
    )

    st.divider()

    # Judicial Context
    st.subheader("🏛️ Judicial Context")

    judicial_context = st.selectbox(
        "Select Judicial Decision Type",
        ["Risk Assessment", "Bail Decisions", "Sentencing",
         "Parole Decisions", "Pretrial Detention", "Evidence Evaluation"],
        help="Type of judicial decision being simulated"
    )

    st.divider()

    # Bias Configuration with Cultural Context
    st.subheader("🎭 Bias & Cultural Factors")

    bias_categories = {
        "Universal Biases": ["demographic", "socioeconomic", "historical"],
        "Legal System Biases": ["bureaucratic", "procedural", "institutional"],
        "Cultural Biases": ["religious", "ethnic", "linguistic", "tribal"],
        "Political Biases": ["ideological", "state_interest", "corruption"]
    }

    selected_biases = []
    for category, biases in bias_categories.items():
        if st.checkbox(f"Include {category}", value=True if "Universal" in category else False):
            selected_biases.extend(biases)

    bias_intensity = st.slider(
        "**Bias Intensity**",
        0.0, simulation_config.MAX_BIAS_FACTOR, 0.35, 0.05,
        help="Overall strength of bias in judicial systems"
    )
    poison_rate = st.slider(
        "**Data Poisoning Rate**",
        0.0, 0.5, 0.05, 0.01,
        help="Proportion of training data that is poisoned"
    )
    st.divider()

    # Global System Parameters
    st.subheader("🌐 Global Parameters")

    development_level = st.slider(
        "**Development Level Variation**",
        0.0, 1.0, 0.5,
        help="0 = Similar development levels, 1 = Maximum variation between countries"
    )

    rule_of_law_variation = st.slider(
        "**Rule of Law Variation**",
        0.0, 1.0, 0.6,
        help="0 = Similar rule of law, 1 = Maximum variation in legal system strength"
    )

    corruption_level = st.slider(
        "**Global Corruption Level**",
        0.0, 1.0, 0.3,
        help="Overall level of corruption affecting judicial systems"
    )

    st.divider()

    # International Standards
    st.subheader("📜 International Standards")

    human_rights_protection = st.slider(
        "**Human Rights Protection**",
        0.0, 1.0, 0.6,
        help="Level of adherence to international human rights standards"
    )

    un_standards = st.select_slider(
        "**UN Basic Principles on Judiciary**",
        options=["Non-compliant", "Partially Compliant", "Moderately Compliant", "Largely Compliant",
                 "Fully Compliant"],
        value="Moderately Compliant"
    )

    international_oversight = st.checkbox(
        "Include International Oversight Mechanisms",
        value=True,
        help="Simulate effects of international monitoring and pressure"
    )

    st.divider()

    # Simulation Configuration
    st.subheader("📊 Simulation Parameters")

    sample_size_per_country = st.number_input(
        "**Sample Size per Country**",
        500, 50000, 2000, step=500
    )

    n_runs = st.slider(
        "**Number of Simulation Runs**",
        1, 10, 3,
        help="More runs provide more reliable comparative statistics"
    )

    include_cross_country_analysis = st.checkbox(
        "Include Cross-Country Analysis",
        value=True,
        help="Compare performance across countries and legal systems"
    )

    st.divider()

    # Run Button
    col_run, col_reset = st.columns(2)
    with col_run:
        run_button = st.button(
            "🌍 **Run**",
            type="primary",
            use_container_width=True
        )
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.intl_judicial_history = []
            st.session_state.country_comparisons = {}
            st.rerun()


# ───────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────
def generate_country_specific_data(country, n_samples, judicial_context, legal_system_info):
    """Generate judicial data specific to a country's legal system."""
    # Base features
    n_features = 15
    X, y_base, demo_info = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features
    )

    # Country-specific adjustments
    country_info = COUNTRIES[country]
    system_info = legal_system_info

    # Adjust features based on legal system
    if system_info["type"] == "common":
        # Common law: emphasize precedent, adversarial process
        X[:, 0] *= 1.2  # Prior cases weight
        X[:, 1] += np.random.normal(0, 0.1, n_samples)  # More individual rights emphasis
    elif system_info["type"] == "civil":
        # Civil law: emphasize codes, investigative process
        X[:, 2] *= 1.1  # Procedural compliance
        X[:, 3] += np.random.normal(0.05, 0.1, n_samples)  # System consistency
    elif system_info["type"] == "islamic":
        # Islamic law: religious principles
        X[:, 4] = np.random.beta(3, 3, n_samples)  # Religious compliance factor
        X[:, 5] = np.random.beta(2, 4, n_samples)  # Community standards
    elif system_info["type"] == "socialist":
        # Socialist law: state interest
        X[:, 6] = np.random.beta(4, 2, n_samples)  # State interest factor
        X[:, 7] = np.random.beta(3, 3, n_samples)  # Social stability
    elif system_info["type"] == "customary":
        # Customary law: community focus
        X[:, 8] = np.random.beta(2, 2, n_samples)  # Community harmony
        X[:, 9] = np.random.beta(3, 2, n_samples)  # Restorative justice
    else:  # hybrid
        # Mixed systems: blend of factors
        X[:, :5] += np.random.normal(0, 0.05, (n_samples, 5))

    # Development level effects
    hdi = country_info["hdi"]
    rule_of_law = country_info["rule_of_law"]

    # Higher HDI = better data quality, more resources
    data_quality = hdi * 0.5 + 0.5
    X += np.random.normal(0, 1 - data_quality, X.shape) * 0.3

    # Rule of law effects
    procedural_fairness = rule_of_law * 0.7 + np.random.uniform(0, 0.3, n_samples)
    X[:, 10] = procedural_fairness

    # Corruption effects
    corruption_level = 1 - rule_of_law  # Simplified assumption
    if corruption_level > 0.3:
        # Introduce systematic bias in corruption
        corruption_bias = np.random.choice([0, 1], n_samples, p=[1 - corruption_level, corruption_level])
        X[corruption_bias == 1, 11] = np.random.beta(2, 5, np.sum(corruption_bias == 1))

    # Generate outcome based on judicial context
    risk_factors = np.sum(X[:, :8], axis=1) / 8
    system_factors = np.sum(X[:, 8:12], axis=1) / 4

    # Weight factors by legal system
    if system_info["adversarial"]:
        weight_individual = 0.7
        weight_system = 0.3
    else:
        weight_individual = 0.4
        weight_system = 0.6

    # Presumption of innocence effect
    innocence_presumption = system_info["presumption_of_innocence"]
    risk_score = (risk_factors * weight_individual + system_factors * weight_system)
    risk_score *= (1 - innocence_presumption * 0.3)  # Higher presumption reduces risk scores

    # Add country-specific noise
    risk_score += np.random.normal(0, 0.1 * (1 - rule_of_law), n_samples)

    # Create binary labels
    y = (risk_score > np.median(risk_score)).astype(int)

    # Defendant groups (0 = disadvantaged, 1 = advantaged)
    # Based on socioeconomic factors and country-specific disparities
    socioeconomic = X[:, 1] if X.shape[1] > 1 else np.random.rand(n_samples)
    poverty_threshold = np.percentile(socioeconomic, 40)  # 40% poorest are disadvantaged
    defendant_groups = (socioeconomic > poverty_threshold).astype(int)

    # Ethnic/racial groups (simplified)
    n_ethnic_groups = 3 if country in ["United States", "India", "South Africa"] else 2
    ethnic_groups = np.random.randint(0, n_ethnic_groups, n_samples)

    return X, y, defendant_groups, ethnic_groups, {
        "country": country,
        "legal_system": system_info["type"],
        "rule_of_law": rule_of_law,
        "hdi": hdi,
        "system_info": system_info
    }


def calculate_cross_cultural_metrics(y_true, y_pred, defendant_groups, ethnic_groups,
                                     country_info, legal_system_info):
    """Calculate metrics with cultural and legal system context."""
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
        "positive_prediction_rate": (tp + fp) / len(y_pred)
    })

    # Group-based fairness metrics
    disadvantaged_mask = (defendant_groups == 0)
    advantaged_mask = (defendant_groups == 1)

    if np.any(disadvantaged_mask):
        metrics["fpr_disadvantaged"] = np.sum((y_pred[disadvantaged_mask] == 1) &
                                              (y_true[disadvantaged_mask] == 0))
        metrics["fpr_disadvantaged"] /= np.sum(y_true[disadvantaged_mask] == 0) if np.sum(
            y_true[disadvantaged_mask] == 0) > 0 else 1

        metrics["detention_rate_disadvantaged"] = np.mean(y_pred[disadvantaged_mask] == 1)
    else:
        metrics["fpr_disadvantaged"] = 0
        metrics["detention_rate_disadvantaged"] = 0

    if np.any(advantaged_mask):
        metrics["fpr_advantaged"] = np.sum((y_pred[advantaged_mask] == 1) &
                                           (y_true[advantaged_mask] == 0))
        metrics["fpr_advantaged"] /= np.sum(y_true[advantaged_mask] == 0) if np.sum(
            y_true[advantaged_mask] == 0) > 0 else 1

        metrics["detention_rate_advantaged"] = np.mean(y_pred[advantaged_mask] == 1)
    else:
        metrics["fpr_advantaged"] = 0
        metrics["detention_rate_advantaged"] = 0

    # Calculate disparities
    if metrics["fpr_advantaged"] > 0:
        metrics["fpr_ratio"] = metrics["fpr_disadvantaged"] / metrics["fpr_advantaged"]
    else:
        metrics["fpr_ratio"] = float('inf') if metrics["fpr_disadvantaged"] > 0 else 1.0

    metrics["detention_ratio"] = (metrics["detention_rate_disadvantaged"] /
                                  metrics["detention_rate_advantaged"] if metrics[
                                                                              "detention_rate_advantaged"] > 0 else 1.0)

    # Legal system specific metrics
    system_type = legal_system_info["type"]

    if system_type == "common":
        # Common law: focus on procedural fairness and precedent
        metrics["procedural_fairness"] = 1.0 - min(1.0, abs(metrics["fpr_ratio"] - 1) * 0.5)
        metrics["precedent_consistency"] = metrics["accuracy"] * 0.7 + metrics["precision"] * 0.3
    elif system_type == "civil":
        # Civil law: focus on legal certainty and systematic application
        metrics["legal_certainty"] = 1.0 - metrics["false_positive_rate"]
        metrics["systematic_fairness"] = (1.0 - abs(metrics["detention_ratio"] - 1) * 0.3) * 0.7 + metrics[
            "accuracy"] * 0.3
    elif system_type == "islamic":
        # Islamic law: focus on religious compliance and community justice
        metrics["sharia_compliance"] = 0.6 + np.random.uniform(0, 0.4)  # Placeholder
        metrics["community_justice"] = 1.0 - min(1.0, metrics["false_positive_rate"] * 2)
    elif system_type == "socialist":
        # Socialist law: focus on social stability and state interests
        metrics["social_stability"] = 0.7 + np.random.uniform(0, 0.3)  # Placeholder
        metrics["state_security"] = metrics["recall"] * 0.6 + metrics["precision"] * 0.4
    elif system_type == "customary":
        # Customary law: focus on community harmony and restorative justice
        metrics["community_harmony"] = 0.8 + np.random.uniform(0, 0.2)  # Placeholder
        metrics["restorative_justice"] = 1.0 - metrics["false_positive_rate"]
    else:  # hybrid
        # Mixed systems: balance multiple values
        metrics["pluralistic_fairness"] = (metrics["accuracy"] * 0.4 +
                                           (1.0 - min(1.0, abs(metrics["fpr_ratio"] - 1))) * 0.6)
        metrics["cross_cultural_justice"] = 0.7 + np.random.uniform(0, 0.3)

    # Overall justice score (weighted by legal system values)
    if system_type == "common":
        weights = {"procedural_fairness": 0.6, "accuracy": 0.3, "precedent_consistency": 0.1}
    elif system_type == "civil":
        weights = {"legal_certainty": 0.5, "systematic_fairness": 0.3, "accuracy": 0.2}
    elif system_type == "islamic":
        weights = {"sharia_compliance": 0.5, "community_justice": 0.3, "accuracy": 0.2}
    elif system_type == "socialist":
        weights = {"social_stability": 0.4, "state_security": 0.4, "accuracy": 0.2}
    elif system_type == "customary":
        weights = {"community_harmony": 0.5, "restorative_justice": 0.3, "accuracy": 0.2}
    else:  # hybrid
        weights = {"pluralistic_fairness": 0.5, "cross_cultural_justice": 0.3, "accuracy": 0.2}

    justice_score = 0
    for metric, weight in weights.items():
        if metric in metrics:
            justice_score += metrics[metric] * weight
        else:
            justice_score += 0.5 * weight  # Default if metric not calculated

    metrics["justice_score"] = justice_score

    # Liberty protection (universal value)
    metrics["liberty_protection"] = 1.0 - min(1.0, metrics["false_positive_rate"] * 1.5)

    # Overall fairness score (balancing justice and liberty)
    metrics["fairness_score"] = (metrics["justice_score"] * 0.6 +
                                 metrics["liberty_protection"] * 0.4)

    # Rule of law adjustment
    metrics["rule_of_law_adjusted"] = metrics["fairness_score"] * country_info["rule_of_law"]

    # Human rights compliance
    metrics["human_rights_score"] = (metrics["liberty_protection"] * 0.6 +
                                     (1.0 - min(1.0, abs(metrics["fpr_ratio"] - 1))) * 0.4)

    return metrics


# ───────────────────────────────────────────────
# Main Content Area
# ───────────────────────────────────────────────
if run_button or st.session_state.intl_judicial_history:

    if run_button:
        # Clear previous results
        st.session_state.intl_judicial_history = []

        # Run simulations for each country
        progress_bar = st.progress(0)
        total_simulations = len(selected_countries) * n_runs

        current_sim = 0

        for country in selected_countries:
            country_results = []
            legal_system = COUNTRIES[country]["system"]
            system_info = LEGAL_SYSTEMS[legal_system]

            for i in range(n_runs):
                with st.spinner(f"Running simulation for {country} ({i + 1}/{n_runs})..."):

                    # Generate country-specific data
                    X, y, defendant_groups, ethnic_groups, country_metadata = generate_country_specific_data(
                        country=country,
                        n_samples=sample_size_per_country,
                        judicial_context=judicial_context,
                        legal_system_info=system_info
                    )

                    # Apply selected biases with cultural context
                    for bias_type in selected_biases:
                        if bias_type in ["demographic", "ethnic", "tribal"]:
                            # Apply ethnic bias
                            majority_mask = (ethnic_groups == 0)
                            minority_mask = (ethnic_groups > 0)

                            if np.any(minority_mask):
                                bias_strength = bias_intensity * 0.4
                                # Increase risk scores for minority groups
                                X[minority_mask, 0] *= (1 + bias_strength)
                                X[minority_mask, 3] *= (1 + bias_strength * 0.5)

                        elif bias_type == "socioeconomic":
                            # Socioeconomic bias
                            poor_mask = (defendant_groups == 0)
                            if np.any(poor_mask):
                                bias_strength = bias_intensity * 0.3
                                X[poor_mask, 4] *= (1 - bias_strength * 0.5)  # Reduce employment factor
                                X[poor_mask, 5] *= (1 + bias_strength * 0.3)  # Increase substance abuse suspicion

                    # Apply corruption effects
                    if corruption_level > 0:
                        corruption_mask = np.random.rand(len(X)) < corruption_level * 0.3
                        if np.any(corruption_mask):
                            # Corrupt decisions: randomize outcomes
                            y[corruption_mask] = np.random.choice([0, 1], np.sum(corruption_mask))

                    # Apply poisoning attack
                    X_p, y_p, defendant_groups = simulate_data_poisoning(
                        X, y, poison_rate * (1 + (1 - country_metadata["rule_of_law"])),
                        attack_type="label_flipping",
                        demographic_info=defendant_groups,
                        targeted=True
                    )

                    # Split data
                    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
                        X_p, y_p, defendant_groups, test_size=0.3, random_state=42 + i + hash(country) % 1000
                    )

                    # Standardize features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Train model
                    model = RandomForestClassifier(
                        n_estimators=100,
                        class_weight='balanced',
                        random_state=42 + i + hash(country) % 1000
                    )

                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    # Calculate comprehensive metrics
                    metrics = calculate_cross_cultural_metrics(
                        y_test, y_pred, groups_test, ethnic_groups[groups_test],
                        country_metadata, system_info
                    )

                    # Store results
                    run_result = {
                        "run_id": i + 1,
                        "country": country,
                        "legal_system": legal_system,
                        "legal_system_type": system_info["type"],
                        "judicial_context": judicial_context,
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                        "justice_score": metrics["justice_score"],
                        "liberty_protection": metrics["liberty_protection"],
                        "fairness_score": metrics["fairness_score"],
                        "rule_of_law_adjusted": metrics["rule_of_law_adjusted"],
                        "human_rights_score": metrics["human_rights_score"],
                        "fpr_disadvantaged": metrics.get("fpr_disadvantaged", 0),
                        "fpr_advantaged": metrics.get("fpr_advantaged", 0),
                        "fpr_ratio": min(metrics.get("fpr_ratio", 1), 10),  # Cap at 10
                        "detention_rate_disadvantaged": metrics.get("detention_rate_disadvantaged", 0),
                        "detention_rate_advantaged": metrics.get("detention_rate_advantaged", 0),
                        "detention_ratio": min(metrics.get("detention_ratio", 1), 10),
                        "rule_of_law": country_metadata["rule_of_law"],
                        "hdi": country_metadata["hdi"],
                        "bias_intensity": bias_intensity,
                        "poison_rate": poison_rate,
                        "corruption_level": corruption_level,
                        "human_rights_protection": human_rights_protection,
                        "un_standards": un_standards,
                        "selected_biases": ", ".join(selected_biases) if selected_biases else "None"
                    }

                    # Add legal system specific metrics
                    for key in ["procedural_fairness", "legal_certainty", "sharia_compliance",
                                "social_stability", "community_harmony", "pluralistic_fairness"]:
                        if key in metrics:
                            run_result[key] = metrics[key]

                    country_results.append(run_result)

                    current_sim += 1
                    progress_bar.progress(current_sim / total_simulations)

            # Add country results to global history
            st.session_state.intl_judicial_history.extend(country_results)

        progress_bar.empty()

    # ── Display Results ───────────────────────────────
    if st.session_state.intl_judicial_history:
        df = pd.DataFrame(st.session_state.intl_judicial_history)

        # Summary Metrics Section
        st.markdown("## 🌍 International Judicial Fairness Dashboard")

        # Calculate global averages
        avg_fairness = df["fairness_score"].mean()
        avg_justice = df["justice_score"].mean()
        avg_liberty = df["liberty_protection"].mean()
        avg_fpr_ratio = df["fpr_ratio"].mean()

        # Display global metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="global-metric">', unsafe_allow_html=True)
            st.metric(
                label="🌐 Global Fairness Score",
                value=f"{avg_fairness:.2f}/1.0",
                delta=None,
                help="Average fairness across all countries"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="fairness-metric-intl">', unsafe_allow_html=True)
            st.metric(
                label="⚖️ Global Justice Score",
                value=f"{avg_justice:.2f}/1.0",
                delta=None,
                help="Average justice score across legal systems"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="liberty-metric-intl">', unsafe_allow_html=True)
            st.metric(
                label="🕊️ Global Liberty Protection",
                value=f"{avg_liberty:.2f}/1.0",
                delta=None,
                help="Average liberty protection across countries"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="cultural-metric">', unsafe_allow_html=True)
            st.metric(
                label="⚠️ Average FPR Disparity",
                value=f"{avg_fpr_ratio:.1f}x",
                delta_color="inverse",
                help="Average False Positive Rate ratio (disadvantaged:advantaged)"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Global disparity alert
        if avg_fpr_ratio > 2.5:
            st.markdown(f"""
            <div class="international-alert">
                <strong>🌍🚨 GLOBAL DISPARITY ALERT</strong><br>
                The average False Positive Rate for disadvantaged defendants is 
                <strong>{avg_fpr_ratio:.1f}x higher</strong> than for advantaged defendants 
                across all countries. This indicates systemic bias requiring international attention.
            </div>
            """, unsafe_allow_html=True)

        # Country Comparison Visualization
        st.markdown("### 📊 Country-by-Country Comparison")

        # Aggregate by country
        country_stats = df.groupby("country").agg({
            "fairness_score": "mean",
            "justice_score": "mean",
            "liberty_protection": "mean",
            "fpr_ratio": "mean",
            "rule_of_law": "first",
            "hdi": "first",
            "legal_system_type": "first"
        }).reset_index()

        # Add colors based on legal system
        country_stats["color"] = country_stats["legal_system_type"].apply(
            lambda x: LEGAL_SYSTEMS.get(
                next((k for k, v in LEGAL_SYSTEMS.items() if v["type"] == x), ""),
                {"color": "#95a5a6"}
            )["color"]
        )

        # Create comparison bar chart
        fig = px.bar(
            country_stats.sort_values("fairness_score", ascending=False),
            x="country",
            y=["fairness_score", "justice_score", "liberty_protection"],
            color="legal_system_type",
            barmode="group",
            title="Judicial Fairness Comparison Across Countries",
            labels={"value": "Score", "variable": "Metric"},
            color_discrete_map={
                "common": "#3498db",
                "civil": "#2ecc71",
                "islamic": "#e74c3c",
                "socialist": "#9b59b6",
                "customary": "#f39c12",
                "hybrid": "#1abc9c"
            }
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Legal System Comparison
        if compare_systems and len(df["legal_system_type"].unique()) > 1:
            st.markdown("### ⚖️ Legal System Comparison")

            legal_system_stats = df.groupby("legal_system_type").agg({
                "fairness_score": ["mean", "std"],
                "justice_score": "mean",
                "liberty_protection": "mean",
                "fpr_ratio": "mean",
                "accuracy": "mean"
            }).round(3)

            legal_system_stats.columns = ['_'.join(col).strip() for col in legal_system_stats.columns.values]
            legal_system_stats = legal_system_stats.reset_index()

            # Create radar chart for legal systems
            legal_system_names = {
                "common": "Common Law",
                "civil": "Civil Law",
                "islamic": "Islamic Law",
                "socialist": "Socialist Law",
                "customary": "Customary Law",
                "hybrid": "Mixed/Hybrid"
            }

            legal_system_stats["system_name"] = legal_system_stats["legal_system_type"].map(legal_system_names)

            # Create radar chart
            categories = ['fairness_score_mean', 'justice_score_mean',
                          'liberty_protection_mean', 'accuracy_mean']

            fig = go.Figure()

            for idx, row in legal_system_stats.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[cat] for cat in categories],
                    theta=['Fairness', 'Justice', 'Liberty', 'Accuracy'],
                    fill='toself',
                    name=row['system_name'],
                    line_color=LEGAL_SYSTEMS.get(
                        next((k for k, v in LEGAL_SYSTEMS.items() if v["type"] == row["legal_system_type"]), ""),
                        {"color": "#95a5a6"}
                    )["color"]
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title="Legal System Performance Comparison",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Detailed Analysis Tabs ───────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Overview", "⚖️ Cross-Cultural Analysis",
                                          "🌐 Global Trends", "📋 Detailed Results"])

        with tab1:
            # Performance by development level
            st.markdown("#### 📈 Performance by Development Level")

            fig = px.scatter(
                country_stats,
                x="hdi",
                y="fairness_score",
                size="rule_of_law",
                color="legal_system_type",
                hover_data=["country"],
                title="Judicial Fairness vs Human Development Index",
                labels={
                    "hdi": "Human Development Index",
                    "fairness_score": "Fairness Score",
                    "rule_of_law": "Rule of Law Score"
                },
                size_max=30,
                color_discrete_map={
                    "common": "#3498db",
                    "civil": "#2ecc71",
                    "islamic": "#e74c3c",
                    "socialist": "#9b59b6",
                    "customary": "#f39c12",
                    "hybrid": "#1abc9c"
                }
            )

            st.plotly_chart(fig, use_container_width=True)

            # FPR Ratio by Legal System
            st.markdown("#### ⚠️ Disparity Analysis by Legal System")

            fig2 = px.box(
                df,
                x="legal_system_type",
                y="fpr_ratio",
                color="legal_system_type",
                title="False Positive Rate Ratio Distribution by Legal System",
                labels={
                    "legal_system_type": "Legal System",
                    "fpr_ratio": "FPR Ratio (Disadvantaged:Advantaged)"
                },
                color_discrete_map={
                    "common": "#3498db",
                    "civil": "#2ecc71",
                    "islamic": "#e74c3c",
                    "socialist": "#9b59b6",
                    "customary": "#f39c12",
                    "hybrid": "#1abc9c"
                }
            )

            fig2.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Parity")
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            # Cross-cultural analysis
            st.markdown("### 🌍 Cross-Cultural Justice Analysis")

            # Create a matrix of key metrics by country
            matrix_data = country_stats.pivot_table(
                index="country",
                values=["fairness_score", "justice_score", "liberty_protection", "fpr_ratio"]
            )

            # Display as heatmap
            fig = px.imshow(
                matrix_data.T,
                labels=dict(x="Country", y="Metric", color="Score"),
                x=matrix_data.index,
                y=matrix_data.columns,
                color_continuous_scale="RdYlGn",
                title="Judicial Metrics Heatmap by Country"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Cultural context analysis
            st.markdown("#### 🎭 Cultural Context Factors")

            cultural_factors = pd.DataFrame({
                "Factor": ["Rule of Law Strength", "Human Development",
                           "Legal Tradition", "Corruption Level", "International Standards"],
                "Impact on Fairness": [
                    df["rule_of_law"].corr(df["fairness_score"]),
                    df["hdi"].corr(df["fairness_score"]),
                    0.3,  # Estimated correlation for legal tradition
                    -df["corruption_level"].mean() * 0.5,
                    0.4 if un_standards in ["Largely Compliant", "Fully Compliant"] else 0.1
                ]
            })

            fig2 = px.bar(
                cultural_factors,
                x="Factor",
                y="Impact on Fairness",
                title="Cultural and Systemic Factors Impacting Judicial Fairness",
                color="Impact on Fairness",
                color_continuous_scale="RdYlGn"
            )

            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            # Global trends analysis
            st.markdown("### 🌐 Global Justice Trends")

            # Rule of Law vs Fairness
            fig = px.scatter(
                df,
                x="rule_of_law",
                y="fairness_score",
                color="legal_system_type",
                trendline="ols",
                title="Rule of Law vs Judicial Fairness",
                labels={
                    "rule_of_law": "Rule of Law Index",
                    "fairness_score": "Judicial Fairness Score"
                },
                color_discrete_map={
                    "common": "#3498db",
                    "civil": "#2ecc71",
                    "islamic": "#e74c3c",
                    "socialist": "#9b59b6",
                    "customary": "#f39c12",
                    "hybrid": "#1abc9c"
                }
            )

            st.plotly_chart(fig, use_container_width=True)

            # International standards impact
            st.markdown("#### 📜 Impact of International Standards")

            # Simulate UN standards impact
            un_impact = {
                "Non-compliant": 0.1,
                "Partially Compliant": 0.3,
                "Moderately Compliant": 0.5,
                "Largely Compliant": 0.7,
                "Fully Compliant": 0.9
            }

            standards_data = pd.DataFrame({
                "UN Compliance Level": list(un_impact.keys()),
                "Estimated Fairness Improvement": list(un_impact.values())
            })

            fig2 = px.bar(
                standards_data,
                x="UN Compliance Level",
                y="Estimated Fairness Improvement",
                title="Impact of UN Judicial Standards Compliance",
                color="Estimated Fairness Improvement",
                color_continuous_scale="Greens"
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
                    "justice_score": "{:.2f}",
                    "liberty_protection": "{:.2f}",
                    "fairness_score": "{:.2f}",
                    "rule_of_law_adjusted": "{:.2f}",
                    "human_rights_score": "{:.2f}",
                    "fpr_disadvantaged": "{:.1%}",
                    "fpr_advantaged": "{:.1%}",
                    "detention_rate_disadvantaged": "{:.1%}",
                    "detention_rate_advantaged": "{:.1%}",
                    "rule_of_law": "{:.2f}",
                    "hdi": "{:.3f}"
                }).background_gradient(subset=["fairness_score"], cmap="RdYlGn")
                .background_gradient(subset=["fpr_ratio"], cmap="Reds_r")
                .background_gradient(subset=["rule_of_law"], cmap="Blues")
            )

            # Download options
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "gags_international_judicial_results.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col2:
                # Export configuration
                config_data = {
                    "selected_countries": selected_countries,
                    "judicial_context": judicial_context,
                    "selected_biases": selected_biases,
                    "bias_intensity": bias_intensity,
                    "poison_rate": poison_rate,
                    "corruption_level": corruption_level,
                    "human_rights_protection": human_rights_protection,
                    "un_standards": un_standards,
                    "development_level_variation": development_level,
                    "rule_of_law_variation": rule_of_law_variation,
                    "average_global_fairness": f"{avg_fairness:.2f}",
                    "average_fpr_ratio": f"{avg_fpr_ratio:.1f}x"
                }
                import json

                json_str = json.dumps(config_data, indent=2)
                st.download_button(
                    "📋 Export Configuration (JSON)",
                    json_str,
                    "international_judicial_config.json",
                    "application/json",
                    use_container_width=True
                )

        # ── International Policy Recommendations ───────────────────────────────
        st.divider()

        # Generate actionable recommendations
        st.markdown("## 💡 International Policy Recommendations")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.info("""
            **🌍 For Global Governance:**

            1. **International Standards:** Develop AI-specific judicial standards
            2. **Cross-Border Audits:** Allow international algorithmic audits
            3. **Data Sharing:** Share de-identified fairness data for research
            4. **Capacity Building:** Support developing countries in AI governance
            5. **Harmonization:** Work towards compatible fairness frameworks

            **⚖️ For National Implementation:**

            1. **Legal System Adaptation:** Tailor AI systems to local legal traditions
            2. **Cultural Sensitivity:** Account for local norms and values
            3. **Capacity Development:** Build local technical and legal expertise
            4. **Public Consultation:** Involve diverse stakeholders in design
            """)

        with rec_col2:
            st.success("""
            **🔬 For Technical Adaptation:**

            1. **Cultural Calibration:** Adjust fairness metrics for cultural contexts
            2. **Legal System Modules:** Develop plug-ins for different legal traditions
            3. **Multilingual Systems:** Support local languages in interfaces
            4. **Contextual Explanations:** Provide explanations appropriate to local contexts

            **🤝 For International Cooperation:**

            1. **Best Practice Sharing:** Create platforms for sharing lessons
            2. **Joint Research:** Collaborate on cross-cultural fairness research
            3. **Technical Assistance:** Provide support for system implementation
            4. **Monitoring Networks:** Establish networks for ongoing monitoring
            """)

        # UN Principles reminder
        st.markdown("""
        <div class="cultural-note">
            <p style="margin:0; color:#856404; font-weight:bold;">
                🇺🇳 <em>"The Universal Declaration of Human Rights (Article 10): 
                Everyone is entitled in full equality to a fair and public hearing 
                by an independent and impartial tribunal."</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "⚠️ **Disclaimer:** This simulation is for educational and research purposes. Real judicial AI systems must respect national sovereignty while upholding international human rights standards.")

else:
    # Welcome/Instruction state
    st.markdown("## 🌍⚖️ Welcome to International Judicial Systems Simulation")

    st.markdown("""
    This module allows you to compare how AI impacts judicial fairness across different countries, 
    legal traditions, and cultural contexts. Explore how universal principles and local contexts 
    interact in algorithmic justice systems.
    """)

    # Quick start examples
    st.markdown("### 🚀 Quick Start Comparisons")

    col1, col2, col3 = st.columns(3)

    with col1:
        system_color = LEGAL_SYSTEMS["Common Law (US/UK/Canada/Australia)"]["color"]
        st.markdown(f"""
        <div class="country-card" style="--country-color:{system_color};">
            <h4>🇺🇸 Common Law Systems</h4>
            <p>Compare US, UK, Canada, Australia:</p>
            <ul>
                <li>Adversarial procedures</li>
                <li>Precedent-based decisions</li>
                <li>Strong individual rights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        system_color = LEGAL_SYSTEMS["Civil Law (France/Germany/Japan/Brazil)"]["color"]
        st.markdown(f"""
        <div class="country-card" style="--country-color:{system_color};">
            <h4>🇫🇷 Civil Law Systems</h4>
            <p>Compare France, Germany, Japan, Brazil:</p>
            <ul>
                <li>Codified legal systems</li>
                <li>Inquisitorial procedures</li>
                <li>Systematic application</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        system_color = LEGAL_SYSTEMS["Islamic Law (Saudi Arabia/Iran/Pakistan)"]["color"]
        st.markdown(f"""
        <div class="country-card" style="--country-color:{system_color};">
            <h4>🇸🇦 Islamic Law Systems</h4>
            <p>Compare Saudi Arabia, Iran, Pakistan:</p>
            <ul>
                <li>Sharia-based principles</li>
                <li>Religious jurisprudence</li>
                <li>Community-focused justice</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # How to use guide
    with st.expander("📖 How to Use This Simulation", expanded=True):
        st.markdown("""
        1. **Select countries** from different legal traditions for comparison
        2. **Configure judicial context** (type of decision being simulated)
        3. **Adjust bias factors** with attention to cultural contexts
        4. **Set global parameters** (development levels, corruption, international standards)
        5. **Run simulations** to compare across countries
        6. **Analyze results** by country, legal system, and development level
        7. **Explore policy recommendations** for international governance

        **Key Metrics to Compare:**
        - **Fairness Score:** Context-appropriate fairness measure
        - **Justice Score:** Legal system-specific justice evaluation
        - **Liberty Protection:** Protection against wrongful decisions
        - **FPR Ratio:** Disparity in false positive rates
        - **Rule of Law Impact:** Effect of institutional strength
        - **Development Correlation:** Relationship with HDI

        **Legal Traditions Covered:**
        - Common Law (adversarial, precedent-based)
        - Civil Law (codified, inquisitorial)
        - Islamic Law (Sharia-based)
        - Socialist Law (state-interest focused)
        - Customary Law (community-based)
        - Mixed/Hybrid Systems
        """)

    # Global context
    st.warning("""
    **🌐 Global Justice Context:**

    Judicial AI systems must navigate complex international considerations:
    - **Cultural Relativism vs Universal Rights:** Balancing local traditions with human rights
    - **Sovereignty vs International Standards:** Respecting national sovereignty while upholding standards
    - **Development Disparities:** Different capacities for AI implementation
    - **Legal Pluralism:** Multiple legal systems operating within single countries
    - **Cross-Border Effects:** Decisions in one country affecting others

    Responsible international AI governance requires sensitivity to these complexities while upholding fundamental rights.
    """)

# Footer
st.divider()
st.caption(
    "🌍⚖️ International Judicial Systems Simulation • GAGS Framework • v2.0 • Promoting Justice Across Borders and Cultures")