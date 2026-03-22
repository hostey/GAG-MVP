# pages/03_🌾_Agrotech_Equity.py
"""
Agrotech Equity Simulation — GAGS Framework v3.0

The missing domain page. Covers:
  - Smallholder farming scenarios (Plateau State, Abuja FCT)
  - Climate-resilient agriculture bias testing
  - AI Agent Economy for resource allocation (irrigation, fertilizer, drones)
  - Gender equity audit (52% female smallholder farmers)
  - Multilingual fairness (Hausa/Yoruba/Igbo/English)
  - Full XAI layer: feature importance, instance explanation, counterfactuals
  - Compliance report: NITDA, UNESCO, AU AI Policy, ISO 42001
  - Model card auto-generation
  - Intersectional fairness: gender × income × connectivity
"""

import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from components.pdf_report import generate_pdf_compliance_report
from components.governance_logic import (
    # Data
    generate_africa_centric_data,
    generate_synthetic_data,
    AFRICA_SCENARIO_PRESETS,
    # Bias / attack
    apply_bias,
    simulate_data_poisoning,
    # Fairness
    calculate_fairness_metrics,
    run_gender_equity_audit,
    # Feature modules
    AgentEconomySandbox,
    MultimodalRedTeamer,
    HybridGovernanceLayer,
    StrategicReasoningArena,
    run_simple_simulation,
    simulate_longitudinal_bias,
    simulate_federated_learning,
    # XAI (new)
    ExplainableModel,
    generate_intersectional_fairness,
    generate_compliance_report,
    generate_model_card,
)
from utils.config import simulation_config, settings
from components.i18n import t, get_lang, language_switcher, language_badge
from components.ussd_simulator import ussd_interface, accessibility_gap_report, format_sms_result
from components.nigeria_regulatory import nigeria_compliance_panel
from components.ux_utils import (
    guided_tour_banner, preset_selector,
    metric_glossary_expander, history_browser, save_to_history,
    share_url_panel, load_config_from_url, apply_url_config,
    annotation_panel,
    role_switcher, get_active_role, role_banner,
    board_member_summary, ROLE_TAB_VISIBILITY,
)


st.set_page_config(page_title="Agrotech Equity • GAGS", layout="wide", page_icon="🌾")

# ── Constants ─────────────────────────────────────────────────────────────────
# simulation_config.BIAS_TYPES holds the core list defined before Feature 3.
# Gender and linguistic were added as BiasType enum values in governance_logic §1
# but may not be present in the config object — merge them in explicitly.
_FEATURE3_BIAS_TYPES = ["gender", "linguistic"]
_VALID_BIAS_TYPES = list(simulation_config.BIAS_TYPES) + [
    b for b in _FEATURE3_BIAS_TYPES if b not in simulation_config.BIAS_TYPES
]

_AGRO_POLICY       = "Deploy AI crop-advisory system to smallholder farmers in FCT"

_AGRO_FEATURE_NAMES = [
    "age", "income_level", "connectivity_score",
    "land_size_ha", "soil_quality", "rainfall_index",
    "market_distance_km", "fertilizer_access", "irrigation_access",
    "prior_yield_kg", "climate_stress",
]

_CROP_SCENARIOS = {
    "Plateau State Smallholder": {
        "desc": "Climate-resilient maize/sorghum for smallholders in Plateau State",
        "target": "crop_failure_risk",
        "key_features": ["rainfall_index", "soil_quality", "climate_stress"],
    },
    "FCT Market Access": {
        "desc": "Predicting whether farmers can access Abuja FCT markets profitably",
        "target": "market_access_viable",
        "key_features": ["market_distance_km", "income_level", "connectivity_score"],
    },
    "Fertilizer Allocation": {
        "desc": "AI-driven fertilizer quota allocation across smallholder groups",
        "target": "fertilizer_need_high",
        "key_features": ["soil_quality", "land_size_ha", "prior_yield_kg"],
    },
    "Climate Risk Assessment": {
        "desc": "Flood/drought risk classification for crop insurance eligibility",
        "target": "climate_high_risk",
        "key_features": ["rainfall_index", "climate_stress", "irrigation_access"],
    },
}

_STATE_DEFAULTS = {
    "agro_run_history":      [],
    "agro_baseline":         None,
    "agro_feature_outputs":  {},
    "agro_xai_results":      {},
    "agro_dataset_info":     {},
    "agro_snapshot_history": [],
    "agro_longitudinal":     None,
    "agro_federated":        None,  # UX history browser
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Page setup ────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_agro_data(scenario_key: str, n_samples: int, noise_level: float = 0.1):
    """Generate Abuja FCT-calibrated agrotech data."""
    try:
        X, y_base, demo, preset = generate_africa_centric_data(
            scenario="smallholder_agrotech", n_samples=n_samples)
    except Exception:
        X, y_base, demo = generate_synthetic_data(n_samples=n_samples, n_features=11)
        preset = AFRICA_SCENARIO_PRESETS.get("smallholder_agrotech", {})

    n = X.shape[0]
    rng = np.random.RandomState(42)

    land_size   = rng.lognormal(0.5, 0.6, n)
    soil_qual   = rng.beta(2, 2, n)
    rainfall    = rng.beta(3, 2, n)
    mkt_dist    = rng.exponential(15, n)
    fertilizer  = rng.beta(2, 3, n)
    irrigation  = rng.beta(1.5, 3, n)
    prior_yield = rng.lognormal(7, 0.5, n)
    climate_str = rng.beta(2, 2, n)

    # Resize X to match feature names
    n_needed = len(_AGRO_FEATURE_NAMES)
    if X.shape[1] < n_needed:
        X = np.hstack([X, np.zeros((n, n_needed - X.shape[1]))])
    elif X.shape[1] > n_needed:
        X = X[:, :n_needed]

    # Fill agrotech-specific columns
    col_map = {
        "land_size_ha":      land_size,
        "soil_quality":      soil_qual,
        "rainfall_index":    rainfall,
        "market_distance_km": mkt_dist,
        "fertilizer_access": fertilizer,
        "irrigation_access": irrigation,
        "prior_yield_kg":    prior_yield / 1000,
        "climate_stress":    climate_str,
    }
    for col_name, values in col_map.items():
        if col_name in _AGRO_FEATURE_NAMES:
            idx = _AGRO_FEATURE_NAMES.index(col_name)
            if idx < X.shape[1]:
                X[:, idx] = values

    # Build outcome
    scenario_info = _CROP_SCENARIOS.get(scenario_key, list(_CROP_SCENARIOS.values())[0])
    target = scenario_info.get("target", "crop_failure_risk")

    if target == "crop_failure_risk":
        score = (1 - rainfall) * 0.35 + climate_str * 0.25 + (1 - soil_qual) * 0.20 + rng.normal(0, 0.1, n)
    elif target == "market_access_viable":
        score = (1 / (mkt_dist + 1)) * 0.30 + X[:, 1] * 0.25 + X[:, 2] * 0.20 + rng.normal(0, 0.1, n)
    elif target == "fertilizer_need_high":
        score = (1 - fertilizer) * 0.40 + (1 - soil_qual) * 0.30 + rng.normal(0, 0.1, n)
    else:
        score = (1 - rainfall) * 0.30 + climate_str * 0.30 + (1 - irrigation) * 0.20 + rng.normal(0, 0.1, n)

    threshold = np.percentile(score, 50)
    y = (score > threshold).astype(int)

    # income_group: 0=low, 1=higher
    income_group = (X[:, 1] > np.median(X[:, 1])).astype(int)

    return X.astype(np.float64), y, demo.astype(int), income_group, _AGRO_FEATURE_NAMES


def _train_and_score(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """Train a RandomForest and return metrics + model + splits."""
    if len(np.unique(y)) < 2:
        return None, None, None, None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.3, random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None)

    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                  random_state=random_state)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    metrics = {
        "accuracy":  float(accuracy_score(y_te, y_pred)),
        "recall":    float(recall_score(y_te, y_pred, zero_division=0)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_te, y_pred, zero_division=0)),
        "fpr":       float(np.mean(y_pred[y_te == 0] == 1)) if (y_te == 0).any() else 0.0,
    }
    return metrics, clf, scaler, (X_tr, X_te, y_tr, y_te)


def _run_one(
    scenario_key, n_samples, selected_biases, bias_intensity,
    poison_rate, run_idx,
    enable_xai=True, enable_redteam=False, enable_governance=True,
    enable_arena=False, enable_agent_economy=False, enable_gender_audit=True,
):
    """Execute one agrotech simulation run."""
    X, y, demo, income_group, feat_names = _generate_agro_data(scenario_key, n_samples)
    X = X.astype(np.float64)

    # Bias injection
    valid_biases = [b for b in selected_biases if b in _VALID_BIAS_TYPES]
    for bt in valid_biases:
        try:
            X, y, demo = apply_bias(X, y, bt, bias_intensity, demographic_info=demo)
        except Exception:
            pass

    # Poisoning
    try:
        X, y, demo = simulate_data_poisoning(
            X, y, poison_rate,
            attack_type="label_flipping",
            demographic_info=demo,
            targeted=False,
        )
    except Exception:
        pass

    metrics, clf, scaler, splits = _train_and_score(X, y, random_state=42 + max(run_idx, 0))
    if clf is None:
        return None

    X_tr, X_te, y_tr, y_te = splits
    y_pred = clf.predict(X_te)
    fair   = calculate_fairness_metrics(y_te, y_pred, demo[:len(y_te)])

    # Gender equity audit
    gender_audit = None
    if enable_gender_audit:
        try:
            dig_base = AFRICA_SCENARIO_PRESETS.get(
                "smallholder_agrotech", {}).get("digital_inclusion_baseline", 0.34)
            gender_audit = run_gender_equity_audit(
                y_te, y_pred, demo[:len(y_te)], dig_base)
        except Exception:
            pass

    # XAI (first run only)
    xai_results = {}
    if enable_xai and run_idx == 0:
        try:
            xm = ExplainableModel(domain="agrotech")
            xm.model         = clf
            xm.feature_names = feat_names[:X_te.shape[1]]
            xm._X_train      = X_tr
            xm._is_fitted    = True

            fi = xm.feature_importance(X_te, y_te, n_repeats=8)
            xai_results["feature_importance"] = fi.__dict__

            high_risk_idx = np.where(y_te == 1)[0]
            if len(high_risk_idx) > 0:
                inst = X_te[high_risk_idx[0]]
                expl = xm.explain_instance(inst, instance_idx=int(high_risk_idx[0]))
                xai_results["instance_explanation"] = expl.__dict__
                cf = xm.counterfactual(inst)
                xai_results["counterfactual"] = cf.__dict__

            mc = xm.model_card(metrics, fair, domain="agrotech")
            cr = generate_compliance_report(
                mc, fair, metrics,
                frameworks=["EU AI Act", "ISO 42001", "NIST AI RMF", "NITDA", "UNESCO"])
            ix = generate_intersectional_fairness(
                y_te, y_pred,
                {"gender": demo[:len(y_te)], "income": income_group[:len(y_te)]},
                min_group_size=20,
            )
            xai_results["model_card"]        = mc.__dict__
            xai_results["compliance_report"] = cr
            xai_results["intersectional"]    = ix.__dict__
            st.session_state.agro_xai_results = xai_results
        except Exception as e:
            st.session_state.agro_xai_results = {"error": str(e)}

    # Feature module: Agent Economy
    agent_economy_result = None
    if enable_agent_economy:
        try:
            sandbox = AgentEconomySandbox(
                n_agents=min(20, n_samples // 100),
                n_goods=5,
                domain="agrotech",
            )
            ae = sandbox.run_auction(X_te[:50], y_te[:50], demo[:50])
            agent_economy_result = ae
        except Exception:
            pass

    # Longitudinal + federated on first run
    if run_idx == 0:
        try:
            bi = bias_intensity if bias_intensity > 0 else 0.15
            bt = valid_biases[0] if valid_biases else "demographic"
            lng = simulate_longitudinal_bias(
                X, y, demo,
                initial_bias_type=bt, initial_bias_intensity=bi,
                n_generations=5, random_state=42)
            st.session_state.agro_longitudinal = lng.__dict__
        except Exception:
            st.session_state.agro_longitudinal = None
        try:
            fed = simulate_federated_learning(
                X, y, demo,
                n_clients=4, n_rounds=3,
                bias_heterogeneity=(bias_intensity or 0.15) * 0.5,
                random_state=42)
            st.session_state.agro_federated = fed.__dict__
        except Exception:
            st.session_state.agro_federated = None

    return {
        "run_id":              run_idx + 1,
        "scenario":            scenario_key,
        "accuracy":            metrics["accuracy"],
        "recall":              metrics["recall"],
        "precision":           metrics["precision"],
        "f1":                  metrics["f1"],
        "fpr":                 metrics["fpr"],
        "equity_score":        fair.get("fairness_score", 0.5),
        "fairness_score":      fair.get("fairness_score", 0.5),
        "demographic_parity":  fair.get("demographic_parity_difference", 0),
        "equalized_odds":      fair.get("equalized_odds_difference", 0),
        "bias_intensity":      bias_intensity,
        "poison_rate":         poison_rate,
        "biases":              ", ".join(valid_biases) or "None",
        "gender_gap":          gender_audit.overall_gender_gap if gender_audit else 0.0,
        "digital_exclusion":   gender_audit.digital_inclusion_score if gender_audit else 0.0,
        "agent_economy":       agent_economy_result,
        "xai":                 xai_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    language_switcher(location="sidebar")
    st.divider()
    role_switcher("agrotech")
    st.divider()

    st.markdown("""<div style="text-align:center;padding:.5rem 0;">
      <h2 style="color:#39ff7a;margin:0;">⚙️ Agrotech Config</h2>
      <p style="color:#888;font-size:.82rem;">AI Bias in Agricultural Technology</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.subheader("🌾 Crop Scenario")
    scenario_key = st.selectbox("Scenario",
        ["smallholder_agrotech","climate_resilient_maize",
         "irrigation_equity","market_linkage"],
        format_func=lambda k: {
            "smallholder_agrotech":   "Smallholder Farming (Abuja FCT)",
            "climate_resilient_maize":"Climate-Resilient Maize (Plateau State)",
            "irrigation_equity":      "Irrigation Access Equity",
            "market_linkage":         "Market Linkage AI",
        }.get(k, k))
    st.divider()

    st.subheader("🎭 Bias Configuration")
    _FEATURE3_BIAS_TYPES = ["gender","linguistic"]
    _ag_valid = list(simulation_config.BIAS_TYPES) + [
        b for b in _FEATURE3_BIAS_TYPES if b not in simulation_config.BIAS_TYPES]
    _ag_defaults = [b for b in ["demographic","socioeconomic","gender","geographic"] if b in _ag_valid]
    selected_biases = st.multiselect("Bias Types", options=_ag_valid, default=_ag_defaults,
        format_func=lambda x: f"🔴 {x}" if x in ("gender","demographic") else f"⚠️ {x}")
    bias_intensity = st.slider("Bias Intensity", 0.0, float(simulation_config.MAX_BIAS_FACTOR), 0.3, 0.05)
    st.divider()

    st.subheader("⚠️ Adversarial Attacks")
    poison_rate = st.slider("Poisoning Rate", 0.0, 0.5, 0.05, 0.01, format="%.2f")
    st.divider()

    st.subheader("📊 Simulation Parameters")
    n_samples = st.number_input("Farmer Records", 500, 50000, settings.DEFAULT_N_SAMPLES, 500)
    n_runs    = st.slider("Simulation Runs", 1, 8, 3)
    include_baseline = st.toggle("Include Baseline (no bias/attack)", value=True)
    st.divider()

    st.subheader("🔬 Feature Modules")
    enable_xai           = st.toggle("Explainable AI",       value=True)
    enable_redteam       = st.toggle("Multimodal Red Team",  value=False)
    enable_governance    = st.toggle("Governance Layer",     value=True)
    enable_arena         = st.toggle("Strategic Arena",      value=False)
    enable_agent_economy = st.toggle("Agent Economy",        value=True)
    enable_gender_audit  = st.toggle("Gender Equity Audit",  value=True)
    st.divider()

    col_r, col_x = st.columns(2)
    run_button = col_r.button("🌾 Run", type="primary", use_container_width=True)
    if col_x.button("🔄 Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("agro_"):
                del st.session_state[k]
        st.rerun()

st.markdown(
    f"""<div class="page-header" style="--ac:#39ff7a;"><p style="font-family:'DM Mono',monospace;font-size:.69rem;letter-spacing:.16em;text-transform:uppercase;opacity:.5;margin:0 0 .55rem;display:flex;align-items:center;gap:.45rem;"><span style="width:16px;height:1px;background:#39ff7a;opacity:.55;display:inline-block;"></span>AGROTECH · GAGS v3.0 · FCT Nigeria</p><h1 style="font-family:'Syne',sans-serif!important;font-size:2.5rem!important;font-weight:800!important;line-height:1.08!important;letter-spacing:-.03em!important;margin:0 0 .6rem!important;">Agrotech Equity Simulation</h1><p style="margin:0;opacity:.72;font-size:.96rem;max-width:660px;line-height:1.65;">AI fairness for FCT smallholder farmers — gender audit (52% female farmers), Vickrey auction agent economy, NITDA alignment.</p><div style="margin-top:.9rem;"><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#39ff7a;">FCT Nigeria</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#39ff7a;">52% Female Farmers</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#39ff7a;">Agent Economy</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#39ff7a;">USSD Mode</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#39ff7a;">NITDA</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#39ff7a;">UNESCO SDG2</span></div></div>""",
    unsafe_allow_html=True
)

# Africa context banner
preset_info = AFRICA_SCENARIO_PRESETS.get("smallholder_agrotech", {})
st.markdown(f"""
<div class="alert-info">
    <strong>🌍 Context:</strong> {preset_info.get('description','Abuja FCT smallholder agrotech')} 
    &nbsp;|&nbsp; Languages: {', '.join(preset_info.get('languages',['Hausa','Yoruba','Igbo','English']))} 
    &nbsp;|&nbsp; Digital inclusion baseline: {preset_info.get('digital_inclusion_baseline', 0.34):.0%} 
    &nbsp;|&nbsp; Female farmers: {preset_info.get('gender_distribution',{}).get('female',0.52):.0%} 
    &nbsp;|&nbsp; Climate stress factor: {preset_info.get('climate_stress_factor', 0.65):.2f}
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════════════════════════

if run_button:
    st.session_state.agro_run_history    = []
    st.session_state.agro_baseline       = None
    st.session_state.agro_feature_outputs= {}
    st.session_state.agro_xai_results    = {}

    if include_baseline:
        with st.spinner("Computing baseline…"):
            base = _run_one(
                scenario_key, n_samples, [], 0.0, 0.0, -1,
                enable_xai=False, enable_redteam=False, enable_governance=False,
                enable_arena=False, enable_agent_economy=False, enable_gender_audit=False,
            )
            st.session_state.agro_baseline = base

    prog = st.progress(0, text="Starting…")
    for i in range(n_runs):
        prog.progress(i / n_runs, text=f"Run {i+1}/{n_runs}…")
        with st.spinner(f"Simulation {i+1}/{n_runs}"):
            result = _run_one(
                scenario_key, n_samples, selected_biases, bias_intensity,
                poison_rate, i,
                enable_xai=enable_xai and i == 0,  # XAI only on first run
                enable_redteam=enable_redteam,
                enable_governance=enable_governance,
                enable_arena=enable_arena,
                enable_agent_economy=enable_agent_economy,
                enable_gender_audit=enable_gender_audit,
            )
            if result:
                st.session_state.agro_run_history.append(result)
                save_to_history(
                    "agro_snapshot_history",
                    label=f"Run {i+1} | bias={bias_intensity:.2f} | {scenario_key[:16]}",
                    metrics={
                        "accuracy":          result.get("accuracy", 0),
                        "fairness_score":    result.get("fairness_score", 0),
                        "demographic_parity":result.get("demographic_parity", 0),
                        "recall":            result.get("recall", 0),
                    },
                    config={
                        "scenario_key":   scenario_key,
                        "bias_intensity": bias_intensity,
                        "poison_rate":    poison_rate,
                    },
                )
    prog.progress(1.0, text="Complete ✓"); prog.empty()

    # ── Longitudinal + Federated ──────────────────────────────────────────────
    if st.session_state.agro_run_history:
        _bl = bias_intensity if bias_intensity > 0 else 0.15
        _bt_list = [b for b in selected_biases if b in _VALID_BIAS_TYPES]
        _bt = _bt_list[0] if _bt_list else "demographic"
        try:
            _Xl, _yl, _dl, _, _fnms = _generate_agro_data(scenario_key, min(n_samples, 1000))
            lng = simulate_longitudinal_bias(_Xl.astype(float), _yl, _dl,
                initial_bias_type=_bt, initial_bias_intensity=_bl,
                n_generations=5, apply_mitigation=False, random_state=42)
            st.session_state.agro_longitudinal = lng.__dict__
        except Exception:
            st.session_state.agro_longitudinal = None
        try:
            fed = simulate_federated_learning(_Xl.astype(float), _yl, _dl,
                n_clients=4, n_rounds=3,
                bias_heterogeneity=_bl * 0.5, random_state=42)
            st.session_state.agro_federated = fed.__dict__
        except Exception:
            st.session_state.agro_federated = None


# ═══════════════════════════════════════════════════════════════════════════════
# Results dashboard
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.agro_run_history:
    df_r   = pd.DataFrame(st.session_state.agro_run_history)
    base   = st.session_state.agro_baseline
    feats  = st.session_state.agro_feature_outputs
    xai    = st.session_state.agro_xai_results
    info   = st.session_state.agro_dataset_info

    avg_acc  = df_r["accuracy"].mean()
    avg_fair = df_r["fairness_score"].mean()
    avg_rec  = df_r["recall"].mean()
    avg_dp   = df_r["demographic_parity"].mean()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown("## 📊 Agrotech Equity Dashboard")
    role_banner("agrotech")

    if info:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scenario",       info.get("scenario","—")[:20])
        c2.metric("Samples",        f"{info.get('samples',0):,}")
        c3.metric("Female Farmers", f"{info.get('female_ratio',0):.0%}")
        c4.metric("Positive Rate",  f"{info.get('positive_rate',0):.1%}")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="card-yield">
            <p style="margin:0;font-size:.8rem;color:#555;">🌱 Prediction Accuracy</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:#3b6d11;">{avg_acc:.1%}</p>
            {'<p style="font-size:.8rem;color:#888;margin:0;">vs baseline: ' + f"{avg_acc - base['accuracy']:+.1%}" + '</p>' if base else ''}
        </div>""", unsafe_allow_html=True)
    with k2:
        fc = "#27ae60" if avg_fair >= 0.7 else "#e67e22" if avg_fair >= 0.5 else "#e74c3c"
        st.markdown(f"""
        <div class="card-equity">
            <p style="margin:0;font-size:.8rem;color:#555;">⚖️ Fairness Score</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:{fc};">{avg_fair:.2f}<span style="font-size:.9rem">/1.0</span></p>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="card-risk">
            <p style="margin:0;font-size:.8rem;color:#555;">🎯 Recall (Detection)</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:#ba7517;">{avg_rec:.1%}</p>
        </div>""", unsafe_allow_html=True)
    with k4:
        dpc = "#e74c3c" if avg_dp > 0.1 else "#e67e22" if avg_dp > 0.05 else "#27ae60"
        st.markdown(f"""
        <div class="card-access">
            <p style="margin:0;font-size:.8rem;color:#555;">📊 Demographic Parity Gap</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:{dpc};">{avg_dp:.1%}</p>
        </div>""", unsafe_allow_html=True)

    # Governance banner
    if "governance" in feats:
        gov = feats["governance"]
        cls = {"approved":"alert-success","rejected":"alert-warning",
               "deferred":"alert-info","reversed":"alert-danger"}.get(gov["outcome"],"alert-info")
        flags_html = "".join(f"<li>{f}</li>" for f in gov["ai_flags"]) or "<li>No drift detected</li>"
        t = gov["tally"]
        st.markdown(f"""
        <div class="{cls}" style="margin-top:1rem;">
            <strong>🏛️ Governance Vote — "{gov['policy']}"</strong><br>
            Outcome: <strong>{gov['outcome'].upper()}</strong> &nbsp;|&nbsp;
            For: {t.get('for',0)} &nbsp; Against: {t.get('against',0)} &nbsp; Abstain: {t.get('abstain',0)}<br>
            <strong>AI Flags:</strong><ul style="margin:.3rem 0 0 1rem;">{flags_html}</ul>
            <span style="font-size:.75rem;opacity:.7;">Ledger hash: <code>{gov['ledger_hash']}</code></span>
        </div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    # ── Share URL panel ───────────────────────────────────────────────
    _share_cfg = {"domain":"agrotech","scenario_key":scenario_key,"bias_intensity":bias_intensity,"poison_rate":poison_rate}
    share_url_panel("agrotech", config=_share_cfg)

    # ── Board Member view (role-specific executive summary) ───────────
    _role_now = get_active_role("agrotech")
    if _role_now == "Board Member":
        _fair_val = avg_fair if "avg_fair" in dir() else 0.5
        _acc_val  = avg_acc if "avg_acc" in dir() else 0.5
        _ok = _fair_val >= 0.7
        _finding = ("Fairness score is within acceptable range. No critical disparities detected."
                    if _ok else "Fairness score below 0.70 — demographic disparities detected.")
        _rec = ("Continue quarterly monitoring and maintain current governance oversight."
                if _ok else "Bias mitigation required before deployment. Consult Data Science team.")
        board_member_summary("agrotech", _acc_val, _fair_val, _ok, _finding, _rec)
    else:
        metric_glossary_expander(["fairness score", "demographic parity", "false positive rate", "false negative rate", "gender gap", "digital inclusion score", "permeability score", "bias intensity", "poison rate"])
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Performance", "⚖️ Equity & Gender",
        "🧠 Explainable AI", "📋 Compliance",
        "🔁 Longitudinal", "🌐 Federated",
        "🤖 Feature Modules", "📊 Agent Economy", "📋 Raw Results",
    ])

    # ── Tab 1: Performance ─────────────────────────────────────────────────────
    with tab1:
        fig_g = make_subplots(rows=1, cols=3,
            specs=[[{"type":"indicator"}]*3],
            subplot_titles=("Accuracy","Recall","Fairness Score"))
        for ci, (val, col) in enumerate([(avg_acc*100,"darkgreen"),(avg_rec*100,"darkorange"),(avg_fair*100,"darkblue")], 1):
            fig_g.add_trace(go.Indicator(
                mode="gauge+number", value=round(val,1),
                gauge={"axis":{"range":[0,100]},"bar":{"color":col},
                       "steps":[{"range":[0,60],"color":"#f0f0f0"},{"range":[60,80],"color":"#d0d0d0"}]},
            ), row=1, col=ci)
        fig_g.update_layout(height=260, showlegend=False, margin=dict(t=40,b=0))
        st.plotly_chart(fig_g, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_bar = px.bar(df_r, x="run_id", y=["accuracy","recall","precision","f1"],
                barmode="group", title="Metrics per Run",
                color_discrete_sequence=["#3b6d11","#639922","#97c459","#c0dd97"])
            st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            if len(df_r) > 1:
                fig_sc = px.scatter(df_r, x="fairness_score", y="accuracy",
                    color="bias_intensity", size=[10]*len(df_r),
                    title="Fairness vs Accuracy",
                    labels={"fairness_score":"Fairness Score","accuracy":"Accuracy"},
                    color_continuous_scale="RdYlGn_r")
                fig_sc.add_shape(type="rect",x0=0.7,x1=1.0,y0=0.7,y1=1.0,
                    line=dict(color="green",width=2,dash="dash"),fillcolor="rgba(0,180,0,0.07)")
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                c1i,c2i,c3i = st.columns(3)
                c1i.metric("Fairness", f"{df_r['fairness_score'].iloc[0]:.3f}")
                c2i.metric("Accuracy", f"{df_r['accuracy'].iloc[0]:.3f}")
                c3i.metric("F1 Score", f"{df_r['f1'].iloc[0]:.3f}")

    # ── Tab 2: Equity & Gender ─────────────────────────────────────────────────
    with tab2:
        st.markdown("### ⚖️ Fairness & Gender Equity Analysis")

        if "gender_audit" in feats:
            ga = feats["gender_audit"]
            st.markdown("#### 🌍 Gender Equity Audit (UNESCO Women4EthicalAI)")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Gender Gap",            f"{ga.get('overall_gender_gap',0):.3f}")
            g2.metric("Representation Score",  f"{ga.get('representation_score',0):.3f}")
            g3.metric("Digital Inclusion",     f"{ga.get('digital_inclusion_score',0):.3f}")
            g4.metric("Audit",                 "✅ PASS" if ga.get("audit_passed") else "❌ FAIL")

            recs = ga.get("incentive_recommendations",[])
            if ga.get("audit_passed"):
                st.markdown(f'<div class="alert-success">✅ {recs[0] if recs else "Audit passed."}</div>', unsafe_allow_html=True)
            else:
                recs_html = "".join(f"<li>{r}</li>" for r in recs)
                st.markdown(f'<div class="alert-danger">❌ Audit failed.<ul style="margin:.3rem 0 0 1rem;">{recs_html}</ul></div>', unsafe_allow_html=True)

        # Intersectional fairness
        if xai and "intersectional" in xai:
            ix = xai["intersectional"]
            st.markdown("#### 🔀 Intersectional Fairness (Gender × Income)")
            st.markdown(f'<div class="alert-info">{ix.get("narrative","")}</div>', unsafe_allow_html=True)

            if ix.get("group_performances"):
                gp_df = pd.DataFrame([
                    {"Group": k, **{kk: round(vv, 3) for kk, vv in v.items()}}
                    for k, v in ix["group_performances"].items()
                ])
                st.dataframe(gp_df.style.background_gradient(subset=["accuracy"], cmap="RdYlGn"),
                             use_container_width=True)
                c1, c2 = st.columns(2)
                c1.metric("Worst Group",          ix.get("worst_intersectional_group","—")[:30])
                c1.metric("Intersectional Gap",   f"{ix.get('intersectional_gap',0):.1%}")
                c2.metric("Best Group",           ix.get("best_intersectional_group","—")[:30])
                c2.metric("Amplification Factor", f"{ix.get('amplification_factor',1):.2f}×")

        c1, c2 = st.columns(2)
        with c1:
            fig_dp = px.bar(df_r, x="run_id", y="demographic_parity",
                title="Demographic Parity Gap per Run",
                labels={"demographic_parity":"Gap","run_id":"Run"},
                color="demographic_parity", color_continuous_scale="RdYlGn_r",
                range_y=[0, max(0.3, df_r["demographic_parity"].max()*1.2)])
            fig_dp.add_hline(y=0.1, line_dash="dot", line_color="red",
                             annotation_text="Acceptable threshold (0.10)")
            st.plotly_chart(fig_dp, use_container_width=True)
        with c2:
            fig_fs = px.bar(df_r, x="run_id", y="fairness_score",
                title="Fairness Score per Run",
                color="fairness_score", color_continuous_scale="RdYlGn",
                range_y=[0, 1])
            fig_fs.add_hline(y=0.7, line_dash="dot", line_color="green",
                             annotation_text="Target (0.70)")
            st.plotly_chart(fig_fs, use_container_width=True)

        if avg_dp > 0.15:
            st.markdown('<div class="alert-danger"><strong>⚠️ High demographic parity gap.</strong> AI recommendations are significantly less accurate for certain farmer groups. Immediate bias mitigation required before deployment.</div>', unsafe_allow_html=True)
        elif avg_fair < 0.7:
            st.markdown('<div class="alert-warning"><strong>⚠️ Fairness below target.</strong> Apply gender-stratified resampling and validate across linguistic groups before deployment.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-success"><strong>✅ Fairness within acceptable range.</strong> Continue quarterly monitoring with local community participation.</div>', unsafe_allow_html=True)

    # ── Tab 3: Explainable AI ──────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🧠 Explainable AI — Understanding Model Decisions")

        if not xai:
            st.info("Enable **XAI** in the sidebar and run at least one simulation to see explanations.")
        else:
            # Feature importance
            if "feature_importance" in xai:
                fi = xai["feature_importance"]
                st.markdown("#### Feature Importance (Permutation-based)")
                st.markdown(f'<div class="xai-box"><em>{fi.get("narrative","")}</em></div>',
                            unsafe_allow_html=True)

                fi_df = pd.DataFrame({
                    "Feature":    fi["feature_names"][:10],
                    "Importance": fi["importances"][:10],
                    "Std Dev":    fi["std_devs"][:10],
                })
                fig_fi = px.bar(fi_df, x="Importance", y="Feature",
                    orientation="h", error_x="Std Dev",
                    title=f"Feature Importance ({fi.get('method','permutation')} method)",
                    color="Importance", color_continuous_scale="Greens",
                    labels={"Feature":"Feature","Importance":"Importance Score"})
                fig_fi.update_layout(yaxis={"categoryorder":"total ascending"}, height=360)
                st.plotly_chart(fig_fi, use_container_width=True)

            c_l, c_r = st.columns(2)

            # Instance explanation
            with c_l:
                if "instance_explanation" in xai:
                    ex = xai["instance_explanation"]
                    st.markdown("#### Instance Explanation (LIME-lite)")
                    pred_word = "HIGH RISK" if ex["predicted_class"] == 1 else "LOW RISK"
                    conf      = ex.get("confidence", 0)
                    stability = ex.get("lime_stability", 0)
                    st.markdown(f"""
                    <div class="xai-box">
                        <strong>Prediction:</strong> {pred_word} &nbsp;|&nbsp;
                        <strong>Confidence:</strong> {conf:.0%} &nbsp;|&nbsp;
                        <strong>Explanation stability:</strong> {stability:.0%}<br>
                        <hr style="margin:.5rem 0;border-color:var(--color-border-tertiary);">
                        <em>{ex.get('decision_path','')}</em>
                    </div>""", unsafe_allow_html=True)

                    contribs = ex.get("feature_contributions", {})
                    if contribs:
                        sorted_c = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
                        cf_df = pd.DataFrame(sorted_c, columns=["Feature","Contribution"])
                        fig_c = px.bar(cf_df, x="Contribution", y="Feature",
                            orientation="h", title="Feature contributions for this farmer",
                            color="Contribution", color_continuous_scale="RdYlGn",
                            color_continuous_midpoint=0)
                        fig_c.update_layout(yaxis={"categoryorder":"total ascending"}, height=300)
                        st.plotly_chart(fig_c, use_container_width=True)

            # Counterfactual
            with c_r:
                if "counterfactual" in xai:
                    cf = xai["counterfactual"]
                    st.markdown("#### Counterfactual Explanation")
                    st.markdown(f"""
                    <div class="xai-box">
                        <strong>Original prediction:</strong> Class {cf.get('original_prediction',0)}<br>
                        <strong>Counterfactual:</strong> Class {cf.get('counterfactual_prediction',1)}<br>
                        <strong>Features changed:</strong> {cf.get('n_features_changed',0)}<br>
                        <strong>New confidence:</strong> {cf.get('confidence_after',0):.0%}<br>
                        <hr style="margin:.5rem 0;border-color:var(--color-border-tertiary);">
                        <em>{cf.get('plain_language','')}</em>
                    </div>""", unsafe_allow_html=True)

                    changes = cf.get("changes", {})
                    if changes:
                        ch_df = pd.DataFrame([
                            {"Feature": k, "Original": v[0], "Counterfactual": v[1],
                             "Change": round(v[1]-v[0], 3)}
                            for k, v in changes.items()
                        ])
                        st.dataframe(ch_df.style.background_gradient(subset=["Change"],
                            cmap="RdYlGn"), use_container_width=True)

    # ── Tab 4: Compliance ─────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 📋 Regulatory Compliance Report")

        if not xai or "compliance_report" not in xai:
            st.info("Enable **XAI** in the sidebar and run a simulation to generate the compliance report.")
        else:
            cr   = xai["compliance_report"]
            summ = cr.get("summary", {})
            mc   = xai.get("model_card", {})

            st.markdown(f"""
            <div class="{'alert-success' if summ.get('overall_compliant') else 'alert-danger'}">
                <strong>Overall Compliance: {'✅ COMPLIANT' if summ.get('overall_compliant') else '❌ NOT COMPLIANT'}</strong>
                &nbsp;|&nbsp; Model: {cr.get('model_name','—')}
                &nbsp;|&nbsp; Risk Level: <strong>{cr.get('overall_risk_level','—')}</strong>
                &nbsp;|&nbsp; Generated: {cr.get('generated_at','')[:10]}
            </div>""", unsafe_allow_html=True)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Fairness Score",    f"{summ.get('fairness_score',0):.3f}")
            s2.metric("Parity Gap",        f"{summ.get('demographic_parity_gap',0):.1%}")
            s3.metric("Eqzd Odds Gap",     f"{summ.get('equalized_odds_gap',0):.1%}")
            s4.metric("Bias Findings",     summ.get("bias_findings_count", 0))

            # Framework-by-framework results
            frameworks = cr.get("frameworks", {})
            for fw_name, fw_data in frameworks.items():
                with st.expander(f"📑 {fw_name}", expanded=fw_name in ("EU AI Act","NITDA")):
                    if "checks" in fw_data:
                        for check, status in fw_data["checks"].items():
                            icon = "✅" if status == "PASS" else "❌"
                            colour = "compliance-pass" if status == "PASS" else "compliance-fail"
                            st.markdown(f'<span class="{colour}">{icon}</span> {check}',
                                        unsafe_allow_html=True)
                    if "recommendations" in fw_data and fw_data["recommendations"]:
                        st.markdown("**Recommendations:**")
                        for r in fw_data["recommendations"]:
                            st.markdown(f"  • {r}")
                    if "note" in fw_data:
                        st.caption(fw_data["note"])
                    # Show non-check fields
                    skip = {"checks","recommendations","note"}
                    for k, v in fw_data.items():
                        if k not in skip and isinstance(v, str):
                            st.markdown(f"**{k}:** {v}")

            # Model card
            if mc:
                st.markdown("#### 📄 Auto-Generated Model Card")
                mc_export = {k: v for k, v in mc.items() if k != "timestamp"}
                st.download_button(
                    "📥 Download Model Card (JSON)",
                    json.dumps(mc_export, indent=2),
                    f"model_card_{scenario_key.lower().replace(' ','_')}.json",
                    "application/json", use_container_width=True,
                )

                # PDF compliance report
                try:
                    pdf_bytes = generate_pdf_compliance_report(
                        compliance_report=cr,
                        model_card=mc,
                        simulation_metadata={"accuracy": xai.get("model_card",{}).get("performance_metrics",{}).get("accuracy",0)},
                        domain="agrotech",
                    )
                    st.download_button(
                        "📄 Download Compliance Report (PDF)",
                        pdf_bytes,
                        f"compliance_report_{scenario_key.lower().replace(' ','_')}.pdf",
                        "application/pdf",
                        use_container_width=True,
                    )
                except Exception as _pdf_err:
                    st.caption(f"PDF generation unavailable: {_pdf_err}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Intended Uses:**")
                    for u in mc.get("intended_uses",[]): st.markdown(f"  • {u}")
                    st.markdown("**Out-of-Scope:**")
                    for u in mc.get("out_of_scope_uses",[]): st.markdown(f"  • {u}")
                with col2:
                    st.markdown("**Bias Findings:**")
                    for f_ in mc.get("bias_findings",[]): st.markdown(f"  • {f_}")
                    st.markdown("**Ethical Considerations:**")
                    for e_ in mc.get("ethical_considerations",[]): st.markdown(f"  • {e_}")

    # ── Tab 5: Longitudinal ────────────────────────────────────────────────────
    with tab5:
        _lng = st.session_state.get("agro_longitudinal")
        st.markdown("### 🔁 Longitudinal Bias Analysis")
        st.markdown('<div class="alert-info">Simulates how crop advisory bias compounds when model predictions feed back into <strong>future training data</strong> over annual retraining cycles.</div>', unsafe_allow_html=True)
        if not _lng:
            st.info("Run a simulation to see longitudinal bias evolution.")
        else:
            c1,c2,c3 = st.columns(3)
            c1.metric("Initial Bias",f'{_lng["initial_bias"]:.1%}')
            c2.metric("Final Bias",f'{_lng["final_bias"]:.1%}',f'{_lng["final_bias"]-_lng["initial_bias"]:+.1%}')
            c3.metric("Amplification",f'{_lng["amplification_factor"]:.2f}×')
            if _lng.get("self_reinforcing"):
                st.error("⚠️ Bias became self-reinforcing. Recommend retraining from clean community-validated data.")
            st.markdown(f'<div class="alert-info"><em>{_lng["narrative"]}</em></div>', unsafe_allow_html=True)
            gm = _lng.get("generation_metrics",[])
            if gm:
                import plotly.express as _pxA
                fig_lng=_pxA.line(pd.DataFrame(gm),x="generation",
                    y=["demographic_parity","fairness_score","accuracy"],
                    title="Bias Evolution Across Retraining Cycles",
                    color_discrete_sequence=["#e74c3c","#27ae60","#3b6d11"])
                fig_lng.add_hline(y=0.1,line_dash="dot",line_color="red")
                st.plotly_chart(fig_lng,use_container_width=True)

    # ── Tab 6: Federated ────────────────────────────────────────────────────────
    with tab6:
        _fed = st.session_state.get("agro_federated")
        st.markdown("### 🌐 Federated Learning Simulation")
        st.markdown('<div class="alert-info">Tests whether crop prediction bias persists when models train <strong>across farming regions</strong> without centralising sensitive household data.</div>', unsafe_allow_html=True)
        if not _fed:
            st.info("Run a simulation to see federated learning results.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Regions",_fed["n_clients"])
            c2.metric("Global Accuracy",f'{_fed["global_accuracy"]:.1%}')
            c3.metric("Global Fairness",f'{_fed["global_fairness"]:.3f}')
            c4.metric("Bias Persisted","Yes ⚠️" if _fed["bias_persisted"] else "No ✅")
            st.markdown(f'<div class="{"alert-danger" if _fed["bias_persisted"] else "alert-success"}"><em>{_fed["narrative"]}</em></div>', unsafe_allow_html=True)
            cr_ = _fed.get("client_results",[])
            if cr_:
                import plotly.express as _pxB
                cr_df = pd.DataFrame([c.__dict__ if hasattr(c,"__dict__") else c for c in cr_])
                if not cr_df.empty and "local_bias" in cr_df.columns:
                    fig_fed=_pxB.bar(cr_df,x="client_id",
                        y=["local_accuracy","local_bias","local_fairness"],
                        barmode="group",title="Per-Region Metrics",
                        color_discrete_sequence=["#3b6d11","#e74c3c","#27ae60"])
                    st.plotly_chart(fig_fed,use_container_width=True)

    # ── Tab 7: Feature Modules ─────────────────────────────────────────────────
    with tab7:
        st.markdown("### 🤖 Advanced Feature Module Results")

        if "multimodal_redteam" in feats:
            st.markdown("#### Feature 2 — Multimodal Red Teaming")
            rt = feats["multimodal_redteam"]
            rt_rows = [{"Modality": r["modality"], "Attack Vector": r["attack_vector"],
                        "Severity": r["severity"], "Bypass Rate": f"{r['bypass_rate']:.1%}",
                        "Sociotechnical Risk": f"{r['sociotechnical_risk']:.2f}",
                        "VR Scenario": "Yes" if r.get("vr_scenario") else "No"}
                       for r in rt.get("modality_results",[])]
            if rt_rows:
                st.dataframe(pd.DataFrame(rt_rows), use_container_width=True)
            for r in rt.get("modality_results",[]):
                if r.get("vr_scenario"):
                    st.markdown(f'<div class="alert-info">🥽 <strong>VR Scenario:</strong><br>{r["vr_scenario"]}</div>',
                                unsafe_allow_html=True)
                    break

        if "strategic_arena" in feats:
            st.markdown("#### Feature 5 — Strategic Arena")
            arena = feats["strategic_arena"]
            standings = pd.DataFrame(arena.get("final_standings",[]))
            if not standings.empty:
                st.dataframe(standings.style.background_gradient(subset=["score"],cmap="YlGn"),
                             use_container_width=True)

        if not any(k in feats for k in ["multimodal_redteam","strategic_arena"]):
            st.info("Enable Feature 2 or Feature 5 modules in the sidebar to see results here.")

    # ── Tab 8: Agent Economy ───────────────────────────────────────────────────
    with tab8:
        st.markdown("### 💰 AI Agent Economy — Agricultural Resource Allocation")
        st.markdown("""
        <div class="alert-info">
            <strong>How it works:</strong> Autonomous AI agents bid for shared agricultural resources 
            (irrigation water, fertilizer quota, drone hours, market access) via a 
            <strong>Vickrey (second-price) auction</strong>. The permeability score measures 
            how unequal spending becomes — high permeability signals that large commercial 
            farms may be outbidding smallholders.
        </div>""", unsafe_allow_html=True)

        # ── USSD/SMS Accessibility Gap ────────────────────────────────────────
        st.divider()
        st.markdown("#### 📱 USSD/SMS Accessibility Analysis")
        _ussd_metrics = {"accuracy": avg_acc, "fairness_score": avg_fair, "recall": avg_rec}
        accessibility_gap_report(_ussd_metrics, domain="agrotech", connectivity_rate=0.34)
        st.divider()
        st.markdown("#### 📱 USSD Interface Simulator")
        _pred_for_ussd = {"risk_level": "high" if avg_rec > 0.5 else "low",
                          "top_features": ["rainfall_index", "soil_quality"],
                          "counterfactual_hint": "Irrigate within 3 days"}
        ussd_interface(prediction_result=_pred_for_ussd, domain="agrotech")

        if "agent_economy" in feats:
            ae = feats["agent_economy"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Economy Stability",  ae.get("economy_stability","—"))
            c2.metric("Permeability Score", f"{ae.get('permeability_score',0):.4f}",
                      help="Higher = more unequal resource distribution")
            c3.metric("Auction Rounds",     ae.get("rounds",0))

            if ae.get("permeability_score",0) >= 0.5:
                st.markdown("""
                <div class="alert-danger">
                    <strong>⚠️ High permeability detected.</strong> Resource allocation is 
                    unequal — aggressive-strategy agents are crowding out smallholders. 
                    Recommend price caps or reserved quotas for small-scale farmers.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-success"><strong>✅ Stable economy.</strong> Resource distribution is reasonably equitable across agent strategies.</div>', unsafe_allow_html=True)

            agents_df = pd.DataFrame(ae.get("agent_summary",[]))
            if not agents_df.empty:
                st.markdown("#### Agent Performance")
                st.dataframe(agents_df.style.background_gradient(
                    subset=["reputation","total_spent"], cmap="Blues"),
                    use_container_width=True)

                fig_spend = px.bar(agents_df, x="name", y="total_spent",
                    color="strategy", title="Total Spending by Agent Strategy",
                    labels={"total_spent":"Spend","name":"Agent"},
                    color_discrete_sequence=["#3b6d11","#e74c3c","#3498db"])
                st.plotly_chart(fig_spend, use_container_width=True)

            # Show auction log summary
            auction_results = ae.get("auction_results",[])
            if auction_results:
                st.markdown("#### Resource Auction Results")
                auction_df = pd.DataFrame(auction_results)
                if not auction_df.empty:
                    fig_auc = px.bar(
                        auction_df.groupby("resource")["price_paid"].mean().reset_index(),
                        x="resource", y="price_paid",
                        title="Average Clearing Price per Resource",
                        color="price_paid", color_continuous_scale="Greens",
                    )
                    st.plotly_chart(fig_auc, use_container_width=True)
        else:
            st.info("Enable **Feature 1 — Agent Economy** in the sidebar to see resource auction results.")

    # ── Tab 9: Raw Results ─────────────────────────────────────────────────────
    with tab9:
        show_cols = ["run_id","scenario","accuracy","recall","precision","f1",
                     "fairness_score","demographic_parity","equalized_odds",
                     "bias_intensity","poison_rate","positive_rate","biases"]
        show_df = df_r[[c for c in show_cols if c in df_r.columns]]
        fmt = {c: "{:.3f}" for c in show_cols
               if c not in ("run_id","scenario","biases")}
        st.dataframe(
            show_df.style
                .format({k: v for k, v in fmt.items() if k in show_df.columns})
                .background_gradient(subset=["accuracy"],     cmap="Greens")
                .background_gradient(subset=["fairness_score"], cmap="RdYlGn"),
            use_container_width=True,
        )

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("📥 Download CSV",
                df_r.to_csv(index=False).encode(),
                f"gags_agrotech_{scenario_key.lower().replace(' ','_')}.csv",
                "text/csv", use_container_width=True)
        with dl2:
            cfg = {
                "scenario": scenario_key, "selected_biases": selected_biases,
                "bias_intensity": bias_intensity, "poison_rate": poison_rate,
                "enable_xai": enable_xai, "enable_gender_audit": enable_gender_audit,
                "avg_accuracy": f"{avg_acc:.3f}", "avg_fairness": f"{avg_fair:.3f}",
            }
            st.download_button("📋 Export Config JSON",
                json.dumps(cfg, indent=2),
                f"agro_config_{scenario_key.lower().replace(' ','_')}.json",
                "application/json", use_container_width=True)

    # ── Simulation history ─────────────────────────────────────────────────────
    history_browser("agro_snapshot_history", domain="agrotech",
        key_metrics=["accuracy","fairness_score","demographic_parity"])

    # ── Annotation layer ────────────────────────────────────────────────
    annotation_panel("agro_annotations", context_label=f"{len(st.session_state.agro_run_history)} Agrotech run(s)")

    # ── Recommendations ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("## 💡 Agrotech AI Recommendations")
    r1, r2 = st.columns(2)
    with r1:
        if avg_dp > 0.1 or avg_fair < 0.7:
            st.markdown("""
            <div class="alert-danger">
                <strong>⚠️ Equity intervention required before deployment</strong><br>
                • Apply gender-stratified resampling (see Gender Equity Audit)<br>
                • Validate model across Hausa/Yoruba/Igbo language groups<br>
                • Deploy USSD/SMS fallback for farmers without smartphones<br>
                • Establish community advisory board before launch
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
                <strong>✅ Equity metrics acceptable for pilot deployment</strong><br>
                • Continue monthly monitoring with local extension officers<br>
                • Run gender audit quarterly aligned with planting seasons<br>
                • Expand to additional local languages in next version
            </div>""", unsafe_allow_html=True)
    with r2:
        st.info("""
        **For Agricultural Extension Officers:**
        1. Use counterfactual explanations to advise individual farmers
        2. Share feature importance charts at community meetings
        3. Flag any systematic differences across villages to the AI team
        4. Maintain a feedback log — report cases where AI was wrong

        **For Policymakers:**
        1. Require NITDA compliance report before any government AI-advisory deployment
        2. Mandate gender equity audit for all agricultural AI tools
        3. Ensure digital inclusion alternatives (USSD) are funded alongside AI systems
        """)

    st.caption("⚠️ Educational simulation. Real agricultural AI requires community validation, local language testing, and regulatory approval.")


# ── Welcome state ─────────────────────────────────────────────────────────────
else:
    st.markdown("## 🌾 Welcome to Agrotech Equity Simulation")
    st.markdown(
        "Configure your scenario in the sidebar and click **Run**. "
        "This module focuses on Nigeria's smallholder farming context — "
        "with Africa-centric data, gender equity audits, explainable AI, and compliance reporting."
    )

    c1, c2, c3 = st.columns(3)
    for col, (title, bg, desc) in zip([c1,c2,c3],[
        ("🌱 Crop failure risk",    "#3b6d11", "Plateau State maize/sorghum. Bias: geographic + climate."),
        ("🏪 Market access",        "#ba7517", "FCT market viability. Bias: income + connectivity."),
        ("💧 Resource allocation",  "#185fa5", "Fertilizer & irrigation. Agent economy active."),
    ]):
        col.markdown(
            f'<div style="background:{bg};padding:1.25rem;border-radius:10px;color:white;">'
            f'<h4 style="margin:0;color:white;">{title}</h4>'
            f'<p style="margin:.5rem 0 0;font-size:.87rem;opacity:.9;">{desc}</p></div>',
            unsafe_allow_html=True)

    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        1. Select a **crop scenario** matching your research question
        2. Enable **XAI** to get feature importance, instance explanations, and counterfactuals
        3. Enable **Gender Equity Audit** for UNESCO Women4EthicalAI alignment
        4. Enable **Agent Economy** to simulate resource auction fairness
        5. Enable **Governance Vote** to test policy proposals against AI drift detection
        6. Review the **Compliance tab** for auto-generated regulatory reports
        7. Use the **Model Card** export for procurement documentation
        """)

st.divider()
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;padding:1rem 0;'>"
    "🌾 Agrotech Equity Simulation · GAGS Framework v3.0 · "
    "Abuja FCT · Gender Equity · Explainable AI · NITDA / UNESCO Compliance"
    "</div>", unsafe_allow_html=True,
)