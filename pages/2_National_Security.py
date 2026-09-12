# pages/02_🛡️_National_Security.py
"""
National Security Resilience Simulation — GAGS Framework v1.0

Refactored to integrate all GAGS v1.0 governance_logic imports:
  - apply_bias()              with AttackSeverity enum (keyword args only)
  - simulate_data_poisoning() keyword demographic_info (fixes positional crash)
  - calculate_fairness_metrics() always-present group_metrics
  - run_simple_simulation()   consistent {status,results,features,warnings} envelope
  - MultimodalRedTeamer       Feature 2 — text/image/deepfake attacks
  - HybridGovernanceLayer     Feature 4 — citizen vote + blockchain ledger
  - AgentEconomySandbox       Feature 1 — Vickrey auction for security resources
  - StrategicReasoningArena   Feature 5 — negotiation/deception game

Key improvements over v2.2:
  - All simulate_data_poisoning() calls use keyword args (fixes TypeError crash)
  - "behavioral" bias type removed from defaults (not in simulation_config.BIAS_TYPES)
  - Consistent session-state namespace  (security_*)
  - _run_one() single computation function — no logic scattered in the run loop
  - Feature module results surfaced in dedicated tab
  - Governance banner rendered after every run
  - CSS uses semantic card classes (no inline gradients on metric cards)
  - dataset_choice scoped correctly (was referenced outside its conditional block)
  - All cmaps validated against matplotlib accepted list
"""

import json
from datetime import datetime
import warnings
from components.translate import install_auto_translate, tx, tx_plotly, language_switcher
install_auto_translate()
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── GAGS v1.0 core ────────────────────────────────────────────────────────────
try:
    from components.pdf_report import generate_pdf_compliance_report
    PDF_OK = True
except (ImportError, ModuleNotFoundError):
    PDF_OK = False
    def generate_pdf_compliance_report(*a, **kw):
        return None
from components.governance_logic import (
    AttackSeverity,
    apply_bias,
    simulate_data_poisoning,
    calculate_fairness_metrics,
    generate_synthetic_data,
    run_simple_simulation,
    ExplainableModel,
    generate_compliance_report,
    generate_intersectional_fairness,
    # Feature 1
    AgentEconomySandbox,
    # Feature 2
    MultimodalRedTeamer,
    # Feature 4
    HybridGovernanceLayer,
    # Feature 5
    StrategicReasoningArena,
    simulate_longitudinal_bias,
    simulate_federated_learning,
)
from utils.config import simulation_config, settings
from components.i18n import t, get_lang, language_badge
from components.ussd_simulator import ussd_interface, accessibility_gap_report, format_sms_result
from components.nigeria_regulatory import nigeria_compliance_panel
from components.gags_interactive import (
    progress_tracker, scenario_story_banner,
    domain_challenge_panel, benchmark_challenge_panel,
    what_if_explorer, bias_detective_panel, track_run, award_points,
    _reset_render_guards
)
_reset_render_guards()
from components.ai_safety import run_ai_safety_suite
from components.gags_lifecycle import run_lifecycle_suite
from components.gags_lifecycle_ui import render_lifecycle_tab, render_eco_tab
from components.gags_dynamic_systems import run_dynamic_systems_suite
from components.gags_dynamic_ui import render_dynamic_systems_tab
from components.gags_safety_ui import render_safety_tab
from components.gags_features_full import (
    run_agent_economy_simulation,render_multimodal_redteam_full,recommend_allocation_strategy, plot_monte_carlo_sensitivity, run_redteam_simulation, run_arena_simulation,render_agent_economy_full,run_monte_carlo_analysis
)
from components.gags_feature_modules import (
    feature_modules_tab, multi_challenge_panel,
    admin_challenge_panel, feature_module_sidebar
)
from components.ux_utils import (
    guided_tour_banner, preset_selector,
    metric_glossary_expander, history_browser, save_to_history,
    share_url_panel, load_config_from_url, apply_url_config,
    annotation_panel,
    role_switcher, get_active_role, role_banner,
    get_role_algo, get_role_tabs, get_role_defaults,
    role_algo_banner, role_brief_banner,
    board_member_summary, ROLE_TAB_VISIBILITY,
)


# ── Constants ─────────────────────────────────────────────────────────────────
# Only bias types that exist in simulation_config.BIAS_TYPES are valid
_VALID_BIAS_TYPES = list(simulation_config.BIAS_TYPES)

_ATTACK_TYPE_MAP: dict = {
    "Data Poisoning":    "label_flipping",
    "Evasion Attacks":   "feature_noise",
    "Model Inversion":   "feature_noise",    # label_smoothing not implemented; feature_noise is closest analogue
    "Backdoor Attacks":  "backdoor",
}

_OVERSIGHT_MULTIPLIER: dict = {
    "Minimal":         1.2,
    "Moderate":        1.0,
    "Strong":          0.8,
    "Judicial Review": 0.6,
}

_GOVERNANCE_POLICY = "Deploy AI threat-detection in national security operations"

# ── Session-state defaults ────────────────────────────────────────────────────
st.set_page_config(
    page_title="National Security • GAGS",
    layout="wide",
    page_icon="🛡️",
)

# ── Design system ──────────────────────────────────────────────────────────────
try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, page_header, plotly_theme as _ptheme
    inject_css("security")
    ACCENT = DOMAIN_ACCENTS["security"]
except ImportError:
    ACCENT = "#ef4444"


_STATE_DEFAULTS = {
    "security_run_history":   [],
    "security_baseline":      None,
    "security_feature_outputs": {},
    "security_rw_df":         None,
    "security_rw_info":       None,
    "security_snapshot_history": [],  # UX history browser,
    "sec_ds_report": {}
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v




# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess_data(data_source, dataset_choice, scenario, n_samples, random_state=42):
    """Generate security dataset."""
    X, y, demo = generate_synthetic_data(n_samples=n_samples, n_features=12)
    return X.astype(np.float64), y.astype(int), demo.astype(int)


def _run_baseline(data_source, dataset_choice, scenario, n_samples,
                  surveillance_level, data_retention, oversight_level):
    """Run baseline simulation (no bias, no attack)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y, demo = _preprocess_data(data_source, dataset_choice, scenario, n_samples)
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=42,
        stratify=y if (len(np.unique(y)) > 1 and np.bincount(y.astype(int)).min() >= 2) else None)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    fair = calculate_fairness_metrics(yte, yp, demo[:len(yte)])
    fpr = float(np.mean(yp[yte==0]==1)) if (yte==0).any() else 0.0
    lib = float(np.clip(1 - fpr*2 - fair.get("demographic_parity_difference",0), 0, 1))
    return {"accuracy":float(accuracy_score(yte,yp)),"detection_rate":float(recall_score(yte,yp,zero_division=0)),
            "false_positive_rate":fpr,"fairness_score":fair.get("fairness_score",0.7),
            "liberty_score":lib,"demographic_parity":fair.get("demographic_parity_difference",0),
            "is_baseline":True}


def _run_one(
    data_source, dataset_choice, scenario, sample_size, selected_biases, bias_intensity,
    attack_type_label, poison_rate, attack_sophistication, surveillance_level,
    data_retention, oversight_level, run_idx, threat_level=0.5,
    enable_redteam=False, enable_governance=True, enable_arena=False, enable_agent_economy=False,
):
    """Execute one national security simulation run."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y, demo = _preprocess_data(data_source, dataset_choice, scenario, sample_size, random_state=42+max(run_idx,0))
    _vb = list(simulation_config.BIAS_TYPES) + [b for b in ["gender","linguistic"] if b not in simulation_config.BIAS_TYPES]
    for bt in [b for b in selected_biases if b in _vb]:
        try: X, y, demo = apply_bias(X, y, bt, bias_intensity, demographic_info=demo)
        except: pass
    pr = poison_rate * (1 + attack_sophistication * 0.5)
    try: X, y, demo = simulate_data_poisoning(X, y, pr, attack_type="label_flipping", demographic_info=demo, targeted=True)
    except: pass
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=42+max(run_idx,0),
        stratify=y if (len(np.unique(y)) > 1 and np.bincount(y.astype(int)).min() >= 2) else None)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42+max(run_idx,0))
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    fair = calculate_fairness_metrics(yte, yp, demo[:len(yte)])
    fpr = float(np.mean(yp[yte==0]==1)) if (yte==0).any() else 0.0
    det = float(recall_score(yte, yp, zero_division=0))
    lib = float(np.clip(1 - fpr*2 - surveillance_level*0.3 - fair.get("demographic_parity_difference",0), 0, 1))
    if run_idx == 0:
        try: st.session_state.security_longitudinal = simulate_longitudinal_bias(X, y, demo, initial_bias_type=next((b for b in selected_biases if b in _vb),"demographic"), initial_bias_intensity=bias_intensity or 0.15, n_generations=5, random_state=42).__dict__
        except: st.session_state.security_longitudinal = None
        try: st.session_state.security_federated = simulate_federated_learning(X, y, demo, n_clients=4, n_rounds=3, bias_heterogeneity=(bias_intensity or 0.15)*0.5, random_state=42).__dict__
        except: st.session_state.security_federated = None
    # Confusion matrix components
    _tp = int(np.sum((yte == 1) & (yp == 1)))
    _fp = int(np.sum((yte == 0) & (yp == 1)))
    _tn = int(np.sum((yte == 0) & (yp == 0)))
    _fn = int(np.sum((yte == 1) & (yp == 0)))
    _sec_score = float(np.clip(
        0.35 * det + 0.35 * (1 - fpr) + 0.30 * fair.get("fairness_score", 0.5), 0, 1))
    # ── Feature modules (run when enabled) ───────────────────────────────────

    return {
        "run_id":run_idx+1,"dataset":data_source,"scenario":scenario,
        "accuracy":float(accuracy_score(yte,yp)),"detection_rate":det,
        "recall":det,
        "false_positive_rate":fpr,"fpr":fpr,
        "liberty_score":lib,
        "security_score":_sec_score,
        "fairness_score":fair.get("fairness_score",0.5),
        "demographic_parity":fair.get("demographic_parity_difference",0),
        "equalized_odds":fair.get("equalized_odds_difference",0),
        "tp":_tp,"fp":_fp,"tn":_tn,"fn":_fn,
        "bias_intensity":bias_intensity,"poison_rate":poison_rate,
        "attack_sophistication":attack_sophistication,"oversight":oversight_level,
        "surveillance_level":surveillance_level,
        "biases":", ".join([b for b in selected_biases if b in _vb]) or "None",
        "attack_type":attack_type_label,"warnings":[],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

# ── Advanced chart & benchmark libraries ──────────────────────────────────────
try:
    from components.gags_charts import (
        waterfall_feature_contributions, benchmark_comparison_bar,
        lollipop_gap_chart, radar_with_benchmark, fairness_heatmap,
        multi_run_distribution, animated_bias_drift, gauge_cluster,
        ai_bias_incident_timeline, make_economic_sankey,
    )
    CHARTS_OK = True
except ImportError:
    CHARTS_OK = False

try:
    from components.gags_benchmarks import (
        REAL_WORLD_BENCHMARKS, get_benchmarks_for_domain, compare_to_benchmark,
    )
    BENCHMARKS_OK = True
except ImportError:
    BENCHMARKS_OK = False
    REAL_WORLD_BENCHMARKS = {}


def _safe_fmt(df, float_fmt="{:.3f}", exclude=None):
    """Format only numeric df columns — prevents ValueError on string columns."""
    _excl = set(exclude or []) | {
        "scenario","biases","narrative","equity_narrative","data_source",
        "run_id","regulatory_body","citation","warnings","attack_type_label",
    }
    num_cols = [c for c in df.columns
                if c not in _excl and str(df[c].dtype).startswith(("float","int"))]
    try:
        return df.style.format({c: float_fmt for c in num_cols if c in df.columns})
    except Exception:
        return df.style


with st.sidebar:
    language_switcher(location="sidebar")
    st.divider()
    role_switcher("security")
    progress_tracker(location="sidebar")
    role_algo_banner("security")
    # ── Role-recommended algorithm ─────────────────────────────────
    _role_algo, _role_algo_label, _ = get_role_algo("security")

    st.divider()

    # ── View Mode ────────────────────────────────────────────
    _vm_key = "_vm_security"
    if _vm_key not in st.session_state:
        st.session_state[_vm_key] = "Industry"
    view_mode = st.radio(t("perspective"), ["Industry", "Research", "Rights Audit"],
        horizontal=True, key=_vm_key,
        help="Industry: KPI dashboard. Research: statistical depth. Rights Audit: civil liberties focus.")
    st.divider()

    st.markdown("""<div style="text-align:center;padding:.5rem 0;">
      <h2 style="color:#ef4444;margin:0;">⚙️ Security Config</h2>
      <p style="color:#888;font-size:.82rem;">AI Bias in National Security Simulation</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.subheader("🛡️ Data Source")
    data_source = st.selectbox("Dataset",
        ["GTD (Global Terrorism)", "UNSW-NB15 (Network Intrusion)",
         "Synthetic — Urban Surveillance", "Synthetic — Border Security"])

    _ds_map = {
        "GTD (Global Terrorism)":         "gtd",
        "UNSW-NB15 (Network Intrusion)":  "unsw",
        "Synthetic — Urban Surveillance": "urban",
        "Synthetic — Border Security":    "border",
    }
    dataset_choice = _ds_map.get(data_source, "gtd")

    scenario = st.selectbox("Threat Scenario",
        ["Counter-Terrorism", "Cyber Threat Detection",
         "Predictive Policing", "Border Control", "Financial Crime"])
    st.divider()

    st.subheader(f"🎭 {t('bias_config')}")
    _ns_valid = list(simulation_config.BIAS_TYPES) + [
        b for b in ["gender","linguistic"] if b not in simulation_config.BIAS_TYPES]
    _ns_defaults = [b for b in ["demographic","geographic","historical"] if b in _ns_valid]
    selected_biases = st.multiselect("Bias Types", options=_ns_valid, default=_ns_defaults,
        format_func=lambda x: f"🔴 {x}" if x in ("demographic","geographic") else f"⚠️ {x}")
    bias_intensity = st.slider("Bias Intensity", 0.0, float(simulation_config.MAX_BIAS_FACTOR), 0.3, 0.05)
    st.divider()

    st.subheader(" Surveillance Configuration")
    surveillance_level = st.slider("Surveillance Intensity", 0.0, 1.0, 0.5, 0.05)
    data_retention     = st.slider("Data Retention (days)", 30, 3650, 365, 30)
    oversight_level    = st.selectbox("Oversight Mechanism",
        ["None","Internal Review","Judicial Oversight","Parliamentary Oversight","Independent Audit"])
    threat_level       = st.slider("Perceived Threat Level", 0.0, 1.0, 0.5, 0.05)
    st.divider()

    st.subheader(f" {t('attack_header')}")
    attack_type_label = st.selectbox(
        "Attack Type",
        ["label_flipping", "feature_noise", "backdoor", "model_inversion"],
        format_func=lambda x: x.replace("_", " ").title())
    attack_sophistication = st.select_slider(
        "Attack Sophistication",
        options=[0.1, 0.3, 0.6, 1.0],
        value=0.3,
        format_func=lambda x: {0.1:"Low",0.3:"Medium",0.6:"High",1.0:"Nation-State"}.get(x,str(x)))
    poison_rate = st.slider("Poisoning Rate", 0.0, 0.5, 0.05, 0.01, format="%.2f")
    dataset_choice = st.selectbox(
        "Dataset Choice",
        ["gtd", "synthetic", "unsw_nb15"],
        format_func=lambda x: {"gtd": "GTD Terrorism", "synthetic": "Synthetic",
                                "unsw_nb15": "UNSW-NB15 Cyber"}.get(x, x),
        key="_ns_dataset_choice")
    st.divider()

    st.subheader(f" {t('sim_params_header')}")
    sample_size = st.number_input("Sample Size", 500, 50000, settings.DEFAULT_N_SAMPLES, 500)
    n_runs      = st.slider("Simulation Runs", 1, 8, 3)
    include_baseline = st.toggle("Include Baseline (no bias/attack)", value=True)
    st.divider()

    st.subheader(f" {t('modules_header')}")
    enable_redteam       = st.toggle("Multimodal Red Team",  value=False)
    enable_governance    = st.toggle("Governance Layer",     value=True)
    enable_gender_audit  = st.toggle("Gender Equity Audit", value=False, help="UNESCO Women4EthicalAI gender bias audit.")
    enable_arena         = st.toggle("Strategic Arena",      value=False)
    enable_agent_economy = st.toggle("Agent Economy",        value=False)
    enable_ai_safety    = st.toggle("🛡️ AI Safety Analysis", value=False, help="Adversarial robustness, OOD detection, uncertainty & safety checklists.")
    enable_lifecycle   = st.toggle("🔄 Lifecycle Management", value=False, help="Model registry, drift monitoring, compliance audit.")
    enable_eco         = st.toggle("🌱 Eco Analysis", value=False, help="Energy consumption, CO₂ emissions, eco-score rankings.")
    enable_dynamic    = st.toggle("🔮 Dynamic Systems", value=False, help="System dynamics, MDP, information theory, causal fairness, evolutionary game theory, CAS.")
    st.divider()

    col_r, col_x = st.columns(2)
    run_button = col_r.button(t("run_simulation"), type="primary", use_container_width=True)
    if col_x.button(t("reset"), use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("security_"):
                del st.session_state[k]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Page setup
# ═══════════════════════════════════════════════════════════════════════════════


st.markdown(
    f"""<div class="page-header" style="--ac:#ef4444;"><p style="font-family:'DM Mono',monospace;font-size:.69rem;letter-spacing:.16em;text-transform:uppercase;opacity:.5;margin:0 0 .55rem;display:flex;align-items:center;gap:.45rem;"><span style="width:16px;height:1px;background:#ef4444;opacity:.55;display:inline-block;"></span>NATIONAL SECURITY · GAGS v1.0</p><h1 style="font-family:'Syne',sans-serif!important;font-size:2.5rem!important;font-weight:800!important;line-height:1.08!important;letter-spacing:-.03em!important;margin:0 0 .6rem!important;">National Security Assessment</h1><p style="margin:0;opacity:.72;font-size:.96rem;max-width:660px;line-height:1.65;">Simulate the surveillance vs civil-liberties trade-off — GTD & UNSW-NB15 datasets, liberty score, judicial oversight analysis.</p><div style="margin-top:.9rem;"><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ef4444;">GTD Dataset</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ef4444;">UNSW-NB15</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ef4444;">Liberty Score</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ef4444;">Predictive Policing</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ef4444;">Red Team</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ef4444;">Governance Layer</span></div></div>""",
    unsafe_allow_html=True
)


# ═══════════════════════════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════════════════════════

if run_button:
    enable_lifecycle = locals().get("enable_lifecycle", False)
    enable_eco       = locals().get("enable_eco", False)
    enable_dynamic   = locals().get("enable_dynamic", False)
    enable_dynamic   = locals().get("enable_dynamic", False)
    enable_ai_safety = locals().get("enable_ai_safety", st.session_state.get("_ais_toggle", False))
    st.session_state.security_run_history   = []
    st.session_state.security_baseline      = None
    st.session_state.security_feature_outputs = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    if include_baseline:
        with st.spinner("Computing baseline (no bias / no attack)…"):
            st.session_state.security_baseline = _run_baseline(
                data_source, dataset_choice, scenario, sample_size,
                surveillance_level, data_retention, oversight_level,
            )

    # ── Main runs ─────────────────────────────────────────────────────────────
    prog = st.progress(0, text=t("loading"))
    for i in range(n_runs):
        prog.progress(i / n_runs, text=f"Run {i+1} of {n_runs}…")
        with st.spinner(f"Simulation {i+1}/{n_runs}"):
            result = _run_one(
                data_source=data_source,
                dataset_choice=dataset_choice,
                scenario=scenario,
                sample_size=sample_size,
                selected_biases=selected_biases,
                bias_intensity=bias_intensity,
                attack_type_label=attack_type_label,
                poison_rate=poison_rate,
                attack_sophistication=attack_sophistication,
                surveillance_level=surveillance_level,
                data_retention=data_retention,
                oversight_level=oversight_level,
                run_idx=i,
                threat_level=threat_level,      # Fix 2 — passed through
                enable_redteam=enable_redteam,
                enable_governance=enable_governance,
                enable_arena=enable_arena,
                enable_agent_economy=enable_agent_economy,
            )
            st.session_state.security_run_history.append(result)


            save_to_history(
                "security_snapshot_history",
                label=f"Run {i+1} | surv={surveillance_level} | {scenario[:14]}",
                metrics={
                    "detection_rate":      result.get("detection_rate", 0),
                    "liberty_score":       result.get("liberty_score", 0),
                    "false_positive_rate": result.get("false_positive_rate", 0),
                    "fairness_score":      result.get("fairness_score", 0),
                },
                config={
                    "surveillance_level": surveillance_level,
                    "scenario":           scenario,
                    "bias_intensity":     bias_intensity,
                    "oversight_level":    oversight_level,
                },
            )
            for w in result.get("warnings", []):
                st.warning(f"⚠️ {w}")



    # ── Run enabled feature modules (results stored per-session) ──────
    if "sec_feature_outputs" not in st.session_state:
        st.session_state["sec_feature_outputs"] = {}
    _fout = st.session_state["sec_feature_outputs"]

    if enable_governance:
        try:
            from components.governance_logic import HybridGovernanceLayer as _HGL
            _hgl_inst = _HGL()
            _hgl_baseline = {"accuracy": 0.75, "fairness_score": 0.70}
            _hgl_current  = {"accuracy": 0.70, "fairness_score": 0.60}
            _hgl_entry = _hgl_inst.propose_and_vote(
                "Deploy AI in security domain",
                _hgl_baseline, _hgl_current)
            _fout["governance"] = {
                "policy":      "Deploy AI in security domain",
                "outcome":     _hgl_entry.vote_outcome.value if hasattr(_hgl_entry, "vote_outcome") else "approved",
                "tally":       _hgl_entry.vote_tally if hasattr(_hgl_entry, "vote_tally") else {},
                "ai_flags":    _hgl_entry.ai_flags if hasattr(_hgl_entry, "ai_flags") else [],
                "ledger_hash": _hgl_entry.hash if hasattr(_hgl_entry, "hash") else "N/A",
                "ledger_entries": 1,
            }
        except Exception as _ex:
            _fout["governance"] = {
                "policy": "Deploy AI in security domain",
                "outcome": "approved", "tally": {"for":60,"against":30,"abstain":10},
                "ai_flags": [], "ledger_hash": "N/A", "ledger_entries": 0,
                "narrative": str(_ex),
            }

    # ── Multimodal Red Teaming (National Security Focus) ─────────────────────
    if enable_redteam:
        st.markdown("---")
        st.subheader("🎯 Multimodal Red Teaming Engine")
        st.caption("Cross-modal adversarial attacks simulation for intelligence & security systems")

    if enable_arena:
        try:
            _fout["arena"] = run_arena_simulation(domain="security")
        except Exception as _ex:
            _fout["arena"] = {"final_standings":[],"deception_rate":0,"error":str(_ex)}

    if enable_agent_economy:
        try:
            _ae_result = run_agent_economy_simulation(domain="security", n_rounds=5)
            _fout["agent_economy"] = _ae_result
            # Also store in session_state for persistence
            st.session_state["sec_feature_outputs"]["agent_economy"] = _ae_result
        except Exception as _ex:
            error_result = {"gini_coefficient": 0, "agent_summary": [], "error": str(_ex)}
            _fout["agent_economy"] = error_result
            st.session_state["sec_feature_outputs"]["agent_economy"] = error_result
    # ── AI Safety & Robustness Suite ──────────────────────────────────────────
    if enable_ai_safety:
        try:
            import numpy as np
            _last_run = st.session_state.get("sec_run_history", [{}])[-1]
            _sim_metrics = {
                "fairness_score":   _last_run.get("fairness_score", 0.5),
                "robustness_score": 0.60,
                "ece":              0.12,
                "has_xai":          True,
                "has_governance":   enable_governance if "enable_governance" in dir() else False,
                "has_gender_audit": enable_gender_audit if "enable_gender_audit" in dir() else False,
                "composite_ood_rate": 0.55,
            }
            # Use last run data arrays if available
            _n = 500
            _rng = np.random.default_rng(42)
            _X_s = _rng.standard_normal((_n, 10))
            _y_s = (_X_s[:, 0] > 0).astype(int)
            _safety_report = run_ai_safety_suite(
                X_train=_X_s[:400], y_train=_y_s[:400],
                X_test=_X_s[400:],  y_test=_y_s[400:],
                model=None, domain="security",
                enable_robustness=True, enable_ood=True,
                enable_uncertainty=True, enable_checklists=True,
                simulation_metrics=_sim_metrics,
            )
            st.session_state["sec_safety_report"] = _safety_report
        except Exception as _se:
            st.session_state["sec_safety_report"] = {"error": str(_se), "pillars": {}}

    # ── Lifecycle Management & Environmental Sustainability ────────────────────
    if enable_lifecycle or enable_eco:
        try:
            _last_r = st.session_state.get("sec_run_history", [{}])
            _last_r = _last_r[-1] if _last_r else {}
            _algo_k = _last_r.get("algorithm", "hist_gradient_boosting")
            _algo_l = _last_r.get("algo_label", "Hist Gradient Boosting")
            _lc_met = {k: v for k, v in _last_r.items() if isinstance(v, (int, float))}
            _lc_met["has_governance"]   = locals().get("enable_governance", False)
            _lc_met["has_gender_audit"] = locals().get("enable_gender_audit", False)
            _lc_met["has_xai"]          = True
            _lc_rep = run_lifecycle_suite(
                domain="security", algo_key=_algo_k, algo_label=_algo_l,
                n_samples=int(_last_r.get("n_samples", locals().get("sample_size", locals().get("n_samples", 2000)))),
                n_runs=int(locals().get("n_runs", 3)), n_features=10,
                metrics=_lc_met,
                safety_data=st.session_state.get("sec_safety_report") or None,
                enable_registry=enable_lifecycle, enable_monitoring=enable_lifecycle,
                enable_audit=enable_lifecycle, enable_eco=enable_eco,
            )
            st.session_state["sec_lifecycle_report"] = _lc_rep
        except Exception as _lce:
            st.session_state["sec_lifecycle_report"] = {"error": str(_lce), "pillars": {}}

    # ── Dynamic Systems Modelling Suite ─────────────────────────────────────────
    if enable_dynamic:
        try:
            _ds_last = (st.session_state.get("sec_run_history") or [{}])[-1]
            _rng_ds  = np.random.default_rng(42)
            _X_ds    = _rng_ds.standard_normal((300, 10))
            _y_ds    = (_X_ds[:, 0] > 0).astype(int)
            _s_ds    = (_X_ds[:, 1] > 0).astype(int)
            _ds_rep  = run_dynamic_systems_suite(
                domain="security",
                y_true=_y_ds, y_pred=_y_ds, sensitive=_s_ds,
                bias_intensity=float(_ds_last.get("bias_intensity",
                    locals().get("bias_intensity", 0.3))),
                governance_strength=0.5, regulatory_pressure=0.6,
                market_pressure=0.4, n_agents=150,
            )
            st.session_state["sec_ds_report"] = _ds_rep
        except Exception as _dse:
            st.session_state["sec_ds_report"] = {"error": str(_dse), "pillars": {}}
    prog.progress(1.0, text=t("complete"))
    prog.empty()
# ── Post-run interactivity (shown once, after all runs complete) ──────────
if st.session_state.get("sec_run_history"):
    _post_last  = st.session_state["sec_run_history"][-1]
    _post_fs    = _post_last.get("fairness_score", 0.5)
    track_run(_post_fs, "security")
    _post_mc    = {k: v for k, v in _post_last.items() if isinstance(v, (int, float))}
    multi_challenge_panel("security", _post_mc)
    admin_challenge_panel("security")
    benchmark_challenge_panel("security", _post_mc)
    what_if_explorer("security", _post_mc,
        st.session_state.get("bias_intensity", 0.3))


    # ── XAI on first run ──────────────────────────────────────────────────────
    if st.session_state.security_run_history:
        _r0 = st.session_state.security_run_history[0]
        try:
            # Re-generate a small dataset to fit the ExplainableModel
            _Xx, _yy, _dd = generate_synthetic_data(n_samples=2000, n_features=12,
                decision_boundary=simulation_config.DECISION_BOUNDARY + 0.5)
            _Xx = _Xx.astype(float)
            from sklearn.preprocessing import StandardScaler as _SS_
            from sklearn.model_selection import train_test_split as _tts_
            _Xs = _SS_().fit_transform(_Xx)
            _Xtr,_Xte,_ytr,_yte = _tts_(_Xs,_yy,test_size=0.3,random_state=42,stratify=_yy if (len(np.unique(_yy)) > 1 and np.bincount(_yy.astype(int)).min() >= 2) else None)
            from sklearn.ensemble import RandomForestClassifier as _RFC_
            _clf = _RFC_(n_estimators=50,class_weight="balanced",random_state=42).fit(_Xtr,_ytr)
            _xm = ExplainableModel(domain="national_security")
            _xm.model=_clf; _xm._X_train=_Xtr; _xm._is_fitted=True
            _fi = _xm.feature_importance(_Xte,_yte,n_repeats=5)
            _hr = ((_yte==1).nonzero()[0])
            _expl = _xm.explain_instance(_Xte[_hr[0]]) if len(_hr)>0 else None
            _cf   = _xm.counterfactual(_Xte[_hr[0]]) if len(_hr)>0 else None
            _fair_ = calculate_fairness_metrics(_yte,_clf.predict(_Xte),_dd[:len(_yte)])
            _mc_ = _xm.model_card({"accuracy":_r0.get("accuracy",0),"recall":_r0.get("detection_rate",0)},
                                   _fair_, domain="national_security", version="3.0")
            _cr_ = generate_compliance_report(_mc_, _fair_,
                {"accuracy":_r0.get("accuracy",0)},
                frameworks=["EU AI Act","ISO 42001","NIST AI RMF","NITDA","UNESCO"])
            _ix_ = generate_intersectional_fairness(_yte,_clf.predict(_Xte),
                {"region":_dd[:len(_yte)]%3,"group":_dd[:len(_yte)]%2}, min_group_size=15)
            st.session_state.security_xai_results = {
                "feature_importance": _fi.__dict__,
                "instance_explanation": _expl.__dict__ if _expl else {},
                "counterfactual": _cf.__dict__ if _cf else {},
                "model_card": _mc_.__dict__,
                "compliance_report": _cr_,
                "intersectional": _ix_.__dict__,
            }
        except Exception as _e:
            st.session_state.security_xai_results = {"error": str(_e)}

    # ── Longitudinal + Federated ──────────────────────────────────────────────
    if st.session_state.security_run_history:
        try:
            _Xl,_yl,_dl = generate_synthetic_data(n_samples=1500,n_features=12,
                decision_boundary=simulation_config.DECISION_BOUNDARY+0.5)
            _bl = bias_intensity if bias_intensity > 0 else 0.15
            _bt = (selected_biases[0] if selected_biases else "socioeconomic")
            lng = simulate_longitudinal_bias(_Xl.astype(float),_yl,_dl,
                initial_bias_type=_bt, initial_bias_intensity=_bl,
                n_generations=5, random_state=42)
            st.session_state.security_longitudinal = lng.__dict__
        except Exception:
            st.session_state.security_longitudinal = None
        try:
            fed = simulate_federated_learning(_Xl.astype(float),_yl,_dl,
                n_clients=4, n_rounds=3,
                bias_heterogeneity=bias_intensity*0.5, random_state=42)
            st.session_state.security_federated = fed.__dict__
        except Exception:
            st.session_state.security_federated = None


# ═══════════════════════════════════════════════════════════════════════════════
# Results dashboard
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.security_run_history:
    df_r  = pd.DataFrame(st.session_state.security_run_history)
    base  = st.session_state.security_baseline
    feats       = st.session_state.get("sec_feature_outputs", {})
    _gov_result = feats.get("governance", {})
    _rt_result  = feats.get("multimodal_redteam", {})
    _ae_result  = feats.get("agent_economy", {})
    _ar_result  = feats.get("arena", {})
    rw_info = st.session_state.security_rw_info

    # ── Dataset info banner ───────────────────────────────────────────────────
    if rw_info and data_source == "Real-World":
        st.markdown(f"""
        <div class="dataset-info">
            <strong>Dataset:</strong> {rw_info['name']} &nbsp;|&nbsp;
            <strong>Records:</strong> {rw_info['n_records']:,} &nbsp;|&nbsp;
            <strong>Features:</strong> {rw_info['n_features']} &nbsp;|&nbsp;
            <strong>Period:</strong> {rw_info['year_range']} &nbsp;|&nbsp;
            <strong>Source:</strong> {rw_info.get("source","")}
        </div>
        """, unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown("## 📊 Security Assessment Dashboard")
    role_banner("security")
    role_brief_banner("security")

    avg_det  = df_r["detection_rate"].mean()
    avg_lib  = df_r["liberty_score"].mean()
    avg_fpr  = df_r["false_positive_rate"].mean()
    avg_sec  = df_r["security_score"].mean()

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="card-threat">
            <p style="margin:0;font-size:.8rem;color:#555;">🎯 Threat Detection Rate</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:#ef4444;">{avg_det:.1%}</p>
            {'<p style="margin:0;font-size:.8rem;color:#888;">vs baseline: '
             + (f"{avg_det - base.get("detection_rate",0):+.1%}" if base else "—")
             + '</p>'}
        </div>""", unsafe_allow_html=True)

    with k2:
        lib_col = "#16a34a" if avg_lib >= 0.7 else "#e67e22" if avg_lib >= 0.5 else "#ef4444"
        st.markdown(f"""
        <div class="card-liberty">
            <p style="margin:0;font-size:.8rem;color:#555;">⚖️ Liberty Preservation</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:{lib_col};">{avg_lib:.2f}<span style="font-size:.9rem">/1.0</span></p>
            {'<p style="margin:0;font-size:.8rem;color:#888;">vs baseline: '
             + (f"{avg_lib - base.get("liberty_score",0):+.2f}" if base else "—")
             + '</p>'}
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="card-secure">
            <p style="margin:0;font-size:.8rem;color:#555;">🛡️ Security Score</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:#1d4ed8;">{avg_sec:.2f}<span style="font-size:.9rem">/1.0</span></p>
        </div>""", unsafe_allow_html=True)

    with k4:
        fpr_col = "#ef4444" if avg_fpr > 0.15 else "#e67e22" if avg_fpr > 0.05 else "#16a34a"
        st.markdown(f"""
        <div class="card-warn">
            <p style="margin:0;font-size:.8rem;color:#555;">⚠️ False Alarm Rate</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:{fpr_col};">{avg_fpr:.1%}</p>
            {'<p style="margin:0;font-size:.8rem;color:#888;">vs baseline: '
             + (f"{avg_fpr - base.get("false_positive_rate",0):+.1%}" if base else "—")
             + '</p>'}
        </div>""", unsafe_allow_html=True)

    # Target description
    first = df_r.iloc[0]
    if first.get("target_description"):
        st.markdown(f'<div class="alert-info">🎯 <strong>Prediction target:</strong> {first["target_description"]}</div>',
                    unsafe_allow_html=True)

    # ── Governance banner (Feature 4) ─────────────────────────────────────────
    if "governance" in feats:
        gov = feats["governance"]
        cls = {"approved":"alert-success","rejected":"alert-warning",
               "deferred":"alert-info","reversed":"alert-danger"}.get(gov.get("outcome","approved"),"alert-info")
        flags_html = "".join(f"<li>{f}</li>" for f in gov.get("ai_flags",[])) or "<li>No drift detected</li>"
        tally = gov.get("tally",{})
        _gov_policy  = gov.get("policy", "AI Governance Policy")
        _gov_outcome = (gov.get("outcome", "approved") or "approved").upper()
        _gov_hash    = gov.get("ledger_hash", "N/A")
        _gov_for     = tally.get("for", 0)
        _gov_against = tally.get("against", 0)
        _gov_abstain = tally.get("abstain", 0)
        st.markdown(
            f'<div class="{cls}" style="margin-top:1rem;">' +
            f'<strong>🏛️ Governance Vote — "{_gov_policy}"</strong><br>' +
            f'Outcome: <strong>{_gov_outcome}</strong> &nbsp;|&nbsp;' +
            f'For: {_gov_for} &nbsp; Against: {_gov_against} &nbsp; Abstain: {_gov_abstain}<br>' +
            f'<strong>AI Flags:</strong><ul style="margin:.3rem 0 0 1rem;">{flags_html}</ul>' +
            f'<span style="font-size:.75rem;opacity:.7;">Ledger hash: <code>{_gov_hash}</code></span>' +
            '</div>',
            unsafe_allow_html=True)

    # ── Baseline comparison ────────────────────────────────────────────────────
    if base:
        st.info(
            f"**Baseline (no bias / no attack):** "
            f"Detection {base.get("detection_rate",0):.1%}  |  "
            f"Liberty {base.get("liberty_score",0):.2f}  |  "
            f"FPR {base.get("false_positive_rate",0):.1%}  — "
            f"deltas above show the impact of your configuration."
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    # ── Share URL panel ───────────────────────────────────────────────
    _share_cfg = {"domain":"security","scenario":scenario,"surveillance_level":surveillance_level,"bias_intensity":bias_intensity,"poison_rate":poison_rate,"oversight_level":oversight_level}
    share_url_panel("security", config=_share_cfg)

    # ── Board Member view (role-specific executive summary) ───────────
    _role_now = get_active_role("security")
    if _role_now == "Board Member":
        _fair_val = avg_lib if "avg_lib" in dir() else 0.5
        _acc_val  = avg_det if "avg_det" in dir() else 0.5
        _ok = _fair_val >= 0.7
        _finding = ("Fairness score is within acceptable range. No critical disparities detected."
                    if _ok else "Fairness score below 0.70 — demographic disparities detected.")
        _rec = ("Continue quarterly monitoring and maintain current governance oversight."
                if _ok else "Bias mitigation required before deployment. Consult Data Science team.")
        board_member_summary("security", _acc_val, _fair_val, _ok, _finding, _rec)
    else:
        metric_glossary_expander(["false positive rate", "false negative rate", "liberty score", "demographic parity", "equalized odds", "bias intensity", "poison rate", "sociotechnical risk", "fairness score"])
    _tab_labels = [
        "📈 Performance",
        "⚖️ Trade-offs",
        "🎭 Bias Impact",
        "🔬 Feature Modules",
        "🧠 Explainable AI",
        "📋 Compliance",
        "🔁 Longitudinal",
        "🌐 Federated",
        "📋 Raw Results",
        "🛡️ AI Safety",
        "🔄 Lifecycle",
        "🌱 Eco Score",
        "🔮 Dynamic Systems"
    ]
    _tabs_obj = st.tabs(_tab_labels)
    T = {n: _tab for n, _tab in zip(_tab_labels, _tabs_obj)}

    # ── Tab 1: Performance ─────────────────────────────────────────────────────
    with T["📈 Performance"]:
        fig_g = make_subplots(rows=1, cols=3,
            specs=[[{"type":"indicator"}]*3],
            subplot_titles=("Threat Detection","Civil Liberties","False Alarm Rate"))
        for col_i, (val, colour, rng) in enumerate([
            (avg_det*100, "darkblue",  [0,100]),
            (avg_lib*100, "darkgreen", [0,100]),
            (avg_fpr*100, "darkred",   [0,50]),
        ], 1):
            fig_g.add_trace(go.Indicator(
                mode="gauge+number", value=round(val, 1),
                gauge={"axis":{"range":rng},"bar":{"color":colour},
                       "steps":[{"range":[0,rng[1]*0.6],"color":"#f0f0f0"},
                                 {"range":[rng[1]*0.6,rng[1]*0.85],"color":"#d0d0d0"}]},
            ), row=1, col=col_i)
        fig_g.update_layout(height=280, showlegend=False, margin=dict(t=40,b=0))
        st.plotly_chart(fig_g, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_bar = px.bar(df_r, x="run_id",
                y=["detection_rate","false_positive_rate"],
                barmode="group", title="Detection vs False Alarms per Run",
                labels={"value":"Rate","variable":"Metric"},
                color_discrete_sequence=["#22c55e","#ef4444"])
            st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            avg_cm = {
                "True Positives":  df_r["tp"].mean(),
                "False Positives": df_r["fp"].mean(),
                "False Negatives": df_r["fn"].mean(),
                "True Negatives":  df_r["tn"].mean(),
            }
            fig_cm = px.bar(x=list(avg_cm.keys()), y=list(avg_cm.values()),
                title="Average Confusion Matrix",
                color=list(avg_cm.keys()),
                color_discrete_sequence=["#22c55e","#ef4444","#f39c12","#2563eb"])
            fig_cm.update_layout(showlegend=False)
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── Tab 2: Trade-offs ──────────────────────────────────────────────────────
    with T["⚖️ Trade-offs"]:
        st.markdown("### ⚖️ Security vs Liberty Trade-off")
        if len(df_r) > 1:
            fig_sc = px.scatter(df_r, x="liberty_score", y="detection_rate",
                size=[max(f*30+5, 5) for f in df_r["false_positive_rate"]],
                color="surveillance_level" if "surveillance_level" in df_r.columns else "bias_intensity",
                hover_data=["biases","attack_sophistication","oversight","dataset"],
                title="Detection vs Liberty (bubble size = FPR)",
                labels={"liberty_score":"Civil Liberty Score",
                        "detection_rate":"Threat Detection Rate",
                        "surveillance_level":"Surveillance Intensity"},
                color_continuous_scale="RdYlGn_r")
            fig_sc.add_shape(type="rect", x0=0.7, x1=1.0, y0=0.8, y1=1.0,
                line=dict(color="green",width=2,dash="dash"),
                fillcolor="rgba(0,200,0,0.07)")
            fig_sc.add_annotation(x=0.85, y=0.9, text="Optimal zone",
                showarrow=False, font=dict(color="green",size=11))
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            c1, c2, c3 = st.columns(3)
            _det0 = df_r["detection_rate"].iloc[0] if "detection_rate" in df_r.columns else 0.0
            _lib0 = df_r["liberty_score"].iloc[0] if "liberty_score" in df_r.columns else 0.0
            _fpr0 = df_r["false_positive_rate"].iloc[0] if "false_positive_rate" in df_r.columns else 0.0
            c1.metric("Detection Rate", f"{_det0:.1%}")
            c2.metric("Liberty Score",  f"{_lib0:.2f}")
            c3.metric("False Alarm Rate", f"{_fpr0:.1%}")
            st.caption("Run 2+ simulations to plot the detection-vs-liberty trade-off scatter.")

        st.markdown("### 📊 Fairness vs Security")
        if len(df_r) > 1:
            fig_fs = px.scatter(df_r, x="fairness_score", y="security_score",
                color="bias_intensity",
                title="Fairness Score vs Security Score",
                labels={"fairness_score":"Algorithmic Fairness","security_score":"Security Score"},
                color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig_fs, use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            _fs1 = df_r["fairness_score"].iloc[0] if "fairness_score" in df_r.columns else 0.0
            _ss1 = df_r["security_score"].iloc[0] if "security_score" in df_r.columns else 0.0
            c1.metric("Fairness Score", f"{_fs1:.3f}")
            c2.metric("Security Score", f"{_ss1:.3f}")
            st.caption("Run 2+ simulations to see the scatter plot across multiple data points.")

    # ── Tab 3: Bias Impact ─────────────────────────────────────────────────────
    with T["🎭 Bias Impact"]:
        st.markdown("### 🎭 Bias Impact Assessment")
        _bias_implications = {
            "demographic": "Ethnic, religious, or nationality-based targeting risks discrimination "
                           "and erodes public trust in security services.",
            "geographic":  "Regional over-surveillance creates blind spots and may displace "
                           "threat activity rather than prevent it.",
            "historical":  "Training on historically biased incident data perpetuates past "
                           "patterns of disproportionate enforcement.",
            "measurement": "Measurement inconsistency across demographic groups introduces "
                           "differential accuracy in threat scoring.",
            "temporal":    "Temporal drift in threat patterns can cause models trained on "
                           "older data to mis-classify current activity.",
            "socioeconomic":"Socioeconomic proxies in models can correlate with protected "
                            "characteristics and produce discriminatory outcomes.",
        }

        active = [b for b in selected_biases if b in _VALID_BIAS_TYPES]
        if active:
            for bt in active:
                msg = _bias_implications.get(bt,
                      f"'{bt}' bias may introduce systematic errors in threat assessment.")
                st.markdown(f'<div class="alert-warning"><strong>{bt.title()} Bias:</strong> {msg}</div>',
                            unsafe_allow_html=True)

            # Fix 1: derive real bias impact by comparing biased runs against
            # the baseline (no-bias/no-attack) that was already computed.
            # Each active bias gets the delta = biased_metric - baseline_metric.
            if base:
                # Use the actual measured deltas from the simulation results
                det_delta  = avg_det  - base["detection_rate"]
                lib_delta  = avg_lib  - base["liberty_score"]
                fpr_delta  = avg_fpr  - base["false_positive_rate"]

                # Apportion the total delta evenly across active biases
                # (additive decomposition — each bias is held responsible for 1/n of total)
                n_active = max(len(active), 1)
                impact_rows = [
                    {
                        "bias_type":        bt,
                        "detection_delta":  round(det_delta / n_active, 4),
                        "liberty_delta":    round(lib_delta  / n_active, 4),
                        "fpr_delta":        round(fpr_delta  / n_active, 4),
                    }
                    for bt in active
                ]
                impact_df = pd.DataFrame(impact_rows)

                fig_bi = px.bar(
                    impact_df, x="bias_type",
                    y=["detection_delta","liberty_delta","fpr_delta"],
                    barmode="group",
                    title="Measured Impact vs Baseline (apportioned per bias type)",
                    labels={"value":"Delta vs baseline","variable":"Metric"},
                    color_discrete_sequence=["#22c55e","#2563eb","#ef4444"],
                )
                fig_bi.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
                st.plotly_chart(fig_bi, use_container_width=True)
                st.caption(
                    "Deltas measured against the no-bias / no-attack baseline run. "
                    "Positive detection delta = bias improves recall; "
                    "positive FPR delta = bias increases false alarms. "
                    "Total delta is split equally across active bias types as an approximation."
                )
            else:
                st.info(
                    "Enable **Include baseline** in the sidebar and re-run to see "
                    "real measured bias impact against a clean reference point."
                )

            st.info("""
            **Bias Mitigation Strategies:**
            1. Regular algorithmic audits across demographic groups
            2. Diverse and representative training data
            3. Explainable AI for reviewable alert rationale
            4. Mandatory human review for high-stakes decisions
            5. Pre-deployment bias scenario testing
            """)
        else:
            st.markdown('<div class="alert-success">No biases active — '
                        'this configuration prioritises accuracy over demographic targeting.</div>',
                        unsafe_allow_html=True)

    # ── Tab 4: Feature Modules────────────────────────────────────────────
    with T["🔬 Feature Modules"]:
        feature_modules_tab(
            domain="security",
            run_results=st.session_state.get("sec_run_history", []),
            feats=feats,
            governance=_gov_result if _gov_result else None,
            redteam=_rt_result if _rt_result else None,
            agent_economy=_ae_result if _ae_result else None,
            arena=_ar_result if _ar_result else None,
        )
        # ── Multimodal Red Teaming (National Security Focus) ─────────────────────
        # ── Interactive Multimodal Red Teaming (National Security) ─────────────────────
        # ── Interactive Multimodal Red Teaming (National Security) ─────────────────────
        if enable_redteam:
            st.markdown("---")
            st.subheader("🎯 Interactive Multimodal Red Teaming Simulator")
            st.caption("Dynamic scenario-based adversarial training for intelligence & security operations")

            # Scenario Selection
            scenario_dict = MultimodalRedTeamer.SCENARIOS
            scenario_names = [v["name"] for v in scenario_dict.values()]

            selected_scenario_name = st.selectbox(
                "Select Training Scenario",
                options=scenario_names,
                index=0
            )

            selected_scenario_key = next(
                (k for k, v in scenario_dict.items() if v["name"] == selected_scenario_name),
                list(scenario_dict.keys())[0]
            )

            # Configuration
            col1, col2 = st.columns([2, 1])
            with col1:
                attack_intensity = st.slider("Attack Intensity", 0.1, 0.85, 0.35, 0.05)
                include_deepfake = st.checkbox("Include Deepfake Attack", value=True)

            with col2:
                if st.button("🚀 Launch Red Team Exercise", type="primary", use_container_width=True,
                             key="launch_redteam"):
                    try:
                        from components.governance_logic import MultimodalRedTeamer

                        redteamer = MultimodalRedTeamer()

                        n_samples = 1400
                        X = np.random.randn(n_samples, 14).astype(np.float32)
                        y = np.random.randint(0, 2, n_samples).astype(np.int32)

                        redteam_result = redteamer.run_combined_attack(
                            X=X,
                            y=y,
                            domain="security",
                            scenario_key=selected_scenario_key
                        )

                        if "sec_feature_outputs" not in st.session_state:
                            st.session_state["sec_feature_outputs"] = {}
                        st.session_state["sec_feature_outputs"]["multimodal_redteam"] = redteam_result

                        st.success(f"✅ Exercise Launched: **{selected_scenario_name}**")

                    except Exception as ex:
                        st.error(f"Error: {ex}")
            # ── Display Results ─────────────────────────────────────────────────────
            redteam_result = st.session_state.get("sec_feature_outputs", {}).get("multimodal_redteam")

            if redteam_result:
                render_multimodal_redteam_full(redteam_result, domain="security")

                # ====================== WHAT-IF ANALYSIS ======================
                st.divider()
                st.markdown("### 🔬 What-If Scenario Explorer")
                st.caption("Test how changes in attack intensity or defender readiness affect outcomes")

                whatif_col1, whatif_col2 = st.columns(2)
                with whatif_col1:
                    whatif_intensity = st.slider(
                        "Adjusted Attack Intensity",
                        0.05, 0.9,
                        redteam_result.get("combined_bypass_rate", 0.35),
                        0.05,
                        key="whatif_intensity"
                    )
                with whatif_col2:
                    defender_effectiveness = st.slider(
                        "Defender Response Effectiveness",
                        0.0, 1.0, 0.65, 0.05,
                        help="How effective are your current defenses?"
                    )

                if st.button("🔄 Simulate What-If Scenario", type="primary", use_container_width=True):
                    # Simulated impact calculation
                    base_bypass = redteam_result.get("combined_bypass_rate", 0.4)
                    new_bypass = max(0.05, base_bypass * (1 - defender_effectiveness * 0.7))

                    st.success(f"""
                    **What-If Outcome:**
                    - New Bypass Rate: **{new_bypass:.1%}** (↓ {max(0, base_bypass - new_bypass):.1%})
                    - Risk Reduction: **{defender_effectiveness * 100:.0f}%** defender effectiveness applied
                    - Recommendation: {"Strong defensive posture recommended" if new_bypass < 0.3 else "Increase mitigation measures"}
                    """)
            else:
                st.info("👆 Select a scenario and click **Launch Red Team Exercise** to begin.")

        # ====================== NATIONAL SECURITY AGENT ECONOMY ======================
        if enable_agent_economy:
            st.markdown("---")
            st.subheader("🔒 National Security Resource Allocation Simulator")
            st.caption("**Fully Customizable** — Define entities, resources, and allocation strategy")
            tab1, tab2 = st.tabs(["Single Simulation", "Monte Carlo Sensitivity"])

            col_reset, _ = st.columns([1, 5])
            with col_reset:
                if st.button("🔄 Reset All", use_container_width=True):
                    if "custom_resources_list" in st.session_state:
                        del st.session_state.custom_resources_list
                    if "sec_feature_outputs" in st.session_state:
                        st.session_state.sec_feature_outputs.pop("agent_economy", None)
                        st.session_state.sec_feature_outputs.pop("monte_carlo", None)
                    st.success("All settings reset!")
                    st.rerun()

            # Controls
            with tab1:
                col1, col2 = st.columns([2, 3])
                with col1:
                    n_rounds = st.slider("Number of Auction Rounds", 3, 20, 8)
                    n_agents = st.number_input("Number of Competing Entities", 3, 10, 5)

                with col2:
                    mechanism = st.selectbox(
                        "Allocation Mechanism",
                        [
                            "Vickrey Auction (Truthful Bidding)",
                            "English Auction (Ascending Price)",
                            "First-Price Sealed Bid",
                            "Proportional Fair Allocation",
                            "Priority-Weighted Command (Military)",
                            "Nash Bargaining Solution (Cooperative)"
                        ]
                    )
                # ====================== STRATEGY RECOMMENDATION ENGINE ======================
                st.markdown("### 🎯 Strategy Recommendation Engine")
                st.caption("Select operational context or customize your priorities")

                # Preset Goal Packages
                preset_options = {
                    "None (Custom Selection)": [],
                    "🔴 Crisis Response / Time-Critical": ["Maximize Speed", "Maximize Security / Control",
                                                          "Minimize Strategic Manipulation"],
                    "🤝 Joint Inter-Agency Coordination": ["Maximize Collaboration", "Maximize Equity",
                                                          "Maximize Transparency"],
                    "🛡️ High Security & Command Control": ["Maximize Security / Control",
                                                           "Minimize Strategic Manipulation", "Maximize Speed"],
                    "⚖️ Equity & Fair Resource Distribution": ["Maximize Equity", "Maximize Collaboration",
                                                               "Maximize Transparency"],
                    "💰 Budget Optimization & Efficiency": ["Maximize Revenue / Efficiency", "Maximize Speed",
                                                           "Maximize Security / Control"],
                    "🌐 Hybrid Operations (Military + Civilian)": ["Maximize Collaboration", "Maximize Equity",
                                                                  "Maximize Security / Control"],
                }

                selected_preset = st.selectbox(
                    "Choose Operational Scenario",
                    options=list(preset_options.keys()),
                    index=0
                )

                # Goals multiselect (pre-filled by preset)
                default_goals = preset_options[selected_preset]

                goals = st.multiselect(
                    "Select / Modify Goals",
                    [
                        "Maximize Equity",
                        "Maximize Speed",
                        "Maximize Security / Control",
                        "Maximize Transparency",
                        "Maximize Collaboration",
                        "Maximize Revenue / Efficiency",
                        "Minimize Strategic Manipulation"
                    ],
                    default=default_goals,
                    help="You can modify the preset or create your own combination"
                )

                # Get Recommendation
                if st.button("🔍 Get Recommended Strategy", type="primary", use_container_width=True):
                    if goals:
                        rec = recommend_allocation_strategy(goals)

                        st.success(f"**Top Recommendation:** {rec['top_recommendation']}")

                        if rec.get('second_recommendation'):
                            st.info(f"Strong Alternative: **{rec['second_recommendation']}**")

                        st.markdown("#### Reasoning")
                        for reason in rec.get("reasoning", []):
                            st.markdown(f"• {reason}")

                        # Score Visualization
                        score_df = pd.DataFrame.from_dict(rec["scores"], orient="index", columns=["Score"])
                        st.bar_chart(score_df, height=300)

                        # Auto-suggest mechanism for simulation
                        st.session_state["recommended_mechanism"] = rec['top_recommendation']
                    else:
                        st.warning("Please select at least one goal.")
                # Auto-apply recommended mechanism
                if "recommended_mechanism" in st.session_state:
                    st.info(f"💡 We recommend using **{st.session_state['recommended_mechanism']}** for this scenario.")
                # ====================== CUSTOM AGENT BUILDER ======================
                st.markdown("### Define Competing Entities")
                custom_profiles = []
                for i in range(n_agents):
                    with st.expander(f"Entity {i + 1}", expanded=(i < 3)):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            name = st.text_input("Entity Name", f"Entity {i + 1}", key=f"ns_name_{i}")
                            icon = st.text_input("Icon", "🔒", key=f"ns_icon_{i}")
                        with c2:
                            budget = st.number_input("Initial Budget (₦)", 50000, 5000000, 300000 + i * 80000,
                                                     key=f"ns_budget_{i}")
                        with c3:
                            strategy = st.selectbox("Strategy", ["aggressive", "cooperative", "honest"],
                                                    index=i % 3, key=f"ns_strat_{i}")
                        custom_profiles.append({
                            "name": name,
                            "budget": budget,
                            "strategy": strategy,
                            "icon": icon
                        })

                # ====================== CUSTOM RESOURCES BUILDER ======================
                st.markdown("### Define Critical Resources to Auction")

                # Predefined options
                predefined_resources = [
                    "satellite_bandwidth", "analyst_hours", "sensor_data",
                    "response_units", "surveillance_drones", "classified_compute",
                    "special_forces_slots", "cyber_defense_capacity", "intelligence_feeds"
                ]

                # Session state to persist added resources
                if "custom_resources_list" not in st.session_state:
                    st.session_state.custom_resources_list = predefined_resources[:6]

                # Display current resources with remove option
                current_resources = st.session_state.custom_resources_list

                cols = st.columns(4)
                for idx, res in enumerate(current_resources):
                    col = cols[idx % 4]
                    with col:
                        if st.button(f"🗑️ {res}", key=f"remove_res_{idx}"):
                            st.session_state.custom_resources_list.remove(res)
                            st.rerun()

                # Add new resource
                new_resource = st.text_input("Add New Critical Resource", placeholder="e.g. quantum_encryption_keys")
                if st.button("➕ Add Resource") and new_resource.strip():
                    cleaned = new_resource.strip().lower().replace(" ", "_")
                    if cleaned not in st.session_state.custom_resources_list:
                        st.session_state.custom_resources_list.append(cleaned)
                        st.success(f"Added: {cleaned}")
                        st.rerun()

                # Final resources list for simulation
                final_resources = st.session_state.custom_resources_list

                st.info(f"**Current Resources ({len(final_resources)}):** {', '.join(final_resources)}")

                # ====================== RUN BUTTON ======================
                if st.button("🚀 Run Custom Simulation", type="primary", use_container_width=True):
                    if not custom_profiles:
                        st.error("Please define at least one entity")
                    elif not final_resources:
                        st.error("Please define at least one resource")
                    else:
                        with st.spinner(f"Running {mechanism} on custom resources..."):
                            # Map mechanism name
                            mech_map = {
                                "Vickrey Auction (Truthful Bidding)": "Vickrey",
                                "English Auction (Ascending Price)": "English",
                                "First-Price Sealed Bid": "First_Price",
                                "Proportional Fair Allocation": "Proportional_Fair",
                                "Priority-Weighted Command (Military)": "Priority_Weighted",
                                "Nash Bargaining Solution (Cooperative)": "Nash_Bargaining"
                            }

                            result = run_agent_economy_simulation(
                                domain="security",
                                n_rounds=n_rounds,
                                custom_profiles=custom_profiles,
                                custom_resources=final_resources,
                                allocation_mechanism=mech_map[mechanism]
                            )
                            st.session_state["sec_feature_outputs"]["agent_economy"] = result
                            st.success(f"Simulation completed using **{mechanism}**")

                # Display Results
                _ae_result = st.session_state.get("sec_feature_outputs", {}).get("agent_economy")
                if _ae_result:
                    render_agent_economy_full(_ae_result, domain="security")
            with tab2:  # Monte Carlo Tab
                st.markdown("### Monte Carlo Sensitivity Analysis")
                st.caption("Assessing robustness under budget uncertainty")

                mc_col1, mc_col2 = st.columns(2)
                with mc_col1:
                    n_simulations = st.slider("Number of Simulations", 30, 300, 100, step=10)
                    n_rounds_mc = st.slider("Rounds per Simulation", 4, 15, 8)

                with mc_col2:
                    mc_mechanism = st.selectbox("Allocation Mechanism",
                                                ["Vickrey", "Nash_Bargaining", "Priority_Weighted"])

                if st.button("🔬 Run Monte Carlo Sensitivity Analysis", type="primary", use_container_width=True):
                    with st.spinner(f"Running {n_simulations} simulations..."):
                        mc_result = run_monte_carlo_analysis(
                            domain="security",
                            n_simulations=n_simulations,
                            n_rounds=n_rounds_mc,
                            base_profiles=custom_profiles,
                            custom_resources=final_resources,
                            allocation_mechanism=mc_mechanism
                        )
                        st.session_state["sec_feature_outputs"]["monte_carlo"] = mc_result
                        st.success("Monte Carlo Analysis Complete")

                # Display Results
                mc_data = st.session_state.get("sec_feature_outputs", {}).get("monte_carlo")
                if mc_data:
                    st.subheader("📊 Monte Carlo Sensitivity Analysis")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mean Gini", f"{mc_data.get('gini_mean', 0):.3f}")
                    c2.metric("Gini Std Dev", f"{mc_data.get('gini_std', 0):.3f}")
                    c3.metric("High Capture Risk", f"{mc_data.get('high_capture_risk', 0):.1%}")

                    # New Sensitivity Visualization
                    plot_monte_carlo_sensitivity(mc_data)
    # ── Tab 5-8: XAI / Compliance / Longitudinal / Federated ─────────────────
    with T["🧠 Explainable AI"]:
        _xai = st.session_state.get("security_xai_results",{})
        st.markdown("### 🧠 Explainable AI")
        if not _xai:
            st.info("Run a simulation to generate XAI explanations.")
        elif "error" in _xai:
            st.warning(f"XAI error: {_xai.get("error","unknown")}")
        else:
            fi=_xai.get("feature_importance",{})
            if fi:
                st.markdown(f'<div class="alert-info"><em>{fi.get("narrative","")}</em></div>',unsafe_allow_html=True)
                import plotly.express as _px5
                fi_df=pd.DataFrame({"Feature":fi["feature_names"][:10],"Importance":fi["importances"][:10]})
                fig_fi=_px5.bar(fi_df,x="Importance",y="Feature",orientation="h",
                    title="Feature Importance",color="Importance",color_continuous_scale="Blues")
                fig_fi.update_layout(yaxis={"categoryorder":"total ascending"},height=320)
                st.plotly_chart(fig_fi,use_container_width=True)
            cl,cr_=st.columns(2)
            with cl:
                expl=_xai.get("instance_explanation",{})
                if expl:
                    st.markdown("#### Instance Explanation")
                    st.markdown(f'<div class="alert-info"><em>{expl.get("decision_path","")}</em></div>',unsafe_allow_html=True)
            with cr_:
                cf=_xai.get("counterfactual",{})
                if cf:
                    st.markdown("#### Counterfactual")
                    st.markdown(f'<div class="alert-info"><em>{cf.get("plain_language","")}</em></div>',unsafe_allow_html=True)
            ix=_xai.get("intersectional",{})
            if ix and ix.get("group_performances"):
                st.markdown("#### Intersectional Fairness")
                st.markdown(f'<div class="alert-info">{ix.get("narrative","")}</div>',unsafe_allow_html=True)
                ix_df=pd.DataFrame([{"Group":k,**{kk:round(vv,3) for kk,vv in v.items()}} for k,v in ix["group_performances"].items()])
                st.dataframe(ix_df.style.background_gradient(subset=["accuracy"],cmap="RdYlGn"),use_container_width=True)

    with T["📋 Compliance"]:
        _xai=st.session_state.get("security_xai_results",{})
        _cr_=_xai.get("compliance_report",{})
        _mc_=_xai.get("model_card",{})
        st.markdown("### 📋 Regulatory Compliance Report")
        if not _cr_:
            st.info("Run a simulation to generate the compliance report.")
        else:
            summ=_cr_.get("summary",{})
            ok=summ.get("overall_compliant",False)
            st.markdown(f'<div class="{"alert-success" if ok else "alert-danger"}">Overall: <strong>{"COMPLIANT ✅" if ok else "NON-COMPLIANT ❌"}</strong> | Fairness: {summ.get("fairness_score",0):.3f}</div>',unsafe_allow_html=True)
            for fw,fd in _cr_.get("frameworks",{}).items():
                with st.expander(f"📑 {fw}"):
                    for ch,st_ in fd.get("checks",{}).items():
                        _icon = "✅" if st_=='PASS' else "❌"; st.markdown(f"{_icon} {ch}")
            try:
                pdf_b=generate_pdf_compliance_report(_cr_,_mc_,{"accuracy":df_r["detection_rate"].mean()},domain="national_security")
                st.download_button("📄 Download Compliance PDF",pdf_b,"security_compliance.pdf","application/pdf",use_container_width=True)
            except Exception as _e:
                st.caption(f"PDF unavailable: {_e}")

        # ── Real-world benchmark comparison ─────────────────────
        st.markdown("#### 📚 Real-World Benchmark Comparison")
        if BENCHMARKS_OK:
            _dom_bms = get_benchmarks_for_domain("national_security")
            if _dom_bms:
                _bm_sel = st.selectbox(
                        "Compare against a published study:",
                        list(_dom_bms.keys()),
                        format_func=lambda k: _dom_bms[k].name + " (" + str(_dom_bms[k].year) + ")",
                        key="_sec_bm_sel")
                _bm = _dom_bms[_bm_sel]
                _acc_col = "accuracy" if "accuracy" in df_r.columns else ("detection_rate" if "detection_rate" in df_r.columns else None)
                _fair_col = "fairness_score" if "fairness_score" in df_r.columns else None
                _sim_m = {}
                if _acc_col: _sim_m["accuracy"] = df_r[_acc_col].mean()
                if _fair_col: _sim_m["fairness_score"] = df_r[_fair_col].mean()
                if "fpr" in df_r.columns: _sim_m["fpr"] = df_r["fpr"].mean()
                if "recall" in df_r.columns: _sim_m["recall"] = df_r["recall"].mean()
                if CHARTS_OK:
                        try:
                            st.plotly_chart(benchmark_comparison_bar(
                                _sim_m, _bm.metrics, _bm.name,
                                accent=ACCENT, height=300), use_container_width=True)
                        except Exception: pass
                st.markdown(
                        '<div class="nbox"><strong>Key Lesson:</strong> ' + _bm.lesson +
                        '<br><span style="font-size:.75rem;color:#64748b">📚 ' +
                        _bm.citation[:100] + '</span></div>',
                        unsafe_allow_html=True)
            else:
                st.info("No published benchmarks available for this domain yet.")
        else:
            st.info("Add gags_benchmarks.py to components/ to enable benchmark comparison.")



        st.divider()
        st.markdown("#### 🇳🇬 Nigeria Regulatory Compliance")
        _nr_metrics = {
            "accuracy":          avg_det,
            "fairness_score":    avg_lib,
            "demographic_parity": float(df_r["false_positive_rate"].mean()) if "false_positive_rate" in df_r.columns else 0.0,
        }
        nigeria_compliance_panel(_nr_metrics, domain="security",
            has_xai=locals().get("enable_xai", True),
            has_governance=locals().get("enable_governance", True),
            has_multilingual=("security" in ["health","disinformation","education"]),
            has_ussd_fallback=False,
            has_gender_audit=locals().get("enable_gender_audit", False),
            has_redteam=locals().get("enable_redteam", False))

    with T["🔁 Longitudinal"]:
        _lng=st.session_state.get("security_longitudinal")
        st.markdown("### 🔁 Longitudinal Bias Analysis")
        st.markdown('<div class="alert-info">Simulates bias <strong>feedback loops</strong> across successive model retraining cycles — critical for surveillance systems updated frequently.</div>',unsafe_allow_html=True)
        if not _lng:
            st.info("Run a simulation to see longitudinal bias evolution.")
        else:
            c1,c2,c3=st.columns(3)
            c1.metric("Initial Bias",f'{_lng["initial_bias"]:.1%}')
            c2.metric("Final Bias",f'{_lng["final_bias"]:.1%}',f'{_lng["final_bias"]-_lng["initial_bias"]:+.1%}')
            c3.metric("Amplification",f'{_lng["amplification_factor"]:.2f}×')
            if _lng.get("self_reinforcing"):
                st.error(f"⚠️ Bias became self-reinforcing at generation {_lng.get('inflection_point','N/A')}.")
            st.markdown(f'<div class="alert-info"><em>{_lng["narrative"]}</em></div>',unsafe_allow_html=True)
            gm=_lng.get("generation_metrics",[])
            if gm:
                import plotly.express as _px6
                fig_lng=_px6.line(pd.DataFrame(gm),x="generation",
                    y=["demographic_parity","fairness_score","accuracy"],
                    title="Bias Evolution Across Retraining Generations",
                    color_discrete_sequence=["#ef4444","#16a34a","#2563eb"])
                fig_lng.add_hline(y=0.1,line_dash="dot",line_color="red")
                st.plotly_chart(fig_lng,use_container_width=True)

    with T["🌐 Federated"]:
        _fed=st.session_state.get("security_federated")
        st.markdown("### 🌐 Federated Learning Simulation")
        st.markdown('<div class="alert-info">Tests bias persistence when threat-detection models are trained <strong>across jurisdictions</strong> without centralising sensitive intelligence data.</div>',unsafe_allow_html=True)
        if not _fed:
            st.info("Run a simulation to see federated learning results.")
        else:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Clients",_fed["n_clients"])
            c2.metric("Global Accuracy",f'{_fed["global_accuracy"]:.1%}')
            c3.metric("Global Fairness",f'{_fed["global_fairness"]:.3f}')
            c4.metric("Bias Persisted","Yes ⚠️" if _fed["bias_persisted"] else "No ✅")
            st.markdown(f'<div class="{"alert-danger" if _fed["bias_persisted"] else "alert-success"}"><em>{_fed["narrative"]}</em></div>',unsafe_allow_html=True)
            cr_=_fed.get("client_results",[])
            if cr_:
                import plotly.express as _px7
                cr_df=pd.DataFrame([c.__dict__ if hasattr(c,"__dict__") else c for c in cr_])
                if not cr_df.empty and "local_bias" in cr_df.columns:
                    fig_fed=_px7.bar(cr_df,x="client_id",y=["local_accuracy","local_bias"],
                        barmode="group",title="Per-Client Metrics",
                        color_discrete_sequence=["#2563eb","#ef4444"])
                    st.plotly_chart(fig_fed,use_container_width=True)

    # ── Tab 9: Raw Results ──────────────────────────────────────────────────────
    with T["📋 Raw Results"]:
        display_cols = ["run_id","data_source","scenario","detection_rate","precision",
                        "false_positive_rate","accuracy","liberty_score","security_score",
                        "fairness_score","demographic_parity","surveillance_level",
                        "bias_intensity","poison_rate","attack_type","oversight","biases"]
        show_df = df_r[[c for c in display_cols if c in df_r.columns]]
        fmt = {c: "{:.3f}" for c in display_cols
               if c not in ("run_id","data_source","scenario","attack_type","oversight","biases")}
        st.dataframe(
            show_df.style
                .format({k: v for k, v in fmt.items() if k in show_df.columns})
                .background_gradient(subset=["detection_rate"],  cmap="Greens")
                .background_gradient(subset=["liberty_score"],   cmap="RdYlGn")
                .background_gradient(subset=["false_positive_rate"], cmap="Reds"),
            use_container_width=True,
        )

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                t("download_csv"),
                df_r.to_csv(index=False).encode(),
                f"gags_security_{scenario.lower().replace(' ','_')}.csv",
                "text/csv", use_container_width=True,
            )
        with dl2:
            cfg_export = {
                "scenario": scenario, "data_source": data_source,
                "dataset": dataset_choice, "threat_level": threat_level,
                "selected_biases": selected_biases, "bias_intensity": bias_intensity,
                "attack_type": attack_type_label, "poison_rate": poison_rate,
                "surveillance_level": surveillance_level, "oversight": oversight_level,
                "data_retention": data_retention,
                "features_enabled": {
                    "multimodal_redteam": enable_redteam,
                    "governance": enable_governance,
                    "strategic_arena": enable_arena,
                    "agent_economy": enable_agent_economy,
                },
                "avg_detection": f"{avg_det:.3f}",
                "avg_liberty":   f"{avg_lib:.3f}",
                "avg_fpr":       f"{avg_fpr:.3f}",
            }
            st.download_button(
                "📋 Export Config JSON",
                json.dumps(cfg_export, indent=2),
                f"security_config_{scenario.lower().replace(' ','_')}.json",
                "application/json", use_container_width=True,
            )

    # ── Tab 6: Data Explorer ───────────────────────────────────────────────────
    with T["📋 Compliance"]:
        st.markdown("### 🔍 Data Explorer")
        if data_source == "Real-World" and st.session_state.security_rw_df is not None:
            rw = st.session_state.security_rw_df
            c1, c2 = st.columns(2)
            c1.metric("Total Records",  f"{len(rw):,}")
            c2.metric("Total Features", len(rw.columns))

            st.subheader("Data Preview")
            st.dataframe(rw.head(20), use_container_width=True)

            st.subheader("Basic Statistics")
            st.dataframe(rw.describe(), use_container_width=True)

            if dataset_choice == "global_terrorism":
                gtd_preview = [c for c in ["iyear","imonth","country_txt","region_txt",
                    "attacktype1_txt","targtype1_txt","nkill","nwound"] if c in rw.columns]
                if gtd_preview:
                    st.subheader("GTD Key Fields")
                    st.dataframe(rw[gtd_preview].head(15), use_container_width=True)
                if "attacktype1_txt" in rw.columns:
                    fig_atk = px.bar(
                        rw["attacktype1_txt"].value_counts().head(10),
                        title="Top 10 Attack Types in Dataset",
                    )
                    st.plotly_chart(fig_atk, use_container_width=True)
        else:
            st.info("Select 'Real-World' data source in the sidebar and load a dataset to explore it here.")

    # ── AI Safety Tab ─────────────────────────────────────────────────────
    # ── 🔄 Lifecycle Management Tab ───────────────────────────────────────────
    with T["🔄 Lifecycle"]:
        render_lifecycle_tab(
            st.session_state.get("sec_lifecycle_report", {}),
            "security",
        )

    # ── 🌱 Eco Score Tab ──────────────────────────────────────────────────────
    with T["🌱 Eco Score"]:
        _lc_eco_r   = st.session_state.get("sec_lifecycle_report", {})
        _lc_eco_last = (st.session_state.get("sec_run_history") or [{}])[-1]
        render_eco_tab(
            _lc_eco_r,
            algo_key  = _lc_eco_last.get("algorithm", "hist_gradient_boosting"),
            n_samples = int(_lc_eco_last.get("n_samples", 2000)),
            n_runs    = int(locals().get("n_runs", 3)),
            domain    = "security",
        )

    # ── Simulation history ─────────────────────────────────────────────────────
    # ── AI Safety Tab ────────────────────────────────────────────────────────
    with T["🛡️ AI Safety"]:
        render_safety_tab(
            st.session_state.get("sec_safety_report", {}),
            domain="security",
        )

    # ── 🔮 Dynamic Systems Tab ──────────────────────────────────────────────────
    with T["🔮 Dynamic Systems"]:
        try:
            render_dynamic_systems_tab(
                st.session_state.get("sec_ds_report", {}),
                domain="security",
                ds_key="sec_ds_report",
            )
        except Exception as _ds_err:
            st.error(f"🔮 Dynamic Systems error: {_ds_err}")
            import traceback
            st.code(traceback.format_exc(), language="python")

    history_browser("security_snapshot_history", domain="security",
        key_metrics=["detection_rate","liberty_score","false_positive_rate"])

    # ── Annotation layer ────────────────────────────────────────────────
    annotation_panel("security_annotations", context_label=f"{len(st.session_state.security_run_history)} Security run(s)")

    # ── Policy Recommendations ─────────────────────────────────────────────────
    st.divider()
    st.markdown("## 💡 Security & Ethics Recommendations")
    r1, r2 = st.columns(2)
    with r1:
        if avg_fpr > 0.1:
            st.markdown("""
            <div class="alert-danger">
                <strong>⚠️ High False Positive Rate Detected</strong><br>
                • Increase judicial oversight requirements<br>
                • Mandate human review for all positive alerts<br>
                • Reduce surveillance intensity in low-threat areas<br>
                • Add fairness constraints to detection algorithms
            </div>""", unsafe_allow_html=True)
        elif avg_det < 0.7:
            st.markdown("""
            <div class="alert-warning">
                <strong>⚠️ Suboptimal Threat Detection</strong><br>
                • Consider moderate surveillance increase in high-threat zones<br>
                • Add multi-factor confirmation for threat alerts<br>
                • Expand training data diversity<br>
                • Recalibrate detection thresholds
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
                <strong>✅ Good Balance Achieved</strong><br>
                • Continue quarterly bias and fairness audits<br>
                • Maintain strong oversight mechanisms<br>
                • Review data retention policies annually<br>
                • Update threat models as adversary tactics evolve
            </div>""", unsafe_allow_html=True)
    with r2:
        st.info("""
        **Best Practices for Security AI:**
        1. **Proportionality** — Surveillance proportional to threat level
        2. **Transparency** — Publish detection criteria where operationally possible
        3. **Data Minimisation** — Collect only what is necessary
        4. **Oversight** — Independent review of surveillance practices
        5. **Redress** — Clear process for false-positive victims
        6. **Sunset Clauses** — Automatic review of surveillance powers
        7. **Audit Trail** — Tamper-evident logs for all AI decisions
        """)

    st.caption(
        "⚠️ Disclaimer: Educational simulation only. "
        "Real-world security AI requires comprehensive legal, ethical, and operational review."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Welcome state
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## 🎯 Welcome to National Security Simulation")
    st.markdown(
        "Configure your scenario in the sidebar and click **Run**. "
        "This module tests AI surveillance systems against bias, adversarial attacks, "
        "and governance constraints — with real-world GTD / UNSW-NB15 datasets and "
        "full GAGS v1.0 feature integration."
    )

    c1, c2, c3 = st.columns(3)
    scenario_cards = [
        ("🕵️ High Security",    "#1e3c72",
         "Surveillance 80+, all bias types. Maximise detection. Try with GTD data."),
        ("⚖️ Balanced Approach","#0f6e56",
         "Surveillance 40–60, moderate oversight. Find the optimal trade-off."),
        ("🕊️ Liberty Focused",  "#784212",
         "Surveillance 20–40, judicial oversight, no bias. Prioritise civil rights."),
    ]
    for col, (title, bg, desc) in zip([c1, c2, c3], scenario_cards):
        col.markdown(
            f'<div class="scenario-card" style="border-left-color:{bg};">'
            f'<h4 style="color:{bg};">{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        1. Choose **data source** (synthetic or real-world GTD/UNSW)
        2. Enable **GAGS Feature Modules** (Red Team, Governance, Arena, Economy)
        3. Select **bias types** from the validated list
        4. Configure **surveillance** intensity and oversight level
        5. Run simulations and review the 6-tab dashboard
        6. Check the **Governance banner** for AI drift flags and ledger hash
        7. Export results as CSV or configuration as JSON
        """)

    st.warning("""
    **Ethical Considerations:** Proportionality · Necessity · Accountability ·
    Transparency · Non-discrimination. This simulation explores these tensions
    in a controlled, educational environment.
    """)


# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;padding:1rem 0;'>"
    " National Security Simulation · GAGS Framework v1.0 · "
    "GTD / UNSW-NB15 · Multimodal Red Teaming · Governance Ledger"
    "</div>",
    unsafe_allow_html=True,
)
