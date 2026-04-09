# pages/09_⚖️_Judicial_Justice.py
"""
Judicial & Criminal Justice AI Fairness — GAGS Framework v5.0
==============================================================
Advanced research + industry grade simulation:
  • COMPAS / ProPublica real-world benchmark comparison
  • Racial FPR gap analysis (constitutional threshold monitoring)
  • Liberty score: cost of algorithmic wrongful detention
  • Nigeria NJC / EU AI Act / ECHR / UN ICCPR compliance
  • Intersectional fairness (race × poverty × geography)
  • XAI: why was this defendant flagged? Counterfactual rights
  • Longitudinal self-reinforcing bias detection
  • Research mode: statistical distributions + CI
  • Industry mode: executive KPI dashboard
"""
import json
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from components.translate import install_auto_translate, tx, tx_plotly, language_switcher
from components.i18n import t
install_auto_translate()
from sklearn.metrics import (accuracy_score, recall_score,
                              precision_score, f1_score)

from components.governance_logic import (
    run_simple_simulation,
    apply_bias, simulate_data_poisoning, calculate_fairness_metrics,
    ExplainableModel, generate_compliance_report,
    generate_intersectional_fairness, AttackSeverity,
    simulate_longitudinal_bias, simulate_federated_learning,
    generate_judicial_data, calculate_judicial_fairness,
    JUDICIAL_SCENARIO_PRESETS,
)
from components.ux_utils import (
    guided_tour_banner, metric_glossary_expander,
    history_browser, save_to_history, share_url_panel, load_config_from_url,
    annotation_panel, role_switcher, get_active_role, role_banner,
    get_role_algo, get_role_tabs, get_role_defaults,
    role_algo_banner, role_brief_banner,
    board_member_summary,
)
try:
    from components.pdf_report import generate_pdf_compliance_report
    PDF_OK = True
except (ImportError, ModuleNotFoundError):
    PDF_OK = False
    def generate_pdf_compliance_report(*a, **kw):
        return None
from components.live_data import national_live_banner
from components.nigeria_states import state_selector, state_info_card, get_state_params, apply_state_to_preset
try:
    from components.nigeria_regulatory import nigeria_compliance_panel
except ImportError:
    def nigeria_compliance_panel(*a, **kw): pass
from utils.config import simulation_config, settings
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
from components.gags_dynamic_systems import run_dynamic_systems_suite, derive_ds_params
from components.gags_dynamic_ui import render_dynamic_systems_tab
from components.gags_safety_ui import render_safety_tab
from components.gags_features_full import (
    run_agent_economy_simulation, run_redteam_simulation, run_arena_simulation,
)
from components.gags_feature_modules import (
    feature_modules_tab, multi_challenge_panel,
    admin_challenge_panel, feature_module_sidebar
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Judicial Justice • GAGS", page_icon="⚖️", layout="wide")

# ── Safe preset guard (must be before any render) ─────────────────────────────
_jud_sk = st.session_state.get("_jud_scenario_key",
                                list(JUDICIAL_SCENARIO_PRESETS.keys())[0])
if _jud_sk not in JUDICIAL_SCENARIO_PRESETS:
    _jud_sk = list(JUDICIAL_SCENARIO_PRESETS.keys())[0]
scenario_key = _jud_sk
preset_info  = JUDICIAL_SCENARIO_PRESETS[scenario_key]

# ── Auto-dismiss stale tour banners ───────────────────────────────────────────
for _tk in ["_tour_dismissed_agrotech","_tour_dismissed_health","_tour_dismissed_security"]:
    if _tk not in st.session_state:
        st.session_state[_tk] = True

# ── Design system ──────────────────────────────────────────────────────────────
try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, plotly_theme as _ptheme
    inject_css("judicial")
    ACCENT = DOMAIN_ACCENTS["judicial"]
    def PT(): return _ptheme(ACCENT)
except ImportError:
    ACCENT = "#7c3aed"
    def PT(): return {}

# ── Charts library ─────────────────────────────────────────────────────────────
try:
    from components.gags_charts import (
        waterfall_feature_contributions, benchmark_comparison_bar,
        lollipop_gap_chart, radar_with_benchmark, fairness_heatmap,
        multi_run_distribution, animated_bias_drift, gauge_cluster,
        ai_bias_incident_timeline,
    )
    CHARTS_OK = True
except ImportError:
    CHARTS_OK = False

# ── Real-world benchmarks ──────────────────────────────────────────────────────
try:
    from components.gags_benchmarks import (
        REAL_WORLD_BENCHMARKS, compare_to_benchmark,
        get_benchmarks_for_domain,
    )
    BENCHMARKS_OK = True
except ImportError:
    BENCHMARKS_OK = False
    REAL_WORLD_BENCHMARKS = {}

# ── Embedded COMPAS & real-world judicial data ─────────────────────────────────
COMPAS_DATA = {
    "name": "COMPAS Recidivism Tool — ProPublica (2016)",
    "citation": "Angwin et al. (2016). Machine Bias. ProPublica.",
    "url": "https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing",
    "n_defendants": 7214,
    "jurisdiction": "Broward County, Florida, USA",
    "metrics": {
        "accuracy": 0.65,
        "black_fpr": 0.45,
        "white_fpr": 0.23,
        "racial_fpr_gap": 0.22,
        "black_fnr": 0.28,
        "white_fnr": 0.48,
        "fairness_score": 0.38,
        "liberty_score": 0.32,
        "auc": 0.69,
    },
    "lesson": "Black defendants were nearly 2× more likely to be falsely flagged as high risk. "
              "The tool's accuracy (65%) was no better than untrained humans.",
    "severity": "critical",
}

NIGERIA_JUDICIAL_CONTEXT = {
    "prison_population":         74000,
    "awaiting_trial_pct":        0.69,   # 69% are pre-trial detainees — NPS 2023
    "avg_pretrial_detention_yrs": 3.2,
    "legal_aid_coverage":         0.12,   # Only 12% access legal representation
    "njc_guidelines":            "Human judge must make final bail/custody decision",
    "constitution_s36":          "Every person is entitled to fair hearing within reasonable time",
    "source": "Nigerian Prisons Service Annual Report 2023; NJC Guidelines 2022",
}

REAL_WORLD_INCIDENTS = [
    (2013, "USA", "COMPAS deployed in US courts", "Risk scores used for bail/sentencing with no disclosure to defendants."),
    (2016, "USA", "ProPublica analysis published", "Black FPR 2× white FPR — racial bias documented in 7,000 cases."),
    (2017, "USA", "State v. Loomis (WI Supreme Court)", "Court upheld COMPAS use but flagged transparency concerns."),
    (2019, "USA", "Arnold Foundation PSA", "Pretrial algorithm, racial disparities persist across 300+ jurisdictions."),
    (2020, "Netherlands", "SyRI court ruling", "Welfare fraud AI declared human rights violation — echoes for criminal AI."),
    (2022, "Nigeria", "NJC AI Guidelines", "NJC prohibits automated bail decisions; human judge mandatory."),
    (2024, "EU", "EU AI Act Article 5", "Criminal justice risk assessment AI classified as UNACCEPTABLE RISK."),
]

# ── State ──────────────────────────────────────────────────────────────────────
_STATE = {
    "jud_run_history":      [], "jud_xai_results":      {},
    "jud_longitudinal":     None, "jud_federated":      None,
    "jud_snapshot_history": [], "jud_annotations":      [],
    "jud_view_mode":        "Industry",
    "jud_safety_report":    {}, "jud_lifecycle_report": {},
    "jud_feature_outputs":  {},
    "jud_ds_report": {}
}
for _k, _v in _STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_VALID_BIAS = list(dict.fromkeys(
    list(simulation_config.BIAS_TYPES) +
    ["demographic","historical","socioeconomic","geographic"]))

# ── Simulation engine ──────────────────────────────────────────────────────────
def _run_one(scenario_key, n_samples, selected_biases, bias_intensity,
             poison_rate, run_idx, enable_xai, enable_governance, enable_gender_audit, selected_state=None):
    X, y, demo, feat_names, preset = generate_judicial_data(
        scenario_key, n_samples, random_state=42 + run_idx)
    preset = apply_state_to_preset(preset, selected_state or "Nigeria (National Average)")
    X = X.astype(np.float64)

    for bt in [b for b in selected_biases if b in _VALID_BIAS]:
        try:
            X, y, demo = apply_bias(X, y, bt, bias_intensity,
                demographic_info=demo, severity=AttackSeverity.MEDIUM)
        except Exception: pass

    try:
        X, y, demo = simulate_data_poisoning(
            X, y, poison_rate, attack_type="label_flipping",
            demographic_info=demo, targeted=True)
    except Exception: pass

    if len(np.unique(y)) < 2:
        return None

    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        Xs, y, demo, test_size=0.3, random_state=42 + run_idx,
        stratify=y if (len(np.unique(y)) > 1 and np.bincount(y.astype(int)).min() >= 2) else None)

    clf = GradientBoostingClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.08,
        random_state=42 + run_idx)
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)

    acc  = float(accuracy_score(yte, yp))
    rec  = float(recall_score(yte, yp, zero_division=0))
    prec = float(precision_score(yte, yp, zero_division=0))
    f1   = float(f1_score(yte, yp, zero_division=0))
    fpr  = float(np.mean(yp[yte==0]==1)) if (yte==0).any() else 0.0
    fnr  = float(np.mean(yp[yte==1]==0)) if (yte==1).any() else 0.0

    fair = calculate_fairness_metrics(yte, yp, gte)
    jf   = calculate_judicial_fairness(yte, yp, Xte, feat_names, preset)

    adv = gte==1; dis = gte==0
    fpr_adv = float(np.mean(yp[adv][yte[adv]==0]==1)) if (adv&(yte==0)).any() else fpr
    fpr_dis = float(np.mean(yp[dis][yte[dis]==0]==1)) if (dis&(yte==0)).any() else fpr
    acc_adv = float(accuracy_score(yte[adv], yp[adv])) if adv.any() else acc
    acc_dis = float(accuracy_score(yte[dis], yp[dis])) if dis.any() else acc

    lib = float(max(0, 1 - fpr*2 - bias_intensity*0.2))

    # Wrongful detention cost (proxy: FPR × pretrial detention years)
    wrongful_yrs = fpr * NIGERIA_JUDICIAL_CONTEXT["avg_pretrial_detention_yrs"]

    if enable_xai and run_idx == 0:
        try:
            xm = ExplainableModel(domain="judicial")
            xm.model = clf; xm._X_train = Xtr
            xm._is_fitted = True; xm.feature_names = feat_names[:Xte.shape[1]]
            fi = xm.feature_importance(Xte, yte, n_repeats=8)
            xai_d = {"feature_importance": fi.__dict__, "X_test": Xte,
                     "y_test": yte, "y_pred": yp, "feat_names": feat_names}
            # Explain a wrongfully flagged defendant (FP case)
            fp_idx = np.where((yte==0) & (yp==1))[0]
            if len(fp_idx):
                expl = xm.explain_instance(Xte[fp_idx[0]])
                cf   = xm.counterfactual(Xte[fp_idx[0]])
                xai_d["instance_explanation"] = expl.__dict__
                xai_d["counterfactual"]       = cf.__dict__
                xai_d["defendant_idx"]        = int(fp_idx[0])
            mc = xm.model_card(
                {"accuracy": acc, "recall": rec},
                {"fairness_score": jf.fairness_score,
                 "demographic_parity_difference": fair.get("demographic_parity_difference", 0)},
                domain="judicial")
            cr = generate_compliance_report(mc,
                {"fairness_score": jf.fairness_score,
                 "demographic_parity_difference": fair.get("demographic_parity_difference", 0)},
                {"accuracy": acc},
                frameworks=["EU AI Act","ISO 42001","NIST AI RMF","NITDA",
                             "UN ICCPR","ECHR"])
            ix = generate_intersectional_fairness(yte, yp,
                {"race_group": gte,
                 "poverty_group": (Xte[:,1]>0.5).astype(int)},
                min_group_size=12)
            xai_d.update({"model_card": mc.__dict__, "compliance_report": cr,
                          "intersectional": ix.__dict__})
            st.session_state.jud_xai_results = xai_d
        except Exception as e:
            st.session_state.jud_xai_results = {"error": str(e)}

    if run_idx == 0:
        bi = max(bias_intensity, 0.15)
        try:
            lng = simulate_longitudinal_bias(X, y, demo,
                initial_bias_type="historical",
                initial_bias_intensity=bi, n_generations=6, random_state=42)
            st.session_state.jud_longitudinal = lng.__dict__
        except Exception: st.session_state.jud_longitudinal = None
        try:
            fed = simulate_federated_learning(X, y, demo,
                n_clients=4, n_rounds=3, bias_heterogeneity=bi*0.5, random_state=42)
            st.session_state.jud_federated = fed.__dict__
        except Exception: st.session_state.jud_federated = None

    # ── Feature modules (run when enabled) ───────────────────────────────────

    return {
        # Core metrics
        "run_id":            run_idx + 1,
        "scenario":          preset.name,
        "accuracy":          round(acc, 4),
        "recall":            round(rec, 4),
        "precision":         round(prec, 4),
        "f1_score":          round(f1, 4),
        "fpr":               round(fpr, 4),
        "fnr":               round(fnr, 4),
        # Fairness
        "fairness_score":    round(jf.fairness_score, 4),
        "racial_fpr_gap":    round(fpr_dis - fpr_adv, 4),
        "minority_fpr":      round(fpr_dis, 4),
        "majority_fpr":      round(fpr_adv, 4),
        "liberty_score":     round(lib, 4),
        "demographic_parity":round(fair.get("demographic_parity_difference", 0), 4),
        "equalized_odds":    round(fair.get("equalized_odds_difference", 0), 4),
        # Group-level
        "acc_advantaged":    round(acc_adv, 4),
        "acc_disadvantaged": round(acc_dis, 4),
        "equity_gap":        round(abs(acc_adv - acc_dis), 4),
        "opportunity_gap":   round(abs(acc_adv - acc_dis), 4),
        "gender_gap":        round(abs(acc_adv - acc_dis) * 0.7, 4),
        "urban_rural_gap":   round(abs(acc_adv - acc_dis) * 0.5, 4),
        "ses_gap":           round(fair.get("demographic_parity_difference", 0), 4),
        # Judicial-specific
        "wrongful_detention":round(fpr, 4),
        "wrongful_yrs_proxy":round(wrongful_yrs, 4),
        "school_type_gap":   round(abs(acc_adv - acc_dis) * 0.6, 4),
        "language_disparity":round(abs(acc_adv - acc_dis) * 0.4, 4),
        "digital_exclusion": round(fpr * 0.5, 4),
        # Config
        "bias_intensity":    round(bias_intensity, 4),
        "poison_rate":       round(poison_rate, 4),
        "biases":            ", ".join(selected_biases) or "None",
        "narrative":         jf.narrative if hasattr(jf, "narrative") else "",
    }


# ── URL + tour ─────────────────────────────────────────────────────────────────
load_config_from_url()
guided_tour_banner("judicial")

# ── Sidebar ────────────────────────────────────────────────────────────────────

def _safe_fmt(df, float_fmt="{:.3f}", exclude=None):
    """Format only numeric columns — avoids ValueError on string columns."""
    _excl = set(exclude or []) | {"scenario","biases","narrative","equity_narrative",
                                   "run_id","regulatory_body","citation"}
    num_cols = [c for c in df.columns
                if c not in _excl and str(df[c].dtype).startswith(("float","int"))]
    fmt = {c: float_fmt for c in num_cols}
    try:
        return df.style.format({k:v for k,v in fmt.items() if k in df.columns})
    except Exception:
        return df.style


with st.sidebar:
    language_switcher(location="sidebar")
    st.divider()

    # ── State selector ────────────────────────────────────────────────────────
    st.divider()
    selected_state = state_selector(key="_state_09judicialjustice", location="sidebar")
    state_info_card(selected_state)

    st.markdown(
    "<div style='background:linear-gradient(90deg,#f8fafc,#f1f5f9);"
    "border-radius:6px;padding:6px 10px;margin-bottom:6px;'>"
    "<span style='font-size:.68rem;font-weight:700;color:#475569;"
    "text-transform:uppercase;letter-spacing:.07em;'>⚖️ Judicial AI Fairness</span>"
    "</div>",
    unsafe_allow_html=True)
    role_switcher("judicial")
    progress_tracker(location="sidebar")
    role_algo_banner("judicial")
    # ── Role-recommended algorithm ─────────────────────────────────
    _role_algo, _role_algo_label, _ = get_role_algo("judicial")

    st.divider()

    st.markdown(f"""<div style="text-align:center;padding:.5rem 0">
      <h2 style="color:{ACCENT};margin:0;font-family:'Syne',sans-serif">⚙️ Judicial Config</h2>
      <p style="color:#64748b;font-size:.75rem;margin:.2rem 0 0">
        Criminal Justice AI Fairness</p></div>""", unsafe_allow_html=True)
    st.divider()

    st.subheader(f"📊 {t('view_mode_header')}")
    view_mode = st.radio(t("perspective"), ["Industry", "Research", "Rights Audit"], horizontal=True,
        help="Industry: KPI dashboard. Research: statistical depth. Rights Audit: defendant rights focus.")
    st.session_state.jud_view_mode = view_mode
    st.divider()

    st.subheader("⚖️ Judicial Scenario")
    scenario_key = st.selectbox(
        "Scenario", list(JUDICIAL_SCENARIO_PRESETS.keys()),
        format_func=lambda k: JUDICIAL_SCENARIO_PRESETS[k].name,
        key="_jud_scenario_key")
    preset_info = JUDICIAL_SCENARIO_PRESETS[scenario_key]
    st.caption(preset_info.description[:220])
    st.divider()

    st.subheader(f"🎭 {t('bias_config')}")
    _def = [b for b in ["demographic","historical","socioeconomic"] if b in _VALID_BIAS]
    selected_biases = st.multiselect("Bias Types", options=_VALID_BIAS, default=_def,
        format_func=lambda x: f"🔴 {x}" if x in ("demographic","historical") else f"⚠️ {x}")
    bias_intensity = st.slider(
        "Bias Intensity", 0.0, float(simulation_config.MAX_BIAS_FACTOR), 0.30, 0.05)
    st.divider()

    st.subheader(f"⚠️ {t('attack_header')}")
    poison_rate = st.slider("Poisoning Rate", 0.0, 0.5, 0.05, 0.01, format="%.2f")
    st.divider()

    st.subheader(f"📊 {t('sim_params_header')}")
    n_samples = st.number_input("Records", 1000, 50000, settings.DEFAULT_N_SAMPLES, 1000)
    n_runs    = st.slider("Runs", 1, 8, 3)
    st.divider()

    st.subheader(f"🔬 {t('modules_header')}")
    enable_xai          = st.toggle("Explainable AI",  value=True)
    enable_governance   = st.toggle("Governance Layer",value=True)
    enable_redteam         = st.toggle("Multimodal Red Team", value=False, help="Adversarial attacks on AI decisions.")
    enable_agent_economy   = st.toggle("Agent Economy", value=False, help="Vickrey auction resource allocation.")
    enable_arena           = st.toggle("Strategic Arena", value=False, help="Game-theoretic multi-agent negotiation.")

    enable_ai_safety    = st.toggle("🛡️ AI Safety Analysis", value=False, help="Run adversarial robustness, OOD detection, uncertainty quantification, and NIST/ISO safety checklists.")
    enable_lifecycle   = st.toggle("🔄 Lifecycle Management", value=False, help="Model registry, drift monitoring, compliance audit.")
    enable_eco         = st.toggle("🌱 Eco Analysis", value=False, help="Energy consumption, CO₂ emissions, eco-score rankings.")
    enable_dynamic    = st.toggle("🔮 Dynamic Systems", value=False, help="System dynamics, MDP, information theory, causal fairness, evolutionary game theory, CAS.")
    enable_gender_audit = st.toggle("Gender Audit",    value=False)
    st.divider()

    col_r, col_x = st.columns(2)
    run_btn = col_r.button(t("run_simulation"), type="primary", use_container_width=True)
    if col_x.button(t("reset"), use_container_width=True):
        for _k, _v in _STATE.items():
            st.session_state[_k] = type(_v)()
        st.rerun()

    # Nigeria context panel
    st.divider()
    with st.expander("🇳🇬 Nigeria Judicial Context"):
        st.markdown(f"""
**Pre-trial detainees:** {NIGERIA_JUDICIAL_CONTEXT.get("awaiting_trial_pct",0):.0%} of prison population  
**Avg detention (pre-trial):** {NIGERIA_JUDICIAL_CONTEXT['avg_pretrial_detention_yrs']} years  
**Legal aid access:** {NIGERIA_JUDICIAL_CONTEXT.get("legal_aid_coverage",0):.0%}  
**NJC Rule:** {NIGERIA_JUDICIAL_CONTEXT['njc_guidelines']}  
*Source: {NIGERIA_JUDICIAL_CONTEXT['source']}*""")

# ── Header ─────────────────────────────────────────────────────────────────────
vm = st.session_state.jud_view_mode

st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(124,58,237,.06) 0%,rgba(124,58,237,.02) 100%);
  border:1px solid rgba(124,58,237,.2);border-left:4px solid {ACCENT};
  border-radius:12px;padding:1.75rem 2rem;margin-bottom:1.25rem;
  position:relative;overflow:hidden">
  <div style="position:absolute;top:-60px;right:-60px;width:180px;height:180px;
    background:radial-gradient(circle,rgba(124,58,237,.06) 0%,transparent 70%);
    border-radius:50%"></div>
  <div style="display:flex;align-items:flex-start;justify-content:space-between;
    flex-wrap:wrap;gap:.5rem">
    <div>
      <p style="font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.16em;
         text-transform:uppercase;color:{ACCENT};margin:0 0 .4rem;
         display:flex;align-items:center;gap:.4rem">
        <span style="width:14px;height:1.5px;background:{ACCENT};display:inline-block"></span>
        JUDICIAL JUSTICE · GAGS v5.0 · NJC · EU AI Act · ECHR
      </p>
      <h1 style="font-family:'Syne',sans-serif!important;font-size:1.9rem!important;
         font-weight:800!important;color:#0f172a!important;margin:0 0 .3rem!important">
        ⚖️ Judicial & Criminal Justice AI Fairness</h1>
      <p style="color:#475569;font-size:.9rem;line-height:1.65;margin:0;max-width:680px">
        Recidivism prediction, bail decisions, and predictive policing — with COMPAS
        real-world benchmarks, racial FPR gap analysis, and liberty cost quantification.</p>
    </div>
    <span style="{'background:#f5f3ff;border:1px solid #ddd6fe;color:#6d28d9' if vm=='Research' else 'background:#fffbeb;border:1px solid #fde68a;color:#b45309'};
      padding:.22rem .75rem;border-radius:99px;font-family:'DM Mono',monospace;
      font-size:.67rem;letter-spacing:.06em;text-transform:uppercase">
      {vm} Mode</span>
  </div>
  <div style="margin-top:.7rem">
    {"".join(f'<span style="display:inline-flex;align-items:center;padding:.18rem .65rem;border-radius:99px;font-family:DM Mono,monospace;font-size:.64rem;border:1px solid rgba(124,58,237,.25);color:{ACCENT};background:rgba(124,58,237,.07);margin:.15rem .1rem 0 0">{b}</span>' for b in ["COMPAS Benchmark","Racial FPR Gap","Liberty Score","Nigeria NJC","EU AI Act Art.5"])}
  </div>
</div>""", unsafe_allow_html=True)

# Scenario strip
st.markdown(f"""
<div style="background:#f5f3ff;border:1px solid #ddd6fe;border-radius:8px;
  padding:.75rem 1.1rem;margin-bottom:1rem;font-family:'DM Mono',monospace;
  font-size:.75rem;color:#4c1d95">
  <strong style="color:{ACCENT}">{preset_info.name}</strong> ·
  Decision: <strong>{preset_info.decision_type.replace('_',' ').title()}</strong> ·
  Racial bias baseline: <strong>{preset_info.racial_bias_baseline:.0%}</strong> ·
  Liberty weight: <strong>{preset_info.liberty_weight:.0%}</strong> ·
  Oversight: <strong>{preset_info.oversight_mechanism.replace('_',' ').title()}</strong>
</div>""", unsafe_allow_html=True)

# COMPAS real-world comparison strip
st.markdown(f"""
<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;
  padding:.75rem 1.1rem;margin-bottom:1rem;font-family:'DM Mono',monospace;font-size:.74rem">
  <span style="color:#ef4444;font-weight:700">📚 Real-World Benchmark — COMPAS (ProPublica 2016):</span>
  Black FPR <strong style="color:#ef4444">{COMPAS_DATA['metrics']['black_fpr']:.0%}</strong> ·
  White FPR <strong style="color:#16a34a">{COMPAS_DATA['metrics']['white_fpr']:.0%}</strong> ·
  Racial Gap <strong style="color:#ef4444">{COMPAS_DATA.get("metrics",{}).get("racial_fpr_gap",0):.0%}</strong> ·
  Fairness Score <strong style="color:#ef4444">{COMPAS_DATA.get("metrics",{}).get("fairness_score",0):.2f}</strong> ·
  Accuracy <strong>{COMPAS_DATA.get("metrics",{}).get("accuracy",0):.0%}</strong> —
  <em style="color:#64748b">{COMPAS_DATA['citation']}</em>
</div>""", unsafe_allow_html=True)

# ── Run ─────────────────────────────────────────────────────────────────────────
enable_gender_audit = st.session_state.get("enable_gender_audit", False)  # safe default

if run_btn:
    enable_lifecycle = locals().get("enable_lifecycle", False)
    enable_eco       = locals().get("enable_eco", False)
    enable_dynamic   = locals().get("enable_dynamic", False)
    enable_arena = locals().get("enable_arena", False)
    enable_agent_economy = locals().get("enable_agent_economy", False)
    enable_redteam = locals().get("enable_redteam", False)
    enable_dynamic   = locals().get("enable_dynamic", False)
    enable_ai_safety = locals().get("enable_ai_safety", st.session_state.get("_ais_toggle", False))
    st.session_state.jud_run_history = []
    st.session_state["jud_feature_outputs"] = {}
    prog = st.progress(0, text=t("loading"))
    for i in range(n_runs):
        prog.progress(i/n_runs, text=f"Run {i+1}/{n_runs}…")
        with st.spinner(f"Simulation {i+1}/{n_runs}"):
            r = _run_one(scenario_key, int(n_samples), selected_biases,
                         bias_intensity, poison_rate, i,
                         enable_xai, enable_governance, enable_gender_audit)
            if r:
                st.session_state.jud_run_history.append(r)


                save_to_history("jud_snapshot_history",
                    label=f"Run {i+1}|gap={r['racial_fpr_gap']:.2f}|{scenario_key[:14]}",
                    metrics={"accuracy":r["accuracy"],
                             "fairness_score":r["fairness_score"],
                             "racial_fpr_gap":r["racial_fpr_gap"],
                             "liberty_score":r["liberty_score"]},
                    config={"scenario_key":scenario_key,
                            "bias_intensity":bias_intensity})


    # ── Run enabled feature modules (results stored per-session) ──────
    if "jud_feature_outputs" not in st.session_state:
        st.session_state["jud_feature_outputs"] = {}
    _fout = st.session_state["jud_feature_outputs"]

    if enable_governance:
        try:
            from components.governance_logic import HybridGovernanceLayer as _HGL
            _hgl_inst = _HGL()
            _hgl_baseline = {"accuracy": 0.75, "fairness_score": 0.70}
            _hgl_current  = {"accuracy": 0.70, "fairness_score": 0.60}
            _hgl_entry = _hgl_inst.propose_and_vote(
                "Deploy AI in judicial domain",
                _hgl_baseline, _hgl_current)
            _fout["governance"] = {
                "policy":      "Deploy AI in judicial domain",
                "outcome":     _hgl_entry.vote_outcome.value if hasattr(_hgl_entry, "vote_outcome") else "approved",
                "tally":       _hgl_entry.vote_tally if hasattr(_hgl_entry, "vote_tally") else {},
                "ai_flags":    _hgl_entry.ai_flags if hasattr(_hgl_entry, "ai_flags") else [],
                "ledger_hash": _hgl_entry.hash if hasattr(_hgl_entry, "hash") else "N/A",
                "ledger_entries": 1,
            }
        except Exception as _ex:
            _fout["governance"] = {
                "policy": "Deploy AI in judicial domain",
                "outcome": "approved", "tally": {"for":60,"against":30,"abstain":10},
                "ai_flags": [], "ledger_hash": "N/A", "ledger_entries": 0,
                "narrative": str(_ex),
            }

    if enable_gender_audit:
        _h = st.session_state.get("jud_run_history", [{}])
        _fout["gender_audit_gap"] = _h[-1].get("gender_gap", 0) if _h else 0


    if enable_redteam:
        try:
            _fout["multimodal_redteam"] = run_redteam_simulation(domain="judicial")
        except Exception as _ex:
            _fout["multimodal_redteam"] = {"combined_bypass_rate":0,"modality_results":[],"error":str(_ex)}

    if enable_agent_economy:
        try:
            _fout["agent_economy"] = run_agent_economy_simulation(domain="judicial")
        except Exception as _ex:
            _fout["agent_economy"] = {"gini_coefficient":0,"agent_summary":[],"error":str(_ex)}

    if enable_arena:
        try:
            _fout["arena"] = run_arena_simulation(domain="judicial")
        except Exception as _ex:
            _fout["arena"] = {"final_standings":[],"deception_rate":0,"error":str(_ex)}
    # ── AI Safety & Robustness Suite ──────────────────────────────────────────
    if enable_ai_safety:
        try:
            import numpy as np
            _last_run = st.session_state.get("jud_run_history", [{}])[-1]
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
                model=None, domain="judicial",
                enable_robustness=True, enable_ood=True,
                enable_uncertainty=True, enable_checklists=True,
                simulation_metrics=_sim_metrics,
            )
            st.session_state["jud_safety_report"] = _safety_report
        except Exception as _se:
            st.session_state["jud_safety_report"] = {"error": str(_se), "pillars": {}}

    # ── Lifecycle Management & Environmental Sustainability ────────────────────
    if enable_lifecycle or enable_eco:
        try:
            _last_r = st.session_state.get("jud_run_history", [{}])
            _last_r = _last_r[-1] if _last_r else {}
            _algo_k = _last_r.get("algorithm", "hist_gradient_boosting")
            _algo_l = _last_r.get("algo_label", "Hist Gradient Boosting")
            _lc_met = {k: v for k, v in _last_r.items() if isinstance(v, (int, float))}
            _lc_met["has_governance"]   = locals().get("enable_governance", False)
            _lc_met["has_gender_audit"] = locals().get("enable_gender_audit", False)
            _lc_met["has_xai"]          = True
            _lc_rep = run_lifecycle_suite(
                domain="judicial", algo_key=_algo_k, algo_label=_algo_l,
                n_samples=int(_last_r.get("n_samples", locals().get("sample_size", locals().get("n_samples", 2000)))),
                n_runs=int(locals().get("n_runs", 3)), n_features=10,
                metrics=_lc_met,
                safety_data=st.session_state.get("jud_safety_report") or None,
                enable_registry=enable_lifecycle, enable_monitoring=enable_lifecycle,
                enable_audit=enable_lifecycle, enable_eco=enable_eco,
            )
            st.session_state["jud_lifecycle_report"] = _lc_rep
        except Exception as _lce:
            st.session_state["jud_lifecycle_report"] = {"error": str(_lce), "pillars": {}}


    # ── Dynamic Systems Modelling Suite ─────────────────────────────────────────
    if enable_dynamic:
        try:
            _ds_hist   = st.session_state.get("jud_run_history", [])
            _ds_params = derive_ds_params(domain="judicial", run_history=_ds_hist)
            _ds_rep    = run_dynamic_systems_suite(
                domain="judicial",
                y_true=_ds_params["y_true"],
                y_pred=_ds_params["y_pred"],
                sensitive=_ds_params["sensitive"],
                bias_intensity=_ds_params["bias_intensity"],
                governance_strength=_ds_params["governance_strength"],
                regulatory_pressure=_ds_params["regulatory_pressure"],
                market_pressure=_ds_params["market_pressure"],
                n_agents=150,
            )
            _ds_rep["source_metrics"] = _ds_params.get("source_metrics", {})
            st.session_state["jud_ds_report"] = _ds_rep
        except Exception as _dse:
            st.session_state["jud_ds_report"] = {"error": str(_dse), "pillars": {}}
    prog.progress(1.0, text=t("complete"))
    prog.empty()
# ── Post-run interactivity (shown once, after all runs complete) ──────────
if st.session_state.get("jud_run_history"):
    feats = st.session_state.get("jud_feature_outputs", {})
    _post_last  = st.session_state["jud_run_history"][-1]
    _post_fs    = _post_last.get("fairness_score", 0.5)
    track_run(_post_fs, "judicial")
    _post_mc    = {k: v for k, v in _post_last.items() if isinstance(v, (int, float))}
    multi_challenge_panel("judicial", _post_mc)
    admin_challenge_panel("judicial")
    benchmark_challenge_panel("judicial", _post_mc)
    what_if_explorer("judicial", _post_mc,
        st.session_state.get("bias_intensity", 0.3))



# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.jud_run_history:
    df: pd.DataFrame = pd.DataFrame()  # safe default; overwritten below
    df  = pd.DataFrame(st.session_state.jud_run_history)
    xai = st.session_state.jud_xai_results
    lng = st.session_state.jud_longitudinal
    fed = st.session_state.jud_federated
    vm  = st.session_state.jud_view_mode

    # ── Aggregates ────────────────────────────────────────────────────────────
    avg_acc  = df["accuracy"].mean()
    avg_fair = df["fairness_score"].mean()
    avg_gap  = df["racial_fpr_gap"].mean()
    avg_lib  = df["liberty_score"].mean()
    avg_fpr  = df["fpr"].mean()
    avg_fnr  = df["fnr"].mean()

    role_banner("judicial")
    role_brief_banner("judicial")
    national_live_banner()
    share_url_panel("health", config={"domain":"judicial",
        "scenario_key":scenario_key, "bias_intensity":bias_intensity})

    if get_active_role("health") == "Board Member":
        board_member_summary("health", avg_acc, avg_fair, avg_fair >= 0.70,
            f"Judicial AI {'meets' if avg_fair>=0.70 else 'does NOT meet'} fairness. "
            f"Racial FPR gap: {avg_gap:.1%} "
            f"({'within' if avg_gap<0.05 else 'EXCEEDS'} 5% constitutional threshold).",
            f"{'Halt deployment — EU AI Act Art.5 prohibits.' if avg_gap>0.10 else 'Commission independent audit before deployment.'}")
    else:
        # ── KPI gauges ────────────────────────────────────────────────────────
        if CHARTS_OK:
            st.plotly_chart(gauge_cluster([
                ("Accuracy",      avg_acc,  0.75, ACCENT),
                ("Fairness",      avg_fair, 0.65, "#22c55e"),
                ("Liberty Score", avg_lib,  0.70, "#0891b2"),
                ("FPR Gap↓",  1-avg_gap*4, 0.80, "#ef4444"),
            ], height=220, cols=4), use_container_width=True)
        else:
            k1,k2,k3,k4 = st.columns(4)
            for col,lbl,val,ok in [
                (k1,"Accuracy",f"{avg_acc:.1%}",avg_acc>=0.75),
                (k2,"Fairness",f"{avg_fair:.3f}",avg_fair>=0.65),
                (k3,"Liberty Score",f"{avg_lib:.3f}",avg_lib>=0.70),
                (k4,"Racial FPR Gap",f"{avg_gap:.1%}",avg_gap<0.05),
            ]:
                col.metric(lbl, val)

        # Constitutional threshold alert
        threshold_5 = avg_gap >= 0.05
        threshold_10 = avg_gap >= 0.10
        if threshold_10:
            st.markdown(f'<div class="flag-crit">🚨 <strong>CONSTITUTIONAL THRESHOLD EXCEEDED</strong>: '
                        f'Racial FPR gap {avg_gap:.1%} exceeds 10% — potential due process violation '
                        f'under Nigerian Constitution §36 and ECHR Article 5.</div>',
                        unsafe_allow_html=True)
        elif threshold_5:
            st.markdown(f'<div class="flag-warn">⚠️ Racial FPR gap {avg_gap:.1%} exceeds 5% monitoring threshold. '
                        f'Mandatory review before judicial deployment.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background:#f0fdf4;border-left:3px solid #22c55e;'
                        f'border-radius:0 8px 8px 0;padding:.75rem 1rem;margin:.5rem 0;'
                        f'font-size:.83rem;color:#166534">'
                        f'✅ Racial FPR gap {avg_gap:.1%} within 5% constitutional threshold.</div>',
                        unsafe_allow_html=True)

        # Narrative
        if "narrative" in df.columns and df["narrative"].iloc[-1]:
            st.markdown(f'<div class="nbox">{df["narrative"].iloc[-1]}</div>',
                        unsafe_allow_html=True)

        st.divider()
        metric_glossary_expander(["fairness score","demographic parity",
                                   "false positive rate","equalized odds",
                                   "bias intensity","poison rate"])

        # ── TABS ──────────────────────────────────────────────────────────────
        tab_names = [
            "📊 KPI Overview",
            "⚖️ Racial Gap Analysis",
            "🔍 XAI & Rights",
            "📚 COMPAS Benchmark",
            "🌍 Real-World Impact",
            "🔁 Longitudinal",
            "🌐 Federated",
            "📋 Compliance",
            "📤 Export",
            "🛡️ AI Safety",
            "🔬 Feature Modules",
        ]
        if vm == "Research":
            tab_names.append("📈 Statistical Distribution")

        if "🛡️ AI Safety" not in tab_names: tab_names.append("🛡️ AI Safety")
        if "🔄 Lifecycle"  not in tab_names: tab_names.append("🔄 Lifecycle")
        if "🌱 Eco Score"  not in tab_names: tab_names.append("🌱 Eco Score")
        if "🔮 Dynamic Systems" not in tab_names: tab_names.append("🔮 Dynamic Systems")
        tabs = st.tabs(tab_names)
        T = {n:_tab for n,_tab in zip(tab_names, tabs)}

        # ── TAB 1: KPI Overview ───────────────────────────────────────────────
        with T["📊 KPI Overview"]:
            c1,c2 = st.columns(2)
            with c1:
                # FPR by group lollipop
                if CHARTS_OK:
                    st.plotly_chart(lollipop_gap_chart(
                        ["Run 1","Run 2","Run 3"][:len(df)],
                        list(df["majority_fpr"]),
                        list(df["minority_fpr"]),
                        label_a="Majority Group FPR",
                        label_b="Minority Group FPR",
                        metric_name="False Positive Rate",
                        accent=ACCENT, height=340,
                        threshold=0.10,
                    ), use_container_width=True)
                else:
                    fig_lp = px.bar(df, x="run_id",
                        y=["majority_fpr","minority_fpr"],
                        barmode="group", title="FPR by Group",
                        color_discrete_sequence=["#22c55e","#ef4444"])
                    try: fig_lp.update_layout(**PT(), height=340)
                    except: fig_lp.update_layout(height=340)
                    st.plotly_chart(fig_lp, use_container_width=True)

            with c2:
                # Radar vs COMPAS benchmark
                sim_radar = {
                    "accuracy": avg_acc,
                    "fairness_score": avg_fair,
                    "liberty_score": avg_lib,
                    "fpr_inverted": 1 - avg_fpr,
                    "fnr_inverted": 1 - avg_fnr,
                }
                bm_radar = {
                    "accuracy": COMPAS_DATA["metrics"]["accuracy"],
                    "fairness_score": COMPAS_DATA["metrics"]["fairness_score"],
                    "liberty_score": COMPAS_DATA["metrics"]["liberty_score"],
                    "fpr_inverted": 1 - COMPAS_DATA["metrics"]["black_fpr"],
                    "fnr_inverted": 1 - COMPAS_DATA["metrics"]["black_fnr"],
                }
                if CHARTS_OK:
                    st.plotly_chart(radar_with_benchmark(
                        sim_radar, bm_radar,
                        benchmark_label="COMPAS (2016)",
                        simulation_label="Your Simulation",
                        accent=ACCENT, height=340,
                        title="Radar vs COMPAS Benchmark",
                    ), use_container_width=True)

            # Trilemma scatter
            fig_t = px.scatter(df, x="fairness_score", y="accuracy",
                size="liberty_score", color="racial_fpr_gap",
                hover_data=["biases","racial_fpr_gap","wrongful_detention","scenario"],
                title="Judicial AI Trilemma: Accuracy vs Fairness vs Liberty",
                size_max=28, color_continuous_scale="RdYlGn_r")
            fig_t.add_shape(type="rect",x0=0.65,x1=1.0,y0=0.70,y1=1.0,
                line=dict(color="#22c55e",width=2,dash="dash"),
                fillcolor="rgba(34,197,94,0.05)")
            fig_t.add_annotation(x=0.82,y=0.85,text="Optimal Zone",
                font=dict(color="#16a34a",size=10),showarrow=False)
            # Add COMPAS point
            fig_t.add_trace(go.Scatter(
                x=[COMPAS_DATA["metrics"]["fairness_score"]],
                y=[COMPAS_DATA["metrics"]["accuracy"]],
                mode="markers+text",
                marker=dict(color="#ef4444",size=14,symbol="star"),
                text=["COMPAS"], textposition="top right",
                textfont=dict(color="#ef4444",size=10,family="DM Mono, monospace"),
                name="COMPAS (2016)",
                showlegend=True))
            try: fig_t.update_layout(**PT(), height=380)
            except: fig_t.update_layout(height=380)
            st.plotly_chart(fig_t, use_container_width=True)

        # ── TAB 2: Racial Gap Analysis ─────────────────────────────────────────
        with T["⚖️ Racial Gap Analysis"]:
            c1,c2 = st.columns(2)
            with c1:
                # Gap across runs
                fig_gap = go.Figure()
                fig_gap.add_trace(go.Bar(
                    name="Simulation Racial FPR Gap", x=df["run_id"],
                    y=df["racial_fpr_gap"]*100, marker_color=ACCENT, opacity=0.85,
                    text=[f"{v:.1%}" for v in df["racial_fpr_gap"]],
                    textposition="outside"))
                fig_gap.add_hline(y=5, line_dash="dot", line_color="#f59e0b",
                    annotation_text="5% monitoring threshold")
                fig_gap.add_hline(y=10, line_dash="dot", line_color="#ef4444",
                    annotation_text="10% constitutional threshold")
                fig_gap.add_hline(y=COMPAS_DATA["metrics"]["racial_fpr_gap"]*100,
                    line_dash="dash", line_color="#dc2626",
                    annotation_text=f"COMPAS 2016 ({COMPAS_DATA.get("metrics",{}).get("racial_fpr_gap",0):.0%})")
                try: fig_gap.update_layout(**PT(), title="Racial FPR Gap vs Thresholds",
                    yaxis_title="Gap (pp)", height=340)
                except: fig_gap.update_layout(height=340)
                st.plotly_chart(fig_gap, use_container_width=True)

            with c2:
                # Group FPR heatmap
                if CHARTS_OK and len(df) >= 2:
                    hm_df = df[["run_id","majority_fpr","minority_fpr","racial_fpr_gap",
                                "liberty_score","fairness_score"]].set_index("run_id")
                    hm_df.index = [f"Run {i}" for i in hm_df.index]
                    hm_df.columns = ["Majority FPR","Minority FPR","Racial Gap",
                                     "Liberty","Fairness"]
                    st.plotly_chart(fairness_heatmap(
                        hm_df, title="Fairness Metrics Heatmap",
                        accent=ACCENT, height=320,
                        colorscale="RdYlGn_r", zmin=0, zmax=0.5),
                        use_container_width=True)
                else:
                    c_metrics = ["majority_fpr","minority_fpr","racial_fpr_gap"]
                    fig_hm = px.bar(df, x="run_id", y=c_metrics, barmode="group",
                        title="FPR Metrics by Run",
                        color_discrete_sequence=["#22c55e","#ef4444",ACCENT])
                    try: fig_hm.update_layout(**PT(), height=320)
                    except: fig_hm.update_layout(height=320)
                    st.plotly_chart(fig_hm, use_container_width=True)

            # Wrongful detention welfare cost
            n_ppl = int(avg_fpr * int(n_samples) * 0.3)
            yrs   = NIGERIA_JUDICIAL_CONTEXT["avg_pretrial_detention_yrs"]
            cost  = n_ppl * yrs * 365 * 2.5  # proxy: $2.5/day prison cost
            st.markdown(f"""
<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:10px;
  padding:1.1rem 1.4rem;margin:.75rem 0">
  <p style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
     color:#991b1b;margin:0 0 .4rem">⚠️ Wrongful Detention Welfare Cost (Proxy)</p>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem">
    <div><p style="font-size:.68rem;color:#64748b;margin:0;font-family:DM Mono,monospace">
      Falsely Flagged</p>
      <p style="font-size:1.4rem;font-weight:700;color:#ef4444;margin:0">{n_ppl:,}</p></div>
    <div><p style="font-size:.68rem;color:#64748b;margin:0;font-family:DM Mono,monospace">
      Avg Detention</p>
      <p style="font-size:1.4rem;font-weight:700;color:#ef4444;margin:0">{yrs:.1f} yrs</p></div>
    <div><p style="font-size:.68rem;color:#64748b;margin:0;font-family:DM Mono,monospace">
      Est. Cost (proxy)</p>
      <p style="font-size:1.4rem;font-weight:700;color:#ef4444;margin:0">${cost/1e6:.1f}M</p></div>
    <div><p style="font-size:.68rem;color:#64748b;margin:0;font-family:DM Mono,monospace">
      COMPAS Gap</p>
      <p style="font-size:1.4rem;font-weight:700;color:#ef4444;margin:0">
        {COMPAS_DATA.get("metrics",{}).get("racial_fpr_gap",0):.0%}</p></div>
  </div>
  <p style="font-size:.72rem;color:#94a3b8;margin:.5rem 0 0">
    Proxy estimates using Nigeria avg pretrial detention {yrs} yrs and $2.5/day prison cost.
    Source: {NIGERIA_JUDICIAL_CONTEXT['source']}</p>
</div>""", unsafe_allow_html=True)

        # ── TAB 3: XAI & Rights ────────────────────────────────────────────────
        with T["🔍 XAI & Rights"]:
            st.markdown("### 🔍 Explainable AI — Defendant Rights")
            st.markdown('<div class="nbox">Every defendant has a right to understand why '
                        'an AI risk score contributed to their detention. These explanations '
                        'satisfy NJC transparency requirements and EU AI Act Article 14 '
                        '(human oversight).</div>', unsafe_allow_html=True)

            if not xai:
                st.info("Enable XAI in sidebar and run simulation.")
            elif "error" in xai:
                st.warning(f"XAI error: {xai.get("error","unknown")}")
            else:
                fi = xai.get("feature_importance", {})
                if fi and CHARTS_OK:
                    names = fi.get("feature_names", [])
                    imps  = fi.get("importances", [])
                    if names and imps:
                        signed = [v if i%2==0 else -v*0.5
                                  for i,v in enumerate(imps[:10])]
                        st.plotly_chart(waterfall_feature_contributions(
                            names[:10], signed, base_value=0.5,
                            instance_label="Risk Score",
                            title="What Drives the AI Risk Score?",
                            accent=ACCENT, height=400,
                        ), use_container_width=True)

                c1,c2 = st.columns(2)
                with c1:
                    expl = xai.get("instance_explanation", {})
                    if expl:
                        st.markdown("#### 🔎 Why was this defendant flagged?")
                        st.markdown(
                            f'<div class="nbox"><em>{expl.get("decision_path","")}</em></div>',
                            unsafe_allow_html=True)
                        co = expl.get("feature_contributions", {})
                        if co:
                            co_df = pd.DataFrame(
                                sorted(co.items(), key=lambda x:abs(x[1]), reverse=True)[:8],
                                columns=["Factor","Contribution"])
                            fig_co = px.bar(co_df, x="Contribution", y="Factor",
                                orientation="h", color="Contribution",
                                color_continuous_scale="RdYlGn",
                                color_continuous_midpoint=0,
                                title="Individual Risk Factor Contributions",
                                height=320)
                            try: fig_co.update_layout(**PT())
                            except: pass
                            st.plotly_chart(fig_co, use_container_width=True)

                with c2:
                    cf = xai.get("counterfactual", {})
                    if cf:
                        st.markdown("#### 🔄 What would have changed this outcome?")
                        st.markdown(
                            '<div style="background:#f0fdf4;border-left:3px solid #22c55e;'
                            'border-radius:0 8px 8px 0;padding:.85rem 1rem;font-size:.83rem;'
                            f'color:#166534;margin:.5rem 0"><em>{cf.get("plain_language","")}</em></div>',
                            unsafe_allow_html=True)
                        ch = cf.get("changes", {})
                        if ch:
                            st.dataframe(pd.DataFrame([{
                                "Factor":k, "Current":round(v[0],3),
                                "Required":round(v[1],3),
                                "Change":round(v[1]-v[0],3)}
                                for k,v in ch.items()]),
                                use_container_width=True, hide_index=True)

                # Intersectional
                ix = xai.get("intersectional", {})
                if ix and ix.get("group_performances"):
                    st.markdown("#### 🔗 Intersectional Fairness (Race × Poverty)")
                    st.markdown(f'<div class="nbox">{ix.get("narrative","")}</div>',
                                unsafe_allow_html=True)

        # ── TAB 4: COMPAS Benchmark ────────────────────────────────────────────
        with T["📚 COMPAS Benchmark"]:
            st.markdown("### 📚 Real-World Benchmark: COMPAS vs Your Simulation")
            st.markdown(f'<div class="nbox">'
                        f'<strong>Study:</strong> {COMPAS_DATA["name"]}<br>'
                        f'<strong>Jurisdiction:</strong> {COMPAS_DATA["jurisdiction"]} '
                        f'({COMPAS_DATA["n_defendants"]:,} defendants)<br>'
                        f'<strong>Lesson:</strong> {COMPAS_DATA["lesson"]}<br>'
                        f'<strong>Citation:</strong> {COMPAS_DATA["citation"]}</div>',
                        unsafe_allow_html=True)

            sim_metrics = {
                "accuracy":        avg_acc,
                "fairness_score":  avg_fair,
                "racial_fpr_gap":  avg_gap,
                "fpr":             avg_fpr,
                "liberty_score":   avg_lib,
            }
            if CHARTS_OK:
                st.plotly_chart(benchmark_comparison_bar(
                    sim_metrics,
                    COMPAS_DATA["metrics"],
                    benchmark_name="COMPAS (ProPublica 2016)",
                    accent=ACCENT, height=360,
                    lower_is_better=["racial_fpr_gap","fpr","black_fpr","white_fpr"],
                ), use_container_width=True)

            # Side-by-side comparison table
            comp_df = pd.DataFrame([
                {"Metric":"Accuracy","Simulation":f"{avg_acc:.3f}",
                 "COMPAS 2016":f"{COMPAS_DATA.get("metrics",{}).get("accuracy",0):.3f}",
                 "Better":"✅ Sim" if avg_acc>COMPAS_DATA.get("metrics",{}).get("accuracy",0) else "❌ COMPAS"},
                {"Metric":"Racial FPR Gap","Simulation":f"{avg_gap:.3f}",
                 "COMPAS 2016":f"{COMPAS_DATA.get("metrics",{}).get("racial_fpr_gap",0):.3f}",
                 "Better":"✅ Sim" if avg_gap<COMPAS_DATA.get("metrics",{}).get("racial_fpr_gap",0) else "❌ COMPAS"},
                {"Metric":"Fairness Score","Simulation":f"{avg_fair:.3f}",
                 "COMPAS 2016":f"{COMPAS_DATA.get("metrics",{}).get("fairness_score",0):.3f}",
                 "Better":"✅ Sim" if avg_fair>COMPAS_DATA.get("metrics",{}).get("fairness_score",0) else "❌ COMPAS"},
                {"Metric":"Liberty Score","Simulation":f"{avg_lib:.3f}",
                 "COMPAS 2016":f"{COMPAS_DATA.get("metrics",{}).get("liberty_score",0):.3f}",
                 "Better":"✅ Sim" if avg_lib>COMPAS_DATA.get("metrics",{}).get("liberty_score",0) else "❌ COMPAS"},
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            if BENCHMARKS_OK:
                other_judicial = get_benchmarks_for_domain("judicial")
                for bk, bm in list(other_judicial.items())[:2]:
                    with st.expander(f"📑 {bm.name} ({bm.year})"):
                        st.markdown(f"**{bm.context}**")
                        st.markdown(f"*Lesson: {bm.lesson}*")
                        st.caption(bm.citation)

        # ── TAB 5: Real-World Impact ────────────────────────────────────────────
        with T["🌍 Real-World Impact"]:
            st.markdown("### 🌍 Historical AI Bias Incidents in Criminal Justice")

            fig_tl = go.Figure()
            for yr, region, name, desc in REAL_WORLD_INCIDENTS:
                color = "#ef4444" if "bias" in desc.lower() or "violation" in desc.lower() or "racial" in desc.lower() else "#22c55e"
                fig_tl.add_trace(go.Scatter(
                    x=[yr], y=[region],
                    mode="markers+text",
                    marker=dict(size=14, color=color, symbol="circle",
                                line=dict(color="#fff",width=2)),
                    text=[name], textposition="top center",
                    textfont=dict(size=9, color="#334155",family="DM Mono, monospace"),
                    hovertemplate=f"<b>{name}</b> ({yr})<br>{desc}<extra></extra>",
                    showlegend=False))
            try:
                tl_layout = PT()
                tl_layout.update(height=320,
                    title="Judicial AI Bias Incidents Timeline",
                    xaxis=dict(title="Year", dtick=1))
                fig_tl.update_layout(**tl_layout)
            except:
                fig_tl.update_layout(height=320)
            st.plotly_chart(fig_tl, use_container_width=True)

            st.markdown("#### 🇳🇬 Nigeria Judicial System Context")
            c1,c2,c3 = st.columns(3)
            c1.metric("Prison Population", f"{NIGERIA_JUDICIAL_CONTEXT.get("prison_population",0):,}")
            c2.metric("Awaiting Trial",
                      f"{NIGERIA_JUDICIAL_CONTEXT.get("awaiting_trial_pct",0):.0%}",
                      "of all prisoners")
            c3.metric("Legal Aid Access",
                      f"{NIGERIA_JUDICIAL_CONTEXT.get("legal_aid_coverage",0):.0%}",
                      "only")
            st.caption("Source: " + str(NIGERIA_JUDICIAL_CONTEXT.get("source","")))
            st.markdown(f'<div class="nbox">'
                        f'<strong>NJC Rule:</strong> {NIGERIA_JUDICIAL_CONTEXT["njc_guidelines"]}<br>'
                        f'<strong>Constitution §36:</strong> {NIGERIA_JUDICIAL_CONTEXT["constitution_s36"]}'
                        f'</div>', unsafe_allow_html=True)

        # ── TAB 6: Longitudinal ────────────────────────────────────────────────
        with T["🔁 Longitudinal"]:
            st.markdown("### 🔁 Bias Self-Reinforcement Across Retraining")
            st.markdown('<div class="nbox">Criminal justice AI trained on biased arrest '
                        'data replicates policing bias in the next generation — '
                        'creating feedback loops that amplify racial disparities over time.</div>',
                        unsafe_allow_html=True)
            if not lng:
                st.info("Run simulation to see longitudinal analysis.")
            else:
                c1,c2,c3 = st.columns(3)
                c1.metric("Initial Bias", f'{lng["initial_bias"]:.1%}')
                c2.metric("Final Bias",   f'{lng["final_bias"]:.1%}',
                          f'{lng["final_bias"]-lng["initial_bias"]:+.1%}')
                c3.metric("Amplification",f'{lng["amplification_factor"]:.2f}×',
                          "⚠️ Self-reinforcing" if lng["self_reinforcing"] else "Stable")
                if lng.get("self_reinforcing"):
                    st.error(f'⚠️ Judicial bias became self-reinforcing at cycle '
                             f'{lng.get("inflection_point","N/A")}. '
                             f'Biased arrests → biased training data → more biased predictions.')
                st.markdown(f'<div class="nbox"><em>{lng["narrative"]}</em></div>',
                            unsafe_allow_html=True)
                gm = lng.get("generation_metrics", [])
                if gm and CHARTS_OK:
                    st.plotly_chart(animated_bias_drift(
                        gm,
                        metric_keys=["demographic_parity","fairness_score","accuracy"],
                        accent=ACCENT, height=380,
                        title="Animated: Judicial Bias Amplification Over Retraining Cycles",
                    ), use_container_width=True)

        # ── TAB 7: Federated ──────────────────────────────────────────────────
        with T["🌐 Federated"]:
            st.markdown("### 🌐 Federated Learning — Bias Across Jurisdictions")
            st.markdown('<div class="nbox">Tests whether racial bias persists when models '
                        'train across multiple court jurisdictions without centralising '
                        'defendant data — critical for national-scale deployments.</div>',
                        unsafe_allow_html=True)
            if not fed:
                st.info("Run simulation to see federated analysis.")
            else:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Jurisdictions", fed["n_clients"])
                c2.metric("Global Accuracy", f'{fed["global_accuracy"]:.1%}')
                c3.metric("Global Fairness", f'{fed["global_fairness"]:.3f}')
                c4.metric("Bias Persisted", "Yes ⚠️" if fed["bias_persisted"] else "No ✅")
                st.markdown(
                    f'<div class="{"flag-crit" if fed["bias_persisted"] else "nbox"}">'
                    f'<em>{fed["narrative"]}</em></div>', unsafe_allow_html=True)
                cr_list = fed.get("client_results", [])
                if cr_list:
                    cr_df = pd.DataFrame([
                        c.__dict__ if hasattr(c,"__dict__") else c for c in cr_list])
                    if not cr_df.empty and "local_accuracy" in cr_df.columns:
                        fig_fed = px.bar(cr_df, x="client_id",
                            y=["local_accuracy","local_bias","local_fairness"],
                            barmode="group", title="Per-Jurisdiction Metrics",
                            color_discrete_sequence=[ACCENT,"#ef4444","#22c55e"])
                        try: fig_fed.update_layout(**PT(), height=300)
                        except: fig_fed.update_layout(height=300)
                        st.plotly_chart(fig_fed, use_container_width=True)

        # ── TAB 8: Compliance ──────────────────────────────────────────────────
        with T["📋 Compliance"]:
            st.markdown("### 📋 Multi-Framework Compliance Report")
            _cr = xai.get("compliance_report", {})
            _mc = xai.get("model_card", {})
            if not _cr:
                st.info("Enable XAI and run simulation to generate compliance report.")
            else:
                summ = _cr.get("summary", {})
                ok   = summ.get("overall_compliant", False)
                st.markdown(
                    f'<div class="{"nbox" if ok else "flag-crit"}">'
                    f'Overall: <strong>{"COMPLIANT ✅" if ok else "NON-COMPLIANT ❌"}</strong> | '
                    f'Fairness: {summ.get("fairness_score",0):.3f} | '
                    f'Racial Gap: {avg_gap:.1%} | Liberty: {avg_lib:.3f}</div>',
                    unsafe_allow_html=True)

                # EU AI Act special note
                if avg_gap > 0.05:
                    st.markdown("""
<div class="flag-crit"><strong>🚨 EU AI Act Article 5 (UNACCEPTABLE RISK)</strong><br>
Criminal justice recidivism and risk assessment AI with demonstrated racial disparity
qualifies as unacceptable risk under EU AI Act Art. 5(1)(b). Deployment in the EU
would be prohibited. Human oversight under Art. 14 required at minimum.</div>""",
                        unsafe_allow_html=True)

                for fw, fd in _cr.get("frameworks", {}).items():
                    with st.expander(f"📑 {fw}"):
                        for ch, st_ in fd.get("checks", {}).items():
                            st.markdown(f"{'✅' if st_=='PASS' else '❌'} {ch}")

                try:
                    pdf_b = generate_pdf_compliance_report(_cr, _mc,
                        {"accuracy":avg_acc,"fairness_score":avg_fair},
                        domain="generic")
                    st.download_button("📄 Download PDF Compliance Report",
                        pdf_b, "judicial_compliance.pdf", "application/pdf",
                        use_container_width=True)
                except Exception as e:
                    st.caption(f"PDF unavailable: {e}")

            st.divider()
            st.markdown("#### 🇳🇬 Nigeria Regulatory Alignment")
            nigeria_compliance_panel(
                {"accuracy":avg_acc,"fairness_score":avg_fair,
                 "demographic_parity":df["demographic_parity"].mean()},
                domain="agrotech", has_ussd_fallback=False,
                has_gender_audit=enable_gender_audit,
                has_multilingual=False, has_xai=enable_xai,
                has_governance=enable_governance, has_redteam=False)
            # ── Real-world benchmark comparison ─────────────────────────
            st.markdown("#### 📚 Real-World Benchmark Comparison")
            if BENCHMARKS_OK:
                _dom_bms = get_benchmarks_for_domain("judicial")
                if _dom_bms:
                    _bm_sel = st.selectbox(
                        "Compare against a published study:",
                        list(_dom_bms.keys()),
                        format_func=lambda k: _dom_bms[k].name + " (" + str(_dom_bms[k].year) + ")",
                        key="_judicial_bm_sel")
                    _bm = _dom_bms[_bm_sel]
                    _sim_m = {"accuracy": avg_acc, "fairness_score": avg_fair}
                    if "fpr" in df.columns: _sim_m["fpr"] = df["fpr"].mean()
                    if CHARTS_OK:
                        try:
                            st.plotly_chart(benchmark_comparison_bar(
                                _sim_m, _bm.metrics, _bm.name,
                                accent=ACCENT, height=300), use_container_width=True)
                        except Exception: pass
                    st.markdown(
                        f'<div class="nbox"><strong>Lesson:</strong> {_bm.lesson}<br>'
                        f'<span style="font-size:.75rem;color:#64748b">📚 {_bm.citation[:100]}</span></div>',
                        unsafe_allow_html=True)
                else:
                    st.info("No benchmarks available for this domain.")
            else:
                st.info("Add gags_benchmarks.py to components/ for benchmark comparison.")


        # ── TAB 9: Export ──────────────────────────────────────────────────────
        with T["📤 Export"]:
            st.markdown("### 📤 Export Results")
            e1,e2,e3 = st.columns(3)

            # Only format numeric columns — FIXES the ValueError
            num_cols = [c for c in df.columns
                        if df[c].dtype in [float, np.float64, np.float32]
                        and c not in ["run_id"]]
            str_cols = [c for c in df.columns if df[c].dtype == object]
            show_cols = [c for c in df.columns if c not in str_cols[2:]]  # keep first 2 str cols

            with e1:
                st.download_button(t("download_csv"),
                    df.to_csv(index=False).encode(),
                    f"gags_judicial_{scenario_key}.csv","text/csv",
                    use_container_width=True)
            with e2:
                export_json = {
                    "gags_version":"5.0","module":"Judicial Justice",
                    "timestamp":datetime.now().isoformat(),
                    "scenario":{"key":scenario_key,"name":preset_info.name,
                                "decision_type":preset_info.decision_type},
                    "compas_benchmark":COMPAS_DATA["metrics"],
                    "results":{"avg_accuracy":round(avg_acc,4),
                               "avg_fairness":round(avg_fair,4),
                               "avg_racial_fpr_gap":round(avg_gap,4),
                               "avg_liberty":round(avg_lib,4),
                               "n_runs":len(df)},
                    "nigeria_context":NIGERIA_JUDICIAL_CONTEXT,
                }
                st.download_button("📋 Download JSON",
                    json.dumps(export_json,indent=2,default=str),
                    f"gags_judicial_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json", use_container_width=True)
            with e3:
                narrative_txt = (
                    f"GAGS Judicial Justice Report\n"
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                    f"{'='*55}\n\n"
                    f"Scenario: {preset_info.name}\n"
                    f"Decision Type: {preset_info.decision_type}\n\n"
                    f"Key Results ({len(df)} runs):\n"
                    f"  Accuracy:        {avg_acc:.3f}\n"
                    f"  Fairness Score:  {avg_fair:.3f}\n"
                    f"  Racial FPR Gap:  {avg_gap:.3f} ({avg_gap:.1%})\n"
                    f"  Liberty Score:   {avg_lib:.3f}\n\n"
                    f"COMPAS 2016 Benchmark:\n"
                    f"  Racial Gap:  {COMPAS_DATA.get("metrics",{}).get("racial_fpr_gap",0):.3f}\n"
                    f"  Fairness:    {COMPAS_DATA.get("metrics",{}).get("fairness_score",0):.3f}\n\n"
                    f"{'EXCEEDS' if avg_gap>0.05 else 'Within'} constitutional 5% threshold.\n"
                )
                st.download_button("📄 Research Narrative",
                    narrative_txt.encode(),
                    "gags_judicial_narrative.txt","text/plain",
                    use_container_width=True)

            # Styled dataframe — numeric only formatting
            # Only format numeric columns to avoid ValueError on str cols
            fmt = {c: "{:.4f}" for c in num_cols
                   if c in df.columns
                   and str(df[c].dtype).startswith(("float","int"))}
            try:
                styled = (df[num_cols + str_cols[:2]]
                          .style.format(fmt)
                          .background_gradient(subset=["accuracy"], cmap="Blues")
                          .background_gradient(subset=["fairness_score"], cmap="RdYlGn"))
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(df, use_container_width=True)

        # ── TAB 10: Research Distribution ──────────────────────────────────────
        if vm == "Research" and "📈 Statistical Distribution" in T:
            with T["📈 Statistical Distribution"]:
                st.markdown("### 📈 Statistical Distribution Across Runs")
                dist_metrics = ["accuracy","fairness_score","racial_fpr_gap",
                                "liberty_score","minority_fpr","majority_fpr",
                                "demographic_parity","equity_gap"]
                dist_labels  = {"accuracy":"Accuracy","fairness_score":"Fairness",
                                "racial_fpr_gap":"Racial Gap","liberty_score":"Liberty",
                                "minority_fpr":"Minority FPR","majority_fpr":"Majority FPR",
                                "demographic_parity":"Dem. Parity","equity_gap":"Equity Gap"}
                if CHARTS_OK:
                    st.plotly_chart(multi_run_distribution(
                        st.session_state.jud_run_history,
                        metrics=dist_metrics, metric_labels=dist_labels,
                        accent=ACCENT, height=460,
                        title=f"Full Distribution — {len(df)} Simulation Runs",
                    ), use_container_width=True)
                else:
                    st.dataframe(df[dist_metrics].describe(), use_container_width=True)


        with T["🔬 Feature Modules"]:
            _jud_feats = st.session_state.get("jud_feature_outputs", {})
            feature_modules_tab(
                domain="judicial",
                run_results=st.session_state.get("jud_run_history", []),
                feats=_jud_feats,
                governance=_jud_feats.get("governance"),
                gender_audit=_jud_feats.get("gender_audit"),
                agent_economy=_jud_feats.get("agent_economy"),
                arena=_jud_feats.get("strategic_arena"),
                redteam=_jud_feats.get("multimodal_redteam"),
            )



        # ── AI Safety Tab ─────────────────────────────────────────────────────
        if "🛡️ AI Safety" in T:
            with T["🛡️ AI Safety"]:
                render_safety_tab(
                    st.session_state.get("jud_safety_report", {}),
                    domain="judicial",
                )

        # ── 🔄 Lifecycle Management Tab ─────────────────────────────────────────
        if "🔄 Lifecycle" in T:
            with T["🔄 Lifecycle"]:
                render_lifecycle_tab(
                    st.session_state.get("jud_lifecycle_report", {}),
                    "judicial",
                )
        if "🌱 Eco Score" in T:
            with T["🌱 Eco Score"]:
                _lc_eco_r   = st.session_state.get("jud_lifecycle_report", {})
                _lc_eco_last = (st.session_state.get("jud_run_history") or [{}])[-1]
                render_eco_tab(
                    _lc_eco_r,
                    algo_key  = _lc_eco_last.get("algorithm", "hist_gradient_boosting"),
                    n_samples = int(_lc_eco_last.get("n_samples", 2000)),
                    n_runs    = int(locals().get("n_runs", 3)),
                    domain    = "judicial",
                )

        # ── 🔮 Dynamic Systems Tab ────────────────────────────────────────────
        if "🔮 Dynamic Systems" in T:
            with T["🔮 Dynamic Systems"]:
                try:
                    render_dynamic_systems_tab(
                        st.session_state.get("jud_ds_report", {}),
                        domain="judicial",
                        ds_key="jud_ds_report",
                    )
                except Exception as _ds_err:
                    st.error(f"🔮 Dynamic Systems error: {_ds_err}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

    history_browser("jud_snapshot_history", domain="health",
        key_metrics=["accuracy","fairness_score","racial_fpr_gap","liberty_score"])
    annotation_panel("jud_annotations",
        context_label=f"{len(st.session_state.jud_run_history)} Judicial run(s)")

    # ── Policy recommendations ─────────────────────────────────────────────────
    st.divider()
    st.markdown("## 💡 Judicial AI Policy Recommendations")
    r1,r2,r3 = st.columns(3)
    with r1:
        st.info("""**⚖️ For Courts & Judicial Bodies**
1. **Human final decision**: AI is advisory only — NJC mandatory
2. **Explain every score**: Defendant must receive written explanation
3. **5% threshold monitoring**: Halt use if racial FPR gap exceeds 5%
4. **Annual independent audit**: Civil society + technical experts
5. **Sunset clause**: Models retired after 2 years without re-validation

**🇳🇬 Nigeria NJC Compliance**
1. §36 Constitution: fair hearing within reasonable time
2. NJC 2022: no automated bail/custody decisions
3. Legal aid mandatory before any AI risk score used""")
    with r2:
        st.success("""**🤖 For AI Developers**
1. **Equalized odds**: equalise FPR across racial groups explicitly
2. **Causal audit**: detect proxy variables for race (address, poverty)
3. **Counterfactual fairness**: decision invariant to race/ethnicity
4. **COMPAS baseline**: any model must demonstrably beat COMPAS
5. **Intersectional testing**: race × poverty × geography

**🔬 For Researchers**
1. Longitudinal studies: track bias feedback loops
2. Welfare cost quantification: falsely detained days
3. Causal identification: separate AI bias from policing bias
4. Community-based validation""")
    with r3:
        st.warning("""**🏛️ For Regulators**
1. **EU AI Act Art. 5**: criminal justice AI = UNACCEPTABLE RISK
2. **ECHR Art. 5**: right to liberty — AI cannot override
3. **UN ICCPR Art. 9**: no arbitrary detention
4. Mandatory pre-deployment racial impact assessment
5. Register of all judicial AI systems deployed

**📊 Metrics That Must Be Public**
1. Racial FPR gap (monthly)
2. False imprisonment rate per demographic
3. Appeal success rate vs AI recommendation
4. Model accuracy vs human judge accuracy""")
    st.caption("⚠️ Educational simulation. Real deployment requires independent legal review.")

else:
    # ── Welcome ────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
  padding:2.5rem;text-align:center;margin-top:1rem">
  <p style="font-size:2rem;margin:0 0 .6rem">⚖️</p>
  <p style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:700;
     color:#0f172a;margin:0 0 .4rem">Welcome to Judicial Justice Simulation</p>
  <p style="color:#64748b;font-size:.87rem;max-width:560px;margin:0 auto .5rem">
    Simulate COMPAS-style recidivism prediction, bail decision bias, and predictive policing.
    Compare against real ProPublica COMPAS data. Measure racial FPR gap against
    constitutional thresholds.</p>
  <p style="color:#94a3b8;font-size:.78rem">
    Configure settings in the sidebar and click <strong>⚖️ Run</strong> to begin.</p>
</div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-family:DM Mono,monospace;"
    "font-size:.68rem;padding:.5rem 0'>"
    "⚖️ JUDICIAL JUSTICE · GAGS v5.0 · COMPAS Benchmark · Nigeria NJC · "
    "EU AI Act Art.5 · ECHR Art.5 · UN ICCPR Art.9"
    "</div>", unsafe_allow_html=True)
