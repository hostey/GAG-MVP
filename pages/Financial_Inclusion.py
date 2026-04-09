# pages/08_💰_Financial_Inclusion.py
"""
Financial Inclusion & Credit Scoring Fairness — GAGS Framework v4.0
====================================================================
Credit scoring bias, ECOA disparate impact, informal economy exclusion.
Nigeria CBN/NDIC context. Redlining prevention. Light-mode design.
"""
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from components.translate import install_auto_translate, tx, tx_plotly, language_switcher
from components.i18n import t
install_auto_translate()

from components.governance_logic import (
    run_simple_simulation,
    apply_bias, simulate_data_poisoning, calculate_fairness_metrics,
    ExplainableModel, generate_compliance_report, generate_intersectional_fairness,
    AttackSeverity, simulate_longitudinal_bias, simulate_federated_learning,
    calculate_financial_fairness, FINANCIAL_SCENARIO_PRESETS,
)
from components.ux_utils import (
    guided_tour_banner, preset_selector, metric_glossary_expander,
    history_browser, save_to_history, share_url_panel, load_config_from_url,
    annotation_panel, role_switcher, get_active_role, role_banner,
    board_member_summary,
    get_role_algo, get_role_tabs, get_role_defaults,
    role_algo_banner, role_brief_banner,
)
try:
    from components.pdf_report import generate_pdf_compliance_report
    PDF_OK = True
except (ImportError, ModuleNotFoundError):
    PDF_OK = False
    def generate_pdf_compliance_report(*a, **kw):
        return None
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

st.set_page_config(page_title="Financial Inclusion • GAGS", page_icon="💰", layout="wide")

# ── Safe top-level preset_info guard ──────────────────────────────────────────
# preset_info must be defined before ANY st.markdown() calls, even if the
# sidebar hasn't executed yet (Streamlit executes top-to-bottom each rerun).
__fina_sk = st.session_state.get("_fin_scenario_key", list(FINANCIAL_SCENARIO_PRESETS.keys())[0])
if __fina_sk not in FINANCIAL_SCENARIO_PRESETS:
    __fina_sk = list(FINANCIAL_SCENARIO_PRESETS.keys())[0]
scenario_key = __fina_sk
preset_info  = FINANCIAL_SCENARIO_PRESETS[scenario_key]



# ── Auto-dismiss stale guided tour banners from other pages ───────────────────
for _tk in ["_tour_dismissed_agrotech","_tour_dismissed_health","_tour_dismissed_security"]:
    if _tk not in st.session_state:
        st.session_state[_tk] = True

# ── Design system ──────────────────────────────────────────────────────────────
try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, plotly_theme as _ptheme
    inject_css("finance")
    ACCENT = DOMAIN_ACCENTS["finance"]
    def PT(): return _ptheme(ACCENT)
except ImportError:
    ACCENT = "#0d9488"
    def PT(): return {}

# ── State ──────────────────────────────────────────────────────────────────────
_STATE = {
    "fin_run_history":      [], "fin_xai_results":      {},
    "fin_longitudinal":     None, "fin_federated":      None,
    "fin_snapshot_history": [], "fin_annotations":      [],
    "fin_ds_report": {}
}
for k, v in _STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

_VALID_BIAS_TYPES = list(simulation_config.BIAS_TYPES) + [
    b for b in ["gender","linguistic"] if b not in simulation_config.BIAS_TYPES]

_COUNTRY_META = {
    "Low-Income Country":  {"financial_inclusion":0.35,"regulatory_capacity":0.30},
    "Lower-Middle Income": {"financial_inclusion":0.55,"regulatory_capacity":0.50},
    "Upper-Middle Income": {"financial_inclusion":0.75,"regulatory_capacity":0.70},
    "High-Income Country": {"financial_inclusion":0.90,"regulatory_capacity":0.85},
}

_FN = ["monthly_income","credit_score","debt_to_income","employment_years","assets","age",
       "geography_urban","bank_relationship_yrs","credit_inquiries","delinquencies",
       "education_level","mobile_usage"]

# ── Data generator ─────────────────────────────────────────────────────────────
def _gen(n, low_inc=0.5, minority=0.3, bias_int=0.3, biases=None,
         alt_w=0.5, country_meta=None, scenario_key="sme_credit_nigeria", rs=42):
    rng    = np.random.RandomState(rs)
    preset = FINANCIAL_SCENARIO_PRESETS[scenario_key]
    biases = biases or []
    income       = rng.lognormal(10.5, 0.6, n)
    credit_score = np.clip(rng.normal(650, 100, n), 300, 850)
    dti          = np.clip(rng.exponential(0.3, n)*100, 0, 100)
    emp          = np.clip(rng.exponential(5, n), 0, 40)
    assets       = rng.lognormal(9, 1.2, n)
    age          = np.clip(rng.normal(45, 15, n), 18, 85)
    geo          = rng.beta(2, 2, n)
    bank_yrs     = rng.exponential(3, n)
    inq          = rng.poisson(2, n).astype(float)
    delq         = rng.poisson(0.5, n).astype(float)
    edu          = rng.beta(3, 2, n)
    mob          = rng.beta(5, 2, n)
    X = np.column_stack([income, credit_score, dti, emp, assets, age, geo,
                         bank_yrs, inq, delq, edu, mob])
    groups  = np.zeros(n); groups[int(n*low_inc):]  = 1
    min_arr = np.zeros(n)
    min_arr[rng.choice(n, int(n*minority), replace=False)] = 1
    if "historical_redlining" in biases:
        X[geo<0.3, 1] = np.maximum(300, X[geo<0.3, 1] - bias_int*50)
    if "proxy_discrimination" in biases:
        m = min_arr==1
        if m.any(): X[m] += rng.normal(0, bias_int*0.5, (m.sum(), 12))
    if "group_disparity" in biases:
        X[groups==0, 1] = np.maximum(300, X[groups==0, 1] - bias_int*30)
    for bt in [b for b in biases if b in _VALID_BIAS_TYPES]:
        try: X, _, _ = apply_bias(X, np.zeros(n,dtype=int), bt, bias_int,
                                   demographic_info=groups.astype(int),
                                   severity=AttackSeverity.MEDIUM)
        except: pass
    X[:, 11] *= (1 + alt_w*0.5)
    true_risk = ((np.log(np.maximum(income,1))-10)*0.30
                 + (credit_score-300)/550*0.35
                 - dti/100*0.15 + emp/40*0.10 - delq*0.05
                 + rng.normal(0, 0.08, n))
    true_risk[min_arr==1] -= preset.gender_credit_gap*0.5
    true_risk[groups==0]  -= (1-preset.bvn_coverage)*0.2
    true_risk[geo<0.3]    -= 0.1
    y = (true_risk <= np.percentile(true_risk, 35)).astype(int)
    return X.astype(np.float64), y, groups.astype(int), min_arr.astype(int), preset


def _metrics(yte, yp, gt, mt, Xte, scenario_key, preset):
    m = {"accuracy": float(accuracy_score(yte,yp)),
         "precision": float(precision_score(yte,yp,zero_division=0)),
         "recall": float(recall_score(yte,yp,zero_division=0)),
         "f1_score": float(f1_score(yte,yp,zero_division=0)),
         "approval_rate": float(np.mean(yp==0))}
    li, hi = gt==0, gt==1
    m["approval_rate_low"]  = float(np.mean(yp[li]==0)) if li.any() else 0.0
    m["approval_rate_high"] = float(np.mean(yp[hi]==0)) if hi.any() else 0.0
    m["default_rate_low"]   = float(np.mean(yte[li]==1)) if li.any() else 0.0
    m["default_rate_high"]  = float(np.mean(yte[hi]==1)) if hi.any() else 0.0
    m["accuracy_low"]       = float(accuracy_score(yte[li],yp[li])) if li.any() else 0.0
    m["accuracy_high"]      = float(accuracy_score(yte[hi],yp[hi])) if hi.any() else 0.0
    m["approval_gap"]       = abs(m["approval_rate_high"] - m["approval_rate_low"])
    di = m["approval_rate_low"] / (m["approval_rate_high"]+1e-8)
    m["disparate_impact_ratio"] = round(di, 4)
    m["ecoa_compliant"]    = di >= 0.80
    fair = calculate_fairness_metrics(yte, yp, gt)
    m["fairness_score"]    = fair.get("fairness_score", 0.5)
    m["demographic_parity"]= fair.get("demographic_parity_difference", 0)
    m["inclusion_score"]   = float(np.clip(1 - m["approval_gap"]*0.5
                                           - abs(m["accuracy_high"]-m["accuracy_low"])*0.3
                                           - (1-di)*0.2, 0, 1))
    ff = calculate_financial_fairness(yte, yp, Xte, _FN, preset)
    m["financial_inclusion_score"] = ff.financial_inclusion_score
    m["fnr_gap"]    = ff.fnr_gap
    m["narrative"]  = ff.narrative
    ap = yp==0
    m["income_disparity"] = float(Xte[ap,0].mean()/(Xte[~ap,0].mean()+1e-8)) if ap.any() and (~ap).any() else 1.0
    if len(np.unique(yte)) > 1:
        tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
        m["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    else:
        m["fpr"] = 0.0
    return m


def _run_one(scenario_key, n_samples, biases, bias_int, poison_rate, run_idx,
             low_inc, minority, alt_w, reg, country_meta, enable_xai, enable_gov, selected_state=None):
    X, y, groups, minority_arr, preset = _gen(
        n_samples, low_inc, minority, bias_int, biases, alt_w, country_meta,
        scenario_key, rs=42+run_idx)
    X = X.astype(np.float64)
    try:
        X, y, groups = simulate_data_poisoning(
            X, y, poison_rate*(1+(1-country_meta.get("regulatory_capacity",0.5))),
            attack_type="label_flipping", demographic_info=groups, targeted=True)
    except: pass
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte, gtr, gte, mtr, mte = train_test_split(
        Xs, y, groups, minority_arr, test_size=0.3, random_state=42+run_idx,
        stratify=y if (len(np.unique(y)) > 1 and np.bincount(y.astype(int)).min() >= 2) else None)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42+run_idx)
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    m = _metrics(yte, yp, gte, mte, Xte, scenario_key, preset)
    if enable_xai and run_idx == 0:
        try:
            xm = ExplainableModel(domain="finance"); xm.model=clf
            xm._X_train=Xtr; xm._is_fitted=True; xm.feature_names=_FN
            fi = xm.feature_importance(Xte, yte, n_repeats=6)
            di_idx = np.where(yte==1)[0]
            xai = {"feature_importance": fi.__dict__}
            if len(di_idx):
                expl = xm.explain_instance(Xte[di_idx[0]])
                cf   = xm.counterfactual(Xte[di_idx[0]])
                xai["instance_explanation"] = expl.__dict__
                xai["counterfactual"]       = cf.__dict__
            mc = xm.model_card({"accuracy":m["accuracy"],"recall":m["recall"]},
                {"fairness_score":m["fairness_score"],
                 "demographic_parity_difference":m["demographic_parity"]}, domain="finance")
            cr = generate_compliance_report(mc,
                {"fairness_score":m["fairness_score"],
                 "demographic_parity_difference":m["demographic_parity"]},
                {"accuracy":m["accuracy"]},
                frameworks=["EU AI Act","ISO 42001","NIST AI RMF","NITDA","UNESCO"])
            ix = generate_intersectional_fairness(yte, yp,
                {"income":gte,"minority":mte}, min_group_size=15)
            xai.update({"model_card":mc.__dict__,"compliance_report":cr,"intersectional":ix.__dict__})
            st.session_state.fin_xai_results = xai
        except Exception as e:
            st.session_state.fin_xai_results = {"error": str(e)}
    if run_idx == 0:
        bl = bias_int if bias_int > 0 else 0.15
        try:
            lng = simulate_longitudinal_bias(X, y, groups, initial_bias_type="socioeconomic",
                initial_bias_intensity=bl, n_generations=5, random_state=42)
            st.session_state.fin_longitudinal = lng.__dict__
        except: st.session_state.fin_longitudinal = None
        try:
            fed = simulate_federated_learning(X, y, groups, n_clients=4, n_rounds=3,
                bias_heterogeneity=bl*0.5, random_state=42)
            st.session_state.fin_federated = fed.__dict__
        except: st.session_state.fin_federated = None
    # ── Feature modules (run when enabled) ───────────────────────────────────

    return {
        "run_id":run_idx+1, "scenario":preset.name,
        "accuracy":m["accuracy"], "precision":m["precision"],
        "recall":m["recall"], "f1_score":m["f1_score"],
        "fpr": m["fpr"],
        "approval_rate":m["approval_rate"],
        "approval_rate_low":m["approval_rate_low"],
        "approval_rate_high":m["approval_rate_high"],
        "default_rate_low":m["default_rate_low"],
        "default_rate_high":m["default_rate_high"],
        "approval_gap":m["approval_gap"],
        "disparate_impact_ratio":m["disparate_impact_ratio"],
        "ecoa_compliant":m["ecoa_compliant"],
        "fairness_score":m["fairness_score"],
        "inclusion_score":m["inclusion_score"],
        "financial_inclusion_score":m["financial_inclusion_score"],
        "demographic_parity":m["demographic_parity"],
        "fnr_gap":m["fnr_gap"],
        "income_disparity":m["income_disparity"],
        "bias_intensity":bias_int, "poison_rate":poison_rate,
        "regulatory_compliance":reg,
        "biases":", ".join(biases) or "None",
        "narrative":m["narrative"],
    }


# ── URL / tour ─────────────────────────────────────────────────────────────────
load_config_from_url()
guided_tour_banner("finance")

# ── Sidebar ────────────────────────────────────────────────────────────────────

def _safe_fmt(df, float_fmt="{:.3f}", exclude=None):
    """Format only numeric df columns — prevents ValueError on string columns."""
    _excl = set(exclude or []) | {"scenario","biases","narrative","equity_narrative",
                                   "run_id","regulatory_body","citation","institution_type"}
    num_cols = [c for c in df.columns
                if c not in _excl and str(df[c].dtype).startswith(("float","int"))]
    try:
        return df.style.format({c: float_fmt for c in num_cols if c in df.columns})
    except Exception:
        return df.style


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


with st.sidebar:
    language_switcher(location="sidebar"); st.divider()

    # ── State selector ────────────────────────────────────────────────────────
    st.divider()
    selected_state = state_selector(key="_state_08financialinclusion", location="sidebar")
    state_info_card(selected_state)

    role_switcher("financial")
    progress_tracker(location="sidebar")
    role_algo_banner("financial"); st.divider()
    # ── View Mode ────────────────────────────────────────────
    _vm_key = "_vm_finance"
    if _vm_key not in st.session_state:
        st.session_state[_vm_key] = "Industry"
    view_mode = st.radio(t("perspective"), ["Industry", "Research", "Consumer Impact"],
        horizontal=True, key=_vm_key,
        help="Industry: KPI dashboard. Research: statistical depth. Consumer Impact: borrower exclusion focus.")
    st.divider()

    st.markdown(f"""<div style="text-align:center;padding:.5rem 0">
      <h2 style="color:{ACCENT};margin:0;font-family:'Syne',sans-serif">⚙️ Financial Config</h2>
      <p style="color:#64748b;font-size:.75rem;margin:.2rem 0 0">
        Financial Inclusion & Credit Fairness</p></div>""", unsafe_allow_html=True)
    st.divider()

    st.subheader("🏦 Financial Context")
    country_income_level = st.selectbox("Country Income Level", list(_COUNTRY_META.keys()), index=1)
    country_meta = _COUNTRY_META[country_income_level]
    institution_type = st.selectbox("Institution Type",
        ["Traditional Bank","Digital Bank","Microfinance",
         "Credit Union","Fintech Platform","Payday Lender"])

    st.subheader("💰 GAGS Scenario")
    scenario_key = st.selectbox("Financial Scenario",
        list(FINANCIAL_SCENARIO_PRESETS.keys()),
        format_func=lambda k: FINANCIAL_SCENARIO_PRESETS[k].name)
    preset_info = FINANCIAL_SCENARIO_PRESETS[scenario_key]
    st.caption(preset_info.description[:200])
    st.divider()

    st.subheader(f"🎭 {t('bias_config')}")
    _fo = list(dict.fromkeys(
        list(_VALID_BIAS_TYPES) +
        ["historical_redlining","proxy_discrimination","group_disparity",
         "feature_correlation","measurement_bias","sample_selection"]))
    _fs = [b for b in ["demographic","socioeconomic","historical_redlining","proxy_discrimination"]
           if b in _fo]
    selected_biases = st.multiselect("Bias Types", options=_fo, default=_fs,
        format_func=lambda x: (f"🏘️ {x}" if x=="historical_redlining"
                               else f"🔍 {x}" if x=="proxy_discrimination"
                               else f"🔴 {x}" if x in ("gender","demographic")
                               else f"⚠️ {x}"))
    if selected_biases:
        st.markdown("Active: " + "".join(
            f'<span style="display:inline-block;background:#fef2f2;border:1px solid #fecaca;'
            f'border-radius:4px;padding:1px 7px;font-size:.7rem;color:#991b1b;margin:2px">{b}</span>'
            for b in selected_biases), unsafe_allow_html=True)
    bias_intensity = st.slider("Bias Intensity", 0.0, float(simulation_config.MAX_BIAS_FACTOR), 0.3, 0.05)
    st.divider()

    st.subheader("👥 Demographics")
    low_income_ratio = st.slider("Low-Income Customers", 0.0, 1.0, 0.50, 0.05)
    minority_ratio   = st.slider("Minority Representation", 0.0, 1.0, 0.30, 0.05)
    st.divider()

    st.subheader("💳 Credit Scoring")
    traditional_data_weight = st.slider("Traditional Data Weight", 0.0, 1.0, 0.70, 0.05)
    alternative_data_use    = st.select_slider("Alternative Data Usage",
        options=["None","Limited","Moderate","Extensive","Primary"], value="Moderate")
    alt_data_weight = {"None":0.0,"Limited":0.2,"Moderate":0.5,"Extensive":0.8,"Primary":1.0}[alternative_data_use]
    regulatory_compliance   = st.slider("Regulatory Compliance", 0.0, 1.0, 0.60, 0.05)
    st.divider()

    st.subheader(f"⚠️ {t('attack_header')}")
    poison_rate = st.slider("Attack Strength", 0.0, 0.5, 0.05, 0.01, format="%.2f")
    st.divider()

    st.subheader(f"📊 {t('sim_params_header')}")
    n_samples = st.number_input("Customer Records", 1000, 100000, settings.DEFAULT_N_SAMPLES, 1000)
    n_runs    = st.slider("Simulation Runs", 1, 10, 3)
    st.divider()

    st.subheader(f"🔬 {t('modules_header')}")
    enable_xai        = st.toggle("Explainable AI", value=True)
    enable_governance   = st.toggle("Governance Layer", value=True)
    enable_redteam         = st.toggle("Multimodal Red Team", value=False, help="Adversarial attacks on AI decisions.")
    enable_agent_economy   = st.toggle("Agent Economy", value=False, help="Vickrey auction resource allocation.")
    enable_arena           = st.toggle("Strategic Arena", value=False, help="Game-theoretic multi-agent negotiation.")

    enable_ai_safety    = st.toggle("🛡️ AI Safety Analysis", value=False, help="Run adversarial robustness, OOD detection, uncertainty quantification, and NIST/ISO safety checklists.")
    enable_lifecycle   = st.toggle("🔄 Lifecycle Management", value=False, help="Model registry, drift monitoring, compliance audit.")
    enable_eco         = st.toggle("🌱 Eco Analysis", value=False, help="Energy consumption, CO₂ emissions, eco-score rankings.")
    enable_dynamic    = st.toggle("🔮 Dynamic Systems", value=False, help="System dynamics, MDP, information theory, causal fairness, evolutionary game theory, CAS.")
    enable_gender_audit = st.toggle("Gender Equity Audit", value=False)
    st.divider()

    col_r, col_x = st.columns(2)
    run_btn = col_r.button(t("run_simulation"), type="primary", use_container_width=True)
    if col_x.button(t("reset"), use_container_width=True):
        for k, v in _STATE.items(): st.session_state[k] = type(v)()
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(13,148,136,.06) 0%,rgba(13,148,136,.02) 100%);
  border:1px solid rgba(13,148,136,.2);border-left:4px solid {ACCENT};
  border-radius:12px;padding:1.75rem 2rem;margin-bottom:1.25rem;position:relative;overflow:hidden">
  <p style="font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.16em;
     text-transform:uppercase;color:{ACCENT};margin:0 0 .4rem;
     display:flex;align-items:center;gap:.4rem">
    <span style="display:inline-block;width:14px;height:1.5px;background:{ACCENT}"></span>
    FINANCIAL INCLUSION · GAGS v4.0 · Nigeria CBN/NDIC
  </p>
  <h1 style="font-family:'Syne',sans-serif!important;font-size:1.9rem!important;
     font-weight:800!important;color:#0f172a!important;margin:0 0 .3rem!important">
    💰 Financial Inclusion & Credit Scoring Fairness</h1>
  <p style="color:#475569;font-size:.9rem;line-height:1.65;margin:0;max-width:680px">
    Algorithmic bias in credit scoring, loan approval, insurance pricing, and mobile money KYC —
    with ECOA disparate impact analysis and Nigeria CBN/NDIC regulatory context.</p>
  <div style="margin-top:.7rem">
    {"".join(f'<span style="display:inline-flex;align-items:center;padding:.18rem .65rem;'
             f'border-radius:99px;font-family:DM Mono,monospace;font-size:.64rem;'
             f'border:1px solid rgba(13,148,136,.25);color:{ACCENT};'
             f'background:rgba(13,148,136,.08);margin:.15rem .1rem 0 0">{b}</span>'
             for b in ["Nigeria CBN/NDIC","Credit Scoring Bias","ECOA Disparate Impact",
                       "Informal Economy Exclusion","Redlining Prevention"])}
  </div>
</div>""", unsafe_allow_html=True)

# Scenario banner
st.markdown(f"""
<div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:8px;
  padding:.75rem 1.1rem;margin-bottom:1rem;font-family:'DM Mono',monospace;
  font-size:.75rem;color:#134e4a">
  <strong style="color:{ACCENT}">{preset_info.name}</strong> ·
  Product: <strong>{preset_info.product_type.replace('_',' ').title()}</strong> ·
  Exclusion rate: <strong>{preset_info.exclusion_rate:.0%}</strong> ·
  Gender credit gap: <strong>{preset_info.gender_credit_gap:.0%}</strong> ·
  Informal income: <strong>{preset_info.informal_income_pct:.0%}</strong> ·
  Regulator: <strong>{preset_info.regulatory_body}</strong>
</div>""", unsafe_allow_html=True)

# ── Run ─────────────────────────────────────────────────────────────────────────
if run_btn:
    enable_lifecycle = locals().get("enable_lifecycle", False)
    enable_eco       = locals().get("enable_eco", False)
    enable_dynamic   = locals().get("enable_dynamic", False)
    enable_arena = locals().get("enable_arena", False)
    enable_agent_economy = locals().get("enable_agent_economy", False)
    enable_redteam = locals().get("enable_redteam", False)
    enable_dynamic   = locals().get("enable_dynamic", False)
    enable_ai_safety = locals().get("enable_ai_safety", st.session_state.get("_ais_toggle", False))
    st.session_state.fin_run_history = []
    st.session_state["fin_feature_outputs"] = {}
    prog = st.progress(0, text=t("loading"))
    for i in range(n_runs):
        prog.progress(i/n_runs, text=f"Run {i+1}/{n_runs}…")
        with st.spinner(f"Simulation {i+1}/{n_runs}"):
            r = _run_one(scenario_key, int(n_samples), selected_biases, bias_intensity,
                         poison_rate, i, low_income_ratio, minority_ratio, alt_data_weight,
                         regulatory_compliance, country_meta, enable_xai, enable_governance)
            if r:
                st.session_state.fin_run_history.append(r)


                save_to_history("fin_snapshot_history",
                    label=f"Run {i+1} | DI={r['disparate_impact_ratio']:.2f} | {scenario_key[:14]}",
                    metrics={"accuracy":r["accuracy"],"fairness_score":r["fairness_score"],
                             "inclusion_score":r["inclusion_score"],
                             "disparate_impact_ratio":r["disparate_impact_ratio"]},
                    config={"scenario_key":scenario_key,"bias_intensity":bias_intensity,
                            "regulatory_compliance":regulatory_compliance,
                            "country_income_level":country_income_level})


    # ── Run enabled feature modules (results stored per-session) ──────
    if "fin_feature_outputs" not in st.session_state:
        st.session_state["fin_feature_outputs"] = {}
    _fout = st.session_state["fin_feature_outputs"]

    if enable_governance:
        try:
            from components.governance_logic import HybridGovernanceLayer as _HGL
            _hgl_inst = _HGL()
            _hgl_baseline = {"accuracy": 0.75, "fairness_score": 0.70}
            _hgl_current  = {"accuracy": 0.70, "fairness_score": 0.60}
            _hgl_entry = _hgl_inst.propose_and_vote(
                "Deploy AI in financial domain",
                _hgl_baseline, _hgl_current)
            _fout["governance"] = {
                "policy":      "Deploy AI in financial domain",
                "outcome":     _hgl_entry.vote_outcome.value if hasattr(_hgl_entry, "vote_outcome") else "approved",
                "tally":       _hgl_entry.vote_tally if hasattr(_hgl_entry, "vote_tally") else {},
                "ai_flags":    _hgl_entry.ai_flags if hasattr(_hgl_entry, "ai_flags") else [],
                "ledger_hash": _hgl_entry.hash if hasattr(_hgl_entry, "hash") else "N/A",
                "ledger_entries": 1,
            }
        except Exception as _ex:
            _fout["governance"] = {
                "policy": "Deploy AI in financial domain",
                "outcome": "approved", "tally": {"for":60,"against":30,"abstain":10},
                "ai_flags": [], "ledger_hash": "N/A", "ledger_entries": 0,
                "narrative": str(_ex),
            }

    if enable_gender_audit:
        _h = st.session_state.get("fin_run_history", [{}])
        _fout["gender_audit_gap"] = _h[-1].get("gender_gap", 0) if _h else 0


    if enable_redteam:
        try:
            _fout["multimodal_redteam"] = run_redteam_simulation(domain="financial")
        except Exception as _ex:
            _fout["multimodal_redteam"] = {"combined_bypass_rate":0,"modality_results":[],"error":str(_ex)}

    if enable_agent_economy:
        try:
            _fout["agent_economy"] = run_agent_economy_simulation(domain="financial")
        except Exception as _ex:
            _fout["agent_economy"] = {"gini_coefficient":0,"agent_summary":[],"error":str(_ex)}

    if enable_arena:
        try:
            _fout["arena"] = run_arena_simulation(domain="financial")
        except Exception as _ex:
            _fout["arena"] = {"final_standings":[],"deception_rate":0,"error":str(_ex)}
    # ── AI Safety & Robustness Suite ──────────────────────────────────────────
    if enable_ai_safety:
        try:
            import numpy as np
            _last_run = st.session_state.get("fin_run_history", [{}])[-1]
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
                model=None, domain="financial",
                enable_robustness=True, enable_ood=True,
                enable_uncertainty=True, enable_checklists=True,
                simulation_metrics=_sim_metrics,
            )
            st.session_state["fin_safety_report"] = _safety_report
        except Exception as _se:
            st.session_state["fin_safety_report"] = {"error": str(_se), "pillars": {}}

    # ── Lifecycle Management & Environmental Sustainability ────────────────────
    if enable_lifecycle or enable_eco:
        try:
            _last_r = st.session_state.get("fin_run_history", [{}])
            _last_r = _last_r[-1] if _last_r else {}
            _algo_k = _last_r.get("algorithm", "hist_gradient_boosting")
            _algo_l = _last_r.get("algo_label", "Hist Gradient Boosting")
            _lc_met = {k: v for k, v in _last_r.items() if isinstance(v, (int, float))}
            _lc_met["has_governance"]   = locals().get("enable_governance", False)
            _lc_met["has_gender_audit"] = locals().get("enable_gender_audit", False)
            _lc_met["has_xai"]          = True
            _lc_rep = run_lifecycle_suite(
                domain="financial", algo_key=_algo_k, algo_label=_algo_l,
                n_samples=int(_last_r.get("n_samples", locals().get("sample_size", locals().get("n_samples", 2000)))),
                n_runs=int(locals().get("n_runs", 3)), n_features=10,
                metrics=_lc_met,
                safety_data=st.session_state.get("fin_safety_report") or None,
                enable_registry=enable_lifecycle, enable_monitoring=enable_lifecycle,
                enable_audit=enable_lifecycle, enable_eco=enable_eco,
            )
            st.session_state["fin_lifecycle_report"] = _lc_rep
        except Exception as _lce:
            st.session_state["fin_lifecycle_report"] = {"error": str(_lce), "pillars": {}}


    # ── Dynamic Systems Modelling Suite ─────────────────────────────────────────
    if enable_dynamic:
        try:
            _ds_hist   = st.session_state.get("fin_run_history", [])
            _ds_params = derive_ds_params(domain="financial", run_history=_ds_hist)
            _ds_rep    = run_dynamic_systems_suite(
                domain="financial",
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
            st.session_state["fin_ds_report"] = _ds_rep
        except Exception as _dse:
            st.session_state["fin_ds_report"] = {"error": str(_dse), "pillars": {}}
    prog.progress(1.0, text=t("complete"))
    prog.empty()
# ── Post-run interactivity (shown once, after all runs complete) ──────────
if st.session_state.get("fin_run_history"):
    feats = st.session_state.get("fin_feature_outputs", {})
    _post_last  = st.session_state["fin_run_history"][-1]
    _post_fs    = _post_last.get("fairness_score", 0.5)
    track_run(_post_fs, "financial")
    _post_mc    = {k: v for k, v in _post_last.items() if isinstance(v, (int, float))}
    multi_challenge_panel("financial", _post_mc)
    admin_challenge_panel("financial")
    benchmark_challenge_panel("financial", _post_mc)
    what_if_explorer("financial", _post_mc,
        st.session_state.get("bias_intensity", 0.3))


# ── Dashboard ──────────────────────────────────────────────────────────────────
if st.session_state.fin_run_history:
    df: pd.DataFrame = pd.DataFrame()  # safe default; overwritten below
    df  = pd.DataFrame(st.session_state.fin_run_history)
    xai = st.session_state.fin_xai_results
    lng = st.session_state.fin_longitudinal
    fed = st.session_state.fin_federated

    avg_acc  = df["accuracy"].mean()
    avg_fair = df["fairness_score"].mean()
    avg_incl = df["inclusion_score"].mean()
    avg_gap  = df["approval_gap"].mean()
    avg_apr  = df["approval_rate"].mean()
    avg_di   = df["disparate_impact_ratio"].mean()

    role_banner("financial")
    role_brief_banner("financial")
    share_url_panel("health", config={"domain":"finance","scenario_key":scenario_key,
                                       "bias_intensity":bias_intensity})

    if get_active_role("health") == "Board Member":
        board_member_summary("health", avg_acc, avg_fair, avg_fair >= 0.70,
            f"Financial AI {'meets' if avg_fair>=0.70 else 'does NOT meet'} fairness threshold. "
            f"DI ratio: {avg_di:.2f} ({'ECOA compliant' if avg_di>=0.80 else 'NON-COMPLIANT'}).",
            "Conduct ECOA audit before deployment. Publish per-demographic approval rates quarterly.")
    else:
        # KPI cards
        k1,k2,k3,k4 = st.columns(4)
        def _kpi(col, label, value, status=""):
            col.markdown(f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;'
                         f'padding:1rem 1.25rem;box-shadow:0 1px 3px rgba(0,0,0,.05);'
                         f'position:relative;overflow:hidden">'
                         f'<p style="font-family:DM Mono,monospace;font-size:.67rem;'
                         f'letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin:0 0 .15rem">'
                         f'{label}</p>'
                         f'<p style="font-family:Syne,sans-serif;font-size:1.85rem;font-weight:700;'
                         f'color:{ACCENT};margin:0">{value}</p>'
                         f'<div style="position:absolute;bottom:0;left:0;right:0;height:3px;'
                         f'background:{"#22c55e" if status=="ok" else "#ef4444" if status=="crit" else ACCENT}'
                         f';opacity:.7"></div></div>', unsafe_allow_html=True)
        _kpi(k1, "✅ Overall Approval Rate", f"{avg_apr:.1%}")
        _kpi(k2, "⚖️ Fairness Score", f"{avg_fair:.2f}/1.0", "ok" if avg_fair>=0.70 else "crit")
        _kpi(k3, "🌍 Inclusion Score", f"{avg_incl:.2f}/1.0", "ok" if avg_incl>=0.65 else "warn")
        _kpi(k4, "📉 Approval Gap", f"{avg_gap:.1%}", "ok" if avg_gap<0.10 else "crit")

        if avg_di < 0.80:
            st.markdown(f'<div class="alert-danger">🚨 <strong>ECOA NON-COMPLIANT</strong>: '
                        f'Disparate impact ratio {avg_di:.2f} is below the 0.80 (80%) threshold. '
                        f'The approval rate for the minority group is significantly lower than the majority.</div>',
                        unsafe_allow_html=True)

        # Per-group charts
        c1,c2 = st.columns(2)
        with c1:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(name="Low-Income", x=df["run_id"],
                y=df["approval_rate_low"]*100, marker_color="#ef4444", opacity=0.85))
            fig1.add_trace(go.Bar(name="Higher-Income", x=df["run_id"],
                y=df["approval_rate_high"]*100, marker_color="#22c55e", opacity=0.85))
            try: fig1.update_layout(**PT(), title="Approval Rates by Income Group",
                barmode="group", yaxis_title="Approval Rate (%)", height=320)
            except: fig1.update_layout(barmode="group", height=320)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Low-Income Default Rate", x=df["run_id"],
                y=df["default_rate_low"]*100, marker_color="#2563eb", opacity=0.85))
            fig2.add_trace(go.Bar(name="Higher-Income Default Rate", x=df["run_id"],
                y=df["default_rate_high"]*100, marker_color="#7c3aed", opacity=0.85))
            try: fig2.update_layout(**PT(), title="Actual Default Rates by Income Group",
                barmode="group", yaxis_title="Default Rate (%)", height=320)
            except: fig2.update_layout(barmode="group", height=320)
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        metric_glossary_expander(["fairness score","demographic parity","false positive rate",
                                   "equalized odds","bias intensity","poison rate"])

        _tab_labels = [
            "📈 Performance",
            "⚖️ Fairness",
            "💰 Economic Impact",
            "🧠 Explainable AI",
            "📋 Compliance",
            "🔁 Longitudinal",
            "🌐 Federated",
            "📋 Raw Results",
            "🛡️ AI Safety",
            "🔄 Lifecycle",
            "🌱 Eco Score",
            "🔬 Feature Modules",
            "🔮 Dynamic Systems"
        ]
        _tabs_obj = st.tabs(_tab_labels)
        T = {n: _tab for n, _tab in zip(_tab_labels, _tabs_obj)}

        with T["📈 Performance"]:
            fg = make_subplots(rows=1, cols=4,
                specs=[[{"type":"indicator"}]*4],
                subplot_titles=("Accuracy","Fairness","Inclusion","Parity"))
            for ci,(val,bc,th) in enumerate([
                (avg_acc*100,"#0d9488",80),(avg_fair*100,"#2563eb",75),
                (avg_incl*100,"#d97706",65),(100-avg_gap*100,"#dc2626",85)],1):
                fg.add_trace(go.Indicator(mode="gauge+number", value=val,
                    gauge={"axis":{"range":[0,100]},"bar":{"color":bc},
                           "threshold":{"line":{"color":"red","width":4},
                                        "thickness":0.75,"value":th}}),row=1,col=ci)
            fg.update_layout(height=280, showlegend=False,
                             paper_bgcolor="#ffffff",
                             font=dict(family="DM Mono, monospace", color="#334155"))
            st.plotly_chart(fg, use_container_width=True)

            fig_sc = px.scatter(df, x="fairness_score", y="accuracy",
                size="inclusion_score", color="bias_intensity",
                hover_data=["biases","regulatory_compliance","disparate_impact_ratio"],
                title="Financial Inclusion Trilemma: Accuracy vs Fairness vs Inclusion",
                size_max=30, color_continuous_scale="YlOrRd")
            fig_sc.add_shape(type="rect",x0=0.70,x1=1.0,y0=0.70,y1=1.0,
                line=dict(color="#22c55e",width=2,dash="dash"),
                fillcolor="rgba(34,197,94,0.06)")
            fig_sc.add_annotation(x=0.85,y=0.85,text="Optimal Zone",
                font=dict(color="#16a34a",size=10),showarrow=False)
            try: fig_sc.update_layout(**PT(), height=360)
            except: fig_sc.update_layout(height=360)
            st.plotly_chart(fig_sc, use_container_width=True)

        with T["⚖️ Fairness"]:
            c1,c2 = st.columns(2)
            with c1:
                fig_fg = px.bar(df, x="run_id",
                    y=["approval_gap","demographic_parity"],
                    barmode="group", title="Fairness Gaps Across Runs",
                    color_discrete_sequence=["#ef4444","#7c3aed"],
                    labels={"value":"Gap","variable":"Metric"})
                try: fig_fg.update_layout(**PT(), height=320)
                except: fig_fg.update_layout(height=320)
                st.plotly_chart(fig_fg, use_container_width=True)
            with c2:
                factors = pd.DataFrame({
                    "Factor":["Bias Intensity","Regulatory Compliance",
                              "Alternative Data","Traditional Data Weight"],
                    "Impact":[-bias_intensity*0.8, regulatory_compliance*0.6,
                               alt_data_weight*0.4, -traditional_data_weight*0.3]})
                fig_fac = px.bar(factors, x="Factor", y="Impact",
                    title="Factors Affecting Financial Fairness",
                    color="Impact", color_continuous_scale="RdYlGn")
                try: fig_fac.update_layout(**PT(), height=320)
                except: fig_fac.update_layout(height=320)
                st.plotly_chart(fig_fac, use_container_width=True)

            comp_label = "Strict" if regulatory_compliance>0.7 else "Moderate" if regulatory_compliance>0.4 else "Lax"
            st.markdown(
                f'<div class="nbox"><strong>Regulatory Setting: '
                f'<span style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:4px;'
                f'padding:1px 8px;font-family:DM Mono,monospace;font-size:.78rem">{comp_label}</span>'
                f'</strong> — Compliance: {regulatory_compliance:.1%} | '
                f'Bias reduction: {1-regulatory_compliance/2:.1%} | '
                f'Inclusion boost: {regulatory_compliance*0.3:.1%}</div>',
                unsafe_allow_html=True)
            if avg_fair < 0.70:
                st.markdown("""
<div class="alert-warning">
  <strong>⚠️ SIGNIFICANT FAIRNESS GAPS DETECTED</strong><br>
  Recommended: Alternative data (mobile money, utility payments), bias audits,
  explainable denials, redlining prevention.
</div>""", unsafe_allow_html=True)
            else:
                st.success("✅ Good fairness achieved. Continue monitoring approval rates by demographic group.")

        with T["💰 Economic Impact"]:
            base_incl = country_meta["financial_inclusion"]
            segs   = ["Low-Income","Minority","Rural","Young Adults","General"]
            cur    = [base_incl*f for f in [0.50,0.60,0.40,0.70,1.0]]
            pot    = [min(1.0,c+avg_incl*g) for c,g in zip(cur,[0.30,0.25,0.35,0.20,0.10])]
            fig_seg = go.Figure()
            fig_seg.add_trace(go.Bar(name="Current Inclusion", x=segs, y=cur,
                marker_color="#94a3b8", text=[f"{r:.0%}" for r in cur], textposition="auto"))
            fig_seg.add_trace(go.Bar(name="Potential with Fair AI", x=segs, y=pot,
                marker_color=ACCENT, text=[f"{r:.0%}" for r in pot], textposition="auto"))
            try: fig_seg.update_layout(**PT(), title="Financial Inclusion by Segment",
                barmode="group", yaxis=dict(range=[0,1],title="Inclusion Rate"), height=320)
            except: fig_seg.update_layout(barmode="group", height=320)
            st.plotly_chart(fig_seg, use_container_width=True)
            econ = [(pot[i]-cur[i])*[0.30,0.20,0.25,0.15,1.0][i]*100*[1000,2000,1500,3000,5000][i]*2.5
                    for i in range(5)]
            total = sum(econ)
            st.markdown(
                f'<div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:10px;'
                f'padding:1.25rem 1.5rem;margin:.75rem 0">'
                f'<p style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;'
                f'color:#134e4a;margin:0 0 .25rem">Total Economic Impact</p>'
                f'<p style="font-size:2rem;margin:0;font-weight:700;color:{ACCENT}">'
                f'${total/1e6:,.1f} Million</p>'
                f'<p style="font-size:.82rem;color:#0d9488;margin:.25rem 0 0">'
                f'Estimated annual boost from improved financial inclusion</p>'
                f'</div>', unsafe_allow_html=True)

        with T["🧠 Explainable AI"]:
            st.markdown("### 🧠 Explainable AI")
            if not xai: st.info("Enable XAI in sidebar and run simulation.")
            elif "error" in xai: st.warning(f"XAI error: {xai.get("error","unknown")}")
            else:
                fi = xai.get("feature_importance",{})
                if fi:
                    st.markdown(f'<div class="nbox"><em>{fi.get("narrative","")}</em></div>',
                                unsafe_allow_html=True)
                    fi_df = pd.DataFrame({"Feature":fi["feature_names"][:10],
                                          "Importance":fi["importances"][:10]})
                    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        title="What drives credit decisions?", color="Importance",
                        color_continuous_scale="YlOrRd")
                    fig_fi.update_layout(yaxis={"categoryorder":"total ascending"}, height=340)
                    try: fig_fi.update_layout(**PT())
                    except: pass
                    st.plotly_chart(fig_fi, use_container_width=True)
                c1,c2 = st.columns(2)
                with c1:
                    ex = xai.get("instance_explanation",{})
                    if ex:
                        st.markdown("#### Why was this customer denied?")
                        st.markdown(f'<div class="nbox"><em>{ex.get("decision_path","")}</em></div>',
                                    unsafe_allow_html=True)
                        co = ex.get("feature_contributions",{})
                        if co:
                            c_df = pd.DataFrame(
                                sorted(co.items(),key=lambda x:abs(x[1]),reverse=True)[:8],
                                columns=["Feature","Contribution"])
                            fco = px.bar(c_df, x="Contribution", y="Feature", orientation="h",
                                color="Contribution", color_continuous_scale="RdYlGn",
                                color_continuous_midpoint=0, height=300)
                            try: fco.update_layout(**PT())
                            except: pass
                            st.plotly_chart(fco, use_container_width=True)
                with c2:
                    cf = xai.get("counterfactual",{})
                    if cf:
                        st.markdown("#### What would get this customer approved?")
                        st.markdown(f'<div class="nbox"><em>{cf.get("plain_language","")}</em></div>',
                                    unsafe_allow_html=True)
                        ch = cf.get("changes",{})
                        if ch:
                            st.dataframe(pd.DataFrame([{
                                "Feature":k,"Current":round(v[0],3),
                                "Required":round(v[1],3),"Change":round(v[1]-v[0],3)}
                                for k,v in ch.items()]), use_container_width=True)

        with T["📋 Compliance"]:
            _cr = xai.get("compliance_report",{}); _mc = xai.get("model_card",{})
            st.markdown("### 📋 Compliance Report")
            if not _cr: st.info("Enable XAI and run simulation to generate compliance report.")
            else:
                summ = _cr.get("summary",{}); ok = summ.get("overall_compliant",False)
                st.markdown(
                    f'<div class="{"alert-success" if ok else "alert-danger"}">'
                    f'Overall: <strong>{"COMPLIANT ✅" if ok else "NON-COMPLIANT ❌"}</strong> | '
                    f'Fairness: {summ.get("fairness_score",0):.3f} | '
                    f'ECOA DI: {avg_di:.2f} ({"≥0.80 ✅" if avg_di>=0.80 else "<0.80 ❌"})</div>',
                    unsafe_allow_html=True)
                for fw,fd in _cr.get("frameworks",{}).items():
                    with st.expander(f"📑 {fw}"):
                        for ch,st_ in fd.get("checks",{}).items():
                            st.markdown(f"{'✅' if st_=='PASS' else '❌'} {ch}")
                try:
                    pdf_b = generate_pdf_compliance_report(_cr,_mc,
                        {"accuracy":avg_acc,"fairness_score":avg_fair},domain="generic")
                    st.download_button("📄 Download PDF Report",pdf_b,
                        "financial_compliance.pdf","application/pdf",use_container_width=True)
                except Exception as e:
                    st.caption(f"PDF unavailable: {e}")
            st.divider()
            st.markdown("#### 🇳🇬 Nigeria Regulatory Alignment")
            nigeria_compliance_panel(
                {"accuracy":avg_acc,"fairness_score":avg_fair,
                 "demographic_parity":df["demographic_parity"].mean()},
                domain="agrotech", has_ussd_fallback=False, has_gender_audit=True,
                has_multilingual=False, has_xai=enable_xai,
                has_governance=enable_governance, has_redteam=False)
            # ── Real-world benchmark comparison ─────────────────────────
            st.markdown("#### 📚 Real-World Benchmark Comparison")
            if BENCHMARKS_OK:
                _dom_bms = get_benchmarks_for_domain("finance")
                if _dom_bms:
                    _bm_sel = st.selectbox(
                        "Compare against published study:",
                        list(_dom_bms.keys()),
                        format_func=lambda k: _dom_bms[k].name + " (" + str(_dom_bms[k].year) + ")",
                        key="_fin_bm_sel")
                    _bm = _dom_bms[_bm_sel]
                    _sim_m = {"accuracy": avg_acc, "fairness_score": avg_fair}
                    if "fpr" in df.columns: _sim_m["fpr"] = df["fpr"].mean()
                    if "recall" in df.columns: _sim_m["recall"] = df["recall"].mean()
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
                    st.info("No benchmarks for this domain yet.")
            else:
                st.info("Add gags_benchmarks.py to components/ to enable.")


        with T["🔁 Longitudinal"]:
            st.markdown("### 🔁 Longitudinal Bias Analysis")
            st.markdown('<div class="nbox">Credit scoring bias compounds across retraining cycles — '
                        'biased decisions feed back into training data, deepening inequality.</div>',
                        unsafe_allow_html=True)
            if not lng: st.info("Run simulation to see longitudinal analysis.")
            else:
                c1,c2,c3 = st.columns(3)
                c1.metric("Initial Bias",f'{lng["initial_bias"]:.1%}')
                c2.metric("Final Bias",f'{lng["final_bias"]:.1%}',
                          f'{lng["final_bias"]-lng["initial_bias"]:+.1%}')
                c3.metric("Amplification",f'{lng["amplification_factor"]:.2f}×',
                          "⚠️ Self-reinforcing" if lng["self_reinforcing"] else "Stable")
                if lng.get("self_reinforcing"):
                    st.error(f'⚠️ Bias self-reinforcing at generation {lng.get("inflection_point","N/A")}.')
                st.markdown(f'<div class="nbox"><em>{lng["narrative"]}</em></div>',
                            unsafe_allow_html=True)
                gm = lng.get("generation_metrics",[])
                if gm:
                    fig_lng = px.line(pd.DataFrame(gm), x="generation",
                        y=["demographic_parity","fairness_score","accuracy"],
                        title="Credit Bias Evolution Across Retraining Cycles",
                        color_discrete_sequence=["#ef4444","#22c55e","#2563eb"],
                        labels={"value":"Score","generation":"Retraining Cycle"})
                    fig_lng.add_hline(y=0.1,line_dash="dot",line_color="#ef4444",
                                      annotation_text="ECOA parity threshold")
                    try: fig_lng.update_layout(**PT(), height=360)
                    except: fig_lng.update_layout(height=360)
                    st.plotly_chart(fig_lng, use_container_width=True)

        with T["🌐 Federated"]:
            st.markdown("### 🌐 Federated Learning")
            st.markdown('<div class="nbox">Tests whether credit scoring bias persists when models '
                        'train across banks/regions without centralising customer data.</div>',
                        unsafe_allow_html=True)
            if not fed: st.info("Run simulation to see federated analysis.")
            else:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Banks/Regions",fed["n_clients"])
                c2.metric("Global Accuracy",f'{fed["global_accuracy"]:.1%}')
                c3.metric("Global Fairness",f'{fed["global_fairness"]:.3f}')
                c4.metric("Bias Persisted","Yes ⚠️" if fed["bias_persisted"] else "No ✅")
                st.markdown(
                    f'<div class="{"alert-danger" if fed["bias_persisted"] else "alert-success"}">'
                    f'<em>{fed["narrative"]}</em></div>', unsafe_allow_html=True)

        with T["📋 Raw Results"]:
            sc = [c for c in df.columns if c != "narrative"]
            st.dataframe(df[sc].style
                .background_gradient(subset=["accuracy"],cmap="Blues")
                .background_gradient(subset=["fairness_score"],cmap="RdYlGn")
                .background_gradient(subset=["approval_rate_low"],cmap="Greens"),
                use_container_width=True)
            d1,d2 = st.columns(2)
            with d1:
                st.download_button(t("download_csv"),df[sc].to_csv(index=False).encode(),
                    f"gags_finance_{scenario_key}.csv","text/csv",use_container_width=True)
            with d2:
                st.download_button(t("export_config"),json.dumps({
                    "scenario_key":scenario_key,"country_income_level":country_income_level,
                    "institution_type":institution_type,"selected_biases":selected_biases,
                    "bias_intensity":bias_intensity,"regulatory_compliance":regulatory_compliance,
                    "avg_accuracy":f"{avg_acc:.3f}","avg_fairness":f"{avg_fair:.3f}",
                    "avg_di_ratio":f"{avg_di:.3f}"},indent=2),
                    f"finance_config_{scenario_key}.json","application/json",use_container_width=True)


        with T["🛡️ AI Safety"]:
            feats = st.session_state.get("fin_feature_outputs", {})
            feature_modules_tab(
                domain="financial",
                run_results=st.session_state.get("fin_run_history", []),
                feats=feats,
                governance=feats.get("governance"),
                gender_audit=feats.get("gender_audit"),
                agent_economy=feats.get("agent_economy"),
                arena=feats.get("strategic_arena"),
                redteam=feats.get("multimodal_redteam"),
            )

    # ── AI Safety Tab ────────────────────────────────────────────────────────
    with T["🔄 Lifecycle"]:
        render_safety_tab(
            st.session_state.get("fin_safety_report", {}),
            domain="financial",
        )

    # ── 🔄 Lifecycle Management Tab ───────────────────────────────────────────
    with T["🌱 Eco Score"]:
        render_lifecycle_tab(
            st.session_state.get("fin_lifecycle_report", {}),
            "financial",
        )

    # ── 🌱 Eco Score Tab ──────────────────────────────────────────────────────
    with T["🔬 Feature Modules"]:
        _lc_eco_r   = st.session_state.get("fin_lifecycle_report", {})
        _lc_eco_last = (st.session_state.get("fin_run_history") or [{}])[-1]
        render_eco_tab(
            _lc_eco_r,
            algo_key  = _lc_eco_last.get("algorithm", "hist_gradient_boosting"),
            n_samples = int(_lc_eco_last.get("n_samples", 2000)),
            n_runs    = int(locals().get("n_runs", 3)),
            domain    = "financial",
        )

    # ── 🔮 Dynamic Systems Tab ──────────────────────────────────────────────────
    with T["🔮 Dynamic Systems"]:
        try:
            render_dynamic_systems_tab(
                st.session_state.get("fin_ds_report", {}),
                domain="financial",
                ds_key="fin_ds_report",
            )
        except Exception as _ds_err:
            st.error(f"🔮 Dynamic Systems error: {_ds_err}")
            import traceback
            st.code(traceback.format_exc(), language="python")

    history_browser("fin_snapshot_history",domain="health",
        key_metrics=["accuracy","fairness_score","inclusion_score","disparate_impact_ratio"])
    annotation_panel("fin_annotations",
        context_label=f"{len(st.session_state.fin_run_history)} Financial run(s)")

    st.divider()
    st.markdown("## 💡 Financial Inclusion Policy Recommendations")
    r1,r2 = st.columns(2)
    with r1:
        st.info("""**🏦 For Financial Institutions**
1. Transparent algorithms: publish credit criteria
2. Human review for algorithmic denials
3. Quarterly ECOA/disparate impact audits
4. Alternative data: mobile money, utility payments
5. Explainable denials: clear reasons for every decision

**🇳🇬 Nigeria CBN/NDIC Alignment**
1. ECOA equivalent: DI ratio ≥ 0.80 required
2. Gender-disaggregated approval rates published annually
3. BVN-alternative pathways for informal income earners""")
    with r2:
        st.success("""**🤖 For AI Development**
1. Fairness constraints built into algorithm design
2. Proxy detection: remove discriminatory variables
3. Continuous monitoring across demographic groups
4. Adversarial testing against fraud and poisoning

**🏛️ For Regulators**
1. Strict ECOA / fair lending enforcement
2. Mandatory algorithmic fairness audits
3. Transparency: key factors in credit decisions disclosed
4. Fintech sandbox with consumer protection guardrails""")
    st.caption("⚠️ Disclaimer: Educational simulation. Real financial AI requires extensive testing.")

else:
    # ── Welcome state ──────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
  padding:2.5rem;text-align:center;margin-top:1rem">
  <p style="font-size:2rem;margin:0 0 .6rem">💰</p>
  <p style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:700;
     color:#0f172a;margin:0 0 .4rem">Welcome to Financial Inclusion Simulation</p>
  <p style="color:#64748b;font-size:.87rem;max-width:540px;margin:0 auto .5rem">
    Explore credit scoring bias, ECOA disparate impact, and informal economy exclusion.
    Six Nigeria-specific and global financial scenarios included.</p>
  <p style="color:#94a3b8;font-size:.78rem">
    Configure settings in the sidebar and click <strong>💰 Run</strong> to begin.
  </p>
</div>""", unsafe_allow_html=True)

st.divider()
st.markdown("<div style='text-align:center;color:#94a3b8;font-family:DM Mono,monospace;"
            "font-size:.7rem;padding:.5rem 0'>💰 Financial Inclusion · GAGS v4.0 · "
            "Nigeria CBN/NDIC · ECOA Disparate Impact Analysis</div>",
            unsafe_allow_html=True)
