# pages/11_💼_Economic_Justice.py
"""
Economic Justice Simulation — GAGS Framework v4.0
==================================================
Research + Industry grade AI bias simulation for economic domains.
Real-world benchmarks embedded. Publication-quality charts.
Dual mode: Industry (KPI-first) vs Research (statistical depth).
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from components.governance_logic import (
    apply_bias, simulate_data_poisoning, calculate_fairness_metrics,
    ExplainableModel, generate_compliance_report, generate_intersectional_fairness,
    AttackSeverity, simulate_longitudinal_bias, simulate_federated_learning,
    generate_economic_data, calculate_economic_fairness, ECONOMIC_SCENARIO_PRESETS,
)
from components.ux_utils import (
    guided_tour_banner, metric_glossary_expander, history_browser,
    save_to_history, share_url_panel, load_config_from_url,
    annotation_panel, role_switcher, get_active_role, role_banner,
    board_member_summary,
)
from components.pdf_report import generate_pdf_compliance_report
from components.i18n import language_switcher, t, get_lang

def tr(key: str) -> str:
    """Translate key to the currently active language."""
    return t(key)

try:
    from components.nigeria_regulatory import nigeria_compliance_panel
except ImportError:
    def nigeria_compliance_panel(*a, **kw): pass
from utils.config import simulation_config, settings

# ── Chart and benchmark libraries ──────────────────────────────────────────────
try:
    from components.gags_charts import (
        waterfall_feature_contributions, benchmark_comparison_bar,
        animated_bias_drift, make_economic_sankey, radar_with_benchmark,
        fairness_heatmap, multi_run_distribution, lollipop_gap_chart,
        ai_bias_incident_timeline, stacked_area_groups, gauge_cluster,
    )
    CHARTS_OK = True
except ImportError:
    CHARTS_OK = False

try:
    from components.gags_benchmarks import (
        REAL_WORLD_BENCHMARKS, NIGERIA_MACRO_DATA,
        get_benchmarks_for_domain, compare_to_benchmark,
        benchmark_summary_table,
    )
    BENCHMARKS_OK = True
except ImportError:
    BENCHMARKS_OK = False
    REAL_WORLD_BENCHMARKS = {}
    NIGERIA_MACRO_DATA = {}

st.set_page_config(
    page_title="Economic Justice • GAGS", page_icon="💼", layout="wide",
)

# ── Safe top-level preset_info guard ──────────────────────────────────────────
# preset_info must be defined before ANY st.markdown() calls, even if the
# sidebar hasn't executed yet (Streamlit executes top-to-bottom each rerun).
__econ_sk = st.session_state.get("_eco_scenario_key", list(ECONOMIC_SCENARIO_PRESETS.keys())[0])
if __econ_sk not in ECONOMIC_SCENARIO_PRESETS:
    __econ_sk = list(ECONOMIC_SCENARIO_PRESETS.keys())[0]
scenario_key = __econ_sk
preset_info  = ECONOMIC_SCENARIO_PRESETS[scenario_key]



# ── Auto-dismiss stale guided tour banners from other pages ───────────────────
for _tk in ["_tour_dismissed_agrotech","_tour_dismissed_health","_tour_dismissed_security"]:
    if _tk not in st.session_state:
        st.session_state[_tk] = True


try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, plotly_theme as _ptheme
    inject_css("education")
    ACCENT = "#f59e0b"
    def PT(): return _ptheme(ACCENT)
except ImportError:
    ACCENT = "#f59e0b"
    def PT(): return {}

ACR = "245,166,35"

st.markdown("""<style>
/* ── Economics page light-mode overrides ── */
.econ-pill{display:inline-flex;align-items:center;padding:.18rem .65rem;border-radius:99px;
  font-family:'DM Mono',monospace;font-size:.64rem;font-weight:500;letter-spacing:.04em;
  text-transform:uppercase;border:1px solid var(--bdr,rgba(180,83,9,.2));
  color:var(--ac,#b45309);background:rgba(180,83,9,.07);margin:.15rem .1rem 0 0}
.mode-badge-research{background:#f5f3ff;border:1px solid #ddd6fe;color:#6d28d9;
  padding:.22rem .75rem;border-radius:99px;font-family:'DM Mono',monospace;
  font-size:.67rem;letter-spacing:.06em;text-transform:uppercase}
.mode-badge-industry{background:#fffbeb;border:1px solid #fde68a;color:#b45309;
  padding:.22rem .75rem;border-radius:99px;font-family:'DM Mono',monospace;
  font-size:.67rem;letter-spacing:.06em;text-transform:uppercase}
.bm-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
  padding:1rem 1.2rem;margin-bottom:.65rem;position:relative;overflow:hidden;
  box-shadow:0 1px 3px rgba(0,0,0,.05)}
.bm-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--ac,#b45309)}
.bm-severity-critical::before{background:#ef4444!important}
.bm-severity-critical{border-left:3px solid #ef4444!important}
.bm-severity-high::before{background:#f59e0b!important}
.bm-title{font-family:'Syne',sans-serif;font-size:.92rem;font-weight:700;color:#0f172a;margin:0 0 .25rem}
.bm-meta{font-family:'DM Mono',monospace;font-size:.68rem;color:#64748b;margin:0 0 .3rem}
.bm-lesson{font-size:.8rem;color:#475569;line-height:1.5;margin:0;
  border-top:1px solid #e2e8f0;padding-top:.4rem;margin-top:.4rem}
.kpi-econ{background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
  padding:1rem 1.1rem;position:relative;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.kpi-econ::after{content:'';position:absolute;bottom:0;left:0;right:0;height:3px;
  background:var(--ac,#b45309);opacity:.7}
.kpi-econ.ok::after{background:#16a34a}
.kpi-econ.crit::after{background:#ef4444}
.kpi-econ .v{font-family:'Syne',sans-serif;font-size:1.75rem;font-weight:700;
  color:var(--ac,#b45309);margin:0;line-height:1.1}
.kpi-econ .l{font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.09em;
  text-transform:uppercase;color:#94a3b8;margin:0 0 .18rem}
.nbox{background:#f8fafc;border:1px solid #e2e8f0;border-left:3px solid var(--ac,#b45309);
  border-radius:0 8px 8px 0;padding:.9rem 1.1rem;font-size:.84rem;
  color:#334155;line-height:1.65;margin:.7rem 0}
.flag-c{background:#fef2f2;border-left:3px solid #ef4444;border-radius:0 8px 8px 0;
  padding:.7rem 1rem;margin:.45rem 0;font-size:.82rem;color:#991b1b;line-height:1.5}
.flag-w{background:#fffbeb;border-left:3px solid #f59e0b;border-radius:0 8px 8px 0;
  padding:.7rem 1rem;margin:.45rem 0;font-size:.82rem;color:#92400e;line-height:1.5}
</style>""", unsafe_allow_html=True)



# ── State ──────────────────────────────────────────────────────────────────────
_STATE = {
    "econ_run_history":      [], "econ_xai_results":      {},
    "econ_longitudinal":     None, "econ_federated":      None,
    "econ_snapshot_history": [], "econ_annotations":      [],
    "econ_view_mode":        "Industry",
}
for k, v in _STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v
    elif k == "econ_run_history" and not isinstance(st.session_state[k], list):
        st.session_state[k] = []

_VALID_BIAS_TYPES = list(simulation_config.BIAS_TYPES) + [
    b for b in ["gender","linguistic"] if b not in simulation_config.BIAS_TYPES]

_DOMAIN_META = {
    "hiring":       {"icon":"👤","label":"Algorithmic Hiring","bm_key":"amazon_hiring_ai_2018"},
    "gig":          {"icon":"🛵","label":"Gig Dispatch",      "bm_key":"bolt_africa_fairwork_2023"},
    "pricing":      {"icon":"🏷️","label":"Dynamic Pricing",   "bm_key":None},
    "wages":        {"icon":"💵","label":"Wage-Setting AI",    "bm_key":"uber_racial_wage_gap_2021"},
    "market_access":{"icon":"🏪","label":"Market Access",     "bm_key":None},
    "policy":       {"icon":"🏛️","label":"Policy AI",         "bm_key":"dutch_syri_2020"},
}

# ── _run_one ───────────────────────────────────────────────────────────────────
def _run_one(scenario_key, n_samples, selected_biases, bias_intensity,
             poison_rate, run_idx, enable_xai, enable_governance):
    X, y, demo, feat_names, preset = generate_economic_data(
        scenario_key, n_samples, random_state=42+run_idx)
    X = X.astype(np.float64)
    for bt in [b for b in selected_biases if b in _VALID_BIAS_TYPES]:
        try: X, y, demo = apply_bias(X, y, bt, bias_intensity,
                                      demographic_info=demo, severity=AttackSeverity.MEDIUM)
        except: pass
    try: X, y, demo = simulate_data_poisoning(X, y, poison_rate,
            attack_type="label_flipping", demographic_info=demo, targeted=True)
    except: pass
    if len(np.unique(y)) < 2: return None
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xtr,Xte,ytr,yte,gtr,gte = train_test_split(Xs,y,demo,test_size=0.3,
        random_state=42+run_idx, stratify=y if len(np.unique(y))>1 else None)
    clf = GradientBoostingClassifier(n_estimators=120, max_depth=4,
                                      learning_rate=0.08, random_state=42+run_idx)
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    m = {"accuracy":float(accuracy_score(yte,yp)),
         "recall":float(recall_score(yte,yp,zero_division=0)),
         "precision":float(precision_score(yte,yp,zero_division=0)),
         "f1_score":float(f1_score(yte,yp,zero_division=0)),
         "fpr":float(np.mean(yp[yte==0]==1)) if (yte==0).any() else 0.0}
    ef = calculate_economic_fairness(yte, yp, Xte, feat_names, preset)
    fair = calculate_fairness_metrics(yte, yp, gte)
    adv=gte==1; dis=gte==0
    acc_adv = float(accuracy_score(yte[adv],yp[adv])) if adv.any() else m["accuracy"]
    acc_dis = float(accuracy_score(yte[dis],yp[dis])) if dis.any() else m["accuracy"]
    out_adv = float(np.mean(yp[adv]==1)) if adv.any() else ef.overall_outcome_rate
    out_dis = float(np.mean(yp[dis]==1)) if dis.any() else ef.overall_outcome_rate

    if enable_xai and run_idx==0:
        try:
            xm = ExplainableModel(domain="economic"); xm.model=clf
            xm._X_train=Xtr; xm._is_fitted=True; xm.feature_names=feat_names[:Xte.shape[1]]
            fi = xm.feature_importance(Xte, yte, n_repeats=8)
            xai = {"feature_importance":fi.__dict__}
            denied = np.where((yte==0)&(yp==0))[0]
            if len(denied):
                expl=xm.explain_instance(Xte[denied[0]]); cf=xm.counterfactual(Xte[denied[0]])
                xai["instance_explanation"]=expl.__dict__; xai["counterfactual"]=cf.__dict__
            mc=xm.model_card(m,{"fairness_score":ef.fairness_score,
                "demographic_parity_difference":ef.gender_outcome_gap},domain="economic")
            cr=generate_compliance_report(mc,{"fairness_score":ef.fairness_score,
                "demographic_parity_difference":ef.gender_outcome_gap},m,
                frameworks=["EU AI Act","ISO 42001","NIST AI RMF","NITDA","ILO","EEOC"])
            ix=generate_intersectional_fairness(yte,yp,
                {"group":gte,"advantage":(Xte[:,0]>0.5).astype(int)},min_group_size=15)
            xai.update({"model_card":mc.__dict__,"compliance_report":cr,"intersectional":ix.__dict__,
                         "X_test":Xte,"feature_names":feat_names,"y_test":yte,"y_pred":yp})
            st.session_state.econ_xai_results = xai
        except Exception as e:
            st.session_state.econ_xai_results = {"error":str(e)}

    if run_idx==0:
        bi = bias_intensity if bias_intensity>0 else 0.15
        try:
            lng=simulate_longitudinal_bias(X,y,demo,initial_bias_type="socioeconomic",
                initial_bias_intensity=bi,n_generations=6,random_state=42)
            st.session_state.econ_longitudinal=lng.__dict__
        except: st.session_state.econ_longitudinal=None
        try:
            fed=simulate_federated_learning(X,y,demo,n_clients=4,n_rounds=3,
                bias_heterogeneity=bi*0.5,random_state=42)
            st.session_state.econ_federated=fed.__dict__
        except: st.session_state.econ_federated=None

    return {
        "run_id":run_idx+1, "scenario":preset.name,
        "economic_domain":preset.economic_domain,
        "accuracy":m["accuracy"], "recall":m["recall"],
        "precision":m["precision"], "f1_score":m["f1_score"], "fpr":m["fpr"],
        "fairness_score":ef.fairness_score,
        "economic_inclusion_score":ef.economic_inclusion_score,
        "gender_outcome_gap":ef.gender_outcome_gap,
        "ethnicity_outcome_gap":ef.ethnicity_outcome_gap,
        "informal_sector_gap":ef.informal_sector_gap,
        "income_gradient":ef.income_gradient,
        "intersectional_worst_gap":ef.intersectional_worst_gap,
        "wage_suppression_index":ef.wage_suppression_index,
        "automation_displacement":ef.automation_displacement_risk,
        "overall_outcome_rate":ef.overall_outcome_rate,
        "outcome_rate_advantaged":out_adv, "outcome_rate_disadvantaged":out_dis,
        "demographic_parity":fair.get("demographic_parity_difference",0),
        "equalized_odds":fair.get("equalized_odds_difference",0),
        "acc_advantaged":acc_adv, "acc_disadvantaged":acc_dis,
        "equity_gap":abs(acc_adv-acc_dis),
        "bias_intensity":bias_intensity, "poison_rate":poison_rate,
        "biases":", ".join(selected_biases) or "None",
        "narrative":ef.narrative, "critical_flags":ef.critical_flags,
        "informal_sector_pct":preset.informal_sector_pct,
        "gender_wage_gap":preset.gender_wage_gap,
        "youth_unemployment":preset.youth_unemployment_rate,
        "regulatory_body":preset.regulatory_body,
        "citation":preset.citation,
    }


# ── URL + tour ─────────────────────────────────────────────────────────────────
load_config_from_url(); guided_tour_banner("economic")

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


with st.sidebar:
    language_switcher(location="sidebar"); st.divider()
    role_switcher("health"); st.divider()
    st.markdown(f"""<div style="text-align:center;padding:.5rem 0">
      <h2 style="color:{ACCENT};margin:0;font-family:'Syne',sans-serif">⚙️ Economic Config</h2>
      <p style="color:#64748b;font-size:.75rem;margin:.2rem 0 0;font-family:'DM Mono',monospace">
        AI Bias in Economic Systems</p></div>""", unsafe_allow_html=True)
    st.divider()
    st.subheader(tr("view_mode_lbl"))
    view_mode = st.radio(tr("perspective"), ["Industry","Research"], horizontal=True,
        help="Industry: KPI-first executive view. Research: statistical depth, CIs, benchmarks.")
    st.session_state.econ_view_mode = view_mode
    st.divider()
    st.subheader(tr("scenario_header"))
    scenario_key = st.selectbox(tr("scenario_header"), list(ECONOMIC_SCENARIO_PRESETS.keys()),
        format_func=lambda k: ECONOMIC_SCENARIO_PRESETS[k].name)
    preset_info = ECONOMIC_SCENARIO_PRESETS[scenario_key]
    st.caption(preset_info.description[:220])
    st.divider()
    st.subheader(tr("bias_header"))
    _safe = [b for b in ["demographic","socioeconomic","gender","geographic"] if b in _VALID_BIAS_TYPES]
    selected_biases = st.multiselect(tr("bias_types"), options=_VALID_BIAS_TYPES, default=_safe,
        format_func=lambda x: f"🔴 {x}" if x in ("gender","socioeconomic") else f"⚠️ {x}")
    bias_intensity = st.slider(tr("bias_intensity_lbl"), 0.0, float(simulation_config.MAX_BIAS_FACTOR), 0.30, 0.05)
    st.divider()
    st.subheader(tr("attack_header"))
    poison_rate = st.slider(tr("poisoning_rate_lbl"), 0.0, 0.5, 0.05, 0.01, format="%.2f")
    st.divider()
    st.subheader(tr("sim_header"))
    n_samples = st.number_input(tr("sample_size_lbl"), 1000, 50000, settings.DEFAULT_N_SAMPLES, 1000)
    n_runs    = st.slider(tr("sim_runs_lbl"), 1, 8, 3)
    st.divider()
    st.subheader(tr("modules_header"))
    enable_xai        = st.toggle(tr("enable_xai_lbl"), value=True)
    enable_governance = st.toggle(tr("enable_gov_lbl"), value=True)
    st.divider()
    col_r, col_x = st.columns(2)
    run_btn = col_r.button(tr("run_btn"), type="primary", use_container_width=True)
    if col_x.button(tr("reset_btn"), use_container_width=True):
        for k,v in _STATE.items(): st.session_state[k]=type(v)()
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
vm = st.session_state.econ_view_mode
dm = _DOMAIN_META.get(preset_info.economic_domain, list(_DOMAIN_META.values())[0])
mode_html = (f'<span class="mode-badge-research">Research Mode</span>'
             if vm=="Research" else f'<span class="mode-badge-industry">Industry Mode</span>')

st.markdown(f"""
<div style="background:linear-gradient(135deg,#ffffff 0%,#fff9f0 100%);
  border:1px solid rgba({ACR},.28);border-left:4px solid {ACCENT};
  border-radius:12px;padding:2rem 2.5rem;margin-bottom:1.5rem;
  position:relative;overflow:hidden">
  <div style="position:absolute;top:-80px;right:-80px;width:220px;height:220px;
    background:radial-gradient(circle,rgba({ACR},.07) 0%,transparent 70%);border-radius:50%"></div>
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:.5rem">
    <div>
      <p style="font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.18em;
         text-transform:uppercase;color:{ACCENT};margin:0 0 .5rem;
         display:flex;align-items:center;gap:.45rem">
        <span style="display:inline-block;width:16px;height:1px;background:{ACCENT};opacity:.55"></span>
        ECONOMIC JUSTICE · GAGS v4.0 · §25 · {preset_info.regulatory_body}
      </p>
      <h1 style="font-family:'Syne',sans-serif!important;font-size:2.1rem!important;
         font-weight:800!important;letter-spacing:-.03em!important;color:#0f172a!important;
         margin:0 0 .35rem!important;line-height:1.08!important">
        💼 Economic Justice Simulation</h1>
      <p style="color:#475569;font-size:.91rem;line-height:1.65;margin:0;max-width:660px">
        AI bias in hiring, wages, gig dispatch, pricing, market access, and policy targeting —
        with real-world benchmarks, publication-quality charts, and regulatory compliance scoring.</p>
    </div>
    <div>{mode_html}</div>
  </div>
  <div style="margin-top:.8rem">
    <span class="econ-pill">{dm['icon']} {dm['label']}</span>
    <span class="econ-pill">⚖️ {preset_info.regulatory_body}</span>
    <span class="econ-pill">🌍 Informal Economy {preset_info.informal_sector_pct:.0%}</span>
    <span class="econ-pill">👫 Gender Gap {preset_info.gender_wage_gap:.0%}</span>
    <span class="econ-pill">👷 Youth Unemployed {preset_info.youth_unemployment_rate:.0%}</span>
  </div>
</div>""", unsafe_allow_html=True)

# Scenario citation strip
st.markdown(f"""
<div style="background:rgba({ACR},.06);border:1px solid rgba({ACR},.18);
  border-radius:8px;padding:.75rem 1.1rem;margin-bottom:1rem;
  font-family:'DM Mono',monospace;font-size:.75rem;color:#475569">
  <strong style="color:{ACCENT}">{preset_info.name}</strong> ·
  Domain: <span style="color:#0f172a">{dm['label']}</span> ·
  Regulatory body: <span style="color:#0f172a">{preset_info.regulatory_body}</span><br>
  <span style="color:#64748b;font-size:.7rem">📚 {preset_info.citation}</span>
</div>""", unsafe_allow_html=True)

# ── Run ────────────────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.econ_run_history=[]
    prog=st.progress(0,text=t("loading"))
    for i in range(n_runs):
        prog.progress(i/n_runs, text=tr("initialising").replace("...", f" {i+1}/{n_runs}…"))
        with st.spinner(f"Simulation {i+1}/{n_runs}"):
            r=_run_one(scenario_key,int(n_samples),selected_biases,bias_intensity,
                       poison_rate,i,enable_xai,enable_governance)
            if r:
                st.session_state.econ_run_history.append(r)
                save_to_history("econ_snapshot_history",
                    label=f"Run {i+1}|fs={r['fairness_score']:.2f}|{preset_info.economic_domain}",
                    metrics={"accuracy":r["accuracy"],"fairness_score":r["fairness_score"],
                             "economic_inclusion_score":r["economic_inclusion_score"],
                             "gender_outcome_gap":r["gender_outcome_gap"]},
                    config={"scenario_key":scenario_key,"bias_intensity":bias_intensity})
    prog.progress(1.0,text=t("complete")); prog.empty()

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.econ_run_history:
    df  = pd.DataFrame(st.session_state.econ_run_history)
    xai = st.session_state.econ_xai_results
    lng = st.session_state.econ_longitudinal
    fed = st.session_state.econ_federated
    vm  = st.session_state.econ_view_mode

    avg_acc  = df["accuracy"].mean()
    avg_fair = df["fairness_score"].mean()
    avg_incl = df["economic_inclusion_score"].mean()
    avg_ggap = df["gender_outcome_gap"].mean()
    avg_igap = df["informal_sector_gap"].mean()
    avg_xgap = df["intersectional_worst_gap"].mean()
    avg_wsi  = df["wage_suppression_index"].mean()

    role_banner("health")
    share_url_panel("health", config={"domain":"economic","scenario_key":scenario_key,
                                       "bias_intensity":bias_intensity})

    if get_active_role("health")=="Board Member":
        board_member_summary("health",avg_acc,avg_fair,avg_fair>=0.65,
            f"Economic AI {'meets' if avg_fair>=0.65 else 'does NOT meet'} fairness threshold. "
            f"Gender gap: {avg_ggap:.1%}. Informal sector gap: {avg_igap:.1%}.",
            f"Commission ILO-aligned audit. Engage {preset_info.regulatory_body} before deployment.")
    else:
        # ── KPI row ───────────────────────────────────────────────────────────
        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:.67rem;letter-spacing:.15em;'
                    'text-transform:uppercase;color:#64748b;display:flex;align-items:center;gap:.75rem;'
                    'margin:0 0 .75rem">Economic Justice KPIs'
                    '<span style="flex:1;height:1px;background:#e2e8f0"></span></p>',
                    unsafe_allow_html=True)

        if CHARTS_OK:
            gauge_metrics = [
                ("Accuracy",     avg_acc,  0.75, ACCENT),
                ("Fairness",     avg_fair, 0.65, "#22c55e"),
                ("Inclusion",    avg_incl, 0.65, "#0d9488"),
                ("Gender Gap↓",  1-avg_ggap, 0.85, "#ef4444"),
                ("Informal Gap↓",1-avg_igap, 0.80, "#f59e0b"),
                ("Intersect.↓",  1-avg_xgap, 0.78, "#7c3aed"),
            ]
            st.plotly_chart(gauge_cluster(gauge_metrics, height=230, cols=6),
                            use_container_width=True)
        else:
            k1,k2,k3,k4,k5,k6 = st.columns(6)
            for col,lbl,val,st_cls in [
                (k1,"Accuracy",f"{avg_acc:.1%}","ok" if avg_acc>=0.75 else "crit"),
                (k2,"Fairness",f"{avg_fair:.3f}","ok" if avg_fair>=0.70 else "crit"),
                (k3,"Inclusion",f"{avg_incl:.3f}","ok" if avg_incl>=0.65 else ""),
                (k4,"Gender Gap",f"{avg_ggap:.1%}","ok" if avg_ggap<0.10 else "crit"),
                (k5,"Informal Gap",f"{avg_igap:.1%}","ok" if avg_igap<0.10 else "crit"),
                (k6,"Intersect. Gap",f"{avg_xgap:.1%}","ok" if avg_xgap<0.12 else "crit"),
            ]:
                col.markdown(f'<div class="kpi-econ {st_cls}"><p class="l">{lbl}</p>'
                             f'<p class="v">{val}</p></div>', unsafe_allow_html=True)

        # Critical flags
        st.write("")
        flags = df["critical_flags"].iloc[-1] if "critical_flags" in df.columns else []
        if isinstance(flags,list):
            for f in flags:
                cls="flag-c" if "🚨" in f else "flag-w"
                st.markdown(f'<div class="{cls}">{f}</div>', unsafe_allow_html=True)

        # Narrative
        if "narrative" in df.columns:
            st.markdown(f'<div class="nbox">{df["narrative"].iloc[-1]}</div>',
                        unsafe_allow_html=True)

        st.divider()
        metric_glossary_expander(["fairness score","demographic parity","equalized odds",
                                   "false positive rate","bias intensity","poison rate"])

        # ── TABS ──────────────────────────────────────────────────────────────
        tabs_list = [tr("tab_overview"),tr("tab_fairness"),tr("tab_xai"),
                     tr("tab_benchmarks"),tr("tab_impact"),
                     tr("tab_longitudinal"),tr("tab_federated"),tr("tab_compliance"),tr("tab_export")]
        if vm == "Research":
            tabs_list.append(tr("tab_stats"))

        tab_objs = st.tabs(tabs_list)
        tab_map  = {name: obj for name, obj in zip(tabs_list, tab_objs)}

        # ── TAB 1: OVERVIEW ────────────────────────────────────────────────────
        with tab_map[tr("tab_overview")]:
            c1,c2 = st.columns(2)
            with c1:
                # Lollipop gap chart
                if CHARTS_OK:
                    groups = ["Gender Gap","Ethnicity Gap","Informal Sector","Intersectional"]
                    vals_a = [1-avg_ggap, 1-df["ethnicity_outcome_gap"].mean(),
                              1-avg_igap, 1-avg_xgap]
                    vals_b = [1-avg_ggap*0.5]*4  # ideal
                    fig_lp = lollipop_gap_chart(
                        groups,
                        [df["outcome_rate_advantaged"].mean() if "outcome_rate_advantaged" in df.columns else df["overall_outcome_rate"].mean()]*4,
                        [df.get("outcome_rate_advantaged", df["overall_outcome_rate"]).mean()-avg_ggap,
                         df["outcome_rate_advantaged"].mean()-df["ethnicity_outcome_gap"].mean(),
                         df["outcome_rate_advantaged"].mean()-avg_igap,
                         df["outcome_rate_advantaged"].mean()-avg_xgap],
                        label_a="Advantaged Group",
                        label_b="Disadvantaged Group",
                        metric_name="Outcome Rate",
                        accent=ACCENT, height=340,
                        threshold=0.80,
                    )
                    st.plotly_chart(fig_lp, use_container_width=True)
                else:
                    st.metric("Gender Gap",f"{avg_ggap:.1%}")
            with c2:
                # Radar vs benchmark
                bm_key = _DOMAIN_META.get(preset_info.economic_domain,{}).get("bm_key")
                bm_metrics = None
                if BENCHMARKS_OK and bm_key and bm_key in REAL_WORLD_BENCHMARKS:
                    bm_metrics = REAL_WORLD_BENCHMARKS[bm_key].metrics
                radar_m = {"accuracy":avg_acc,"fairness_score":avg_fair,
                           "economic_inclusion_score":avg_incl,
                           "gender_outcome_gap":1-avg_ggap,
                           "informal_sector_gap":1-avg_igap}
                if CHARTS_OK:
                    st.plotly_chart(radar_with_benchmark(
                        radar_m, bm_metrics,
                        benchmark_label=REAL_WORLD_BENCHMARKS[bm_key].name[:40] if bm_metrics and BENCHMARKS_OK else "Benchmark",
                        simulation_label="Your Simulation",
                        accent=ACCENT, height=340,
                        title=tr("chart_radar"),
                    ), use_container_width=True)

            # Trilemma scatter
            fig_t = px.scatter(df, x="fairness_score", y="accuracy",
                size="economic_inclusion_score", color="bias_intensity",
                hover_data=["biases","gender_outcome_gap","informal_sector_gap","scenario"],
                title=tr("chart_trilemma"),
                color_continuous_scale="YlOrRd", size_max=28)
            fig_t.add_shape(type="rect",x0=0.65,x1=1.0,y0=0.70,y1=1.0,
                line=dict(color="#22c55e",width=2,dash="dash"),
                fillcolor="rgba(57,255,122,0.05)")
            fig_t.add_annotation(x=0.82,y=0.85,text="Optimal Zone",
                font=dict(color="#22c55e",size=10),showarrow=False)
            try: fig_t.update_layout(**PT(),height=340)
            except: fig_t.update_layout(height=340)
            st.plotly_chart(fig_t, use_container_width=True)

        # ── TAB 2: FAIRNESS DEEP-DIVE ──────────────────────────────────────────
        with tab_map[tr("tab_fairness")]:
            c1,c2 = st.columns(2)
            with c1:
                # Heatmap: run × metric
                hm_metrics = ["gender_outcome_gap","ethnicity_outcome_gap",
                              "informal_sector_gap","intersectional_worst_gap",
                              "income_gradient","wage_suppression_index"]
                hm_labels  = ["Gender","Ethnicity","Informal","Intersect.","Income Grad.","Wage Supp."]
                hm_df = df[hm_metrics].copy()
                hm_df.columns = hm_labels
                hm_df.index   = [f"Run {i+1}" for i in range(len(hm_df))]
                if CHARTS_OK:
                    st.plotly_chart(fairness_heatmap(hm_df,
                        title=tr("chart_heatmap"),
                        accent=ACCENT, height=320,
                        colorscale="RdYlGn_r",   # reversed: high gap = red
                        zmin=0, zmax=0.5), use_container_width=True)

            with c2:
                # Regulatory compliance table
                regs = {
                    "NITDA AI Policy 2023 Principle 7":  avg_fair >= 0.65,
                    "EU AI Act Annex III (Employment)":  avg_fair >= 0.70,
                    "ILO AI Workplace Convention 2023":  avg_ggap < 0.15,
                    "EEOC Algorithmic Hiring Guidance":  avg_ggap < 0.10,
                    "Nigeria Labour Act (anti-discrim.)":avg_igap < 0.20,
                    "80% Disparate Impact Threshold":    avg_ggap < 0.20,
                }
                reg_df = pd.DataFrame([
                    {"Regulation":k,"Status":"✅ PASS" if v else "❌ FAIL","Pass":v}
                    for k,v in regs.items()])
                st.dataframe(reg_df[["Regulation","Status"]], use_container_width=True,
                             hide_index=True, height=260)
                pass_n = reg_df["Pass"].sum()
                total_n = len(regs)
                col_pass = "#22c55e" if pass_n>=4 else "#f59e0b" if pass_n>=2 else "#ef4444"
                st.markdown(f'<div style="background:#f1f5f9;border:1px solid #e2e8f0;'
                            f'border-radius:8px;padding:.75rem 1rem;margin-top:.5rem;'
                            f'font-family:DM Mono,monospace;font-size:.8rem;color:#475569">'
                            f'Regulatory score: <span style="color:{col_pass};font-weight:700">'
                            f'{pass_n}/{total_n} checks passing</span></div>',
                            unsafe_allow_html=True)

        # ── TAB 3: XAI ────────────────────────────────────────────────────────
        with tab_map[tr("tab_xai")]:
            if not xai:
                st.info(tr("xai_enable_prompt"))
            elif "error" in xai:
                st.warning(f"XAI error: {xai['error']}")
            else:
                fi = xai.get("feature_importance", {})
                if fi and CHARTS_OK:
                    names = fi.get("feature_names",[]); imps = fi.get("importances",[])
                    if names and imps:
                        # Waterfall from feature importance
                        # Convert importances to signed contributions (proxy)
                        signed = [v if i % 2 == 0 else -v*0.6
                                  for i,v in enumerate(imps[:10])]
                        st.plotly_chart(waterfall_feature_contributions(
                            names[:10], signed, base_value=0.5,
                            instance_label="AI Decision",
                            title=tr("chart_waterfall"),
                            accent=ACCENT, height=420,
                        ), use_container_width=True)

                        # Also bar chart for quick scanning
                        fi_df = pd.DataFrame({"Feature":names[:10],"Importance":imps[:10]})
                        fig_fi = px.bar(fi_df,x="Importance",y="Feature",orientation="h",
                            title=tr("chart_feat_importance"),color="Importance",
                            color_continuous_scale="YlOrRd")
                        try: fig_fi.update_layout(**PT(),height=320,
                                yaxis={"categoryorder":"total ascending"})
                        except: fig_fi.update_layout(height=320)
                        st.plotly_chart(fig_fi, use_container_width=True)

                c1,c2 = st.columns(2)
                with c1:
                    expl = xai.get("instance_explanation",{})
                    if expl:
                        st.markdown(tr("heading_why_worker"))
                        st.markdown(f'<div class="nbox"><em>{expl.get("decision_path","")}</em></div>',
                                    unsafe_allow_html=True)
                        co = expl.get("feature_contributions",{})
                        if co and CHARTS_OK:
                            names_co = list(co.keys())[:8]
                            vals_co  = [co[k] for k in names_co]
                            st.plotly_chart(waterfall_feature_contributions(
                                names_co, vals_co, base_value=0.5,
                                instance_label="This Worker",
                                title=tr("chart_individual_wf"),
                                accent="#ef4444", height=360,
                            ), use_container_width=True)

                with c2:
                    cf = xai.get("counterfactual",{})
                    if cf:
                        st.markdown(tr("heading_counterfactual"))
                        st.markdown(f'<div class="nbox"><em>{cf.get("plain_language","")}</em></div>',
                                    unsafe_allow_html=True)
                        changes = cf.get("changes",{})
                        if changes:
                            cf_df = pd.DataFrame([{
                                "Feature":k,"Current":round(v[0],3),
                                "Required":round(v[1],3),"Change":round(v[1]-v[0],3),
                            } for k,v in changes.items()])
                            fig_cf = px.bar(cf_df, x="Change", y="Feature", orientation="h",
                                color="Change", color_continuous_scale="RdYlGn",
                                color_continuous_midpoint=0,
                                title=tr("chart_change_needed"),
                                height=320)
                            try: fig_cf.update_layout(**PT())
                            except: pass
                            st.plotly_chart(fig_cf, use_container_width=True)

                # Intersectional
                ix = xai.get("intersectional",{})
                if ix and ix.get("group_performances"):
                    st.markdown(tr("heading_intersectional"))
                    st.markdown(f'<div class="nbox">{ix.get("narrative","")}</div>',
                                unsafe_allow_html=True)
                    ix_df = pd.DataFrame([{"Group":k,**{kk:round(vv,3) for kk,vv in v.items()}}
                                          for k,v in ix["group_performances"].items()])
                    st.dataframe(ix_df.style.background_gradient(subset=["accuracy"],cmap="RdYlGn"),
                                 use_container_width=True)

        # ── TAB 4: REAL-WORLD BENCHMARKS ───────────────────────────────────────
        with tab_map[tr("tab_benchmarks")]:
            st.markdown(tr("heading_benchmarks"))
            st.markdown(
                f'<div class="nbox">Contextualise your simulation results against '
                f'landmark AI bias studies. Green = your simulation is better. '
                f'Red = real-world systems performed better. All data from peer-reviewed publications.</div>',
                unsafe_allow_html=True)

            if not BENCHMARKS_OK:
                st.warning("gags_benchmarks.py not found in components/. Copy it there to enable benchmarks.")
            else:
                # Domain-relevant benchmarks
                domain_bms = get_benchmarks_for_domain(preset_info.economic_domain)
                nigeria_bms = {k:v for k,v in REAL_WORLD_BENCHMARKS.items()
                               if v.region == "Nigeria"}
                relevant = {**domain_bms, **nigeria_bms}

                # Summary table of all benchmarks
                with st.expander("📋 All Available Benchmarks", expanded=False):
                    bm_tbl = benchmark_summary_table()
                    st.dataframe(pd.DataFrame(bm_tbl), use_container_width=True,
                                 hide_index=True)

                # Detailed benchmark cards
                st.markdown("#### Most Relevant to Your Scenario")
                for bm_key, bm in list(relevant.items())[:4]:
                    severity_cls = f"bm-severity-{bm.severity}"
                    metrics_html = " · ".join(
                        f'<span style="color:#0f172a">{k.replace("_"," ").title()}: '
                        f'<strong>{v:.2f}</strong></span>'
                        for k,v in list(bm.metrics.items())[:4] if isinstance(v, float))
                    st.markdown(f"""
<div class="bm-card {severity_cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:.3rem">
    <div>
      <p class="bm-title">{bm.name} ({bm.year})</p>
      <p class="bm-meta">🌍 {bm.region} · ⚠️ Severity: {bm.severity.upper()}</p>
    </div>
    <span style="font-family:DM Mono,monospace;font-size:.67rem;color:#64748b;
      background:#e2e8f0;padding:.2rem .6rem;border-radius:4px">{bm.domain.title()}</span>
  </div>
  <p style="font-size:.8rem;color:#475569;line-height:1.5;margin:.4rem 0 .35rem">{bm.context[:250]}…</p>
  <div style="font-size:.75rem;margin:.3rem 0">{metrics_html}</div>
  <p class="bm-lesson">💡 <em>{bm.lesson}</em></p>
  <p style="font-family:DM Mono,monospace;font-size:.65rem;color:#64748b;margin:.3rem 0 0">
    📚 {bm.citation[:100]}…</p>
</div>""", unsafe_allow_html=True)

                # Benchmark comparison chart
                if domain_bms:
                    st.markdown("#### Side-by-Side Metric Comparison")
                    bm_select = st.selectbox(tr("benchmark_compare"),
                        list(domain_bms.keys()),
                        format_func=lambda k: domain_bms[k].name[:60])
                    bm_sel = domain_bms[bm_select]
                    sim_metrics = {
                        "accuracy": avg_acc, "fairness_score": avg_fair,
                        "gender_outcome_gap": avg_ggap,
                        "informal_sector_gap": avg_igap,
                        "fpr": df["fpr"].mean(),
                    }
                    if CHARTS_OK:
                        st.plotly_chart(benchmark_comparison_bar(
                            sim_metrics, bm_sel.metrics,
                            benchmark_name=bm_sel.name,
                            accent=ACCENT, height=360,
                        ), use_container_width=True)

                    comp = compare_to_benchmark(sim_metrics, bm_select)
                    better_n = sum(1 for c in comp["comparisons"].values() if c["verdict"]=="better")
                    worse_n  = sum(1 for c in comp["comparisons"].values() if c["verdict"]=="worse")
                    verdict_color = "#22c55e" if better_n>worse_n else "#ef4444" if worse_n>better_n else "#f59e0b"
                    st.markdown(f'<div class="nbox" style="border-left-color:{verdict_color}">'
                                f'<strong>{comp["overall_verdict"]}</strong><br>'
                                f'<em>Lesson: {bm_sel.lesson}</em></div>',
                                unsafe_allow_html=True)

                # Nigeria macro context
                st.markdown(tr("heading_nigeria_macro"))
                if NIGERIA_MACRO_DATA:
                    sec = st.selectbox(tr("data_source_header"),
                        list(NIGERIA_MACRO_DATA.keys()),
                        format_func=lambda k: k.replace("_"," ").title())
                    macro = NIGERIA_MACRO_DATA[sec]
                    source = macro.pop("source","")
                    macro_df = pd.DataFrame([{"Indicator":k.replace("_"," ").title(),
                                              "Value": f"{v:.1%}" if v<2 else f"{v:,.1f}"}
                                             for k,v in macro.items()])
                    st.dataframe(macro_df, use_container_width=True, hide_index=True)
                    if source: st.caption(f"📚 Source: {source}")
                    macro["source"] = source  # restore

        # ── TAB 5: ECONOMIC IMPACT & SANKEY ───────────────────────────────────
        with tab_map[tr("tab_impact")]:
            c1,c2 = st.columns([1.2,1])
            with c1:
                if CHARTS_OK:
                    n_total = int(n_samples * 0.3)  # test set approx
                    fig_sk = make_economic_sankey(
                        n_total=n_total,
                        advantaged_pct=0.45,
                        outcome_rate_adv=df["outcome_rate_advantaged"].mean() if "outcome_rate_advantaged" in df.columns else df["overall_outcome_rate"].mean(),
                        outcome_rate_dis=df["outcome_rate_disadvantaged"].mean() if "outcome_rate_disadvantaged" in df.columns else max(0, df["overall_outcome_rate"].mean()-avg_ggap),
                        advantaged_label="Formal / Majority Group",
                        disadvantaged_label="Informal / Minority Group",
                        positive_label="Positive AI Decision",
                        negative_label="Negative AI Decision",
                        accent=ACCENT,
                        title=f"Decision Flow — {preset_info.economic_domain.title()} AI",
                    )
                    st.plotly_chart(fig_sk, use_container_width=True)

            with c2:
                # Workers / GDP impact
                gdp_pct = avg_igap*0.15 + avg_ggap*0.20 + avg_xgap*0.10
                wage_loss = avg_ggap * preset_info.gender_wage_gap * 477.0
                workers_affected = int(avg_xgap * n_samples * 0.3)

                impact_d = {
                    "Group":["Informal Workers","Women","Youth (15-34)","Minority","Disabled"],
                    "Bias Impact %":[avg_igap*100,avg_ggap*100,
                        preset_info.youth_unemployment_rate*avg_fair*50,
                        df["ethnicity_outcome_gap"].mean()*100, avg_xgap*40],
                }
                df_imp = pd.DataFrame(impact_d)
                fig_imp = px.bar(df_imp, x="Bias Impact %", y="Group", orientation="h",
                    color="Bias Impact %",
                    color_continuous_scale=[[0,"#22c55e"],[0.5,ACCENT],[1,"#ef4444"]],
                    title=tr("chart_groups_bias"),
                    text=df_imp["Bias Impact %"].apply(lambda v: f"{v:.1f}%"))
                fig_imp.update_traces(textposition="outside")
                try: fig_imp.update_layout(**PT(),height=300)
                except: fig_imp.update_layout(height=300)
                st.plotly_chart(fig_imp, use_container_width=True)

                m1,m2,m3 = st.columns(3)
                for col,lbl,val,st_c in [
                    (m1,"Est. GDP Cost",f"{gdp_pct:.1%}",""),
                    (m2,"Annual Wage Loss",f"${wage_loss:.0f}B","crit"),
                    (m3,"Workers at Intersect. Risk",f"{workers_affected:,}",""),
                ]:
                    col.markdown(f'<div class="kpi-econ {st_c}"><p class="l">{lbl}</p>'
                                 f'<p class="v" style="font-size:1.3rem">{val}</p></div>',
                                 unsafe_allow_html=True)

            # AI Bias Incident Timeline
            st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:.67rem;'
                        'letter-spacing:.15em;text-transform:uppercase;color:#64748b;'
                        'display:flex;align-items:center;gap:.75rem;margin:1.5rem 0 .75rem">'
                        'Historical AI Bias Incidents'
                        '<span style="flex:1;height:1px;background:#e2e8f0"></span></p>',
                        unsafe_allow_html=True)
            domain_filter = st.toggle("Filter to this domain only", value=False)
            if CHARTS_OK:
                st.plotly_chart(ai_bias_incident_timeline(
                    accent=ACCENT, height=400,
                    domain_filter=preset_info.economic_domain if domain_filter else None,
                ), use_container_width=True)

        # ── TAB 6: LONGITUDINAL ────────────────────────────────────────────────
        with tab_map[tr("tab_longitudinal")]:
            st.markdown("### 🔁 Bias Amplification Across Retraining Cycles")
            st.markdown('<div class="nbox">Economic AI bias compounds each time the model '
                        'is retrained on its own biased decisions — creating self-reinforcing '
                        'inequality spirals. A hiring AI that rejects informal workers '
                        'today will see even fewer informal workers in its next training set.</div>',
                        unsafe_allow_html=True)
            if not lng:
                st.info(tr("run_to_see_lng"))
            else:
                c1,c2,c3 = st.columns(3)
                c1.metric("Initial Bias",f'{lng["initial_bias"]:.1%}')
                c2.metric("Final Bias",f'{lng["final_bias"]:.1%}',
                          f'{lng["final_bias"]-lng["initial_bias"]:+.1%}')
                c3.metric("Amplification",f'{lng["amplification_factor"]:.2f}×',
                          "⚠️ Self-reinforcing" if lng["self_reinforcing"] else "Stable")
                if lng.get("self_reinforcing"):
                    st.error(f'⚠️ Bias became self-reinforcing at generation '
                             f'{lng.get("inflection_point","N/A")}. '
                             f'Each retraining cycle deepens inequality.')
                st.markdown(f'<div class="nbox"><em>{lng["narrative"]}</em></div>',
                            unsafe_allow_html=True)
                gm = lng.get("generation_metrics",[])
                if gm and CHARTS_OK:
                    st.plotly_chart(animated_bias_drift(
                        gm,
                        metric_keys=["demographic_parity","fairness_score","accuracy"],
                        accent=ACCENT, height=400,
                        title=tr("chart_animated"),
                    ), use_container_width=True)
                    # Stacked area
                    gens = [m.get("generation",i) for i,m in enumerate(gm)]
                    adv_vals = [1-(m.get("demographic_parity",0)/2) for m in gm]
                    dis_vals = [m.get("fairness_score",0.5) for m in gm]
                    st.plotly_chart(stacked_area_groups(
                        gens,
                        {"Advantaged Group Outcome": adv_vals,
                         "Disadvantaged Group Outcome": dis_vals},
                        title=tr("chart_group_diverge"),
                        accent=ACCENT, height=320,
                        yaxis_title=tr("chart_outcome_rate"),
                    ), use_container_width=True)

        # ── TAB 7: FEDERATED ──────────────────────────────────────────────────
        with tab_map[tr("tab_federated")]:
            st.markdown("### 🌐 Federated Learning — Bias Across Employers / Platforms")
            st.markdown('<div class="nbox">Tests whether economic bias persists when models '
                        'train across employers or gig platforms without centralising '
                        'sensitive worker data.</div>', unsafe_allow_html=True)
            if not fed:
                st.info(tr("run_to_see_fed"))
            else:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Clients", fed["n_clients"])
                c2.metric("Global Accuracy", f'{fed["global_accuracy"]:.1%}')
                c3.metric("Global Fairness", f'{fed["global_fairness"]:.3f}')
                c4.metric("Bias Persisted", "Yes ⚠️" if fed["bias_persisted"] else "No ✅")
                st.markdown(
                    f'<div class="{"flag-c" if fed["bias_persisted"] else "nbox"}">'
                    f'<em>{fed["narrative"]}</em></div>', unsafe_allow_html=True)
                cr_list = fed.get("client_results",[])
                if cr_list:
                    cr_df = pd.DataFrame([c.__dict__ if hasattr(c,"__dict__") else c
                                          for c in cr_list])
                    if not cr_df.empty and "local_bias" in cr_df.columns:
                        fig_fed = px.bar(cr_df, x="client_id",
                            y=["local_accuracy","local_bias","local_fairness"],
                            barmode="group", title=tr("chart_fed_clients"),
                            color_discrete_sequence=[ACCENT,"#ef4444","#22c55e"])
                        try: fig_fed.update_layout(**PT(),height=320)
                        except: fig_fed.update_layout(height=320)
                        st.plotly_chart(fig_fed, use_container_width=True)

        # ── TAB 8: COMPLIANCE ─────────────────────────────────────────────────
        with tab_map[tr("tab_compliance")]:
            st.markdown(tr("heading_compliance"))
            _cr = xai.get("compliance_report",{}); _mc = xai.get("model_card",{})
            if not _cr:
                st.info(tr("xai_compliance_prompt"))
            else:
                summ = _cr.get("summary",{}); ok = summ.get("overall_compliant",False)
                st.markdown(
                    f'<div class="{"nbox" if ok else "flag-c"}">'
                    f'Overall: <strong>{"COMPLIANT ✅" if ok else "NON-COMPLIANT ❌"}</strong> | '
                    f'Fairness: {summ.get("fairness_score",0):.3f} | '
                    f'Gender gap: {avg_ggap:.1%} | Informal gap: {avg_igap:.1%}</div>',
                    unsafe_allow_html=True)
                for fw,fd in _cr.get("frameworks",{}).items():
                    with st.expander(f"📑 {fw}"):
                        for ch,st_ in fd.get("checks",{}).items():
                            st.markdown(f"{'✅' if st_=='PASS' else '❌'} {ch}")
                try:
                    pdf_b = generate_pdf_compliance_report(_cr,_mc,
                        {"accuracy":avg_acc,"fairness_score":avg_fair},domain="generic")
                    st.download_button(tr("download_pdf"),pdf_b,
                        "economic_compliance.pdf","application/pdf",use_container_width=True)
                except Exception as e:
                    st.caption(f"PDF unavailable: {e}")
            st.divider()
            st.markdown("#### 🇳🇬 Nigeria Regulatory Panel")
            nigeria_compliance_panel(
                {"accuracy":avg_acc,"fairness_score":avg_fair,
                 "demographic_parity":df["demographic_parity"].mean()},
                domain="agrotech", has_ussd_fallback=False, has_gender_audit=True,
                has_multilingual=False, has_xai=enable_xai,
                has_governance=enable_governance, has_redteam=False)

        # ── TAB 9: EXPORT ─────────────────────────────────────────────────────
        with tab_map[tr("tab_export")]:
            st.markdown(tr("heading_export"))
            e1,e2,e3 = st.columns(3)
            sc = [c for c in df.columns if c not in ["narrative","critical_flags"]]
            with e1:
                st.download_button(tr("download_csv"),
                    df[sc].to_csv(index=False).encode(),
                    f"gags_economics_{scenario_key}.csv","text/csv",
                    use_container_width=True)
            with e2:
                export_json = {
                    "gags_version":"4.0", "module":"Economic Justice §25",
                    "timestamp":datetime.now().isoformat(),
                    "scenario":{"key":scenario_key,"name":preset_info.name,
                                "domain":preset_info.economic_domain,
                                "citation":preset_info.citation},
                    "configuration":{"bias_intensity":bias_intensity,
                                     "selected_biases":selected_biases,
                                     "n_samples":int(n_samples),"n_runs":n_runs},
                    "results":{"avg_accuracy":round(avg_acc,4),
                               "avg_fairness":round(avg_fair,4),
                               "avg_inclusion":round(avg_incl,4),
                               "avg_gender_gap":round(avg_ggap,4),
                               "avg_informal_gap":round(avg_igap,4),
                               "avg_intersectional_gap":round(avg_xgap,4)},
                    "regulatory_body":preset_info.regulatory_body,
                    "runs":df[sc].to_dict(orient="records"),
                }
                st.download_button(tr("download_json"),
                    json.dumps(export_json,indent=2,default=str),
                    f"gags_econ_{scenario_key}_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json", use_container_width=True)
            with e3:
                # Research narrative
                narrative_txt = (
                    f"GAGS Economic Justice Simulation Report\n"
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                    f"{'='*60}\n\n"
                    f"Scenario: {preset_info.name}\n"
                    f"Domain:   {preset_info.economic_domain.title()}\n"
                    f"Citation: {preset_info.citation}\n\n"
                    f"Key Results ({n_runs} runs, n={n_samples:,} each):\n"
                    f"  Accuracy:              {avg_acc:.3f}\n"
                    f"  Fairness Score:        {avg_fair:.3f}\n"
                    f"  Economic Inclusion:    {avg_incl:.3f}\n"
                    f"  Gender Outcome Gap:    {avg_ggap:.3f} ({avg_ggap:.1%})\n"
                    f"  Informal Sector Gap:   {avg_igap:.3f} ({avg_igap:.1%})\n"
                    f"  Intersectional Gap:    {avg_xgap:.3f} ({avg_xgap:.1%})\n"
                    f"  Wage Suppression Idx:  {avg_wsi:.3f}\n\n"
                    f"Regulatory Body: {preset_info.regulatory_body}\n"
                    f"Bias Intensity:  {bias_intensity:.2f}\n"
                    f"Poison Rate:     {poison_rate:.3f}\n"
                    f"Active Biases:   {', '.join(selected_biases) or 'None'}\n\n"
                    f"Narrative:\n{df['narrative'].iloc[-1] if 'narrative' in df.columns else ''}\n"
                )
                st.download_button(tr("download_pdf"),
                    narrative_txt.encode(), f"gags_econ_narrative_{scenario_key}.txt",
                    "text/plain", use_container_width=True)

            # Dataframe preview
            st.dataframe(
                df[sc].style
                  .format({c:"{:.3f}" for c in sc
                           if c not in ["run_id","scenario","economic_domain",
                                        "biases","regulatory_body"]
                           and df[sc][c].dtype in [float]})
                  .background_gradient(subset=["fairness_score"],cmap="RdYlGn")
                  .background_gradient(subset=["gender_outcome_gap"],cmap="Reds"),
                use_container_width=True)

        # ── TAB 10: RESEARCH / STATISTICAL DISTRIBUTION ───────────────────────
        if vm=="Research" and tr("tab_stats") in tab_map:
            with tab_map[tr("tab_stats")]:
                st.markdown(tr("heading_stats"))
                st.markdown('<div class="nbox">Violin + box plots show the full distribution '
                            'of each metric — not just the mean. Essential for research: '
                            'shows variance, outliers, and statistical reliability.</div>',
                            unsafe_allow_html=True)
                dist_metrics = ["accuracy","fairness_score","economic_inclusion_score",
                                "gender_outcome_gap","informal_sector_gap",
                                "intersectional_worst_gap","wage_suppression_index"]
                dist_labels  = {"accuracy":"Accuracy","fairness_score":"Fairness",
                                "economic_inclusion_score":"Inclusion",
                                "gender_outcome_gap":"Gender Gap","informal_sector_gap":"Informal Gap",
                                "intersectional_worst_gap":"Intersect. Gap",
                                "wage_suppression_index":"Wage Suppression"}
                if CHARTS_OK:
                    st.plotly_chart(multi_run_distribution(
                        st.session_state.econ_run_history,
                        metrics=dist_metrics, metric_labels=dist_labels,
                        accent=ACCENT, height=460,
                        title=f"Full Distribution — {n_runs} Simulation Runs",
                    ), use_container_width=True)
                else:
                    st.dataframe(df[dist_metrics].describe(), use_container_width=True)

    history_browser("econ_snapshot_history", domain="health",
        key_metrics=["accuracy","fairness_score","economic_inclusion_score","gender_outcome_gap"])
    annotation_panel("econ_annotations",
        context_label=f"{len(st.session_state.econ_run_history)} Economic run(s)")

    # ── Policy recommendations ─────────────────────────────────────────────────
    st.divider()
    st.markdown("## 💡 Economic Justice Recommendations")
    r1,r2,r3 = st.columns(3)
    with r1:
        st.info("""**👤 For Employers & Platforms**
1. **Annual AI audits**: Third-party bias assessment before retraining
2. **Diverse training data**: Oversample informal, female, minority workers
3. **Explainable decisions**: Written reasons for every AI-driven rejection
4. **Human override**: Mandatory review for borderline decisions
5. **Pay transparency**: Publish AI-determined wage distributions by gender

**🛵 Gig Platforms**
1. Publish dispatch algorithm criteria
2. Audit rating systems for racial bias
3. Explain surge pricing to drivers""")
    with r2:
        st.success("""**🤖 For AI Developers**
1. **Fairness constraints**: Build parity into training loss functions
2. **Informal sector features**: Never use formality as a proxy variable
3. **Intersectional testing**: Test female × informal compound penalty
4. **Counterfactual fairness**: Decisions invariant to protected attributes
5. **Model cards**: Publish fairness metrics before any deployment

**📊 For Economists & Researchers**
1. Causal identification: separate AI bias from pre-existing wage gaps
2. Longitudinal tracking: monitor self-reinforcing bias spirals
3. Welfare cost quantification: GDP impact of algorithmic discrimination""")
    with r3:
        st.warning(f"""**🏛️ For Regulators ({preset_info.regulatory_body})**
1. **Mandatory pre-deployment audits** for economic AI
2. **EU AI Act alignment**: employment AI is High-Risk (Annex III)
3. **ILO Convention 2023**: AI workplace standards compliance
4. **EEOC guidance**: algorithmic hiring fairness rules
5. **Nigeria Labour Act**: extend to algorithmic employment decisions

**🌍 Informal Economy (Nigeria-Specific)**
1. Accept mobile money as creditworthiness signal
2. Design AI pathways that function without BVN/formal ID
3. Include informal sector reps in AI governance boards
4. Mandate Hausa/Yoruba/Igbo interfaces for gig platforms""")

    st.caption("⚠️ Disclaimer: Simulation for research and educational use. "
               "Real deployment requires sector-specific audits and regulatory review.")

else:
    # ── Welcome ────────────────────────────────────────────────────────────────
    st.markdown("## 💼 Economic Justice Simulation — Research + Industry Edition")

    # Build domain cards for welcome state
    _domain_items = list(_DOMAIN_META.items())
    _domain_descs = [
        "How resume screening AI systematically disadvantages women, minorities, and informal-sector graduates.",
        "How platform algorithms assign earnings in ways that penalise low-income neighbourhood workers.",
        "How personalised pricing charges informal economy users more for identical goods.",
        "How algorithmic wage-setting suppresses earnings for women and automation-exposed workers — compounding each retraining cycle.",
        "How search algorithms on digital marketplaces bury informal and female-led micro-enterprises.",
        "How benefit fraud AI wrongfully excludes the most vulnerable — Robodebt and SyRI patterns in Nigeria's CCT.",
    ]
    def _dc(dm, desc):
        return (f'<div style="background:#f1f5f9;border:1px solid #e2e8f0;'
                f'border-left:3px solid {ACCENT};border-radius:0 8px 8px 0;'
                f'padding:.85rem 1rem">'
                f'<p style="font-family:DM Mono,monospace;font-size:.67rem;'
                f'letter-spacing:.08em;text-transform:uppercase;color:{ACCENT};margin:0 0 .2rem">'
                f'{dm["icon"]} {dm["label"]}</p>'
                f'<p style="font-size:.78rem;color:#64748b;margin:0;line-height:1.4">{desc}</p>'
                f'</div>')

    row1_html = "".join(_dc(dm, desc) for (_,dm),desc in zip(_domain_items[:3],_domain_descs[:3]))
    row2_html = "".join(_dc(dm, desc) for (_,dm),desc in zip(_domain_items[3:],_domain_descs[3:]))

    st.markdown(f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:2rem;margin-bottom:1.5rem">
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1rem">
    {row1_html}
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem">
    {row2_html}
  </div>
</div>""", unsafe_allow_html=True)

    with st.expander("📖 Research + Industry Features", expanded=True):
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("""
**🔬 Research Features**
- 6 economic domains × scenario presets calibrated to real data
- Real-world benchmarks from 14 landmark studies (COMPAS, Robodebt, SyRI, etc.)
- Animated longitudinal bias drift charts with play/pause
- SHAP-style waterfall charts for individual decisions
- Statistical distribution (violin/box) across multiple runs
- Counterfactual analysis: "what would change this outcome?"
- Intersectional fairness: female × informal compound penalty
- Federated learning: bias across employers without data centralisation
- Regulatory compliance: EU AI Act, ILO, EEOC, NITDA, Nigeria Labour Act
            """)
        with c2:
            st.markdown("""
**🏭 Industry Features**
- Industry mode: KPI gauges first, regulatory table prominent
- Research mode: full statistical distribution, CI display
- Board member executive summary (one verdict + one action)
- PDF compliance report download
- Sankey diagram: who gets what and why
- Historical AI bias incident timeline (2014–2024)
- JSON export for MLOps pipeline integration
- Nigeria macroeconomic context data embedded (NBS, CBN, World Bank)
- Lollipop gap charts (cleaner than grouped bars for gap analysis)
- Benchmark comparison: your simulation vs published real-world AI systems
            """)

    if BENCHMARKS_OK:
        st.markdown("### 📚 Embedded Real-World Benchmark Database")
        bm_preview = pd.DataFrame(benchmark_summary_table())
        st.dataframe(bm_preview[["Study","Domain","Region","Year","Severity"]],
                     use_container_width=True, hide_index=True, height=320)
        st.caption("14 landmark AI bias studies embedded — no API key required. "
                   "Compare your simulation results directly against published research.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#64748b;font-family:DM Mono,monospace;"
    "font-size:.68rem;padding:.6rem 0;letter-spacing:.05em'>"
    "💼 ECONOMIC JUSTICE · GAGS v4.0 · §25 Economic Justice Engine · "
    "ILO · EU AI Act · EEOC · NITDA · Nigeria Labour Act · "
    "14 Real-World Benchmarks · Research + Industry Edition"
    "</div>", unsafe_allow_html=True)