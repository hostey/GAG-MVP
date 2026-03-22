# pages/08_💰_Financial_Inclusion.py
"""
Financial Inclusion & Credit Scoring Fairness — GAGS Framework v4.0
====================================================================
Credit scoring bias, ECOA disparate impact, informal economy exclusion.
Nigeria CBN/NDIC context. Redlining prevention. Light-mode design.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from components.governance_logic import (
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
)
from components.pdf_report import generate_pdf_compliance_report
from components.i18n import language_switcher
try:
    from components.nigeria_regulatory import nigeria_compliance_panel
except ImportError:
    def nigeria_compliance_panel(*a, **kw): pass
from utils.config import simulation_config, settings

st.set_page_config(page_title="Financial Inclusion • GAGS", page_icon="💰", layout="wide")

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
    return m


def _run_one(scenario_key, n_samples, biases, bias_int, poison_rate, run_idx,
             low_inc, minority, alt_w, reg, country_meta, enable_xai, enable_gov):
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
        stratify=y if len(np.unique(y))>1 else None)
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
    return {
        "run_id":run_idx+1, "scenario":preset.name,
        "accuracy":m["accuracy"], "precision":m["precision"],
        "recall":m["recall"], "f1_score":m["f1_score"],
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
with st.sidebar:
    language_switcher(location="sidebar"); st.divider()
    role_switcher("health"); st.divider()
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

    st.subheader("🎭 Bias Configuration")
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

    st.subheader("⚠️ Adversarial Attacks")
    poison_rate = st.slider("Attack Strength", 0.0, 0.5, 0.05, 0.01, format="%.2f")
    st.divider()

    st.subheader("📊 Simulation")
    n_samples = st.number_input("Customer Records", 1000, 100000, settings.DEFAULT_N_SAMPLES, 1000)
    n_runs    = st.slider("Simulation Runs", 1, 10, 3)
    st.divider()

    st.subheader("🔬 Feature Modules")
    enable_xai        = st.toggle("Explainable AI", value=True)
    enable_governance = st.toggle("Governance Layer", value=True)
    st.divider()

    col_r, col_x = st.columns(2)
    run_btn = col_r.button("💰 Run", type="primary", use_container_width=True)
    if col_x.button("🔄 Reset", use_container_width=True):
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
    st.session_state.fin_run_history = []
    prog = st.progress(0, text="Initialising…")
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
    prog.progress(1.0, text="Complete ✓"); prog.empty()

# ── Dashboard ──────────────────────────────────────────────────────────────────
if st.session_state.fin_run_history:
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

    role_banner("health")
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

        tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
            "📈 Performance","⚖️ Fairness","💰 Economic Impact",
            "🧠 Explainable AI","📋 Compliance","🔁 Longitudinal",
            "🌐 Federated","📋 Raw Results"])

        with tab1:
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

        with tab2:
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

        with tab3:
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

        with tab4:
            st.markdown("### 🧠 Explainable AI")
            if not xai: st.info("Enable XAI in sidebar and run simulation.")
            elif "error" in xai: st.warning(f"XAI error: {xai['error']}")
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

        with tab5:
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

        with tab6:
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

        with tab7:
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

        with tab8:
            sc = [c for c in df.columns if c != "narrative"]
            st.dataframe(df[sc].style
                .background_gradient(subset=["accuracy"],cmap="Blues")
                .background_gradient(subset=["fairness_score"],cmap="RdYlGn")
                .background_gradient(subset=["approval_rate_low"],cmap="Greens"),
                use_container_width=True)
            d1,d2 = st.columns(2)
            with d1:
                st.download_button("📥 Download CSV",df[sc].to_csv(index=False).encode(),
                    f"gags_finance_{scenario_key}.csv","text/csv",use_container_width=True)
            with d2:
                st.download_button("📋 Export Config",json.dumps({
                    "scenario_key":scenario_key,"country_income_level":country_income_level,
                    "institution_type":institution_type,"selected_biases":selected_biases,
                    "bias_intensity":bias_intensity,"regulatory_compliance":regulatory_compliance,
                    "avg_accuracy":f"{avg_acc:.3f}","avg_fairness":f"{avg_fair:.3f}",
                    "avg_di_ratio":f"{avg_di:.3f}"},indent=2),
                    f"finance_config_{scenario_key}.json","application/json",use_container_width=True)

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