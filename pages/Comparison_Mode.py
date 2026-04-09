# pages/Comparison_Mode.py
"""
GAGS Comparison Mode v4.0
Compare any two simulations side-by-side across all 7 domains.
Domain-specific scenarios, metrics, composite scores, and delta analysis.
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
from sklearn.metrics import (accuracy_score, recall_score,
                              precision_score, f1_score)

from components.governance_logic import (
    apply_bias, simulate_data_poisoning, calculate_fairness_metrics,
    generate_synthetic_data, generate_africa_centric_data,
    generate_education_data, EDUCATION_SCENARIO_PRESETS,
    generate_financial_data, FINANCIAL_SCENARIO_PRESETS,
    generate_judicial_data, JUDICIAL_SCENARIO_PRESETS,
    generate_disinformation_data, DISINFORMATION_SCENARIO_PRESETS,
    AttackSeverity,
)
from utils.config import simulation_config, settings

st.set_page_config(
    page_title="Comparison Mode • GAGS",
    page_icon="", layout="wide",
)

try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, plotly_theme as _ptheme
    inject_css("home")
    ACCENT = DOMAIN_ACCENTS["home"]
except ImportError:
    ACCENT = "#00e5ff"

# ── Extra comparison-specific styles ──────────────────────────────────────────
st.markdown("""
<style>
.slot-a{background:rgba(41,128,185,.12);border:1px solid rgba(41,128,185,.35);
  border-radius:8px;padding:.5rem .85rem;margin:.5rem 0;font-family:var(--ff-m);
  font-size:.78rem;color:#7fbfff}
.slot-b{background:rgba(231,76,60,.12);border:1px solid rgba(231,76,60,.35);
  border-radius:8px;padding:.5rem .85rem;margin:.5rem 0;font-family:var(--ff-m);
  font-size:.78rem;color:#ff9999}
.winner-a{color:#7fbfff;font-family:var(--ff-m);font-size:.72rem;
  letter-spacing:.05em;text-transform:uppercase}
.winner-b{color:#ff9999;font-family:var(--ff-m);font-size:.72rem;
  letter-spacing:.05em;text-transform:uppercase}
.winner-tie{color:var(--t2);font-family:var(--ff-m);font-size:.72rem;
  letter-spacing:.05em;text-transform:uppercase}
.domain-badge{display:inline-flex;align-items:center;gap:.4rem;
  padding:.25rem .75rem;border-radius:99px;font-family:var(--ff-m);
  font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;
  border:1px solid;margin-bottom:.5rem}
.delta-pos{color:#39ff7a;font-family:var(--ff-m);font-size:.8rem}
.delta-neg{color:#ff3b5c;font-family:var(--ff-m);font-size:.8rem}
.delta-neu{color:var(--t2);font-family:var(--ff-m);font-size:.8rem}
.summary-card{background:var(--bg2);border:1px solid var(--bg3);border-radius:10px;
  padding:1.25rem;height:100%}
.scenario-pill{display:inline-block;background:var(--bg2);border:1px solid var(--bg3);
  border-radius:4px;padding:2px 8px;font-family:var(--ff-m);font-size:.68rem;
  color:var(--t2);margin:2px}
</style>
""", unsafe_allow_html=True)

# ── Domain registry — all 7 domains + scenarios ───────────────────────────────
_DOMAINS = {
    "Healthcare": {
        "icon": "🏥", "accent": "#00e5ff", "key": "health",
        "scenarios": [
            "UCI Heart Disease",
            "PIMA Diabetes (Women)",
            "Breast Cancer Wisconsin",
            "Abuja FCT (Africa-centric)",
            "Abuja Maternal Health",
        ],
        "primary_metrics": ["accuracy","recall","fairness_score","composite_score"],
        "domain_label": "Sensitivity / Equity Score",
    },
    "National Security": {
        "icon": "🛡️", "accent": "#ff3b5c", "key": "security",
        "scenarios": [
            "Counter-Terrorism (GTD)",
            "Cyber Threat (UNSW-NB15)",
            "Predictive Policing — Urban",
            "Border Control",
            "Financial Crime",
        ],
        "primary_metrics": ["accuracy","false_positive_rate","fairness_score","composite_score"],
        "domain_label": "Detection Rate / Liberty Score",
    },
    "Agrotech": {
        "icon": "🌾", "accent": "#39ff7a", "key": "agrotech",
        "scenarios": [
            "Smallholder Farming (Abuja FCT)",
            "Climate-Resilient Maize (Plateau State)",
            "Irrigation Access Equity",
            "Market Linkage AI",
        ],
        "primary_metrics": ["accuracy","fairness_score","demographic_parity","composite_score"],
        "domain_label": "Crop Risk / Gender Gap",
    },
    "Education": {
        "icon": "🎓", "accent": "#f5a623", "key": "education",
        "scenarios": list(EDUCATION_SCENARIO_PRESETS.keys()),
        "primary_metrics": ["accuracy","fairness_score","demographic_parity","composite_score"],
        "domain_label": "Opportunity Gap / SES Equity",
        "scenario_labels": {k: EDUCATION_SCENARIO_PRESETS[k].name
                            for k in EDUCATION_SCENARIO_PRESETS},
    },
    "Financial": {
        "icon": "💰", "accent": "#00d4aa", "key": "finance",
        "scenarios": list(FINANCIAL_SCENARIO_PRESETS.keys()),
        "primary_metrics": ["accuracy","fairness_score","false_positive_rate","composite_score"],
        "domain_label": "DI Ratio / Inclusion Score",
        "scenario_labels": {k: FINANCIAL_SCENARIO_PRESETS[k].name
                            for k in FINANCIAL_SCENARIO_PRESETS},
    },
    "Judicial": {
        "icon": "⚖️", "accent": "#c084fc", "key": "judicial",
        "scenarios": list(JUDICIAL_SCENARIO_PRESETS.keys()),
        "primary_metrics": ["accuracy","false_positive_rate","fairness_score","composite_score"],
        "domain_label": "Racial FPR Gap / Liberty",
        "scenario_labels": {k: JUDICIAL_SCENARIO_PRESETS[k].name
                            for k in JUDICIAL_SCENARIO_PRESETS},
    },
    "Disinformation": {
        "icon": "📡", "accent": "#fb923c", "key": "disinformation",
        "scenarios": list(DISINFORMATION_SCENARIO_PRESETS.keys()),
        "primary_metrics": ["accuracy","recall","fairness_score","composite_score"],
        "domain_label": "Language FPR Gap / Over-Removal",
        "scenario_labels": {k: DISINFORMATION_SCENARIO_PRESETS[k].name
                            for k in DISINFORMATION_SCENARIO_PRESETS},
    },
}

_LOWER_IS_BETTER = {
    "false_positive_rate","fpr","demographic_parity",
    "equalized_odds","poison_rate","bias_intensity","fpr_disparity",
    "racial_fpr_gap","language_fpr_gap","over_removal_rate","speech_suppression",
}

_VALID_BIAS_TYPES = list(simulation_config.BIAS_TYPES) + [
    b for b in ["gender","linguistic"] if b not in simulation_config.BIAS_TYPES]

_STATE = {
    "cmp_result_a": None, "cmp_result_b": None,
    "cmp_config_a": None, "cmp_config_b": None,
    # Plain-value keys for preset population (never used as widget keys)
    "_pv_a_domain":   None, "_pv_b_domain":   None,
    "_pv_a_scenario": None, "_pv_b_scenario": None,
    "_pv_a_bias_int": None, "_pv_b_bias_int": None,
    "_pv_a_poison":   None, "_pv_b_poison":   None,
}
for k, v in _STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Core simulation engine ────────────────────────────────────────────────────
def _run_simulation(domain: str, scenario: str, selected_biases: list,
                    bias_intensity: float, poison_rate: float,
                    n_samples: int, random_seed: int = 42) -> dict:
    """
    Generic simulator for all 7 domains.
    Returns a normalised metrics dict comparable across domains.
    """
    valid = [b for b in selected_biases if b in _VALID_BIAS_TYPES]
    rng   = np.random.RandomState(random_seed)

    # ── Data generation — domain aware ───────────────────────────────────────
    try:
        if domain == "Education":
            X, y, demo, feat_names, preset = generate_education_data(
                scenario, n_samples, random_state=random_seed)
        elif domain == "Financial":
            X, y, demo, feat_names, preset = generate_financial_data(
                scenario, n_samples, random_state=random_seed)
        elif domain == "Judicial":
            X, y, demo, feat_names, preset = generate_judicial_data(
                scenario, n_samples, random_state=random_seed)
        elif domain == "Disinformation":
            X, y, demo, feat_names, preset = generate_disinformation_data(
                scenario, n_samples, random_state=random_seed)
        elif domain == "Agrotech":
            X, y, demo, _ = generate_africa_centric_data(
                "smallholder_agrotech", n_samples)
        else:
            db = simulation_config.DECISION_BOUNDARY
            if domain == "National Security":
                db += 0.5
            X, y, demo = generate_synthetic_data(
                n_samples=n_samples, n_features=12, decision_boundary=db)
    except Exception:
        X, y, demo = generate_synthetic_data(n_samples=n_samples, n_features=12)
        feat_names = [f"f{i}" for i in range(12)]

    X = X.astype(np.float64)

    # ── Bias injection ────────────────────────────────────────────────────────
    for bt in valid:
        try:
            X, y, demo = apply_bias(X, y, bt, bias_intensity,
                                    demographic_info=demo,
                                    severity=AttackSeverity.MEDIUM)
        except Exception:
            pass

    # ── Poisoning ─────────────────────────────────────────────────────────────
    try:
        X, y, demo = simulate_data_poisoning(
            X, y, poison_rate, attack_type="label_flipping",
            demographic_info=demo, targeted=False)
    except Exception:
        pass

    # ── Train / evaluate ──────────────────────────────────────────────────────
    if len(np.unique(y)) < 2:
        return {"error": "Single-class target after bias injection. Lower bias intensity."}

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.3, random_state=random_seed,
        stratify=y if len(np.unique(y)) > 1 else None)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                  random_state=random_seed)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)

    acc  = float(accuracy_score(yte, yp))
    rec  = float(recall_score(yte, yp, zero_division=0))
    prec = float(precision_score(yte, yp, zero_division=0))
    f1   = float(f1_score(yte, yp, zero_division=0))
    fpr  = float(np.mean(yp[yte==0]==1)) if (yte==0).any() else 0.0
    spec = float(np.mean(yp[yte==0]==0)) if (yte==0).any() else 1.0
    fair = calculate_fairness_metrics(yte, yp, demo[:len(yte)])
    fs   = fair.get("fairness_score", 0.5)
    dp   = fair.get("demographic_parity_difference", 0)
    eo   = fair.get("equalized_odds_difference", 0)

    # Per-group accuracy gap
    adv = demo[:len(yte)] == 1
    dis = demo[:len(yte)] == 0
    acc_hi = float(accuracy_score(yte[adv], yp[adv])) if adv.any() else acc
    acc_lo = float(accuracy_score(yte[dis], yp[dis])) if dis.any() else acc

    # ── Domain-specific composite score ──────────────────────────────────────
    if domain == "National Security":
        # Balance detection vs liberty (low FPR)
        composite = max(0, 0.35*rec + 0.35*(1-fpr) + 0.30*fs)
    elif domain == "Healthcare":
        composite = 0.30*acc + 0.40*fs + 0.30*rec
    elif domain == "Agrotech":
        composite = 0.40*acc + 0.40*fs + 0.20*(1-dp)
    elif domain == "Education":
        composite = 0.35*acc + 0.40*fs + 0.25*(1-dp)
    elif domain == "Financial":
        # ECOA disparate impact proxy
        di_ratio = min(acc_lo/(acc_hi+1e-8), 1.0)
        composite = 0.25*acc + 0.40*fs + 0.35*di_ratio
    elif domain == "Judicial":
        # Penalise FPR heavily — wrongful detention
        composite = max(0, 0.20*acc + 0.30*fs + 0.50*(1-fpr*2))
    elif domain == "Disinformation":
        composite = 0.30*prec + 0.30*rec + 0.40*fs
    else:
        composite = 0.50*acc + 0.50*fs

    # Domain-specific extra metrics
    extra = {}
    if domain == "National Security":
        extra["liberty_score"] = round(max(0, 1 - fpr*2 - bias_intensity*0.3), 4)
        extra["detection_rate"] = round(rec, 4)
    elif domain == "Financial":
        di = acc_lo / (acc_hi + 1e-8)
        extra["disparate_impact_ratio"] = round(min(di, 1.0), 4)
        extra["ecoa_compliant"] = di >= 0.80
    elif domain == "Judicial":
        extra["racial_fpr_gap"] = round(fpr, 4)
        extra["liberty_score"]  = round(max(0, 1 - fpr*2), 4)
    elif domain == "Disinformation":
        extra["language_fpr_gap"]  = round(fpr * 1.2, 4)   # proxy
        extra["over_removal_rate"] = round(float(np.mean((yp==1)&(yte==0))), 4)
    elif domain == "Education":
        extra["opportunity_gap"] = round(abs(acc_hi - acc_lo), 4)
    elif domain == "Agrotech":
        extra["gender_gap"] = round(abs(acc_hi - acc_lo), 4)

    return {
        "domain": domain,
        "scenario": scenario,
        "accuracy":           round(acc,  4),
        "recall":             round(rec,  4),
        "precision":          round(prec, 4),
        "f1_score":           round(f1,   4),
        "specificity":        round(spec, 4),
        "false_positive_rate":round(fpr,  4),
        "fairness_score":     round(fs,   4),
        "demographic_parity": round(dp,   4),
        "equalized_odds":     round(eo,   4),
        "fpr_disparity":      round(fair.get("fpr_disparity", 0), 4),
        "acc_hi":             round(acc_hi, 4),
        "acc_lo":             round(acc_lo, 4),
        "equity_gap":         round(abs(acc_hi - acc_lo), 4),
        "composite_score":    round(composite, 4),
        "bias_intensity":     round(bias_intensity, 4),
        "poison_rate":        round(poison_rate, 4),
        "n_samples":          n_samples,
        "biases_applied":     ", ".join(valid) or "None",
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in extra.items()},
    }


# ── Apply any pending preset (must happen before widgets are instantiated) ─────
_DOMAIN_KEYS  = list(_DOMAINS.keys())
_DOMAIN_INDEX = {d: i for i, d in enumerate(_DOMAIN_KEYS)}


# ── Preset staging (read BEFORE widgets, never write to widget keys) ───────────
# Presets write to _pv_* (plain value keys, not widget keys).
# Widgets read _pv_* as their index/value. This never touches a widget key.

def _pv(key, default):
    """Read a plain-value key; return default if absent OR None."""
    val = st.session_state.get(key)
    return default if val is None else val


with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:.5rem 0 1rem">
      <h2 style="color:var(--ac,#38bdf8);margin:0;font-family:var(--ff-d,'Syne',sans-serif)">
         Comparison Mode</h2>
      <p style="color:var(--t2,#5a6a85);font-size:.78rem;margin:.3rem 0 0;
         font-family:var(--ff-m,'DM Mono',monospace)">
        Configure Slot A and Slot B independently</p>
    </div>""", unsafe_allow_html=True)

    # ── SLOT A ────────────────────────────────────────────────────────────────
    st.markdown('<div class="slot-a">🔵 Slot A — Configuration</div>',
                unsafe_allow_html=True)
    a_domain   = st.selectbox("Domain A", _DOMAIN_KEYS, key="a_domain",
                    index=_DOMAIN_INDEX.get(_pv("_pv_a_domain", _DOMAIN_KEYS[0]), 0))
    a_info     = _DOMAINS[a_domain]
    _a_scen_opts    = a_info["scenarios"]
    _a_scen_default = _pv("_pv_a_scenario", _a_scen_opts[0])
    _a_scen_idx     = _a_scen_opts.index(_a_scen_default) if _a_scen_default in _a_scen_opts else 0
    a_scen_raw = st.selectbox("Scenario A", _a_scen_opts, index=_a_scen_idx, key="a_scenario",
        format_func=lambda k: a_info.get("scenario_labels", {}).get(k, k))
    a_biases   = st.multiselect("Biases A", _VALID_BIAS_TYPES,
        default=[b for b in ["demographic","socioeconomic"] if b in _VALID_BIAS_TYPES],
        key="a_biases")
    a_bias_int = st.slider("Bias intensity A", 0.0, 1.0,
                    float(_pv("_pv_a_bias_int", 0.20)), 0.05, key="a_bias_int")
    a_poison   = st.slider("Poison rate A", 0.0, 0.5,
                    float(_pv("_pv_a_poison", 0.05)), 0.01, key="a_poison")
    a_samples  = st.number_input("Samples A", 500, 50_000, 5000, 500, key="a_samples")

    st.divider()

    # ── SLOT B ────────────────────────────────────────────────────────────────
    st.markdown('<div class="slot-b">🔴 Slot B — Configuration</div>',
                unsafe_allow_html=True)
    b_domain   = st.selectbox("Domain B", _DOMAIN_KEYS, key="b_domain",
                    index=_DOMAIN_INDEX.get(_pv("_pv_b_domain", _DOMAIN_KEYS[1]), 1))
    b_info     = _DOMAINS[b_domain]
    _b_scen_opts    = b_info["scenarios"]
    _b_scen_default = _pv("_pv_b_scenario", _b_scen_opts[0])
    _b_scen_idx     = _b_scen_opts.index(_b_scen_default) if _b_scen_default in _b_scen_opts else 0
    b_scen_raw = st.selectbox("Scenario B", _b_scen_opts, index=_b_scen_idx, key="b_scenario",
        format_func=lambda k: b_info.get("scenario_labels", {}).get(k, k))
    b_biases   = st.multiselect("Biases B", _VALID_BIAS_TYPES,
        default=[b for b in ["demographic","geographic"] if b in _VALID_BIAS_TYPES],
        key="b_biases")
    b_bias_int = st.slider("Bias intensity B", 0.0, 1.0,
                    float(_pv("_pv_b_bias_int", 0.35)), 0.05, key="b_bias_int")
    b_poison   = st.slider("Poison rate B", 0.0, 0.5,
                    float(_pv("_pv_b_poison", 0.08)), 0.01, key="b_poison")
    b_samples  = st.number_input("Samples B", 500, 50_000, 5000, 500, key="b_samples")

    st.divider()

    col_r, col_x = st.columns(2)
    run_btn = col_r.button("🔀 Run", type="primary", use_container_width=True)
    if col_x.button("🗑 Clear", use_container_width=True):
        for k, v in _STATE.items():
            st.session_state[k] = v
        st.rerun()

    # Quick-start presets
    st.divider()
    st.markdown('<p style="font-family:var(--ff-m,monospace);font-size:.7rem;'
                'letter-spacing:.1em;text-transform:uppercase;color:var(--t2,#666);">'
                'Quick-start presets</p>', unsafe_allow_html=True)
    PRESETS = [
        ("Low vs High Bias",   "Healthcare", "Healthcare", "UCI Heart Disease", "UCI Heart Disease", 0.1, 0.4, 0.03, 0.03),
        ("Health vs Finance",  "Healthcare", "Financial",  "UCI Heart Disease", "sme_credit_nigeria", 0.25, 0.25, 0.05, 0.05),
        ("Security vs Judicial","National Security","Judicial","Counter-Terrorism (GTD)","bail_decision_nigeria",0.3,0.3,0.05,0.05),
        ("Agro vs Education",  "Agrotech",   "Education",  "Smallholder Farming (Abuja FCT)","jamb_admission",0.2,0.2,0.04,0.04),
        ("Disinfo vs Judicial","Disinformation","Judicial","election_nigeria_2027","bail_decision_nigeria",0.3,0.3,0.06,0.06),
        ("Finance vs Judicial","Financial",  "Judicial",   "sme_credit_nigeria","bail_decision_nigeria",0.35,0.35,0.05,0.05),
    ]
    for label, da, db, sa, sb, bia, bib, pa, pb in PRESETS:
        if st.button(label, key=f"_preset_{label}", use_container_width=True):
            st.session_state["_pv_a_domain"]   = da
            st.session_state["_pv_b_domain"]   = db
            st.session_state["_pv_a_scenario"] = sa
            st.session_state["_pv_b_scenario"] = sb
            st.session_state["_pv_a_bias_int"] = bia
            st.session_state["_pv_b_bias_int"] = bib
            st.session_state["_pv_a_poison"]   = pa
            st.session_state["_pv_b_poison"]   = pb
            st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header fade-in">
  <p class="eyebrow">GAGS · Comparison Mode · All 7 Domains</p>
  <h1> Head-to-Head Simulation Comparison</h1>
  <p>Run any two simulations side-by-side — same domain different settings,
     or entirely different domains. Every metric diffed, every gap quantified.</p>
  <div style="margin-top:.75rem">
    {"".join(f'<span class="badge" style="color:{d["accent"]};border-color:{d["accent"]}44;background:{d["accent"]}11">{d["icon"]} {name}</span>' for name, d in _DOMAINS.items())}
  </div>
</div>""", unsafe_allow_html=True)

# ── Run ────────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running Slot A…"):
        ra = _run_simulation(a_domain, a_scen_raw, a_biases, a_bias_int,
                             a_poison, int(a_samples), 42)
    with st.spinner("Running Slot B…"):
        rb = _run_simulation(b_domain, b_scen_raw, b_biases, b_bias_int,
                             b_poison, int(b_samples), 99)
    st.session_state.cmp_result_a = ra
    st.session_state.cmp_result_b = rb
    st.session_state.cmp_config_a = {
        "domain": a_domain, "scenario": a_scen_raw,
        "biases": a_biases, "bias_intensity": a_bias_int,
        "poison_rate": a_poison, "n_samples": int(a_samples)}
    st.session_state.cmp_config_b = {
        "domain": b_domain, "scenario": b_scen_raw,
        "biases": b_biases, "bias_intensity": b_bias_int,
        "poison_rate": b_poison, "n_samples": int(b_samples)}

res_a = st.session_state.cmp_result_a
res_b = st.session_state.cmp_result_b

if not res_a or not res_b:
    # Welcome state
    st.markdown("""
<div style="background:var(--bg1);border:1px solid var(--bg3);border-radius:12px;
  padding:2.5rem;text-align:center;margin-top:1rem">
  <p style="font-size:2.5rem;margin:0 0 .75rem"></p>
  <p style="font-family:var(--ff-d);font-size:1.3rem;font-weight:700;
     color:var(--t0);margin:0 0 .5rem">Configure &amp; Run a Comparison</p>
  <p style="color:var(--t2);font-size:.88rem;max-width:520px;margin:0 auto">
    Choose two domain/scenario/bias configurations in the sidebar,
    then click <strong style="color:var(--ac)">Run</strong>.
    Compare Healthcare vs Judicial, low bias vs high bias, or any combination across all 7 domains.
  </p>
</div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-label" style="margin-top:2rem">Quick-start ideas</p>',
                unsafe_allow_html=True)
    idea_cols = st.columns(3)
    IDEAS = [
        ("🏥 vs ⚖️", "Healthcare vs Judicial", "Does the same bias pattern cause the same fairness degradation in clinical vs criminal justice AI?"),
        ("💰 vs 🎓", "Financial vs Education", "Compare how socioeconomic bias compounds differently in credit scoring vs JAMB exam prediction."),
        ("📡 vs 🛡️", "Disinformation vs Security", "Both use classification on sensitive populations. Which domain takes the harder fairness hit?"),
        ("Low vs High Bias", "Same domain, 0.1 vs 0.4", "Quantify the exact accuracy-fairness cost of increasing bias intensity in a controlled experiment."),
        ("Nigeria vs Global", "Agrotech FCT vs Financial Global", "How does Nigeria-specific calibration change the equity outcomes vs a global baseline?"),
        ("Poison Attack", "Clean vs 15% poison rate", "Measure how data poisoning degrades fairness and accuracy differently from bias injection."),
    ]
    for col, (title, subtitle, desc) in zip(idea_cols * 2, IDEAS):
        col.markdown(f"""
<div style="background:var(--bg2);border:1px solid var(--bg3);border-radius:10px;
  padding:1rem 1.1rem;margin-bottom:.75rem">
  <p style="font-family:var(--ff-d);font-size:.9rem;font-weight:700;
     color:var(--t0);margin:0 0 .2rem">{title}</p>
  <p style="font-family:var(--ff-m);font-size:.68rem;color:var(--ac);
     margin:0 0 .4rem;letter-spacing:.04em">{subtitle}</p>
  <p style="font-size:.78rem;color:var(--t2);line-height:1.45;margin:0">{desc}</p>
</div>""", unsafe_allow_html=True)
    st.stop()

# ── Error check ────────────────────────────────────────────────────────────────
for slot, res in [("A", res_a), ("B", res_b)]:
    if isinstance(res, dict) and "error" in res:
        st.error(f"Slot {slot} error: {res['error']}")
        st.stop()

# ── Summary cards ──────────────────────────────────────────────────────────────
cfg_a = st.session_state.cmp_config_a
cfg_b = st.session_state.cmp_config_b
da    = _DOMAINS[res_a["domain"]]
db    = _DOMAINS[res_b["domain"]]

st.markdown('<p class="section-label fade-in">Simulation Results</p>',
            unsafe_allow_html=True)

ca, arrow_col, cb = st.columns([1, 0.12, 1])

def _summary_card(res, cfg, info, slot_label):
    comp = res.get("composite_score", 0)
    comp_col = "#39ff7a" if comp >= 0.70 else "#f5a623" if comp >= 0.55 else "#ff3b5c"
    pills = "".join(f'<span class="scenario-pill">{b}</span>'
                    for b in (cfg["biases"] or ["None"]))
    return f"""
<div class="summary-card">
  <div class="domain-badge" style="color:{info['accent']};
       border-color:{info['accent']}55;background:{info['accent']}11">
    {info['icon']} {slot_label}
  </div>
  <p style="font-family:var(--ff-d);font-size:1.1rem;font-weight:700;
     color:var(--t0);margin:.2rem 0 .1rem;letter-spacing:-.01em">
    {res['domain']} — {res.get('scenario','')[:32]}
  </p>
  <p style="font-family:var(--ff-m);font-size:.7rem;color:var(--t2);margin:0 0 .8rem">
    n={cfg['n_samples']:,} · bias={cfg['bias_intensity']:.2f} · poison={cfg['poison_rate']:.2f}
  </p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-bottom:.75rem">
    <div style="background:var(--bg3);border-radius:7px;padding:.6rem .8rem">
      <p style="font-family:var(--ff-m);font-size:.65rem;text-transform:uppercase;
         letter-spacing:.07em;color:var(--t2);margin:0">Accuracy</p>
      <p style="font-family:var(--ff-d);font-size:1.4rem;font-weight:700;
         color:{info['accent']};margin:0">{res['accuracy']:.1%}</p>
    </div>
    <div style="background:var(--bg3);border-radius:7px;padding:.6rem .8rem">
      <p style="font-family:var(--ff-m);font-size:.65rem;text-transform:uppercase;
         letter-spacing:.07em;color:var(--t2);margin:0">Fairness</p>
      <p style="font-family:var(--ff-d);font-size:1.4rem;font-weight:700;
         color:{info['accent']};margin:0">{res['fairness_score']:.3f}</p>
    </div>
    <div style="background:var(--bg3);border-radius:7px;padding:.6rem .8rem">
      <p style="font-family:var(--ff-m);font-size:.65rem;text-transform:uppercase;
         letter-spacing:.07em;color:var(--t2);margin:0">FPR</p>
      <p style="font-family:var(--ff-d);font-size:1.4rem;font-weight:700;
         color:{info['accent']};margin:0">{res['false_positive_rate']:.1%}</p>
    </div>
    <div style="background:var(--bg3);border-radius:7px;padding:.6rem .8rem">
      <p style="font-family:var(--ff-m);font-size:.65rem;text-transform:uppercase;
         letter-spacing:.07em;color:var(--t2);margin:0">Composite</p>
      <p style="font-family:var(--ff-d);font-size:1.4rem;font-weight:700;
         color:{comp_col};margin:0">{comp:.3f}</p>
    </div>
  </div>
  <div>Active biases: {pills}</div>
</div>"""

with ca:
    st.markdown(_summary_card(res_a, cfg_a, da, "🔵 Slot A"), unsafe_allow_html=True)
with arrow_col:
    st.markdown("""
<div style="display:flex;align-items:center;justify-content:center;height:100%;
  font-size:1.6rem;color:var(--t2);padding-top:3rem">VS</div>""", unsafe_allow_html=True)
with cb:
    st.markdown(_summary_card(res_b, cfg_b, db, "🔴 Slot B"), unsafe_allow_html=True)

# ── Overall winner ─────────────────────────────────────────────────────────────
comp_a = res_a.get("composite_score", 0)
comp_b = res_b.get("composite_score", 0)
diff   = comp_a - comp_b
if abs(diff) < 0.02:
    verdict = "🤝 Too close to call — effectively tied"
    vcls = "winner-tie"
elif diff > 0:
    verdict = f"🔵 Slot A wins — composite score advantage {diff:+.3f}"
    vcls = "winner-a"
else:
    verdict = f"🔴 Slot B wins — composite score advantage {abs(diff):+.3f}"
    vcls = "winner-b"

st.markdown(f"""
<div style="background:var(--bg2);border:1px solid var(--bg3);border-radius:10px;
  padding:.85rem 1.25rem;margin:.75rem 0;display:flex;align-items:center;gap:.75rem">
  <span class="{vcls}" style="font-size:.9rem">{verdict}</span>
  <span style="margin-left:auto;font-family:var(--ff-m);font-size:.68rem;color:var(--t2)">
    Composite = weighted accuracy + fairness + domain-specific penalty
  </span>
</div>""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visual Comparison",
    "⚖️ Metric Diff Table",
    "🎯 Domain-Specific Metrics",
    "🔧 Config Diff",
    "📤 Export",
])

# ── Tab 1: Visual Comparison ───────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-label">Core Metrics — Side by Side</p>',
                unsafe_allow_html=True)

    CORE_METRICS = ["accuracy","recall","precision","f1_score",
                    "fairness_score","demographic_parity",
                    "equalized_odds","false_positive_rate","composite_score"]
    METRIC_LABELS = ["Accuracy","Recall","Precision","F1","Fairness","Dem. Parity",
                     "Eq. Odds","FPR","Composite"]

    vals_a = [res_a.get(m, 0) for m in CORE_METRICS]
    vals_b = [res_b.get(m, 0) for m in CORE_METRICS]

    c1, c2 = st.columns(2)
    with c1:
        # Grouped bar
        fig = go.Figure()
        fig.add_trace(go.Bar(name=f"🔵 {res_a['domain']}", x=METRIC_LABELS, y=vals_a,
            marker_color=da["accent"], opacity=0.85))
        fig.add_trace(go.Bar(name=f"🔴 {res_b['domain']}", x=METRIC_LABELS, y=vals_b,
            marker_color=db["accent"], opacity=0.85))
        try:
            fig.update_layout(**_ptheme(ACCENT), barmode="group",
                title="All Core Metrics Compared", height=360,
                yaxis=dict(range=[0,1], **_ptheme(ACCENT).get("yaxis",{})))
        except Exception:
            fig.update_layout(barmode="group", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Radar
        fig_r = go.Figure()
        radar_metrics = ["accuracy","recall","fairness_score",
                         "specificity","composite_score","f1_score"]
        rl = ["Accuracy","Recall","Fairness","Specificity","Composite","F1"]
        rv_a = [res_a.get(m, 0) for m in radar_metrics]
        rv_b = [res_b.get(m, 0) for m in radar_metrics]
        for vals, name, color in [(rv_a+[rv_a[0]], f"🔵 {res_a['domain']}", da["accent"]),
                                   (rv_b+[rv_b[0]], f"🔴 {res_b['domain']}", db["accent"])]:
            h = color.lstrip("#")
            fc = f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},0.08)"
            fig_r.add_trace(go.Scatterpolar(r=vals, theta=rl+[rl[0]],
                fill="toself", name=name,
                line=dict(color=color, width=2),
                fillcolor=fc))
        try:
            pt = _ptheme(ACCENT)
            fig_r.update_layout(
                paper_bgcolor=pt["paper_bgcolor"],
                font=pt["font"],
                polar=dict(bgcolor="#090d13",
                    angularaxis=dict(linecolor="#1e2a3d", gridcolor="#1e2a3d",
                        tickfont=dict(color="#9aa8c4", size=10)),
                    radialaxis=dict(visible=True, range=[0,1],
                        gridcolor="#1e2a3d", linecolor="#1e2a3d",
                        tickfont=dict(color="#5a6a85", size=8))),
                height=360, title="Radar — Core Metrics",
                title_font=pt.get("title_font", {}),
                legend=pt.get("legend", {}),
                margin=dict(t=42, b=20, l=40, r=40),
            )
        except Exception:
            fig_r.update_layout(height=360)
        st.plotly_chart(fig_r, use_container_width=True)

    # Delta bars
    st.markdown('<p class="section-label">Delta Analysis (A − B)</p>',
                unsafe_allow_html=True)
    deltas = [res_a.get(m, 0) - res_b.get(m, 0) for m in CORE_METRICS]
    colors = []
    for m, d in zip(CORE_METRICS, deltas):
        if m in _LOWER_IS_BETTER:
            colors.append("#39ff7a" if d < 0 else "#ff3b5c")
        else:
            colors.append("#39ff7a" if d > 0 else "#ff3b5c")
    fig_d = go.Figure(go.Bar(x=METRIC_LABELS, y=deltas,
        marker_color=colors, text=[f"{d:+.3f}" for d in deltas],
        textposition="outside",
        textfont=dict(family="DM Mono, monospace", size=10)))
    try:
        fig_d.update_layout(**_ptheme(ACCENT), title="Δ A − B (green = A better)",
            height=300, yaxis=dict(zeroline=True, zerolinecolor="#1e2a3d",
            **_ptheme(ACCENT).get("yaxis",{})))
    except Exception:
        fig_d.update_layout(height=300)
    st.plotly_chart(fig_d, use_container_width=True)

# ── Tab 2: Metric Diff Table ───────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-label">Full Metric Comparison</p>',
                unsafe_allow_html=True)

    all_metrics = sorted(set(list(res_a.keys()) + list(res_b.keys())) -
                         {"domain","scenario","biases_applied","ecoa_compliant"})
    rows = []
    for m in all_metrics:
        va = res_a.get(m)
        vb = res_b.get(m)
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            continue
        delta = va - vb
        lib   = m in _LOWER_IS_BETTER
        winner = ("A" if (delta < -0.005 if lib else delta > 0.005)
                  else "B" if (delta > 0.005 if lib else delta < -0.005)
                  else "Tie")
        rows.append({
            "Metric": m.replace("_"," ").title(),
            f"A — {res_a['domain']}": f"{va:.4f}",
            f"B — {res_b['domain']}": f"{vb:.4f}",
            "Δ (A−B)": f"{delta:+.4f}",
            "Winner": winner,
        })

    df_diff = pd.DataFrame(rows)
    st.dataframe(df_diff, use_container_width=True, height=480,
                 column_config={
                     "Winner": st.column_config.TextColumn("Winner",
                         help="Which slot has the better value for this metric"),
                     "Δ (A−B)": st.column_config.TextColumn("Δ (A−B)"),
                 })

    # Summary
    wins_a = sum(1 for r in rows if r["Winner"]=="A")
    wins_b = sum(1 for r in rows if r["Winner"]=="B")
    ties   = sum(1 for r in rows if r["Winner"]=="Tie")
    w1, w2, w3 = st.columns(3)
    w1.metric(f"🔵 Slot A wins", wins_a)
    w2.metric(f"🔴 Slot B wins", wins_b)
    w3.metric("🤝 Ties", ties)

# ── Tab 3: Domain-Specific Metrics ─────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-label">Domain-Specific Metrics</p>',
                unsafe_allow_html=True)

    for slot_label, res, info in [("🔵 Slot A", res_a, da), ("🔴 Slot B", res_b, db)]:
        domain = res["domain"]
        accent = info["accent"]
        st.markdown(f"""
<div style="border-left:3px solid {accent};padding:.5rem .9rem;margin:.75rem 0;
  background:{accent}0d;border-radius:0 8px 8px 0">
  <p style="font-family:var(--ff-m);font-size:.7rem;letter-spacing:.08em;
     text-transform:uppercase;color:{accent};margin:0 0 .35rem">
    {slot_label} — {domain} · {res.get("scenario","")}
  </p>""", unsafe_allow_html=True)

        dmc = st.columns(4)
        domain_extras = {
            "National Security": [
                ("Liberty Score", "liberty_score"),
                ("Detection Rate", "detection_rate"),
                ("FPR",           "false_positive_rate"),
                ("Fairness",      "fairness_score"),
            ],
            "Financial": [
                ("DI Ratio",  "disparate_impact_ratio"),
                ("ECOA Pass", "ecoa_compliant"),
                ("Fairness",  "fairness_score"),
                ("Equity Gap","equity_gap"),
            ],
            "Judicial": [
                ("Racial FPR Gap", "racial_fpr_gap"),
                ("Liberty Score",  "liberty_score"),
                ("Fairness",       "fairness_score"),
                ("Composite",      "composite_score"),
            ],
            "Disinformation": [
                ("Language FPR Gap",  "language_fpr_gap"),
                ("Over-Removal Rate", "over_removal_rate"),
                ("Recall",            "recall"),
                ("Fairness",          "fairness_score"),
            ],
            "Education": [
                ("Opportunity Gap",  "opportunity_gap"),
                ("Dem. Parity",      "demographic_parity"),
                ("Fairness",         "fairness_score"),
                ("Accuracy",         "accuracy"),
            ],
            "Agrotech": [
                ("Gender Gap",   "gender_gap"),
                ("Eq. Odds Gap", "equalized_odds"),
                ("Fairness",     "fairness_score"),
                ("Accuracy",     "accuracy"),
            ],
            "Healthcare": [
                ("Sensitivity",  "recall"),
                ("Specificity",  "specificity"),
                ("Fairness",     "fairness_score"),
                ("Equity Gap",   "equity_gap"),
            ],
        }
        for col, (label, key) in zip(dmc, domain_extras.get(domain, [
                ("Accuracy","accuracy"),("Fairness","fairness_score"),
                ("FPR","false_positive_rate"),("Composite","composite_score")])):
            val = res.get(key, "—")
            if isinstance(val, bool):
                display = "✅ Yes" if val else "❌ No"
            elif isinstance(val, float):
                display = f"{val:.3f}"
            else:
                display = str(val)
            col.metric(label, display)
        st.markdown("</div>", unsafe_allow_html=True)

    # Side-by-side domain KPI comparison
    if res_a["domain"] == res_b["domain"]:
        st.markdown('<p class="section-label">Same-Domain Deep Comparison</p>',
                    unsafe_allow_html=True)
        domain = res_a["domain"]
        metrics_to_compare = domain_extras.get(domain,
            [("Accuracy","accuracy"),("Fairness","fairness_score"),
             ("FPR","false_positive_rate"),("Composite","composite_score")])
        fig_dd = go.Figure()
        for slot_label, res, color in [
            (f"🔵 A · bias={cfg_a['bias_intensity']}", res_a, da["accent"]),
            (f"🔴 B · bias={cfg_b['bias_intensity']}", res_b, db["accent"]),
        ]:
            vals = [float(res.get(k, 0)) if isinstance(res.get(k), (int,float,bool)) else 0
                    for _, k in metrics_to_compare]
            labels = [l for l,_ in metrics_to_compare]
            fig_dd.add_trace(go.Bar(name=slot_label, x=labels, y=vals,
                marker_color=color, opacity=0.85))
        try:
            fig_dd.update_layout(**_ptheme(ACCENT), barmode="group",
                title=f"{domain} — Domain KPI Comparison", height=320)
        except Exception:
            fig_dd.update_layout(barmode="group", height=320)
        st.plotly_chart(fig_dd, use_container_width=True)

# ── Tab 4: Config Diff ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-label">Configuration Comparison</p>',
                unsafe_allow_html=True)
    cfg_rows = [
        {"Parameter": "Domain",         "Slot A": cfg_a["domain"],            "Slot B": cfg_b["domain"]},
        {"Parameter": "Scenario",        "Slot A": str(cfg_a["scenario"])[:50],"Slot B": str(cfg_b["scenario"])[:50]},
        {"Parameter": "Bias Types",      "Slot A": ", ".join(cfg_a["biases"]) or "None",
                                          "Slot B": ", ".join(cfg_b["biases"]) or "None"},
        {"Parameter": "Bias Intensity",  "Slot A": f"{cfg_a['bias_intensity']:.2f}",
                                          "Slot B": f"{cfg_b['bias_intensity']:.2f}"},
        {"Parameter": "Poison Rate",     "Slot A": f"{cfg_a['poison_rate']:.3f}",
                                          "Slot B": f"{cfg_b['poison_rate']:.3f}"},
        {"Parameter": "Sample Size",     "Slot A": f"{cfg_a['n_samples']:,}",  "Slot B": f"{cfg_b['n_samples']:,}"},
        {"Parameter": "Composite Score", "Slot A": f"{res_a.get('composite_score',0):.4f}",
                                          "Slot B": f"{res_b.get('composite_score',0):.4f}"},
        {"Parameter": "Fairness Score",  "Slot A": f"{res_a.get('fairness_score',0):.4f}",
                                          "Slot B": f"{res_b.get('fairness_score',0):.4f}"},
    ]
    df_cfg = pd.DataFrame(cfg_rows)
    df_cfg["Match"] = df_cfg.apply(
        lambda r: "✅" if r["Slot A"] == r["Slot B"] else "⚠️", axis=1)
    st.dataframe(df_cfg, use_container_width=True, hide_index=True)

# ── Tab 5: Export ──────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<p class="section-label">Export Comparison Report</p>',
                unsafe_allow_html=True)

    e1, e2, e3 = st.columns(3)
    with e1:
        export = {
            "gags_version": "4.0",
            "comparison_timestamp": datetime.now().isoformat(),
            "slot_a": {"config": cfg_a, "results": res_a},
            "slot_b": {"config": cfg_b, "results": res_b},
            "winner": ("A" if comp_a > comp_b + 0.02
                       else "B" if comp_b > comp_a + 0.02 else "Tie"),
            "composite_delta": round(comp_a - comp_b, 4),
        }
        st.download_button("📥 Download JSON Report",
            json.dumps(export, indent=2, default=str),
            f"gags_comparison_{res_a['domain'].lower()}_vs_{res_b['domain'].lower()}.json",
            "application/json", use_container_width=True)

    with e2:
        df_export = pd.DataFrame([
            {"slot": "A", "domain": res_a["domain"], **{k:v for k,v in res_a.items()
             if isinstance(v,(int,float,str,bool))}},
            {"slot": "B", "domain": res_b["domain"], **{k:v for k,v in res_b.items()
             if isinstance(v,(int,float,str,bool))}},
        ])
        st.download_button("📊 Download CSV",
            df_export.to_csv(index=False).encode(),
            f"gags_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv", use_container_width=True)

    with e3:
        narrative = (
            f"GAGS Comparison Report — {datetime.now().strftime('%Y-%m-%d')}\n"
            f"{'='*55}\n\n"
            f"SLOT A: {cfg_a['domain']} · {cfg_a['scenario']}\n"
            f"  Bias: {cfg_a['bias_intensity']:.2f} | Poison: {cfg_a['poison_rate']:.3f}\n"
            f"  Accuracy: {res_a['accuracy']:.1%} | Fairness: {res_a['fairness_score']:.3f}\n"
            f"  Composite: {comp_a:.3f}\n\n"
            f"SLOT B: {cfg_b['domain']} · {cfg_b['scenario']}\n"
            f"  Bias: {cfg_b['bias_intensity']:.2f} | Poison: {cfg_b['poison_rate']:.3f}\n"
            f"  Accuracy: {res_b['accuracy']:.1%} | Fairness: {res_b['fairness_score']:.3f}\n"
            f"  Composite: {comp_b:.3f}\n\n"
            f"VERDICT: {verdict}\n"
        )
        st.download_button("📄 Download Text Report",
            narrative.encode(),
            "gags_comparison_report.txt", "text/plain",
            use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:var(--t2);font-family:var(--ff-m);"
    "font-size:.7rem;padding:.5rem 0;letter-spacing:.05em'>"
    "COMPARISON MODE · GAGS v4.0 · All 7 Domains · Configurable Diff Engine"
    "</div>", unsafe_allow_html=True)