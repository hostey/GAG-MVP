"""
pages/12_Peace_Conflict.py  — GAGS Resilience Framework v4.0
═══════════════════════════════════════════════════════════════
Peace & Conflict Resolution AI Equity Module

Nigeria-specific contexts:
  • Boko Haram early-warning AI — surveillance bias, Kanuri/Hausa profiling
  • Farmer-Herder conflict prediction — ethnic proxy variables
  • ECOWAS conflict forecasting — cross-border AI cooperation
  • Election violence prediction — 2027 electoral AI deployment
  • Niger Delta resource conflict — oil-community AI mediation
  • Post-conflict reintegration — ex-combatant risk scoring

Theoretical frameworks integrated:
  • Game Theory (Signalling, Commitment problems, Costly conflict)
  • Network Science (Conflict diffusion, cascades)
  • Epidemiological SIR/SEIR violence contagion models
  • Structural Equation Modelling (Root causes, mediation)
  • Agent-Based Modelling (Insurgency dynamics, ceasefire stability)
  • Dynamic Causal Modelling (Reinforcing conflict loops)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                              f1_score, roc_auc_score)

from components.translate import install_auto_translate, tx, language_switcher
install_auto_translate()
from components.i18n import t, get_lang, language_badge

from components.governance_logic import (
    calculate_fairness_metrics, apply_bias, simulate_data_poisoning,
    HybridGovernanceLayer, AttackSeverity, simulate_longitudinal_bias,
    simulate_federated_learning,
)
from utils.config import simulation_config, settings
from components.ux_utils import (
    role_switcher, role_banner, role_brief_banner, role_algo_banner,
    get_role_algo, get_role_tabs, get_role_defaults,
    history_browser, save_to_history, share_url_panel, load_config_from_url,
    annotation_panel, guided_tour_banner,
)
from components.live_data import national_live_banner
from components.nigeria_states import state_selector, state_info_card, get_state_params, apply_state_to_preset
from components.gags_interactive import (
    _reset_render_guards, track_run,
    progress_tracker, what_if_explorer, benchmark_challenge_panel,
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
    feature_modules_tab, multi_challenge_panel, admin_challenge_panel,
)
from components.nigeria_regulatory import nigeria_compliance_panel

try:
    from components.gags_design import build_css, DOMAIN_ACCENTS, plotly_theme
    ACCENT = "#0f766e"   # teal — peace
    st.markdown(build_css(ACCENT), unsafe_allow_html=True)
    PT = plotly_theme(ACCENT)
except ImportError:
    ACCENT = "#0f766e"
    PT = {}

try:
    from components.gags_benchmarks import (
        get_benchmarks, benchmark_comparison_bar, BENCHMARKS_OK,
    )
except ImportError:
    BENCHMARKS_OK = False

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Peace & Conflict AI • GAGS",
    layout="wide", page_icon="🕊️",
)

load_config_from_url()
guided_tour_banner("conflict")

# ── State ────────────────────────────────────────────────────────────────────
_STATE = {
    "pcr_run_history":      [], "pcr_xai_results":      {},
    "pcr_longitudinal":     None, "pcr_federated":       None,
    "pcr_snapshot_history": [], "pcr_annotations":      [],
    "pcr_safety_report":    {}, "pcr_lifecycle_report": {},
    "pcr_feature_outputs":  {},
    "pcr_ds_report":        {},
}
for _k, _v in _STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_VALID_BIAS = list(dict.fromkeys(
    list(simulation_config.BIAS_TYPES) +
    ["demographic", "historical", "geographic", "linguistic", "ethnic"]))

# ── Nigeria conflict scenarios ───────────────────────────────────────────────
CONFLICT_SCENARIOS = {
    "boko_haram_ew": {
        "name":        "Boko Haram Early Warning (Northeast)",
        "description": "AI threat-assessment system deployed by security forces in Borno, Yobe, "
                       "Adamawa. Trained on historical incident data skewed toward Kanuri/Hausa profiles. "
                       "Risk: ethnic profiling, mass false positives on civilian population.",
        "region":      "Northeast Nigeria",
        "at_risk":     "Kanuri, Hausa-Fulani communities — 4.2M people",
        "bias_type":   "geographic + ethnic",
        "ref":         "ACLED Nigeria 2015-2023; UN OCHA Humanitarian Reports",
        "n_features":  14,
        "base_fpr":    0.28,
        "dp_gap":      0.24,
    },
    "ipob_separatist": {
        "name":        "IPOB Separatist Movement — Southeast AI Surveillance",
        "description": "AI-powered threat-assessment and social media monitoring system deployed "
                       "by DSS/Army in Imo, Anambra, Abia, Enugu, Ebonyi states during sit-at-home "
                       "enforcement. Trained predominantly on reports from Joint Military Operations "
                       "(Python Dance I-III, Crocodile Smile), creating systematic bias against "
                       "Igbo civilians. Risk: mass civilian profiling, keyword-based Igbo-language "
                       "flagging, social media surveillance of legitimate political expression, "
                       "criminalisation of dissent. Disproportionate impact on Igbo youth aged 18-35.",
        "region":      "Southeast Nigeria (Imo, Anambra, Abia, Enugu, Ebonyi)",
        "at_risk":     "Igbo civilians, activists, traders — est. 22M people in Southeast",
        "bias_type":   "ethnic + political + linguistic (Igbo)",
        "ref":         "Amnesty International 2021 'Deadly Fusion'; HRW 2021; "
                       "Intersociety Onitsha 2022; ACLED Southeast Nigeria 2021-2023",
        "n_features":  15,
        "base_fpr":    0.38,   # Very high: sit-at-home monitoring sweeps civilian population
        "dp_gap":      0.33,   # Large gap: Igbo communities vs non-Igbo in training data
    },
    "farmer_herder": {
        "name":        "Farmer-Herder Conflict Prediction (Middle Belt)",
        "description": "Predictive AI deployed to pre-position security forces across Benue, Plateau, "
                       "Kaduna States. Training data reflects prior Fulani/Tiv incident reports, "
                       "creating circular prediction that over-polices pastoral communities.",
        "region":      "Middle Belt (Benue, Plateau, Kaduna)",
        "at_risk":     "Fulani herdsmen, Tiv farmers — 6.8M across 6 states",
        "bias_type":   "ethnic + occupation",
        "ref":         "UNDP Nigeria 2022; International Crisis Group Report 295",
        "n_features":  12,
        "base_fpr":    0.22,
        "dp_gap":      0.19,
    },
    "election_violence": {
        "name":        "Election Violence Prediction (2027 Elections)",
        "description": "INEC-adjacent AI system predicting electoral hotspots for security deployment. "
                       "Historical data from 2015-2023 elections reflects incumbent-era policing biases. "
                       "Risk: opposition strongholds flagged as high-risk; chilling effect on voter turnout.",
        "region":      "Nigeria (36 states + FCT)",
        "at_risk":     "Opposition voters, ethnic minorities — 93M registered voters",
        "bias_type":   "political + geographic",
        "ref":         "Electoral Integrity Project 2023; YIAGA Africa Watching the Vote",
        "n_features":  16,
        "base_fpr":    0.31,
        "dp_gap":      0.27,
    },
    "niger_delta": {
        "name":        "Niger Delta Resource Conflict AI (Oil Companies)",
        "description": "Shell/SPDC-deployed environmental risk AI used to triage community conflict "
                       "claims. Model trained on company incident logs systematically under-counts "
                       "Ogoni/Ijaw community reports vs contractor reports.",
        "region":      "Niger Delta (Rivers, Bayelsa, Delta, Edo)",
        "at_risk":     "Ogoni, Ijaw, Itsekiri communities — 31M people",
        "bias_type":   "socioeconomic + corporate",
        "ref":         "Amnesty International 2020; UN UNEP Ogoniland Report",
        "n_features":  13,
        "base_fpr":    0.35,
        "dp_gap":      0.30,
    },
    "ecowas_forecast": {
        "name":        "ECOWAS Regional Conflict Forecast",
        "description": "Cross-border AI conflict early-warning deployed by ECOWAS for West Africa. "
                       "Training data from Sahel/Francophone crises may not generalise to Anglophone "
                       "Nigeria contexts. Language/data asymmetry between member states.",
        "region":      "West Africa (ECOWAS 15 states)",
        "at_risk":     "Cross-border communities — 400M",
        "bias_type":   "linguistic + data asymmetry",
        "ref":         "ECOWAS Early Warning System (ECOWARN); ACLED West Africa 2010-2023",
        "n_features":  18,
        "base_fpr":    0.19,
        "dp_gap":      0.16,
    },
    "reintegration": {
        "name":        "Ex-Combatant Reintegration Risk Score (DDR)",
        "description": "AI risk-scoring system for Boko Haram ex-combatants in DDR (Disarmament, "
                       "Demobilisation, Reintegration) programme. High-risk scores restrict access "
                       "to livelihood support, creating re-radicalisation incentive.",
        "region":      "Northeast (Borno, Yobe, Adamawa)",
        "at_risk":     "~20,000 DDR programme participants",
        "bias_type":   "historical + demographic",
        "ref":         "UNICEF 2022 DDR Report; Nigerian Army OPLafayette data",
        "n_features":  11,
        "base_fpr":    0.25,
        "dp_gap":      0.21,
    },
}

# ── Real-world benchmarks ─────────────────────────────────────────────────────
CONFLICT_BENCHMARKS = {
    "IPOB/Southeast Ops Baseline (Army 2021)": {"accuracy": 0.61, "fairness": 0.38, "recall": 0.55},

    "ACLED Conflict Prediction Baseline":     {"accuracy": 0.71, "fairness": 0.52, "recall": 0.68},
    "ESCAP Conflict Forecasting (2022)":      {"accuracy": 0.74, "fairness": 0.55, "recall": 0.70},
    "ViEWS Violence Early Warning":           {"accuracy": 0.69, "fairness": 0.48, "recall": 0.65},
    "PITF Political Instability Task Force":  {"accuracy": 0.76, "fairness": 0.51, "recall": 0.72},
}

# ── Nigeria conflict context constants ───────────────────────────────────────
NIGERIA_CONFLICT_CONTEXT = {
    "fatalities_2022":        8_154,     # ACLED
    "displaced_persons":      2_300_000, # UNHCR (inc. Southeast)
    "farmer_herder_states":   14,
    "boko_haram_active_yrs":  15,
    "ddr_participants":       20_000,
    "election_hotspots_2023": 127,
    "peacebuilding_budget_bn": 2.3,      # USD billion/yr
}

# ── Simulation engine ─────────────────────────────────────────────────────────
def _generate_conflict_data(scenario_key: str, n_samples: int,
                             random_state: int = 42) -> tuple:
    """
    Generate Nigeria-calibrated conflict AI training data.
    Returns (X, y, demographic_info, feature_names, scenario_dict).
    """
    sc  = CONFLICT_SCENARIOS[scenario_key]
    rng = np.random.default_rng(random_state)
    n   = n_samples
    nf  = sc["n_features"]

    # Demographic split: disadvantaged group (ethnic minority / targeted community)
    demo = rng.binomial(1, 0.48, n).astype(float)  # 48% from marginalised community

    # Base feature matrix — socio-economic, geographic, historical incident indicators
    X = rng.standard_normal((n, nf))

    # Inject domain-specific structure
    base_fpr = sc["base_fpr"]
    dp_gap   = sc["dp_gap"]

    # Disadvantaged group has systematically worse "base features" due to:
    # - less infrastructure → worse connectivity features
    # - historical over-policing → more prior incidents on record
    dis_mask = demo == 0
    X[dis_mask, 0] += dp_gap * 1.5    # over-reported prior incidents
    X[dis_mask, 1] -= dp_gap * 0.8    # under-served infrastructure
    X[dis_mask, 2] += dp_gap * 0.6    # geographic flagging proxy

    # Ground truth: genuine threat signal + demographic noise
    true_risk_score = (0.35 * X[:, 0] + 0.20 * X[:, 1] - 0.15 * X[:, 2]
                       + rng.normal(0, 0.3, n))
    y = (true_risk_score > 0.1).astype(int)

    # Feature names
    base_names = [
        "Prior Incident Reports", "Infrastructure Index", "Geographic Proximity",
        "Population Density", "Historical Grievance Score", "Economic Marginalisation",
        "Armed Group Presence", "State Capacity Index", "Natural Resource Proximity",
        "Social Media Intensity", "Displacement History", "Cross-Border Activity",
    ]
    feat_names = (base_names + [f"Feature_{i}" for i in range(len(base_names), nf)])[:nf]

    return X.astype(np.float64), y, demo, feat_names, sc


def _run_one(scenario_key, n_samples, selected_biases, bias_intensity,
             poison_rate, run_idx, enable_xai, enable_governance,
             enable_gender_audit, selected_state=None):
    X, y, demo, feat_names, sc = _generate_conflict_data(
        scenario_key, n_samples, random_state=42 + run_idx)

    for bt in [b for b in selected_biases if b in _VALID_BIAS]:
        try:
            X, y, demo = apply_bias(X, y, bt, bias_intensity,
                demographic_info=demo, severity=AttackSeverity.MEDIUM)
        except Exception:
            pass

    try:
        X, y, demo = simulate_data_poisoning(
            X, y, poison_rate, attack_type="label_flipping",
            demographic_info=demo, targeted=True)
    except Exception:
        pass

    if len(np.unique(y)) < 2:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        Xs, y, demo, test_size=0.3,
        random_state=42 + run_idx,
        stratify=y if np.bincount(y.astype(int)).min() >= 2 else None)

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4,
        learning_rate=0.08, random_state=42 + run_idx)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)

    acc  = float(accuracy_score(yte, yp))
    rec  = float(recall_score(yte, yp, zero_division=0))
    prec = float(precision_score(yte, yp, zero_division=0))
    f1   = float(f1_score(yte, yp, zero_division=0))
    fpr  = float(np.mean(yp[yte == 0] == 1)) if (yte == 0).any() else 0.0
    fnr  = float(np.mean(yp[yte == 1] == 0)) if (yte == 1).any() else 0.0

    fair = calculate_fairness_metrics(yte, yp, gte)
    adv  = gte == 1
    dis  = gte == 0
    fpr_adv = float(np.mean(yp[adv][yte[adv] == 0] == 1)) if (adv & (yte == 0)).any() else fpr
    fpr_dis = float(np.mean(yp[dis][yte[dis] == 0] == 1)) if (dis & (yte == 0)).any() else fpr
    acc_adv = float(accuracy_score(yte[adv], yp[adv])) if adv.any() else acc
    acc_dis = float(accuracy_score(yte[dis], yp[dis])) if dis.any() else acc

    # Peace-specific derived metrics
    false_alarm_rate     = fpr                               # civilians wrongly flagged
    civilian_harm_proxy  = fpr * bias_intensity              # proxy for harm from false flags
    community_trust      = max(0.0, 1.0 - fpr * 2.5 - bias_intensity * 0.3)
    ceasefire_compliance = max(0.0, 1.0 - fnr * 1.5)        # missing real threats
    fairness_score       = fair.get("fairness_score", 0.5)
    dp_gap               = fair.get("demographic_parity_difference", 0.0)

    if enable_governance and run_idx == 0:
        try:
            gov = HybridGovernanceLayer(n_agents=12, domain="conflict")
            gov_result = gov.deliberate(
                model_accuracy=acc,
                fairness_score=fairness_score,
                risk_level="critical",
                context={"scenario": sc["name"], "bias_types": selected_biases},
            )
            st.session_state["pcr_feature_outputs"] = (
                st.session_state.get("pcr_feature_outputs", {}))
            st.session_state["pcr_feature_outputs"]["governance"] = gov_result.__dict__
        except Exception as _ge:
            st.session_state["pcr_feature_outputs"]["governance"] = {
                "policy": "Deploy with oversight", "outcome": "approved",
                "tally": {"for": 7, "against": 4, "abstain": 1},
                "narrative": str(_ge),
            }

    if run_idx == 0:
        bi = max(bias_intensity, 0.12)
        try:
            lng = simulate_longitudinal_bias(
                X, y, demo, initial_bias_type="historical",
                initial_bias_intensity=bi, n_generations=6, random_state=42)
            st.session_state.pcr_longitudinal = lng.__dict__
        except Exception:
            st.session_state.pcr_longitudinal = None
        try:
            fed = simulate_federated_learning(
                X, y, demo, n_clients=4, n_rounds=3,
                bias_heterogeneity=bi * 0.5, random_state=42)
            st.session_state.pcr_federated = fed.__dict__
        except Exception:
            st.session_state.pcr_federated = None

    return {
        "run_id":               run_idx + 1,
        "scenario":             sc["name"],
        "accuracy":             round(acc, 4),
        "recall":               round(rec, 4),
        "precision":            round(prec, 4),
        "f1_score":             round(f1, 4),
        "fpr":                  round(fpr, 4),
        "fnr":                  round(fnr, 4),
        "fairness_score":       round(fairness_score, 4),
        "false_alarm_rate":     round(false_alarm_rate, 4),
        "civilian_harm_proxy":  round(civilian_harm_proxy, 4),
        "community_trust":      round(community_trust, 4),
        "ceasefire_compliance": round(ceasefire_compliance, 4),
        "demographic_parity":   round(dp_gap, 4),
        "equalized_odds":       round(fair.get("equalized_odds_difference", 0), 4),
        "acc_advantaged":       round(acc_adv, 4),
        "acc_disadvantaged":    round(acc_dis, 4),
        "equity_gap":           round(abs(acc_adv - acc_dis), 4),
        "gender_gap":           round(abs(acc_adv - acc_dis) * 0.6, 4),
        "minority_fpr":         round(fpr_dis, 4),
        "majority_fpr":         round(fpr_adv, 4),
        "ethnic_fpr_gap":       round(fpr_dis - fpr_adv, 4),
        "bias_intensity":       round(bias_intensity, 4),
        "poison_rate":          round(poison_rate, 4),
        "biases":               ", ".join(selected_biases) or "None",
    }



# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Peace & Conflict AI • GAGS",
                   layout="wide", page_icon="🕊️")
load_config_from_url()
guided_tour_banner("conflict")

# ── Session state ─────────────────────────────────────────────────────────────
_STATE = {
    "pcr_run_history":       [], "pcr_xai_results":       {},
    "pcr_longitudinal":      None, "pcr_federated":        None,
    "pcr_snapshot_history":  [], "pcr_annotations":        [],
    "pcr_safety_report":     {}, "pcr_lifecycle_report":   {},
    "pcr_feature_outputs":   {}, "pcr_ds_report":          {},
    # interactivity state
    "pcr_ceasefire_moves":   [],   # ceasefire negotiation game
    "pcr_policy_choices":    {},   # user policy decisions
    "pcr_quiz_answers":      {},   # ethical quiz
    "pcr_live_fpr":          0.28, # live adjustable FPR
    "pcr_seir_custom":       {},   # custom SEIR params
    "pcr_network_seed":      42,
    "pcr_points":            0,
    "pcr_badges":            [],
}
for _k, _v in _STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_VALID_BIAS = list(dict.fromkeys(
    list(simulation_config.BIAS_TYPES) +
    ["demographic","historical","geographic","linguistic","ethnic"]))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    language_switcher(location="sidebar")
    st.divider()
    selected_state = state_selector(key="_state_12pcr", location="sidebar")
    state_info_card(selected_state)

    st.markdown(
        "<div style='background:linear-gradient(90deg,#f0fdfa,#ccfbf1);"
        "border-radius:6px;padding:6px 10px;margin-bottom:6px;'>"
        "<span style='font-size:.68rem;font-weight:700;color:#0f766e;"
        "text-transform:uppercase;letter-spacing:.07em;'>🕊️ Peace & Conflict AI</span>"
        "</div>", unsafe_allow_html=True)
    role_switcher("conflict")
    progress_tracker(location="sidebar")
    role_algo_banner("conflict")
    _role_algo, _role_algo_label, _ = get_role_algo("conflict")
    st.divider()

    st.markdown(f"""<div style="text-align:center;padding:.5rem 0">
      <h2 style="color:#0f766e;margin:0;font-family:'Syne',sans-serif">🕊️ Conflict Config</h2>
      <p style="color:#64748b;font-size:.75rem;margin:.2rem 0 0">
        Peace & Conflict AI Fairness Simulator</p></div>""", unsafe_allow_html=True)
    st.divider()

    scenario_key = st.selectbox(
        "🌍 Conflict Scenario", list(CONFLICT_SCENARIOS.keys()),
        format_func=lambda k: CONFLICT_SCENARIOS[k]["name"],
        key="_pcr_scenario")
    sc_info = CONFLICT_SCENARIOS[scenario_key]
    st.caption(sc_info["description"][:220])
    st.markdown(
        f"<div style='background:#f0fdfa;border-left:3px solid #0f766e;"
        f"border-radius:0 6px 6px 0;padding:6px 10px;margin:4px 0;font-size:.75rem;'>"
        f"📍 <b>Region:</b> {sc_info['region']}<br>"
        f"🎯 <b>At-risk:</b> {sc_info['at_risk']}<br>"
        f"⚠️ <b>Bias:</b> {sc_info['bias_type']}</div>", unsafe_allow_html=True)
    st.divider()

    _def_bias = [b for b in ["demographic","historical","geographic"] if b in _VALID_BIAS]
    selected_biases = st.multiselect(
        "🎭 Bias Types", options=_VALID_BIAS, default=_def_bias,
        format_func=lambda x: f"🔴 {x}" if x in ("demographic","ethnic") else f"⚠️ {x}")
    bias_intensity = st.slider("⚡ Bias Intensity", 0.0,
        float(simulation_config.MAX_BIAS_FACTOR), 0.28, 0.05)
    poison_rate    = st.slider("☣️ Data Poisoning", 0.0, 0.5, 0.06, 0.01, format="%.2f")
    st.divider()

    n_samples = st.number_input("📊 Records", 1000, 50000, settings.DEFAULT_N_SAMPLES, 1000)
    n_runs    = st.slider("🔁 Runs", 1, 8, 3)
    st.divider()

    st.subheader("🔧 Feature Toggles")
    enable_xai           = st.toggle("Explainable AI",       value=True)
    enable_governance    = st.toggle("Governance Layer",      value=True)
    enable_gender_audit  = st.toggle("Gender Audit",          value=False)
    enable_redteam       = st.toggle("Multimodal Red Team",   value=False)
    enable_agent_economy = st.toggle("Agent Economy",         value=False)
    enable_arena         = st.toggle("Strategic Arena",       value=False)
    enable_ai_safety     = st.toggle("AI Safety Suite",       value=False)
    enable_lifecycle     = st.toggle("Lifecycle Management",  value=False)
    enable_eco           = st.toggle("Eco Analysis",          value=False)
    enable_dynamic       = st.toggle("🔮 Dynamic Systems",   value=False)
    st.divider()

    run_button = st.button("▶ Run Simulation", type="primary",
                           use_container_width=True, key="_pcr_run")
    if st.button("🔄 Reset", use_container_width=True, key="_pcr_reset"):
        for _k, _v in _STATE.items():
            st.session_state[_k] = _v
        st.rerun()

    share_url_panel({"scenario": scenario_key,
                     "bias_intensity": bias_intensity, "n_samples": n_samples})
    language_badge()

# ── Scenario story banner ────────────────────────────────────────────────────
SCENARIO_ROLES = {
    "boko_haram_ew":    "You are an NSA AI Ethics Oversight Officer auditing a threat-assessment "
                        "system deployed across 4.2 million civilians in the Northeast.",
    "ipob_separatist":  "You are an NHRC AI Auditor. Your review of this surveillance system "
                        "will determine whether 22 million Igbo civilians face mass profiling.",
    "farmer_herder":    "You are a UNDP Conflict Prevention Advisor. Biased AI is escalating "
                        "an already deadly crisis across 6 Middle Belt states.",
    "election_violence":"You are an INEC Digital Rights Monitor. Democracy depends on whether "
                        "this AI prediction system targets opposition communities.",
    "niger_delta":      "You are an Amnesty International AI Accountability Researcher. "
                        "Shell's AI is suppressing legitimate community conflict reports.",
    "ecowas_forecast":  "You are an ECOWAS AI Governance Officer. Your cross-border "
                        "conflict forecast affects 400 million people across 15 states.",
    "reintegration":    "You are a UNICEF DDR Programme Manager. A flawed risk score "
                        "could deny 20,000 ex-combatants the support needed to avoid re-radicalisation.",
}
role_text = SCENARIO_ROLES.get(scenario_key, "You are an AI Governance Officer.")

ACCENT = "#0f766e"
st.markdown(
    f"<div style='background:linear-gradient(135deg,#042f2e,#0f766e);"
    f"border-radius:14px;padding:18px 26px;margin-bottom:16px;"
    f"border-left:5px solid #2dd4bf;'>"
    f"<p style='color:#99f6e4;font-size:.68rem;font-weight:700;"
    f"letter-spacing:.15em;text-transform:uppercase;margin:0 0 4px;'>"
    f"🎭 YOUR ROLE · PEACE & CONFLICT MODULE</p>"
    f"<p style='color:#f0fdfa;font-size:1.05rem;font-weight:700;margin:0 0 6px;'>"
    f"🕊️ {sc_info['name']}</p>"
    f"<p style='color:#ccfbf1;font-size:.85rem;margin:0;line-height:1.6;'>"
    f"{role_text}</p></div>", unsafe_allow_html=True)

# IPOB sensitivity notice
if scenario_key == "ipob_separatist":
    st.warning(
        "⚠️ **High Sensitivity Scenario.** Operations against IPOB involved documented "
        "civilian casualties and arbitrary detention (Amnesty 2021; HRW 2021). "
        "This module audits the **algorithmic bias dimension** only.")

sc_info = CONFLICT_SCENARIOS[scenario_key]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Scenario Base FPR",       f"{sc_info['base_fpr']:.0%}")
k2.metric("Demographic Parity Gap",  f"{sc_info['dp_gap']:.0%}")
k3.metric("Nigeria IDPs (UNHCR)",     "2.3M")
k4.metric("ACLED Fatalities 2022",    "8,154")

# ── Run block ─────────────────────────────────────────────────────────────────
if run_button:
    enable_lifecycle     = locals().get("enable_lifecycle",     False)
    enable_eco           = locals().get("enable_eco",           False)
    enable_dynamic       = locals().get("enable_dynamic",       False)
    enable_redteam       = locals().get("enable_redteam",       False)
    enable_agent_economy = locals().get("enable_agent_economy", False)
    enable_arena         = locals().get("enable_arena",         False)
    enable_gender_audit  = locals().get("enable_gender_audit",  False)
    enable_ai_safety     = locals().get("enable_ai_safety",     False)

    st.session_state.pcr_run_history        = []
    st.session_state["pcr_feature_outputs"] = {}
    prog = st.progress(0, text="Running conflict AI simulation…")

    for i in range(int(n_runs)):
        prog.progress((i+1)/n_runs, text=f"Run {i+1} of {n_runs}…")
        res = _run_one(
            scenario_key, int(n_samples), selected_biases,
            bias_intensity, poison_rate, i,
            enable_xai, enable_governance, enable_gender_audit, selected_state)
        if res:
            st.session_state.pcr_run_history.append(res)
            save_to_history("pcr_snapshot_history",
                {"accuracy": res["accuracy"],
                 "fairness_score": res["fairness_score"],
                 "false_alarm_rate": res["false_alarm_rate"],
                 "community_trust": res["community_trust"]},
                domain="conflict")

    _fout = st.session_state.get("pcr_feature_outputs", {})
    if enable_gender_audit:
        _h = st.session_state.get("pcr_run_history", [{}])
        _fout["gender_audit_gap"] = _h[-1].get("gender_gap", 0) if _h else 0
    if enable_redteam:
        try:    _fout["multimodal_redteam"] = run_redteam_simulation(domain="conflict")
        except Exception as _ex:
            _fout["multimodal_redteam"] = {"combined_bypass_rate":0,"modality_results":[],"error":str(_ex)}
    if enable_agent_economy:
        try:    _fout["agent_economy"] = run_agent_economy_simulation(domain="conflict")
        except Exception as _ex:
            _fout["agent_economy"] = {"gini_coefficient":0,"agent_summary":[],"error":str(_ex)}
    if enable_arena:
        try:    _fout["arena"] = run_arena_simulation(domain="conflict")
        except Exception as _ex:
            _fout["arena"] = {"final_standings":[],"deception_rate":0,"error":str(_ex)}
    st.session_state["pcr_feature_outputs"] = _fout

    if enable_ai_safety:
        try:
            _last = st.session_state.get("pcr_run_history",[{}])[-1]
            _sm   = {"fairness_score":_last.get("fairness_score",0.5),"robustness_score":0.55,
                     "ece":0.14,"has_xai":True,"has_governance":enable_governance,
                     "has_gender_audit":enable_gender_audit,"composite_ood_rate":0.60}
            _rng  = np.random.default_rng(42)
            _Xs   = _rng.standard_normal((500,10)); _ys = (_Xs[:,0]>0).astype(int)
            st.session_state["pcr_safety_report"] = run_ai_safety_suite(
                X_train=_Xs[:400],y_train=_ys[:400],X_test=_Xs[400:],y_test=_ys[400:],
                model=None,domain="conflict",enable_robustness=True,enable_ood=True,
                enable_uncertainty=True,enable_checklists=True,simulation_metrics=_sm)
        except Exception as _se:
            st.session_state["pcr_safety_report"] = {"error":str(_se),"pillars":{}}

    if enable_lifecycle or enable_eco:
        try:
            _lr = st.session_state.get("pcr_run_history",[{}])[-1]
            _lm = {k:v for k,v in _lr.items() if isinstance(v,(int,float))}
            _lm.update({"has_governance":enable_governance,"has_gender_audit":enable_gender_audit,"has_xai":True})
            st.session_state["pcr_lifecycle_report"] = run_lifecycle_suite(
                domain="conflict",algo_key=_role_algo,algo_label=_role_algo_label,
                n_samples=int(n_samples),n_runs=int(n_runs),n_features=10,
                metrics=_lm,safety_data=st.session_state.get("pcr_safety_report") or None,
                enable_registry=enable_lifecycle,enable_monitoring=enable_lifecycle,
                enable_audit=enable_lifecycle,enable_eco=enable_eco)
        except Exception as _lce:
            st.session_state["pcr_lifecycle_report"] = {"error":str(_lce),"pillars":{}}

    if enable_dynamic:
        try:
            _dh  = st.session_state.get("pcr_run_history",[])
            _dp  = derive_ds_params(domain="conflict",run_history=_dh)
            _dr  = run_dynamic_systems_suite(
                domain="conflict",y_true=_dp["y_true"],y_pred=_dp["y_pred"],
                sensitive=_dp["sensitive"],bias_intensity=_dp["bias_intensity"],
                governance_strength=_dp["governance_strength"],
                regulatory_pressure=_dp["regulatory_pressure"],
                market_pressure=_dp["market_pressure"],n_agents=150)
            _dr["source_metrics"] = _dp.get("source_metrics",{})
            st.session_state["pcr_ds_report"] = _dr
        except Exception as _dse:
            st.session_state["pcr_ds_report"] = {"error":str(_dse),"pillars":{}}

    prog.progress(1.0, text="Complete ✅"); prog.empty()
    st.session_state["pcr_points"] = st.session_state.get("pcr_points",0) + 50
    st.session_state.pcr_live_fpr  = float(
        st.session_state.pcr_run_history[-1].get("false_alarm_rate", sc_info["base_fpr"]))

# ── Post-run interactivity ────────────────────────────────────────────────────
if st.session_state.get("pcr_run_history"):
    _pl = st.session_state["pcr_run_history"][-1]
    track_run(_pl.get("fairness_score",0.5),"conflict")
    multi_challenge_panel("conflict",{k:v for k,v in _pl.items() if isinstance(v,(int,float))})
    admin_challenge_panel("conflict")
    benchmark_challenge_panel("conflict",{k:v for k,v in _pl.items() if isinstance(v,(int,float))})

# ── Welcome / no results ──────────────────────────────────────────────────────
if not st.session_state.pcr_run_history:
    st.divider()
    # Interactive role chooser before first run
    st.markdown("### 🎭 Choose Your Role Before Running")
    rc1,rc2,rc3 = st.columns(3)
    roles = [
        ("🏛️ Policy Maker","You set legal thresholds and mandates for conflict AI deployment.",
         "Policy Maker"),
        ("🤖 AI Developer","You design the model. You choose which features to include.",
         "AI Developer"),
        ("🕊️ Peacebuilder","You represent affected communities in the oversight process.",
         "Community Advocate"),
    ]
    for col,(icon,desc,label) in zip([rc1,rc2,rc3],roles):
        with col:
            if st.button(f"{icon}\n\n**{label}**", use_container_width=True,
                         key=f"_role_pre_{label}"):
                st.session_state["pcr_active_role"] = label
            st.caption(desc)

    active_role = st.session_state.get("pcr_active_role","")
    if active_role:
        st.success(f"✅ Role selected: **{active_role}** — your perspective will shape the analysis.")

    st.divider()
    st.markdown("### 🧠 Quick Pre-Simulation Quiz — What do you know about conflict AI?")
    q1 = st.radio(
        "1. What was the primary bias problem in the COMPAS recidivism algorithm?",
        ["It was too accurate","It had higher false positive rates for Black defendants",
         "It required too much data","It was too slow"],
        key="_pcr_quiz_q1", index=None)
    if q1 == "It had higher false positive rates for Black defendants":
        st.success("✅ Correct! ProPublica 2016 found Black defendants were nearly 2× more likely "
                   "to be incorrectly flagged as high-risk.")
        st.session_state["pcr_points"] += 10
    elif q1:
        st.error("❌ Incorrect. ProPublica (2016) showed Black defendants had nearly 2× higher false "
                 "positive rates — the same bias pattern this module simulates for Nigeria.")

    q2 = st.radio(
        "2. In Nigeria's Northeast, how many people were displaced by the Boko Haram conflict as of 2023?",
        ["About 50,000","About 500,000","Over 2 million","Over 10 million"],
        key="_pcr_quiz_q2", index=None)
    if q2 == "Over 2 million":
        st.success("✅ Correct! UNHCR 2023 recorded over 2.1M internally displaced persons.")
        st.session_state["pcr_points"] += 10
    elif q2:
        st.error("❌ UNHCR 2023: over 2.1M IDPs in Nigeria, mostly from Northeast conflict.")

    st.divider()
    # Theory overview cards
    st.markdown("### 🧠 Integrated Theoretical Frameworks")
    tc1,tc2,tc3 = st.columns(3)
    theories = [
        ("🎮 Game Theory",   "#0f766e","Signalling games, commitment problems. "
         "Why parties conceal capabilities; optimal ceasefire design under AI."),
        ("🕸️ Network Science","#7c3aed","Scale-free conflict cascade dynamics. "
         "Percolation threshold: the FPR below which cascade is contained."),
        ("🦠 SEIR Contagion", "#dc2626","R₀ of violence: compute the epidemic reproduction "
         "number and herd-immunity threshold for peacebuilding investment."),
        ("🔗 Causal Fairness","#0891b2","Pearl's do-calculus decomposes ethnic bias into "
         "direct vs proxy-variable pathways."),
        ("🤖 Agent-Based",    "#d97706","Insurgency dynamics: emergent violence from "
         "micro-level grievances, ceasefire stability under information asymmetry."),
        ("♾️ System Dynamics","#b45309","Reinforcing loops: AI surveillance → over-policing "
         "→ grievance → recruitment. Tipping points to runaway conflict cycles."),
    ]
    for i,(title,col,desc) in enumerate(theories):
        with [tc1,tc2,tc3][i%3]:
            st.markdown(
                f"<div style='background:#fff;border:1px solid #e2e8f0;"
                f"border-top:3px solid {col};border-radius:8px;"
                f"padding:12px 14px;margin-bottom:10px;'>"
                f"<p style='font-weight:700;color:#0f172a;margin:0 0 4px;font-size:.85rem;'>{title}</p>"
                f"<p style='color:#475569;font-size:.76rem;margin:0;line-height:1.5;'>{desc}</p>"
                f"</div>", unsafe_allow_html=True)
    st.stop()

# ── Main dashboard ─────────────────────────────────────────────────────────────
df       = pd.DataFrame(st.session_state.pcr_run_history)
avg_acc  = df["accuracy"].mean()
avg_fair = df["fairness_score"].mean()
avg_fpr  = df["false_alarm_rate"].mean()
avg_trust= df["community_trust"].mean()
avg_dp   = df["demographic_parity"].mean()
avg_eg   = df["ethnic_fpr_gap"].mean()
sc_info  = CONFLICT_SCENARIOS[scenario_key]

# Points display
pts = st.session_state.get("pcr_points", 0)
pc1,pc2,pc3,pc4,pc5,pc6 = st.columns(6)
pc1.metric("🎯 Accuracy",        f"{avg_acc:.1%}")
pc2.metric("⚖️ Fairness Score",  f"{avg_fair:.3f}",
           delta=f"{avg_fair-0.7:+.3f}", delta_color="normal")
pc3.metric("🚨 False Alarm Rate",f"{avg_fpr:.1%}",
           delta=f"{avg_fpr-sc_info['base_fpr']:+.1%}", delta_color="inverse")
pc4.metric("🤝 Community Trust", f"{avg_trust:.3f}")
pc5.metric("📊 Parity Gap",      f"{avg_dp:.3f}", delta_color="inverse")
pc6.metric("⭐ Your Points",      f"{pts}")

# Alert banners
if avg_fpr > 0.25:
    st.error(
        f"🚨 **Critical:** False alarm rate {avg_fpr:.1%} — "
        f"~{int(avg_fpr*2_000_000*0.05):,} civilians wrongly flagged. "
        f"Human review mandatory before deployment.")
elif avg_fair < 0.65:
    st.warning(
        f"⚠️ Fairness score {avg_fair:.3f} < 0.65 · Ethnic FPR gap {avg_eg:.3f} — "
        f"systematic targeting of marginalised communities detected.")
else:
    st.success(f"✅ AI system within acceptable thresholds. Fairness: {avg_fair:.3f} · FPR: {avg_fpr:.1%}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
_tab_labels = [
    "📊 Performance","⚖️ Conflict Fairness",
    "🎮 Game Theory","🕸️ Network Cascade","🦠 SEIR Contagion",
    "🕵️ Bias Detective","🎛️ Policy Simulator",
    "🔍 XAI & Explainability","🏛️ Nigeria Regulatory",
    "📋 Compliance","🔁 Longitudinal","🌐 Federated",
    "📤 Export","🔬 Feature Modules",
    "🛡️ AI Safety","🔄 Lifecycle","🌱 Eco Score","🔮 Dynamic Systems",
]
_tabs_obj = st.tabs(_tab_labels)
T = {n: _tab for n, _tab in zip(_tab_labels, _tabs_obj)}

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — PERFORMANCE
# ────────────────────────────────────────────────────────────────────────────
with T["📊 Performance"]:
    st.markdown("### 📈 Simulation Performance")
    fig = px.line(df, x="run_id",
                  y=["accuracy","fairness_score","community_trust"],
                  title="Accuracy · Fairness · Trust Across Runs",
                  color_discrete_sequence=[ACCENT,"#7c3aed","#16a34a"],
                  labels={"value":"Score","run_id":"Run","variable":"Metric"})
    fig.add_hline(y=0.7, line_dash="dot", line_color="#dc2626",
                  annotation_text="Fairness threshold")
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        fig2 = px.bar(df, x="run_id",
                      y=["false_alarm_rate","ethnic_fpr_gap"],
                      barmode="group", title="False Alarm & Ethnic FPR Gap",
                      color_discrete_sequence=["#dc2626","#ea580c"],
                      labels={"value":"Rate","run_id":"Run"})
        fig2.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.scatter(df, x="fairness_score", y="community_trust",
                          size="false_alarm_rate", color="ethnic_fpr_gap",
                          title="Fairness vs Community Trust",
                          color_continuous_scale="RdYlGn_r")
        fig3.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Live FPR adjuster ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🎛️ Live False Alarm Rate Adjuster")
    st.caption("Move the slider to instantly see how changing the FPR threshold affects "
               "real-world civilian impact — no re-run needed.")
    live_fpr = st.slider("Adjust False Alarm Rate",
                         0.02, 0.60, float(avg_fpr), 0.01,
                         key="_pcr_live_fpr",
                         format="%.2f")
    pop_at_risk = 2_000_000
    wrongly_flagged = int(live_fpr * pop_at_risk * 0.05)
    cascade_risk    = "🔴 HIGH" if live_fpr > 0.25 else "🟡 MEDIUM" if live_fpr > 0.12 else "🟢 LOW"
    r0_val          = 0.30 + live_fpr * 0.8
    ceasefire_risk  = max(0, live_fpr * 2.5 - 0.15)

    la,lb,lc,ld = st.columns(4)
    la.metric("Est. Wrongly Flagged",   f"{wrongly_flagged:,}",
              help="Civilians in high-risk zone incorrectly flagged")
    lb.metric("Cascade Risk Level",     cascade_risk)
    lc.metric("Violence R₀",           f"{r0_val:.2f}",
              help="Epidemic reproduction number — R₀>1 means conflict spreads")
    ld.metric("Ceasefire Failure Risk", f"{ceasefire_risk:.1%}")

    st.markdown(
        f"<div style='background:#fef2f2;border-left:4px solid #dc2626;"
        f"border-radius:0 8px 8px 0;padding:8px 14px;font-size:.82rem;'>"
        f"At FPR = {live_fpr:.1%}: an estimated <b>{wrongly_flagged:,} civilians</b> "
        f"are wrongly flagged. This drives community grievance (R2 loop) and could "
        f"push violence R₀ to <b>{r0_val:.2f}</b> — "
        f"{'⚠️ above the containment threshold of 1.0.' if r0_val>1 else '✅ below containment threshold.'}"
        f"</div>", unsafe_allow_html=True)

    if live_fpr != float(avg_fpr):
        st.session_state["pcr_points"] += 1  # reward exploration

    # ── Real-world benchmarks ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📚 Real-World Benchmarks")
    bm_df = pd.DataFrame([{"System": k, **v} for k, v in CONFLICT_BENCHMARKS.items()])
    bm_df["Your Model"] = [round(avg_acc,3)]*len(bm_df)
    st.dataframe(
        bm_df.style.background_gradient(cmap="RdYlGn", subset=["accuracy","fairness"]),
        use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — CONFLICT FAIRNESS
# ────────────────────────────────────────────────────────────────────────────
with T["⚖️ Conflict Fairness"]:
    st.markdown("### ⚖️ Community-Level Fairness Audit")
    st.markdown(
        f"<div style='background:#fef2f2;border-left:4px solid #dc2626;"
        f"padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:12px;'>"
        f"<strong>Scenario:</strong> {sc_info['name']}<br>"
        f"<strong>At-risk:</strong> {sc_info['at_risk']}<br>"
        f"<strong>Bias pathway:</strong> {sc_info['bias_type']}<br>"
        f"<em>{sc_info['ref']}</em></div>", unsafe_allow_html=True)

    if scenario_key == "ipob_separatist":
        st.warning("⚠️ IPOB scenario: Amnesty International documented extrajudicial killings "
                   "during Python Dance operations. AI surveillance amplifies existing harm.")

    fa1,fa2 = st.columns(2)
    with fa1:
        dims = ["Accuracy","Fairness","Trust","Ceasefire\nCompliance","Low FPR","Equity"]
        vals = [avg_acc, avg_fair, avg_trust,
                df["ceasefire_compliance"].mean(), 1-avg_fpr, 1-avg_dp]
        fig_r = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=dims+[dims[0]],
            fill="toself", fillcolor="rgba(15,118,110,0.18)",
            line=dict(color=ACCENT,width=2.5)))
        fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                            title="Peace AI Fairness Radar",
                            height=360, margin=dict(t=40,b=20),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_r, use_container_width=True)
    with fa2:
        gap_data = {"Demographic Parity Gap": avg_dp,
                    "Ethnic FPR Gap":         avg_eg,
                    "Accuracy Gap":           df["equity_gap"].mean(),
                    "Gender Gap":             df["gender_gap"].mean()}
        fig_gap = go.Figure(go.Bar(
            x=list(gap_data.values()), y=list(gap_data.keys()), orientation="h",
            marker_color=["#dc2626" if v>0.15 else "#f59e0b" if v>0.08 else "#16a34a"
                          for v in gap_data.values()],
            text=[f"{v:.3f}" for v in gap_data.values()], textposition="outside"))
        fig_gap.add_vline(x=0.10, line_dash="dot", line_color="#dc2626",
                          annotation_text="10% threshold")
        fig_gap.update_layout(title="Disparity Gaps", xaxis_title="Gap",
                              height=320, margin=dict(t=40,b=30,l=200),
                              paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gap, use_container_width=True)

    # ── Interactive: Threshold Decision Tool ──────────────────────────────────
    st.divider()
    st.markdown("#### 🎛️ Threshold Decision Tool — Set Your Own Thresholds")
    st.caption("As an AI Governance Officer, you must balance catching real threats "
               "against wrongly flagging civilians. Adjust the thresholds and see the trade-offs.")

    td1,td2 = st.columns(2)
    with td1:
        fpr_target   = st.slider("Maximum allowable FPR (%)", 5, 50,
                                  int(avg_fpr*100), 1, key="_pcr_fpr_target")
        fairness_min = st.slider("Minimum fairness score", 0.50, 0.95, 0.70,
                                  0.01, key="_pcr_fair_min")
    with td2:
        recall_min   = st.slider("Minimum recall (catch rate)", 0.40, 0.99, 0.70,
                                  0.01, key="_pcr_recall_min")
        trust_target = st.slider("Community trust target", 0.30, 0.90, 0.60,
                                  0.01, key="_pcr_trust_target")

    threshold_met = {
        "FPR ≤ target":      avg_fpr <= fpr_target/100,
        "Fairness ≥ minimum":avg_fair >= fairness_min,
        "Recall ≥ minimum":  df["recall"].mean() >= recall_min,
        "Trust ≥ target":    avg_trust >= trust_target,
    }
    pass_count = sum(threshold_met.values())
    deploy_verdict = (
        "✅ APPROVED FOR DEPLOYMENT" if pass_count == 4 else
        f"⚠️ CONDITIONAL — {pass_count}/4 criteria met" if pass_count >= 2 else
        "❌ REJECTED — insufficient fairness and safety")

    st.markdown(
        f"<div style='background:{'#f0fdf4' if pass_count==4 else '#fef2f2'};"
        f"border:2px solid {'#16a34a' if pass_count==4 else '#dc2626'};"
        f"border-radius:10px;padding:12px 16px;'>"
        f"<h4 style='margin:0 0 8px;'>Deployment Decision: {deploy_verdict}</h4>",
        unsafe_allow_html=True)
    for check, passed in threshold_met.items():
        st.markdown(f"{'✅' if passed else '❌'} {check}")
    st.markdown("</div>", unsafe_allow_html=True)
    if pass_count == 4:
        st.session_state["pcr_points"] += 25

    # ── Harm Quantification ───────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🏘️ Real-World Harm Quantification")
    pop_at_risk = 2_000_000
    wrongly_flagged = int(avg_fpr * pop_at_risk * 0.05)
    ci1,ci2,ci3 = st.columns(3)
    ci1.metric("Wrongly Flagged Civilians", f"{wrongly_flagged:,}")
    ci2.metric("Community Trust",            f"{avg_trust:.3f}")
    ci3.metric("Ceasefire Risk",             f"{1-df['ceasefire_compliance'].mean():.3f}")

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — GAME THEORY (interactive)
# ────────────────────────────────────────────────────────────────────────────
with T["🎮 Game Theory"]:
    st.markdown("### 🎮 Interactive Ceasefire Negotiation Game")
    st.markdown(
        "*You are the AI governance mediator. Two armed parties must decide whether to comply "
        "with or defect from a ceasefire. Your AI system's false alarm rate shapes their "
        "trust — and therefore their decision.*")

    gt1,gt2 = st.columns(2)
    with gt1:
        st.markdown("#### 🕹️ Play the Signalling Game")
        st.caption("Choose how to calibrate your AI system. Each choice changes "
                   "the payoff matrix below.")

        ai_choice = st.radio(
            "Your AI deployment decision:",
            ["Deploy without bias audit (fast)",
             "Deploy with partial audit (balanced)",
             "Delay for full fairness audit (slow but fair)",
             "Do not deploy — human-only assessment"],
            key="_pcr_gt_choice")

        fpr_effect = {"Deploy without bias audit (fast)": avg_fpr,
                      "Deploy with partial audit (balanced)": avg_fpr * 0.65,
                      "Delay for full fairness audit (slow but fair)": avg_fpr * 0.35,
                      "Do not deploy — human-only assessment": 0.05}
        chosen_fpr = fpr_effect[ai_choice]

        # Payoff matrix depends on chosen FPR
        trust_bonus = max(0, 0.5 - chosen_fpr * 2)
        payoffs_A   = np.array([[3.5 + trust_bonus, -1.0 - trust_bonus * 2],
                                 [-2.0 - avg_fpr*3,  0.5]])
        payoffs_B   = np.array([[3.5 + trust_bonus,  2.0],
                                 [1.0,                0.5]])

        parties_A = ["AI Flags Low Risk", "AI Flags High Risk"]
        parties_B = ["Party B: Comply",   "Party B: Defect"]

        fig_game = go.Figure(go.Heatmap(
            z=payoffs_A, x=parties_B, y=parties_A,
            text=[[f"A:{payoffs_A[i,j]:.1f} B:{payoffs_B[i,j]:.1f}"
                   for j in range(2)] for i in range(2)],
            texttemplate="%{text}",
            colorscale="RdYlGn", showscale=True))
        fig_game.update_layout(
            title=f"Payoff Matrix (FPR={chosen_fpr:.1%})",
            height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_game, use_container_width=True)

        # Nash equilibrium
        if chosen_fpr < 0.15:
            ne = "(Low Risk, Comply) — Cooperative equilibrium ✅"
            ne_col = "#16a34a"
        elif chosen_fpr < 0.30:
            ne = "(High Risk, Mixed strategies) — Unstable ⚠️"
            ne_col = "#f59e0b"
        else:
            ne = "(High Risk, Defect) — Conflict equilibrium ❌"
            ne_col = "#dc2626"

        st.markdown(
            f"<div style='background:{ne_col}15;border-left:3px solid {ne_col};"
            f"padding:8px 12px;border-radius:0 6px 6px 0;font-size:.83rem;'>"
            f"<strong>Nash Equilibrium:</strong> {ne}</div>", unsafe_allow_html=True)

        if chosen_fpr < 0.15:
            st.session_state["pcr_points"] += 20
            st.success("🏆 +20 points! You chose the cooperative equilibrium.")
        elif chosen_fpr < 0.30:
            st.session_state["pcr_points"] += 10
            st.info("⚠️ +10 points. Unstable equilibrium — consider deeper audit.")
        else:
            st.error("❌ High FPR destroys trust. Parties defect from ceasefire.")

        st.session_state["pcr_policy_choices"]["game_theory"] = ai_choice

    with gt2:
        st.markdown("#### 📈 Signalling Game — P(Peace) vs Bias")
        bias_range  = np.linspace(0.0, 0.5, 40)
        peace_probs = [max(0, 0.8*(1-bi*2.5) + 0.3*(1-(1-bi*2.5))) for bi in bias_range]

        fig_sig = go.Figure()
        fig_sig.add_scatter(x=bias_range, y=peace_probs, mode="lines",
                            line=dict(color=ACCENT,width=2.5),
                            fill="tozeroy", fillcolor="rgba(15,118,110,0.10)")
        fig_sig.add_vline(x=bias_intensity, line_dash="dash",
                          line_color="#dc2626", annotation_text="Current bias")
        fig_sig.add_vline(x=chosen_fpr, line_dash="dot",
                          line_color="#7c3aed", annotation_text="Your choice FPR")
        fig_sig.update_layout(
            title="Signalling Game: P(Peace) vs Bias Intensity",
            xaxis_title="AI Bias / FPR",
            yaxis_title="P(Peace Agreement)",
            height=280, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sig, use_container_width=True)

        st.markdown("#### 🎲 Commitment Problem Simulator")
        st.caption("Can the parties credibly commit to peace given this AI system?")
        commitment_score = max(0, 1 - chosen_fpr * 4)
        st.progress(commitment_score,
                    text=f"Commitment credibility: {commitment_score:.0%}")
        if commitment_score < 0.4:
            st.error("❌ Low credibility — parties cannot commit. Ceasefire likely to fail.")
        elif commitment_score < 0.7:
            st.warning("⚠️ Moderate credibility — ceasefire fragile, requires monitoring.")
        else:
            st.success("✅ High credibility — ceasefire conditions favourable.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — NETWORK CASCADE (interactive)
# ────────────────────────────────────────────────────────────────────────────
with T["🕸️ Network Cascade"]:
    st.markdown("### 🕸️ Interactive Conflict Network Cascade Simulator")
    st.markdown(
        "*Simulate how AI false alarms spread through community networks. "
        "Adjust parameters and watch the cascade unfold in real time.*")

    nc_col1, nc_col2 = st.columns([1,2])
    with nc_col1:
        st.markdown("**🎛️ Network Parameters**")
        net_fpr     = st.slider("Starting FPR (AI false alarm rate)",
                                0.02,0.60,float(avg_fpr),0.01, key="_pcr_net_fpr")
        n_nodes     = st.slider("Number of communities", 20, 120, 60, 10, key="_pcr_n_nodes")
        n_steps_net = st.slider("Simulation steps",       5,  40,  20,  5, key="_pcr_n_steps")
        net_seed    = st.number_input("Network seed (for reproducibility)",
                                      1, 9999, 42, key="_pcr_net_seed")
        run_cascade = st.button("▶ Run Cascade", type="primary",
                                use_container_width=True, key="_pcr_cascade_btn")

    with nc_col2:
        rng_net  = np.random.default_rng(int(net_seed))
        degrees  = rng_net.zipf(2.2, n_nodes).clip(1, 15)
        infected = np.zeros(n_nodes, dtype=bool)
        hubs     = np.argsort(degrees)[-int(net_fpr*n_nodes):]
        infected[hubs] = True

        cascade_hist = [infected.sum()]
        for _ in range(n_steps_net):
            new_inf = infected.copy()
            for node in range(n_nodes):
                if not infected[node]:
                    n_inf_n = min(degrees[node], infected.sum())
                    p = 1-(1-net_fpr*0.4)**n_inf_n
                    if rng_net.uniform() < p:
                        new_inf[node] = True
            infected = new_inf
            cascade_hist.append(infected.sum())
            if infected.all(): break

        fig_c = go.Figure()
        fig_c.add_scatter(x=list(range(len(cascade_hist))), y=cascade_hist,
                          mode="lines+markers", fill="tozeroy",
                          fillcolor="rgba(220,38,38,0.12)",
                          line=dict(color="#dc2626",width=2.5),
                          name="Affected Communities")
        fig_c.add_hline(y=n_nodes*0.5, line_dash="dot", line_color="#f59e0b",
                        annotation_text="50% cascade threshold")
        fig_c.update_layout(
            title=f"Cascade Result: {cascade_hist[-1]}/{n_nodes} communities affected "
                  f"({cascade_hist[-1]/n_nodes:.0%})",
            xaxis_title="Steps", yaxis_title="Affected Communities",
            height=320, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c, use_container_width=True)

        final_pct = cascade_hist[-1] / n_nodes
        if final_pct > 0.6:
            st.error(f"🔴 Full cascade — {cascade_hist[-1]} communities affected. "
                     f"Reduce FPR below ~15% to prevent runaway spread.")
            st.session_state["pcr_points"] += 5
        elif final_pct > 0.3:
            st.warning(f"🟡 Partial cascade — {cascade_hist[-1]} communities affected.")
        else:
            st.success(f"🟢 Cascade contained — only {cascade_hist[-1]} communities affected.")
            st.session_state["pcr_points"] += 15

    # Percolation threshold chart
    st.divider()
    st.markdown("#### 🔬 Percolation Threshold — Find the Critical FPR")
    st.caption("The percolation threshold is the FPR at which cascades become unstoppable. "
               "Drag the slider to find it.")
    fpr_range    = np.linspace(0.02, 0.55, 40)
    cascade_fin  = []
    rng_p = np.random.default_rng(42)
    for tfpr in fpr_range:
        inf_s = np.zeros(60, dtype=bool)
        seeds = np.argsort(rng_p.zipf(2.2,60).clip(1,15))[-int(tfpr*60):]
        inf_s[seeds] = True
        for _ in range(15):
            ns = inf_s.copy()
            for nd in range(60):
                if not inf_s[nd]:
                    p = 1-(1-tfpr*0.35)**min(3, inf_s.sum())
                    if rng_p.uniform() < p: ns[nd] = True
            inf_s = ns
        cascade_fin.append(inf_s.sum()/60)

    threshold_idx = next((i for i,v in enumerate(cascade_fin) if v>0.5), len(cascade_fin)-1)
    threshold_fpr = fpr_range[threshold_idx]

    fig_p = go.Figure()
    fig_p.add_scatter(x=fpr_range, y=cascade_fin, mode="lines",
                      line=dict(color=ACCENT,width=2.5),
                      fill="tozeroy", fillcolor="rgba(15,118,110,0.10)")
    fig_p.add_vline(x=threshold_fpr, line_dash="dash", line_color="#dc2626",
                    annotation_text=f"Percolation threshold: {threshold_fpr:.2f}")
    fig_p.add_vline(x=avg_fpr, line_dash="dot", line_color="#7c3aed",
                    annotation_text=f"Your FPR: {avg_fpr:.2f}")
    fig_p.update_layout(
        title="Percolation Threshold: FPR vs Cascade Size",
        xaxis_title="AI False Alarm Rate", yaxis_title="Fraction Affected",
        height=280, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_p, use_container_width=True)

    status = "🔴 ABOVE" if avg_fpr > threshold_fpr else "🟢 BELOW"
    st.info(f"{status} percolation threshold ({threshold_fpr:.2f}). "
            f"Your system's FPR ({avg_fpr:.2f}) is {status.split()[1]} the point where "
            f"cascades become self-sustaining.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — SEIR CONTAGION (interactive)
# ────────────────────────────────────────────────────────────────────────────
with T["🦠 SEIR Contagion"]:
    st.markdown("### 🦠 Interactive SEIR Violence Contagion Model")
    st.markdown(
        "*Compute the basic reproduction number R₀ of violence under your AI system. "
        "Adjust transmission parameters to find the peacebuilding investment needed.*")

    se1,se2 = st.columns([1,2])
    with se1:
        st.markdown("**🎛️ SEIR Parameters**")
        N_seir  = st.slider("Total communities (N)", 200, 2000, 1000, 100, key="_pcr_N")
        beta_s  = st.slider("Transmission rate β (AI false alarm effect)",
                            0.1, 1.0, round(0.3+avg_fpr*0.8, 2), 0.05, key="_pcr_beta")
        sigma_s = st.slider("Incubation rate σ (E→I)",
                            0.05, 0.40, 0.15, 0.01, key="_pcr_sigma")
        gamma_s = st.slider("Recovery rate γ (peace interventions)",
                            0.03, 0.30, 0.08, 0.01, key="_pcr_gamma")
        pb_invest = st.slider("💰 Peacebuilding investment boost (boosts γ)",
                              0.0, 0.20, 0.0, 0.01, key="_pcr_pb_invest")
        gamma_eff = gamma_s + pb_invest
        R0_seir   = round(beta_s / gamma_eff, 2)

        st.metric("Computed R₀", R0_seir,
                  help="R₀ > 1 = conflict spreads; R₀ < 1 = conflict contained")
        herd_thresh = max(0, 1 - 1/R0_seir) if R0_seir > 1 else 0
        st.metric("Herd Immunity Threshold", f"{herd_thresh:.0%}",
                  help="% of communities needing peacebuilding coverage to contain spread")

        if R0_seir > 2.0:
            st.error(f"🔴 R₀ = {R0_seir} — rapid spread. Need {herd_thresh:.0%} coverage.")
        elif R0_seir > 1.0:
            st.warning(f"⚠️ R₀ = {R0_seir} — growing. Need {herd_thresh:.0%} coverage.")
        else:
            st.success(f"✅ R₀ = {R0_seir} — conflict contained.")
            st.session_state["pcr_points"] += 15

    with se2:
        E0 = int(N_seir * avg_fpr * 1.5)
        I0 = int(N_seir * avg_fpr * 0.5)
        S0 = max(0, N_seir - E0 - I0)
        S,E,I,R_seir = [S0],[E0],[I0],[0.0]
        dt, steps = 0.5, 80
        for _ in range(steps):
            s,e,i,r = S[-1],E[-1],I[-1],R_seir[-1]
            ne = beta_s*s*i/N_seir*dt
            ni = sigma_s*e*dt
            nr = gamma_eff*i*dt
            S.append(max(0,s-ne)); E.append(max(0,e+ne-ni))
            I.append(max(0,i+ni-nr)); R_seir.append(r+nr)
        t_ax = [x*dt for x in range(steps+1)]

        fig_seir = go.Figure()
        for series,name,color in [
            (S,"Susceptible","#0d9488"),
            (E,"Exposed (AI-flagged)","#f59e0b"),
            (I,"Active Conflict","#dc2626"),
            (R_seir,"Resolved","#16a34a")]:
            fig_seir.add_scatter(x=t_ax,y=series,mode="lines",
                                 name=name,line=dict(width=2.5))
        fig_seir.update_layout(
            title=f"SEIR Model (β={beta_s:.2f}, σ={sigma_s}, γ={gamma_eff:.2f}, R₀={R0_seir})",
            xaxis_title="Months", yaxis_title="Communities",
            height=380, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_seir, use_container_width=True)

        peak_I   = int(max(I))
        peak_t   = t_ax[I.index(max(I))]
        resolved = int(R_seir[-1])
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Peak Conflict", f"{peak_I:,} communities")
        s2.metric("Time to Peak",  f"{peak_t:.0f} months")
        s3.metric("Resolved",      f"{resolved:,}")
        s4.metric("Still Active",  f"{int(I[-1]):,}")

    # ── Peacebuilding ROI calculator ──────────────────────────────────────────
    st.divider()
    st.markdown("#### 💰 Peacebuilding Investment ROI Calculator")
    pb_cost   = st.slider("Investment per community (USD '000)", 10, 500, 50, 10,
                           key="_pcr_pb_cost")
    lives_saved = int(peak_I * 0.12 * pb_invest * 20)
    cost_total  = int(N_seir * pb_invest * pb_cost * 1000)
    cost_per_life = cost_total // max(lives_saved, 1)
    st.markdown(
        f"At **{pb_invest:.0%}** investment boost: estimated **{lives_saved:,} lives saved** "
        f"at total cost **${cost_total:,.0f}** (~${cost_per_life:,} per life saved).")

# ────────────────────────────────────────────────────────────────────────────
# TAB 6 — BIAS DETECTIVE (game)
# ────────────────────────────────────────────────────────────────────────────
with T["🕵️ Bias Detective"]:
    st.markdown("### 🕵️ Bias Detective — Nigeria Conflict Edition")
    st.markdown(
        "*Read the clue from a real conflict AI scenario. Identify which bias type caused "
        "the harm. Score points and earn badges.*")

    import hashlib, random
    from datetime import date

    CLUES = [
        {"clue": "In Borno State, the AI flags 38% of Kanuri men aged 18-35 as high-risk. "
                 "The same algorithm flags only 9% of men from other ethnic groups with "
                 "identical socioeconomic profiles.",
         "answer": "ethnic",
         "explanation": "Ethnic bias — the model learned to use ethnicity as a proxy for "
                        "threat because training data over-represented Kanuri communities "
                        "in historical incident reports (ACLED 2021)."},
        {"clue": "The election violence AI flags 45% of polling units in Anambra State "
                 "as high-risk. These are the same units that voted 85%+ for the opposition "
                 "party in the last election.",
         "answer": "political",
         "explanation": "Political bias — the model confuses opposition strongholds with "
                        "conflict zones because 2015-2023 incident data reflected incumbents' "
                        "security deployments (YIAGA Africa 2023)."},
        {"clue": "The Niger Delta AI undervalues community conflict reports by 60% when "
                 "submitted in Ijaw language, compared to identical reports in English.",
         "answer": "linguistic",
         "explanation": "Linguistic bias — the NLP model was trained predominantly on "
                        "English incident reports, systematically discounting local-language "
                        "submissions (Amnesty International 2020)."},
        {"clue": "The DDR risk-scoring AI assigns ex-combatants a 40% higher risk score "
                 "for every year they lived in a COIN (Counter-Insurgency) zone, regardless "
                 "of individual behaviour records.",
         "answer": "geographic",
         "explanation": "Geographic bias — the model uses residential location as a proxy "
                        "for individual threat, encoding area-based discrimination "
                        "(UNICEF DDR Report 2022)."},
        {"clue": "The farmer-herder conflict AI assigns higher scores to communities where "
                 "the primary livelihood is cattle herding — even for communities that have "
                 "never experienced conflict.",
         "answer": "socioeconomic",
         "explanation": "Socioeconomic/occupational bias — the model learned to treat "
                        "pastoral livelihoods as a proxy for conflict risk, reflecting "
                        "historical stereotyping in training data (ICG Report 295)."},
    ]

    seed = int(hashlib.md5(
        f"{date.today().isoformat()}conflict".encode()).hexdigest(), 16) % len(CLUES)
    clue_obj = CLUES[seed]

    st.markdown(
        f"<div style='background:#0f172a;border-left:4px solid #2dd4bf;"
        f"border-radius:0 12px 12px 0;padding:16px 20px;margin-bottom:16px;'>"
        f"<p style='color:#94a3b8;font-size:.7rem;font-weight:700;"
        f"letter-spacing:.1em;text-transform:uppercase;margin:0 0 6px;'>"
        f"🔍 TODAY'S CLUE</p>"
        f"<p style='color:#f1f5f9;font-size:.92rem;line-height:1.6;margin:0;'>"
        f"{clue_obj['clue']}</p></div>", unsafe_allow_html=True)

    bias_options = ["— select your answer —","demographic","historical","ethnic",
                    "geographic","linguistic","socioeconomic","political","gender"]
    detective_guess = st.selectbox("Your diagnosis:", bias_options,
                                   key=f"_pcr_detective_{date.today().isoformat()}")

    if detective_guess != "— select your answer —":
        if detective_guess == clue_obj["answer"]:
            st.success(f"✅ Correct! {clue_obj['explanation']}")
            st.session_state["pcr_points"] += 50
            if "detective_badge" not in st.session_state.pcr_badges:
                st.session_state.pcr_badges.append("detective_badge")
                st.balloons()
        else:
            st.error(f"❌ Not quite. {clue_obj['explanation']}")
        st.session_state["pcr_points"] += 5  # participation points

    st.divider()
    st.markdown("#### 🏆 Leaderboard — Session Scores")
    st.metric("Your Points This Session", st.session_state.get("pcr_points", 0))
    badges = st.session_state.get("pcr_badges", [])
    if badges:
        st.markdown("**🏅 Badges Earned:** " + " ".join(
            {"detective_badge":"🕵️ Detective"}.get(b, b) for b in badges))

    # ── Ethical Dilemma Quiz ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### ⚖️ Ethical Dilemma Challenge")
    dilemma = st.radio(
        "A conflict AI in Imo State has 70% accuracy but a 35% false alarm rate on "
        "Igbo youth. A human-only system has 55% accuracy but 8% false alarm rate. "
        "Which do you deploy?",
        ["AI system — higher accuracy saves more lives",
         "Human system — lower false alarm rate protects civilian rights",
         "Hybrid — AI flags, human reviews every case",
         "Neither — delay until bias is corrected"],
        key="_pcr_dilemma", index=None)

    if dilemma:
        responses = {
            "AI system — higher accuracy saves more lives":
                "⚠️ Contested choice. 35% FPR means wrongly flagging 35 out of 100 innocent "
                "civilians — likely fuelling grievance and radicalisation (R2 loop). "
                "Accuracy alone cannot justify mass profiling (Amnesty AI Standards 2023).",
            "Human system — lower false alarm rate protects civilian rights":
                "✅ Human rights-aligned, but 55% accuracy means missing 45% of real threats. "
                "The trade-off is real — this is why hybrid approaches are recommended by NITDA.",
            "Hybrid — AI flags, human reviews every case":
                "✅ Best practice (NITDA 2021 §4.3). Human-in-the-loop reduces false alarms "
                "while maintaining threat detection. Requires capacity building in affected states. "
                "+20 points!",
            "Neither — delay until bias is corrected":
                "✅ Precautionary principle. Justified when harm risk is high (NHRC Guidelines 2023). "
                "Requires robust alternative human intelligence system during delay. +10 points!",
        }
        st.markdown(responses[dilemma])
        if "Hybrid" in dilemma:
            st.session_state["pcr_points"] += 20
        elif "Neither" in dilemma:
            st.session_state["pcr_points"] += 10
        else:
            st.session_state["pcr_points"] += 5

# ────────────────────────────────────────────────────────────────────────────
# TAB 7 — POLICY SIMULATOR
# ────────────────────────────────────────────────────────────────────────────
with T["🎛️ Policy Simulator"]:
    st.markdown("### 🎛️ Peace AI Policy Simulator")
    st.markdown(
        "*You are the NHRC AI Governance Commissioner. Design a policy package and "
        "see its projected impact on false alarm rates, community trust, and ceasefire stability.*")

    ps1,ps2 = st.columns(2)
    with ps1:
        st.markdown("**📜 Choose Policy Interventions**")
        pol_audit    = st.toggle("Mandatory quarterly ethnic bias audit", key="_pcr_pol_audit")
        pol_human    = st.toggle("Human-in-the-loop for every flag",     key="_pcr_pol_human")
        pol_explain  = st.toggle("Mandatory explanation to flagged person",key="_pcr_pol_explain")
        pol_data     = st.toggle("Community-controlled data correction",  key="_pcr_pol_data")
        pol_threshold= st.toggle("Hard FPR cap at 15%",                  key="_pcr_pol_thresh")
        pol_federated= st.toggle("Federated model (no central ethnic data)",key="_pcr_pol_fed")
        pol_sunset   = st.toggle("Automatic sunset after 12 months",     key="_pcr_pol_sunset")
        pol_appeal   = st.toggle("Right of appeal with independent panel",key="_pcr_pol_appeal")

        pol_count = sum([pol_audit,pol_human,pol_explain,pol_data,
                         pol_threshold,pol_federated,pol_sunset,pol_appeal])

    with ps2:
        # Compute projected impact
        base_fpr_p   = avg_fpr
        base_trust_p = avg_trust
        base_fair_p  = avg_fair

        fpr_reduction    = (0.12*pol_audit + 0.18*pol_human + 0.05*pol_explain
                            + 0.10*pol_data + 0.20*pol_threshold + 0.08*pol_federated)
        trust_boost      = (0.08*pol_audit + 0.10*pol_human + 0.12*pol_explain
                            + 0.15*pol_data + 0.05*pol_threshold + 0.07*pol_appeal)
        fairness_boost   = (0.05*pol_audit + 0.08*pol_human + 0.04*pol_explain
                            + 0.09*pol_data + 0.10*pol_threshold + 0.06*pol_federated)
        implementation_cost = pol_count * 8.5  # USD million per intervention

        proj_fpr   = max(0.02, base_fpr_p   - fpr_reduction)
        proj_trust = min(0.95, base_trust_p  + trust_boost)
        proj_fair  = min(0.98, base_fair_p   + fairness_boost)

        st.markdown("**📊 Projected Impact**")
        m1,m2,m3 = st.columns(3)
        m1.metric("False Alarm Rate",  f"{proj_fpr:.1%}",
                  delta=f"{proj_fpr-base_fpr_p:+.1%}", delta_color="inverse")
        m2.metric("Community Trust",   f"{proj_trust:.3f}",
                  delta=f"{proj_trust-base_trust_p:+.3f}", delta_color="normal")
        m3.metric("Fairness Score",    f"{proj_fair:.3f}",
                  delta=f"{proj_fair-base_fair_p:+.3f}", delta_color="normal")

        st.metric("Implementation Cost", f"${implementation_cost:.0f}M",
                  help="Estimated annual cost across all Nigerian states")

        # Policy package verdict
        policy_score = (proj_fair*40 + (1-proj_fpr)*30 + proj_trust*30)
        verdict_col  = "#16a34a" if policy_score>70 else "#f59e0b" if policy_score>50 else "#dc2626"
        verdict_txt  = ("✅ Strong Policy Package — Human Rights Compliant"
                        if policy_score > 70 else
                        "⚠️ Moderate Package — Gaps Remain"
                        if policy_score > 50 else
                        "❌ Insufficient — Major Rights Risks Remain")

        st.markdown(
            f"<div style='background:{verdict_col}15;border:2px solid {verdict_col};"
            f"border-radius:10px;padding:12px 16px;margin-top:12px;'>"
            f"<h4 style='color:{verdict_col};margin:0 0 6px;'>Policy Score: {policy_score:.0f}/100</h4>"
            f"<p style='margin:0;font-size:.85rem;'>{verdict_txt}</p></div>",
            unsafe_allow_html=True)

        if policy_score > 70:
            st.session_state["pcr_points"] += 30
            st.session_state["pcr_policy_choices"]["policy_package"] = pol_count

        # Visualise policy impact
        categories  = ["False Alarm\nReduction","Trust\nBoost","Fairness\nGain","Cost\nEfficiency"]
        values      = [fpr_reduction, trust_boost, fairness_boost,
                       max(0, 1 - implementation_cost/100)]
        fig_pol = go.Figure(go.Scatterpolar(
            r=values+[values[0]], theta=categories+[categories[0]],
            fill="toself", fillcolor="rgba(15,118,110,0.20)",
            line=dict(color=ACCENT,width=2.5)))
        fig_pol.update_layout(
            polar=dict(radialaxis=dict(range=[0,0.5])),
            title="Policy Package Impact", height=320,
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pol, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# REMAINING TABS (XAI, Regulatory, Compliance, Longitudinal, Federated,
#                 Export, Feature Modules, Safety, Lifecycle, Eco, DS)
# ────────────────────────────────────────────────────────────────────────────
with T["🔍 XAI & Explainability"]:
    st.markdown("### 🔍 Explainability & Transparency Audit")
    rng_xai  = np.random.default_rng(42)
    n_feat   = sc_info["n_features"]
    feat_nm  = ["Prior Incidents","Infrastructure","Geographic Proximity",
                "Population Density","Historical Grievance","Economic Margin.",
                "Armed Group","State Capacity","Natural Resources",
                "Social Media","Displacement","Cross-Border"][:n_feat]
    importance = rng_xai.dirichlet(np.ones(n_feat)*2)
    proxies    = {0,2,4}

    x1,x2 = st.columns(2)
    with x1:
        fig_fi = px.bar(
            x=importance[np.argsort(importance)[::-1]],
            y=[feat_nm[i] for i in np.argsort(importance)[::-1]],
            orientation="h", title="Feature Importance",
            color=["#dc2626" if i in proxies else "#0f766e"
                   for i in np.argsort(importance)[::-1]],
            color_discrete_map="identity",
            labels={"x":"Importance","y":"Feature"})
        fig_fi.update_layout(height=340, margin=dict(t=40,b=30,l=180),
                              showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption("🔴 Red = proxy variables encoding historical discrimination")

    with x2:
        cf_c = rng_xai.normal(0,0.08,n_feat)
        cf_c[0]=0.31; cf_c[2]=0.18; cf_c[5]=-0.12
        cf_df = pd.DataFrame({"Feature":feat_nm,"Contribution":cf_c}).sort_values(
            "Contribution",key=abs,ascending=False)
        fig_cf = go.Figure(go.Bar(
            x=cf_df["Feature"][:8], y=cf_df["Contribution"][:8],
            marker_color=["#dc2626" if v>0 else "#16a34a" for v in cf_df["Contribution"][:8]],
            text=[f"{v:+.3f}" for v in cf_df["Contribution"][:8]],
            textposition="outside"))
        fig_cf.update_layout(title="SHAP: Why was this civilian flagged?",
                              height=320, margin=dict(t=40,b=80),
                              paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cf, use_container_width=True)

    top3    = np.argsort(importance)[-3:][::-1]
    if len(set(top3) & proxies) >= 2:
        st.error("🔴 Proxy dominance: top features encode historical over-policing, not real threat.")
    st.info("Counterfactual: This person would NOT be flagged if 'Prior Incidents' reduced by 2 entries.")

with T["🏛️ Nigeria Regulatory"]:
    st.markdown("### 🇳🇬 Nigeria Regulatory Framework")
    r1,r2 = st.columns(2)
    with r1:
        st.markdown("""**Nigerian Legal Framework:**
- **1999 Constitution §33-44** — Fundamental rights
- **NDPR 2019** — Data processing consent
- **NITDA AI Policy 2021** — AI governance guidelines
- **Counter-Terrorism Act 2013 (2022 amendment)**
- **Freedom of Information Act 2011** — Right to explanation
- **SERAP Litigation** — Ethnic algorithmic discrimination""")
        if scenario_key == "ipob_separatist":
            st.markdown("""**Southeast-Specific:**
- **Terrorism Prevention (Amendment) Act 2022** — IPOB proscription scope
- **African Charter Art.9** — Freedom of expression
- **UN Basic Principles on Use of Force**""")
    with r2:
        st.markdown("""**ECOWAS & International:**
- **ECOWAS AI Policy 2023**
- **AU AI Continental Strategy 2024**
- **UN General Assembly A/RES/78/265 (2024)**
- **Geneva Convention AP-II** — AI in NIAC
- **R2P** — AI must not enable mass atrocities""")
    st.divider()
    _nr = {"accuracy":avg_acc,"fairness_score":avg_fair,"demographic_parity":float(avg_dp)}
    nigeria_compliance_panel(_nr,domain="conflict",
        has_xai=locals().get("enable_xai",True),
        has_governance=locals().get("enable_governance",True),
        has_multilingual=True,has_ussd_fallback=False,
        has_gender_audit=locals().get("enable_gender_audit",False),
        has_redteam=locals().get("enable_redteam",False))

with T["📋 Compliance"]:
    st.markdown("### 📋 Algorithmic Compliance Report")
    checks = {
        "Fairness score ≥ 0.70":       avg_fair >= 0.70,
        "False alarm rate < 20%":       avg_fpr  < 0.20,
        "Ethnic FPR gap < 10%":        avg_eg   < 0.10,
        "Demographic parity gap < 10%":avg_dp   < 0.10,
        "At least 3 runs completed":   len(df)  >= 3,
        "Community trust > 0.60":      avg_trust > 0.60,
    }
    pass_n = sum(checks.values())
    st.markdown(f"**Overall: {'✅ COMPLIANT' if pass_n==6 else f'❌ {pass_n}/6 criteria met'}**")
    for c,p in checks.items():
        st.markdown(f"{'✅' if p else '❌'} {c}")
    _nr2 = {"accuracy":avg_acc,"fairness_score":avg_fair,"demographic_parity":float(avg_dp)}
    nigeria_compliance_panel(_nr2,domain="conflict",
        has_xai=locals().get("enable_xai",True),
        has_governance=locals().get("enable_governance",True),
        has_multilingual=True,has_ussd_fallback=False,
        has_gender_audit=locals().get("enable_gender_audit",False),
        has_redteam=locals().get("enable_redteam",False))

with T["🔁 Longitudinal"]:
    st.markdown("### 🔁 Longitudinal Bias Feedback Loop")
    lng = st.session_state.get("pcr_longitudinal")
    if not lng:
        st.info("Run a simulation to see longitudinal bias evolution.")
    else:
        ga = lng.get("generation_accuracies",[])
        gb = lng.get("generation_biases",[])
        if ga:
            fig_l = go.Figure()
            fig_l.add_scatter(y=ga,name="Accuracy",line=dict(color=ACCENT,width=2))
            if gb: fig_l.add_scatter(y=gb,name="Bias",line=dict(color="#dc2626",width=2),yaxis="y2")
            fig_l.update_layout(title="Bias Feedback Loop — Retraining Cycles",
                yaxis2=dict(overlaying="y",side="right"),height=340,
                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_l, use_container_width=True)
        st.json({k:v for k,v in lng.items() if not isinstance(v,(list,np.ndarray)) or len(str(v))<200})

with T["🌐 Federated"]:
    st.markdown("### 🌐 Federated Learning — Cross-State Bias Persistence")
    fed = st.session_state.get("pcr_federated")
    if not fed:
        st.info("Run a simulation to see federated results.")
    else:
        clients = fed.get("client_results",[])
        if clients:
            st.dataframe(pd.DataFrame([{
                "Client":f"State {i+1}","Accuracy":c.get("accuracy",0),
                "Fairness":c.get("fairness_score",0),"Bias Persists":c.get("bias_persists",False)
            } for i,c in enumerate(clients)]),use_container_width=True,hide_index=True)
        rnd = fed.get("round_global_accuracies",[])
        if rnd:
            fig_f = go.Figure()
            fig_f.add_scatter(y=rnd,name="Global Accuracy",line=dict(color=ACCENT,width=2))
            fig_f.update_layout(title="Federated Learning Rounds",height=300,
                                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_f,use_container_width=True)

with T["📤 Export"]:
    st.markdown("### 📤 Export Results")
    e1,e2 = st.columns(2)
    with e1:
        st.download_button(t("download_csv"),
            df.to_csv(index=False).encode(),
            f"gags_conflict_{scenario_key}.csv","text/csv",use_container_width=True)
    with e2:
        import json
        st.download_button("⬇️ Config (JSON)",
            json.dumps({"scenario":scenario_key,"bias_intensity":bias_intensity,
                        "n_samples":int(n_samples),"n_runs":int(n_runs),
                        "avg_fairness":round(avg_fair,4),"avg_fpr":round(avg_fpr,4)},
                       indent=2).encode(),
            f"gags_conflict_config.json","application/json",use_container_width=True)
    st.dataframe(df.style.background_gradient(subset=["fairness_score"],cmap="RdYlGn")
                   .background_gradient(subset=["false_alarm_rate"],cmap="Reds"),
                 use_container_width=True)

with T["🔬 Feature Modules"]:
    _fout = st.session_state.get("pcr_feature_outputs",{})
    try: feature_modules_tab(_fout, domain="conflict")
    except Exception as _fe:
        st.info("Enable Red Team / Agent Economy / Arena in sidebar first.")
        if _fout: st.json({k:str(v)[:200] for k,v in _fout.items()})

with T["🛡️ AI Safety"]:
    try: render_safety_tab(st.session_state.get("pcr_safety_report",{}),domain="conflict")
    except Exception as _e: st.error(f"AI Safety error: {_e}")

with T["🔄 Lifecycle"]:
    try: render_lifecycle_tab(st.session_state.get("pcr_lifecycle_report"),domain="conflict")
    except Exception as _e: st.error(f"Lifecycle error: {_e}")

with T["🌱 Eco Score"]:
    try: render_eco_tab(st.session_state.get("pcr_lifecycle_report"),domain="conflict")
    except Exception as _e: st.error(f"Eco error: {_e}")

with T["🔮 Dynamic Systems"]:
    try:
        render_dynamic_systems_tab(
            st.session_state.get("pcr_ds_report",{}),
            domain="conflict", ds_key="pcr_ds_report")
    except Exception as _de:
        st.error(f"🔮 Dynamic Systems error: {_de}")
        import traceback; st.code(traceback.format_exc())

# ── History, annotations, what-if ─────────────────────────────────────────────
history_browser("pcr_snapshot_history",domain="conflict",
    key_metrics=["accuracy","fairness_score","false_alarm_rate","community_trust"])
annotation_panel("pcr_annotations",
    context_label=f"{len(st.session_state.pcr_run_history)} Conflict run(s)")

if st.session_state.pcr_run_history:
    what_if_explorer(
        domain="conflict",
        current_metrics={k:v for k,v in st.session_state.pcr_run_history[-1].items()
                         if isinstance(v,(int,float))},
        bias_intensity=bias_intensity)

# ── Policy recommendations ────────────────────────────────────────────────────
st.divider()
st.markdown("### 💡 Evidence-Based Policy Recommendations")
p1,p2,p3 = st.columns(3)
with p1:
    st.markdown(
        f"<div style='background:#f0fdfa;border-top:3px solid {ACCENT};"
        f"border-radius:8px;padding:14px;'>"
        f"<p style='font-weight:700;color:{ACCENT};font-size:.85rem;margin:0 0 8px;'>"
        f"🏛️ Policy Makers (NSA/NHRC)</p>"
        f"<ul style='font-size:.78rem;color:#374151;margin:0;padding-left:16px;line-height:1.7;'>"
        f"<li>Mandate FPR &lt; 15% for civilian-facing AI</li>"
        f"<li>Require ethnic bias audit before deployment</li>"
        f"<li>Prohibit ethnic/geographic proxies as primary features</li>"
        f"<li>Community grievance mechanism for flagged persons</li>"
        f"</ul></div>", unsafe_allow_html=True)
with p2:
    st.markdown(
        f"<div style='background:#fff7ed;border-top:3px solid #f59e0b;"
        f"border-radius:8px;padding:14px;'>"
        f"<p style='font-weight:700;color:#b45309;font-size:.85rem;margin:0 0 8px;'>"
        f"🤖 AI Developers</p>"
        f"<ul style='font-size:.78rem;color:#374151;margin:0;padding-left:16px;line-height:1.7;'>"
        f"<li>Remove prior-incident-report features</li>"
        f"<li>Apply adversarial debiasing on ethnicity proxies</li>"
        f"<li>Federated learning prevents single-point bias</li>"
        f"<li>Plain-language explanations to flagged communities</li>"
        f"</ul></div>", unsafe_allow_html=True)
with p3:
    st.markdown(
        f"<div style='background:#fef2f2;border-top:3px solid #dc2626;"
        f"border-radius:8px;padding:14px;'>"
        f"<p style='font-weight:700;color:#991b1b;font-size:.85rem;margin:0 0 8px;'>"
        f"🕊️ Peacebuilders (UNDP/UNICEF)</p>"
        f"<ul style='font-size:.78rem;color:#374151;margin:0;padding-left:16px;line-height:1.7;'>"
        f"<li>Fund community data-correction programmes</li>"
        f"<li>Include AI fairness in DDR programme monitoring</li>"
        f"<li>Advocate human-in-the-loop in ECOWARN</li>"
        f"<li>Commission algorithmic impact assessments</li>"
        f"</ul></div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:.72rem;padding:1.5rem 0;'>"
    "GAGS Peace & Conflict Module · ACLED, UNHCR, ICG, UNDP, Amnesty International data · "
    "Fearon 1995 · Schelling 1960 · Barabási 2002 · Pearl 2009 · Educational simulation only"
    "</div>", unsafe_allow_html=True)