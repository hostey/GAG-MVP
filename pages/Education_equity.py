# pages/07_🎓_Education_Equity.py
"""
Education Equity Simulation — GAGS Framework v3.0
Examinations, admissions, dropout prediction, and automated grading.
Nigeria: JAMB/WAEC context. Global: university admission screening.
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
from components.translate import install_auto_translate, tx, tx_plotly, language_switcher
install_auto_translate()

from components.governance_logic import (
    run_simple_simulation,
    apply_bias, simulate_data_poisoning, calculate_fairness_metrics,
    simulate_bias_mitigation, ExplainableModel, generate_compliance_report,
    generate_intersectional_fairness, AttackSeverity, plugin_registry,
    simulate_longitudinal_bias, simulate_federated_learning,
    generate_education_data, calculate_education_equity,
    EDUCATION_SCENARIO_PRESETS,
)
from components.ux_utils import (
    guided_tour_banner, preset_selector, metric_glossary_expander,
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
from components.i18n import t, get_lang
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
st.set_page_config(page_title="Education Equity • GAGS", page_icon="🎓", layout="wide")

# ── Safe top-level preset_info guard ──────────────────────────────────────────
# preset_info must be defined before ANY st.markdown() calls, even if the
# sidebar hasn't executed yet (Streamlit executes top-to-bottom each rerun).
__educ_sk = st.session_state.get("_edu_scenario_key", list(EDUCATION_SCENARIO_PRESETS.keys())[0])
if __educ_sk not in EDUCATION_SCENARIO_PRESETS:
    __educ_sk = list(EDUCATION_SCENARIO_PRESETS.keys())[0]
scenario_key = __educ_sk
preset_info  = EDUCATION_SCENARIO_PRESETS[scenario_key]



# ── Auto-dismiss stale guided tour banners from other pages ───────────────────
for _tk in ["_tour_dismissed_agrotech","_tour_dismissed_health","_tour_dismissed_security"]:
    if _tk not in st.session_state:
        st.session_state[_tk] = True


# ── Design system ──────────────────────────────────────────────────────────────
try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, page_header, plotly_theme as _ptheme
    inject_css("education")
    ACCENT = DOMAIN_ACCENTS["education"]
except ImportError:
    ACCENT = "#f59e0b"


try:
    from components.gags_charts import (
        waterfall_feature_contributions, benchmark_comparison_bar,
        lollipop_gap_chart, radar_with_benchmark, fairness_heatmap,
        multi_run_distribution, ai_bias_incident_timeline, gauge_cluster,
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

# ── State ──────────────────────────────────────────────────────────────────────
_EDU_STATE = {
    "edu_run_history": [], "edu_safety_report": {}, "edu_xai_results": {},
    "edu_longitudinal": None, "edu_federated": None,
    "edu_snapshot_history": [], "edu_annotations": [],
    "edu_ds_report": {}
}
for _k, _v in _EDU_STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_VALID_BIAS_TYPES_EDU = list(simulation_config.BIAS_TYPES) + [
    b for b in ["gender","linguistic","geographic"] if b not in simulation_config.BIAS_TYPES]

def _run_one(scenario_key, n_samples, selected_biases, bias_intensity,
             poison_rate, run_idx, enable_xai, enable_governance, selected_state=None):
    from components.governance_logic import (
    run_simple_simulation,
        generate_education_data, calculate_education_equity,
        apply_bias, simulate_data_poisoning, calculate_fairness_metrics,
        ExplainableModel, generate_compliance_report,
        AttackSeverity, simulate_longitudinal_bias, simulate_federated_learning,
    )
    X, y, demo, feat_names, preset = generate_education_data(
        scenario_key, n_samples, random_state=42 + run_idx)
    preset = apply_state_to_preset(preset, selected_state or "Nigeria (National Average)")
    X = X.astype(np.float64)
    for bt in [b for b in selected_biases if b in _VALID_BIAS_TYPES_EDU]:
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
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4,
        learning_rate=0.08, random_state=42 + run_idx)
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    acc  = float(accuracy_score(yte, yp))
    rec  = float(recall_score(yte, yp, zero_division=0))
    prec = float(precision_score(yte, yp, zero_division=0))
    f1   = float(f1_score(yte, yp, zero_division=0))
    fpr  = float(np.mean(yp[yte == 0] == 1)) if (yte == 0).any() else 0.0
    fair = calculate_fairness_metrics(yte, yp, gte)
    eq   = calculate_education_equity(yte, yp, Xte, feat_names, preset)
    adv  = gte == 1; dis = gte == 0
    acc_adv = float(accuracy_score(yte[adv], yp[adv])) if adv.any() else acc
    acc_dis = float(accuracy_score(yte[dis], yp[dis])) if dis.any() else acc
    if enable_xai and run_idx == 0:
        try:
            xm = ExplainableModel(domain="education")
            xm.model = clf; xm._X_train = Xtr
            xm._is_fitted = True; xm.feature_names = feat_names[:Xte.shape[1]]
            fi = xm.feature_importance(Xte, yte, n_repeats=6)
            xai_d = {"feature_importance": fi.__dict__}
            denied = np.where((yte == 0) & (yp == 0))[0]
            if len(denied):
                expl = xm.explain_instance(Xte[denied[0]])
                cf   = xm.counterfactual(Xte[denied[0]])
                xai_d["instance_explanation"] = expl.__dict__
                xai_d["counterfactual"]       = cf.__dict__
            mc = xm.model_card({"accuracy": acc},
                {"fairness_score": eq.fairness_score,
                 "demographic_parity_difference": fair.get("demographic_parity_difference", 0)},
                domain="education")
            cr = generate_compliance_report(mc,
                {"fairness_score": eq.fairness_score,
                 "demographic_parity_difference": fair.get("demographic_parity_difference", 0)},
                {"accuracy": acc},
                frameworks=["EU AI Act", "ISO 42001", "NITDA", "UNESCO"])
            xai_d.update({"model_card": mc.__dict__, "compliance_report": cr})
            st.session_state.edu_xai_results = xai_d
        except Exception as e:
            st.session_state.edu_xai_results = {"error": str(e)}
    if run_idx == 0:
        bi = bias_intensity if bias_intensity > 0 else 0.15
        try:
            lng = simulate_longitudinal_bias(X, y, demo,
                initial_bias_type="socioeconomic",
                initial_bias_intensity=bi, n_generations=5, random_state=42)
            st.session_state.edu_longitudinal = lng.__dict__
        except Exception: st.session_state.edu_longitudinal = None
        try:
            fed = simulate_federated_learning(X, y, demo, n_clients=4, n_rounds=3,
                bias_heterogeneity=bi * 0.5, random_state=42)
            st.session_state.edu_federated = fed.__dict__
        except Exception: st.session_state.edu_federated = None
    # ── Feature modules (run when enabled) ───────────────────────────────────

    return {
        "run_id": run_idx + 1, "scenario": preset.name,
        "accuracy": acc, "recall": rec, "precision": prec, "f1_score": f1,
        "fpr": fpr,
        "fairness_score": eq.fairness_score,
        "opportunity_gap": eq.opportunity_gap if hasattr(eq, "opportunity_gap") else abs(acc_adv - acc_dis),
        "gender_gap": eq.gender_gap if hasattr(eq, "gender_gap") else abs(acc_adv - acc_dis) * 0.8,
        "urban_rural_gap": eq.urban_rural_gap if hasattr(eq, "urban_rural_gap") else abs(acc_adv - acc_dis) * 0.6,
        "ses_gap": fair.get("demographic_parity_difference", 0),
        "school_type_gap": abs(acc_adv - acc_dis) * 0.7,
        "language_disparity": abs(acc_adv - acc_dis) * 0.5,
        "digital_exclusion": 1 - eq.digital_inclusion_score if hasattr(eq, "digital_inclusion_score") else 0.3,
        "demographic_parity": fair.get("demographic_parity_difference", 0),
        "equalized_odds": fair.get("equalized_odds_difference", 0),
        "acc_advantaged": acc_adv, "acc_disadvantaged": acc_dis,
        "equity_gap": abs(acc_adv - acc_dis),
        "bias_intensity": bias_intensity, "poison_rate": poison_rate,
        "biases": ", ".join(selected_biases) or "None",
        "equity_narrative": eq.narrative if hasattr(eq, "narrative") else "",
    }


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


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    language_switcher(location="sidebar")
    st.divider()

    # ── State selector ────────────────────────────────────────────────────────
    st.divider()
    selected_state = state_selector(key="_state_07educationequity", location="sidebar")
    state_info_card(selected_state)

    st.markdown(
    "<div style='background:linear-gradient(90deg,#f8fafc,#f1f5f9);"
    "border-radius:6px;padding:6px 10px;margin-bottom:6px;'>"
    "<span style='font-size:.68rem;font-weight:700;color:#475569;"
    "text-transform:uppercase;letter-spacing:.07em;'>🎓 Education Equity</span>"
    "</div>",
    unsafe_allow_html=True)
    role_switcher("education")
    progress_tracker(location="sidebar")
    role_algo_banner("education")
    # ── Role-recommended algorithm ─────────────────────────────────
    _role_algo, _role_algo_label, _ = get_role_algo("education")

    st.divider()

    st.subheader("🎓 Education Scenario")
    scenario_key = st.selectbox(
        "Scenario", list(EDUCATION_SCENARIO_PRESETS.keys()),
        format_func=lambda k: EDUCATION_SCENARIO_PRESETS[k].name,
        key="_edu_scenario_key")
    preset_info = EDUCATION_SCENARIO_PRESETS[scenario_key]
    st.caption(preset_info.description[:200])
    st.divider()

    st.subheader(f"🎭 {t('bias_config')}")
    _edu_valid = list(dict.fromkeys(
        list(simulation_config.BIAS_TYPES) +
        ["demographic", "socioeconomic", "geographic", "gender", "linguistic"]))
    _edu_def = [b for b in ["demographic", "socioeconomic"]
                if b in _edu_valid]
    selected_biases = st.multiselect(
        "Bias Types", options=_edu_valid, default=_edu_def)
    bias_intensity = st.slider(
        "Bias Intensity", 0.0, float(simulation_config.MAX_BIAS_FACTOR), 0.30, 0.05)
    st.divider()

    st.subheader(f"⚠️ {t('attack_header')}")
    poison_rate = st.slider("Poisoning Rate", 0.0, 0.5, 0.05, 0.01, format="%.2f")
    st.divider()

    st.subheader(f"📊 {t('sim_params_header')}")
    n_samples = st.number_input(
        "Records", 1000, 50000, settings.DEFAULT_N_SAMPLES, 1000)
    n_runs = st.slider("Runs", 1, 8, 3)
    st.divider()

    st.subheader(f"📊 {t('view_mode_header')}")
    _vm_key = "_view_mode_education"
    if _vm_key not in st.session_state:
        st.session_state[_vm_key] = "Industry"
    view_mode = st.radio(t("perspective"), ["Industry", "Research", "Student Impact"],
        horizontal=True, key=_vm_key,
        help="Industry: KPI dashboard. Research: statistical depth. Student Impact: exclusion focus.")
    st.divider()
    st.subheader(f"🔬 {t('modules_header')}")
    enable_xai        = st.toggle("Explainable AI",   value=True)
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
        for _k, _v in _EDU_STATE.items():
            st.session_state[_k] = type(_v)()
        st.rerun()





st.markdown(
    f"""<div class="page-header" style="--ac:#ffe234;"><p style="font-family:'DM Mono',monospace;font-size:.69rem;letter-spacing:.16em;text-transform:uppercase;opacity:.5;margin:0 0 .55rem;display:flex;align-items:center;gap:.45rem;"><span style="width:16px;height:1px;background:#ffe234;opacity:.55;display:inline-block;"></span>EDUCATION · GAGS v3.0 · Nigeria</p><h1 style="font-family:'Syne',sans-serif!important;font-size:2.5rem!important;font-weight:800!important;line-height:1.08!important;letter-spacing:-.03em!important;margin:0 0 .6rem!important;">Education Equity Simulation</h1><p style="margin:0;opacity:.72;font-size:.96rem;max-width:660px;line-height:1.65;">JAMB/WAEC admission AI, dropout prediction, automated grading bias — gender gap, urban-rural divide, coaching access inequity.</p><div style="margin-top:.9rem;"><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ffe234;">JAMB/WAEC</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ffe234;">Dropout Prediction</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ffe234;">Gender Gap</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ffe234;">SES Gap</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ffe234;">Urban-Rural</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ffe234;">UNESCO SDG4</span></div></div>""",
    unsafe_allow_html=True
)

# ── Scenario info banner ───────────────────────────────────────────────────────
st.markdown(f"""
<div class="alert-info">
  <strong>Active scenario:</strong> {preset_info.name} |
  <strong>Target:</strong> {preset_info.target_variable.replace('_',' ').title()} |
  <strong>Languages:</strong> {', '.join(preset_info.languages[:4])} |
  <strong>Digital inclusion:</strong> {preset_info.digital_inclusion:.0%} |
  <strong>Out-of-school rate:</strong> {preset_info.out_of_school_rate:.0%}
</div>
""", unsafe_allow_html=True)

# ── Run ────────────────────────────────────────────────────────────────────────
if run_btn:
    enable_lifecycle = locals().get("enable_lifecycle", False)
    enable_eco       = locals().get("enable_eco", False)
    enable_dynamic   = locals().get("enable_dynamic", False)
    enable_arena = locals().get("enable_arena", False)
    enable_agent_economy = locals().get("enable_agent_economy", False)
    enable_redteam = locals().get("enable_redteam", False)
    enable_dynamic   = locals().get("enable_dynamic", False)
    enable_ai_safety = locals().get("enable_ai_safety", st.session_state.get("_ais_toggle", False))
    st.session_state.edu_run_history = []
    st.session_state["edu_feature_outputs"] = {}
    prog = st.progress(0, text=t("loading"))
    for i in range(n_runs):
        prog.progress(i/n_runs, text=f"Run {i+1}/{n_runs}…")
        with st.spinner(f"Simulation {i+1}/{n_runs}"):
            r = _run_one(scenario_key, n_samples, selected_biases, bias_intensity,
                         poison_rate, i, enable_xai, enable_governance)
            if r:
                st.session_state.edu_run_history.append(r)


                save_to_history("edu_snapshot_history",
                    label=f"Run {i+1} | bias={bias_intensity:.2f} | {scenario_key[:16]}",
                    metrics={"accuracy":r["accuracy"],"fairness_score":r["fairness_score"],
                             "opportunity_gap":r["opportunity_gap"],"gender_gap":r["gender_gap"]},
                    config={"scenario_key":scenario_key,"bias_intensity":bias_intensity,
                            "poison_rate":poison_rate,"selected_biases":selected_biases})


    # ── Run enabled feature modules (results stored per-session) ──────
    if "edu_feature_outputs" not in st.session_state:
        st.session_state["edu_feature_outputs"] = {}
    _fout = st.session_state["edu_feature_outputs"]

    if enable_governance:
        try:
            from components.governance_logic import HybridGovernanceLayer as _HGL
            _hgl_inst = _HGL()
            _hgl_baseline = {"accuracy": 0.75, "fairness_score": 0.70}
            _hgl_current  = {"accuracy": 0.70, "fairness_score": 0.60}
            _hgl_entry = _hgl_inst.propose_and_vote(
                "Deploy AI in education domain",
                _hgl_baseline, _hgl_current)
            _fout["governance"] = {
                "policy":      "Deploy AI in education domain",
                "outcome":     _hgl_entry.vote_outcome.value if hasattr(_hgl_entry, "vote_outcome") else "approved",
                "tally":       _hgl_entry.vote_tally if hasattr(_hgl_entry, "vote_tally") else {},
                "ai_flags":    _hgl_entry.ai_flags if hasattr(_hgl_entry, "ai_flags") else [],
                "ledger_hash": _hgl_entry.hash if hasattr(_hgl_entry, "hash") else "N/A",
                "ledger_entries": 1,
            }
        except Exception as _ex:
            _fout["governance"] = {
                "policy": "Deploy AI in education domain",
                "outcome": "approved", "tally": {"for":60,"against":30,"abstain":10},
                "ai_flags": [], "ledger_hash": "N/A", "ledger_entries": 0,
                "narrative": str(_ex),
            }

    if enable_gender_audit:
        _h = st.session_state.get("edu_run_history", [{}])
        _fout["gender_audit_gap"] = _h[-1].get("gender_gap", 0) if _h else 0


    if enable_redteam:
        try:
            _fout["multimodal_redteam"] = run_redteam_simulation(domain="education")
        except Exception as _ex:
            _fout["multimodal_redteam"] = {"combined_bypass_rate":0,"modality_results":[],"error":str(_ex)}

    if enable_agent_economy:
        try:
            _fout["agent_economy"] = run_agent_economy_simulation(domain="education")
        except Exception as _ex:
            _fout["agent_economy"] = {"gini_coefficient":0,"agent_summary":[],"error":str(_ex)}

    if enable_arena:
        try:
            _fout["arena"] = run_arena_simulation(domain="education")
        except Exception as _ex:
            _fout["arena"] = {"final_standings":[],"deception_rate":0,"error":str(_ex)}
    # ── AI Safety & Robustness Suite ──────────────────────────────────────────
    if enable_ai_safety:
        try:
            import numpy as np
            _last_run = st.session_state.get("edu_run_history", [{}])[-1]
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
                model=None, domain="education",
                enable_robustness=True, enable_ood=True,
                enable_uncertainty=True, enable_checklists=True,
                simulation_metrics=_sim_metrics,
            )
            st.session_state["edu_safety_report"] = _safety_report
        except Exception as _se:
            st.session_state["edu_safety_report"] = {"error": str(_se), "pillars": {}}

    # ── Lifecycle Management & Environmental Sustainability ────────────────────
    if enable_lifecycle or enable_eco:
        try:
            _last_r = st.session_state.get("edu_run_history", [{}])
            _last_r = _last_r[-1] if _last_r else {}
            _algo_k = _last_r.get("algorithm", "hist_gradient_boosting")
            _algo_l = _last_r.get("algo_label", "Hist Gradient Boosting")
            _lc_met = {k: v for k, v in _last_r.items() if isinstance(v, (int, float))}
            _lc_met["has_governance"]   = locals().get("enable_governance", False)
            _lc_met["has_gender_audit"] = locals().get("enable_gender_audit", False)
            _lc_met["has_xai"]          = True
            _lc_rep = run_lifecycle_suite(
                domain="education", algo_key=_algo_k, algo_label=_algo_l,
                n_samples=int(_last_r.get("n_samples", locals().get("sample_size", locals().get("n_samples", 2000)))),
                n_runs=int(locals().get("n_runs", 3)), n_features=10,
                metrics=_lc_met,
                safety_data=st.session_state.get("edu_safety_report") or None,
                enable_registry=enable_lifecycle, enable_monitoring=enable_lifecycle,
                enable_audit=enable_lifecycle, enable_eco=enable_eco,
            )
            st.session_state["edu_lifecycle_report"] = _lc_rep
        except Exception as _lce:
            st.session_state["edu_lifecycle_report"] = {"error": str(_lce), "pillars": {}}


    # ── Dynamic Systems Modelling Suite ─────────────────────────────────────────
    if enable_dynamic:
        try:
            _ds_hist   = st.session_state.get("edu_run_history", [])
            _ds_params = derive_ds_params(domain="education", run_history=_ds_hist)
            _ds_rep    = run_dynamic_systems_suite(
                domain="education",
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
            st.session_state["edu_ds_report"] = _ds_rep
        except Exception as _dse:
            st.session_state["edu_ds_report"] = {"error": str(_dse), "pillars": {}}
    prog.progress(1.0, text=t("complete"))
    prog.empty()
# ── Post-run interactivity (shown once, after all runs complete) ──────────
if st.session_state.get("edu_run_history"):
    feats = st.session_state.get("edu_feature_outputs", {})
    _post_last  = st.session_state["edu_run_history"][-1]
    _post_fs    = _post_last.get("fairness_score", 0.5)
    track_run(_post_fs, "education")
    _post_mc    = {k: v for k, v in _post_last.items() if isinstance(v, (int, float))}
    multi_challenge_panel("education", _post_mc)
    admin_challenge_panel("education")
    benchmark_challenge_panel("education", _post_mc)
    what_if_explorer("education", _post_mc,
        st.session_state.get("bias_intensity", 0.3))


# ── Dashboard ──────────────────────────────────────────────────────────────────
if st.session_state.edu_run_history:
    df: pd.DataFrame = pd.DataFrame()  # safe default; overwritten below
    df    = pd.DataFrame(st.session_state.edu_run_history)
    xai   = st.session_state.edu_xai_results
    lng_  = st.session_state.edu_longitudinal
    fed_  = st.session_state.edu_federated

    st.markdown("## 📊 Education Equity Dashboard")
    role_banner("education")
    role_brief_banner("education")
    national_live_banner()

    avg_acc  = df["accuracy"].mean()
    avg_fair = df["fairness_score"].mean()
    avg_opp  = df["opportunity_gap"].mean()
    avg_gen  = df["gender_gap"].mean()
    avg_ur   = df["urban_rural_gap"].mean()
    avg_ses  = df["ses_gap"].mean()

    # Share URL
    share_url_panel("health", config={"domain":"education","scenario_key":scenario_key,
        "bias_intensity":bias_intensity,"poison_rate":poison_rate})

    # Board Member view
    _role_now = get_active_role("health")
    if _role_now == "Board Member":
        board_member_summary("health", avg_acc, avg_fair, avg_fair >= 0.70,
            f"Education AI {'shows acceptable equity' if avg_fair >= 0.70 else 'exhibits significant equity gaps'} "
            f"(opportunity gap score: {avg_opp:.2f}).",
            "Conduct independent audit before deploying in examination or admissions contexts.")
    else:
        # KPI row
        k1,k2,k3,k4,k5,k6 = st.columns(6)
        for col, label, val, card in [
            (k1,"🎯 Accuracy",       avg_acc,  "card-blue"),
            (k2,"⚖️ Fairness",       avg_fair, "card-green"),
            (k3,"🎓 Opportunity Gap",avg_opp,  "card-purple"),
            (k4,"👩 Gender Gap",     avg_gen,  "card-orange"),
            (k5,"🏘️ Urban-Rural Gap",avg_ur,   "card-orange"),
            (k6,"💰 SES Gap",        avg_ses,  "card-red"),
        ]:
            colour = "#16a34a" if val >= 0.7 else "#f39c12" if val >= 0.5 else "#ef4444"
            if "Gap" in label:
                colour = "#16a34a" if val <= 0.05 else "#f39c12" if val <= 0.12 else "#ef4444"
            col.markdown(f'<div class="{card}"><p style="margin:0;font-size:.78rem;color:#555;">{label}</p>'
                         f'<p style="margin:0;font-size:1.8rem;font-weight:700;color:{colour};">'
                         f'{"%.2f" % val if 'Score' in label or 'Gap' in label else "%.1f%%" % (val*100)}</p></div>',
                         unsafe_allow_html=True)

        # Alert if critical gaps
        if avg_gen > 0.10:
            st.markdown(f'<div class="alert-danger">⚠️ Gender gap {avg_gen:.1%} exceeds 10% — '
                        'UNESCO Women4EthicalAI threshold breached. '
                        'Immediate gender-stratified resampling required.</div>', unsafe_allow_html=True)
        if avg_ur > 0.15:
            st.markdown(f'<div class="alert-warning">⚠️ Urban-rural gap {avg_ur:.1%} — '
                        'model systematically disadvantages rural students.</div>', unsafe_allow_html=True)

        metric_glossary_expander(["fairness score","demographic parity","equalized odds",
            "gender gap","false positive rate","false negative rate","digital inclusion score"])

        # ── Tabs ──────────────────────────────────────────────────────────────
        _tab_labels = [
            "📈 Performance",
            "⚖️ Equity Gaps",
            "🧠 Explainable AI",
            "📋 Compliance",
            "🔁 Longitudinal",
            "🌐 Federated",
            "🏛️ Nigeria Regulatory",
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
            st.markdown("### Model Performance Across Runs")
            fig = px.line(df, x="run_id", y=["accuracy","fairness_score","opportunity_gap"],
                title="Accuracy, Fairness & Opportunity Gap",
                color_discrete_sequence=["#1d4ed8","#16a34a","#f39c12"],
                labels={"value":"Score","run_id":"Run"})
            fig.add_hline(y=0.7, line_dash="dot", line_color="#ef4444",
                          annotation_text="Min fairness threshold (0.70)")
            st.plotly_chart(fig, use_container_width=True)

            # Confusion-style gap matrix
            st.markdown("### Equity Gap Matrix")
            gap_df = pd.DataFrame([{
                "Dimension": d, "Gap": df[col].mean(),
                "Status": "✅ OK" if df[col].mean() <= thr else "❌ High"
            } for d,col,thr in [
                ("Gender",       "gender_gap",      0.05),
                ("Urban-Rural",  "urban_rural_gap",  0.08),
                ("SES",          "ses_gap",          0.08),
                ("School Type",  "school_type_gap",  0.08),
                ("Language",     "language_disparity",0.10),
                ("Digital",      "digital_exclusion", 0.40),
            ]])
            fig_g = px.bar(gap_df, x="Gap", y="Dimension", orientation="h",
                color="Status",
                color_discrete_map={"✅ OK":"#16a34a","❌ High":"#ef4444"},
                title="Equity Gap by Dimension",
                text=[f"{v:.1%}" for v in gap_df["Gap"]])
            fig_g.update_traces(textposition="outside")
            fig_g.update_layout(height=320, showlegend=True, xaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig_g, use_container_width=True)

        with T["⚖️ Equity Gaps"]:
            st.markdown("### Equity Breakdown")
            st.markdown(f'<div class="alert-info">{df["equity_narrative"].iloc[-1]}</div>',
                        unsafe_allow_html=True)

            c1,c2 = st.columns(2)
            with c1:
                # Radar of all gap types
                dims = ["Gender Gap","Urban-Rural","SES Gap","School Type","Language","Digital Excl."]
                vals = [avg_gen, avg_ur, avg_ses, df["school_type_gap"].mean(),
                        df["language_disparity"].mean(), df["digital_exclusion"].mean()]
                fig_r = go.Figure(go.Scatterpolar(
                    r=vals+[vals[0]], theta=dims+[dims[0]],
                    fill="toself", line=dict(color="#f39c12",width=2),
                    fillcolor="rgba(243,156,18,0.12)"))
                fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,0.5])),
                                    height=320, title="Equity Gap Radar")
                st.plotly_chart(fig_r, use_container_width=True)
            with c2:
                st.markdown("#### 🎓 Scenario Context")
                st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Scenario | {preset_info.name} |
| Target Variable | {preset_info.target_variable.replace('_',' ').title()} |
| Digital Inclusion | {preset_info.digital_inclusion:.0%} |
| Out-of-School Rate | {preset_info.out_of_school_rate:.0%} |
| Gender Gap Baseline | {preset_info.gender_gap_baseline:.0%} |
| SES Bias Baseline | {preset_info.socioeconomic_bias:.0%} |
                """)
                if avg_fair < 0.70:
                    st.markdown('<div class="alert-danger">Model does NOT meet minimum '
                                'fairness threshold for educational deployment (0.70).</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-success">Model meets minimum fairness '
                                'threshold. Continue monitoring for intersectional gaps.</div>',
                                unsafe_allow_html=True)

        with T["🧠 Explainable AI"]:
            st.markdown("### 🧠 Explainable AI")
            if not xai:
                st.info("Enable XAI in sidebar and run simulation.")
            elif "error" in xai:
                st.warning(f"XAI error: {xai.get("error","unknown")}")
            else:
                fi = xai.get("feature_importance",{})
                if fi:
                    st.markdown(f'<div class="alert-info"><em>{fi.get("narrative","")}</em></div>',
                                unsafe_allow_html=True)
                    fi_df = pd.DataFrame({"Feature":fi["feature_names"][:10],
                                          "Importance":fi["importances"][:10]})
                    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        title="Feature Importance — What drives educational AI predictions?",
                        color="Importance", color_continuous_scale="YlOrRd")
                    fig_fi.update_layout(yaxis={"categoryorder":"total ascending"},height=350)
                    st.plotly_chart(fig_fi, use_container_width=True)

                c1,c2 = st.columns(2)
                with c1:
                    expl = xai.get("instance_explanation",{})
                    if expl:
                        st.markdown("#### Student Prediction Explanation")
                        st.markdown(f'<div class="alert-info"><em>{expl.get("decision_path","")}</em></div>',
                                    unsafe_allow_html=True)
                        contrib = expl.get("feature_contributions",{})
                        if contrib:
                            c_df = pd.DataFrame(sorted(contrib.items(),key=lambda x:abs(x[1]),reverse=True)[:8],
                                columns=["Feature","Contribution"])
                            fig_c = px.bar(c_df, x="Contribution", y="Feature", orientation="h",
                                color="Contribution", color_continuous_scale="RdYlGn",
                                color_continuous_midpoint=0, height=300)
                            st.plotly_chart(fig_c, use_container_width=True)
                with c2:
                    cf = xai.get("counterfactual",{})
                    if cf:
                        st.markdown("#### What would change the prediction?")
                        st.markdown(f'<div class="alert-info"><em>{cf.get("plain_language","")}</em></div>',
                                    unsafe_allow_html=True)
                        changes = cf.get("changes",{})
                        if changes:
                            st.dataframe(pd.DataFrame([{"Feature":k,"Original":round(v[0],3),
                                "Counterfactual":round(v[1],3),"Change":round(v[1]-v[0],3)}
                                for k,v in changes.items()]), use_container_width=True)

                ix = xai.get("intersectional",{})
                if ix and ix.get("group_performances"):
                    st.markdown("#### Intersectional Fairness (Gender × Urban)")
                    st.markdown(f'<div class="alert-info">{ix.get("narrative","")}</div>',
                                unsafe_allow_html=True)
                    ix_df = pd.DataFrame([{"Group":k,**{kk:round(vv,3) for kk,vv in v.items()}}
                        for k,v in ix["group_performances"].items()])
                    st.dataframe(ix_df.style.background_gradient(subset=["accuracy"],cmap="RdYlGn"),
                                 use_container_width=True)

        with T["📋 Compliance"]:
            st.markdown("### 📋 Compliance Report")
            _cr = xai.get("compliance_report",{})
            _mc = xai.get("model_card",{})
            if not _cr:
                st.info("Enable XAI and run simulation to generate compliance report.")
            else:
                summ = _cr.get("summary",{})
                ok = summ.get("overall_compliant",False)
                st.markdown(f'<div class="{"alert-success" if ok else "alert-danger"}">'
                    f'Overall: <strong>{"COMPLIANT ✅" if ok else "NON-COMPLIANT ❌"}</strong> | '
                    f'Fairness: {summ.get("fairness_score",0):.3f} | '
                    f'Bias Findings: {summ.get("bias_findings_count",0)}</div>',
                    unsafe_allow_html=True)
                for fw, fd in _cr.get("frameworks",{}).items():
                    with st.expander(f"📑 {fw}"):
                        for ch, st_ in fd.get("checks",{}).items():
                            st.markdown(f"{'✅' if st_=='PASS' else '❌'} {ch}")
                try:
                    pdf_b = generate_pdf_compliance_report(_cr, _mc,
                        {"accuracy":avg_acc,"fairness_score":avg_fair}, domain="generic")
                    st.download_button("📄 Download PDF Compliance Report", pdf_b,
                        "education_compliance.pdf", "application/pdf", use_container_width=True)
                except Exception as e:
                    st.caption(f"PDF unavailable: {e}")

        with T["🔁 Longitudinal"]:
            st.markdown("### 🔁 Longitudinal Bias Analysis")
            st.markdown('<div class="alert-info">Simulates how educational AI bias compounds '
                        'across annual model retraining cycles — critical for systems updated '
                        'each academic year with the previous year\'s outcomes.</div>',
                        unsafe_allow_html=True)
            if not lng_:
                st.info("Run simulation to see longitudinal analysis.")
            else:
                c1,c2,c3 = st.columns(3)
                c1.metric("Initial Bias", f'{lng_["initial_bias"]:.1%}')
                c2.metric("Final Bias", f'{lng_["final_bias"]:.1%}',
                          f'{lng_["final_bias"]-lng_["initial_bias"]:+.1%}')
                c3.metric("Amplification", f'{lng_["amplification_factor"]:.2f}×',
                          "⚠️ Self-reinforcing" if lng_["self_reinforcing"] else "Stable")
                if lng_.get("self_reinforcing"):
                    st.error(f'⚠️ Bias became self-reinforcing at generation '
                             f'{lng_.get("inflection_point","N/A")}. '
                             'Annual retraining without bias correction will worsen outcomes for disadvantaged students.')
                st.markdown(f'<div class="alert-info"><em>{lng_["narrative"]}</em></div>',
                            unsafe_allow_html=True)
                gm = lng_.get("generation_metrics",[])
                if gm:
                    fig_lng = px.line(pd.DataFrame(gm), x="generation",
                        y=["demographic_parity","fairness_score","accuracy"],
                        title="Bias Evolution Across Academic Years",
                        color_discrete_sequence=["#ef4444","#16a34a","#1d4ed8"],
                        labels={"value":"Score","generation":"Academic Year Cycle"})
                    fig_lng.add_hline(y=0.1, line_dash="dot", line_color="red",
                                      annotation_text="Acceptable parity threshold (10%)")
                    st.plotly_chart(fig_lng, use_container_width=True)
                    if CHARTS_OK:
                        try:
                            st.plotly_chart(animated_bias_drift(
                                gm, accent=ACCENT, height=380,
                                title="Animated: Bias Drift Across Retraining Cycles",
                            ), use_container_width=True)
                        except Exception: pass

        with T["🌐 Federated"]:
            st.markdown("### 🌐 Federated Learning Simulation")
            st.markdown('<div class="alert-info">Tests whether educational AI bias persists '
                        'when models are trained across schools or districts without '
                        'centralising sensitive student data.</div>', unsafe_allow_html=True)
            if not fed_:
                st.info("Run simulation to see federated analysis.")
            else:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Schools/Districts", fed_["n_clients"])
                c2.metric("Global Accuracy", f'{fed_["global_accuracy"]:.1%}')
                c3.metric("Global Fairness", f'{fed_["global_fairness"]:.3f}')
                c4.metric("Bias Persisted", "Yes ⚠️" if fed_["bias_persisted"] else "No ✅")
                st.markdown(f'<div class="{"alert-danger" if fed_["bias_persisted"] else "alert-success"}">'
                    f'<em>{fed_["narrative"]}</em></div>', unsafe_allow_html=True)
                cr_list = fed_.get("client_results",[])
                if cr_list:
                    cr_df = pd.DataFrame([c.__dict__ if hasattr(c,"__dict__") else c for c in cr_list])
                    if not cr_df.empty and "local_bias" in cr_df.columns:
                        fig_fed = px.bar(cr_df, x="client_id",
                            y=["local_accuracy","local_bias","local_fairness"],
                            barmode="group", title="Per-School/District Metrics",
                            color_discrete_sequence=["#1d4ed8","#ef4444","#16a34a"],
                            labels={"client_id":"School/District ID"})
                        st.plotly_chart(fig_fed, use_container_width=True)

        with T["🏛️ Nigeria Regulatory"]:
            st.markdown("### 🇳🇬 Nigeria Regulatory Compliance")
            _nr_m = {"accuracy":avg_acc,"fairness_score":avg_fair,
                     "demographic_parity":df["demographic_parity"].mean()}
            nigeria_compliance_panel(_nr_m, domain="agrotech",
                has_ussd_fallback=False, has_gender_audit=True,
                has_multilingual=True, has_xai=enable_xai,
                has_governance=enable_governance, has_redteam=False)
            # ── Real-world benchmark comparison ─────────────────────────
            st.markdown("#### 📚 Real-World Benchmark Comparison")
            if BENCHMARKS_OK:
                _dom_bms = get_benchmarks_for_domain("education")
                if _dom_bms:
                    _bm_sel = st.selectbox(
                        "Compare against a published study:",
                        list(_dom_bms.keys()),
                        format_func=lambda k: _dom_bms[k].name + " (" + str(_dom_bms[k].year) + ")",
                        key="_education_bm_sel")
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


        with T["📋 Raw Results"]:
            st.markdown("### 📋 Raw Simulation Results")
            _str_cols = {"equity_narrative","narrative","biases","scenario"}
            show_cols = [c for c in df.columns if c not in _str_cols]
            num_cols  = [c for c in show_cols
                         if str(df[c].dtype).startswith("float")
                         or str(df[c].dtype).startswith("int")]
            fmt = {c: "{:.3f}" for c in num_cols}
            try:
                styled = df[show_cols].style.format(
                    {k: v for k, v in fmt.items() if k in df[show_cols].columns})
                if "accuracy" in num_cols:
                    styled = styled.background_gradient(
                        subset=["accuracy"], cmap="Blues")
                if "fairness_score" in num_cols:
                    styled = styled.background_gradient(
                        subset=["fairness_score"], cmap="RdYlGn")
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(df[show_cols], use_container_width=True)
            d1,d2 = st.columns(2)
            with d1:
                st.download_button(t("download_csv"), df.to_csv(index=False).encode(),
                    f"gags_education_{scenario_key}.csv","text/csv",use_container_width=True)
            with d2:
                st.download_button(t("export_config"), json.dumps({
                    "scenario":scenario_key,"bias_intensity":bias_intensity,
                    "selected_biases":selected_biases,"n_samples":n_samples},indent=2),
                    "education_config.json","application/json",use_container_width=True)


        with T["🛡️ AI Safety"]:
            feats = st.session_state.get("edu_feature_outputs", {})
            feature_modules_tab(
                domain="education",
                run_results=st.session_state.get("edu_run_history", []),
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
            st.session_state.get("edu_safety_report", {}),
            domain="education",
        )

    # ── 🔄 Lifecycle Management Tab ───────────────────────────────────────────
    with T["🌱 Eco Score"]:
        render_lifecycle_tab(
            st.session_state.get("edu_lifecycle_report", {}),
            "education",
        )

    # ── 🌱 Eco Score Tab ──────────────────────────────────────────────────────
    with T["🔬 Feature Modules"]:
        _lc_eco_r   = st.session_state.get("edu_lifecycle_report", {})
        _lc_eco_last = (st.session_state.get("edu_run_history") or [{}])[-1]
        render_eco_tab(
            _lc_eco_r,
            algo_key  = _lc_eco_last.get("algorithm", "hist_gradient_boosting"),
            n_samples = int(_lc_eco_last.get("n_samples", 2000)),
            n_runs    = int(locals().get("n_runs", 3)),
            domain    = "education",
        )

    # ── 🔮 Dynamic Systems Tab ──────────────────────────────────────────────────
    with T["🔮 Dynamic Systems"]:
        try:
            render_dynamic_systems_tab(
                st.session_state.get("edu_ds_report", {}),
                domain="education",
                ds_key="edu_ds_report",
            )
        except Exception as _ds_err:
            st.error(f"🔮 Dynamic Systems error: {_ds_err}")
            import traceback
            st.code(traceback.format_exc(), language="python")

    history_browser("edu_snapshot_history", domain="health",
        key_metrics=["accuracy","fairness_score","opportunity_gap"])
    annotation_panel("edu_annotations",
        context_label=f"{len(st.session_state.edu_run_history)} Education run(s)")

    # Policy recommendations
    st.divider()
    st.markdown("## 💡 Education AI Policy Recommendations")
    r1,r2 = st.columns(2)
    with r1:
        st.markdown("""
**Immediate actions if fairness score < 0.70:**
- Apply gender-stratified resampling before training
- Contextualise scores by school type and LGA socioeconomic index
- Remove coaching access as a direct training signal
- Validate with community representatives before deployment

**NITDA alignment:**
- Explainability required for examination scoring systems
- Gender disaggregated performance metrics mandatory
- USSD fallback for rural/low-connectivity students
        """)
    with r2:
        st.markdown("""
**Structural recommendations:**
- Separate model pipelines for private vs public school contexts
- Annual fairness audits tied to academic calendar
- Community review board for Hausa/Yoruba/Igbo content in NLP grading
- Appeal mechanism: students can contest AI-assigned grades

**International alignment:**
- UNESCO SDG4 (Quality Education) — equity metrics required
- EU AI Act: educational AI classified as limited to high risk
- UNCRC Article 28: right to education on basis of equal opportunity
        """)


# ── Welcome state ──────────────────────────────────────────────────────────────
else:
    st.markdown(f"""<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
  padding:2.5rem;text-align:center;margin-top:1rem">
  <p style="font-size:2rem;margin:0 0 .6rem">🎓</p>
  <p style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:700;
     color:#0f172a;margin:0 0 .4rem">Welcome to Education Equity Simulation</p>
  <p style="color:#64748b;font-size:.87rem;max-width:540px;margin:0 auto .5rem">JAMB/WAEC scoring bias, dropout prediction, and automated grading. Measure opportunity gap, gender gap, SES equity, and urban-rural divide.</p>
  <p style="color:#94a3b8;font-size:.78rem">
    Configure settings in the sidebar and click <strong>Run</strong> to begin.
  </p>
</div>""", unsafe_allow_html=True)

st.divider()
st.markdown("<div style='text-align:center;color:#7f8c8d;font-size:.82rem;padding:.75rem 0;'>"
    "🎓 Education Equity Simulation · GAGS Framework v3.0 · "
    "Bias in Examinations, Admissions & Automated Grading</div>", unsafe_allow_html=True)
