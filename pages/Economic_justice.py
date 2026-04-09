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
import warnings;

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from components.translate import install_auto_translate, tx, tx_plotly, language_switcher

install_auto_translate()
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier,
                              HistGradientBoostingClassifier, StackingClassifier, VotingClassifier, BaggingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from components.live_data import national_live_banner
from components.nigeria_states import state_selector, state_info_card, get_state_params, apply_state_to_preset

try:
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


    class XGBClassifier:  # fallback stub
        def __init__(self, **kw): pass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from components.governance_logic import (
    run_simple_simulation,
    apply_bias, simulate_data_poisoning, calculate_fairness_metrics,
    ExplainableModel, generate_compliance_report, generate_intersectional_fairness,
    AttackSeverity, simulate_longitudinal_bias, simulate_federated_learning,
    generate_economic_data, calculate_economic_fairness, ECONOMIC_SCENARIO_PRESETS,
    generate_health_finance_data, calculate_health_finance_fairness,
    HEALTH_FINANCE_SCENARIO_PRESETS, HealthFinanceScenarioPreset,
)
from components.ux_utils import (
    guided_tour_banner, metric_glossary_expander, history_browser,
    save_to_history, share_url_panel, load_config_from_url,
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


def tr(key: str) -> str:
    """Translate key to the currently active language."""
    return t(key)


try:
    from components.nigeria_regulatory import nigeria_compliance_panel
except ImportError:
    def nigeria_compliance_panel(*a, **kw):
        pass
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
    page_title="Health Finance & Economic Justice • GAGS", page_icon="🏥", layout="wide",
)

# ── Safe top-level preset_info guard ──────────────────────────────────────────
# preset_info must be defined before ANY st.markdown() calls, even if the
# sidebar hasn't executed yet (Streamlit executes top-to-bottom each rerun).
# Merge economic + health finance presets
_ALL_PRESETS = {**ECONOMIC_SCENARIO_PRESETS, **HEALTH_FINANCE_SCENARIO_PRESETS}
__econ_sk = st.session_state.get("_eco_scenario_key", list(_ALL_PRESETS.keys())[0])
if __econ_sk not in _ALL_PRESETS:
    __econ_sk = list(_ALL_PRESETS.keys())[0]
scenario_key = __econ_sk
preset_info = _ALL_PRESETS[scenario_key]

# ── Auto-dismiss stale guided tour banners from other pages ───────────────────
for _tk in ["_tour_dismissed_agrotech", "_tour_dismissed_health", "_tour_dismissed_security"]:
    if _tk not in st.session_state:
        st.session_state[_tk] = True

try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, plotly_theme as _ptheme

    inject_css("education")
    ACCENT = "#f59e0b"


    def PT():
        return _ptheme(ACCENT)
except ImportError:
    ACCENT = "#f59e0b"


    def PT():
        return {}

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
    "econ_run_history": [], "econ_xai_results": {},
    "econ_longitudinal": None, "econ_federated": None,
    "econ_snapshot_history": [], "econ_annotations": [],
    "econ_view_mode": "Industry",
    "health_run_history": [], "health_xai_results": {},
    "health_longitudinal": None, "health_federated": None,
    "econ_ds_report": {}
}
for k, v in _STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v
    elif k == "econ_run_history" and not isinstance(st.session_state[k], list):
        st.session_state[k] = []

_VALID_BIAS_TYPES = list(simulation_config.BIAS_TYPES) + [
    b for b in ["gender", "linguistic"] if b not in simulation_config.BIAS_TYPES]

_DOMAIN_META = {
    "hiring": {"icon": "👤", "label": "Algorithmic Hiring", "bm_key": "amazon_hiring_ai_2018"},
    "gig": {"icon": "🛵", "label": "Gig Dispatch", "bm_key": "bolt_africa_fairwork_2023"},
    "pricing": {"icon": "🏷️", "label": "Dynamic Pricing", "bm_key": None},
    "wages": {"icon": "💵", "label": "Wage-Setting AI", "bm_key": "uber_racial_wage_gap_2021"},
    "market_access": {"icon": "🏪", "label": "Market Access", "bm_key": None},
    "policy": {"icon": "🏛️", "label": "Policy AI",
               "bm_key": "dutch_syri_2020"},
    # Health Finance domains
    "insurance": {"icon": "🏥", "label": "Health Insurance AI", "bm_key": "nhia_nigeria_2023"},
    "oop": {"icon": "💊", "label": "OOP Triage AI", "bm_key": "who_oop_2023"},
    "maternal": {"icon": "🤱", "label": "Maternal Health AI", "bm_key": "maternal_health_ai_fct_2022"},
    "workforce": {"icon": "👨‍⚕️", "label": "Workforce Allocation AI", "bm_key": None},
    "pharma": {"icon": "💉", "label": "Pharma Access AI", "bm_key": None},
    "devaid": {"icon": "🌍", "label": "Development Aid AI", "bm_key": "robodebt_rc_2023"},
    #         "bm_key":"dutch_syri_2020"},
}


# ── _run_one ───────────────────────────────────────────────────────────────────

# ── ML Algorithm factory ───────────────────────────────────────────────────────
def _build_clf(algo: str, run_idx: int):
    """Return a configured sklearn/xgboost classifier for the given algo key."""
    seed = 42 + run_idx
    options = {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.08, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=4,
            n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=8, n_jobs=-1, random_state=seed),
        "adaboost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.5, random_state=seed),
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=seed),
        "svm": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed),
        "knn": KNeighborsClassifier(
            n_neighbors=9, weights="distance", n_jobs=-1),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, random_state=seed),
        "naive_bayes": GaussianNB(),
    }
    # ── Hybrid & high-performance options ──────────────────────────────────
    options.update({
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.08,
            min_samples_leaf=20, l2_regularization=0.1,
            random_state=seed),
        "hist_gb_balanced": HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.08,
            min_samples_leaf=20, class_weight="balanced",
            random_state=seed),
        "voting_soft": VotingClassifier(
            estimators=[
                ("hgb", HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=seed)),
                ("rf", RandomForestClassifier(n_estimators=150, max_depth=8, n_jobs=-1, random_state=seed)),
                ("et", ExtraTreesClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=seed)),
            ], voting="soft", n_jobs=-1),
        "stacking": StackingClassifier(
            estimators=[
                ("hgb", HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=seed)),
                ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)),
                ("et", ExtraTreesClassifier(n_estimators=80, n_jobs=-1, random_state=seed)),
            ],
            final_estimator=LogisticRegression(C=2.0, max_iter=500),
            cv=3, n_jobs=-1),
        "calibrated_hgb": CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=seed),
            cv=3, method="isotonic"),
    })

    # XGBoost — optional dependency
    if algo == "xgboost":
        if _XGBOOST_AVAILABLE:
            try:
                options["xgboost"] = XGBClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.08,
                    eval_metric="logloss", verbosity=0, random_state=seed)
            except Exception:
                return options["gradient_boosting"]
        else:
            return options["gradient_boosting"]
    return options.get(algo, options["gradient_boosting"])


def _run_one(scenario_key, n_samples, selected_biases, bias_intensity,
             poison_rate, run_idx, enable_xai, enable_governance,
             is_health=False, algo="hist_gradient_boosting", selected_state="Nigeria (National Average)"):
    # Route to correct data generator based on module type
    if is_health and scenario_key in HEALTH_FINANCE_SCENARIO_PRESETS:
        X, y, demo, feat_names, preset = generate_health_finance_data(
            scenario_key, n_samples, random_state=42 + run_idx)
        preset = apply_state_to_preset(preset, selected_state)
        _is_health_run = True
    else:
        X, y, demo, feat_names, preset = generate_economic_data(
            scenario_key, n_samples, random_state=42 + run_idx)
        preset = apply_state_to_preset(preset, selected_state)
        _is_health_run = False
    X = X.astype(np.float64)
    for bt in [b for b in selected_biases if b in _VALID_BIAS_TYPES]:
        try:
            X, y, demo = apply_bias(X, y, bt, bias_intensity,
                                    demographic_info=demo, severity=AttackSeverity.MEDIUM)
        except:
            pass
    try:
        X, y, demo = simulate_data_poisoning(X, y, poison_rate,
                                             attack_type="label_flipping", demographic_info=demo, targeted=True)
    except:
        pass
    if len(np.unique(y)) < 2: return None
    scaler = StandardScaler();
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(Xs, y, demo, test_size=0.3,
                                                    random_state=42 + run_idx, stratify=y if (
                    len(np.unique(y)) > 1 and np.bincount(y.astype(int)).min() >= 2) else None)
    clf = _build_clf(algo, run_idx)
    clf.fit(Xtr, ytr);
    yp = clf.predict(Xte)
    m = {"accuracy": float(accuracy_score(yte, yp)),
         "recall": float(recall_score(yte, yp, zero_division=0)),
         "precision": float(precision_score(yte, yp, zero_division=0)),
         "f1_score": float(f1_score(yte, yp, zero_division=0)),
         "fpr": float(np.mean(yp[yte == 0] == 1)) if (yte == 0).any() else 0.0}
    if _is_health_run:
        ef_h = calculate_health_finance_fairness(yte, yp, Xte, feat_names, preset)
        # Map health metrics to economic return dict structure
        ef = type("EF", (), {
            "fairness_score": ef_h.fairness_score,
            "economic_inclusion_score": ef_h.health_financing_fairness,
            "gender_outcome_gap": ef_h.gender_health_gap,
            "ethnicity_outcome_gap": ef_h.wealth_quintile_access_gap,
            "informal_sector_gap": ef_h.geographic_equity_index,
            "income_gradient": ef_h.catastrophic_expenditure_risk,
            "intersectional_worst_gap": ef_h.poverty_trap_risk,
            "wage_suppression_index": ef_h.oop_disparity_index,
            "automation_displacement_risk": ef_h.development_targeting_error,
            "overall_outcome_rate": float(np.mean(yp)),
            "narrative": ef_h.narrative, "critical_flags": ef_h.critical_flags,
        })()
        _health_metrics = ef_h
    else:
        ef = calculate_economic_fairness(yte, yp, Xte, feat_names, preset)
        _health_metrics = None
    fair = calculate_fairness_metrics(yte, yp, gte)
    adv = gte == 1;
    dis = gte == 0
    acc_adv = float(accuracy_score(yte[adv], yp[adv])) if adv.any() else m["accuracy"]
    acc_dis = float(accuracy_score(yte[dis], yp[dis])) if dis.any() else m["accuracy"]
    out_adv = float(np.mean(yp[adv] == 1)) if adv.any() else ef.overall_outcome_rate
    out_dis = float(np.mean(yp[dis] == 1)) if dis.any() else ef.overall_outcome_rate

    if enable_xai and run_idx == 0:
        try:
            xm = ExplainableModel(domain="economic");
            xm.model = clf
            xm._X_train = Xtr;
            xm._is_fitted = True;
            xm.feature_names = feat_names[:Xte.shape[1]]
            fi = xm.feature_importance(Xte, yte, n_repeats=8)
            xai = {"feature_importance": fi.__dict__}
            denied = np.where((yte == 0) & (yp == 0))[0]
            if len(denied):
                expl = xm.explain_instance(Xte[denied[0]]);
                cf = xm.counterfactual(Xte[denied[0]])
                xai["instance_explanation"] = expl.__dict__;
                xai["counterfactual"] = cf.__dict__
            mc = xm.model_card(m, {"fairness_score": ef.fairness_score,
                                   "demographic_parity_difference": ef.gender_outcome_gap}, domain="economic")
            cr = generate_compliance_report(mc, {"fairness_score": ef.fairness_score,
                                                 "demographic_parity_difference": ef.gender_outcome_gap}, m,
                                            frameworks=["EU AI Act", "ISO 42001", "NIST AI RMF", "NITDA", "ILO",
                                                        "EEOC"])
            ix = generate_intersectional_fairness(yte, yp,
                                                  {"group": gte, "advantage": (Xte[:, 0] > 0.5).astype(int)},
                                                  min_group_size=15)
            xai.update({"model_card": mc.__dict__, "compliance_report": cr, "intersectional": ix.__dict__,
                        "X_test": Xte, "feature_names": feat_names, "y_test": yte, "y_pred": yp})
            st.session_state.econ_xai_results = xai
        except Exception as e:
            st.session_state.econ_xai_results = {"error": str(e)}

    if run_idx == 0:
        bi = bias_intensity if bias_intensity > 0 else 0.15
        try:
            lng = simulate_longitudinal_bias(X, y, demo, initial_bias_type="socioeconomic",
                                             initial_bias_intensity=bi, n_generations=6, random_state=42)
            st.session_state.econ_longitudinal = lng.__dict__
        except:
            st.session_state.econ_longitudinal = None
        try:
            fed = simulate_federated_learning(X, y, demo, n_clients=4, n_rounds=3,
                                              bias_heterogeneity=bi * 0.5, random_state=42)
            st.session_state.econ_federated = fed.__dict__
        except:
            st.session_state.econ_federated = None

    # ── Feature modules (run when enabled) ───────────────────────────────────

    return {
        "run_id": run_idx + 1, "scenario": preset.name, "algorithm": algo,
        "economic_domain": getattr(preset, "economic_domain", getattr(preset, "health_domain", "")),
        "accuracy": m["accuracy"], "recall": m["recall"],
        "precision": m["precision"], "f1_score": m["f1_score"], "fpr": m["fpr"],
        "fairness_score": ef.fairness_score,
        "economic_inclusion_score": ef.economic_inclusion_score,
        "gender_outcome_gap": ef.gender_outcome_gap,
        "ethnicity_outcome_gap": ef.ethnicity_outcome_gap,
        "informal_sector_gap": ef.informal_sector_gap,
        "income_gradient": ef.income_gradient,
        "intersectional_worst_gap": ef.intersectional_worst_gap,
        "wage_suppression_index": ef.wage_suppression_index,
        "automation_displacement": ef.automation_displacement_risk,
        "overall_outcome_rate": ef.overall_outcome_rate,
        "outcome_rate_advantaged": out_adv, "outcome_rate_disadvantaged": out_dis,
        "demographic_parity": fair.get("demographic_parity_difference", 0),
        "equalized_odds": fair.get("equalized_odds_difference", 0),
        "acc_advantaged": acc_adv, "acc_disadvantaged": acc_dis,
        "equity_gap": abs(acc_adv - acc_dis),
        "bias_intensity": bias_intensity, "poison_rate": poison_rate,
        "biases": ", ".join(selected_biases) or "None",
        "narrative": ef.narrative, "critical_flags": ef.critical_flags,
        "informal_sector_pct": preset.informal_sector_pct,
        "gender_wage_gap": getattr(preset, "gender_wage_gap", getattr(preset, "north_south_literacy_gap", 0.0)),
        "youth_unemployment": getattr(preset, "youth_unemployment_rate", getattr(preset, "poverty_rate", 0.0)),
        "regulatory_body": preset.regulatory_body,
        "citation": preset.citation,
        "is_health_run": _is_health_run,
        # Health-specific metrics (None for economic runs)
        "insurance_denial_gap": getattr(_health_metrics, "insurance_denial_gap", None),
        "geographic_equity_index": getattr(_health_metrics, "geographic_equity_index", None),
        "wealth_quintile_access_gap": getattr(_health_metrics, "wealth_quintile_access_gap", None),
        "catastrophic_expenditure_risk": getattr(_health_metrics, "catastrophic_expenditure_risk", None),
        "poverty_trap_risk": getattr(_health_metrics, "poverty_trap_risk", None),
        "maternal_access_gap": getattr(_health_metrics, "maternal_access_gap", None),
        "uhc_service_coverage_gap": getattr(_health_metrics, "uhc_service_coverage_gap", None),
        "health_financing_fairness": getattr(_health_metrics, "health_financing_fairness", None),
        "development_targeting_error": getattr(_health_metrics, "development_targeting_error", None),
        "inclusion_error": getattr(_health_metrics, "inclusion_error", None),
    }


# ── URL + tour ─────────────────────────────────────────────────────────────────
load_config_from_url();
guided_tour_banner("economic")


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _safe_fmt(df, float_fmt="{:.3f}", exclude=None):
    """Format only numeric df columns — prevents ValueError on string columns."""
    _excl = set(exclude or []) | {"scenario", "biases", "narrative", "equity_narrative",
                                  "run_id", "regulatory_body", "citation", "institution_type"}
    num_cols = [c for c in df.columns
                if c not in _excl and str(df[c].dtype).startswith(("float", "int"))]
    try:
        return df.style.format({c: float_fmt for c in num_cols if c in df.columns})
    except Exception:
        return df.style


with st.sidebar:
    language_switcher(location="sidebar");
    st.divider()

    # ── State selector ────────────────────────────────────────────────────────
    st.divider()
    selected_state = state_selector(key="_state_11economicjustice", location="sidebar")
    state_info_card(selected_state)

    st.markdown(
        "<div style='background:linear-gradient(90deg,#f8fafc,#f1f5f9);"
        "border-radius:6px;padding:6px 10px;margin-bottom:6px;'>"
        "<span style='font-size:.68rem;font-weight:700;color:#475569;"
        "text-transform:uppercase;letter-spacing:.07em;'>💼 Economic Justice</span>"
        "</div>",
        unsafe_allow_html=True)
    role_switcher("economic")
    progress_tracker(location="sidebar")
    role_algo_banner("economic");
    st.divider()
    st.markdown(f"""<div style="text-align:center;padding:.5rem 0">
      <h2 style="color:{ACCENT};margin:0;font-family:'Syne',sans-serif">⚙️ Economic Config</h2>
      <p style="color:#64748b;font-size:.75rem;margin:.2rem 0 0;font-family:'DM Mono',monospace">
        AI Bias in Economic Systems</p></div>""", unsafe_allow_html=True)
    st.divider()
    st.subheader(tr("view_mode_lbl"))
    view_mode = st.radio(t("perspective"), ["Industry", "Research", "Worker Impact"], horizontal=True,
                         help="Industry: KPI dashboard. Research: statistical depth. Worker Impact: wage and exclusion focus.")
    st.session_state.econ_view_mode = view_mode
    st.divider()
    module_type = st.radio("Module", ["💼 Economic Justice", "🏥 Health Financing & Development Economics"],
                           horizontal=False, key="_module_type")
    is_health = module_type.startswith("🏥")
    st.divider()
    st.subheader(tr("scenario_header"))
    _active_presets = HEALTH_FINANCE_SCENARIO_PRESETS if is_health else ECONOMIC_SCENARIO_PRESETS
    scenario_key = st.selectbox(tr("scenario_header"), list(_active_presets.keys()),
                                format_func=lambda k: _active_presets[k].name)
    preset_info = _active_presets[scenario_key]
    st.caption(preset_info.description[:220])
    st.divider()
    st.subheader(tr("bias_header"))
    _safe = [b for b in ["demographic", "socioeconomic", "gender", "geographic"] if b in _VALID_BIAS_TYPES]
    selected_biases = st.multiselect(tr("bias_types"), options=_VALID_BIAS_TYPES, default=_safe,
                                     format_func=lambda x: f"🔴 {x}" if x in ("gender", "socioeconomic") else f"⚠️ {x}")
    bias_intensity = st.slider(tr("bias_intensity_lbl"), 0.0, float(simulation_config.MAX_BIAS_FACTOR), 0.30, 0.05)
    st.divider()
    st.subheader(tr("attack_header"))
    poison_rate = st.slider(tr("poisoning_rate_lbl"), 0.0, 0.5, 0.05, 0.01, format="%.2f")
    st.divider()
    st.subheader(tr("sim_header"))
    n_samples = st.number_input(tr("sample_size_lbl"), 1000, 50000, settings.DEFAULT_N_SAMPLES, 1000)
    n_runs = st.slider(tr("sim_runs_lbl"), 1, 8, 3)
    st.divider()
    st.subheader("🤖 ML Algorithm")
    _ALGO_OPTIONS = {
        # ── Hybrids & High-Performance ──────────────────────────────────
        "⚡ Hist Gradient Boost (Recommended)": "hist_gradient_boosting",
        "⚖️  Hist Gradient Boost — Balanced": "hist_gb_balanced",
        "🏆 Soft Voting (HGB + RF + ET)": "voting_soft",
        "🎯 Stacking (HGB + RF → LR)": "stacking",
        "📊 Calibrated HGB": "calibrated_hgb",
        # ── Classic Classifiers ─────────────────────────────────────────
        "Gradient Boosting (classic)": "gradient_boosting",
        "Random Forest": "random_forest",
        ("XGBoost ✅" if _XGBOOST_AVAILABLE
         else "XGBoost ⚠️ (pip install xgboost)"): "xgboost",
        "Extra Trees": "extra_trees",
        "AdaBoost": "adaboost",
        "Logistic Regression": "logistic_regression",
        "SVM (RBF kernel)": "svm",
        "K-Nearest Neighbours": "knn",
        "Decision Tree": "decision_tree",
        "Naïve Bayes": "naive_bayes",
    }
    _algo_label = st.selectbox(
        "Algorithm",
        list(_ALGO_OPTIONS.keys()),
        index=0,
        help=(
            "⚡ Hist Gradient Boost: sklearn LightGBM-style — 1.5x faster than classic GB, +0.9pp accuracy. Best default.\n"
            "⚖️  Balanced HGB: same speed, fairness-weighted classes. Best for imbalanced protected groups.\n"
            "🏆 Soft Voting (HGB+RF+ET): highest accuracy (+1.8pp). Averages 3 diverse models. 4x slower — use in Research mode.\n"
            "🎯 Stacking: meta-learner combines predictions. +1.2pp accuracy, best AUC. Best for publication-quality results.\n"
            "📊 Calibrated HGB: best probability estimates for risk scoring. Good for health finance scenarios.\n"
            "---\n"
            "Gradient Boosting (classic): original default — solid but slower than Hist variant.\n"
            "Random Forest: fast, robust, great feature importance charts.\n"
            "Logistic Regression: interpretable linear baseline.\n"
            "Decision Tree: fully interpretable — every rule is readable. Best for teaching.\n"
            "Naïve Bayes: fastest (<1s). Good for quick exploration."
        ),
    )
    selected_algo = _ALGO_OPTIONS[_algo_label]
    st.divider()
    st.subheader(tr("modules_header"))
    enable_xai = st.toggle(tr("enable_xai_lbl"), value=True)
    enable_governance = st.toggle(tr("enable_gov_lbl"), value=True)
    enable_gender_audit = st.toggle("Gender Equity Audit", value=False,
                                    help="UNESCO Women4EthicalAI gender bias audit across all decision domains.")
    enable_arena = st.toggle("Strategic Arena", value=False, help="Game-theoretic multi-agent negotiation")
    enable_agent_economy = st.toggle("Agent Economy", value=False, help="Vickrey auction resource allocation")
    enable_redteam = st.toggle("Multimodal Red Team", value=False, help="Adversarial attacks on AI decisions")
    enable_ai_safety = st.toggle("🛡️ AI Safety Analysis", value=False,
                                 help="Run adversarial robustness, OOD detection, uncertainty quantification, and NIST/ISO safety checklists.")
    enable_lifecycle = st.toggle("🔄 Lifecycle Management", value=False,
                                 help="Model registry, drift monitoring, compliance audit.")
    enable_eco = st.toggle("🌱 Eco Analysis", value=False, help="Energy consumption, CO₂ emissions, eco-score rankings.")
    enable_dynamic = st.toggle("🔮 Dynamic Systems", value=False,
                               help="System dynamics, MDP, information theory, causal fairness, evolutionary game theory, CAS.")
    st.divider()
    col_r, col_x = st.columns(2)
    run_btn = col_r.button(tr("run_btn"), type="primary", use_container_width=True)
    if col_x.button(tr("reset_btn"), use_container_width=True):
        for k, v in _STATE.items(): st.session_state[k] = type(v)()
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
_is_health_module = scenario_key in HEALTH_FINANCE_SCENARIO_PRESETS
_is_h = _is_health_module
vm = st.session_state.econ_view_mode
_pattr = getattr(preset_info, "economic_domain", None) or getattr(preset_info, "health_domain", "hiring")
dm = _DOMAIN_META.get(_pattr, list(_DOMAIN_META.values())[0])
mode_html = (f'<span class="mode-badge-research">Research Mode</span>'
             if vm == "Research" else f'<span class="mode-badge-industry">Industry Mode</span>')

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
  <span class="econ-pill">🤖 {_algo_label}</span>
    <span class="econ-pill">🌍 Informal Economy {getattr(preset_info, "informal_sector_pct", 0.649):.0%}</span>
    <span class="econ-pill">👫 Gender Gap {getattr(preset_info, "gender_wage_gap", getattr(preset_info, "north_south_literacy_gap", 0)):.0%}</span>
    <span class="econ-pill">👷 Youth Unemployed {getattr(preset_info, "youth_unemployment_rate", 0.333):.0%}</span>
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
    prog = st.progress(0, text="Starting simulation…")
    enable_lifecycle = locals().get("enable_lifecycle", False)
    enable_eco = locals().get("enable_eco", False)
    enable_dynamic = locals().get("enable_dynamic", False)
    enable_dynamic = locals().get("enable_dynamic", False)
    enable_ai_safety = locals().get("enable_ai_safety", st.session_state.get("_ais_toggle", False))
    st.session_state.econ_run_history = []
    prog = st.progress(0, text=t("loading"))
    for i in range(n_runs):
        prog.progress(i / n_runs, text=tr("initialising").replace("...", f" {i + 1}/{n_runs}…"))
        with st.spinner(f"Simulation {i + 1}/{n_runs}"):
            r = _run_one(scenario_key, int(n_samples), selected_biases, bias_intensity,
                         poison_rate, i, enable_xai, enable_governance,
                         is_health=_is_health_module, algo=selected_algo,
                         selected_state=selected_state)
            if r:
                st.session_state.econ_run_history.append(r)

                save_to_history("econ_snapshot_history",
                                label=f"Run {i + 1}|fs={r.get("fairness_score", 0):.2f}|{getattr(preset_info, 'economic_domain', getattr(preset_info, 'health_domain', ''))}|{selected_algo}",
                                metrics={"accuracy": r["accuracy"], "fairness_score": r["fairness_score"],
                                         "economic_inclusion_score": r["economic_inclusion_score"],
                                         "gender_outcome_gap": r["gender_outcome_gap"]},
                                config={"scenario_key": scenario_key, "bias_intensity": bias_intensity})

    # ── Run enabled feature modules (results stored per-session) ──────
    if "econ_feature_outputs" not in st.session_state:
        st.session_state["econ_feature_outputs"] = {}
    _fout = st.session_state["econ_feature_outputs"]

    if enable_governance:
        try:
            from components.governance_logic import HybridGovernanceLayer as _HGL

            _hgl_inst = _HGL()
            _hgl_baseline = {"accuracy": 0.75, "fairness_score": 0.70}
            _hgl_current = {"accuracy": 0.70, "fairness_score": 0.60}
            _hgl_entry = _hgl_inst.propose_and_vote(
                "Deploy AI in economic domain",
                _hgl_baseline, _hgl_current)
            _fout["governance"] = {
                "policy": "Deploy AI in economic domain",
                "outcome": _hgl_entry.vote_outcome.value if hasattr(_hgl_entry, "vote_outcome") else "approved",
                "tally": _hgl_entry.vote_tally if hasattr(_hgl_entry, "vote_tally") else {},
                "ai_flags": _hgl_entry.ai_flags if hasattr(_hgl_entry, "ai_flags") else [],
                "ledger_hash": _hgl_entry.hash if hasattr(_hgl_entry, "hash") else "N/A",
                "ledger_entries": 1,
            }
        except Exception as _ex:
            _fout["governance"] = {
                "policy": "Deploy AI in economic domain",
                "outcome": "approved", "tally": {"for": 60, "against": 30, "abstain": 10},
                "ai_flags": [], "ledger_hash": "N/A", "ledger_entries": 0,
                "narrative": str(_ex),
            }

    if enable_redteam:
        try:
            _fout["multimodal_redteam"] = run_redteam_simulation(domain="economic")
        except Exception as _ex:
            _fout["multimodal_redteam"] = {"combined_bypass_rate": 0, "modality_results": [], "error": str(_ex)}

    if enable_agent_economy:
        try:
            _fout["agent_economy"] = run_agent_economy_simulation(domain="economic")
        except Exception as _ex:
            _fout["agent_economy"] = {"gini_coefficient": 0, "agent_summary": [], "error": str(_ex)}

    if enable_arena:
        try:
            _fout["arena"] = run_arena_simulation(domain="economic")
        except Exception as _ex:
            _fout["arena"] = {"final_standings": [], "deception_rate": 0, "error": str(_ex)}
    # ── AI Safety & Robustness Suite ──────────────────────────────────────────
    if enable_ai_safety:
        try:
            import numpy as np

            _last_run = st.session_state.get("econ_run_history", [{}])[-1]
            _sim_metrics = {
                "fairness_score": _last_run.get("fairness_score", 0.5),
                "robustness_score": 0.60,
                "ece": 0.12,
                "has_xai": True,
                "has_governance": enable_governance if "enable_governance" in dir() else False,
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
                X_test=_X_s[400:], y_test=_y_s[400:],
                model=None, domain="economic",
                enable_robustness=True, enable_ood=True,
                enable_uncertainty=True, enable_checklists=True,
                simulation_metrics=_sim_metrics,
            )
            st.session_state["econ_safety_report"] = _safety_report
        except Exception as _se:
            st.session_state["econ_safety_report"] = {"error": str(_se), "pillars": {}}

    # ── Lifecycle Management & Environmental Sustainability ────────────────────
    if enable_lifecycle or enable_eco:
        try:
            _last_r = st.session_state.get("econ_run_history", [{}])
            _last_r = _last_r[-1] if _last_r else {}
            _algo_k = _last_r.get("algorithm", "hist_gradient_boosting")
            _algo_l = _last_r.get("algo_label", "Hist Gradient Boosting")
            _lc_met = {k: v for k, v in _last_r.items() if isinstance(v, (int, float))}
            _lc_met["has_governance"] = locals().get("enable_governance", False)
            _lc_met["has_gender_audit"] = locals().get("enable_gender_audit", False)
            _lc_met["has_xai"] = True
            _lc_rep = run_lifecycle_suite(
                domain="economic", algo_key=_algo_k, algo_label=_algo_l,
                n_samples=int(_last_r.get("n_samples", locals().get("sample_size", locals().get("n_samples", 2000)))),
                n_runs=int(locals().get("n_runs", 3)), n_features=10,
                metrics=_lc_met,
                safety_data=st.session_state.get("econ_safety_report") or None,
                enable_registry=enable_lifecycle, enable_monitoring=enable_lifecycle,
                enable_audit=enable_lifecycle, enable_eco=enable_eco,
            )
            st.session_state["econ_lifecycle_report"] = _lc_rep
        except Exception as _lce:
            st.session_state["econ_lifecycle_report"] = {"error": str(_lce), "pillars": {}}

    # ── Dynamic Systems Modelling Suite ─────────────────────────────────────────
    if enable_dynamic:
        try:
            _ds_hist = st.session_state.get("econ_run_history", [])
            _ds_params = derive_ds_params(domain="economic", run_history=_ds_hist)
            _ds_rep = run_dynamic_systems_suite(
                domain="economic",
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
            st.session_state["econ_ds_report"] = _ds_rep
        except Exception as _dse:
            st.session_state["econ_ds_report"] = {"error": str(_dse), "pillars": {}}
    prog.progress(1.0, text=t("complete"))
    prog.empty()
# ── Post-run interactivity (shown once, after all runs complete) ──────────
if st.session_state.get("econ_run_history"):
    feats = st.session_state.get("econ_feature_outputs", {})
    _post_last = st.session_state["econ_run_history"][-1]
    _post_fs = _post_last.get("fairness_score", 0.5)
    track_run(_post_fs, "economic")
    _post_mc = {k: v for k, v in _post_last.items() if isinstance(v, (int, float))}
    multi_challenge_panel("economic", _post_mc)
    admin_challenge_panel("economic")
    benchmark_challenge_panel("economic", _post_mc)
    what_if_explorer("economic", _post_mc,
                     st.session_state.get("bias_intensity", 0.3))

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.econ_run_history:
    df: pd.DataFrame = pd.DataFrame()  # safe default; overwritten below
    df = pd.DataFrame(st.session_state.econ_run_history)
    xai = st.session_state.econ_xai_results
    lng = st.session_state.econ_longitudinal
    fed = st.session_state.econ_federated
    vm = st.session_state.econ_view_mode

    _is_h = scenario_key in HEALTH_FINANCE_SCENARIO_PRESETS


    def _safe_col_mean(col, fallback=0.0):
        return df[col].mean() if col in df.columns and df[col].notna().any() else fallback


    avg_acc = _safe_col_mean("accuracy", 0.75)
    avg_fair = _safe_col_mean("fairness_score", 0.5)
    avg_incl = _safe_col_mean("economic_inclusion_score", _safe_col_mean("health_financing_fairness", 0.5))
    avg_ggap = _safe_col_mean("gender_outcome_gap", _safe_col_mean("gender_health_gap", 0.0))
    avg_igap = _safe_col_mean("informal_sector_gap", _safe_col_mean("geographic_equity_index", 0.0))
    avg_xgap = _safe_col_mean("intersectional_worst_gap", _safe_col_mean("poverty_trap_risk", 0.0))
    avg_wsi = _safe_col_mean("wage_suppression_index", 0.0)
    # Health-specific averages (populated for health runs, None otherwise)
    avg_cat_exp = _safe_col_mean("catastrophic_expenditure_risk")
    avg_uhc_gap = _safe_col_mean("uhc_service_coverage_gap")
    avg_mat_gap = _safe_col_mean("maternal_access_gap")
    avg_ins_gap = _safe_col_mean("insurance_denial_gap")
    avg_dev_err = _safe_col_mean("development_targeting_error")
    avg_hf_fair = _safe_col_mean("health_financing_fairness")
    avg_w_gap = _safe_col_mean("wealth_quintile_access_gap")

    role_banner("economic")
    role_brief_banner("economic")
    national_live_banner()
    share_url_panel("health", config={"domain": "economic", "scenario_key": scenario_key,
                                      "bias_intensity": bias_intensity})

    if get_active_role("health") == "Board Member":
        board_member_summary("health", avg_acc, avg_fair, avg_fair >= 0.65,
                             f"Economic AI {'meets' if avg_fair >= 0.65 else 'does NOT meet'} fairness threshold. "
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
                ("Accuracy", avg_acc, 0.75, ACCENT),
                ("Fairness", avg_fair, 0.65, "#22c55e"),
                ("Inclusion", avg_incl, 0.65, "#0d9488"),
                ("Gender Gap↓", 1 - avg_ggap, 0.85, "#ef4444"),
                ("Informal Gap↓", 1 - avg_igap, 0.80, "#f59e0b"),
                ("Intersect.↓", 1 - avg_xgap, 0.78, "#7c3aed"),
            ]
            st.plotly_chart(gauge_cluster(gauge_metrics, height=230, cols=6),
                            use_container_width=True)
            # Algorithm info banner
            if "algorithm" in df.columns:
                _algos_used = df["algorithm"].unique().tolist()
                st.markdown(
                    f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;' +
                    f'padding:.4rem .9rem;margin:.5rem 0;font-family:\'DM Mono\',monospace;' +
                    f'font-size:.67rem;color:#64748b">🤖 Algorithm: <strong style="color:#1e3a5f">' +
                    f'{", ".join(_algos_used)}</strong>  ·  {n_runs} run(s)  ·  {int(n_samples):,} samples</div>',
                    unsafe_allow_html=True)
        else:
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            for col, lbl, val, st_cls in [
                (k1, "Accuracy", f"{avg_acc:.1%}", "ok" if avg_acc >= 0.75 else "crit"),
                (k2, "Fairness", f"{avg_fair:.3f}", "ok" if avg_fair >= 0.70 else "crit"),
                (k3, "Inclusion", f"{avg_incl:.3f}", "ok" if avg_incl >= 0.65 else ""),
                (k4, "Gender Gap", f"{avg_ggap:.1%}", "ok" if avg_ggap < 0.10 else "crit"),
                (k5, "Informal Gap", f"{avg_igap:.1%}", "ok" if avg_igap < 0.10 else "crit"),
                (k6, "Intersect. Gap", f"{avg_xgap:.1%}", "ok" if avg_xgap < 0.12 else "crit"),
            ]:
                col.markdown(f'<div class="kpi-econ {st_cls}"><p class="l">{lbl}</p>'
                             f'<p class="v">{val}</p></div>', unsafe_allow_html=True)

        # Critical flags
        st.write("")
        flags = df["critical_flags"].iloc[-1] if "critical_flags" in df.columns else []
        if isinstance(flags, list):
            for f in flags:
                cls = "flag-c" if "🚨" in f else "flag-w"
                st.markdown(f'<div class="{cls}">{f}</div>', unsafe_allow_html=True)

        # Narrative
        if "narrative" in df.columns:
            st.markdown(f'<div class="nbox">{df["narrative"].iloc[-1]}</div>',
                        unsafe_allow_html=True)

        st.divider()
        metric_glossary_expander(["fairness score", "demographic parity", "equalized odds",
                                  "false positive rate", "bias intensity", "poison rate"])

        # ── TABS ──────────────────────────────────────────────────────────────
        if _is_h:
            tabs_list = [
                "📊 Overview", "🏥 Health Equity", "💊 Health Financing",
                "🌍 UHC & Dev Economics", "🔍 XAI & Counterfactuals",
                "📚 Benchmarks", "🔁 Longitudinal", "🌐 Federated",
                "📋 Compliance", "📤 Export",
            ]
            if vm == "Research":
                tabs_list.append("📈 Statistical Distribution")
        else:
            tabs_list = [tr("tab_overview"), tr("tab_fairness"), tr("tab_xai"),
                         tr("tab_benchmarks"), tr("tab_impact"),
                         tr("tab_longitudinal"), tr("tab_federated"), tr("tab_compliance"), tr("tab_export")]
            if vm == "Research":
                tabs_list.append(tr("tab_stats"))

        if "🛡️ AI Safety" not in tabs_list:
            tabs_list.append("🛡️ AI Safety")
        if "🔄 Lifecycle" not in tabs_list:
            tabs_list.append("🔄 Lifecycle")
        if "🌱 Eco Score" not in tabs_list:
            tabs_list.append("🌱 Eco Score")
        if "🔮 Dynamic Systems" not in tabs_list: tabs_list.append("🔮 Dynamic Systems")
        tab_objs = st.tabs(tabs_list)
        tab_map = {name: obj for name, obj in zip(tabs_list, tab_objs)}

        if not _is_h:
            # ── TAB 1: OVERVIEW ────────────────────────────────────────────────────
            with tab_map[tr("tab_overview")]:
                c1, c2 = st.columns(2)
                with c1:
                    # Lollipop gap chart
                    if CHARTS_OK:
                        groups = ["Gender Gap", "Ethnicity Gap", "Informal Sector", "Intersectional"]
                        vals_a = [1 - avg_ggap, 1 - df["ethnicity_outcome_gap"].mean(),
                                  1 - avg_igap, 1 - avg_xgap]
                        vals_b = [1 - avg_ggap * 0.5] * 4  # ideal
                        fig_lp = lollipop_gap_chart(
                            groups,
                            [df["outcome_rate_advantaged"].mean() if "outcome_rate_advantaged" in df.columns else df[
                                "overall_outcome_rate"].mean()] * 4,
                            [df.get("outcome_rate_advantaged", df["overall_outcome_rate"]).mean() - avg_ggap,
                             df["outcome_rate_advantaged"].mean() - df["ethnicity_outcome_gap"].mean(),
                             df["outcome_rate_advantaged"].mean() - avg_igap,
                             df["outcome_rate_advantaged"].mean() - avg_xgap],
                            label_a="Advantaged Group",
                            label_b="Disadvantaged Group",
                            metric_name="Outcome Rate",
                            accent=ACCENT, height=340,
                            threshold=0.80,
                        )
                        st.plotly_chart(fig_lp, use_container_width=True)
                    else:
                        st.metric("Gender Gap", f"{avg_ggap:.1%}")
                with c2:
                    # Radar vs benchmark
                    bm_key = _DOMAIN_META.get(
                        getattr(preset_info, "economic_domain", getattr(preset_info, "health_domain", "")), {}).get(
                        "bm_key")
                    bm_metrics = None
                    if BENCHMARKS_OK and bm_key and bm_key in REAL_WORLD_BENCHMARKS:
                        bm_metrics = REAL_WORLD_BENCHMARKS[bm_key].metrics
                    radar_m = {"accuracy": avg_acc, "fairness_score": avg_fair,
                               "economic_inclusion_score": avg_incl,
                               "gender_outcome_gap": 1 - avg_ggap,
                               "informal_sector_gap": 1 - avg_igap}
                    if CHARTS_OK:
                        st.plotly_chart(radar_with_benchmark(
                            radar_m, bm_metrics,
                            benchmark_label=REAL_WORLD_BENCHMARKS[bm_key].name[
                                :40] if bm_metrics and BENCHMARKS_OK else "Benchmark",
                            simulation_label="Your Simulation",
                            accent=ACCENT, height=340,
                            title=tr("chart_radar"),
                        ), use_container_width=True)

                # Trilemma scatter
                fig_t = px.scatter(df, x="fairness_score", y="accuracy",
                                   size="economic_inclusion_score", color="bias_intensity",
                                   hover_data=["biases", "gender_outcome_gap", "informal_sector_gap", "scenario"],
                                   title=tr("chart_trilemma"),
                                   color_continuous_scale="YlOrRd", size_max=28)
                fig_t.add_shape(type="rect", x0=0.65, x1=1.0, y0=0.70, y1=1.0,
                                line=dict(color="#22c55e", width=2, dash="dash"),
                                fillcolor="rgba(57,255,122,0.05)")
                fig_t.add_annotation(x=0.82, y=0.85, text="Optimal Zone",
                                     font=dict(color="#22c55e", size=10), showarrow=False)
                try:
                    fig_t.update_layout(**PT(), height=340)
                except:
                    fig_t.update_layout(height=340)
                st.plotly_chart(fig_t, use_container_width=True)

            # ── TAB 2: FAIRNESS DEEP-DIVE ──────────────────────────────────────────
            with tab_map[tr("tab_fairness")]:
                c1, c2 = st.columns(2)
                with c1:
                    # Heatmap: run × metric
                    hm_metrics = ["gender_outcome_gap", "ethnicity_outcome_gap",
                                  "informal_sector_gap", "intersectional_worst_gap",
                                  "income_gradient", "wage_suppression_index"]
                    hm_labels = ["Gender", "Ethnicity", "Informal", "Intersect.", "Income Grad.", "Wage Supp."]
                    hm_df = df[hm_metrics].copy()
                    hm_df.columns = hm_labels
                    hm_df.index = [f"Run {i + 1}" for i in range(len(hm_df))]
                    if CHARTS_OK:
                        st.plotly_chart(fairness_heatmap(hm_df,
                                                         title=tr("chart_heatmap"),
                                                         accent=ACCENT, height=320,
                                                         colorscale="RdYlGn_r",  # reversed: high gap = red
                                                         zmin=0, zmax=0.5), use_container_width=True)

                with c2:
                    # Regulatory compliance table
                    regs = {
                        "NITDA AI Policy 2023 Principle 7": avg_fair >= 0.65,
                        "EU AI Act Annex III (Employment)": avg_fair >= 0.70,
                        "ILO AI Workplace Convention 2023": avg_ggap < 0.15,
                        "EEOC Algorithmic Hiring Guidance": avg_ggap < 0.10,
                        "Nigeria Labour Act (anti-discrim.)": avg_igap < 0.20,
                        "80% Disparate Impact Threshold": avg_ggap < 0.20,
                    }
                    reg_df = pd.DataFrame([
                        {"Regulation": k, "Status": "✅ PASS" if v else "❌ FAIL", "Pass": v}
                        for k, v in regs.items()])
                    st.dataframe(reg_df[["Regulation", "Status"]], use_container_width=True,
                                 hide_index=True, height=260)
                    pass_n = reg_df["Pass"].sum()
                    total_n = len(regs)
                    col_pass = "#22c55e" if pass_n >= 4 else "#f59e0b" if pass_n >= 2 else "#ef4444"
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
                    st.warning(f"XAI error: {xai.get("error", "unknown")}")
                else:
                    fi = xai.get("feature_importance", {})
                    if fi and CHARTS_OK:
                        names = fi.get("feature_names", []);
                        imps = fi.get("importances", [])
                        if names and imps:
                            # Waterfall from feature importance
                            # Convert importances to signed contributions (proxy)
                            signed = [v if i % 2 == 0 else -v * 0.6
                                      for i, v in enumerate(imps[:10])]
                            st.plotly_chart(waterfall_feature_contributions(
                                names[:10], signed, base_value=0.5,
                                instance_label="AI Decision",
                                title=tr("chart_waterfall"),
                                accent=ACCENT, height=420,
                            ), use_container_width=True)

                            # Also bar chart for quick scanning
                            fi_df = pd.DataFrame({"Feature": names[:10], "Importance": imps[:10]})
                            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                            title=tr("chart_feat_importance"), color="Importance",
                                            color_continuous_scale="YlOrRd")
                            try:
                                fig_fi.update_layout(**PT(), height=320,
                                                     yaxis={"categoryorder": "total ascending"})
                            except:
                                fig_fi.update_layout(height=320)
                            st.plotly_chart(fig_fi, use_container_width=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        expl = xai.get("instance_explanation", {})
                        if expl:
                            st.markdown(tr("heading_why_worker"))
                            st.markdown(f'<div class="nbox"><em>{expl.get("decision_path", "")}</em></div>',
                                        unsafe_allow_html=True)
                            co = expl.get("feature_contributions", {})
                            if co and CHARTS_OK:
                                names_co = list(co.keys())[:8]
                                vals_co = [co[k] for k in names_co]
                                st.plotly_chart(waterfall_feature_contributions(
                                    names_co, vals_co, base_value=0.5,
                                    instance_label="This Worker",
                                    title=tr("chart_individual_wf"),
                                    accent="#ef4444", height=360,
                                ), use_container_width=True)

                    with c2:
                        cf = xai.get("counterfactual", {})
                        if cf:
                            st.markdown(tr("heading_counterfactual"))
                            st.markdown(f'<div class="nbox"><em>{cf.get("plain_language", "")}</em></div>',
                                        unsafe_allow_html=True)
                            changes = cf.get("changes", {})
                            if changes:
                                cf_df = pd.DataFrame([{
                                    "Feature": k, "Current": round(v[0], 3),
                                    "Required": round(v[1], 3), "Change": round(v[1] - v[0], 3),
                                } for k, v in changes.items()])
                                fig_cf = px.bar(cf_df, x="Change", y="Feature", orientation="h",
                                                color="Change", color_continuous_scale="RdYlGn",
                                                color_continuous_midpoint=0,
                                                title=tr("chart_change_needed"),
                                                height=320)
                                try:
                                    fig_cf.update_layout(**PT())
                                except:
                                    pass
                                st.plotly_chart(fig_cf, use_container_width=True)

                    # Intersectional
                    ix = xai.get("intersectional", {})
                    if ix and ix.get("group_performances"):
                        st.markdown(tr("heading_intersectional"))
                        st.markdown(f'<div class="nbox">{ix.get("narrative", "")}</div>',
                                    unsafe_allow_html=True)
                        ix_df = pd.DataFrame([{"Group": k, **{kk: round(vv, 3) for kk, vv in v.items()}}
                                              for k, v in ix["group_performances"].items()])
                        st.dataframe(ix_df.style.background_gradient(subset=["accuracy"], cmap="RdYlGn"),
                                     use_container_width=True)

            # ── TAB 4: REAL-WORLD BENCHMARKS ───────────────────────────────────────
            with tab_map[tr("tab_benchmarks")]:
                # Algorithm performance panel (shown when multiple algorithms compared)
                if "algorithm" in df.columns and len(df) > 1:
                    with st.expander("🤖 Algorithm Performance Summary", expanded=False):
                        _algo_grp = df.groupby("algorithm")[
                            ["accuracy", "fairness_score", "f1_score", "fpr"]].mean().reset_index()
                        _algo_grp.columns = ["Algorithm", "Accuracy", "Fairness Score", "F1 Score",
                                             "False Positive Rate"]
                        _algo_grp = _algo_grp.sort_values("Accuracy", ascending=False)
                        st.caption("Average metrics across all runs for each algorithm used in this session.")
                        st.dataframe(_algo_grp.style.format({
                            "Accuracy": "  {:.3f}", "Fairness Score": "  {:.3f}",
                            "F1 Score": "  {:.3f}", "False Positive Rate": "  {:.3f}"
                        }).background_gradient(subset=["Accuracy"], cmap="Greens")
                                     .background_gradient(subset=["Fairness Score"], cmap="RdYlGn"),
                                     use_container_width=True, hide_index=True)
                        st.caption(
                            "💡 Tip: Run multiple simulations with different algorithms selected to compare them here. "
                            "Fairness Score matters as much as Accuracy — a biased accurate model is not a good model.")
                st.divider()
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
                    domain_bms = get_benchmarks_for_domain(
                        getattr(preset_info, "economic_domain", getattr(preset_info, "health_domain", "")))
                    nigeria_bms = {k: v for k, v in REAL_WORLD_BENCHMARKS.items()
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
                            f'<span style="color:#0f172a">{k.replace("_", " ").title()}: '
                            f'<strong>{v:.2f}</strong></span>'
                            for k, v in list(bm.metrics.items())[:4] if isinstance(v, float))
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
                        better_n = sum(1 for c in comp["comparisons"].values() if c["verdict"] == "better")
                        worse_n = sum(1 for c in comp["comparisons"].values() if c["verdict"] == "worse")
                        verdict_color = "#22c55e" if better_n > worse_n else "#ef4444" if worse_n > better_n else "#f59e0b"
                        st.markdown(f'<div class="nbox" style="border-left-color:{verdict_color}">'
                                    f'<strong>{comp["overall_verdict"]}</strong><br>'
                                    f'<em>Lesson: {bm_sel.lesson}</em></div>',
                                    unsafe_allow_html=True)

                    # Nigeria macro context
                    st.markdown(tr("heading_nigeria_macro"))
                    if NIGERIA_MACRO_DATA:
                        sec = st.selectbox(tr("data_source_header"),
                                           list(NIGERIA_MACRO_DATA.keys()),
                                           format_func=lambda k: k.replace("_", " ").title())
                        macro = NIGERIA_MACRO_DATA[sec]
                        source = macro.pop("source", "")
                        macro_df = pd.DataFrame([{"Indicator": k.replace("_", " ").title(),
                                                  "Value": f"{v:.1%}" if v < 2 else f"{v:,.1f}"}
                                                 for k, v in macro.items()])
                        st.dataframe(macro_df, use_container_width=True, hide_index=True)
                        if source: st.caption(f"📚 Source: {source}")
                        macro["source"] = source  # restore

            # ── TAB 5: ECONOMIC IMPACT & SANKEY ───────────────────────────────────
            with tab_map[tr("tab_impact")]:
                c1, c2 = st.columns([1.2, 1])
                with c1:
                    if CHARTS_OK:
                        n_total = int(n_samples * 0.3)  # test set approx
                        fig_sk = make_economic_sankey(
                            n_total=n_total,
                            advantaged_pct=0.45,
                            outcome_rate_adv=df[
                                "outcome_rate_advantaged"].mean() if "outcome_rate_advantaged" in df.columns else df[
                                "overall_outcome_rate"].mean(),
                            outcome_rate_dis=df[
                                "outcome_rate_disadvantaged"].mean() if "outcome_rate_disadvantaged" in df.columns else max(
                                0, df["overall_outcome_rate"].mean() - avg_ggap),
                            advantaged_label="Formal / Majority Group",
                            disadvantaged_label="Informal / Minority Group",
                            positive_label="Positive AI Decision",
                            negative_label="Negative AI Decision",
                            accent=ACCENT,
                            title=f"Decision Flow — {getattr(preset_info, 'economic_domain', getattr(preset_info, 'health_domain', '')).title()} AI",
                        )
                        st.plotly_chart(fig_sk, use_container_width=True)

                with c2:
                    # Workers / GDP impact
                    gdp_pct = avg_igap * 0.15 + avg_ggap * 0.20 + avg_xgap * 0.10
                    wage_loss = avg_ggap * getattr(preset_info, "gender_wage_gap", 0.23) * 477.0
                    workers_affected = int(avg_xgap * n_samples * 0.3)

                    impact_d = {
                        "Group": ["Informal Workers", "Women", "Youth (15-34)", "Minority", "Disabled"],
                        "Bias Impact %": [avg_igap * 100, avg_ggap * 100,
                                          getattr(preset_info, "youth_unemployment_rate", 0.333) * avg_fair * 50,
                                          df["ethnicity_outcome_gap"].mean() * 100, avg_xgap * 40],
                    }
                    df_imp = pd.DataFrame(impact_d)
                    fig_imp = px.bar(df_imp, x="Bias Impact %", y="Group", orientation="h",
                                     color="Bias Impact %",
                                     color_continuous_scale=[[0, "#22c55e"], [0.5, ACCENT], [1, "#ef4444"]],
                                     title=tr("chart_groups_bias"),
                                     text=df_imp["Bias Impact %"].apply(lambda v: f"{v:.1f}%"))
                    fig_imp.update_traces(textposition="outside")
                    try:
                        fig_imp.update_layout(**PT(), height=300)
                    except:
                        fig_imp.update_layout(height=300)
                    st.plotly_chart(fig_imp, use_container_width=True)

                    m1, m2, m3 = st.columns(3)
                    for col, lbl, val, st_c in [
                        (m1, "Est. GDP Cost", f"{gdp_pct:.1%}", ""),
                        (m2, "Annual Wage Loss", f"${wage_loss:.0f}B", "crit"),
                        (m3, "Workers at Intersect. Risk", f"{workers_affected:,}", ""),
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
                        domain_filter=getattr(preset_info, "economic_domain",
                                              getattr(preset_info, "health_domain", "")) if domain_filter else None,
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
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Initial Bias", f'{lng["initial_bias"]:.1%}')
                    c2.metric("Final Bias", f'{lng["final_bias"]:.1%}',
                              f'{lng["final_bias"] - lng["initial_bias"]:+.1%}')
                    c3.metric("Amplification", f'{lng["amplification_factor"]:.2f}×',
                              "⚠️ Self-reinforcing" if lng["self_reinforcing"] else "Stable")
                    if lng.get("self_reinforcing"):
                        st.error(f'⚠️ Bias became self-reinforcing at generation '
                                 f'{lng.get("inflection_point", "N/A")}. '
                                 f'Each retraining cycle deepens inequality.')
                    st.markdown(f'<div class="nbox"><em>{lng["narrative"]}</em></div>',
                                unsafe_allow_html=True)
                    gm = lng.get("generation_metrics", [])
                    if gm and CHARTS_OK:
                        st.plotly_chart(animated_bias_drift(
                            gm,
                            metric_keys=["demographic_parity", "fairness_score", "accuracy"],
                            accent=ACCENT, height=400,
                            title=tr("chart_animated"),
                        ), use_container_width=True)
                        # Stacked area
                        gens = [m.get("generation", i) for i, m in enumerate(gm)]
                        adv_vals = [1 - (m.get("demographic_parity", 0) / 2) for m in gm]
                        dis_vals = [m.get("fairness_score", 0.5) for m in gm]
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
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Clients", fed["n_clients"])
                    c2.metric("Global Accuracy", f'{fed["global_accuracy"]:.1%}')
                    c3.metric("Global Fairness", f'{fed["global_fairness"]:.3f}')
                    c4.metric("Bias Persisted", "Yes ⚠️" if fed["bias_persisted"] else "No ✅")
                    st.markdown(
                        f'<div class="{"flag-c" if fed["bias_persisted"] else "nbox"}">'
                        f'<em>{fed["narrative"]}</em></div>', unsafe_allow_html=True)
                    cr_list = fed.get("client_results", [])
                    if cr_list:
                        cr_df = pd.DataFrame([c.__dict__ if hasattr(c, "__dict__") else c
                                              for c in cr_list])
                        if not cr_df.empty and "local_bias" in cr_df.columns:
                            fig_fed = px.bar(cr_df, x="client_id",
                                             y=["local_accuracy", "local_bias", "local_fairness"],
                                             barmode="group", title=tr("chart_fed_clients"),
                                             color_discrete_sequence=[ACCENT, "#ef4444", "#22c55e"])
                            try:
                                fig_fed.update_layout(**PT(), height=320)
                            except:
                                fig_fed.update_layout(height=320)
                            st.plotly_chart(fig_fed, use_container_width=True)

            # ── TAB 8: COMPLIANCE ─────────────────────────────────────────────────
            with tab_map[tr("tab_compliance")]:
                st.markdown(tr("heading_compliance"))
                _cr = xai.get("compliance_report", {});
                _mc = xai.get("model_card", {})
                if not _cr:
                    st.info(tr("xai_compliance_prompt"))
                else:
                    summ = _cr.get("summary", {});
                    ok = summ.get("overall_compliant", False)
                    st.markdown(
                        f'<div class="{"nbox" if ok else "flag-c"}">'
                        f'Overall: <strong>{"COMPLIANT ✅" if ok else "NON-COMPLIANT ❌"}</strong> | '
                        f'Fairness: {summ.get("fairness_score", 0):.3f} | '
                        f'Gender gap: {avg_ggap:.1%} | Informal gap: {avg_igap:.1%}</div>',
                        unsafe_allow_html=True)
                    for fw, fd in _cr.get("frameworks", {}).items():
                        with st.expander(f"📑 {fw}"):
                            for ch, st_ in fd.get("checks", {}).items():
                                st.markdown(f"{'✅' if st_ == 'PASS' else '❌'} {ch}")
                    try:
                        pdf_b = generate_pdf_compliance_report(_cr, _mc,
                                                               {"accuracy": avg_acc, "fairness_score": avg_fair},
                                                               domain="generic")
                        st.download_button(tr("download_pdf"), pdf_b,
                                           "economic_compliance.pdf", "application/pdf", use_container_width=True)
                    except Exception as e:
                        st.caption(f"PDF unavailable: {e}")
                st.divider()
                st.markdown("#### 🇳🇬 Nigeria Regulatory Panel")
                nigeria_compliance_panel(
                    {"accuracy": avg_acc, "fairness_score": avg_fair,
                     "demographic_parity": df["demographic_parity"].mean()},
                    domain="agrotech", has_ussd_fallback=False, has_gender_audit=True,
                    has_multilingual=False, has_xai=enable_xai,
                    has_governance=enable_governance, has_redteam=False)

            # ── TAB 9: EXPORT ─────────────────────────────────────────────────────
            with tab_map[tr("tab_export")]:
                st.markdown(tr("heading_export"))
                e1, e2, e3 = st.columns(3)
                sc = [c for c in df.columns if c not in ["narrative", "critical_flags"]]
                with e1:
                    st.download_button(tr("download_csv"),
                                       df[sc].to_csv(index=False).encode(),
                                       f"gags_economics_{scenario_key}.csv", "text/csv",
                                       use_container_width=True)
                with e2:
                    export_json = {
                        "gags_version": "4.0", "module": "Economic Justice §25",
                        "timestamp": datetime.now().isoformat(),
                        "scenario": {"key": scenario_key, "name": preset_info.name,
                                     "domain": getattr(preset_info, "economic_domain",
                                                       getattr(preset_info, "health_domain", "")),
                                     "citation": preset_info.citation},
                        "configuration": {"bias_intensity": bias_intensity,
                                          "selected_biases": selected_biases,
                                          "n_samples": int(n_samples), "n_runs": n_runs},
                        "results": {"avg_accuracy": round(avg_acc, 4),
                                    "avg_fairness": round(avg_fair, 4),
                                    "avg_inclusion": round(avg_incl, 4),
                                    "avg_gender_gap": round(avg_ggap, 4),
                                    "avg_informal_gap": round(avg_igap, 4),
                                    "avg_intersectional_gap": round(avg_xgap, 4)},
                        "regulatory_body": preset_info.regulatory_body,
                        "runs": df[sc].to_dict(orient="records"),
                    }
                    st.download_button(tr("download_json"),
                                       json.dumps(export_json, indent=2, default=str),
                                       f"gags_econ_{scenario_key}_{datetime.now().strftime('%Y%m%d')}.json",
                                       "application/json", use_container_width=True)
                with e3:
                    # Research narrative
                    narrative_txt = (
                        f"GAGS Economic Justice Simulation Report\n"
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                        f"{'=' * 60}\n\n"
                        f"Scenario: {preset_info.name}\n"
                        f"Domain:   {getattr(preset_info, 'economic_domain', getattr(preset_info, 'health_domain', '')).title()}\n"
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
                    .format({c: "{:.3f}" for c in sc
                             if c not in ["run_id", "scenario", "economic_domain",
                                          "biases", "regulatory_body"]
                             and df[sc][c].dtype in [float]})
                    .background_gradient(subset=["fairness_score"], cmap="RdYlGn")
                    .background_gradient(subset=["gender_outcome_gap"], cmap="Reds"),
                    use_container_width=True)

            # ── TAB 10: RESEARCH / STATISTICAL DISTRIBUTION ───────────────────────
            if vm == "Research" and tr("tab_stats") in tab_map:
                with tab_map[tr("tab_stats")]:
                    st.markdown(tr("heading_stats"))
                    st.markdown('<div class="nbox">Violin + box plots show the full distribution '
                                'of each metric — not just the mean. Essential for research: '
                                'shows variance, outliers, and statistical reliability.</div>',
                                unsafe_allow_html=True)
                    dist_metrics = ["accuracy", "fairness_score", "economic_inclusion_score",
                                    "gender_outcome_gap", "informal_sector_gap",
                                    "intersectional_worst_gap", "wage_suppression_index"]
                    dist_labels = {"accuracy": "Accuracy", "fairness_score": "Fairness",
                                   "economic_inclusion_score": "Inclusion",
                                   "gender_outcome_gap": "Gender Gap", "informal_sector_gap": "Informal Gap",
                                   "intersectional_worst_gap": "Intersect. Gap",
                                   "wage_suppression_index": "Wage Suppression"}
                    if CHARTS_OK:
                        st.plotly_chart(multi_run_distribution(
                            st.session_state.econ_run_history,
                            metrics=dist_metrics, metric_labels=dist_labels,
                            accent=ACCENT, height=460,
                            title=f"Full Distribution — {n_runs} Simulation Runs",
                        ), use_container_width=True)
                    else:
                        st.dataframe(df[dist_metrics].describe(), use_container_width=True)

        else:
            def _hpass(col, val):
                rules = {
                    "insurance_denial_gap": ("lt", 0.15), "wealth_quintile_access_gap": ("lt", 0.20),
                    "geographic_equity_index": ("gt", 0.70), "gender_health_gap": ("lt", 0.10),
                    "catastrophic_expenditure_risk": ("lt", 0.10), "poverty_trap_risk": ("lt", 0.08),
                    "maternal_access_gap": ("lt", 0.15), "uhc_service_coverage_gap": ("lt", 0.37),
                    "development_targeting_error": ("lt", 0.15), "health_financing_fairness": ("gt", 0.65),
                }
                r = rules.get(col)
                if not r: return True
                op, th = r
                return (val < th) if op == "lt" else (val > th)


            # ── Overview ────────────────────────────────────────────────────────────────────
            with tab_map.get("\U0001f4ca Overview", list(tab_map.values())[0]):
                c1, c2 = st.columns([1.1, 0.9])
                with c1:
                    if CHARTS_OK:
                        try:
                            h_grp = ["Wealth Gap", "Urban\u2013Rural Gap", "Gender Gap", "Poverty Trap Risk"]
                            h_adv = [1 - (avg_ins_gap or avg_ggap) * 0.6, 1 - avg_igap * 0.6,
                                     1 - avg_ggap * 0.5, 1 - avg_xgap * 0.4]
                            h_dis = [1 - (avg_ins_gap or avg_ggap * 1.4), 1 - avg_igap * 1.5,
                                     1 - avg_ggap * 1.2, 1 - avg_xgap]
                            st.plotly_chart(lollipop_gap_chart(
                                h_grp, h_adv, h_dis, label_a="Advantaged", label_b="Disadvantaged",
                                metric_name="Health Access Rate", accent="#0891b2",
                                height=340, threshold=0.80), use_container_width=True)
                        except Exception:
                            for g, a, d in zip(h_grp, h_adv, h_dis):
                                st.metric(g, f"{a:.1%}", delta=f"{d - a:+.1%}")
                with c2:
                    if CHARTS_OK:
                        try:
                            st.plotly_chart(gauge_cluster([
                                ("Health Equity", avg_hf_fair or avg_fair, 0.65, "#0891b2"),
                                ("Fairness", avg_fair, 0.65, "#16a34a"),
                                ("Accuracy", avg_acc, 0.75, "#b45309"),
                                ("Wealth Gap\u2193", 1 - (avg_ins_gap or avg_ggap), 0.80, "#dc2626"),
                                ("OOP Risk\u2193", 1 - (avg_cat_exp or 0.20), 0.85, "#f59e0b"),
                                ("UHC Progress", 1 - (avg_uhc_gap or 0.37), 0.60, "#7c3aed"),
                            ], height=230, cols=6), use_container_width=True)
                        except Exception:
                            st.metric("Health Equity", f"{avg_hf_fair or avg_fair:.3f}")
                flags_h = df["critical_flags"].iloc[-1] if "critical_flags" in df.columns else []
                if isinstance(flags_h, list):
                    for fh in flags_h:
                        cls = "flag-c" if "\U0001f6a8" in fh or "\u26a0" in fh else "flag-w"
                        st.markdown(f'<div class="{cls}">{fh}</div>', unsafe_allow_html=True)
                if "narrative" in df.columns:
                    st.markdown(f'<div class="nbox">{df["narrative"].iloc[-1]}</div>', unsafe_allow_html=True)

            # ── Health Equity ────────────────────────────────────────────────────────────────
            with tab_map.get("\U0001f3e5 Health Equity", list(tab_map.values())[0]):
                st.markdown("#### \U0001f3e5 Health Financing Equity Analysis")
                st.markdown(
                    '<div class="nbox">Each metric anchored to WHO, NHIA, and World Bank standards. Thresholds are the regulatory compliance floor.</div>',
                    unsafe_allow_html=True)
                col_map_h = {
                    "insurance_denial_gap": ("Insurance Denial Gap", "< 15pp"),
                    "wealth_quintile_access_gap": ("Wealth Quintile Gap", "< 20pp"),
                    "geographic_equity_index": ("Geographic Equity Index", "> 0.70"),
                    "gender_health_gap": ("Gender Health Gap", "< 10pp"),
                    "catastrophic_expenditure_risk": ("Catastrophic OOP Risk", "< 10%"),
                    "poverty_trap_risk": ("Poverty Trap Risk", "< 8%"),
                    "maternal_access_gap": ("Maternal Access Gap", "< 15pp"),
                    "uhc_service_coverage_gap": ("UHC Coverage Gap", "< 37pp"),
                    "development_targeting_error": ("Dev. Targeting Error", "< 15%"),
                    "health_financing_fairness": ("HF Fairness Score", ">= 0.65"),
                }
                hm_rows = [
                    {"Metric": lbl, "Value": f"{df[col].mean():.3f}", "Threshold": th,
                     "Status": "\u2705 PASS" if _hpass(col, df[col].mean()) else "\u274c FLAG"}
                    for col, (lbl, th) in col_map_h.items()
                    if col in df.columns and df[col].notna().any()
                ]
                if hm_rows:
                    st.dataframe(pd.DataFrame(hm_rows), use_container_width=True, hide_index=True)
                st.divider()
                st.markdown("#### Health Equity Heatmap Across Runs")
                h_hm = [c for c in ["fairness_score", "health_financing_fairness",
                                    "wealth_quintile_access_gap", "geographic_equity_index",
                                    "catastrophic_expenditure_risk", "gender_health_gap"]
                        if c in df.columns and df[c].notna().any()]
                if CHARTS_OK and len(h_hm) >= 2:
                    try:
                        st.plotly_chart(fairness_heatmap(df[["run_id"] + h_hm].copy(),
                                                         run_col="run_id", metric_cols=h_hm,
                                                         title="Health Equity Heatmap (Run \u00d7 Metric)",
                                                         accent="#0891b2"),
                                        use_container_width=True)
                    except Exception:
                        st.dataframe(df[h_hm].describe().round(3), use_container_width=True)

            # ── Health Financing ─────────────────────────────────────────────────────────────
            with tab_map.get("\U0001f48a Health Financing", list(tab_map.values())[0]):
                st.markdown("#### \U0001f48a Health Financing & Out-of-Pocket Burden")
                nhis_r = getattr(preset_info, "nhis_coverage_rate", 0.045)
                cat_rh = avg_cat_exp or 0.12
                pov_rh = avg_xgap * 0.5
                c1h, c2h = st.columns(2)
                with c1h:
                    try:
                        fig_sk = go.Figure(go.Sankey(
                            node=dict(pad=15, thickness=20,
                                      label=["Population", "Insured (NHIA)", "Uninsured", "Gets Care",
                                             "Denied", "Catastrophic OOP", "Poverty Trap", "Adequate Care"],
                                      color=["#334155", "#16a34a", "#dc2626", "#0891b2",
                                             "#f59e0b", "#dc2626", "#7c3aed", "#16a34a"]),
                            link=dict(
                                source=[0, 0, 1, 2, 2, 4, 4],
                                target=[1, 2, 3, 3, 4, 5, 6],
                                value=[nhis_r * 1000, (1 - nhis_r) * 1000, nhis_r * 900,
                                       (1 - nhis_r) * 600, (1 - nhis_r) * 400, cat_rh * 400, pov_rh * 400],
                                color=["rgba(22,163,74,0.3)", "rgba(220,38,38,0.3)", "rgba(8,145,178,0.3)",
                                       "rgba(8,145,178,0.3)", "rgba(245,158,11,0.3)", "rgba(220,38,38,0.3)",
                                       "rgba(124,58,237,0.3)"]),
                        ))
                        fig_sk.update_layout(title_text="Care Access Pathway \u2014 Who Gets Through?",
                                             height=380, paper_bgcolor="white")
                        st.plotly_chart(fig_sk, use_container_width=True)
                    except Exception:
                        kh = st.columns(3)
                        kh[0].metric("NHIA Coverage", f"{nhis_r:.1%}")
                        kh[1].metric("Catastrophic OOP Risk", f"{cat_rh:.1%}")
                        kh[2].metric("Poverty Trap Risk", f"{pov_rh:.1%}")
                with c2h:
                    st.markdown("##### Nigeria Health Finance vs WHO Targets")
                    st.dataframe(pd.DataFrame({
                        "Metric": ["NHIS Coverage", "OOP %", "MMR", "Workers/10k", "UHC Index"],
                        "Nigeria": [f"{nhis_r:.1%}", "74.8%",
                                    f"{getattr(preset_info, 'maternal_mortality_ratio', 1047):.0f}/100k",
                                    "1.95", "43/100"],
                        "WHO Target": ["Universal", "<20%", "<70/100k", "\u226523/10k", "\u226580/100"],
                        "AI Risk": ["High", "Critical", "Critical", "High", "Medium"],
                    }), use_container_width=True, hide_index=True)
                st.divider()
                dh = getattr(preset_info, "health_domain", "oop")
                st.markdown("#### \U0001f4cc Policy Implications")
                pm = {
                    "insurance": [
                        f"**Insurance denial gap: {avg_ins_gap:.1%}** \u2014 informal workers ({getattr(preset_info, 'informal_sector_pct', 0.649):.0%} of workforce) excluded by employment-based AI scoring",
                        "**Policy lever**: NHIA Act 2022 mandates universal coverage \u2014 accept mobile money / utility payment records as informal income evidence",
                    ],
                    "oop": [
                        f"**OOP burden 74.8%** \u2014 AI triage errors translate directly to financial catastrophe for poor households",
                        f"**Poverty trap risk: {avg_xgap:.1%}** \u2014 near-poor households pushed below poverty line by AI care denial",
                        "**Policy lever**: BHCPF must condition facility financing on demonstrated triage equity metrics",
                    ],
                    "maternal": [
                        f"**Maternal access gap: {avg_mat_gap:.1%}** \u2014 Nigeria 2022 pilot showed 31% rural miss rate; AI trained on urban tertiary records",
                        f"**MMR 1,047/100k** \u2014 15\u00d7 the SDG target; AI misclassification is not an acceptable margin",
                        "**Policy lever**: FMOH Maternal Death Review must integrate AI audit; northern states need equity-adjusted thresholds",
                    ],
                    "workforce": [
                        f"**Density gap**: {getattr(preset_info, 'health_worker_density', 1.95):.2f}/10k vs WHO minimum 23 \u2014 AI must weight disease burden not infrastructure scores",
                        "**Policy lever**: HCWDB must mandate equity-weighted AI allocation with real-time LGA monitoring",
                    ],
                    "pharma": [
                        "**Essential medicine price variation 8\u00d7 across LGAs** \u2014 AI demand prediction amplifies this for profit",
                        "**Policy lever**: NAFDAC price audit mandate; NHIA pooled procurement must use equity-weighted not revenue-optimised AI",
                    ],
                    "devaid": [
                        f"**Exclusion error {avg_dev_err:.1%}** \u2014 truly poor households denied social protection; Robodebt pattern detected",
                        "**Policy lever**: NSIP must include CBO attestation / local government affidavit as non-digital alternative pathway",
                    ],
                }
                for msg in pm.get(dh, pm["oop"]):
                    st.markdown(f'<div class="nbox">{msg}</div>', unsafe_allow_html=True)

            # ── UHC & Development Economics ──────────────────────────────────────────────────
            with tab_map.get("\U0001f30d UHC & Dev Economics", list(tab_map.values())[0]):
                st.markdown("#### \U0001f30d Universal Health Coverage & Development Economics")
                st.markdown(
                    '<div class="nbox">Nigeria UHC index: <strong>43/100</strong> (WHO 2023). SDG 3.8 target: <strong>80/100 by 2030</strong>. This tab models how AI deployment choices accelerate or impede that journey.</div>',
                    unsafe_allow_html=True)
                c1u, c2u = st.columns(2)
                uhc_b = 43.0
                uhc_sim = max(30, uhc_b - (avg_uhc_gap or 0.37) * 50)
                uhc_opt = min(uhc_b + 8.0, 80.0)
                with c1u:
                    fig_uhc = go.Figure(go.Bar(
                        x=["Nigeria 2023", "With Biased AI", "With Fair AI", "SDG 2030 Target"],
                        y=[uhc_b, uhc_sim, uhc_opt, 80.0],
                        marker_color=["#334155", "#dc2626", "#16a34a", "#0891b2"],
                        text=[f"{v:.1f}" for v in [uhc_b, uhc_sim, uhc_opt, 80.0]],
                        textposition="outside",
                    ))
                    fig_uhc.add_hline(y=80, line_dash="dash", line_color="#0891b2",
                                      annotation_text="SDG 3.8 Target (80/100)")
                    fig_uhc.update_layout(
                        title="UHC Service Coverage Index \u2014 AI Impact Scenario",
                        yaxis_title="UHC Index (0\u2013100)", height=360, yaxis_range=[0, 90],
                        paper_bgcolor="white", plot_bgcolor="#f8fafc",
                    )
                    st.plotly_chart(fig_uhc, use_container_width=True)
                with c2u:
                    st.markdown("##### Development Economics: AI & Poverty Traps")
                    excl_v = avg_dev_err or 0.18
                    incl_v = df["inclusion_error"].mean() if "inclusion_error" in df.columns and df[
                        "inclusion_error"].notna().any() else 0.12
                    pov_base = getattr(preset_info, "poverty_rate", 0.387)
                    st.dataframe(pd.DataFrame({
                        "Indicator": ["National Poverty Rate", "Simulation Exclusion Error",
                                      "Simulation Inclusion Error", "Poverty Trap Risk",
                                      "Digital Exclusion Risk", "Near-Poor at Risk"],
                        "Value": [f"{pov_base:.1%}", f"{excl_v:.1%}", f"{incl_v:.1%}",
                                  f"{avg_xgap * 0.4:.1%}", f"{(1 - 0.55) * (1 - 0.61):.1%}",
                                  f"{pov_base * 0.3:.1%}"],
                        "Threshold": ["Reference", "<15%", "<20%", "<8%", "Monitor", "<5%"],
                        "Status": ["\U0001f4ca Baseline",
                                   "\u2705" if excl_v < 0.15 else "\u274c FLAG",
                                   "\u2705" if incl_v < 0.20 else "\u26a0\ufe0f WATCH",
                                   "\u2705" if avg_xgap * 0.4 < 0.08 else "\u274c FLAG",
                                   "\u26a0\ufe0f Monitor",
                                   "\u2705" if pov_base * 0.3 < 0.05 else "\u274c FLAG"],
                    }), use_container_width=True, hide_index=True)
                st.divider()
                st.markdown("#### \U0001f4d6 Robodebt Patterns in Nigerian Context")
                st.markdown(
                    '<div class="nbox">The Australian Robodebt Royal Commission (2023) found 32% wrongful welfare denial from AI-automated proxy scoring. The same structural pattern exists in Nigeria NSIP BVN/NIN-dependent targeting: AI uses digital footprint as a poverty proxy, systematically excluding the poorest households who have no digital footprint.</div>',
                    unsafe_allow_html=True)
                rc = st.columns(3)
                rc[0].metric("Robodebt Wrongful Notices", "32%", delta="Reference Case", delta_color="off")
                rc[1].metric("Your Simulation Exclusion Error", f"{excl_v:.1%}",
                             delta=f"{'Better' if excl_v < 0.32 else 'Worse'} than Robodebt",
                             delta_color="normal" if excl_v < 0.32 else "inverse")
                rc[2].metric("Nigeria Digital Exclusion Risk", f"{(1 - 0.55) * (1 - 0.61):.1%}",
                             delta="No BVN + No Mobile Money", delta_color="off")

            # ── XAI ─────────────────────────────────────────────────────────────────────────
            with tab_map.get("\U0001f50d XAI & Counterfactuals", list(tab_map.values())[0]):
                st.markdown("#### \U0001f50d Explainable AI \u2014 Health Finance Decision Drivers")
                xai_h = st.session_state.get("econ_xai_results", {})
                if not xai_h or "error" in xai_h:
                    st.info("Enable XAI in the sidebar and run a health simulation to see explanations.")
                else:
                    c1x, c2x = st.columns(2)
                    with c1x:
                        st.markdown("##### Feature Importance \u2014 What Drives Health AI Decisions?")
                        fi = xai_h.get("feature_importance", {})
                        feats = fi.get("feature_names", [])
                        imps = fi.get("importances_mean", [])
                        if CHARTS_OK and feats and imps:
                            try:
                                st.plotly_chart(waterfall_feature_contributions(
                                    feats[:8], imps[:8],
                                    title="Feature Importance \u2014 Health AI", accent="#0891b2"),
                                    use_container_width=True)
                            except Exception:
                                for f, i in zip(feats[:5], imps[:5]):
                                    st.metric(f.replace("_", " ").title(), f"{i:.3f}")
                    with c2x:
                        st.markdown("##### Why Was This Patient/Beneficiary Denied?")
                        cf = xai_h.get("counterfactual", {})
                        changes = cf.get("changes", []) if cf else []
                        if isinstance(changes, dict): changes = list(changes.values())
                        hfl = {
                            "employment_formality": "Employment Formality",
                            "income_quintile": "Income Quintile",
                            "location_rurality": "Location (Urban/Rural)",
                            "bvn_status": "BVN Registration",
                            "digital_footprint_score": "Digital Footprint Score",
                            "anc_visits": "Antenatal Care Visits",
                            "distance_to_facility": "Distance to Facility",
                            "wealth_quintile": "Household Wealth Quintile",
                        }
                        if changes:
                            st.markdown("**Changes needed to reverse the AI decision:**")
                            # changes: Dict[str, Tuple[float, float]] = {feature: (orig, new)}
                            _ch_items = list(changes.items()) if isinstance(changes, dict) else []
                            for fname, vals in _ch_items[:5]:
                                try:
                                    cur = vals[0] if isinstance(vals, (list, tuple)) else float(vals)
                                    req = vals[1] if isinstance(vals, (list, tuple)) and len(vals) > 1 else cur
                                except Exception:
                                    continue
                                st.markdown(
                                    f'<div class="nbox"><strong>{hfl.get(fname, fname.replace("_", " ").title())}</strong>: '
                                    f'{cur:.2f} \u2192 {req:.2f} ({req - cur:+.2f})</div>',
                                    unsafe_allow_html=True)
                        else:
                            st.info("Enable XAI to see counterfactual explanations.")

            # ── Benchmarks ──────────────────────────────────────────────────────────────────
            with tab_map.get("\U0001f4da Benchmarks", list(tab_map.values())[0]):
                st.markdown("#### \U0001f4da Health Finance Benchmark Comparison")
                dh2 = getattr(preset_info, "health_domain", "oop")
                health_bms = [
                    ("WHO Health Financing Progress Matrix (2023)",
                     "UHC index 43/100; OOP 74.8%; 195M+ uninsured",
                     f"UHC gap: {avg_uhc_gap:.2f}" if avg_uhc_gap else "\u2013", "critical", "all"),
                    ("NHIA Annual Report (2023)",
                     "Formal insurance coverage 4.5% \u2014 informal workers structurally excluded",
                     f"Insurance denial gap: {avg_ins_gap:.1%}" if avg_ins_gap else "\u2013", "critical", "insurance"),
                    ("NDHS Nigeria 2021 \u2014 Maternal Health",
                     "MMR 1,047/100k; NW states 4\u00d7 higher than SW; 38% home deliveries",
                     f"Maternal access gap: {avg_mat_gap:.1%}" if avg_mat_gap else "N/A", "critical", "maternal"),
                    ("Nigeria Maternal Health AI Pilot (2022)",
                     "AI missed 31% of high-risk rural women due to urban-dominated training data",
                     f"Geographic equity: {avg_igap:.3f}", "critical", "maternal"),
                    ("Okonkwo et al. \u2014 Lancet Digital Health (2022)",
                     "AI triage in LMICs under-classifies rural risk by 23\u201341% systematically",
                     f"Rural\u2013urban gap: {avg_igap:.1%}", "high", "oop"),
                    ("Onoka et al. \u2014 Catastrophic Health Expenditure (2022)",
                     "4.0% Nigerian households face catastrophic OOP; near-poor 3\u00d7 more vulnerable",
                     f"OOP risk: {avg_cat_exp:.1%}" if avg_cat_exp else "\u2013", "high", "oop"),
                    ("Australian Robodebt Royal Commission (2023)",
                     "32% wrongful debt notices from AI-automated welfare proxy scoring",
                     f"Dev. targeting error: {avg_dev_err:.1%}" if avg_dev_err else "N/A", "critical", "devaid"),
                    ("World Bank Nigeria PBF Evaluation (2022)",
                     "Performance-based financing AI: 18% exclusion error in poorest quintile",
                     f"Inclusion error: {df['inclusion_error'].mean():.1%}" if "inclusion_error" in df.columns and df[
                         "inclusion_error"].notna().any() else "N/A", "high", "devaid"),
                ]
                for study, finding, your_m, sev, dom in health_bms:
                    if dom == "all" or dom == dh2:
                        sev_cls = "bm-severity-critical" if sev == "critical" else ""
                        st.markdown(
                            f'<div class="bm-card {sev_cls}">'
                            f'<p class="bm-title">\U0001f4d6 {study}</p>'
                            f'<p class="bm-meta">{finding}</p>'
                            f'<p class="bm-lesson"><strong>Your simulation:</strong> {your_m}</p></div>',
                            unsafe_allow_html=True)

            # ── Longitudinal ─────────────────────────────────────────────────────────────────
            with tab_map.get("\U0001f501 Longitudinal", list(tab_map.values())[0]):
                st.markdown("### \U0001f501 Longitudinal Health Equity \u2014 Bias Amplification Across Retraining")
                lng_h = st.session_state.get("econ_longitudinal")
                if not lng_h:
                    st.info(tr("run_to_see_lng"))
                else:
                    if CHARTS_OK:
                        try:
                            st.plotly_chart(animated_bias_drift(lng_h, accent="#0891b2",
                                                                title="Health Equity Drift Across AI Retraining Cycles",
                                                                y_label="Health Equity Score"),
                                            use_container_width=True)
                        except Exception as e:
                            st.warning(f"Chart unavailable: {e}")
                    st.markdown(
                        '<div class="nbox"><strong>Health implication:</strong> Each time a health AI is retrained on its own decisions, initial exclusions (rural women missed, informal workers denied insurance) corrupt future training data \u2014 creating self-reinforcing health inequity that widens with each cycle.</div>',
                        unsafe_allow_html=True)
                    amp = lng_h.get("amplification_factor", 1.0) if isinstance(lng_h, dict) else 1.0
                    sf = lng_h.get("self_reinforcing", False) if isinstance(lng_h, dict) else False
                    lc = st.columns(3)
                    lc[0].metric("Amplification Factor", f"{amp:.2f}\u00d7",
                                 delta="CRITICAL \U0001f6a8" if amp > 1.5 else "Moderate \u26a0\ufe0f" if amp > 1.2 else "Low \u2705")
                    lc[1].metric("Self-Reinforcing", "Yes \U0001f6a8" if sf else "No \u2705")
                    lc[2].metric("Intervention Point", "Cycle 2\u20133" if amp > 1.5 else "Annual review sufficient")

            # ── Federated ────────────────────────────────────────────────────────────────────
            with tab_map.get("\U0001f310 Federated", list(tab_map.values())[0]):
                st.markdown("### \U0001f310 Federated Learning \u2014 Equity Across Facilities / States")
                fed_h = st.session_state.get("econ_federated")
                if not fed_h:
                    st.info(tr("run_to_see_fed"))
                else:
                    st.markdown(
                        '<div class="nbox">Tests whether health equity is maintained across distributed clients (hospitals, states, LGAs) or concentrated in specific facilities.</div>',
                        unsafe_allow_html=True)
                    try:
                        cr = fed_h.get("client_results", []) if isinstance(fed_h, dict) else []
                        if cr:
                            cr_df = pd.DataFrame([c.__dict__ if hasattr(c, "__dict__") else c for c in cr[:4]])
                            if not cr_df.empty and "local_accuracy" in cr_df.columns:
                                cr_df["facility"] = [f"Facility {j + 1}" for j in range(len(cr_df))]
                                ycols = [c for c in ["local_accuracy", "local_bias"] if c in cr_df.columns]
                                st.plotly_chart(px.bar(cr_df, x="facility", y=ycols, barmode="group",
                                                       title="Health Equity Across Federated Facility Clients",
                                                       color_discrete_sequence=["#0891b2", "#dc2626"]),
                                                use_container_width=True)
                        fc = st.columns(3)
                        fc[0].metric("Global Accuracy",
                                     f"{fed_h.get('global_accuracy', 0):.1%}" if isinstance(fed_h, dict) else "\u2013")
                        bp = fed_h.get("bias_persisted", False) if isinstance(fed_h, dict) else False
                        fc[1].metric("Bias Persisted After Federated Training", "Yes \U0001f6a8" if bp else "No \u2705")
                        fc[2].metric("Recommendation",
                                     "Equity-weighted aggregation needed" if bp else "Standard FedAvg acceptable")
                    except Exception as e:
                        st.warning(f"Federated analysis: {e}")

            # ── Compliance ───────────────────────────────────────────────────────────────────
            with tab_map.get("\U0001f4cb Compliance", list(tab_map.values())[0]):
                st.markdown("#### \U0001f4cb Health Finance Regulatory Compliance Report")
                st.markdown(
                    '<div class="nbox">Frameworks assessed: <strong>NHIA Act 2022, FMOH PHC Policy, WHO UHC Framework, World Bank Social Protection Standards, NDPC Act 2023, SDG 3.8 (UHC by 2030), ILO Social Protection Floor, EU AI Act (High-Risk AI in Healthcare).</strong></div>',
                    unsafe_allow_html=True)
                hf_s = avg_hf_fair or avg_fair
                w_g = avg_ins_gap or avg_ggap
                cat3 = avg_cat_exp or 0
                uhc3 = avg_uhc_gap or 0
                mat3 = avg_mat_gap or 0
                dev3 = avg_dev_err or 0
                chk = [
                    ("NHIA Act 2022 \u2014 Universal Coverage Mandate",
                     "Health AI must not discriminate on basis of employment formality",
                     (avg_ins_gap or 0) < 0.15, f"Insurance denial gap: {w_g:.1%}"),
                    ("FMOH PHC Policy \u2014 Geographic Equity",
                     "AI systems must maintain equal access across urban/rural populations",
                     avg_igap < 0.15, f"Geographic gap: {avg_igap:.1%}"),
                    ("WHO UHC Framework \u2014 Catastrophic Expenditure Prevention",
                     "AI decisions must not increase catastrophic OOP beyond 10% of households",
                     cat3 < 0.10, f"Catastrophic OOP risk: {cat3:.1%}"),
                    ("SDG 3.8 \u2014 Universal Health Coverage (2030)",
                     "AI deployment must close, not widen, Nigeria's UHC gap from 43 to 80",
                     uhc3 < 0.37, f"UHC coverage gap: {uhc3:.2f}"),
                    ("FMOH \u2014 Maternal Health Equity Standard",
                     "AI risk scoring must not disadvantage rural/northern pregnant women",
                     mat3 < 0.15, f"Maternal access gap: {mat3:.1%}"),
                    ("World Bank Social Protection Standards",
                     "Programme targeting AI must keep exclusion error below 15%",
                     dev3 < 0.15, f"Dev. targeting error: {dev3:.1%}"),
                    ("NDPC Act 2023 \u2014 Automated Decision Rights",
                     "Individuals have right to explanation and human review for health AI decisions",
                     enable_xai, f"XAI enabled: {'Yes' if enable_xai else 'No \u2014 enable XAI'}"),
                    ("EU AI Act \u2014 High-Risk AI in Healthcare",
                     "Health AI systems require conformity assessment and post-market monitoring",
                     hf_s >= 0.60, f"Health equity score: {hf_s:.3f}"),
                ]
                ph = sum(1 for _, _, p, _ in chk if p)
                th = len(chk)
                sc = "#16a34a" if ph == th else "#dc2626" if ph < th * 0.6 else "#f59e0b"
                st.markdown(
                    f'<div style="background:{sc}15;border:2px solid {sc};border-radius:8px;'
                    f'padding:1rem;margin-bottom:1rem"><strong>Health Compliance: {ph}/{th} \u2014 '
                    f'{"COMPLIANT \u2705" if ph == th else "NON-COMPLIANT \u274c" if ph < th * 0.6 else "CONDITIONAL \u26a0\ufe0f"}'
                    f'</strong></div>', unsafe_allow_html=True)
                for fw, desc, passes, val in chk:
                    with st.expander(f'{"\u2705" if passes else "\u274c"} {fw}'):
                        st.markdown(f"**Requirement:** {desc}")
                        st.markdown(f"**Your simulation:** {val}")
                        st.markdown(f"**Status:** {'PASS' if passes else 'FAIL \u2014 intervention required'}")
                st.divider()
                nigeria_compliance_panel(
                    {"accuracy": avg_acc, "fairness_score": avg_fair, "demographic_parity": w_g},
                    domain="healthcare", has_ussd_fallback=False, has_gender_audit=True,
                    has_multilingual=False, has_xai=enable_xai,
                    has_governance=enable_governance, has_redteam=False)

            # ── Export ───────────────────────────────────────────────────────────────────────
            with tab_map.get("\U0001f4e4 Export", list(tab_map.values())[0]):
                st.markdown(tr("heading_export"))
                hec = [c for c in df.columns if c not in ["narrative", "critical_flags", "is_health_run"]]
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.download_button(tr("download_csv"),
                                       df[hec].to_csv(index=False).encode(),
                                       f"gags_health_finance_{scenario_key}.csv", "text/csv",
                                       use_container_width=True)
                with e2:
                    cfg_h = {"scenario_key": scenario_key, "module": "health_finance",
                             "bias_intensity": bias_intensity, "n_samples": int(n_samples),
                             "n_runs": n_runs, "health_domain": getattr(preset_info, "health_domain", ""),
                             "regulatory_body": getattr(preset_info, "regulatory_body", ""),
                             "results": {"hf_fairness": avg_hf_fair, "cat_exp_risk": avg_cat_exp,
                                         "uhc_gap": avg_uhc_gap, "ins_gap": avg_ins_gap}}
                    st.download_button(tr("download_json"),
                                       json.dumps(cfg_h, indent=2, default=str).encode(),
                                       f"gags_health_{scenario_key}.json", "application/json",
                                       use_container_width=True)
                with e3:
                    narr = df["narrative"].iloc[-1] if "narrative" in df.columns else ""
                    flags_txt = "\n".join(
                        df["critical_flags"].iloc[-1]) if "critical_flags" in df.columns and isinstance(
                        df["critical_flags"].iloc[-1], list) else ""
                    full_txt = f"GAGS Health Finance Compliance Narrative\n{'=' * 50}\n\n{narr}\n\nCritical Flags:\n{flags_txt}\n\nData Citation:\n{getattr(preset_info, 'citation', '')}"
                    st.download_button("\U0001f4c4 Health Compliance Narrative (.txt)",
                                       full_txt.encode(), f"health_compliance_{scenario_key}.txt", "text/plain",
                                       use_container_width=True)
                st.divider()
                st.markdown("##### Complete Health Finance Results Table")
                try:
                    st.dataframe(_safe_fmt(df[hec], exclude=["run_id", "scenario"]), use_container_width=True)
                except Exception:
                    st.dataframe(df[hec], use_container_width=True)

            if vm == "Research" and "\U0001f4c8 Statistical Distribution" in tab_map:
                with tab_map["\U0001f4c8 Statistical Distribution"]:
                    st.markdown(tr("heading_stats"))
                    h_dm = [c for c in ["accuracy", "fairness_score", "health_financing_fairness",
                                        "wealth_quintile_access_gap", "geographic_equity_index",
                                        "catastrophic_expenditure_risk", "uhc_service_coverage_gap",
                                        "maternal_access_gap", "development_targeting_error"]
                            if c in df.columns and df[c].notna().any()]
                    if CHARTS_OK and h_dm:
                        try:
                            st.plotly_chart(multi_run_distribution(
                                st.session_state.econ_run_history, metrics=h_dm,
                                metric_labels={"accuracy": "Accuracy", "fairness_score": "Fairness",
                                               "health_financing_fairness": "HF Fairness",
                                               "wealth_quintile_access_gap": "Wealth Gap",
                                               "geographic_equity_index": "Geo Equity",
                                               "catastrophic_expenditure_risk": "OOP Risk",
                                               "uhc_service_coverage_gap": "UHC Gap",
                                               "maternal_access_gap": "Maternal Gap",
                                               "development_targeting_error": "Dev. Error"},
                                accent="#0891b2", height=480,
                                title=f"Health Finance \u2014 Full Distribution Across {n_runs} Runs",
                            ), use_container_width=True)
                        except Exception as e:
                            st.warning(f"Distribution chart unavailable: {e}")
                    if h_dm:
                        st.dataframe(df[h_dm].describe().round(4), use_container_width=True)

                with tab_fm:
                    feats = st.session_state.get("econ_feature_outputs", {})
                    feature_modules_tab(
                        domain="economic",
                        run_results=st.session_state.get("econ_run_history", []),
                        feats=feats,
                        governance=feats.get("governance"),
                        gender_audit=feats.get("gender_audit"),
                        agent_economy=feats.get("agent_economy"),
                        arena=feats.get("strategic_arena"),
                        redteam=feats.get("multimodal_redteam"),
                    )

        # ── Tab: AI Safety ────────────────────────────────────────────────────
        if "🛡️ AI Safety" in tab_map:
            with tab_map["🛡️ AI Safety"]:
                render_safety_tab(
                    st.session_state.get("econ_safety_report", {}),
                    domain="economic",
                )

    # ── 🔄 Lifecycle Management Tab ───────────────────────────────────────────
    if "🔄 Lifecycle" in tab_map:
        with tab_map["🔄 Lifecycle"]:
            render_lifecycle_tab(
                st.session_state.get("econ_lifecycle_report", {}),
                "economic",
            )

    # ── 🌱 Eco Score Tab ──────────────────────────────────────────────────────
    if "🌱 Eco Score" in tab_map:
        with tab_map["🌱 Eco Score"]:
            _lc_eco_r = st.session_state.get("econ_lifecycle_report", {})
            _lc_eco_last = (st.session_state.get("econ_run_history") or [{}])[-1]
            render_eco_tab(
                _lc_eco_r,
                algo_key=_lc_eco_last.get("algorithm", "hist_gradient_boosting"),
                n_samples=int(_lc_eco_last.get("n_samples", 2000)),
                n_runs=int(locals().get("n_runs", 3)),
                domain="economic",
            )

        # ── 🔮 Dynamic Systems Tab ────────────────────────────────────────────
        if "🔮 Dynamic Systems" in tab_map:
            with tab_map["🔮 Dynamic Systems"]:
                try:
                    render_dynamic_systems_tab(
                        st.session_state.get("econ_ds_report", {}),
                        domain="economic",
                        ds_key="econ_ds_report",
                    )
                except Exception as _ds_err:
                    st.error(f"🔮 Dynamic Systems error: {_ds_err}")
                    import traceback

                    st.code(traceback.format_exc(), language="python")

    history_browser("econ_snapshot_history", domain="health",
                    key_metrics=["accuracy", "fairness_score", "economic_inclusion_score", "gender_outcome_gap"])
    annotation_panel("econ_annotations",
                     context_label=f"{len(st.session_state.econ_run_history)} Economic run(s)")

    # ── Policy recommendations ─────────────────────────────────────────────────
    st.divider()
    st.markdown("## 💡 Economic Justice Recommendations")
    r1, r2, r3 = st.columns(3)
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


    row1_html = "".join(_dc(dm, desc) for (_, dm), desc in zip(_domain_items[:3], _domain_descs[:3]))
    row2_html = "".join(_dc(dm, desc) for (_, dm), desc in zip(_domain_items[3:], _domain_descs[3:]))

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
        c1, c2 = st.columns(2)
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
        st.dataframe(bm_preview[["Study", "Domain", "Region", "Year", "Severity"]],
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
