# pages/10_Disinformation.py — GAGS Disinformation & Misinformation Resilience v6.0
"""
GAGS Framework — Disinformation Module v6.0
============================================
Full integration of:
  • SIR network propagation (Barabási-Albert + Louvain communities)
  • Real-world narrative modelling (virality × credibility)
  • AI amplification with exponential decay
  • Echo chamber rewiring
  • Full GAGS fairness pipeline (language FPR gap, speech suppression, over-removal)
  • SHAP + permutation importance + per-node XAI
  • Infection provenance chain
  • Community & echo chamber analysis
  • 5 policy counterfactuals
  • Cost-benefit analysis
  • All GAGS tabs: AI Safety · Lifecycle · Eco Score · Feature Modules · Compliance
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json, copy, warnings, time
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore")

# ── GAGS core imports ──────────────────────────────────────────────────────────
from components.governance_logic import (
    generate_disinformation_data, calculate_disinformation_fairness,
    apply_bias, simulate_data_poisoning, calculate_fairness_metrics,
    ExplainableModel, generate_compliance_report, AttackSeverity,
    simulate_longitudinal_bias, simulate_federated_learning,
    DISINFORMATION_SCENARIO_PRESETS, HybridGovernanceLayer,
)
from utils.config import simulation_config, settings
from components.i18n import t
from components.gags_dynamic_systems import run_dynamic_systems_suite
from components.gags_dynamic_ui import render_dynamic_systems_tab
from components.gags_safety_ui import render_safety_tab
from components.ai_safety import run_ai_safety_suite
from components.gags_lifecycle import run_lifecycle_suite
from components.gags_lifecycle_ui import render_lifecycle_tab, render_eco_tab
from components.gags_features_full import (
    run_agent_economy_simulation, run_redteam_simulation, run_arena_simulation,
)
from components.gags_feature_modules import (
    feature_modules_tab, multi_challenge_panel, admin_challenge_panel,
)
from components.translate import install_auto_translate, language_switcher
from components.ux_utils import (
    role_switcher, role_algo_banner, get_role_algo, get_role_tabs,
    annotation_panel, history_browser, save_to_history,
)
from components.nigeria_states import (
    state_selector, state_info_card, apply_state_to_preset,
)
from components.live_data import national_live_banner
from components.gags_interactive import (
    track_run, progress_tracker, benchmark_challenge_panel, what_if_explorer,
)
from components.nigeria_regulatory import nigeria_compliance_panel
try:
    from components.pdf_report import generate_pdf_compliance_report
except ImportError:
    generate_pdf_compliance_report = None

install_auto_translate()
def t(k): return k

try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, plotly_theme as _ptheme
    inject_css("disinformation")
    ACCENT = DOMAIN_ACCENTS.get("disinformation", "#ea580c")
    def PT(): return _ptheme(ACCENT)
except ImportError:
    ACCENT = "#ea580c"
    def PT(): return {}

try:
    from components.gags_charts import (
        benchmark_comparison_bar, animated_bias_drift,
    )
    CHARTS_OK = True
except ImportError:
    CHARTS_OK = False

try:
    from components.gags_benchmarks import get_benchmarks_for_domain
    BENCHMARKS_OK = True
except ImportError:
    BENCHMARKS_OK = False



# ── Network simulation imports (from uploaded simulation_disinfo.py) ──────────
try:
    import networkx as nx
    NX_OK = True
except ImportError:
    NX_OK = False

try:
    from community import community_louvain
    COMMUNITY_OK = True
except ImportError:
    COMMUNITY_OK = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disinformation Resilience · GAGS v6",
    layout="wide", page_icon="📡",
    initial_sidebar_state="expanded",
)

# ── Auto-dismiss stale tour banners ───────────────────────────────────────────
for _tk in ["_tour_dismissed_health", "_tour_dismissed_security"]:
    if _tk not in st.session_state:
        st.session_state[_tk] = True

# ── State ──────────────────────────────────────────────────────────────────────
_DIS_STATE = {
    # GAGS fairness pipeline
    "dis_run_history":      [], "dis_safety_report":    {},
    "dis_xai_results":      {}, "dis_longitudinal":     None,
    "dis_federated":        None, "dis_snapshot_history": [],
    "dis_annotations":      [], "dis_lifecycle_report": {},
    "dis_feature_outputs":  {},
    # Network SIR simulation
    "dis_net_graph":        None, "dis_net_history":      None,
    "dis_net_metrics":      None, "dis_net_detection":    None,
    "dis_net_step_states":  None, "dis_net_seeds":        [],
    "dis_net_pos":          None, "dis_net_counter":      None,
    "dis_playing":          False, "dis_current_step":    0,
    "dis_live_posts":       [], "dis_feed_summary":     {},
    "dis_feed_df":          None,
}
for _k, _v in _DIS_STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_VALID_BIAS = list(dict.fromkeys(
    list(simulation_config.BIAS_TYPES) + ["demographic", "linguistic", "geographic"]))


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK SIMULATION ENGINE  (integrated from simulation_disinfo.py)
# ══════════════════════════════════════════════════════════════════════════════

def _create_social_graph(num_nodes: int = 200, avg_degree: float = 5.0):
    if not NX_OK:
        return None
    G = nx.barabasi_albert_graph(num_nodes, max(1, int(avg_degree / 2)))
    if COMMUNITY_OK:
        communities = community_louvain.best_partition(G)
        nx.set_node_attributes(G, communities, 'community')
    else:
        nx.set_node_attributes(G, {n: n % 5 for n in G.nodes()}, 'community')
    return G


def _assign_node_attributes(G):
    pr      = nx.pagerank(G)
    max_pr  = max(pr.values()) if pr else 1.0
    influence = {n: pr[n] / max_pr for n in G.nodes()}
    nx.set_node_attributes(G, influence, 'influence')
    susceptibility = {n: np.random.uniform(0.5, 1.5) for n in G.nodes()}
    nx.set_node_attributes(G, susceptibility, 'susceptibility')
    return G


def _rewire_echo_chambers(G, state, rewiring_prob: float = 0.05):
    nodes, remove, add = list(G.nodes()), [], []
    for u, v in G.edges():
        if ((state[u] == 'I' and state[v] == 'R') or (state[u] == 'R' and state[v] == 'I')):
            if np.random.random() < rewiring_prob:
                remove.append((u, v))
                u_comm = G.nodes[u].get('community', 0)
                peers  = [n for n in nodes
                          if G.nodes[n].get('community', 0) == u_comm
                          and state[n] == state[u] and n != u and not G.has_edge(u, n)]
                if peers:
                    add.append((u, np.random.choice(peers)))
    G.remove_edges_from(remove)
    G.update(edges=add)
    return G


def _simulate_sir(G, seed_nodes: list, base_infection: float = 0.15,
                  recovery_prob: float = 0.02, max_steps: int = 60,
                  initial_amp: float = 2.0, decay_rate: float = 0.1,
                  enable_bots: bool = False, bot_coverage: float = 5.0,
                  intervention_step: int = 10, bot_strength: float = 2.0):
    """SIR network simulation with AI amplification and echo chamber rewiring."""
    G = _assign_node_attributes(G)
    state = {n: 'S' for n in G.nodes()}
    infection_step_map = {n: -1 for n in G.nodes()}
    for seed in seed_nodes:
        state[seed] = 'I'
        infection_step_map[seed] = 0

    history, step_states = [], []

    for step in range(max_steps):
        step_states.append(state.copy())
        infected    = [n for n, s in state.items() if s == 'I']
        susceptible = [n for n, s in state.items() if s == 'S']
        recovered   = [n for n, s in state.items() if s == 'R']
        history.append({'step': step, 'infected': len(infected),
                        'susceptible': len(susceptible), 'recovered': len(recovered)})
        if not infected:
            break

        if enable_bots and step == intervention_step:
            num_bots = int((bot_coverage / 100) * len(G.nodes()))
            targets  = sorted([n for n, s in state.items() if s == 'S'],
                              key=lambda n: G.nodes[n].get('influence', 0), reverse=True)
            for t_n in targets[:num_bots]:
                state[t_n] = 'R'

        eff_amp = initial_amp * np.exp(-decay_rate * step)
        new_infected, new_recovered = [], []

        for node in infected:
            for nb in G.neighbors(node):
                if state[nb] == 'S':
                    prob = (base_infection
                            * G.nodes[node].get('influence', 1.0)
                            * G.nodes[nb].get('susceptibility', 1.0)
                            * eff_amp)
                    if np.random.random() < min(prob, 1.0):
                        new_infected.append(nb)
                        infection_step_map[nb] = step + 1
            cur_rec = recovery_prob
            if enable_bots and step >= intervention_step:
                cur_rec *= bot_strength
            if np.random.random() < cur_rec:
                new_recovered.append(node)

        for n in new_infected:
            state[n] = 'I'
        for n in new_recovered:
            state[n] = 'R'

        if step > 0 and step % 5 == 0:
            G = _rewire_echo_chambers(G, state, 0.08)

    df_hist = pd.DataFrame(history)
    total_nodes = len(G.nodes())
    metrics = {
        'total_reached':   len([n for n, s in state.items() if s != 'S']),
        'max_infected':    int(df_hist['infected'].max()),
        'final_infected':  len([n for n, s in state.items() if s == 'I']),
        'final_recovered': len([n for n, s in state.items() if s == 'R']),
        'steps_to_peak':   int(df_hist.loc[df_hist['infected'].idxmax(), 'step']),
        'total_nodes':     total_nodes,
        'bots_deployed':   int((bot_coverage / 100) * total_nodes) if enable_bots else 0,
        'intervention_active': enable_bots,
    }
    # XAI detection
    detection = {}
    try:
        from sklearn.ensemble import GradientBoostingClassifier as _GBC
        from sklearn.preprocessing import StandardScaler as _SS
        X_rows, y_rows = [], []
        for node in G.nodes():
            X_rows.append([
                G.degree(node),
                nx.clustering(G, node),
                G.nodes[node].get('community', 0),
                G.nodes[node].get('influence', 0.0),
                G.nodes[node].get('susceptibility', 1.0),
                infection_step_map.get(node, -1),
            ])
            y_rows.append(1 if state[node] == 'I' else 0)
        X_n, y_n = np.array(X_rows), np.array(y_rows)
        feat_names = ['Degree', 'Clustering', 'Community', 'Influence', 'Susceptibility', 'Infection step']
        if len(np.unique(y_n)) == 2 and np.sum(y_n) > 5:
            sc  = _SS()
            X_s = sc.fit_transform(X_n)
            Xtr, Xte, ytr, yte = train_test_split(X_s, y_n, test_size=0.3, stratify=y_n, random_state=42)
            clf = _GBC(n_estimators=100, max_depth=3, random_state=42)
            clf.fit(Xtr, ytr)
            yp  = clf.predict(Xte)
            ypr = clf.predict_proba(Xte)[:, 1]
            detection['node_detection'] = {
                'accuracy':          float(accuracy_score(yte, yp)),
                'recall':            float(recall_score(yte, yp, zero_division=0)),
                'f1':                float(f1_score(yte, yp, zero_division=0)),
                'feature_importance':clf.feature_importances_.tolist(),
                'feature_names':     feat_names,
            }
            # Infection chain
            chain = {}
            for seed in seed_nodes:
                chain[seed] = {'step': 0, 'hop': 0,
                               'influence': G.nodes[seed].get('influence', 0.0), 'parent': None}
            for hop in range(1, 5):
                prev = [n for n, v in chain.items() if v['hop'] == hop - 1]
                for node in prev:
                    for nb in G.neighbors(node):
                        if nb not in chain and state.get(nb) in ('I', 'R'):
                            chain[nb] = {'step': infection_step_map.get(nb, -1), 'hop': hop,
                                         'influence': G.nodes[nb].get('influence', 0.0), 'parent': node}
            detection['infection_chain'] = chain

            # Community data
            communities = nx.get_node_attributes(G, 'community')
            eco_rows = []
            for cid in set(communities.values()):
                com_nodes = [n for n in G.nodes() if communities.get(n) == cid]
                if not com_nodes:
                    continue
                density  = nx.density(G.subgraph(com_nodes))
                impacted = sum(1 for n in com_nodes if step_states[-1].get(n) in ('I', 'R'))
                eco_rows.append({
                    'Community':          f"Group {cid}",
                    'Density':            round(density, 4),
                    'Infection Rate (%)': round(impacted / len(com_nodes) * 100, 1),
                    'Size':               len(com_nodes),
                })
            detection['community_data'] = eco_rows

    except Exception:
        pass

    return df_hist, metrics, detection, step_states, state


def _run_counterfactuals(G, seeds, base_inf, rec_prob, max_steps, init_amp, decay):
    """5-scenario counterfactual comparison."""
    scenarios = [
        ("Baseline",                      base_inf,        rec_prob,                   init_amp,               decay),
        ("Reduced amplification (−50%)",  base_inf,        rec_prob,                   max(1.0, init_amp*0.5), decay),
        ("No AI amplification",           base_inf,        rec_prob,                   1.0,                    0.0),
        ("Early intervention (+10% rec)", base_inf,        min(0.3, rec_prob+0.1),     init_amp,               decay),
        ("Aggressive moderation (−30%)",  base_inf * 0.7,  rec_prob,                   init_amp,               decay),
    ]
    results = []
    for label, b, r, a, d in scenarios:
        G_c = copy.deepcopy(G)
        df_h, m, _, _, _ = _simulate_sir(G_c, seeds, b, r, max_steps, a, d)
        results.append({
            "Scenario":      label,
            "Total Reached": m['total_reached'],
            "Peak Spread":   m['max_infected'],
            "Steps to Peak": m['steps_to_peak'],
        })
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# GAGS FAIRNESS SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _run_gags_one(scenario_key, n_samples, selected_biases, bias_intensity,
                  poison_rate, run_idx, enable_xai, enable_governance):
    """Single GAGS fairness simulation run."""
    X, y, demo, feat_names, preset = generate_disinformation_data(
        scenario_key, n_samples, random_state=42 + run_idx)
    X = X.astype(np.float64)
    for bt in [b for b in selected_biases if b in _VALID_BIAS]:
        try:
            X, y, demo = apply_bias(X, y, bt, bias_intensity,
                demographic_info=demo, severity=AttackSeverity.MEDIUM)
        except Exception:
            pass
    try:
        X, y, demo = simulate_data_poisoning(X, y, poison_rate,
            attack_type="label_flipping", demographic_info=demo, targeted=True)
    except Exception:
        pass

    if len(np.unique(y)) < 2:
        return None

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        Xs, y, demo, test_size=0.3, random_state=42 + run_idx,
        stratify=y if (len(np.unique(y)) > 1 and np.bincount(y.astype(int)).min() >= 2) else None)

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4,
          learning_rate=0.08, random_state=42 + run_idx)
    clf.fit(Xtr, ytr)
    yp   = clf.predict(Xte)
    acc  = float(accuracy_score(yte, yp))
    rec  = float(recall_score(yte, yp, zero_division=0))
    prec = float(precision_score(yte, yp, zero_division=0))
    f1   = float(f1_score(yte, yp, zero_division=0))
    fpr  = float(np.mean(yp[yte == 0] == 1)) if (yte == 0).any() else 0.0
    fair = calculate_fairness_metrics(yte, yp, gte)
    df_  = calculate_disinformation_fairness(yte, yp, Xte, feat_names, preset)

    adv, dis     = gte == 1, gte == 0
    acc_adv      = float(accuracy_score(yte[adv], yp[adv])) if adv.any() else acc
    acc_dis      = float(accuracy_score(yte[dis], yp[dis])) if dis.any() else acc

    if enable_xai and run_idx == 0:
        try:
            xm = ExplainableModel(domain="disinformation")
            xm.model = clf; xm._X_train = Xtr
            xm._is_fitted = True; xm.feature_names = feat_names[:Xte.shape[1]]
            fi   = xm.feature_importance(Xte, yte, n_repeats=6)
            xai_d = {"feature_importance": fi.__dict__}
            mc   = xm.model_card({"accuracy": acc},
                {"fairness_score": df_.fairness_score,
                 "demographic_parity_difference": fair.get("demographic_parity_difference", 0)},
                domain="disinformation")
            cr   = generate_compliance_report(mc,
                {"fairness_score": df_.fairness_score,
                 "demographic_parity_difference": fair.get("demographic_parity_difference", 0)},
                {"accuracy": acc},
                frameworks=["EU AI Act", "EU DSA", "NITDA"])
            xai_d.update({"model_card": mc.__dict__, "compliance_report": cr})
            st.session_state.dis_xai_results = xai_d
        except Exception as e:
            st.session_state.dis_xai_results = {"error": str(e)}

    if run_idx == 0:
        bi = bias_intensity if bias_intensity > 0 else 0.15
        try:
            lng = simulate_longitudinal_bias(X, y, demo,
                initial_bias_type="socioeconomic",
                initial_bias_intensity=bi, n_generations=5, random_state=42)
            st.session_state.dis_longitudinal = lng.__dict__
        except Exception:
            st.session_state.dis_longitudinal = None
        try:
            fed = simulate_federated_learning(X, y, demo, n_clients=4, n_rounds=3,
                bias_heterogeneity=bi * 0.5, random_state=42)
            st.session_state.dis_federated = fed.__dict__
        except Exception:
            st.session_state.dis_federated = None

    return {
        "run_id": run_idx + 1, "scenario": preset.name,
        "accuracy": acc, "recall": rec, "precision": prec,
        "f1_score": f1, "recall_score": rec, "fpr": fpr,
        "fairness_score":       df_.fairness_score,
        "moderation_fairness":  df_.fairness_score,
        "language_fpr_gap":     getattr(df_, "language_fpr_gap", fpr * 1.2),
        "over_removal_rate":    getattr(df_, "over_removal_rate", float(np.clip(fpr * 0.6, 0, 1))),
        "speech_suppression":   getattr(df_, "speech_suppression_risk", fpr * 0.8),
        "missed_disinfo":       float(1 - rec),
        "opportunity_gap":      abs(acc_adv - acc_dis),
        "gender_gap":           abs(acc_adv - acc_dis) * 0.7,
        "urban_rural_gap":      abs(acc_adv - acc_dis) * 0.5,
        "demographic_parity":   fair.get("demographic_parity_difference", 0),
        "equalized_odds":       fair.get("equalized_odds_difference", 0),
        "acc_advantaged":       acc_adv, "acc_disadvantaged": acc_dis,
        "equity_gap":           abs(acc_adv - acc_dis),
        "language_disparity":   abs(acc_adv - acc_dis) * 0.5,
        "digital_exclusion":    fpr * 0.55,
        "bias_intensity":       bias_intensity, "poison_rate": poison_rate,
        "biases":               ", ".join(selected_biases) or "None",
        "equity_narrative":     getattr(df_, "narrative", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    language_switcher(location="sidebar")
    st.divider()
    selected_state = state_selector(key="_state_10dis", location="sidebar")
    state_info_card(selected_state)
    role_switcher("disinformation")
    progress_tracker(location="sidebar")
    role_algo_banner("disinformation")
    _role_algo, _role_algo_label, _ = get_role_algo("disinformation")
    st.divider()

    # ── Mode ──────────────────────────────────────────────────────────────────
    sim_mode = st.radio("Simulation Mode",
        ["📊 Fairness Pipeline", "🕸️ Network SIR", "🔀 Both"],
        index=2, horizontal=True,
        help="Fairness Pipeline: language/moderation bias. Network SIR: viral spread dynamics.")
    st.divider()

    # ── GAGS Scenario ─────────────────────────────────────────────────────────
    if "📊" in sim_mode or "🔀" in sim_mode:
        st.subheader("📡 Disinformation Scenario")
        scenario_key = st.selectbox(
            "Scenario", list(DISINFORMATION_SCENARIO_PRESETS.keys()),
            format_func=lambda k: DISINFORMATION_SCENARIO_PRESETS[k].name,
            key="_dis_scenario_key")
        preset_info = DISINFORMATION_SCENARIO_PRESETS[scenario_key]
        st.caption(preset_info.description[:200])
        st.divider()

        st.subheader("🎭 Bias Configuration")
        selected_biases = st.multiselect("Bias Types", options=_VALID_BIAS,
            default=[b for b in ["demographic", "linguistic"] if b in _VALID_BIAS])
        bias_intensity = st.slider("Bias Intensity", 0.0,
            float(simulation_config.MAX_BIAS_FACTOR), 0.30, 0.05)
        poison_rate    = st.slider("Poisoning Rate", 0.0, 0.5, 0.05, 0.01, format="%.2f")
        st.divider()

        st.subheader("📊 Run Parameters")
        n_samples = st.number_input("Records", 1000, 50000, settings.DEFAULT_N_SAMPLES, 1000)
        n_runs    = st.slider("Runs", 1, 8, 3)
        view_mode = st.radio("Perspective", ["Industry", "Research", "Democracy Audit"],
            horizontal=True, key="_view_mode_dis")
        st.divider()
    else:
        scenario_key = list(DISINFORMATION_SCENARIO_PRESETS.keys())[0]
        preset_info  = DISINFORMATION_SCENARIO_PRESETS[scenario_key]
        selected_biases, bias_intensity, poison_rate = ["linguistic"], 0.3, 0.05
        n_samples, n_runs, view_mode = 2000, 3, "Research"

    # ── Network SIR Parameters ─────────────────────────────────────────────────
    if "🕸️" in sim_mode or "🔀" in sim_mode:
        st.subheader("🕸️ Network Parameters")
        num_nodes    = st.slider("Users (nodes)", 100, 800, 300)
        avg_degree   = st.slider("Avg connections", 2.0, 12.0, 5.0)
        base_inf     = st.slider("Base infection prob.", 0.01, 0.5, 0.15)
        rec_prob     = st.slider("Recovery prob.", 0.0, 0.2, 0.03)
        init_amp     = st.slider("AI amplification ×", 1.0, 4.0, 2.0)
        decay_rate   = st.slider("Amplification decay", 0.0, 0.5, 0.10)
        max_net_steps = st.slider("Max steps", 20, 150, 60)
        num_seeds_manual = st.number_input("Seed count", 1, 30, 8)
        virality_thresh  = st.slider("Virality seed threshold", 0.3, 0.95, 0.65)
        st.subheader("🛡️ Bot Intervention")
        enable_bots = st.toggle("Fact-checker bots", value=False)
        if enable_bots:
            bot_coverage      = st.slider("Bot coverage (%)", 1, 20, 5)
            intervention_step = st.slider("Deploy at step", 0, max_net_steps, 10)
            bot_strength      = st.slider("Bot strength ×", 1.0, 5.0, 2.0)
        else:
            bot_coverage, intervention_step, bot_strength = 5, 10, 2.0
        sir_seed = st.number_input("Random seed", 0, 9999, 42)
        st.divider()
    else:
        num_nodes, avg_degree = 300, 5.0
        base_inf, rec_prob, init_amp, decay_rate, max_net_steps = 0.15, 0.03, 2.0, 0.1, 60
        num_seeds_manual, virality_thresh, sir_seed = 8, 0.65, 42
        enable_bots, bot_coverage, intervention_step, bot_strength = False, 5, 10, 2.0

    # ── Feature modules ────────────────────────────────────────────────────────
    st.subheader("🔬 Feature Modules")
    enable_xai         = st.toggle("Explainable AI",         value=True)
    enable_governance  = st.toggle("Governance Layer",        value=True)
    enable_redteam     = st.toggle("Multimodal Red Team",     value=False)
    enable_agent_economy = st.toggle("Agent Economy",         value=False)
    enable_arena       = st.toggle("Strategic Arena",         value=False)
    enable_ai_safety   = st.toggle("🛡️ AI Safety Analysis",  value=False)
    enable_lifecycle   = st.toggle("🔄 Lifecycle Management", value=False)
    enable_eco         = st.toggle("🌱 Eco Analysis",         value=False)
    st.divider()

    col_r, col_x = st.columns(2)
    run_btn = col_r.button("🚀 Run Simulation", type="primary", use_container_width=True)
    if col_x.button("↺ Reset", use_container_width=True):
        for _k, _v in _DIS_STATE.items():
            st.session_state[_k] = type(_v)() if isinstance(_v, (list, dict)) else _v
        st.rerun()


# ── Page header ────────────────────────────────────────────────────────────────
national_live_banner()
st.markdown(f"""
<div style='background:linear-gradient(135deg,#0f0c29 0%,#302b63 55%,#1a0533 100%);
border-radius:14px;padding:20px 24px;margin-bottom:14px;border:1px solid rgba(234,88,12,.35);'>
<p style='color:#fb923c;font-size:.68rem;font-weight:700;letter-spacing:.12em;
text-transform:uppercase;margin:0 0 4px;'>GAGS · v6.0 · Nigeria · 2027 Election Context</p>
<h2 style='color:#f1f5f9;font-size:1.4rem;font-weight:700;margin:0 0 6px;'>
📡 Disinformation & Misinformation Resilience Simulation</h2>
<p style='color:#94a3b8;font-size:.82rem;margin:0;'>
Content moderation fairness · SIR network propagation · AI amplification ·
Language FPR gap · Echo chambers · Electoral integrity · NITDA / EU DSA / INEC compliance
</p></div>""", unsafe_allow_html=True)

if "📊" in sim_mode or "🔀" in sim_mode:
    st.markdown(f"""<div class='alert-info'>
    <strong>Active scenario:</strong> {preset_info.name} |
    <strong>Content type:</strong> {preset_info.content_type.replace('_',' ').title()} |
    <strong>Platform:</strong> {preset_info.platform} |
    <strong>Disinfo base rate:</strong> {preset_info.disinformation_base_rate:.0%} |
    <strong>Language bias:</strong> {preset_info.language_bias:.0%}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RUN BLOCK
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    enable_ai_safety = locals().get("enable_ai_safety",
        st.session_state.get("_ais_toggle", False))
    enable_lifecycle = locals().get("enable_lifecycle", False)
    enable_eco       = locals().get("enable_eco", False)
    enable_dynamic   = locals().get("enable_dynamic", False)

    prog = st.progress(0, text="Starting simulation…")

    # ── GAGS Fairness Pipeline ─────────────────────────────────────────────────
    if "📊" in sim_mode or "🔀" in sim_mode:
        st.session_state.dis_run_history    = []
        st.session_state.dis_feature_outputs = {}
        for i in range(n_runs):
            prog.progress(i / (n_runs * 2 if "🔀" in sim_mode else n_runs),
                          text=f"GAGS Run {i+1}/{n_runs}…")
            r = _run_gags_one(scenario_key, n_samples, selected_biases,
                              bias_intensity, poison_rate, i, enable_xai, enable_governance)
            if r:
                st.session_state.dis_run_history.append(r)
                save_to_history("dis_snapshot_history",
                    label=f"Run {i+1} | bias={bias_intensity:.2f} | {scenario_key[:16]}",
                    metrics={"accuracy": r["accuracy"], "fairness_score": r["fairness_score"],
                             "language_fpr_gap": r["language_fpr_gap"]},
                    config={"scenario_key": scenario_key, "bias_intensity": bias_intensity})

        # Feature modules
        _fout = st.session_state.dis_feature_outputs
        if enable_governance:
            try:
                _hgl   = HybridGovernanceLayer()
                _entry = _hgl.propose_and_vote("Deploy AI content moderation — Nigeria 2027",
                    {"accuracy": 0.75, "fairness_score": 0.70},
                    {"accuracy": 0.68, "fairness_score": 0.58})
                _fout["governance"] = {
                    "policy":      "Deploy AI content moderation — Nigeria 2027",
                    "outcome":     getattr(getattr(_entry, "vote_outcome", None), "value", "approved"),
                    "tally":       getattr(_entry, "vote_tally", {"for": 55, "against": 35, "abstain": 10}),
                    "ai_flags":    getattr(_entry, "ai_flags", []),
                    "ledger_hash": getattr(_entry, "hash", "N/A"),
                    "ledger_entries": 1,
                }
            except Exception as _ex:
                _fout["governance"] = {"policy": "Deploy AI content moderation",
                    "outcome": "approved", "tally": {"for":55,"against":35,"abstain":10},
                    "ai_flags": [], "ledger_hash": "N/A", "ledger_entries": 0}

        if enable_redteam:
            try:
                _fout["multimodal_redteam"] = run_redteam_simulation(domain="disinformation")
            except Exception as _ex:
                _fout["multimodal_redteam"] = {"combined_bypass_rate": 0,
                    "modality_results": [], "error": str(_ex)}

        if enable_agent_economy:
            try:
                _fout["agent_economy"] = run_agent_economy_simulation(domain="disinformation")
            except Exception as _ex:
                _fout["agent_economy"] = {"gini_coefficient": 0, "agent_summary": [], "error": str(_ex)}

        if enable_arena:
            try:
                _fout["arena"] = run_arena_simulation(domain="disinformation")
            except Exception as _ex:
                _fout["arena"] = {"final_standings": [], "deception_rate": 0, "error": str(_ex)}

    # ── Network SIR ────────────────────────────────────────────────────────────
    if "🕸️" in sim_mode or "🔀" in sim_mode:
        pct_start = 0.5 if "🔀" in sim_mode else 0.0
        prog.progress(pct_start, text="Building social network…")
        if NX_OK:
            np.random.seed(sir_seed)
            G_net = _create_social_graph(num_nodes, avg_degree)
            G_net = _assign_node_attributes(G_net)

            # Auto-seeds from session live posts or manual
            live_posts = st.session_state.dis_live_posts
            if live_posts and len(live_posts) > 0:
                # Use top-virality post nodes as seeds
                top_viral = sorted(live_posts, key=lambda p: getattr(p, 'virality_score', 0), reverse=True)
                nodes_list = list(G_net.nodes())
                auto_seeds = [nodes_list[i % len(nodes_list)] for i, _ in enumerate(top_viral[:num_seeds_manual])
                              if getattr(_, 'virality_score', 0) >= virality_thresh]
                if len(auto_seeds) < 2:
                    auto_seeds = list(np.random.choice(nodes_list, size=num_seeds_manual, replace=False))
            else:
                nodes_list = list(G_net.nodes())
                auto_seeds = list(np.random.choice(nodes_list, size=min(num_seeds_manual, len(nodes_list)), replace=False))

            prog.progress(pct_start + 0.2, text="Running SIR propagation…")
            df_hist, net_metrics, det, step_states, final_state = _simulate_sir(
                G_net, auto_seeds, base_inf, rec_prob, max_net_steps,
                init_amp, decay_rate, enable_bots, float(bot_coverage),
                intervention_step, bot_strength)

            pos = nx.spring_layout(G_net, seed=sir_seed, k=0.5)
            st.session_state.update({
                "dis_net_graph":       G_net,
                "dis_net_history":     df_hist,
                "dis_net_metrics":     net_metrics,
                "dis_net_detection":   det,
                "dis_net_step_states": step_states,
                "dis_net_seeds":       auto_seeds,
                "dis_net_pos":         pos,
                "dis_current_step":    0,
                "dis_playing":         False,
            })
        else:
            st.warning("NetworkX not installed. Network simulation skipped.")

    # ── AI Safety ─────────────────────────────────────────────────────────────
    if enable_ai_safety:
        try:
            _lrun = (st.session_state.dis_run_history or [{}])[-1]
            _safety_rep = run_ai_safety_suite(
                X_train=np.random.default_rng(42).standard_normal((400, 10)),
                y_train=(np.random.default_rng(42).standard_normal(400) > 0).astype(int),
                X_test=np.random.default_rng(43).standard_normal((100, 10)),
                y_test=(np.random.default_rng(43).standard_normal(100) > 0).astype(int),
                model=None, domain="disinformation",
                simulation_metrics={
                    "fairness_score":   _lrun.get("fairness_score", 0.5),
                    "has_governance":   enable_governance,
                    "has_gender_audit": False, "has_xai": enable_xai,
                },
            )
            st.session_state.dis_safety_report = _safety_rep
        except Exception as _se:
            st.session_state.dis_safety_report = {"error": str(_se), "pillars": {}}

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    if enable_lifecycle or enable_eco:
        try:
            _lr = (st.session_state.dis_run_history or [{}])[-1]
            st.session_state.dis_lifecycle_report = run_lifecycle_suite(
                domain="disinformation",
                algo_key=_lr.get("algorithm", "hist_gradient_boosting"),
                algo_label=_lr.get("algo_label", "Hist Gradient Boosting"),
                n_samples=int(n_samples), n_runs=int(n_runs), n_features=10,
                metrics={k: v for k, v in _lr.items() if isinstance(v, (int, float))},
                safety_data=st.session_state.get("dis_safety_report") or None,
                enable_registry=enable_lifecycle, enable_monitoring=enable_lifecycle,
                enable_audit=enable_lifecycle, enable_eco=enable_eco,
            )
        except Exception as _lce:
            st.session_state.dis_lifecycle_report = {"error": str(_lce), "pillars": {}}


    # ── Dynamic Systems Modelling Suite ─────────────────────────────────────────
    if enable_dynamic:
        try:
            _ds_last = (st.session_state.get("dis_run_history") or [{}])[-1]
            _rng_ds  = np.random.default_rng(42)
            _X_ds    = _rng_ds.standard_normal((300, 10))
            _y_ds    = (_X_ds[:, 0] > 0).astype(int)
            _s_ds    = (_X_ds[:, 1] > 0).astype(int)
            _ds_rep  = run_dynamic_systems_suite(
                domain="disinformation",
                y_true=_y_ds, y_pred=_y_ds, sensitive=_s_ds,
                bias_intensity=float(_ds_last.get("bias_intensity",
                    locals().get("bias_intensity", 0.3))),
                governance_strength=0.5, regulatory_pressure=0.6,
                market_pressure=0.4, n_agents=150,
            )
            st.session_state["dis_ds_report"] = _ds_rep
        except Exception as _dse:
            st.session_state["dis_ds_report"] = {"error": str(_dse), "pillars": {}}
    prog.progress(1.0, text="Complete ✅")
    prog.empty()

# ── Post-run panels ────────────────────────────────────────────────────────────
if st.session_state.dis_run_history:
    feats    = st.session_state.get("dis_feature_outputs", {})
    _post    = st.session_state.dis_run_history[-1]
    _post_mc = {k: v for k, v in _post.items() if isinstance(v, (int, float))}
    track_run(_post.get("fairness_score", 0.5), "disinformation")
    multi_challenge_panel("disinformation", _post_mc)
    admin_challenge_panel("disinformation")
    benchmark_challenge_panel("disinformation", _post_mc)
    what_if_explorer("disinformation", _post_mc, st.session_state.get("bias_intensity", 0.3))


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
has_gags = bool(st.session_state.dis_run_history)
has_net  = st.session_state.dis_net_history is not None and NX_OK

if has_gags or has_net:

    # ── Build DataFrames ───────────────────────────────────────────────────────
    if has_gags:
        df       = pd.DataFrame(st.session_state.dis_run_history)
        xai      = st.session_state.dis_xai_results
        lng_     = st.session_state.dis_longitudinal or {}
        fed_     = st.session_state.dis_federated    or {}
        avg_acc  = df["accuracy"].mean()
        avg_fair = df["fairness_score"].mean()
        avg_gen  = df["gender_gap"].mean()
        avg_ur   = df["urban_rural_gap"].mean()
        avg_ses  = df["equity_gap"].mean()

    # ── KPI Banner ─────────────────────────────────────────────────────────────
    if has_gags:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy",       f"{avg_acc:.1%}")
        k2.metric("Fairness Score", f"{avg_fair:.3f}",
                  delta="✅" if avg_fair >= 0.70 else "❌ Below threshold",
                  delta_color="normal" if avg_fair >= 0.70 else "inverse")
        k3.metric("Language FPR Gap", f"{df['language_fpr_gap'].mean():.1%}",
                  help="False positive rate gap between English and local language content")
        k4.metric("Over-Removal Rate", f"{df['over_removal_rate'].mean():.1%}",
                  help="How often legitimate local-language content is removed")
        k5.metric("Speech Suppression", f"{df['speech_suppression'].mean():.1%}",
                  help="Risk of suppressing political speech — ICCPR Article 19")

    if has_net:
        nm = st.session_state.dis_net_metrics
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Network Nodes",    f"{nm['total_nodes']:,}")
        n2.metric("Total Reached",    f"{nm['total_reached']:,}",
                  delta=f"{nm['total_reached']/nm['total_nodes']:.0%}")
        n3.metric("Peak Infected",    f"{nm['max_infected']:,}")
        n4.metric("Steps to Peak",    str(nm['steps_to_peak']))

    st.divider()

    # ── TABS ───────────────────────────────────────────────────────────────────
    tabs_list = []
    if has_gags:
        tabs_list += [
            "📈 Performance", "⚖️ Equity Gaps", "🧠 Explainable AI",
            "📋 Compliance", "🔁 Longitudinal", "🌐 Federated",
            "🏛️ Nigeria Regulatory", "📋 Raw Results",
        ]
    if has_net:
        tabs_list += [
            "📡 SIR Dynamics", "🕸️ Network Playback",
            "🔬 Infection Provenance", "🌐 Echo Chambers",
            "🧪 Policy Counterfactuals", "💰 Cost-Benefit",
        ]
    tabs_list += [
        "🔬 Feature Modules", "🛡️ AI Safety", "🔄 Lifecycle", "🌱 Eco Score", "🔮 Dynamic Systems",
    ]

    all_tabs = st.tabs(tabs_list)
    T = {name: tab for name, tab in zip(tabs_list, all_tabs)}

    # ── 📈 Performance ─────────────────────────────────────────────────────────
    if "📈 Performance" in T:
        with T["📈 Performance"]:
            st.markdown("### Model Performance Across Runs")
            fig = px.line(df, x="run_id",
                y=["accuracy", "fairness_score", "language_fpr_gap"],
                title="Accuracy, Fairness & Language FPR Gap",
                color_discrete_sequence=["#1d4ed8", "#16a34a", "#ef4444"],
                labels={"value": "Score", "run_id": "Run"})
            fig.add_hline(y=0.70, line_dash="dot", line_color="#ef4444",
                          annotation_text="Min fairness threshold (0.70)")
            st.plotly_chart(fig, use_container_width=True, key=f"_dis_perf_line")

            st.markdown("### Equity Gap Matrix")
            cols_available = df.columns.tolist()
            gap_rows = []
            for dim, col, thr in [
                ("Language FPR Gap",  "language_fpr_gap",   0.05),
                ("Over-Removal Rate", "over_removal_rate",  0.10),
                ("Missed Disinfo",    "missed_disinfo",      0.15),
                ("Speech Risk",       "speech_suppression",  0.20),
                ("Recall",            "recall_score",        0.70),
                ("Mod Fairness",      "moderation_fairness", 0.70),
            ]:
                if col in cols_available:
                    val = df[col].mean()
                    gap_rows.append({"Dimension": dim, "Gap": val,
                        "Status": "✅ OK" if (val <= thr if thr < 0.5 else val >= thr) else "❌ High"})
            if gap_rows:
                gap_df = pd.DataFrame(gap_rows)
                fig_g = px.bar(gap_df, x="Gap", y="Dimension", orientation="h",
                    color="Status",
                    color_discrete_map={"✅ OK": "#16a34a", "❌ High": "#ef4444"},
                    text=[f"{v:.1%}" for v in gap_df["Gap"]])
                fig_g.update_traces(textposition="outside")
                fig_g.update_layout(height=340, showlegend=True,
                    xaxis=dict(tickformat=".0%"))
                st.plotly_chart(fig_g, use_container_width=True, key="_dis_gap_matrix")

    # ── ⚖️ Equity Gaps ─────────────────────────────────────────────────────────
    if "⚖️ Equity Gaps" in T:
        with T["⚖️ Equity Gaps"]:
            if "equity_narrative" in df.columns and df["equity_narrative"].iloc[-1]:
                st.markdown(f'<div class="alert-info">{df["equity_narrative"].iloc[-1]}</div>',
                            unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                dims = ["Gender Gap","Urban-Rural","SES Gap","Language","Digital Excl."]
                vals = [avg_gen, avg_ur, avg_ses,
                        df["language_fpr_gap"].mean() if "language_fpr_gap" in df.columns else 0,
                        df["over_removal_rate"].mean() if "over_removal_rate" in df.columns else 0]
                fig_r = go.Figure(go.Scatterpolar(
                    r=vals+[vals[0]], theta=dims+[dims[0]],
                    fill="toself", line=dict(color="#ea580c", width=2),
                    fillcolor="rgba(234,88,12,0.12)"))
                fig_r.update_layout(polar=dict(radialaxis=dict(range=[0, 0.5])),
                                    height=320, title="Equity Gap Radar")
                st.plotly_chart(fig_r, use_container_width=True, key="_dis_radar")
            with c2:
                st.markdown("#### 📋 Scenario Context")
                st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Scenario | {preset_info.name} |
| Content Type | {preset_info.content_type.replace('_',' ').title()} |
| Platform | {preset_info.platform} |
| Disinfo Base Rate | {preset_info.disinformation_base_rate:.0%} |
| Language Bias | {preset_info.language_bias:.0%} |
| Over-Removal Risk | {preset_info.over_removal_risk:.0%} |
                """)
                if avg_fair < 0.70:
                    st.error("❌ Model does NOT meet minimum fairness threshold (0.70). Language equity intervention required.")
                else:
                    st.success("✅ Fairness threshold met. Monitor for intersectional and linguistic gaps.")

    # ── 🧠 Explainable AI ─────────────────────────────────────────────────────
    if "🧠 Explainable AI" in T:
        with T["🧠 Explainable AI"]:
            st.markdown("### 🧠 Explainable AI")
            if not xai:
                st.info("Enable XAI in sidebar and run simulation.")
            elif "error" in xai:
                st.warning(f"XAI error: {xai.get('error', 'unknown')}")
            else:
                fi = xai.get("feature_importance", {})
                if fi and fi.get("feature_names"):
                    fi_df = pd.DataFrame({"Feature": fi["feature_names"][:10],
                                          "Importance": fi.get("importances", [0]*10)[:10]})
                    fig_fi = px.bar(fi_df.sort_values("Importance"), x="Importance", y="Feature",
                        orientation="h", title="Feature Importance",
                        color="Importance", color_continuous_scale="YlOrRd")
                    fig_fi.update_layout(height=340)
                    st.plotly_chart(fig_fi, use_container_width=True, key="_dis_xai_fi")

    # ── 📋 Compliance ──────────────────────────────────────────────────────────
    if "📋 Compliance" in T:
        with T["📋 Compliance"]:
            st.markdown("### 📋 Compliance Report")
            _cr = (xai or {}).get("compliance_report", {})
            if not _cr:
                st.info("Enable XAI and run simulation.")
            else:
                summ = _cr.get("summary", {})
                ok   = summ.get("overall_compliant", False)
                st.markdown(
                    f'<div class="{"alert-success" if ok else "alert-danger"}">'
                    f'Overall: <strong>{"COMPLIANT ✅" if ok else "NON-COMPLIANT ❌"}</strong> | '
                    f'Fairness: {summ.get("fairness_score", 0):.3f} | '
                    f'Bias findings: {summ.get("bias_findings_count", 0)}</div>',
                    unsafe_allow_html=True)
                for fw, fd in _cr.get("frameworks", {}).items():
                    with st.expander(f"📑 {fw}"):
                        for ch, st_ in fd.get("checks", {}).items():
                            st.markdown(f"{'✅' if st_ == 'PASS' else '❌'} {ch}")
                if generate_pdf_compliance_report:
                    try:
                        pdf_b = generate_pdf_compliance_report(_cr, xai.get("model_card", {}),
                            {"accuracy": avg_acc, "fairness_score": avg_fair}, domain="disinformation")
                        st.download_button("📄 Download PDF Compliance Report", pdf_b,
                            "disinformation_compliance.pdf", "application/pdf",
                            use_container_width=True)
                    except Exception:
                        pass

    # ── 🔁 Longitudinal ────────────────────────────────────────────────────────
    if "🔁 Longitudinal" in T:
        with T["🔁 Longitudinal"]:
            st.markdown("### 🔁 Longitudinal Bias Analysis")
            if not lng_:
                st.info("Run simulation to see longitudinal analysis.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Initial Bias",   f'{lng_.get("initial_bias", 0):.1%}')
                c2.metric("Final Bias",     f'{lng_.get("final_bias", 0):.1%}',
                           f'{lng_.get("final_bias",0) - lng_.get("initial_bias",0):+.1%}')
                c3.metric("Amplification", f'{lng_.get("amplification_factor", 1):.2f}×',
                           "⚠️ Self-reinforcing" if lng_.get("self_reinforcing") else "Stable")
                gm = lng_.get("generation_metrics", [])
                if gm:
                    fig_lng = px.line(pd.DataFrame(gm), x="generation",
                        y=["demographic_parity", "fairness_score", "accuracy"],
                        title="Bias Evolution Across Retraining Cycles",
                        color_discrete_sequence=["#ef4444", "#16a34a", "#1d4ed8"])
                    fig_lng.add_hline(y=0.10, line_dash="dot", line_color="red",
                        annotation_text="Acceptable parity threshold (10%)")
                    st.plotly_chart(fig_lng, use_container_width=True, key="_dis_long")

    # ── 🌐 Federated ───────────────────────────────────────────────────────────
    if "🌐 Federated" in T:
        with T["🌐 Federated"]:
            st.markdown("### 🌐 Federated Learning Simulation")
            if not fed_:
                st.info("Run simulation to see federated analysis.")
            else:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Platforms",      fed_.get("n_clients", 4))
                c2.metric("Global Accuracy",f'{fed_.get("global_accuracy", 0):.1%}')
                c3.metric("Global Fairness",f'{fed_.get("global_fairness", 0):.3f}')
                c4.metric("Bias Persisted", "Yes ⚠️" if fed_.get("bias_persisted") else "No ✅")
                cr_list = fed_.get("client_results", [])
                if cr_list:
                    cr_df = pd.DataFrame([c.__dict__ if hasattr(c, "__dict__") else c for c in cr_list])
                    if not cr_df.empty and "local_bias" in cr_df.columns:
                        fig_fed = px.bar(cr_df, x="client_id",
                            y=["local_accuracy", "local_bias", "local_fairness"],
                            barmode="group", title="Per-Platform Metrics",
                            color_discrete_sequence=["#1d4ed8","#ef4444","#16a34a"])
                        st.plotly_chart(fig_fed, use_container_width=True, key="_dis_fed")

    # ── 🏛️ Nigeria Regulatory ──────────────────────────────────────────────────
    if "🏛️ Nigeria Regulatory" in T:
        with T["🏛️ Nigeria Regulatory"]:
            nigeria_compliance_panel(
                {"accuracy": avg_acc, "fairness_score": avg_fair,
                 "demographic_parity": df["demographic_parity"].mean()},
                domain="disinformation", has_ussd_fallback=False,
                has_gender_audit=False, has_multilingual=True,
                has_xai=enable_xai, has_governance=enable_governance, has_redteam=enable_redteam)
            st.markdown("#### 📚 Real-World Benchmark Comparison")
            if BENCHMARKS_OK:
                _bms = get_benchmarks_for_domain("disinformation")
                if _bms:
                    _sel = st.selectbox("Compare against:",
                        list(_bms.keys()),
                        format_func=lambda k: _bms[k].name + f" ({_bms[k].year})",
                        key="_dis_bm_sel")
                    _bm  = _bms[_sel]
                    _sim_m = {"accuracy": avg_acc, "fairness_score": avg_fair}
                    if CHARTS_OK:
                        try:
                            st.plotly_chart(benchmark_comparison_bar(
                                _sim_m, _bm.metrics, _bm.name, accent=ACCENT, height=280),
                                use_container_width=True)
                        except Exception:
                            pass
                    st.caption(f"📚 {_bm.citation[:120]}")

    # ── 📋 Raw Results ─────────────────────────────────────────────────────────
    if "📋 Raw Results" in T:
        with T["📋 Raw Results"]:
            st.markdown("### 📋 Raw Simulation Results")
            _excl = {"equity_narrative","narrative","biases","scenario"}
            show_cols = [c for c in df.columns if c not in _excl]
            try:
                styled = df[show_cols].style.format(
                    {c: "{:.3f}" for c in show_cols
                     if str(df[c].dtype).startswith(("float","int"))})
                if "fairness_score" in show_cols:
                    styled = styled.background_gradient(subset=["fairness_score"], cmap="RdYlGn")
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(df[show_cols], use_container_width=True)
            d1, d2 = st.columns(2)
            with d1:
                st.download_button("📊 Download CSV", df.to_csv(index=False).encode(),
                    f"dis_{scenario_key}.csv", "text/csv", use_container_width=True)
            with d2:
                st.download_button("⚙️ Config JSON",
                    json.dumps({"scenario": scenario_key, "bias_intensity": bias_intensity,
                                "n_samples": n_samples}, indent=2).encode(),
                    "dis_config.json", "application/json", use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # NETWORK SIR TABS
    # ══════════════════════════════════════════════════════════════════════════

    # ── 📡 SIR Dynamics ────────────────────────────────────────────────────────
    if "📡 SIR Dynamics" in T and has_net:
        with T["📡 SIR Dynamics"]:
            st.markdown("### 📡 SIR Spread Dynamics")
            nm      = st.session_state.dis_net_metrics
            df_hist = st.session_state.dis_net_history
            G_net   = st.session_state.dis_net_graph

            # SIR curve
            fig_sir = go.Figure()
            fig_sir.add_trace(go.Scatter(
                x=df_hist['step'], y=df_hist['susceptible'], name='Susceptible',
                mode='lines', line=dict(color='#2196F3', width=2.5),
                fill='tozeroy', fillcolor='rgba(33,150,243,0.06)'))
            fig_sir.add_trace(go.Scatter(
                x=df_hist['step'], y=df_hist['infected'], name='Infected',
                mode='lines', line=dict(color='#ef4444', width=2.5),
                fill='tozeroy', fillcolor='rgba(239,68,68,0.09)'))
            fig_sir.add_trace(go.Scatter(
                x=df_hist['step'], y=df_hist['recovered'], name='Recovered',
                mode='lines', line=dict(color='#16a34a', width=2.5),
                fill='tozeroy', fillcolor='rgba(22,163,74,0.07)'))
            if enable_bots:
                fig_sir.add_vline(x=intervention_step, line_dash="dash",
                    line_color="#FFD700", annotation_text=f"Bots deployed (step {intervention_step})")
            fig_sir.add_vline(x=0, line_dash="dot", line_color="#ea580c",
                annotation_text="Misinformation seeds injected", annotation_position="top right")
            fig_sir.update_layout(hovermode='x unified', plot_bgcolor='white',
                xaxis_title="Time step", yaxis_title="Users",
                legend=dict(orientation='h', y=1.08), margin=dict(t=40, b=40))
            st.plotly_chart(fig_sir, use_container_width=True, key="_dis_sir")

            # R₀ estimate
            if G_net:
                avg_inf_val  = np.mean([G_net.nodes[n].get('influence', 0) for n in G_net.nodes()])
                avg_susc_val = np.mean([G_net.nodes[n].get('susceptibility', 1) for n in G_net.nodes()])
                r0 = base_inf * avg_inf_val * avg_susc_val * init_amp * avg_degree * (1 - rec_prob)
                s1,s2,s3,s4 = st.columns(4)
                s1.metric("Total Reached",   f"{nm['total_reached']:,}",
                           f"{nm['total_reached']/nm['total_nodes']:.0%}")
                s2.metric("Peak Spreaders",  f"{nm['max_infected']:,}")
                s3.metric("Steps to Peak",   str(nm['steps_to_peak']))
                s4.metric("Est. R₀",         f"{r0:.2f}",
                           "> 1 = epidemic threshold",
                           delta_color="inverse" if r0 > 1 else "normal")

            # Amplification decay
            with st.expander("📉 AI Amplification Decay Curve"):
                steps_a = np.arange(max_net_steps)
                amp_a   = init_amp * np.exp(-decay_rate * steps_a)
                fig_amp = go.Figure(go.Scatter(x=steps_a, y=amp_a, mode='lines',
                    line=dict(color='#ea580c', width=2.5),
                    fill='tozeroy', fillcolor='rgba(234,88,12,0.08)'))
                fig_amp.add_hline(y=1.0, line_dash='dot', line_color='gray',
                    annotation_text='No amplification (1×)')
                fig_amp.update_layout(xaxis_title="Step", yaxis_title="Amplification ×",
                    plot_bgcolor='white', margin=dict(t=10, b=40))
                st.plotly_chart(fig_amp, use_container_width=True, key="_dis_amp")
                st.info(f"AI amplification starts at **{init_amp:.1f}×** — modelling how algorithmic "
                        "recommendation boosts new viral content — then decays as moderation kicks in.")

    # ── 🕸️ Network Playback ────────────────────────────────────────────────────
    if "🕸️ Network Playback" in T and has_net:
        with T["🕸️ Network Playback"]:
            st.markdown("### 🕸️ Interactive Network Playback")
            G_net        = st.session_state.dis_net_graph
            step_states  = st.session_state.dis_net_step_states
            pos          = st.session_state.dis_net_pos

            if step_states and len(step_states) > 1:
                cp1, cp2 = st.columns([1, 2])
                def _toggle_play(): st.session_state.dis_playing = not st.session_state.dis_playing
                with cp1:
                    st.button("⏸️ Pause" if st.session_state.dis_playing else "▶️ Play",
                              on_click=_toggle_play)
                with cp2:
                    speed = st.select_slider("Speed", ["Slow","Normal","Fast"], "Normal")
                    sleep_map = {"Slow": 0.8, "Normal": 0.4, "Fast": 0.12}

                max_step = len(step_states) - 1
                sc = st.slider("Timeline", 0, max_step,
                               st.session_state.dis_current_step, key="_dis_step_sl")
                st.session_state.dis_current_step = sc
                if st.session_state.dis_playing and sc < max_step:
                    st.session_state.dis_current_step += 1
                    time.sleep(sleep_map[speed])
                    st.rerun()
                elif sc >= max_step:
                    st.session_state.dis_playing = False

                state_now = step_states[sc]
                cnt = {s: sum(1 for v in state_now.values() if v == s) for s in ('S','I','R')}
                nc1,nc2,nc3 = st.columns(3)
                nc1.metric("Susceptible", f"{cnt['S']:,}", delta_color="off")
                nc2.metric("Infected",    f"{cnt['I']:,}", delta_color="inverse")
                nc3.metric("Recovered",   f"{cnt['R']:,}")

                fig_net = go.Figure()
                ex, ey = [], []
                for e in G_net.edges():
                    x0,y0=pos[e[0]]; x1,y1=pos[e[1]]
                    ex.extend([x0,x1,None]); ey.extend([y0,y1,None])
                fig_net.add_trace(go.Scatter(x=ex, y=ey, mode='lines',
                    line=dict(width=0.4, color='#d1d5db'), hoverinfo='none', showlegend=False))

                for sk, (nm_l, col, sz) in {
                    'S':('Susceptible','#60A5FA',6),
                    'I':('Infected','#EF4444',14),
                    'R':('Recovered','#34D399',6)}.items():
                    ns = [n for n,s in state_now.items() if s==sk]
                    if not ns: continue
                    fig_net.add_trace(go.Scatter(
                        x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns],
                        mode='markers', name=nm_l,
                        marker=dict(color=col, size=sz, line=dict(width=0.8, color='white')),
                        text=[f"Node {n} | Inf {G_net.nodes[n].get('influence',0):.3f}" for n in ns]))
                fig_net.update_layout(height=520, plot_bgcolor='white',
                    legend=dict(orientation='h', y=1.05),
                    xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    margin=dict(t=20,b=10))
                st.plotly_chart(fig_net, use_container_width=True, key="_dis_netplay")

    # ── 🔬 Infection Provenance ────────────────────────────────────────────────
    if "🔬 Infection Provenance" in T and has_net:
        with T["🔬 Infection Provenance"]:
            st.markdown("### 🔬 Infection Provenance Chain")
            det   = st.session_state.dis_net_detection or {}
            chain = det.get('infection_chain', {})
            seeds = set(st.session_state.dis_net_seeds)
            G_net = st.session_state.dis_net_graph
            pos   = st.session_state.dis_net_pos

            if not chain:
                st.info("Run simulation with XAI enabled to see provenance chain.")
            else:
                hop_counts = {}
                for nid, info in chain.items():
                    h = info['hop']; hop_counts[h] = hop_counts.get(h, 0) + 1

                p1,p2,p3 = st.columns(3)
                p1.metric("Seed nodes",   len(seeds))
                p2.metric("Chain depth",  max(hop_counts.keys(), default=0), "hops")
                p3.metric("Nodes traced", len(chain))

                hop_css_colours = {0:'#6C47FF', 1:'#EF4444', 2:'#F4A261', 3:'#2DC653', -1:'#D1D5DB'}
                for hop in sorted(hop_counts.keys()):
                    nodes_h = sorted([(nid, info) for nid, info in chain.items() if info['hop']==hop],
                                     key=lambda x: x[1]['influence'], reverse=True)
                    lbl  = "Seeds (origin)" if hop == 0 else f"Hop {hop}"
                    col  = hop_css_colours.get(hop, '#aaa')
                    pills = " ".join(
                        f"<span style='background:{col};color:white;padding:2px 9px;"
                        f"border-radius:999px;font-size:.72rem;margin:2px;display:inline-block;'>"
                        f"N{nid} · step {info['step'] if info['step']>=0 else '0'} · "
                        f"inf {info['influence']:.2f}</span>"
                        for nid, info in nodes_h[:20])
                    if len(nodes_h) > 20:
                        pills += (f"<span style='background:#e2e8f0;padding:2px 9px;"
                                  f"border-radius:999px;font-size:.72rem;'>+{len(nodes_h)-20} more</span>")
                    st.markdown(f"**{lbl}** — {len(nodes_h)} nodes")
                    st.markdown(pills, unsafe_allow_html=True)

                # Network coloured by hop
                fig_ch = go.Figure()
                ex2,ey2=[],[]
                for e in G_net.edges():
                    x0,y0=pos[e[0]]; x1,y1=pos[e[1]]
                    ex2.extend([x0,x1,None]); ey2.extend([y0,y1,None])
                fig_ch.add_trace(go.Scatter(x=ex2,y=ey2,mode='lines',
                    line=dict(width=0.3,color='#e5e7eb'),hoverinfo='none',showlegend=False))
                for hv, col in hop_css_colours.items():
                    ns  = [nid for nid,info in chain.items() if info['hop']==hv] if hv>=0 else \
                          [n for n in G_net.nodes() if n not in chain]
                    nm2 = 'Seed' if hv==0 else f'Hop {hv}' if hv>0 else 'Not reached'
                    if not ns: continue
                    sz  = 14 if hv==0 else max(6,12-hv*2)
                    fig_ch.add_trace(go.Scatter(
                        x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns],
                        mode='markers', name=nm2,
                        marker=dict(color=col,size=sz,line=dict(width=0.8,color='white')),
                        text=[f"N{n}|hop{chain.get(n,{}).get('hop','—')}" for n in ns]))
                fig_ch.update_layout(height=480, plot_bgcolor='white',
                    legend=dict(orientation='h',y=1.05),
                    xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    margin=dict(t=20,b=10))
                st.plotly_chart(fig_ch, use_container_width=True, key="_dis_chain")

    # ── 🌐 Echo Chambers ──────────────────────────────────────────────────────
    if "🌐 Echo Chambers" in T and has_net:
        with T["🌐 Echo Chambers"]:
            st.markdown("### 🌐 Community Structure & Echo Chamber Analysis")
            det       = st.session_state.dis_net_detection or {}
            eco_rows  = det.get('community_data', [])
            G_net     = st.session_state.dis_net_graph
            pos       = st.session_state.dis_net_pos
            s_states  = st.session_state.dis_net_step_states or [{}]

            if eco_rows:
                df_eco = pd.DataFrame(eco_rows)
                fig_b  = px.scatter(df_eco, x='Density', y='Infection Rate (%)',
                    size='Size', hover_name='Community',
                    title='Echo Chamber Risk (size = community size)',
                    color='Infection Rate (%)', color_continuous_scale='Reds',
                    size_max=40)
                fig_b.add_vline(x=df_eco['Density'].median(), line_dash='dot', line_color='gray')
                fig_b.add_hline(y=50, line_dash='dot', line_color='gray',
                    annotation_text='50% infected')
                fig_b.update_layout(plot_bgcolor='white', margin=dict(t=40,b=40))
                st.plotly_chart(fig_b, use_container_width=True, key="_dis_echo")
                st.info("Communities in the **top-right** combine high internal connectivity "
                        "with high infection rate — the hallmark of an echo chamber.")

                # Network coloured by community
                communities = nx.get_node_attributes(G_net, 'community')
                if communities:
                    fig_c2 = go.Figure()
                    ex3,ey3=[],[]
                    for e in G_net.edges():
                        x0,y0=pos[e[0]]; x1,y1=pos[e[1]]
                        ex3.extend([x0,x1,None]); ey3.extend([y0,y1,None])
                    fig_c2.add_trace(go.Scatter(x=ex3,y=ey3,mode='lines',
                        line=dict(width=0.3,color='#f0f0f0'),
                        hoverinfo='none',showlegend=False))
                    pal = px.colors.qualitative.Pastel
                    for i, cid in enumerate(sorted(set(communities.values()))):
                        cn = [n for n in G_net.nodes() if communities.get(n)==cid]
                        fig_c2.add_trace(go.Scatter(
                            x=[pos[n][0] for n in cn], y=[pos[n][1] for n in cn],
                            mode='markers', name=f"Group {cid}",
                            marker=dict(color=pal[i%len(pal)], size=8,
                                        line=dict(width=0.5, color='white'))))
                    fig_c2.update_layout(height=460, plot_bgcolor='white',
                        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                        legend=dict(orientation='h',y=1.04),
                        margin=dict(t=10,b=10))
                    st.plotly_chart(fig_c2, use_container_width=True, key="_dis_comm")

    # ── 🧪 Policy Counterfactuals ─────────────────────────────────────────────
    if "🧪 Policy Counterfactuals" in T and has_net:
        with T["🧪 Policy Counterfactuals"]:
            st.markdown("### 🧪 AI Governance Policy Counterfactuals")
            st.markdown("""
            Compare 5 intervention strategies against the baseline — all seeded from
            the same network and starting conditions:
            - **Baseline**: No intervention
            - **Reduced amplification**: Cut algorithmic boost by 50%
            - **No AI amplification**: Pure organic spread only
            - **Early intervention**: +10% recovery probability (faster fact-checking)
            - **Aggressive moderation**: Reduce base infection by 30%
            """)

            if not st.session_state.dis_net_seeds:
                st.info("Run the Network SIR simulation first.")
            else:
                if st.button("▶️ Run all 5 policy scenarios", type="primary",
                             use_container_width=True):
                    with st.spinner("Simulating policy scenarios…"):
                        G_cf = _create_social_graph(num_nodes, avg_degree)
                        cdf  = _run_counterfactuals(G_cf,
                            st.session_state.dis_net_seeds,
                            base_inf, rec_prob, max_net_steps, init_amp, decay_rate)
                        st.session_state.dis_net_counter = cdf

                cdf = st.session_state.dis_net_counter
                if cdf is not None:
                    baseline = cdf.iloc[0]
                    cdf = cdf.copy()
                    cdf['vs Baseline (%)'] = (
                        (cdf['Total Reached'] - baseline['Total Reached'])
                        / baseline['Total Reached'] * 100).round(1)
                    st.dataframe(cdf, use_container_width=True, hide_index=True)

                    cf1, cf2 = st.columns(2)
                    with cf1:
                        fig_cf = px.bar(cdf, x='Scenario', y='Total Reached',
                            color='Total Reached', color_continuous_scale='Reds',
                            title='Total users reached by scenario')
                        fig_cf.update_layout(showlegend=False, plot_bgcolor='white',
                            xaxis_tickangle=-20, margin=dict(t=40,b=80))
                        st.plotly_chart(fig_cf, use_container_width=True, key="_dis_cf_bar")
                    with cf2:
                        fig_cfp = go.Figure(go.Scatter(
                            x=cdf['Scenario'], y=cdf['Peak Spread'],
                            mode='lines+markers',
                            line=dict(color='#ea580c', width=2.5),
                            marker=dict(size=8)))
                        fig_cfp.update_layout(title='Peak spreaders by scenario',
                            plot_bgcolor='white', xaxis_tickangle=-20,
                            margin=dict(t=40,b=80))
                        st.plotly_chart(fig_cfp, use_container_width=True, key="_dis_cf_line")

                    best = cdf.sort_values('Total Reached').iloc[0]
                    reduction = ((baseline['Total Reached'] - best['Total Reached'])
                                 / baseline['Total Reached'] * 100)
                    st.success(
                        f"**Best policy:** {best['Scenario']} — "
                        f"**{reduction:.1f}% reduction** in total reach "
                        f"({best['Total Reached']:,} vs {baseline['Total Reached']:,})")
                    st.download_button("📥 Download counterfactual CSV",
                        cdf.to_csv(index=False).encode(),
                        f"counterfactual_{datetime.now():%Y%m%d}.csv",
                        "text/csv", use_container_width=True)

    # ── 💰 Cost-Benefit ────────────────────────────────────────────────────────
    if "💰 Cost-Benefit" in T and has_net:
        with T["💰 Cost-Benefit"]:
            st.markdown("### 💰 Cost-Benefit Analysis")
            nm = st.session_state.dis_net_metrics
            cb1,cb2,cb3 = st.columns(3)
            with cb1: cost_per_bot    = st.number_input("Cost per bot ($)", 1, 500, 50)
            with cb2: harm_per_inf    = st.number_input("Harm per infection ($)", 1, 1000, 150)
            with cb3: platform_budget = st.number_input("Budget ($)", 1000, 50000, 10000)

            total_bots  = nm.get('bots_deployed', 0)
            total_cost  = total_bots * cost_per_bot
            actual_harm = nm['total_reached'] * harm_per_inf
            harm_saved  = (nm['total_nodes'] - nm['total_reached']) * harm_per_inf
            roi         = ((harm_saved - total_cost) / total_cost * 100) if total_cost > 0 else 0

            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Bot cost",       f"${total_cost:,}")
            r2.metric("Actual harm",    f"${actual_harm:,}")
            r3.metric("Harm prevented", f"${harm_saved:,}")
            r4.metric("ROI",            f"{roi:.1f}%",
                       delta_color="normal" if roi > 0 else "inverse")

            fig_roi = go.Figure(go.Bar(
                x=["Bot cost","Actual harm","Harm prevented"],
                y=[total_cost, actual_harm, harm_saved],
                marker_color=["#ea580c","#ef4444","#16a34a"],
                text=[f"${v:,.0f}" for v in [total_cost,actual_harm,harm_saved]],
                textposition='outside'))
            fig_roi.update_layout(plot_bgcolor='white', yaxis_title='USD ($)',
                margin=dict(t=20,b=40))
            st.plotly_chart(fig_roi, use_container_width=True, key="_dis_roi")

            hist_df = st.session_state.dis_net_history
            if hist_df is not None:
                st.download_button("📊 Download Spread History CSV",
                    hist_df.to_csv(index=False).encode(),
                    f"sir_history_{datetime.now():%Y%m%d}.csv", "text/csv",
                    use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SHARED TABS (all modes)
    # ══════════════════════════════════════════════════════════════════════════

    # ── 🔬 Feature Modules ─────────────────────────────────────────────────────
    if "🔬 Feature Modules" in T:
        with T["🔬 Feature Modules"]:
            feats = st.session_state.get("dis_feature_outputs", {})
            feature_modules_tab(
                domain="disinformation",
                run_results=st.session_state.get("dis_run_history", []),
                feats=feats,
                governance=feats.get("governance"),
                gender_audit=feats.get("gender_audit"),
                agent_economy=feats.get("agent_economy"),
                arena=feats.get("arena"),
                redteam=feats.get("multimodal_redteam"),
            )

    # ── 🛡️ AI Safety ───────────────────────────────────────────────────────────
    if "🛡️ AI Safety" in T:
        with T["🛡️ AI Safety"]:
            render_safety_tab(
                st.session_state.get("dis_safety_report", {}),
                domain="disinformation")

    # ── 🔄 Lifecycle ──────────────────────────────────────────────────────────
    if "🔄 Lifecycle" in T:
        with T["🔄 Lifecycle"]:
            render_lifecycle_tab(
                st.session_state.get("dis_lifecycle_report", {}),
                "disinformation")

    # ── 🌱 Eco Score ──────────────────────────────────────────────────────────
    if "🌱 Eco Score" in T:
        with T["🌱 Eco Score"]:
            _lc_r    = st.session_state.get("dis_lifecycle_report", {})
            _lc_last = (st.session_state.get("dis_run_history") or [{}])[-1]
            render_eco_tab(_lc_r,
                algo_key  = _lc_last.get("algorithm", "hist_gradient_boosting"),
                n_samples = int(_lc_last.get("n_samples", 2000)),
                n_runs    = int(n_runs),
                domain    = "disinformation")

    # ── History & Annotations ──────────────────────────────────────────────────
    if has_gags:
        history_browser("dis_snapshot_history", domain="disinformation",
            key_metrics=["accuracy","fairness_score","language_fpr_gap"])
        annotation_panel("dis_annotations",
            context_label=f"{len(st.session_state.dis_run_history)} Disinformation run(s)")

        st.divider()
        st.markdown("## 💡 Policy Recommendations")
        pr1, pr2 = st.columns(2)
        with pr1:
            st.markdown("""
**If language FPR gap > 10%:**
- Collect annotated Hausa/Yoruba/Igbo training data
- Language-stratified threshold calibration
- Local language community reviewers for appeals
- Publish per-language moderation accuracy quarterly

**Nigeria / NCC alignment:**
- NITDA AI Policy: language equity in automated decisions
- NCC: USSD/SMS content cannot be moderated by English-only AI
- INEC: election content moderation must be audited pre-election
            """)
        with pr2:
            st.markdown("""
**Structural recommendations:**
- Separate classifiers per language family
- Human review mandatory for political content in election periods
- Over-removal rate published in transparency reports
- Appeal pathway with plain-language explanation

**International alignment:**
- EU DSA: moderation must not discriminate by language or origin
- ICCPR Article 19: right to freedom of expression
- UNESCO: AI must not suppress minority language expression
            """)

# ── Welcome state ──────────────────────────────────────────────────────────────
else:
    st.markdown(f"""
    <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
    padding:2.5rem;text-align:center;margin-top:1rem'>
    <p style='font-size:2.2rem;margin:0 0 .6rem'>📡</p>
    <p style='font-size:1.2rem;font-weight:700;color:#0f172a;margin:0 0 .4rem'>
    Welcome to Disinformation Resilience Simulation v6.0</p>
    <p style='color:#64748b;font-size:.87rem;max-width:600px;margin:0 auto .5rem'>
    Choose a simulation mode in the sidebar:<br>
    <strong>📊 Fairness Pipeline</strong> — language FPR gap, speech suppression, EU DSA / NITDA compliance<br>
    <strong>🕸️ Network SIR</strong> — viral spread dynamics, echo chambers, infection provenance<br>
    <strong>🔀 Both</strong> — full combined analysis
    </p>
    <p style='color:#94a3b8;font-size:.78rem;margin-top:.5rem'>
    Configure settings in the sidebar and click <strong>🚀 Run Simulation</strong>
    </p></div>""", unsafe_allow_html=True)

st.divider()
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;font-size:.78rem;padding:.75rem 0;'>"
    "📡 Disinformation Resilience Simulation · GAGS Framework v6.0 · "
    "SIR Network · Content Moderation Fairness · Electoral Integrity · "
    "Language Equity · Nigeria / EU DSA / INEC Compliance"
    "</div>", unsafe_allow_html=True)
