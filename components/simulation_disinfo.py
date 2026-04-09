# utils/simulation_disinfo_advanced.py
import networkx as nx
import numpy as np
import shap
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from community import community_louvain
import os
from typing import Tuple, Dict, Optional, List


# ─────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────

def create_social_graph(num_nodes: int = 200, avg_degree: float = 5.0) -> nx.Graph:
    """Create a scale-free social network with community structure."""
    G = nx.barabasi_albert_graph(num_nodes, max(1, int(avg_degree / 2)))
    communities = community_louvain.best_partition(G)
    nx.set_node_attributes(G, communities, 'community')
    return G


def assign_node_attributes(G: nx.Graph) -> nx.Graph:
    """Assign influence (PageRank) and susceptibility to each node."""
    pr = nx.pagerank(G)
    max_pr = max(pr.values()) if pr else 1.0
    influence = {n: pr[n] / max_pr for n in G.nodes()}
    nx.set_node_attributes(G, influence, 'influence')
    susceptibility = {n: np.random.uniform(0.5, 1.5) for n in G.nodes()}
    nx.set_node_attributes(G, susceptibility, 'susceptibility')
    return G


# ─────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────

def load_isot_dataset(fake_path: str = 'data/Fake.csv',
                      true_path: str = 'data/True.csv') -> Optional[pd.DataFrame]:
    """Load and merge the ISOT fake/real news dataset."""
    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        return None
    fake_df = pd.read_csv(fake_path);  fake_df['label'] = 1
    true_df = pd.read_csv(true_path);  true_df['label'] = 0
    df = pd.concat([fake_df, true_df], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def train_text_classifier(df: pd.DataFrame) -> Tuple:
    """Train a TF-IDF + GBM classifier for fake-news content detection."""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return vectorizer, clf, accuracy


def assign_articles_to_nodes(G: nx.Graph, dataset_df: pd.DataFrame) -> nx.Graph:
    """Assign a random article to each node from the ISOT dataset."""
    n_nodes = G.number_of_nodes()
    sampled = dataset_df.sample(n=n_nodes, replace=True, random_state=42)
    articles = list(zip(sampled['text'], sampled['label']))
    attrs = {node: {'article_text': text, 'article_label': label}
             for node, (text, label) in zip(G.nodes(), articles)}
    nx.set_node_attributes(G, attrs)
    return G


# ─────────────────────────────────────────────────────────────
# Echo chamber rewiring
# ─────────────────────────────────────────────────────────────

def rewire_for_echo_chambers(G: nx.Graph, state: dict,
                              rewiring_prob: float = 0.05) -> nx.Graph:
    """
    Simulates polarisation: breaks I↔R edges and rewires within same
    state/community, modelling users unfollowing fact-checkers.
    """
    nodes = list(G.nodes())
    edges_to_remove, edges_to_add = [], []

    for u, v in G.edges():
        if ((state[u] == 'I' and state[v] == 'R') or
                (state[u] == 'R' and state[v] == 'I')):
            if np.random.random() < rewiring_prob:
                edges_to_remove.append((u, v))
                u_comm = G.nodes[u]['community']
                potential_peers = [
                    n for n in nodes
                    if G.nodes[n]['community'] == u_comm
                    and state[n] == state[u]
                    and n != u and not G.has_edge(u, n)
                ]
                if potential_peers:
                    edges_to_add.append((u, np.random.choice(potential_peers)))

    G.remove_edges_from(edges_to_remove)
    G.update(edges=edges_to_add)
    return G


# ─────────────────────────────────────────────────────────────
# XAI helpers
# ─────────────────────────────────────────────────────────────

def build_node_feature_matrix(G: nx.Graph, state: dict,
                               infection_step: dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build (X, y, feature_names) for node-level spread prediction.
    Features: degree, clustering, community, influence, susceptibility,
              infection_step (–1 = never infected).
    """
    feature_names = [
        'Degree', 'Clustering coefficient', 'Community ID',
        'Influence (PageRank)', 'Susceptibility', 'Infection step'
    ]
    X_rows, y_rows = [], []
    for node in G.nodes():
        feat = [
            G.degree(node),
            nx.clustering(G, node),
            G.nodes[node].get('community', 0),
            G.nodes[node].get('influence', 0.0),
            G.nodes[node].get('susceptibility', 1.0),
            infection_step.get(node, -1),
        ]
        X_rows.append(feat)
        y_rows.append(1 if state[node] == 'I' else 0)
    return np.array(X_rows), np.array(y_rows), feature_names


def compute_shap_values(clf: GradientBoostingClassifier,
                        X_scaled: np.ndarray,
                        feature_names: List[str],
                        max_samples: int = 100) -> Dict:
    """
    Compute SHAP values for the node-feature detector.
    Returns mean |SHAP|, a sample of per-node values, and feature names.
    """
    sample_idx = np.random.choice(len(X_scaled),
                                  size=min(max_samples, len(X_scaled)),
                                  replace=False)
    X_sample = X_scaled[sample_idx]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    # GBM TreeExplainer returns (n_samples, n_features) for binary
    if isinstance(shap_values, list):
        sv = shap_values[1]   # class-1 shap values
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0).tolist()
    # Waterfall data: top node (index 0 of sample)
    waterfall_values = sv[0].tolist()
    base_value = float(explainer.expected_value[1]
                       if isinstance(explainer.expected_value, (list, np.ndarray))
                       else explainer.expected_value)

    return {
        'mean_abs_shap': mean_abs_shap,
        'waterfall_values': waterfall_values,
        'base_value': base_value,
        'feature_names': feature_names,
        'shap_matrix': sv.tolist(),          # (sample, features) for beeswarm
        'X_sample': X_sample.tolist(),
    }


def compute_permutation_importance(clf: GradientBoostingClassifier,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   feature_names: List[str],
                                   n_repeats: int = 10) -> Dict:
    """
    Permutation feature importance — model-agnostic baseline comparison.
    """
    result = permutation_importance(
        clf, X_test, y_test,
        n_repeats=n_repeats, random_state=42, scoring='roc_auc'
    )
    return {
        'importances_mean': result.importances_mean.tolist(),
        'importances_std': result.importances_std.tolist(),
        'feature_names': feature_names,
    }


def explain_single_node(node_id: int, G: nx.Graph, state: dict,
                        infection_step: dict,
                        clf: GradientBoostingClassifier,
                        scaler: StandardScaler,
                        feature_names: List[str]) -> Dict:
    """
    Per-node counterfactual: what would need to change for a susceptible
    node to become infected (or vice-versa)?  Returns feature values,
    model probability, and simple natural-language explanation.
    """
    feat = np.array([[
        G.degree(node_id),
        nx.clustering(G, node_id),
        G.nodes[node_id].get('community', 0),
        G.nodes[node_id].get('influence', 0.0),
        G.nodes[node_id].get('susceptibility', 1.0),
        infection_step.get(node_id, -1),
    ]])
    feat_scaled = scaler.transform(feat)
    prob = clf.predict_proba(feat_scaled)[0][1]

    # Natural-language explanation
    degree   = G.degree(node_id)
    inf      = G.nodes[node_id].get('influence', 0.0)
    susc     = G.nodes[node_id].get('susceptibility', 1.0)
    status   = state.get(node_id, 'S')

    if status == 'I':
        reason = (f"Node {node_id} is infected. It has {degree} connections "
                  f"and an influence score of {inf:.3f}. "
                  f"Its susceptibility ({susc:.2f}) is "
                  + ("high" if susc > 1.0 else "moderate") +
                  ", making it an active spreader.")
    elif status == 'R':
        reason = (f"Node {node_id} has recovered (fact-checked). "
                  f"With influence {inf:.3f} it may still shape the network "
                  f"by discouraging neighbours from spreading.")
    else:
        reason = (f"Node {node_id} is susceptible. It has {degree} connections "
                  f"and influence {inf:.3f}. "
                  f"Susceptibility {susc:.2f} means it "
                  + ("is at high risk of infection." if susc > 1.1
                     else "has some resistance."))

    return {
        'node_id': node_id,
        'status': status,
        'probability': float(prob),
        'features': {fn: float(feat[0][i]) for i, fn in enumerate(feature_names)},
        'explanation': reason,
    }


def trace_infection_chain(seed_nodes: list, state: dict,
                          infection_step: dict, G: nx.Graph,
                          max_hops: int = 4) -> Dict:
    """
    Trace the most influential infection chains from seeds outward.
    Returns a dict of {node: {'step': int, 'hop': int, 'influence': float}}
    for building a provenance tree.
    """
    chain = {}
    for seed in seed_nodes:
        chain[seed] = {'step': 0, 'hop': 0,
                       'influence': G.nodes[seed].get('influence', 0.0),
                       'parent': None}

    for hop in range(1, max_hops + 1):
        prev_hop_nodes = [n for n, v in chain.items() if v['hop'] == hop - 1]
        for node in prev_hop_nodes:
            for nb in G.neighbors(node):
                if nb not in chain and state.get(nb) in ('I', 'R'):
                    chain[nb] = {
                        'step': infection_step.get(nb, -1),
                        'hop': hop,
                        'influence': G.nodes[nb].get('influence', 0.0),
                        'parent': node,
                    }
    return chain


# ─────────────────────────────────────────────────────────────
# Counterfactual policy scenarios
# ─────────────────────────────────────────────────────────────

def run_counterfactual_simulations(
        G: nx.Graph, seeds: list,
        base_infection: float, recovery_prob: float,
        max_steps: int, initial_amp: float, decay_rate: float,
        dataset_df: Optional[pd.DataFrame] = None,
        content_detector=None,
        fake_news_boost: float = 1.0) -> pd.DataFrame:
    """
    Run five policy scenarios and return a comparison DataFrame.
    Uses a deep-copy of G for each run to avoid cross-contamination.
    """
    import copy
    results = []

    scenarios = [
        ("Baseline",                     base_infection,        recovery_prob,        initial_amp,        decay_rate),
        ("Reduced amplification (50%)",  base_infection,        recovery_prob,        max(1.0, initial_amp * 0.5), decay_rate),
        ("No AI amplification",          base_infection,        recovery_prob,        1.0,                0.0),
        ("Early intervention (+10% rec)", base_infection,       min(0.3, recovery_prob + 0.1), initial_amp, decay_rate),
        ("Aggressive moderation (−30%)", base_infection * 0.7,  recovery_prob,        initial_amp,        decay_rate),
    ]

    for label, b_inf, rec, amp, dec in scenarios:
        G_copy = copy.deepcopy(G)
        hist, metrics, _, _ = simulate_misinfo_propagation_advanced(
            G_copy, seeds, b_inf, rec, max_steps,
            initial_amplification=amp, decay_rate=dec,
            dataset_df=dataset_df, content_detector=content_detector,
            fake_news_boost=fake_news_boost, collect_detection_data=False)
        results.append({
            "Scenario": label,
            "Total Reached": metrics['total_reached'],
            "Peak Spread": metrics['max_infected'],
            "Steps to Peak": metrics['steps_to_peak'],
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────

def simulate_misinfo_propagation_advanced(
        G: nx.Graph,
        seed_nodes: list,
        base_infection_prob: float = 0.15,
        recovery_prob: float = 0.02,
        max_steps: int = 50,
        initial_amplification: float = 2.0,
        decay_rate: float = 0.1,
        dataset_df: Optional[pd.DataFrame] = None,
        content_detector=None,
        fake_news_boost: float = 1.5,
        collect_detection_data: bool = False,
        enable_bots: bool = False,
        bot_coverage: float = 5.0,
        intervention_step: int = 10,
        bot_strength: float = 2.0,
        recovery_boost_step: Optional[int] = None,
        recovery_boost_multiplier: float = 1.0,
) -> Tuple[pd.DataFrame, Dict, Dict, List]:

    # Ensure node attributes exist
    if not nx.get_node_attributes(G, 'influence'):
        G = assign_node_attributes(G)
    if dataset_df is not None:
        G = assign_articles_to_nodes(G, dataset_df)

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

        # Bot deployment
        if enable_bots and step == intervention_step:
            num_bots = int((bot_coverage / 100) * len(G.nodes()))
            targets = sorted([n for n, s in state.items() if s == 'S'],
                             key=lambda n: G.nodes[n].get('influence', 0), reverse=True)
            for t in targets[:num_bots]:
                state[t] = 'R'

        eff_amp = initial_amplification * np.exp(-decay_rate * step)
        new_infected, new_recovered = [], []

        for node in infected:
            node_boost = (fake_news_boost
                          if dataset_df is not None
                          and G.nodes[node].get('article_label', 0) == 1
                          else 1.0)
            for nb in G.neighbors(node):
                if state[nb] == 'S':
                    prob = (base_infection_prob
                            * G.nodes[node].get('influence', 1.0)
                            * G.nodes[nb].get('susceptibility', 1.0)
                            * eff_amp * node_boost)
                    if np.random.random() < min(prob, 1.0):
                        new_infected.append(nb)
                        infection_step_map[nb] = step + 1

            cur_rec = recovery_prob
            if enable_bots and step >= intervention_step:
                cur_rec *= bot_strength
            if np.random.random() < cur_rec:
                new_recovered.append(node)

        for n in new_infected:   state[n] = 'I'
        for n in new_recovered:  state[n] = 'R'

        if step > 0 and step % 5 == 0:
            G = rewire_for_echo_chambers(G, state, rewiring_prob=0.1)

    df_history = pd.DataFrame(history)
    final_metrics = {
        'total_reached':   len([n for n, s in state.items() if s != 'S']),
        'max_infected':    df_history['infected'].max(),
        'final_infected':  len([n for n, s in state.items() if s == 'I']),
        'final_recovered': len([n for n, s in state.items() if s == 'R']),
        'steps_to_peak':   int(df_history.loc[df_history['infected'].idxmax(), 'step']),
        'bots_deployed':   int((bot_coverage / 100) * len(G.nodes())) if enable_bots else 0,
        'total_nodes':     len(G.nodes()),
        'intervention_active': enable_bots,
    }

    # ── Detection + XAI ────────────────────────────────────────
    detection_metrics = {}
    if collect_detection_data and len(G.nodes()) > 20:
        X_nodes, y_nodes, feat_names = build_node_feature_matrix(
            G, state, infection_step_map)

        if len(np.unique(y_nodes)) == 2 and np.sum(y_nodes) > 5:
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_nodes)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_s, y_nodes, test_size=0.3, stratify=y_nodes, random_state=42)

            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=42)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]

            detection_metrics['node_detection'] = {
                'accuracy':          float(accuracy_score(y_te, y_pred)),
                'precision':         float(precision_score(y_te, y_pred, zero_division=0)),
                'recall':            float(recall_score(y_te, y_pred, zero_division=0)),
                'f1':                float(f1_score(y_te, y_pred, zero_division=0)),
                'roc_auc':           float(roc_auc_score(y_te, y_prob)),
                'confusion_matrix':  confusion_matrix(y_te, y_pred).tolist(),
                'feature_importance': clf.feature_importances_.tolist(),
                'feature_names':     feat_names,
            }

            # SHAP values
            try:
                shap_data = compute_shap_values(clf, X_s, feat_names)
                detection_metrics['shap'] = shap_data
            except Exception:
                pass

            # Permutation importance
            try:
                perm_data = compute_permutation_importance(
                    clf, X_te, y_te, feat_names)
                detection_metrics['permutation_importance'] = perm_data
            except Exception:
                pass

            # Per-node explanation for the top-5 highest-influence infected nodes
            top_infected = sorted(
                [n for n, s in state.items() if s == 'I'],
                key=lambda n: G.nodes[n].get('influence', 0), reverse=True)[:5]
            node_explanations = []
            for nid in top_infected:
                try:
                    exp = explain_single_node(
                        nid, G, state, infection_step_map, clf, scaler, feat_names)
                    node_explanations.append(exp)
                except Exception:
                    pass
            detection_metrics['node_explanations'] = node_explanations

            # Infection chain provenance
            try:
                chain = trace_infection_chain(
                    seed_nodes, state, infection_step_map, G)
                detection_metrics['infection_chain'] = chain
            except Exception:
                pass

    return df_history, final_metrics, detection_metrics, step_states