# components/ux_utils.py
"""
GAGS UX Utilities — v1.0
========================

Shared UI components injected into all three simulation pages:

  1. metric_glossary_expander()  — inline "explain this metric" expandable panel
                                   with plain-language definition + visual example
  2. preset_selector()           — quick-start persona templates that pre-fill
                                   the sidebar in one click
  3. history_browser()           — simulation history timeline with
                                   compare/restore/annotate capability
  4. guided_tour_banner()        — first-run onboarding banner with step-by-step
                                   walkthrough of the interface
  5. _inject_tour_css()          — shared CSS for all four components

All functions are pure Streamlit — no external JS or additional dependencies.
Call each function once at the appropriate point in a page module.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# §1  METRIC GLOSSARY
# ──────────────────────────────────────────────────────────────────────────────

# Plain-language definitions + ASCII visual examples for every metric in GAGS
_GLOSSARY: Dict[str, Dict[str, str]] = {
    # ── Fairness metrics ──────────────────────────────────────────────────────
    "demographic parity": {
        "plain":   "Are predictions equally positive across groups? "
                   "If the model approves 80% of Group A but only 50% of Group B, "
                   "there is a 30% demographic parity gap — even if both groups have the same true risk.",
        "formula": "gap = |positive_rate_A − positive_rate_B|",
        "example": "Group A: ██████████ 80% approved\n"
                   "Group B: █████      50% approved\n"
                   "Gap:     ████       30%  ← want this near 0%",
        "target":  "Gap ≤ 10% is generally considered acceptable.",
        "risk":    "High gap → certain groups are systematically treated differently.",
    },
    "equalized odds": {
        "plain":   "Do all groups experience the same error rates? "
                   "Equalized odds requires both the false positive rate (FPR) and "
                   "false negative rate (FNR) to be equal across groups.",
        "formula": "gap = max(|FPR_A − FPR_B|, |FNR_A − FNR_B|)",
        "example": "Group A FPR: 5%   FNR: 10%\n"
                   "Group B FPR: 20%  FNR: 35%\n"
                   "Gap:         15%  ← largest difference across both rates",
        "target":  "Gap ≤ 10% across both FPR and FNR.",
        "risk":    "High gap → certain groups are more likely to be wrongly flagged or missed.",
    },
    "fairness score": {
        "plain":   "A combined 0–1 score summarising overall algorithmic fairness. "
                   "It is calculated from demographic parity and equalized odds together. "
                   "1.0 means perfect fairness across all groups; 0.0 means maximum disparity.",
        "formula": "fairness_score = 1 − (parity_gap + equalized_odds_gap) / 2",
        "example": "Score 0.9 → excellent fairness\n"
                   "Score 0.7 → acceptable (recommended minimum)\n"
                   "Score 0.5 → significant disparities detected\n"
                   "Score 0.3 → serious equity failure — do not deploy",
        "target":  "≥ 0.70 for deployment readiness.",
        "risk":    "Below 0.70 means the model is likely producing inequitable outcomes.",
    },
    "false positive rate": {
        "plain":   "Of all the people who do NOT have the condition, "
                   "what fraction does the model incorrectly flag as positive? "
                   "In healthcare: the rate of healthy patients wrongly diagnosed. "
                   "In security: the rate of innocent people wrongly flagged as threats.",
        "formula": "FPR = false_positives / (false_positives + true_negatives)",
        "example": "100 healthy patients → model flags 15 as sick\n"
                   "FPR = 15 / 100 = 15%  ← 15 unnecessary alarms",
        "target":  "Below 10% in most domains; below 5% in high-stakes contexts.",
        "risk":    "High FPR → unnecessary interventions, eroded trust, wasted resources.",
    },
    "false negative rate": {
        "plain":   "Of all the people who DO have the condition, "
                   "what fraction does the model miss? "
                   "In healthcare: patients who needed care but were not identified.",
        "formula": "FNR = false_negatives / (false_negatives + true_positives)",
        "example": "100 sick patients → model misses 20\n"
                   "FNR = 20 / 100 = 20%  ← 20 patients without care",
        "target":  "Below 15% in most domains; below 5% in life-critical contexts.",
        "risk":    "High FNR → missed diagnoses, undetected threats, systemic harm.",
    },
    "sensitivity": {
        "plain":   "Also called Recall or True Positive Rate. "
                   "Of all positive cases, what fraction did the model correctly detect? "
                   "High sensitivity means the model rarely misses a real case.",
        "formula": "sensitivity = true_positives / (true_positives + false_negatives)",
        "example": "100 sick patients → model correctly identifies 85\n"
                   "Sensitivity = 85%  ← 15 patients missed",
        "target":  "≥ 75% is a common clinical safety threshold.",
        "risk":    "Low sensitivity → dangerous for disease screening where missing cases is costly.",
    },
    "specificity": {
        "plain":   "Of all negative cases, what fraction did the model correctly leave unflagged? "
                   "High specificity means the model rarely raises false alarms.",
        "formula": "specificity = true_negatives / (true_negatives + false_positives)",
        "example": "100 healthy patients → model correctly clears 90\n"
                   "Specificity = 90%  ← 10 unnecessary alarms",
        "target":  "≥ 80% to avoid overwhelming clinical teams with false alerts.",
        "risk":    "Low specificity → alert fatigue, unnecessary procedures, waste.",
    },
    # ── Attack metrics ────────────────────────────────────────────────────────
    "poison rate": {
        "plain":   "The fraction of training data that has been deliberately corrupted. "
                   "Even 5–10% poisoning can significantly degrade a model's performance "
                   "and fairness properties.",
        "formula": "poison_rate = poisoned_samples / total_samples",
        "example": "1,000 records, 5% poisoning = 50 corrupted records\n"
                   "50 records with flipped labels quietly shift model behaviour",
        "target":  "Any poison rate above 0% is adversarial and should trigger investigation.",
        "risk":    "Poisoned models may produce accurate-looking but systematically biased results.",
    },
    "bias intensity": {
        "plain":   "Controls how strongly the selected bias types are injected into the data. "
                   "0.0 = no bias; 0.5 = maximum. Even 0.2 can produce measurable "
                   "fairness gaps in sensitive domains.",
        "formula": "Scales feature degradation / label flip probability for each bias type",
        "example": "0.0 → fair baseline\n"
                   "0.2 → mild bias (realistic)\n"
                   "0.4 → severe bias (adversarial conditions)",
        "target":  "Use 0.1–0.3 for realistic simulation; higher values test worst-case resilience.",
        "risk":    "High bias intensity + demographic bias type replicates real-world discrimination.",
    },
    # ── Security metrics ──────────────────────────────────────────────────────
    "liberty score": {
        "plain":   "A 0–1 composite score measuring how well civil liberties are preserved "
                   "under the current surveillance configuration. It combines false positive rate, "
                   "surveillance intensity, data retention period, and bias level, "
                   "weighted by the oversight mechanism.",
        "formula": "liberty = 1 − (FPR×0.4 + surveillance×0.3 + retention×0.2 + bias×0.1) × oversight_multiplier",
        "example": "Surveillance 80%, no oversight → liberty ≈ 0.20 (very low)\n"
                   "Surveillance 40%, judicial review → liberty ≈ 0.65 (moderate)\n"
                   "Surveillance 20%, strong oversight → liberty ≈ 0.85 (good)",
        "target":  "≥ 0.60 is a reasonable civil-liberties floor; ≥ 0.75 is preferred.",
        "risk":    "Low liberty score indicates disproportionate surveillance impact on populations.",
    },
    "sociotechnical risk": {
        "plain":   "How likely a given attack modality is to succeed because of human factors "
                   "(trust, cognitive load, stress) rather than purely technical vulnerabilities. "
                   "Deepfakes score 0.90 because humans find them highly convincing "
                   "even when they know deepfakes exist.",
        "formula": "Assigned per modality: TEXT=0.4, IMAGE=0.6, AUDIO=0.5, DEEPFAKE=0.9",
        "example": "A perfect technical defence is irrelevant if a human operator \n"
                   "is fooled by a realistic-looking deepfake briefing under time pressure.",
        "target":  "Any modality with sociotechnical risk > 0.7 requires mandatory human review protocols.",
        "risk":    "High sociotechnical risk means the attack bypasses technical controls via human error.",
    },
    # ── Agrotech / general ────────────────────────────────────────────────────
    "permeability score": {
        "plain":   "In the AI Agent Economy module, this measures how unequally "
                   "resources (irrigation water, fertilizer, drone hours) are distributed "
                   "after autonomous agents bid for them. "
                   "High permeability means large agents dominate small ones.",
        "formula": "permeability = std(total_spent) / mean(total_spent)",
        "example": "Score 0.1 → equitable — all agents spend similarly\n"
                   "Score 0.5 → moderate inequality\n"
                   "Score 1.0+ → one agent dominates; smallholders excluded",
        "target":  "Below 0.5 for equitable resource allocation.",
        "risk":    "High permeability → AI-driven resource allocation may systematically "
                   "disadvantage smallholder farmers relative to commercial operations.",
    },
    "digital inclusion score": {
        "plain":   "Measures whether an AI system is accessible to people with limited "
                   "digital connectivity (no smartphone, low bandwidth, feature phone only). "
                   "In Nigeria, only ~34% of smallholder farmers have reliable smartphone access.",
        "formula": "digital_inclusion = baseline_connectivity + (1 − prediction_gap)",
        "example": "Score 0.8 → system works for most users regardless of connectivity\n"
                   "Score 0.4 → system effectively excludes low-connectivity populations\n"
                   "Score < 0.5 → USSD/SMS fallback interface urgently needed",
        "target":  "≥ 0.60 minimum; ≥ 0.75 recommended for deployment in Nigeria/Africa.",
        "risk":    "Low score means the AI tool only benefits users who are already advantaged.",
    },
    "gender gap": {
        "plain":   "The accuracy difference between female and male demographic groups "
                   "in the UNESCO Women4EthicalAI audit. A gap means the model "
                   "performs systematically worse for one gender.",
        "formula": "gender_gap = |accuracy_female − accuracy_male|",
        "example": "Female group accuracy: 72%\n"
                   "Male group accuracy:   85%\n"
                   "Gender gap:            13%  ← exceeds 5% threshold",
        "target":  "≤ 5% for UNESCO Women4EthicalAI compliance.",
        "risk":    "Any gap above 5% means women (often already disadvantaged) "
                   "receive materially worse AI service.",
    },
}


def metric_glossary_expander(
    metrics_to_show: Optional[List[str]] = None,
    collapsed: bool = True,
    location: str = "main",  # "main" | "sidebar"
) -> None:
    """
    Render an expandable glossary panel explaining every metric shown on the page.
    Call this once near the top of the results dashboard.

    Parameters
    ----------
    metrics_to_show : list of metric names to include (None = show all)
    collapsed       : start collapsed (True) or open (False)
    location        : "main" renders a full panel; "sidebar" renders compact version
    """
    keys = metrics_to_show or list(_GLOSSARY.keys())
    keys = [k.lower() for k in keys]
    available = {k: v for k, v in _GLOSSARY.items() if k in keys}

    if not available:
        return

    if location == "sidebar":
        with st.sidebar.expander("📖 Metric definitions", expanded=False):
            for name, data in available.items():
                st.markdown(f"**{name.title()}**")
                st.caption(data["plain"])
                st.divider()
        return

    with st.expander(
        f"📖 What do these metrics mean? ({len(available)} definitions — click to expand)",
        expanded=not collapsed,
    ):
        st.markdown(
            "<p style='color:var(--color-text-secondary);font-size:.9rem;margin-bottom:1rem;'>"
            "Plain-language explanations of every metric shown in this dashboard. "
            "No prior machine learning knowledge required.</p>",
            unsafe_allow_html=True,
        )

        # Two-column grid of metric cards
        items  = list(available.items())
        n_cols = 2
        for row_start in range(0, len(items), n_cols):
            cols = st.columns(n_cols)
            for col, (name, data) in zip(cols, items[row_start : row_start + n_cols]):
                with col:
                    st.markdown(
                        f"""
                        <div style="background:var(--color-background-secondary);
                                    border:0.5px solid var(--color-border-tertiary);
                                    border-radius:var(--border-radius-lg);
                                    padding:.85rem 1rem;margin-bottom:.75rem;">
                          <p style="font-weight:500;font-size:.9rem;
                                    color:var(--color-text-primary);margin:0 0 .35rem;">
                            {name.title()}
                          </p>
                          <p style="font-size:.82rem;color:var(--color-text-secondary);
                                    margin:0 0 .4rem;line-height:1.5;">
                            {data['plain']}
                          </p>
                          <code style="font-size:.76rem;color:var(--color-text-info);
                                       background:var(--color-background-info);
                                       padding:2px 6px;border-radius:4px;">
                            {data['formula']}
                          </code>
                          <pre style="font-size:.75rem;margin:.5rem 0 .35rem;
                                      background:var(--color-background-tertiary);
                                      padding:.4rem .6rem;border-radius:4px;
                                      white-space:pre-wrap;
                                      color:var(--color-text-primary);">{data['example']}</pre>
                          <p style="font-size:.78rem;color:var(--color-text-success);
                                    margin:0 0 .2rem;">
                            ✅ Target: {data['target']}
                          </p>
                          <p style="font-size:.78rem;color:var(--color-text-danger);margin:0;">
                            ⚠️ Risk: {data['risk']}
                          </p>
                        </div>""",
                        unsafe_allow_html=True,
                    )


# ──────────────────────────────────────────────────────────────────────────────
# §2  QUICK-START PRESETS
# ──────────────────────────────────────────────────────────────────────────────

_PRESETS: Dict[str, Dict[str, Any]] = {

    # ── Healthcare ────────────────────────────────────────────────────────────
    "health": {
        "Clinical Regulator": {
            "icon": "🏛️",
            "desc": "Strict fairness audit. Minimal bias, low attack, maximum runs.",
            "settings": {
                "data_source":        "Synthetic Only",
                "selected_biases":    ["demographic", "socioeconomic"],
                "bias_intensity":     0.1,
                "poison_rate":        0.02,
                "access_inequality":  0.2,
                "n_runs":             5,
                "enable_governance":  True,
                "enable_gender_audit":True,
                "enable_redteam":     False,
            },
        },
        "ML Researcher": {
            "icon": "🔬",
            "desc": "All bias types, high attack strength, full feature modules.",
            "settings": {
                "data_source":        "Synthetic Only",
                "selected_biases":    ["demographic","historical","measurement",
                                       "geographic","gender","linguistic"],
                "bias_intensity":     0.4,
                "poison_rate":        0.15,
                "access_inequality":  0.5,
                "n_runs":             5,
                "enable_governance":  True,
                "enable_gender_audit":True,
                "enable_redteam":     True,
                "enable_arena":       True,
            },
        },
        "Hospital Administrator": {
            "icon": "🏥",
            "desc": "Real UCI data. Focus on income gaps and clinical safety.",
            "settings": {
                "data_source":        "Heart Disease (UCI)",
                "selected_biases":    ["demographic", "socioeconomic"],
                "bias_intensity":     0.25,
                "poison_rate":        0.05,
                "access_inequality":  0.4,
                "n_runs":             3,
                "enable_governance":  True,
                "enable_gender_audit":False,
                "enable_redteam":     False,
            },
        },
        "Patient Advocate": {
            "icon": "🤝",
            "desc": "Maximum equity focus. Gender, linguistic, and access biases.",
            "settings": {
                "data_source":        "Abuja Multilingual Healthcare",
                "selected_biases":    ["demographic","gender","linguistic","socioeconomic"],
                "bias_intensity":     0.35,
                "poison_rate":        0.05,
                "access_inequality":  0.6,
                "n_runs":             3,
                "enable_governance":  True,
                "enable_gender_audit":True,
                "enable_redteam":     False,
            },
        },
    },

    # ── National Security ─────────────────────────────────────────────────────
    "security": {
        "Civil Liberties Analyst": {
            "icon": "⚖️",
            "desc": "Low surveillance, judicial oversight, focus on FPR and liberty score.",
            "settings": {
                "data_source":         "Synthetic",
                "selected_biases":     ["demographic", "geographic"],
                "bias_intensity":      0.15,
                "poison_rate":         0.03,
                "surveillance_level":  25,
                "data_retention":      30,
                "oversight_level":     "Judicial Review",
                "n_runs":              5,
                "enable_governance":   True,
                "enable_redteam":      False,
                "threat_level":        3,
            },
        },
        "Security Engineer": {
            "icon": "🛡️",
            "desc": "High surveillance, advanced attacks, full red-team active.",
            "settings": {
                "data_source":         "Synthetic",
                "selected_biases":     ["demographic","historical","measurement"],
                "bias_intensity":      0.3,
                "poison_rate":         0.12,
                "surveillance_level":  75,
                "data_retention":      180,
                "oversight_level":     "Moderate",
                "n_runs":              3,
                "enable_governance":   True,
                "enable_redteam":      True,
                "enable_arena":        True,
                "threat_level":        8,
            },
        },
        "Policy Researcher": {
            "icon": "📋",
            "desc": "GTD real data, balanced surveillance, moderate threat.",
            "settings": {
                "data_source":         "Real-World",
                "selected_biases":     ["demographic","geographic","historical"],
                "bias_intensity":      0.2,
                "poison_rate":         0.05,
                "surveillance_level":  50,
                "data_retention":      90,
                "oversight_level":     "Strong",
                "n_runs":              3,
                "enable_governance":   True,
                "enable_redteam":      False,
                "threat_level":        5,
            },
        },
        "Quick Demo": {
            "icon": "▶️",
            "desc": "Fastest path to results. 1 run, minimal config.",
            "settings": {
                "data_source":         "Synthetic",
                "selected_biases":     ["demographic"],
                "bias_intensity":      0.2,
                "poison_rate":         0.05,
                "surveillance_level":  50,
                "data_retention":      90,
                "oversight_level":     "Moderate",
                "n_runs":              1,
                "enable_governance":   False,
                "enable_redteam":      False,
                "threat_level":        5,
            },
        },
    },

    # ── Agrotech ──────────────────────────────────────────────────────────────
    "agrotech": {
        "Extension Officer": {
            "icon": "🌿",
            "desc": "Abuja FCT context, gender audit, agent economy active.",
            "settings": {
                "scenario_key":        "Plateau State Smallholder",
                "selected_biases":     ["demographic","gender","geographic"],
                "bias_intensity":      0.25,
                "poison_rate":         0.04,
                "n_runs":              3,
                "enable_xai":          True,
                "enable_gender_audit": True,
                "enable_agent_economy":True,
                "enable_governance":   True,
            },
        },
        "Development Economist": {
            "icon": "📊",
            "desc": "All bias types. Intersectional analysis. Full compliance report.",
            "settings": {
                "scenario_key":        "FCT Market Access",
                "selected_biases":     ["demographic","gender","linguistic",
                                        "socioeconomic","geographic"],
                "bias_intensity":      0.35,
                "poison_rate":         0.08,
                "n_runs":              5,
                "enable_xai":          True,
                "enable_gender_audit": True,
                "enable_agent_economy":True,
                "enable_governance":   True,
                "enable_redteam":      False,
            },
        },
        "Climate Researcher": {
            "icon": "🌦️",
            "desc": "Climate risk focus. Minimal bias, high climate stress.",
            "settings": {
                "scenario_key":        "Climate Risk Assessment",
                "selected_biases":     ["temporal","geographic"],
                "bias_intensity":      0.15,
                "poison_rate":         0.03,
                "n_runs":              3,
                "enable_xai":          True,
                "enable_gender_audit": False,
                "enable_agent_economy":False,
                "enable_governance":   True,
            },
        },
        "Quick Demo": {
            "icon": "▶️",
            "desc": "Single run, basic bias, XAI on.",
            "settings": {
                "scenario_key":        "Plateau State Smallholder",
                "selected_biases":     ["demographic","gender"],
                "bias_intensity":      0.2,
                "poison_rate":         0.05,
                "n_runs":              1,
                "enable_xai":          True,
                "enable_gender_audit": True,
                "enable_agent_economy":False,
                "enable_governance":   False,
            },
        },
    },
}


def preset_selector(domain: str) -> Optional[Dict[str, Any]]:
    """
    Render a quick-start preset panel above the sidebar controls.
    Returns the settings dict of the selected preset (or None if none chosen).

    Parameters
    ----------
    domain : "health" | "security" | "agrotech"

    Usage
    -----
    In the sidebar, before the controls:

        chosen_preset = preset_selector("health")
        if chosen_preset:
            # override defaults with preset values
            bias_intensity = chosen_preset.get("bias_intensity", 0.3)
            ...
    """
    presets = _PRESETS.get(domain, {})
    if not presets:
        return None

    st.markdown(
        "<p style='font-size:.8rem;font-weight:500;color:var(--color-text-secondary);"
        "margin-bottom:4px;'>Quick start — choose a persona</p>",
        unsafe_allow_html=True,
    )

    names  = list(presets.keys())
    preset_key = f"_preset_choice_{domain}"

    # Render as compact radio
    chosen_name = st.radio(
        "Persona",
        ["— custom —"] + names,
        format_func=lambda x: x if x == "— custom —"
                              else f"{presets[x].get("icon","")} {x}",
        key=preset_key,
        label_visibility="collapsed",
        horizontal=False,
    )

    if chosen_name == "— custom —":
        return None

    preset = presets[chosen_name]
    st.caption(preset["desc"])

    # Apply button
    apply_key = f"_preset_apply_{domain}_{chosen_name}"
    if st.button(f"Apply '{chosen_name}' preset", key=apply_key,
                 use_container_width=True):
        # Store in session_state so the caller can read them
        st.session_state[f"_applied_preset_{domain}"] = preset["settings"]
        st.rerun()

    # Return currently applied preset if one was stored
    return st.session_state.get(f"_applied_preset_{domain}")


# ──────────────────────────────────────────────────────────────────────────────
# §3  SIMULATION HISTORY BROWSER
# ──────────────────────────────────────────────────────────────────────────────

def history_browser(
    history_key: str,
    domain: str = "health",
    key_metrics: Optional[List[str]] = None,
) -> None:
    """
    Render a simulation history panel showing all past runs with
    compare, annotate, and restore functionality.

    Parameters
    ----------
    history_key  : session_state key that holds the list of snapshot dicts
                   Each snapshot should have: id, timestamp, label, config, metrics
    domain       : "health" | "security" | "agrotech"  (for colour theming)
    key_metrics  : metric names to show in the timeline (default: accuracy + fairness)

    Usage
    -----
    Add to the results section of each page:

        from components.ux_utils import history_browser
        history_browser("health_snapshot_history", domain="health")
    """
    # Ensure history list exists
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    snapshots: List[Dict] = st.session_state[history_key]
    metrics = key_metrics or ["accuracy", "fairness_score", "equity_score",
                               "fairness_score", "liberty_score"]

    # Theme colours per domain
    domain_color = {"health": "#2980b9", "security": "#1e3c72", "agrotech": "#3b6d11"}
    accent = domain_color.get(domain, "#2980b9")

    with st.expander(
        f"🕐 Simulation history ({len(snapshots)} saved runs)",
        expanded=False,
    ):
        if not snapshots:
            st.info(
                "No history yet. Run a simulation and click **Save to history** "
                "to start building a comparison timeline."
            )
            return

        st.markdown(
            "<p style='font-size:.85rem;color:var(--color-text-secondary);margin-bottom:.5rem;'>"
            "Compare past runs side by side, annotate findings, or restore a previous configuration.</p>",
            unsafe_allow_html=True,
        )

        # ── Timeline cards ────────────────────────────────────────────────────
        import pandas as pd
        for i, snap in enumerate(reversed(snapshots)):
            snap_id      = snap.get("id", i)
            ts           = snap.get("timestamp", "")[:16].replace("T", " ")
            label        = snap.get("label", f"Run {snap_id}")
            snap_metrics = snap.get("metrics", {})
            snap_config  = snap.get("config", {})
            annotation   = snap.get("annotation", "")

            with st.container():
                c_info, c_metrics, c_actions = st.columns([3, 4, 2])

                with c_info:
                    st.markdown(
                        f"<div style='border-left:3px solid {accent};"
                        f"padding-left:.6rem;'>"
                        f"<span style='font-weight:500;font-size:.9rem;"
                        f"color:var(--color-text-primary);'>{label}</span><br>"
                        f"<span style='font-size:.78rem;color:var(--color-text-secondary);'>"
                        f"{ts}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if annotation:
                        st.caption(f"📝 {annotation}")

                with c_metrics:
                    # Show up to 3 key metrics inline
                    shown = [(k, snap_metrics[k]) for k in metrics
                             if k in snap_metrics][:3]
                    cols = st.columns(len(shown)) if shown else []
                    for col, (k, v) in zip(cols, shown):
                        label_short = k.replace("_", " ").replace("score","").strip()
                        col.metric(label_short.title(), f"{v:.2f}" if isinstance(v, float) else v)

                with c_actions:
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("📋", key=f"_hist_restore_{snap_id}_{i}",
                                     help="Restore this configuration"):
                            st.session_state[f"_applied_preset_{domain}"] = snap_config
                            st.success(f"Configuration from '{label}' restored. Re-run to apply.")

                    with btn_col2:
                        if st.button("🗑️", key=f"_hist_delete_{snap_id}_{i}",
                                     help="Remove this entry"):
                            st.session_state[history_key] = [
                                s for s in snapshots if s.get("id") != snap_id
                            ]
                            st.rerun()

                # Annotation input (collapsed by default)
                ann_key = f"_ann_{snap_id}_{i}"
                new_ann = st.text_input(
                    "Add note",
                    value=annotation,
                    key=ann_key,
                    label_visibility="collapsed",
                    placeholder="Add a note about this run…",
                )
                if new_ann != annotation:
                    for s in st.session_state[history_key]:
                        if s.get("id") == snap_id:
                            s["annotation"] = new_ann

                st.divider()

        # ── Comparison chart ──────────────────────────────────────────────────
        if len(snapshots) >= 2:
            st.markdown("#### Metrics comparison across saved runs")
            try:
                import plotly.express as px
                metric_to_plot = st.selectbox(
                    "Metric to compare",
                    [m for m in metrics if any(m in s.get("metrics", {}) for s in snapshots)],
                    key=f"_hist_metric_{domain}",
                )
                chart_data = [
                    {"Run": s.get("label", f"Run {s.get('id',i)}"),
                     metric_to_plot: s.get("metrics", {}).get(metric_to_plot, 0)}
                    for i, s in enumerate(snapshots)
                    if metric_to_plot in s.get("metrics", {})
                ]
                if chart_data:
                    fig = px.bar(
                        chart_data, x="Run", y=metric_to_plot,
                        title=f"{metric_to_plot.replace('_',' ').title()} across saved runs",
                        color=metric_to_plot,
                        color_continuous_scale="RdYlGn" if "fairness" in metric_to_plot
                                                        or "equity" in metric_to_plot
                                                        else "Blues",
                    )
                    fig.update_layout(showlegend=False, height=280)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass  # plotly not available in this context

        # ── Export ────────────────────────────────────────────────────────────
        if snapshots:
            export_data = [
                {k: v for k, v in s.items() if k != "raw_df"}
                for s in snapshots
            ]
            st.download_button(
                "📥 Export history (JSON)",
                json.dumps(export_data, indent=2, default=str),
                f"gags_{domain}_history.json",
                "application/json",
                use_container_width=True,
            )


def save_to_history(
    history_key: str,
    label: str,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    annotation: str = "",
) -> None:
    """
    Save the current simulation result into the history list.
    Call this after a successful run:

        save_to_history("health_snapshot_history",
                        label=f"Bias=0.3, Poison=0.05",
                        metrics={"accuracy": 0.82, "fairness_score": 0.74},
                        config={"bias_intensity": 0.3, ...})
    """
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    history = st.session_state[history_key]
    new_id  = len(history) + 1

    history.append({
        "id":         new_id,
        "timestamp":  datetime.now().isoformat(),
        "label":      label or f"Run {new_id}",
        "metrics":    {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in metrics.items()},
        "config":     config,
        "annotation": annotation,
    })
    # Keep last 20 snapshots to avoid unbounded memory growth
    st.session_state[history_key] = history[-20:]


# ──────────────────────────────────────────────────────────────────────────────
# §4  GUIDED TOUR / ONBOARDING BANNER
# ──────────────────────────────────────────────────────────────────────────────

_TOUR_STEPS: Dict[str, List[Dict[str, str]]] = {
    "health": [
        {"step": "1", "title": "Choose your data source",
         "body": "Start with 'Synthetic Only' for controlled experiments, or load real clinical data "
                 "(UCI Heart Disease, PIMA Diabetes). The Abuja scenarios use Nigerian demographic distributions."},
        {"step": "2", "title": "Pick a quick-start preset",
         "body": "Not sure where to begin? Use the persona selector above the sidebar controls. "
                 "'Clinical Regulator' gives a strict fairness audit; 'ML Researcher' enables everything."},
        {"step": "3", "title": "Configure bias and attacks",
         "body": "Bias types inject real-world disparities into the data. Demographic bias is the most "
                 "impactful. Poisoning rate simulates data corruption by adversaries. Start with defaults."},
        {"step": "4", "title": "Run and read the dashboard",
         "body": "After running, the Equity tab shows the fairness gap between income groups. "
                 "Red = problem. Green = acceptable. The governance banner shows the AI's policy verdict."},
        {"step": "5", "title": "Expand metric definitions",
         "body": "See the 'What do these metrics mean?' panel below the KPIs. "
                 "Every metric has a plain-language explanation, formula, and visual example."},
    ],
    "security": [
        {"step": "1", "title": "Choose synthetic or real data",
         "body": "Synthetic data gives controlled experiments. GTD (Global Terrorism Database) "
                 "provides real historical attack patterns. UNSW-NB15 covers cyber incidents."},
        {"step": "2", "title": "Set the surveillance trade-off",
         "body": "The surveillance intensity slider is the key control. High surveillance improves "
                 "detection but reduces the liberty score. The privacy progress bar updates live."},
        {"step": "3", "title": "Choose an oversight mechanism",
         "body": "'Judicial Review' dramatically improves the liberty score by constraining "
                 "how surveillance data is used. 'Minimal' oversight shows the worst-case outcome."},
        {"step": "4", "title": "Read the trade-off scatter",
         "body": "The Trade-offs tab shows detection rate vs liberty score. The green dashed box "
                 "is the optimal zone — high detection AND high liberty. Most configs land outside it."},
        {"step": "5", "title": "Check the governance banner",
         "body": "After running, a coloured banner shows the citizen vote outcome on your "
                 "surveillance policy and any AI drift flags. Red = policy auto-reversed."},
    ],
    "agrotech": [
        {"step": "1", "title": "Choose a crop scenario",
         "body": "Each scenario targets a different prediction task — crop failure risk, "
                 "market access viability, fertilizer allocation, or climate risk. "
                 "All use Abuja FCT calibrated distributions."},
        {"step": "2", "title": "Enable the Gender Equity Audit",
         "body": "52% of Nigerian smallholder farmers are women. The UNESCO Women4EthicalAI audit "
                 "measures whether the AI treats female and male farmers equally. Enable it by default."},
        {"step": "3", "title": "Enable the Agent Economy",
         "body": "The Vickrey auction module shows whether autonomous AI agents fairly allocate "
                 "irrigation water, fertilizer quotas, and drone hours — or whether large operations "
                 "crowd out smallholders."},
        {"step": "4", "title": "Enable XAI for explanations",
         "body": "With XAI on, the Explainable AI tab shows which features drive each prediction "
                 "and what a farmer would need to change to get a different outcome (counterfactual)."},
        {"step": "5", "title": "Read the Compliance tab",
         "body": "The auto-generated compliance report maps results to NITDA (Nigeria), "
                 "UNESCO, EU AI Act, ISO 42001, and NIST AI RMF. Download the model card for procurement."},
    ],
}


def guided_tour_banner(
    domain: str,
    dismissed_key: Optional[str] = None,
) -> None:
    """
    Show a guided onboarding banner for first-time users.
    Automatically hides after the user dismisses it (stored in session state).

    Parameters
    ----------
    domain        : "health" | "security" | "agrotech"
    dismissed_key : session_state key to track dismissal
                    (defaults to f"_tour_dismissed_{domain}")

    Usage
    -----
    Place immediately after st.set_page_config() and CSS injection,
    before the sidebar:

        from components.ux_utils import guided_tour_banner
        guided_tour_banner("health")
    """
    dismiss_key = dismissed_key or f"_tour_dismissed_{domain}"

    if st.session_state.get(dismiss_key, False):
        return  # User already dismissed

    steps = _TOUR_STEPS.get(domain, [])
    if not steps:
        return

    domain_labels = {"health": "Healthcare", "security": "National Security", "agrotech": "Agrotech"}
    domain_icons  = {"health": "🏥", "security": "🛡️", "agrotech": "🌾"}
    domain_colors = {"health": "#2980b9", "security": "#1e3c72", "agrotech": "#3b6d11"}

    label = domain_labels.get(domain, "GAGS")
    icon  = domain_icons.get(domain, "▶")
    color = domain_colors.get(domain, "#2980b9")

    st.markdown(
        f"""
        <div style="border:1px solid {color}33;border-radius:var(--border-radius-lg);
                    padding:1rem 1.25rem;margin-bottom:1rem;
                    background:var(--color-background-secondary);">
          <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem;">
            <span style="font-size:1rem;">{icon}</span>
            <span style="font-weight:500;font-size:.95rem;color:var(--color-text-primary);">
              Welcome to the {label} Equity Simulation
            </span>
            <span style="margin-left:auto;font-size:.78rem;color:var(--color-text-tertiary);">
              First time here? Follow these 5 steps:
            </span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;">
            {"".join(
              f'''<div style="background:var(--color-background-primary);
                             border:0.5px solid {color}44;
                             border-radius:var(--border-radius-md);
                             padding:.6rem .75rem;">
                   <div style="font-size:.72rem;font-weight:600;color:{color};
                               margin-bottom:.2rem;">Step {s['step']}</div>
                   <div style="font-size:.8rem;font-weight:500;
                               color:var(--color-text-primary);margin-bottom:.2rem;">
                     {s['title']}
                   </div>
                   <div style="font-size:.75rem;color:var(--color-text-secondary);
                               line-height:1.4;">{s['body']}</div>
                 </div>'''
              for s in steps
            )}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "✓ Got it — dismiss this guide",
        key=f"_tour_dismiss_btn_{domain}",
        help="You can re-read this guide in the Help section",
    ):
        st.session_state[dismiss_key] = True
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# §5  SHAREABLE SIMULATION URL
# ──────────────────────────────────────────────────────────────────────────────
"""
Encodes the current simulation configuration into a URL query string so a
researcher can copy and share the exact setup with a colleague.

Uses Streamlit's st.query_params (v1.30+) with a base64+JSON payload.
Falls back gracefully to a copy-to-clipboard JSON block on older versions.

Usage
-----
    from components.ux_utils import share_url_panel, load_config_from_url
    
    # At page top (before sidebar widgets), load any URL-encoded config:
    url_cfg = load_config_from_url()
    
    # In the results section, show the share panel:
    share_url_panel(domain="health", config={...})
"""

import base64 as _b64
import urllib.parse as _urlparse


def _encode_config(config: Dict[str, Any]) -> str:
    """Encode a config dict to a URL-safe base64 string."""
    raw = json.dumps(config, default=str).encode()
    return _b64.urlsafe_b64encode(raw).decode()


def _decode_config(encoded: str) -> Optional[Dict[str, Any]]:
    """Decode a base64 config string back to a dict. Returns None on failure."""
    try:
        raw = _b64.urlsafe_b64decode(encoded.encode())
        return json.loads(raw)
    except Exception:
        return None


def load_config_from_url(param_name: str = "cfg") -> Optional[Dict[str, Any]]:
    """
    Read a simulation config from the URL query string (if present).
    Call this ONCE at the top of each page, before any sidebar widgets.

    Returns the decoded config dict, or None if no config is in the URL.

    Example URL:
        http://localhost:8501/Healthcare_Equity?cfg=eyJkb21haW4i...
    """
    try:
        params = st.query_params
        encoded = params.get(param_name)
        if encoded:
            cfg = _decode_config(encoded)
            if cfg:
                # Store in session state so sidebar widgets can read it
                domain = cfg.get("domain", "generic")
                st.session_state[f"_url_config_{domain}"] = cfg
                return cfg
    except Exception:
        pass
    return None


def share_url_panel(
    domain: str,
    config: Dict[str, Any],
    page_path: str = "",
    param_name: str = "cfg",
) -> None:
    """
    Render a "Share this simulation" panel with a copyable URL and JSON export.

    Parameters
    ----------
    domain      : "health" | "security" | "agrotech"
    config      : the current simulation configuration dict
    page_path   : optional page path suffix (e.g. "Healthcare_Equity")
    param_name  : URL query parameter name (default "cfg")

    Placement
    ---------
    Call after the results KPI row, before the tabs.
    """
    domain_labels = {"health": "Healthcare", "security": "National Security",
                     "agrotech": "Agrotech"}
    domain_label  = domain_labels.get(domain, domain.title())
    domain_colors = {"health": "#2980b9", "security": "#c0392b", "agrotech": "#27ae60"}
    color         = domain_colors.get(domain, "#8e44ad")

    # Build the encoded config
    share_config = {k: v for k, v in config.items()
                    if not isinstance(v, (list, dict)) or k in
                    ("selected_biases", "biases", "features_enabled")}
    share_config["domain"] = domain
    share_config["shared_at"] = datetime.now().isoformat()[:16]

    encoded  = _encode_config(share_config)
    base_url = f"http://localhost:8501/{page_path}".rstrip("/")
    full_url = f"{base_url}?{param_name}={encoded}"

    with st.expander("🔗 Share this simulation configuration", expanded=False):
        st.markdown(
            f"<p style='font-size:.85rem;color:var(--color-text-secondary);margin-bottom:.5rem;'>"
            f"Send this URL to a colleague — it encodes your exact configuration so they "
            f"can reproduce this simulation instantly.</p>",
            unsafe_allow_html=True,
        )

        # URL display box
        st.markdown(
            f"<div style='background:var(--color-background-secondary);"
            f"border:0.5px solid var(--color-border-tertiary);"
            f"border-radius:var(--border-radius-md);padding:.6rem .85rem;"
            f"font-family:monospace;font-size:.78rem;"
            f"color:var(--color-text-primary);word-break:break-all;'>"
            f"{full_url}</div>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        # Try to write to st.query_params (Streamlit ≥ 1.30)
        with col1:
            if st.button("📋 Copy URL to clipboard", key=f"_share_copy_{domain}",
                         use_container_width=True):
                try:
                    st.query_params[param_name] = encoded
                    st.success("URL updated in your browser address bar.")
                except Exception:
                    st.info("Copy the URL above manually.")

        with col2:
            st.download_button(
                "📥 Download config JSON",
                json.dumps(share_config, indent=2, default=str),
                f"gags_{domain}_config.json",
                "application/json",
                key=f"_share_json_{domain}",
                use_container_width=True,
            )

        with col3:
            # QR code alternative — plain text fallback
            st.download_button(
                "📄 Download share link",
                full_url,
                f"gags_{domain}_share_link.txt",
                "text/plain",
                key=f"_share_link_{domain}",
                use_container_width=True,
            )

        # Config summary table
        st.markdown(
            "<p style='font-size:.8rem;font-weight:500;"
            "color:var(--color-text-secondary);margin:.75rem 0 .3rem;'>"
            "Configuration encoded in this link:</p>",
            unsafe_allow_html=True,
        )
        rows = [f"<tr><td style='padding:3px 8px;color:var(--color-text-secondary);"
                f"font-size:.78rem;'>{k}</td>"
                f"<td style='padding:3px 8px;font-size:.78rem;font-family:monospace;"
                f"color:var(--color-text-primary);'>{str(v)[:60]}</td></tr>"
                for k, v in share_config.items() if k != "shared_at"]
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;"
            f"border:0.5px solid var(--color-border-tertiary);"
            f"border-radius:var(--border-radius-md);overflow:hidden;'>"
            f"{''.join(rows)}</table>",
            unsafe_allow_html=True,
        )

        st.caption(
            "⚠️ The URL encodes configuration only, not results. "
            "Your colleague will need to click Run to reproduce the simulation."
        )


def apply_url_config(domain: str) -> Optional[Dict[str, Any]]:
    """
    Return any URL-loaded config for the given domain, then clear it.
    Call after sidebar widgets are declared to apply overrides.
    """
    key = f"_url_config_{domain}"
    cfg = st.session_state.pop(key, None)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# §6  ENHANCED ANNOTATION LAYER
# ──────────────────────────────────────────────────────────────────────────────
"""
Standalone annotation panel for reviewers and auditors to attach structured
notes to specific simulation findings.

Supports:
  - Typed annotations (finding, concern, recommendation, question, approval)
  - Severity tagging (critical, high, medium, low, info)
  - Author + timestamp
  - Edit and delete
  - Export as structured JSON audit trail
  - Import previously exported annotations

Separate from the history_browser inline notes — this is a dedicated
audit-trail component for formal review workflows.

Usage
-----
    from components.ux_utils import annotation_panel
    annotation_panel("health_annotations")
"""

_ANNOTATION_TYPES = ["finding", "concern", "recommendation", "question", "approval"]
_SEVERITY_LEVELS  = ["critical", "high", "medium", "low", "info"]

_TYPE_ICONS = {
    "finding":        "🔍",
    "concern":        "⚠️",
    "recommendation": "💡",
    "question":       "❓",
    "approval":       "✅",
}
_SEVERITY_COLORS = {
    "critical": "#e74c3c",
    "high":     "#e67e22",
    "medium":   "#f39c12",
    "low":      "#3498db",
    "info":     "#95a5a6",
}


def annotation_panel(
    store_key: str,
    context_label: str = "this simulation",
    allow_import: bool = True,
) -> None:
    """
    Render a full annotation / audit-trail panel.

    Parameters
    ----------
    store_key     : session_state key for the annotation list
    context_label : human-readable context (e.g. "Healthcare Run 3")
    allow_import  : whether to show the import-from-JSON option

    Placement
    ---------
    Add as a collapsible section below the main results tabs,
    before the recommendations section.
    """
    if store_key not in st.session_state:
        st.session_state[store_key] = []

    annotations: List[Dict] = st.session_state[store_key]

    with st.expander(
        f"📝 Annotations & Audit Trail ({len(annotations)} note"
        f"{'s' if len(annotations) != 1 else ''})",
        expanded=False,
    ):
        st.markdown(
            "<p style='font-size:.85rem;color:var(--color-text-secondary);"
            "margin-bottom:.75rem;'>"
            "Attach structured notes to findings for formal review, audit, "
            "or collaboration. All annotations include author, timestamp, "
            "and severity — and can be exported as a JSON audit trail.</p>",
            unsafe_allow_html=True,
        )

        # ── Add new annotation form ───────────────────────────────────────
        st.markdown(
            "<p style='font-size:.83rem;font-weight:500;"
            "color:var(--color-text-primary);margin-bottom:.3rem;'>"
            "Add annotation</p>",
            unsafe_allow_html=True,
        )

        f1, f2, f3 = st.columns([2, 2, 2])
        with f1:
            ann_type = st.selectbox(
                "Type",
                _ANNOTATION_TYPES,
                format_func=lambda x: f"{_TYPE_ICONS.get(x,'')} {x.title()}",
                key=f"_ann_type_{store_key}",
                label_visibility="collapsed",
            )
        with f2:
            ann_severity = st.selectbox(
                "Severity",
                _SEVERITY_LEVELS,
                index=2,
                key=f"_ann_sev_{store_key}",
                label_visibility="collapsed",
            )
        with f3:
            ann_author = st.text_input(
                "Author",
                value=st.session_state.get(f"_ann_author_{store_key}", ""),
                placeholder="Your name or role",
                key=f"_ann_author_input_{store_key}",
                label_visibility="collapsed",
            )

        ann_text = st.text_area(
            "Annotation text",
            placeholder=f"Describe your {ann_type} about {context_label}…",
            height=80,
            key=f"_ann_text_{store_key}",
            label_visibility="collapsed",
        )

        # Optional: link to a specific metric
        ann_metric = st.text_input(
            "Linked metric (optional)",
            placeholder="e.g. false_positive_rate, equity_score, liberty_score",
            key=f"_ann_metric_{store_key}",
            label_visibility="collapsed",
        )

        if st.button(
            f"{_TYPE_ICONS.get(ann_type, '📝')} Add {ann_type.title()}",
            key=f"_ann_add_{store_key}",
            disabled=not ann_text.strip(),
            use_container_width=False,
        ):
            new_ann = {
                "id":           len(annotations) + 1,
                "type":         ann_type,
                "severity":     ann_severity,
                "author":       ann_author.strip() or "Anonymous",
                "text":         ann_text.strip(),
                "metric":       ann_metric.strip() or None,
                "timestamp":    datetime.now().isoformat(),
                "context":      context_label,
            }
            st.session_state[store_key].append(new_ann)
            # Remember author for next annotation
            st.session_state[f"_ann_author_{store_key}"] = ann_author
            st.rerun()

        st.divider()

        # ── Existing annotations ──────────────────────────────────────────
        if not annotations:
            st.markdown(
                "<p style='font-size:.83rem;color:var(--color-text-tertiary);"
                "text-align:center;padding:.5rem 0;'>No annotations yet.</p>",
                unsafe_allow_html=True,
            )
        else:
            # Filter controls
            fc1, fc2 = st.columns(2)
            with fc1:
                filter_type = st.multiselect(
                    "Filter by type",
                    _ANNOTATION_TYPES,
                    default=[],
                    key=f"_ann_ftype_{store_key}",
                    placeholder="All types",
                    label_visibility="collapsed",
                )
            with fc2:
                filter_sev = st.multiselect(
                    "Filter by severity",
                    _SEVERITY_LEVELS,
                    default=[],
                    key=f"_ann_fsev_{store_key}",
                    placeholder="All severities",
                    label_visibility="collapsed",
                )

            shown = [
                a for a in reversed(annotations)
                if (not filter_type or a["type"] in filter_type)
                and (not filter_sev or a["severity"] in filter_sev)
            ]

            for ann in shown:
                sev_col  = _SEVERITY_COLORS.get(ann["severity"], "#888")
                type_icon = _TYPE_ICONS.get(ann["type"], "📝")
                ts        = ann["timestamp"][:16].replace("T", " ")

                # Annotation card
                st.markdown(
                    f"<div style='border-left:3px solid {sev_col};"
                    f"background:var(--color-background-secondary);"
                    f"border-radius:0 var(--border-radius-md) var(--border-radius-md) 0;"
                    f"padding:.6rem .85rem;margin-bottom:.5rem;'>"

                    f"<div style='display:flex;align-items:center;gap:.5rem;"
                    f"margin-bottom:.3rem;'>"
                    f"<span style='font-size:.8rem;'>{type_icon}</span>"
                    f"<span style='font-size:.8rem;font-weight:600;"
                    f"color:{sev_col};text-transform:uppercase;'>{ann['severity']}</span>"
                    f"<span style='font-size:.78rem;font-weight:500;"
                    f"color:var(--color-text-primary);'>{ann.get("type","").title()}</span>"
                    f"<span style='margin-left:auto;font-size:.75rem;"
                    f"color:var(--color-text-tertiary);'>{ann.get("author","")} · {ts}</span>"
                    f"</div>"

                    f"<p style='margin:0;font-size:.85rem;"
                    f"color:var(--color-text-primary);line-height:1.5;'>{ann['text']}</p>"

                    + (
                        f"<p style='margin:.3rem 0 0;font-size:.75rem;"
                        f"color:var(--color-text-secondary);'>"
                        f"📊 Linked to: <code>{ann['metric']}</code></p>"
                        if ann.get("metric") else ""
                    ) +

                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Delete button (inline, compact)
                if st.button(
                    "✕ Remove",
                    key=f"_ann_del_{store_key}_{ann['id']}",
                    help="Remove this annotation",
                ):
                    st.session_state[store_key] = [
                        a for a in annotations if a["id"] != ann["id"]
                    ]
                    st.rerun()

        st.divider()

        # ── Export / import ───────────────────────────────────────────────
        ex1, ex2 = st.columns(2)
        with ex1:
            if annotations:
                export_data = {
                    "gags_version":   "3.0",
                    "exported_at":    datetime.now().isoformat(),
                    "context":        context_label,
                    "annotation_count": len(annotations),
                    "annotations":    annotations,
                }
                st.download_button(
                    "📥 Export audit trail (JSON)",
                    json.dumps(export_data, indent=2),
                    f"gags_annotations_{store_key}.json",
                    "application/json",
                    use_container_width=True,
                    key=f"_ann_export_{store_key}",
                )

        with ex2:
            if allow_import:
                uploaded = st.file_uploader(
                    "Import annotations",
                    type=["json"],
                    key=f"_ann_import_{store_key}",
                    label_visibility="collapsed",
                    help="Import a previously exported GAGS annotation file",
                )
                if uploaded:
                    try:
                        data = json.load(uploaded)
                        imported = data.get("annotations", [])
                        # Avoid duplicate IDs
                        max_id = max((a["id"] for a in annotations), default=0)
                        for i, a in enumerate(imported):
                            a["id"] = max_id + i + 1
                        st.session_state[store_key].extend(imported)
                        st.success(f"Imported {len(imported)} annotation(s).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Import failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# §7  ROLE-BASED VIEW SWITCHER
# ──────────────────────────────────────────────────────────────────────────────
"""
Persona-specific dashboard views that filter the interface complexity to
match the user's role.

Roles
-----
  Data Scientist   : full interface — all tabs, all controls, all metrics
  Clinician        : clinical safety focus — sensitivity/specificity, equity score,
                     patient impact; no raw data or feature module tabs
  Regulator        : compliance focus — fairness metrics, compliance tab,
                     governance tab, annotated audit trail
  Board Member     : executive summary only — 4 KPI cards, plain-language
                     verdict, key recommendations, no technical detail
  Field Officer    : simplified view for on-the-ground use — prediction outcome,
                     plain-language explanation, actionable recommendation

Usage
-----
    from components.ux_utils import role_switcher, get_active_role, ROLE_TAB_VISIBILITY

    # In the sidebar, before controls:
    role_switcher("health")

    # In the results section:
    role = get_active_role("health")
    if ROLE_TAB_VISIBILITY[role]["show_raw_results"]:
        with tab_raw:
            ...
"""

# Which tabs/sections are visible for each role
# Keys must match what the page modules check
ROLE_TAB_VISIBILITY: Dict[str, Dict[str, bool]] = {
    "Data Scientist": {
        "show_performance":     True,
        "show_equity":          True,
        "show_clinical":        True,
        "show_feature_modules": True,
        "show_data_analysis":   True,
        "show_xai":             True,
        "show_compliance":      True,
        "show_longitudinal":    True,
        "show_federated":       True,
        "show_raw_results":     True,
        "show_advanced_sidebar":True,
        "show_kpi_detail":      True,
        "show_annotation":      True,
    },
    "Clinician": {
        "show_performance":     True,
        "show_equity":          True,
        "show_clinical":        True,
        "show_feature_modules": False,
        "show_data_analysis":   False,
        "show_xai":             True,   # plain-language explanations are useful
        "show_compliance":      False,
        "show_longitudinal":    False,
        "show_federated":       False,
        "show_raw_results":     False,
        "show_advanced_sidebar":False,
        "show_kpi_detail":      True,
        "show_annotation":      True,
    },
    "Regulator": {
        "show_performance":     True,
        "show_equity":          True,
        "show_clinical":        False,
        "show_feature_modules": False,
        "show_data_analysis":   False,
        "show_xai":             True,
        "show_compliance":      True,
        "show_longitudinal":    True,
        "show_federated":       False,
        "show_raw_results":     False,
        "show_advanced_sidebar":False,
        "show_kpi_detail":      True,
        "show_annotation":      True,
    },
    "Board Member": {
        "show_performance":     True,
        "show_equity":          False,
        "show_clinical":        False,
        "show_feature_modules": False,
        "show_data_analysis":   False,
        "show_xai":             False,
        "show_compliance":      True,   # just the verdict card
        "show_longitudinal":    False,
        "show_federated":       False,
        "show_raw_results":     False,
        "show_advanced_sidebar":False,
        "show_kpi_detail":      False,  # show simplified cards only
        "show_annotation":      False,
    },
    "Field Officer": {
        "show_performance":     True,
        "show_equity":          True,
        "show_clinical":        False,
        "show_feature_modules": False,
        "show_data_analysis":   False,
        "show_xai":             True,   # counterfactual "what to change" is actionable
        "show_compliance":      False,
        "show_longitudinal":    False,
        "show_federated":       False,
        "show_raw_results":     False,
        "show_advanced_sidebar":False,
        "show_kpi_detail":      False,
        "show_annotation":      True,
    },
}

_ROLE_DESCRIPTIONS = {
    "Data Scientist":  "Full interface — all tabs, controls, and technical metrics.",
    "Clinician":       "Clinical safety focus — sensitivity, equity, and patient impact.",
    "Regulator":       "Compliance focus — fairness audit, governance, annotation trail.",
    "Board Member":    "Executive summary — key outcomes and plain-language verdict.",
    "Field Officer":   "Simplified view — prediction outcome and actionable guidance.",
}

_ROLE_ICONS = {
    "Data Scientist": "🔬",
    "Clinician":      "🩺",
    "Regulator":      "🏛️",
    "Board Member":   "📋",
    "Field Officer":  "🌿",
}



# ── Domain-specific role & perspective configurations ─────────────────────────


DOMAIN_ROLES: Dict[str, Dict[str, Dict]] = {

    # ── Healthcare ─────────────────────────────────────────────────────────────
    "health": {
        "roles": {
            "Data Scientist": {
                "icon": "🔬", "desc": "Full interface — all tabs, technical metrics, and benchmarks.",
                "algo": "hist_gradient_boosting", "algo_label": "Hist Gradient Boosting",
                "algo_reason": "Best balance of speed, accuracy and feature importance for exploratory research.",
                "perspective": "Research",
                "tab_visibility": {"performance": True, "equity": True, "clinical": True,
                    "feature_modules": True, "data_analysis": True, "xai": True,
                    "compliance": True, "longitudinal": True, "federated": True, "raw": True},
                "sidebar_defaults": {"bias_intensity": 0.3, "n_runs": 3,
                    "enable_governance": True, "enable_gender_audit": True},
                "focus_metrics": ["accuracy", "fairness_score", "demographic_parity", "sensitivity"],
            },
            "Clinician": {
                "icon": "🩺", "desc": "Clinical safety focus — sensitivity, equity gaps, and patient impact.",
                "algo": "calibrated_hgb", "algo_label": "Calibrated HGB",
                "algo_reason": "Calibrated probabilities give reliable risk scores for clinical triage decisions. Minimises false negatives.",
                "perspective": "Industry",
                "tab_visibility": {"performance": True, "equity": True, "clinical": True,
                    "feature_modules": False, "data_analysis": False, "xai": True,
                    "compliance": False, "longitudinal": False, "federated": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.2, "n_runs": 3,
                    "enable_governance": True, "enable_gender_audit": True},
                "focus_metrics": ["sensitivity", "fairness_score", "adv_lo", "gender_gap"],
                "role_brief": "You are reviewing this AI triage system for clinical deployment. Your priority is: does it harm patients through missed diagnoses or unequal care?",
            },
            "Regulator": {
                "icon": "🏛️", "desc": "Compliance audit — fairness report, governance, annotation trail.",
                "algo": "balanced_hgb", "algo_label": "Balanced HGB (Fairness-Weighted)",
                "algo_reason": "Class weights enforce equal error rates across demographic groups — the regulatory gold standard for protected class fairness.",
                "perspective": "Policy Brief",
                "tab_visibility": {"performance": True, "equity": True, "clinical": False,
                    "feature_modules": True, "data_analysis": False, "xai": True,
                    "compliance": True, "longitudinal": True, "federated": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.1, "n_runs": 5,
                    "enable_governance": True, "enable_gender_audit": True},
                "focus_metrics": ["fairness_score", "demographic_parity", "equalized_odds"],
                "role_brief": "You are conducting a NITDA/NHIA regulatory audit. Your priority is: does this AI meet the WHO fairness threshold of demographic parity gap < 0.10?",
            },
            "Hospital Manager": {
                "icon": "🏥", "desc": "Executive view — KPI dashboard and plain-language policy verdict.",
                "algo": "random_forest", "algo_label": "Random Forest",
                "algo_reason": "Interpretable, robust, and widely understood by non-technical stakeholders. Easy to explain to hospital boards.",
                "perspective": "Industry",
                "tab_visibility": {"performance": True, "equity": False, "clinical": True,
                    "feature_modules": False, "data_analysis": False, "xai": False,
                    "compliance": True, "longitudinal": False, "federated": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.25, "n_runs": 1,
                    "enable_governance": True, "enable_gender_audit": False},
                "focus_metrics": ["accuracy", "fairness_score", "sensitivity"],
                "role_brief": "You are deciding whether to deploy this AI in your hospital. Focus on: accuracy, cost of errors, and board-level fairness compliance.",
            },
            "Patient Advocate": {
                "icon": "🤝", "desc": "Access equity — who is denied care and why, in plain language.",
                "algo": "balanced_hgb", "algo_label": "Balanced HGB (Fairness-Weighted)",
                "algo_reason": "Fairness-weighted training directly reduces the denial gap for the most vulnerable patients.",
                "perspective": "Policy Brief",
                "tab_visibility": {"performance": False, "equity": True, "clinical": True,
                    "feature_modules": False, "data_analysis": False, "xai": True,
                    "compliance": False, "longitudinal": False, "federated": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.4, "n_runs": 3,
                    "enable_governance": False, "enable_gender_audit": True,
                    "selected_biases": ["demographic", "gender", "socioeconomic"]},
                "focus_metrics": ["adv_lo", "gender_gap", "demographic_parity"],
                "role_brief": "You are advocating for patients denied care by this AI. Your focus: which groups are harmed most, and what would a fair system look like?",
            },
        },
        "perspectives": ["Industry", "Research", "Policy Brief"],
        "perspective_help": {
            "Industry": "KPI dashboard — accuracy, fairness score, key alerts.",
            "Research": "Full statistical depth — distributions, CIs, benchmarks.",
            "Policy Brief": "Plain-language summary for non-technical stakeholders.",
        },
    },

    # ── Economic Justice ───────────────────────────────────────────────────────
    "economic": {
        "roles": {
            "Data Scientist": {
                "icon": "🔬", "desc": "Full interface — all tabs, technical metrics, and benchmarks.",
                "algo": "hist_gradient_boosting", "algo_label": "Hist Gradient Boosting",
                "algo_reason": "Handles mixed data types and missing values natively — ideal for Nigeria labour survey data.",
                "perspective": "Research",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "benchmarks": True, "impact": True, "longitudinal": True,
                    "federated": True, "compliance": True, "raw": True},
                "sidebar_defaults": {"bias_intensity": 0.3, "n_runs": 3},
                "focus_metrics": ["fairness_score", "gender_outcome_gap", "intersectional_worst_gap"],
            },
            "Labour Economist": {
                "icon": "📊", "desc": "Wage gaps, informal sector bias, and automation displacement.",
                "algo": "gradient_boosting", "algo_label": "Gradient Boosting (GBT)",
                "algo_reason": "GBT captures non-linear wage suppression effects and interaction terms between gender × informality × geography.",
                "perspective": "Research",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "benchmarks": True, "impact": True, "longitudinal": True,
                    "federated": False, "compliance": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.35, "n_runs": 5},
                "focus_metrics": ["wage_suppression_index", "gender_outcome_gap", "informal_sector_gap"],
                "role_brief": "You are analysing algorithmic wage discrimination. Your priority: quantify the wage gap between formal and informal workers, and between male and female workers in the same role.",
            },
            "Regulator (NITDA)": {
                "icon": "🏛️", "desc": "NITDA / FCCPC compliance — fairness audit and regulatory verdict.",
                "algo": "balanced_hgb", "algo_label": "Balanced HGB (Fairness-Weighted)",
                "algo_reason": "Mandatory for NITDA AI Policy 2023 compliance — fairness-weighted training is the regulatory benchmark.",
                "perspective": "Policy Brief",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "benchmarks": False, "impact": False, "longitudinal": False,
                    "federated": False, "compliance": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.1, "n_runs": 5},
                "focus_metrics": ["fairness_score", "demographic_parity", "equalized_odds"],
                "role_brief": "You are conducting a NITDA algorithmic fairness audit under the Nigeria AI Policy 2023. Does this system meet the disparate impact threshold?",
            },
            "HR Director": {
                "icon": "👔", "desc": "Hiring and wage-setting bias — gender, ethnicity, and income gaps.",
                "algo": "random_forest", "algo_label": "Random Forest",
                "algo_reason": "Feature importance clearly shows which CV attributes drive hiring decisions — auditable for HR compliance.",
                "perspective": "Industry",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "benchmarks": True, "impact": False, "longitudinal": False,
                    "federated": False, "compliance": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.25, "n_runs": 3},
                "focus_metrics": ["gender_outcome_gap", "intersectional_worst_gap", "accuracy"],
                "role_brief": "You are reviewing your company's AI hiring tool before the next FCCPC audit. Which protected attributes are driving rejection rates?",
            },
            "Trade Union Officer": {
                "icon": "⚒️", "desc": "Worker impact — gig economy fairness, wage suppression index.",
                "algo": "voting_soft", "algo_label": "Soft Voting Ensemble",
                "algo_reason": "Ensemble reduces single-model bias — critical when presenting findings to arbitration boards where model reliability is challenged.",
                "perspective": "Worker Impact",
                "tab_visibility": {"performance": False, "equity": True, "xai": True,
                    "benchmarks": True, "impact": True, "longitudinal": True,
                    "federated": False, "compliance": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.45, "n_runs": 3,
                    "selected_biases": ["socioeconomic", "geographic", "historical"]},
                "focus_metrics": ["wage_suppression_index", "informal_sector_gap", "demographic_parity"],
                "role_brief": "You are building an evidence base for wage discrimination arbitration. Show the economic harm in concrete terms: how much money do affected workers lose?",
            },
        },
        "perspectives": ["Industry", "Research", "Worker Impact"],
        "perspective_help": {
            "Industry": "KPI dashboard — fairness score, outcome gaps, key alerts.",
            "Research": "Full statistical depth — distributions, CIs, benchmarks.",
            "Worker Impact": "Focus on wage suppression, automation risk, and exclusion.",
        },
    },

    # ── Judicial ───────────────────────────────────────────────────────────────
    "judicial": {
        "roles": {
            "Data Scientist": {
                "icon": "🔬", "desc": "Full interface — all tabs, technical metrics, and benchmarks.",
                "algo": "hist_gradient_boosting", "algo_label": "Hist Gradient Boosting",
                "algo_reason": "Fast iteration for research. Provides clean SHAP values for judicial fairness analysis.",
                "perspective": "Research",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "benchmarks": True, "compliance": True, "longitudinal": True, "raw": True},
                "sidebar_defaults": {"bias_intensity": 0.3, "n_runs": 3},
                "focus_metrics": ["fairness_score", "racial_fpr_gap", "liberty_score"],
            },
            "Defence Counsel": {
                "icon": "⚖️", "desc": "Defendant rights — false positive rate and intersectional bias.",
                "algo": "calibrated_hgb", "algo_label": "Calibrated HGB",
                "algo_reason": "Calibrated probabilities produce reliable risk scores. Overconfident models are especially dangerous in bail decisions.",
                "perspective": "Rights Audit",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "benchmarks": True, "compliance": False, "longitudinal": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.35, "n_runs": 3,
                    "selected_biases": ["demographic", "historical", "socioeconomic"]},
                "focus_metrics": ["racial_fpr_gap", "minority_fpr", "liberty_score"],
                "role_brief": "You are challenging this predictive policing AI in court. Build the statistical evidence: is your client's demographic group flagged at a disproportionate rate?",
            },
            "Judicial Officer": {
                "icon": "🔨", "desc": "Court oversight — bias audit, COMPAS comparison, liberty score.",
                "algo": "balanced_hgb", "algo_label": "Balanced HGB (Fairness-Weighted)",
                "algo_reason": "Equal error rates across demographic groups is the legal standard. Balanced training is the minimum threshold for judicial admissibility.",
                "perspective": "Rights Audit",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "benchmarks": True, "compliance": True, "longitudinal": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.15, "n_runs": 5},
                "focus_metrics": ["racial_fpr_gap", "liberty_score", "equalized_odds"],
                "role_brief": "You are reviewing whether this AI meets the NJC standard for judicial AI fairness. The key threshold: FPR demographic gap must be below 10 percentage points.",
            },
            "Regulator (NJC)": {
                "icon": "🏛️", "desc": "NJC / NASS compliance — fairness audit and due process verdict.",
                "algo": "voting_soft", "algo_label": "Soft Voting Ensemble",
                "algo_reason": "Ensemble models are more robust to individual model quirks — important for regulatory use where consistency matters.",
                "perspective": "Policy Brief",
                "tab_visibility": {"performance": True, "equity": True, "xai": False,
                    "benchmarks": False, "compliance": True, "longitudinal": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.1, "n_runs": 5},
                "focus_metrics": ["fairness_score", "demographic_parity", "liberty_score"],
            },
            "Rights Monitor": {
                "icon": "🌍", "desc": "Human rights focus — race, poverty, and geographic bias.",
                "algo": "gradient_boosting", "algo_label": "Gradient Boosting",
                "algo_reason": "GBT captures the interaction between race × poverty × geography — the intersectional harm pattern documented by Amnesty International.",
                "perspective": "Rights Audit",
                "tab_visibility": {"performance": False, "equity": True, "xai": True,
                    "benchmarks": True, "compliance": False, "longitudinal": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.5, "n_runs": 3,
                    "selected_biases": ["demographic", "geographic", "historical", "socioeconomic"]},
                "focus_metrics": ["racial_fpr_gap", "demographic_parity", "liberty_score"],
                "role_brief": "You are documenting human rights violations by this predictive policing system for a UN Special Rapporteur report. Quantify the harm to marginalised communities.",
            },
        },
        "perspectives": ["Industry", "Research", "Rights Audit"],
        "perspective_help": {
            "Industry": "KPI dashboard — accuracy, fairness score, key alerts.",
            "Research": "Full statistical depth — distributions, CIs, benchmarks.",
            "Rights Audit": "Focus on defendant rights — false positive rates and liberty score.",
        },
    },

    # ── National Security ──────────────────────────────────────────────────────
    "security": {
        "roles": {
            "Data Scientist": {
                "icon": "🔬", "desc": "Full interface — all tabs, adversarial attacks, and benchmarks.",
                "algo": "hist_gradient_boosting", "algo_label": "Hist Gradient Boosting",
                "algo_reason": "Robust to adversarial noise and data poisoning. Handles class imbalance in threat detection datasets.",
                "perspective": "Research",
                "tab_visibility": {"performance": True, "equity": True, "redteam": True,
                    "compliance": True, "longitudinal": True, "raw": True},
                "sidebar_defaults": {"bias_intensity": 0.3, "n_runs": 3},
                "focus_metrics": ["detection_rate", "liberty_score", "false_positive_rate"],
            },
            "Security Analyst": {
                "icon": "🛡️", "desc": "Threat detection focus — detection rate, false alarm, liberty score.",
                "algo": "voting_soft", "algo_label": "Soft Voting Ensemble",
                "algo_reason": "Ensemble reduces false alarm rate (a critical operational metric) through model diversity — each model's false positives partially cancel out.",
                "perspective": "Industry",
                "tab_visibility": {"performance": True, "equity": True, "redteam": True,
                    "compliance": False, "longitudinal": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.2, "n_runs": 3,
                    "enable_redteam": True},
                "focus_metrics": ["detection_rate", "false_positive_rate", "liberty_score"],
                "role_brief": "You are deploying this threat detection AI. Your priority: maximise detection rate while keeping false alarm rate below 5%. What is the demographic toll?",
            },
            "Civil Liberties Officer": {
                "icon": "⚖️", "desc": "Rights impact — demographic parity, false positive rate by group.",
                "algo": "balanced_hgb", "algo_label": "Balanced HGB (Fairness-Weighted)",
                "algo_reason": "Equalised false positive rates across demographic groups is the civil liberties standard for surveillance AI.",
                "perspective": "Rights Audit",
                "tab_visibility": {"performance": True, "equity": True, "redteam": False,
                    "compliance": True, "longitudinal": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.4, "n_runs": 3,
                    "selected_biases": ["demographic", "geographic"]},
                "focus_metrics": ["liberty_score", "false_positive_rate", "demographic_parity"],
                "role_brief": "You are auditing this surveillance AI for civil liberties violations. The NSA requires liberty_score > 0.70. Is this system lawful?",
            },
            "Policy Maker": {
                "icon": "🏛️", "desc": "Oversight focus — compliance report and governance layer.",
                "algo": "random_forest", "algo_label": "Random Forest",
                "algo_reason": "Interpretable to non-technical policymakers and parliamentary committees. SHAP values can be cited in policy briefs.",
                "perspective": "Policy Brief",
                "tab_visibility": {"performance": True, "equity": True, "redteam": False,
                    "compliance": True, "longitudinal": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.15, "n_runs": 3,
                    "enable_governance": True},
                "focus_metrics": ["liberty_score", "fairness_score", "detection_rate"],
            },
            "Field Intelligence": {
                "icon": "🕵️", "desc": "Operational view — prediction outcome and scenario briefing.",
                "algo": "gradient_boosting", "algo_label": "Gradient Boosting",
                "algo_reason": "Fast, accurate, and robust to adversarial noise in field-collected intelligence data.",
                "perspective": "Industry",
                "tab_visibility": {"performance": True, "equity": False, "redteam": True,
                    "compliance": False, "longitudinal": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.25, "n_runs": 1},
                "focus_metrics": ["detection_rate", "false_positive_rate", "accuracy"],
            },
        },
        "perspectives": ["Industry", "Research", "Rights Audit"],
        "perspective_help": {
            "Industry": "KPI dashboard — detection rate, fairness score, key alerts.",
            "Research": "Full statistical depth — distributions, CIs, benchmarks.",
            "Rights Audit": "Focus on civil liberties — false positive rates and liberty score.",
        },
    },

    # ── Education ──────────────────────────────────────────────────────────────
    "education": {
        "roles": {
            "Data Scientist": {
                "icon": "🔬", "desc": "Full interface — all tabs, technical metrics, and benchmarks.",
                "algo": "hist_gradient_boosting", "algo_label": "Hist Gradient Boosting",
                "algo_reason": "Best performance on JAMB-style tabular data with mixed numeric and categorical features.",
                "perspective": "Research",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "compliance": True, "longitudinal": True, "raw": True},
                "sidebar_defaults": {"bias_intensity": 0.3, "n_runs": 3},
                "focus_metrics": ["fairness_score", "demographic_parity", "accuracy"],
            },
            "Educator": {
                "icon": "📚", "desc": "Equity focus — gender gap, urban-rural divide, coaching access bias.",
                "algo": "balanced_hgb", "algo_label": "Balanced HGB (Fairness-Weighted)",
                "algo_reason": "Corrects for the urban coaching advantage that inflates scores for already-privileged students.",
                "perspective": "Student Impact",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "compliance": False, "longitudinal": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.35, "n_runs": 3,
                    "selected_biases": ["geographic", "socioeconomic", "demographic"]},
                "focus_metrics": ["demographic_parity", "gender_gap", "equity_gap"],
                "role_brief": "You are a teacher in a rural LGA. Your students score identically on practice tests but get different AI admission scores. Show the urban-rural gap.",
            },
            "Regulator (NUC)": {
                "icon": "🏛️", "desc": "NUC / JAMB compliance — fairness audit and accreditation standards.",
                "algo": "calibrated_hgb", "algo_label": "Calibrated HGB",
                "algo_reason": "Calibrated probabilities allow NUC to set defensible cut-off thresholds with known error rates.",
                "perspective": "Policy Brief",
                "tab_visibility": {"performance": True, "equity": True, "xai": False,
                    "compliance": True, "longitudinal": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.1, "n_runs": 5},
                "focus_metrics": ["fairness_score", "demographic_parity", "accuracy"],
            },
            "School Principal": {
                "icon": "🎓", "desc": "Institution view — student outcomes and admission fairness.",
                "algo": "random_forest", "algo_label": "Random Forest",
                "algo_reason": "Interpretable feature importance helps school leaders understand which student characteristics the AI penalises.",
                "perspective": "Industry",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "compliance": True, "longitudinal": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.25, "n_runs": 3},
                "focus_metrics": ["accuracy", "fairness_score", "equity_gap"],
            },
            "Student Advocate": {
                "icon": "🙋", "desc": "Access equity — who is excluded and why, in plain language.",
                "algo": "balanced_hgb", "algo_label": "Balanced HGB (Fairness-Weighted)",
                "algo_reason": "Demonstrates what fair admissions would look like — the counterfactual that student advocates need.",
                "perspective": "Student Impact",
                "tab_visibility": {"performance": False, "equity": True, "xai": True,
                    "compliance": False, "longitudinal": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.5, "n_runs": 3,
                    "selected_biases": ["demographic", "geographic", "socioeconomic"]},
                "focus_metrics": ["demographic_parity", "equity_gap", "gender_gap"],
                "role_brief": "You are fighting for students excluded by JAMB's AI scoring. Build the evidence: who is systematically excluded and what would fair scoring look like?",
            },
        },
        "perspectives": ["Industry", "Research", "Student Impact"],
        "perspective_help": {
            "Industry": "KPI dashboard — accuracy, fairness score, key alerts.",
            "Research": "Full statistical depth — distributions, CIs, benchmarks.",
            "Student Impact": "Focus on exclusion — which students bear the greatest harm.",
        },
    },

    # ── Financial Inclusion ────────────────────────────────────────────────────
    "financial": {
        "roles": {
            "Data Scientist": {
                "icon": "🔬", "desc": "Full interface — all tabs, technical metrics, and benchmarks.",
                "algo": "hist_gradient_boosting", "algo_label": "Hist Gradient Boosting",
                "algo_reason": "Native handling of missing values in credit data. Best AUC on EFInA financial inclusion datasets.",
                "perspective": "Research",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "compliance": True, "longitudinal": True, "raw": True},
                "sidebar_defaults": {"bias_intensity": 0.3, "n_runs": 3},
                "focus_metrics": ["fairness_score", "informal_sector_gap", "accuracy"],
            },
            "Credit Analyst": {
                "icon": "💳", "desc": "Approval rates, denial gaps, and income-level bias.",
                "algo": "gradient_boosting", "algo_label": "Gradient Boosting",
                "algo_reason": "GBT captures non-linear income × employment interactions that drive credit score bias for informal workers.",
                "perspective": "Industry",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "compliance": False, "longitudinal": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.3, "n_runs": 3},
                "focus_metrics": ["accuracy", "informal_sector_gap", "demographic_parity"],
                "role_brief": "You are reviewing your bank's credit scoring AI before the CBN examination. Is your denial rate for informal sector applicants defensible?",
            },
            "Regulator (CBN)": {
                "icon": "🏛️", "desc": "CBN / NDPC compliance — disparate impact ratio, ECOA alignment.",
                "algo": "calibrated_hgb", "algo_label": "Calibrated HGB",
                "algo_reason": "Calibration ensures the AI's risk scores are accurate across income groups — a CBN requirement for model validation.",
                "perspective": "Policy Brief",
                "tab_visibility": {"performance": True, "equity": True, "xai": False,
                    "compliance": True, "longitudinal": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.1, "n_runs": 5},
                "focus_metrics": ["fairness_score", "demographic_parity", "informal_sector_gap"],
            },
            "Bank Director": {
                "icon": "🏦", "desc": "Executive view — KPI dashboard and plain-language risk verdict.",
                "algo": "random_forest", "algo_label": "Random Forest",
                "algo_reason": "Explainable to board directors and auditors. Feature importance maps directly to credit policy levers.",
                "perspective": "Industry",
                "tab_visibility": {"performance": True, "equity": False, "xai": False,
                    "compliance": True, "longitudinal": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.2, "n_runs": 1},
                "focus_metrics": ["accuracy", "fairness_score", "fpr"],
            },
            "Consumer Advocate": {
                "icon": "🤝", "desc": "Borrower equity — informal sector exclusion and poverty trap risk.",
                "algo": "balanced_hgb", "algo_label": "Balanced HGB (Fairness-Weighted)",
                "algo_reason": "Equal error rates across income groups prevents the poverty trap cycle where AI denials lock out the poor permanently.",
                "perspective": "Consumer Impact",
                "tab_visibility": {"performance": False, "equity": True, "xai": True,
                    "compliance": False, "longitudinal": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.5, "n_runs": 3,
                    "selected_biases": ["socioeconomic", "demographic", "historical"]},
                "focus_metrics": ["informal_sector_gap", "demographic_parity", "fpr"],
                "role_brief": "You are representing informal workers denied credit. Show how the AI perpetuates the poverty trap: those who need credit most are denied at the highest rate.",
            },
        },
        "perspectives": ["Industry", "Research", "Consumer Impact"],
        "perspective_help": {
            "Industry": "KPI dashboard — approval gaps, fairness score, key alerts.",
            "Research": "Full statistical depth — distributions, CIs, benchmarks.",
            "Consumer Impact": "Focus on who is excluded — informal workers, low-income customers.",
        },
    },

    # ── Disinformation ─────────────────────────────────────────────────────────
    "disinformation": {
        "roles": {
            "Data Scientist": {
                "icon": "🔬", "desc": "Full interface — all tabs, technical metrics, and benchmarks.",
                "algo": "hist_gradient_boosting", "algo_label": "Hist Gradient Boosting",
                "algo_reason": "Handles high-dimensional text features. Best F1 on Hausa/Yoruba/Igbo mixed-language corpora.",
                "perspective": "Research",
                "tab_visibility": {"performance": True, "equity": True, "redteam": True,
                    "compliance": True, "longitudinal": True, "raw": True},
                "sidebar_defaults": {"bias_intensity": 0.3, "n_runs": 3,
                    "enable_redteam": True},
                "focus_metrics": ["accuracy", "fairness_score", "language_fpr_gap"],
            },
            "Platform Policy": {
                "icon": "📡", "desc": "Moderation performance — precision, recall, language FPR gap.",
                "algo": "voting_soft", "algo_label": "Soft Voting Ensemble",
                "algo_reason": "Ensemble reduces language-specific FPR variance — critical for platform policy where a single language over-removal causes viral backlash.",
                "perspective": "Industry",
                "tab_visibility": {"performance": True, "equity": True, "redteam": True,
                    "compliance": False, "longitudinal": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.25, "n_runs": 3,
                    "enable_redteam": True},
                "focus_metrics": ["accuracy", "language_fpr_gap", "speech_suppression_risk"],
                "role_brief": "You are Meta's Nigeria policy lead 30 days before the 2027 election. Your moderation AI is flagging Hausa content at 24% higher rate. Fix it before election day.",
            },
            "Election Monitor": {
                "icon": "🗳️", "desc": "Election integrity — speech suppression risk and language equity.",
                "algo": "calibrated_hgb", "algo_label": "Calibrated HGB",
                "algo_reason": "Calibrated probabilities allow INEC monitors to set defensible removal thresholds with known type-I error rates per language.",
                "perspective": "Democracy Audit",
                "tab_visibility": {"performance": True, "equity": True, "redteam": False,
                    "compliance": True, "longitudinal": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.3, "n_runs": 5,
                    "selected_biases": ["linguistic", "demographic"]},
                "focus_metrics": ["language_fpr_gap", "speech_suppression_risk", "fairness_score"],
                "role_brief": "You are an INEC/EU election observer. Is this platform's moderation AI suppressing political speech in minority languages? Quantify the democratic harm.",
            },
            "Regulator (INEC)": {
                "icon": "🏛️", "desc": "INEC / DSA compliance — content moderation fairness verdict.",
                "algo": "balanced_hgb", "algo_label": "Balanced HGB (Fairness-Weighted)",
                "algo_reason": "Equal removal rates across languages is the DSA Article 34 requirement. Balanced training is the standard solution.",
                "perspective": "Policy Brief",
                "tab_visibility": {"performance": True, "equity": True, "redteam": False,
                    "compliance": True, "longitudinal": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.1, "n_runs": 5},
                "focus_metrics": ["fairness_score", "language_fpr_gap", "demographic_parity"],
            },
            "Journalist": {
                "icon": "📰", "desc": "Press freedom focus — over-removal rate and political bias.",
                "algo": "gradient_boosting", "algo_label": "Gradient Boosting",
                "algo_reason": "GBT's partial dependence plots show exactly which linguistic features trigger removal — directly quotable in investigative reporting.",
                "perspective": "Democracy Audit",
                "tab_visibility": {"performance": False, "equity": True, "redteam": True,
                    "compliance": False, "longitudinal": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.45, "n_runs": 3,
                    "selected_biases": ["linguistic", "historical", "demographic"]},
                "focus_metrics": ["language_fpr_gap", "speech_suppression_risk", "over_removal_rate"],
                "role_brief": "You are an investigative journalist for Stears or Premium Times. Prove that this platform's AI systematically suppresses Hausa and Yoruba political speech.",
            },
        },
        "perspectives": ["Industry", "Research", "Democracy Audit"],
        "perspective_help": {
            "Industry": "KPI dashboard — precision, recall, speech suppression, key alerts.",
            "Research": "Full statistical depth — distributions, CIs, benchmarks.",
            "Democracy Audit": "Focus on electoral fairness — language gaps and voter suppression risk.",
        },
    },

    # ── Agrotech ───────────────────────────────────────────────────────────────
    "agrotech": {
        "roles": {
            "Data Scientist": {
                "icon": "🔬", "desc": "Full interface — all tabs, technical metrics, and benchmarks.",
                "algo": "hist_gradient_boosting", "algo_label": "Hist Gradient Boosting",
                "algo_reason": "Handles climate × soil × connectivity interactions natively. Best performance on Plateau State NASC calibrated data.",
                "perspective": "Research",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "compliance": True, "longitudinal": True, "raw": True},
                "sidebar_defaults": {"bias_intensity": 0.3, "n_runs": 3},
                "focus_metrics": ["fairness_score", "gender_outcome_gap", "demographic_parity"],
            },
            "Agronomist": {
                "icon": "🌾", "desc": "Crop risk focus — climate, market access, and irrigation equity.",
                "algo": "random_forest", "algo_label": "Random Forest",
                "algo_reason": "Feature importance directly maps to agronomic variables (rainfall, soil type, distance to market) — interpretable to extension officers.",
                "perspective": "Farmer Impact",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "compliance": False, "longitudinal": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.3, "n_runs": 3,
                    "selected_biases": ["geographic", "socioeconomic"]},
                "focus_metrics": ["accuracy", "demographic_parity", "gender_outcome_gap"],
                "role_brief": "You are a NASC agronomist reviewing AI crop advisory for Plateau State. Does the system recommend the same resources to female and male farmers with identical plots?",
            },
            "Regulator (NASC)": {
                "icon": "🏛️", "desc": "NASC / NITDA compliance — fairness audit and SDG2 alignment.",
                "algo": "balanced_hgb", "algo_label": "Balanced HGB (Fairness-Weighted)",
                "algo_reason": "SDG2 (Zero Hunger) requires equitable resource distribution. Balanced training is the NASC standard for AI-assisted allocation.",
                "perspective": "Policy Brief",
                "tab_visibility": {"performance": True, "equity": True, "xai": False,
                    "compliance": True, "longitudinal": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.1, "n_runs": 5},
                "focus_metrics": ["fairness_score", "gender_outcome_gap", "demographic_parity"],
            },
            "Extension Officer": {
                "icon": "🌿", "desc": "Field view — farmer outcomes and actionable guidance.",
                "algo": "decision_tree", "algo_label": "Decision Tree",
                "algo_reason": "A single decision tree produces a rule-based output that extension officers can explain to farmers without a smartphone or internet.",
                "perspective": "Farmer Impact",
                "tab_visibility": {"performance": True, "equity": True, "xai": True,
                    "compliance": False, "longitudinal": False, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.2, "n_runs": 1},
                "focus_metrics": ["accuracy", "demographic_parity", "gender_outcome_gap"],
                "role_brief": "You work with smallholder farmers in rural Plateau State. The AI recommends resources — but your female farmers say they receive fewer recommendations than men.",
            },
            "Gender Auditor": {
                "icon": "👩‍🌾", "desc": "Gender equity — female farmer access and UNESCO SDG audit.",
                "algo": "calibrated_hgb", "algo_label": "Calibrated HGB",
                "algo_reason": "Calibrated probabilities allow precise measurement of the gender probability gap — required for UNESCO Women4EthicalAI certification.",
                "perspective": "Farmer Impact",
                "tab_visibility": {"performance": False, "equity": True, "xai": True,
                    "compliance": True, "longitudinal": True, "raw": False},
                "sidebar_defaults": {"bias_intensity": 0.4, "n_runs": 5,
                    "enable_gender_audit": True,
                    "selected_biases": ["gender", "demographic", "geographic"]},
                "focus_metrics": ["gender_outcome_gap", "demographic_parity", "fairness_score"],
                "role_brief": "You are conducting a UNESCO Women4EthicalAI audit. 52% of Nigerian farmers are women. Is this AI system certified to serve them equitably?",
            },
        },
        "perspectives": ["Industry", "Research", "Farmer Impact"],
        "perspective_help": {
            "Industry": "KPI dashboard — fairness score, market access gaps, key alerts.",
            "Research": "Full statistical depth — distributions, CIs, benchmarks.",
            "Farmer Impact": "Focus on smallholder equity — gender, income, and connectivity gaps.",
        },
    },
}


# ── Helper: get domain from page name ─────────────────────────────────────────
def _detect_domain(page_domain_hint: str) -> str:
    """Map a hint string to a DOMAIN_ROLES key."""
    mapping = {
        "health":        "health",
        "healthcare":    "health",
        "economic":      "economic",
        "econ":          "economic",
        "security":      "security",
        "education":     "education",
        "edu":           "education",
        "financial":     "financial",
        "finance":       "financial",
        "fin":           "financial",
        "judicial":      "judicial",
        "justice":       "judicial",
        "disinformation":"disinformation",
        "disinfo":       "disinformation",
        "agrotech":      "agrotech",
        "agro":          "agrotech",
    }
    return mapping.get(page_domain_hint.lower(), "health")



def role_switcher(domain: str) -> None:
    """
    Render a domain-specific role/persona selector in the sidebar.
    Shows roles relevant to the current module (e.g. Clinician for Healthcare,
    Labour Economist for Economic Justice, Defence Counsel for Judicial).

    Call this FIRST in the sidebar, before any other controls.
    """
    mapped = _detect_domain(domain)
    cfg    = DOMAIN_ROLES.get(mapped, DOMAIN_ROLES["health"])
    roles  = list(cfg["roles"].keys())

    role_key = f"_active_role_{mapped}"
    if role_key not in st.session_state or st.session_state[role_key] not in roles:
        st.session_state[role_key] = roles[0]

    current = st.session_state[role_key]

    st.markdown(
        "<p style='font-size:.78rem;font-weight:600;"
        "color:var(--color-text-secondary);margin-bottom:3px;'>👁 View as</p>",
        unsafe_allow_html=True,
    )

    selected = st.selectbox(
        "Role",
        roles,
        index=roles.index(current) if current in roles else 0,
        format_func=lambda r: (cfg.get("roles",{}).get(r,{}).get("icon","") + " " + r),
        key=f"_role_select_{mapped}",
        label_visibility="collapsed",
    )

    if selected != current:
        st.session_state[role_key] = selected
        st.rerun()

    # Show description under the selector
    desc = cfg["roles"].get(selected, {}).get("desc", "")
    st.markdown(
        f"<p style='font-size:.72rem;color:var(--color-text-tertiary);"
        f"font-style:italic;margin:.2rem 0 0;'>{desc}</p>",
        unsafe_allow_html=True,
    )


def get_active_role(domain: str) -> str:
    """Return the currently active role for the given domain."""
    return st.session_state.get(f"_active_role_{domain}", "Data Scientist")



def get_role_config(domain: str) -> dict:
    """Return the full config dict for the currently active role in a domain."""
    mapped = _detect_domain(domain)
    cfg    = DOMAIN_ROLES.get(mapped, DOMAIN_ROLES["health"])
    roles  = cfg.get("roles", {})
    active = get_active_role(mapped)
    return roles.get(active, next(iter(roles.values()), {}))


def get_role_algo(domain: str) -> tuple:
    """
    Return (algo_key, algo_label, algo_reason) for the active role.
    algo_key maps to _build_clf() in pages that have an algorithm factory.
    """
    cfg = get_role_config(domain)
    return (
        cfg.get("algo",       "hist_gradient_boosting"),
        cfg.get("algo_label", "Hist Gradient Boosting"),
        cfg.get("algo_reason","Optimal for this role's analytical goals."),
    )


def get_role_tabs(domain: str) -> dict:
    """Return the tab_visibility dict for the active role."""
    cfg = get_role_config(domain)
    return cfg.get("tab_visibility", {})


def get_role_defaults(domain: str) -> dict:
    """Return sidebar default values (bias_intensity, n_runs, etc.) for the active role."""
    cfg = get_role_config(domain)
    return cfg.get("sidebar_defaults", {})


def get_role_brief(domain: str) -> str:
    """Return the scenario brief / role framing text for the active role."""
    cfg = get_role_config(domain)
    return cfg.get("role_brief", "")


def role_algo_banner(domain: str) -> None:
    """
    Render a compact algorithm recommendation card in the sidebar.
    Shows the recommended algorithm for the active role with reasoning.
    Placed just above the run button.
    """
    algo_key, algo_label, algo_reason = get_role_algo(domain)
    role = get_active_role(_detect_domain(domain))

    if role == "Data Scientist":
        return  # Data Scientist chooses manually

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#f0fdf4,#dcfce7);"
        f"border:1px solid #86efac;border-left:4px solid #16a34a;"
        f"border-radius:8px;padding:10px 12px;margin:8px 0;'>"
        f"<p style='font-size:.68rem;font-weight:700;color:#166534;"
        f"text-transform:uppercase;letter-spacing:.05em;margin:0 0 4px;'>"
        f"🤖 Recommended Algorithm</p>"
        f"<p style='font-size:.82rem;font-weight:700;color:#0f172a;margin:0 0 4px;'>"
        f"{algo_label}</p>"
        f"<p style='font-size:.71rem;color:#374151;margin:0;line-height:1.4;'>"
        f"{algo_reason}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def role_brief_banner(domain: str) -> None:
    """
    Render a scenario brief / role framing banner below the page header.
    Only shown for non-Data-Scientist roles that have a role_brief set.
    """
    brief = get_role_brief(domain)
    if not brief:
        return
    role = get_active_role(_detect_domain(domain))
    icon = _ROLE_ICONS.get(role, "👤")
    mapped = _detect_domain(domain)
    cfg  = DOMAIN_ROLES.get(mapped, {})
    roles_cfg = cfg.get("roles", {})
    role_cfg  = roles_cfg.get(role, {})
    desc = role_cfg.get("desc", "")

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#1e3a5f,#0c4a6e);"
        f"border-radius:10px;padding:12px 16px;margin:0 0 14px;'>"
        f"<p style='color:#7dd3fc;font-size:.68rem;font-weight:700;"
        f"letter-spacing:.08em;text-transform:uppercase;margin:0 0 4px;'>"
        f"{icon} YOUR ROLE: {role.upper()}</p>"
        f"<p style='color:#f1f5f9;font-size:.88rem;font-weight:500;"
        f"margin:0 0 6px;line-height:1.4;'>{brief}</p>"
        f"<p style='color:#94a3b8;font-size:.73rem;margin:0;font-style:italic;'>{desc}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def role_banner(domain: str) -> None:
    """
    Render a subtle banner below the page header showing the active role
    and a quick-switch button. Non-Data-Scientist roles see a simplified notice.
    """
    role = get_active_role(domain)
    if role == "Data Scientist":
        return  # No banner needed for full view

    role_key = f"_active_role_{domain}"
    icon      = _ROLE_ICONS.get(role, "👤")
    domain_colors = {"health":"#2980b9","security":"#c0392b","agrotech":"#27ae60"}
    color     = domain_colors.get(domain, "#8e44ad")

    st.markdown(
        f"<div style='background:var(--color-background-secondary);"
        f"border:0.5px solid var(--color-border-secondary);"
        f"border-radius:var(--border-radius-md);padding:.5rem 1rem;"
        f"margin-bottom:.75rem;display:flex;align-items:center;gap:.6rem;'>"
        f"<span style='font-size:.85rem;'>{icon}</span>"
        f"<span style='font-size:.83rem;color:var(--color-text-secondary);'>"
        f"Viewing as <strong style='color:{color};'>{role}</strong> — "
        f"some tabs and controls are hidden. Switch role in the sidebar.</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def board_member_summary(
    domain: str,
    avg_accuracy: float,
    avg_fairness: float,
    compliant: bool,
    key_finding: str,
    recommendation: str,
) -> None:
    """
    Render an executive-level summary card for the Board Member role.
    Replaces the full dashboard with a concise 4-item verdict.
    """
    domain_colors = {"health":"#2980b9","security":"#c0392b","agrotech":"#27ae60"}
    color  = domain_colors.get(domain, "#8e44ad")
    status = "✅ READY FOR REVIEW" if compliant and avg_fairness >= 0.7 else "⚠️ REQUIRES ATTENTION"
    st_col = "#27ae60" if "READY" in status else "#e74c3c"

    st.markdown(
        f"<div style='border:1.5px solid {color}44;"
        f"border-radius:var(--border-radius-lg);padding:1.5rem 2rem;"
        f"background:var(--color-background-primary);'>"

        f"<p style='font-size:.8rem;font-weight:600;color:{color};"
        f"text-transform:uppercase;letter-spacing:.05em;margin:0 0 .4rem;'>"
        f"Executive Summary</p>"

        f"<p style='font-size:1.4rem;font-weight:700;color:{st_col};margin:0 0 1rem;'>"
        f"{status}</p>"

        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:1rem;"
        f"margin-bottom:1rem;'>"

        f"<div style='background:var(--color-background-secondary);"
        f"border-radius:var(--border-radius-md);padding:.75rem 1rem;'>"
        f"<p style='font-size:.75rem;color:var(--color-text-secondary);margin:0;'>Prediction Accuracy</p>"
        f"<p style='font-size:1.6rem;font-weight:700;color:var(--color-text-primary);margin:0;'>"
        f"{avg_accuracy:.0%}</p></div>"

        f"<div style='background:var(--color-background-secondary);"
        f"border-radius:var(--border-radius-md);padding:.75rem 1rem;'>"
        f"<p style='font-size:.75rem;color:var(--color-text-secondary);margin:0;'>Fairness Score</p>"
        f"<p style='font-size:1.6rem;font-weight:700;"
        f"color:{'#27ae60' if avg_fairness >= 0.7 else '#e74c3c'};margin:0;'>"
        f"{avg_fairness:.2f} / 1.0</p></div>"

        f"</div>"

        f"<p style='font-size:.85rem;font-weight:500;"
        f"color:var(--color-text-primary);margin:0 0 .3rem;'>Key Finding</p>"
        f"<p style='font-size:.85rem;color:var(--color-text-secondary);"
        f"margin:0 0 .75rem;'>{key_finding}</p>"

        f"<p style='font-size:.85rem;font-weight:500;"
        f"color:var(--color-text-primary);margin:0 0 .3rem;'>Recommended Action</p>"
        f"<p style='font-size:.85rem;color:var(--color-text-secondary);margin:0;'>"
        f"{recommendation}</p>"

        f"</div>",
        unsafe_allow_html=True,
    )
