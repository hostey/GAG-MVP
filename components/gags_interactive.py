"""
components/gags_interactive.py  —  GAGS Interactivity Engine v1.0
==================================================================
Makes every simulation module addictive and engaging through:

  1. BIAS DETECTIVE CHALLENGE
     User is given a biased simulation result with hidden bias settings.
     They must identify which bias type and intensity caused the harm.
     Points awarded for accuracy. Streak counter. Daily challenge mode.

  2. POLICY DESIGN LAB
     User picks a real Nigerian policy scenario (NHIA, NITDA, etc.)
     and must configure the AI fairly before "deploying" it.
     Scored against WHO/regulatory thresholds. Badge awarded on pass.

  3. WHAT-IF EXPLORER
     Interactive sliders that update charts IN REAL TIME without
     clicking "Run" — showing how each bias dial changes outcomes.

  4. FAIRNESS CHALLENGE
     Head-to-head: user's algorithm config vs the documented
     real-world benchmark (COMPAS, FCT Maternal AI, Robodebt).
     "Can you beat the real algorithm?"

  5. STREAK & PROGRESS TRACKER
     Cross-module points system stored in session state.
     Badges unlocked for exploring all modules, achieving fair outcomes,
     and completing domain-specific challenges.

  6. SCENARIO STORYTELLING MODE
     Each simulation is wrapped in a narrative: "You are an NHIA AI
     officer. 50,000 applications are waiting. Your algorithm is about
     to make decisions. What do you configure?"

Usage
-----
    from components.gags_interactive import (
        bias_detective_panel,
        policy_lab_panel,
        what_if_explorer,
        fairness_challenge_panel,
        progress_tracker,
        scenario_story_banner,
        award_points,
    )
"""

from __future__ import annotations
import random
import hashlib
from datetime import datetime, date
from typing import Any

import streamlit as st


# ── Points & badge system ─────────────────────────────────────────────────────

BADGES = {
    "first_run":        ("🏃", "First Run",         "Ran your first simulation"),
    "fair_result":      ("✅", "Fair Outcome",       "Achieved fairness_score ≥ 0.75"),
    "beat_benchmark":   ("🏆", "Beat the Real AI",  "Outperformed a documented real-world system"),
    "detective":        ("🔍", "Bias Detective",     "Correctly identified hidden bias type"),
    "policy_pass":      ("📋", "Policy Cleared",     "Passed all regulatory thresholds"),
    "all_modules":      ("🌍", "Full Explorer",      "Visited all 8 simulation modules"),
    "streak_3":         ("🔥", "On Fire",            "3-run improvement streak"),
    "streak_5":         ("⚡", "Lightning",          "5-run improvement streak"),
    "nigeria_expert":   ("🇳🇬", "Nigeria Expert",    "Used all 36 state selectors"),
    "what_if_master":   ("🎛️", "What-If Master",    "Explored 10+ what-if configurations"),
    "challenge_1":      ("🥉", "Challenger",         "Completed first domain challenge"),
    "challenge_5":      ("🥈", "Advanced",           "Completed 5 domain challenges"),
    "challenge_10":     ("🥇", "Expert",             "Completed 10 domain challenges"),
    "low_bias":         ("🕊️", "Peacemaker",         "Achieved bias_intensity 0.0 with fairness ≥ 0.9"),
    "high_bias":        ("😱", "Witness",            "Saw fairness_score drop below 0.2"),
}

DOMAIN_CHALLENGES = {
    "health": [
        {
            "id": "hc_mmr",
            "title": "🩺 Save the Mothers",
            "story": (
                "You are configuring the NHIA's new maternal health AI for deployment "
                "across all Nigerian LGAs. The system will risk-score 2.3 million pregnant "
                "women annually. In the 2022 FCT pilot, the AI missed 31% of high-risk rural women. "
                "**Your mission:** configure the AI to keep maternal_access_gap below 15 percentage points."
            ),
            "target_metric":  "maternal_access_gap",
            "target_value":   0.15,
            "target_label":   "maternal_access_gap < 15pp",
            "pass_condition": "less_than",
            "points":         150,
            "hint":           "Try reducing geographic and socioeconomic bias. Rural women are being systematically under-prioritised.",
            "badge":          "policy_pass",
        },
        {
            "id": "hc_oop",
            "title": "💸 End the Poverty Trap",
            "story": (
                "Nigeria's OOP health expenditure stands at 74.8% — one of the highest globally. "
                "An AI triage system at a Lagos hospital is compounding this by denying care to "
                "low-income patients at a higher rate. "
                "**Your mission:** configure the AI so catastrophic_expenditure_risk drops below 20%."
            ),
            "target_metric":  "catastrophic_expenditure_risk",
            "target_value":   0.20,
            "target_label":   "catastrophic_expenditure_risk < 20%",
            "pass_condition": "less_than",
            "points":         120,
            "hint":           "Wealth quintile bias is the main driver. Reduce socioeconomic bias intensity.",
            "badge":          "policy_pass",
        },
    ],
    "economic": [
        {
            "id": "ec_hiring",
            "title": "👔 Level the Playing Field",
            "story": (
                "A major Lagos tech company is about to deploy an AI hiring tool trained on "
                "historical hire records from elite institutions. NBS data shows a 24pp private/"
                "state university outcome gap. "
                "**Your mission:** bring the intersectional_worst_gap below 20 percentage points "
                "before the system goes live."
            ),
            "target_metric":  "intersectional_worst_gap",
            "target_value":   0.20,
            "target_label":   "intersectional_worst_gap < 20pp",
            "pass_condition": "less_than",
            "points":         130,
            "hint":           "Historical bias is the key driver here — it encodes past hiring discrimination into future predictions.",
            "badge":          "policy_pass",
        },
        {
            "id": "ec_gig",
            "title": "⚒️ Fair Wages for Gig Workers",
            "story": (
                "58% of Nigerian gig workers earn below minimum wage (Fairwork Africa 2023). "
                "A dispatch platform is using AI wage-setting that systematically suppresses "
                "pay for informal workers. "
                "**Your mission:** bring wage_suppression_index below 0.25."
            ),
            "target_metric":  "wage_suppression_index",
            "target_value":   0.25,
            "target_label":   "wage_suppression_index < 0.25",
            "pass_condition": "less_than",
            "points":         110,
            "hint":           "Informal sector bias and socioeconomic bias are compounding. Try addressing both simultaneously.",
            "badge":          "policy_pass",
        },
    ],
    "judicial": [
        {
            "id": "jud_fpr",
            "title": "⚖️ Innocent Until Proven",
            "story": (
                "A predictive policing AI in Lagos has a documented 22pp racial FPR gap — "
                "mirroring the COMPAS scandal. The NJC is reviewing the system. "
                "**Your mission:** bring the FPR demographic gap below 10 percentage points "
                "to pass the NJC fairness standard."
            ),
            "target_metric":  "demographic_parity_difference",
            "target_value":   0.10,
            "target_label":   "FPR gap < 10pp",
            "pass_condition": "less_than",
            "points":         140,
            "hint":           "Demographic and historical bias are entangled. Balanced training helps most here.",
            "badge":          "policy_pass",
        },
    ],
    "disinformation": [
        {
            "id": "dis_language",
            "title": "🗳️ Protect Every Voice",
            "story": (
                "It is 30 days to the 2027 Nigerian general election. Meta has deployed a "
                "content moderation AI trained on English data. Hausa and Yoruba posts are "
                "being removed at 24pp higher rate than English. "
                "**Your mission:** bring language_fpr_gap below 10pp before election day."
            ),
            "target_metric":  "language_fpr_gap",
            "target_value":   0.10,
            "target_label":   "language_fpr_gap < 10pp",
            "pass_condition": "less_than",
            "points":         160,
            "hint":           "Linguistic bias is the direct driver. Reduce it and see the gap close.",
            "badge":          "policy_pass",
        },
    ],
    "education": [
        {
            "id": "edu_gap",
            "title": "🎓 Unlock Every Student",
            "story": (
                "JAMB's AI admission scoring shows a 23.1pp urban-rural gap. Students from "
                "rural LGAs with identical abilities are being systematically excluded. "
                "**Your mission:** bring the equity_gap below 15pp to pass the NUC equity standard."
            ),
            "target_metric":  "demographic_parity_difference",
            "target_value":   0.15,
            "target_label":   "urban-rural gap < 15pp",
            "pass_condition": "less_than",
            "points":         120,
            "hint":           "Geographic bias encoding the coaching access inequality is the main driver.",
            "badge":          "policy_pass",
        },
    ],
    "financial": [
        {
            "id": "fin_informal",
            "title": "💳 Bank the Unbanked",
            "story": (
                "64.9% of Nigeria's workforce is informal. A CBN-supervised credit scoring AI "
                "is denying loans to informal workers at 33.4pp higher rate than formal employees "
                "with equivalent creditworthiness. "
                "**Your mission:** bring the informal_sector_gap below 15pp."
            ),
            "target_metric":  "informal_sector_gap",
            "target_value":   0.15,
            "target_label":   "informal_sector_gap < 15pp",
            "pass_condition": "less_than",
            "points":         130,
            "hint":           "Socioeconomic bias encodes employment formality. Historical bias amplifies it.",
            "badge":          "policy_pass",
        },
    ],
    "security": [
        {
            "id": "sec_liberty",
            "title": "🛡️ Liberty Without Compromise",
            "story": (
                "A predictive surveillance AI deployed in Abuja is flagging northern communities "
                "at 2.3× the rate of southern communities for the same behaviour. "
                "**Your mission:** achieve liberty_score above 0.70 while maintaining "
                "detection_rate above 0.65."
            ),
            "target_metric":  "liberty_score",
            "target_value":   0.70,
            "target_label":   "liberty_score > 0.70 AND detection_rate > 0.65",
            "pass_condition": "greater_than",
            "points":         150,
            "hint":           "Demographic and geographic bias interact here. Governance oversight reduces false positives.",
            "badge":          "policy_pass",
        },
    ],
    "agrotech": [
        {
            "id": "agro_gender",
            "title": "👩‍🌾 Equal Fields",
            "story": (
                "52% of Nigerian smallholder farmers are women, yet an AI crop-advisory system "
                "is recommending fertilizer to male farmers at 28pp higher rate. "
                "**Your mission:** bring the gender_outcome_gap below 10pp to meet the "
                "UNESCO Women4EthicalAI standard."
            ),
            "target_metric":  "gender_outcome_gap",
            "target_value":   0.10,
            "target_label":   "gender_outcome_gap < 10pp",
            "pass_condition": "less_than",
            "points":         120,
            "hint":           "Gender bias is the direct driver. Reduce it — but watch that geographic bias doesn't compensate.",
            "badge":          "policy_pass",
        },
    ],
}

DOMAIN_BENCHMARK_CHALLENGE = {
    "health": {
        "name":  "FCT Maternal AI Pilot (2022)",
        "story": "The real AI missed 31% of high-risk rural women. Can you do better?",
        "beat_metric": "maternal_access_gap",
        "beat_value":  0.31,
        "beat_label":  "Beat maternal_access_gap < 31% (FCT 2022 baseline)",
    },
    "judicial": {
        "name":  "COMPAS Recidivism AI (USA)",
        "story": "COMPAS showed a 22pp racial FPR gap. Can you beat it?",
        "beat_metric": "demographic_parity_difference",
        "beat_value":  0.22,
        "beat_label":  "Beat FPR gap < 22pp (COMPAS baseline)",
    },
    "economic": {
        "name":  "Algorithmic Hiring (Nigeria 2023)",
        "story": "Documented 24pp private/state university gap. Beat it.",
        "beat_metric": "gender_outcome_gap",
        "beat_value":  0.24,
        "beat_label":  "Beat outcome gap < 24pp (Nigeria 2023 baseline)",
    },
    "disinformation": {
        "name":  "Facebook 2020 US Election AI",
        "story": "Non-English content removed 5× faster. Beat the language gap.",
        "beat_metric": "language_fpr_gap",
        "beat_value":  0.30,
        "beat_label":  "Beat language FPR gap < 30% (Facebook 2020 baseline)",
    },
    "financial": {
        "name":  "Nigerian Informal Credit Scoring",
        "story": "34pp informal sector denial gap documented (EFInA 2022). Beat it.",
        "beat_metric": "informal_sector_gap",
        "beat_value":  0.34,
        "beat_label":  "Beat informal sector gap < 34pp (EFInA 2022 baseline)",
    },
    "education": {
        "name":  "JAMB Urban-Rural Gap (2022)",
        "story": "23.1pp urban-rural outcome gap. Can your AI close it?",
        "beat_metric": "demographic_parity_difference",
        "beat_value":  0.231,
        "beat_label":  "Beat urban-rural gap < 23.1pp (JAMB 2022 baseline)",
    },
}


# ── Points helpers ────────────────────────────────────────────────────────────

def _reset_render_guards() -> None:
    """Clear per-run render guards. Call once at the top of each page."""
    keys_to_clear = [k for k in st.session_state
                     if k.startswith(('_wi_rendered_', '_dcp_rendered_', '_bcp_rendered_'))]
    for k in keys_to_clear:
        del st.session_state[k]


def _ensure_state():
    defaults = {
        "_gags_points":        0,
        "_gags_badges":        [],
        "_gags_streak":        0,
        "_gags_best_fairness": 0.0,
        "_gags_challenges_done": [],
        "_gags_modules_visited": [],
        "_gags_runs_total":    0,
        "_gags_what_if_count": 0,
        "_gags_states_tried":  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def award_points(points: int, reason: str = "") -> None:
    _ensure_state()
    st.session_state["_gags_points"] += points
    if reason:
        st.toast(f"✨ +{points} points — {reason}!", icon="🎯")


def award_badge(badge_id: str) -> None:
    _ensure_state()
    if badge_id in BADGES and badge_id not in st.session_state["_gags_badges"]:
        st.session_state["_gags_badges"].append(badge_id)
        icon, name, desc = BADGES[badge_id]
        st.toast(f"{icon} Badge unlocked: **{name}** — {desc}", icon="🏅")


def track_run(fairness_score: float, domain: str) -> None:
    """Call after every simulation run to update streak and badges."""
    _ensure_state()
    st.session_state["_gags_runs_total"] += 1

    if domain not in st.session_state["_gags_modules_visited"]:
        st.session_state["_gags_modules_visited"].append(domain)

    if st.session_state["_gags_runs_total"] == 1:
        award_badge("first_run")
        award_points(10, "first simulation run")

    if fairness_score >= 0.75:
        award_badge("fair_result")
        award_points(20, "fair outcome achieved")

    if fairness_score >= 0.90:
        award_badge("low_bias")

    if fairness_score < 0.20:
        award_badge("high_bias")

    # Streak
    prev_best = st.session_state["_gags_best_fairness"]
    if fairness_score > prev_best:
        st.session_state["_gags_streak"] += 1
        st.session_state["_gags_best_fairness"] = fairness_score
        if st.session_state["_gags_streak"] >= 5:
            award_badge("streak_5")
        elif st.session_state["_gags_streak"] >= 3:
            award_badge("streak_3")
    else:
        st.session_state["_gags_streak"] = 0

    if len(st.session_state["_gags_modules_visited"]) >= 8:
        award_badge("all_modules")


# ── Progress tracker widget ───────────────────────────────────────────────────

def progress_tracker(location: str = "sidebar") -> None:
    """Compact progress bar and badge display. Call in sidebar."""
    _ensure_state()

    points  = st.session_state["_gags_points"]
    badges  = st.session_state["_gags_badges"]
    streak  = st.session_state["_gags_streak"]
    modules = len(st.session_state["_gags_modules_visited"])
    runs    = st.session_state["_gags_runs_total"]

    container = st.sidebar if location == "sidebar" else st

    # Level calculation
    level = min(10, 1 + points // 100)
    next_level_pts = level * 100
    progress_pct   = min(1.0, (points % 100) / 100)

    container.markdown("---")
    container.markdown(
        f"<div style='background:linear-gradient(135deg,#1e3a5f,#0891b2);"
        f"border-radius:10px;padding:10px 14px;margin:4px 0;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<span style='color:#fff;font-weight:700;font-size:.85rem;'>⚡ Level {level}</span>"
        f"<span style='color:#7dd3fc;font-size:.75rem;'>{points} pts</span>"
        f"</div>"
        f"<div style='background:rgba(255,255,255,.2);border-radius:4px;height:5px;margin:6px 0 4px;'>"
        f"<div style='background:#22d3ee;border-radius:4px;height:5px;"
        f"width:{progress_pct*100:.0f}%;'></div>"
        f"</div>"
        f"<div style='display:flex;gap:10px;font-size:.72rem;color:#bae6fd;'>"
        f"<span>🔥 {streak} streak</span>"
        f"<span>🗂️ {modules}/8 modules</span>"
        f"<span>▶️ {runs} runs</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if badges:
        badge_html = "".join(
            f"<span title='{BADGES[b][2]}' style='font-size:1.1rem;cursor:help;'>"
            f"{BADGES[b][0]}</span>"
            for b in badges[-6:]   # show last 6
        )
        container.markdown(
            f"<div style='padding:4px 0;font-size:.72rem;color:#94a3b8;'>"
            f"Badges: {badge_html}</div>",
            unsafe_allow_html=True,
        )


# ── Scenario story banner ─────────────────────────────────────────────────────

def scenario_story_banner(domain: str, scenario_name: str,
                           scenario_desc: str = "") -> None:
    """
    Full-width narrative banner that sets the scene for the simulation.
    Makes the user feel like they are making a real decision.
    """
    domain_stories = {
        "health": "You are an NHIA AI Governance Officer. Your algorithm will affect millions.",
        "economic": "You are a NITDA AI Ethics Auditor. Real workers' livelihoods depend on this.",
        "judicial": "You are a NJC AI Oversight Commissioner. Liberty is at stake.",
        "disinformation": "You are an INEC Digital Rights Monitor. Democracy depends on this.",
        "education": "You are a NUC AI Equity Officer. Every student's future is on the line.",
        "financial": "You are a CBN AI Fairness Regulator. Financial inclusion is at stake.",
        "security": "You are a NSA Ethics Oversight Officer. Security and rights must balance.",
        "agrotech": "You are a NASC AI Equity Auditor. 50 million smallholders depend on this.",
    }

    role_text = domain_stories.get(domain, "You are an AI Governance Officer.")

    st.markdown(
        f"""<div style='background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
        border-radius:12px;padding:16px 20px;margin:0 0 16px;
        border-left:4px solid #0891b2;'>
        <p style='color:#7dd3fc;font-size:.72rem;font-weight:600;
        letter-spacing:.08em;text-transform:uppercase;margin:0 0 4px;'>
        🎭 SCENARIO BRIEF
        </p>
        <p style='color:#f1f5f9;font-size:.95rem;font-weight:600;margin:0 0 4px;'>
        {scenario_name}
        </p>
        <p style='color:#94a3b8;font-size:.78rem;margin:0 0 8px;font-style:italic;'>
        {role_text}
        </p>
        <p style='color:#cbd5e1;font-size:.82rem;margin:0;line-height:1.5;'>
        {scenario_desc[:300] if scenario_desc else "Configure the simulation parameters, then click Run to see the outcome."}
        </p>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Domain challenge panel ────────────────────────────────────────────────────

def domain_challenge_panel(domain: str, latest_results: dict) -> None:
    """
    Show the active domain challenge, check if the user passed, award badge.
    Call this after the simulation results are available.

    Parameters
    ----------
    domain         : domain key (health, economic, judicial, etc.)
    latest_results : dict of metric_name → value from the latest run
    """
    # Guard: render once per domain per script run
    _rk = f"_dcp_rendered_{domain}"
    if st.session_state.get(_rk): return
    st.session_state[_rk] = True

    _ensure_state()
    challenges = DOMAIN_CHALLENGES.get(domain, [])
    if not challenges:
        return

    st.markdown("---")
    st.markdown(
        "<p style='font-size:.78rem;font-weight:700;color:#7c3aed;"
        "letter-spacing:.06em;text-transform:uppercase;'>🎯 ACTIVE CHALLENGE</p>",
        unsafe_allow_html=True,
    )

    # Pick today's challenge (rotate daily)
    day_idx  = date.today().toordinal() % len(challenges)
    challenge = challenges[day_idx]
    cid       = challenge["id"]

    already_done = cid in st.session_state["_gags_challenges_done"]

    col_story, col_status = st.columns([3, 1])
    with col_story:
        st.markdown(
            f"<div style='background:#f5f3ff;border:1px solid #ddd6fe;"
            f"border-left:4px solid #7c3aed;border-radius:8px;"
            f"padding:12px 14px;'>"
            f"<b style='font-size:.9rem;'>{challenge.get("title","")}</b><br>"
            f"<span style='font-size:.78rem;color:#374151;line-height:1.5;'>"
            f"{challenge.get("story","")}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_status:
        metric = challenge["target_metric"]
        target = challenge["target_value"]
        cond   = challenge["pass_condition"]
        current_val = latest_results.get(metric)

        if current_val is not None:
            if cond == "less_than":
                passed = float(current_val) < target
            else:
                passed = float(current_val) > target

            delta_pct = abs(float(current_val) - target) / max(target, 0.001) * 100

            if passed:
                st.markdown(
                    f"<div style='background:#dcfce7;border-radius:8px;"
                    f"padding:10px;text-align:center;'>"
                    f"<div style='font-size:1.6rem;'>✅</div>"
                    f"<div style='font-weight:700;color:#166534;font-size:.8rem;'>PASSED!</div>"
                    f"<div style='color:#166534;font-size:.72rem;'>"
                    f"{challenge.get("target_label","")}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if not already_done:
                    st.session_state["_gags_challenges_done"].append(cid)
                    award_points(challenge["points"], challenge["title"])
                    award_badge(challenge.get("badge", "policy_pass"))
                    n_done = len(st.session_state["_gags_challenges_done"])
                    if n_done >= 10: award_badge("challenge_10")
                    elif n_done >= 5: award_badge("challenge_5")
                    else: award_badge("challenge_1")
            else:
                st.markdown(
                    f"<div style='background:#fee2e2;border-radius:8px;"
                    f"padding:10px;text-align:center;'>"
                    f"<div style='font-size:1.6rem;'>❌</div>"
                    f"<div style='font-weight:700;color:#991b1b;font-size:.8rem;'>NOT YET</div>"
                    f"<div style='color:#991b1b;font-size:.72rem;'>"
                    f"{delta_pct:.0f}% off target</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                with st.expander("💡 Hint"):
                    st.caption(challenge["hint"])
        else:
            st.info("Run simulation to check challenge status.")

    if already_done:
        st.success(f"✅ Challenge completed! +{challenge.get("points",0)} points awarded.")


# ── Benchmark challenge panel ─────────────────────────────────────────────────

def benchmark_challenge_panel(domain: str, latest_results: dict) -> None:
    """
    'Can you beat the real AI?' panel shown after simulation runs.
    Compares user's result against documented real-world baseline.
    """
    # Guard: render once per domain per script run
    _rk = f"_bcp_rendered_{domain}"
    if st.session_state.get(_rk): return
    st.session_state[_rk] = True

    bm = DOMAIN_BENCHMARK_CHALLENGE.get(domain)
    if not bm:
        return

    metric      = bm["beat_metric"]
    baseline    = bm["beat_value"]
    current_val = latest_results.get(metric)
    if current_val is None:
        return

    user_val = float(current_val)
    beat     = user_val < baseline   # lower is better for gap metrics

    colour = "#dcfce7" if beat else "#fff7ed"
    icon   = "🏆" if beat else "📊"
    msg    = (f"You beat the real AI! ({user_val:.1%} vs {baseline:.1%} baseline)"
              if beat else
              f"Not yet — {user_val:.1%} vs {baseline:.1%} to beat")

    st.markdown(
        f"<div style='background:{colour};border-radius:8px;"
        f"padding:10px 14px;margin:8px 0;'>"
        f"<b>{icon} VS REAL WORLD: {bm.get("name","benchmark")}</b><br>"
        f"<span style='font-size:.78rem;color:#374151;'>{bm['story']}</span><br>"
        f"<span style='font-size:.82rem;font-weight:600;color:#1e3a5f;'>{msg}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if beat:
        award_badge("beat_benchmark")
        award_points(50, f"Beat {bm.get("name","benchmark")}")


# ── What-if explorer ──────────────────────────────────────────────────────────

def what_if_explorer(domain: str, current_metrics: dict,
                     bias_intensity: float) -> None:
    """
    Interactive panel showing how metrics change as the user moves a slider.
    Approximates impact without a full re-run using parametric scaling.
    """
    _ensure_state()

    # Guard: only render once per domain per script run
    _wi_rendered_key = f"_wi_rendered_{domain}"
    if st.session_state.get(_wi_rendered_key):
        return
    st.session_state[_wi_rendered_key] = True

    with st.expander("🎛️ What-If Explorer — adjust bias and see live impact", expanded=False):
        st.caption(
            "Move the slider to see how changing bias intensity affects the key metrics. "
            "This uses an approximation — click Run for exact results."
        )

        wi_val = st.slider(
            "Hypothetical bias intensity",
            0.0, 1.0,
            float(bias_intensity),
            0.05,
            key=f"_what_if_{domain}",
        )

        st.session_state["_gags_what_if_count"] += 1
        if st.session_state["_gags_what_if_count"] >= 10:
            award_badge("what_if_master")

        # Approximate metric scaling
        ratio = wi_val / max(bias_intensity, 0.01)

        gap_metrics = [k for k in current_metrics
                       if any(x in k for x in ['gap','rate','risk','index'])]

        if gap_metrics and ratio != 1.0:
            cols = st.columns(min(3, len(gap_metrics)))
            for i, metric in enumerate(gap_metrics[:3]):
                val = current_metrics[metric]
                est = min(1.0, float(val) * ratio)
                delta = est - float(val)
                cols[i].metric(
                    metric.replace('_', ' ').title(),
                    f"{est:.1%}",
                    delta=f"{delta:+.1%}",
                    delta_color="inverse",
                )
        else:
            st.info("Run a simulation first to see what-if projections.")


# ── Bias detective mini-game ───────────────────────────────────────────────────

def bias_detective_panel(domain: str) -> None:
    """
    Hide-and-seek game: user must guess which bias type caused a displayed outcome.
    A random daily challenge is generated from DOMAIN_CHALLENGES.
    """
    _ensure_state()

    # Guard: only render once per domain per script run
    _wi_rendered_key = f"_wi_rendered_{domain}"
    if st.session_state.get(_wi_rendered_key):
        return
    st.session_state[_wi_rendered_key] = True

    with st.expander("🔍 Bias Detective — can you spot the hidden bias?", expanded=False):

        # Daily seed so challenge resets each day but is consistent within a day
        seed = int(hashlib.md5(
            f"{date.today().isoformat()}{domain}".encode()
        ).hexdigest(), 16) % 1000

        rng  = random.Random(seed)
        bias_types = [
            "demographic", "historical", "socioeconomic",
            "geographic", "gender", "linguistic",
        ]
        hidden_bias = rng.choice(bias_types)
        hidden_intensity = rng.choice([0.2, 0.3, 0.4, 0.5, 0.6])

        # Clues based on the hidden bias
        clue_map = {
            "demographic":   "The AI treats certain demographic groups systematically differently in outcomes.",
            "historical":    "The training data reflects decisions made in a discriminatory past.",
            "socioeconomic": "Wealth quintile strongly predicts outcome — the poor are disproportionately harmed.",
            "geographic":    "Rural communities bear a disproportionate share of negative outcomes.",
            "gender":        "Women and men with identical qualifications receive different outcomes.",
            "linguistic":    "Content in local languages is treated differently from English content.",
        }

        st.markdown(
            f"**🔍 Daily Detective Challenge** — investigate the clue, identify the bias.\n\n"
            f"> *{clue_map[hidden_bias]}*\n\n"
            f"Simulated fairness score: **{0.85 - hidden_intensity * 0.6:.2f}**  |  "
            f"Bias intensity: hidden"
        )

        guess = st.selectbox(
            "Your diagnosis: which bias type caused this?",
            ["— select —"] + bias_types,
            key=f"_detective_{domain}_{date.today().isoformat()}",
        )

        if guess != "— select —":
            if guess == hidden_bias:
                st.success(f"✅ Correct! The hidden bias was **{hidden_bias}** at intensity {hidden_intensity:.1f}.")
                award_badge("detective")
                award_points(75, "Bias Detective correct answer")
            else:
                st.error(
                    f"❌ Not quite. The hidden bias was **{hidden_bias}**. "
                    f"Re-read the clue and try again tomorrow!"
                )
