"""
components/gags_feature_modules.py  —  GAGS Feature Modules v2.0
================================================================
State-of-the-art renderer for all 5 GAGS feature modules with:

  • Rich interactive UI for each module's results
  • Domain-specific "How to Use" guides per module
  • Admin-editable multi-challenge system (stored in session_state)
  • Module-specific explanations, tooltips, and narrative
  • Expandable deep-dive sections per module

THE 5 FEATURE MODULES
---------------------
  1. AI Agent Economy Sandbox       — resource auctions, autonomous bidding
  2. Multimodal Red Teaming         — text/image/deepfake adversarial attacks
  3. Africa-Centric / Gender Audit  — Nigeria presets + UNESCO equity audit
  4. Hybrid Governance Layer        — citizen vote + blockchain-style ledger
  5. Strategic Social Reasoning     — negotiation, coalition, deception game

ADMIN CHALLENGE SYSTEM
----------------------
Challenges are stored in st.session_state["_admin_challenges"][domain].
Admins access the management panel via a password-protected expander.
Default password: "gags2024" (change in ADMIN_PASSWORD below).
"""

from __future__ import annotations
import json
import random
from datetime import date
from typing import Any

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Admin config ───────────────────────────────────────────────────────────────
ADMIN_PASSWORD = "gags2024"   # change this in production

# ── Module metadata ────────────────────────────────────────────────────────────
MODULE_META = {
    "agent_economy": {
        "icon":  "💰",
        "title": "AI Agent Economy Sandbox",
        "tagline": "Autonomous agents compete for scarce resources — who gets left out?",
        "colour": "#0891b2",
        "description": (
            "Simulates a multi-agent resource auction where AI agents bid for scarce "
            "healthcare beds, fertilizer allocations, credit slots, or legal aid. "
            "Agents have different wealth, connectivity, and bargaining power — "
            "exposing how algorithmic resource allocation perpetuates structural inequality."
        ),
        "how_to_use": [
            "Enable **Agent Economy** in the sidebar before running the simulation.",
            "Run the simulation — agents will bid in a Vickrey (second-price) auction.",
            "Read the **Agent Summary** table: agents with lower wealth systematically "
            "lose out even with identical underlying need.",
            "Check the **Auction Results** chart: the Gini coefficient of resource "
            "distribution shows how unequal the allocation is.",
            "Try increasing **Socioeconomic bias** in the sidebar — watch the Gini "
            "coefficient rise as wealthy agents crowd out poor ones.",
            "The **fairness_score** drops when resource access correlates with wealth "
            "rather than need — this is the core bias this module measures.",
        ],
        "key_metrics": ["gini_coefficient", "allocation_equity", "bypass_rate"],
        "research_note": (
            "Based on Vickrey-Clarke-Groves mechanism design and multi-agent simulation "
            "literature. Calibrated to Nigerian healthcare resource scarcity (1.95 health "
            "workers per 10,000 population, WHO 2022)."
        ),
    },
    "multimodal_redteam": {
        "icon":  "🛡️",
        "title": "Multimodal Red Teaming",
        "tagline": "Attack the AI across text, image, and deepfake vectors — find its weakest link.",
        "colour": "#dc2626",
        "description": (
            "Tests the AI system's robustness against three classes of adversarial attack: "
            "(1) text injection — embedding triggering phrases in inputs; "
            "(2) adversarial image perturbation — pixel-level perturbations that fool "
            "vision models; (3) deepfake attacks — synthetic media that bypasses "
            "content moderation. Each attack has a severity score and bypass rate."
        ),
        "how_to_use": [
            "Enable **Multimodal Red Team** in the sidebar before running.",
            "The module automatically tests three attack vectors: Text, Image, and Deepfake.",
            "Read the **Modality Results** table: bypass rate is the fraction of attacks "
            "that successfully fooled the AI.",
            "High **Sociotechnical Risk** (>0.6) means the attack could be deployed "
            "at scale by an adversary with moderate resources.",
            "The **VR/AR Scenario** panel shows the real-world consequence of a successful "
            "deepfake attack in context (e.g. a synthetic ministerial video).",
            "Use this to understand: which attack surface is your AI most vulnerable to?",
        ],
        "key_metrics": ["combined_bypass_rate", "combined_sociotechnical_risk"],
        "research_note": (
            "Red teaming methodology based on Anthropic (2023), NIST AI RMF Adversarial "
            "ML taxonomy, and Stuber et al. (2023) multimodal attack surfaces. "
            "Deepfake calibration uses AfricaCheck 2023 synthetic media incident data."
        ),
    },
    "governance": {
        "icon":  "🏛️",
        "title": "Hybrid Governance Layer",
        "tagline": "Democratic oversight meets blockchain accountability — can citizens control the AI?",
        "colour": "#7c3aed",
        "description": (
            "Models a hybrid AI governance system where: (1) a citizen assembly votes "
            "on AI deployment policies; (2) every AI decision is recorded in an immutable "
            "blockchain-style audit ledger; (3) a policy engine applies the voted rules "
            "in real time. Shows what happens when governance is weak or strong."
        ),
        "how_to_use": [
            "Enable **Governance Layer** in the sidebar (on by default).",
            "The module runs automatically alongside the main simulation.",
            "Read the **Policy Votes** panel: which policies did the citizen assembly adopt?",
            "Check the **Audit Ledger**: every flagged decision is logged with timestamp, "
            "decision type, and the rule that triggered the flag.",
            "The **AI Flags** counter shows how many decisions were caught by governance rules.",
            "Increase **Oversight Mechanism** intensity to see how stronger governance "
            "reduces harmful outcomes — at some cost to efficiency.",
            "This module answers: does governance actually change AI behaviour, or is it "
            "just compliance theatre?",
        ],
        "key_metrics": ["ai_flags", "policy_votes", "ledger_entries"],
        "research_note": (
            "Governance model based on the EU AI Act Article 9 risk management requirements, "
            "NITDA AI Policy 2023, and Ada Lovelace Institute (2023) citizen participation "
            "in AI governance. Blockchain ledger simulates NDPC audit trail requirements."
        ),
    },
    "gender_audit": {
        "icon":  "👩",
        "title": "Gender Equity Audit (UNESCO Women4EthicalAI)",
        "tagline": "Measure the gender gap precisely — then close it.",
        "colour": "#16a34a",
        "description": (
            "Conducts a structured gender equity audit aligned to the UNESCO "
            "Recommendation on the Ethics of AI (2021) and the Women4EthicalAI platform. "
            "Measures: overall gender gap, representation ratio, digital inclusion score, "
            "and the compound penalty for women in multiple disadvantaged categories."
        ),
        "how_to_use": [
            "Enable **Gender Equity Audit** in the sidebar before running.",
            "After the simulation runs, go to the **Equity** or **Feature Modules** tab.",
            "Read the **Gender Gap** metric: this is the raw outcome difference between "
            "men and women with identical non-gender characteristics.",
            "The **Representation Score** measures whether women are proportionally "
            "represented in the positive-outcome group (target: ≥0.85).",
            "The **Digital Inclusion Score** measures whether the AI performs equally "
            "for women with limited digital footprint (low BVN, low connectivity).",
            "The **UNESCO Verdict** panel gives a pass/fail against the Women4EthicalAI "
            "threshold of gender_gap < 0.10.",
            "Try: set gender bias to 0.0 — does the gap disappear? If not, historical "
            "bias is encoding past discrimination into future predictions.",
        ],
        "key_metrics": ["overall_gender_gap", "representation_score", "digital_inclusion_score"],
        "research_note": (
            "Audit framework aligned to UNESCO Recommendation on the Ethics of AI (2021), "
            "Women4EthicalAI platform indicators, and WEF Global Gender Gap Report 2023. "
            "Nigeria calibration: 52% of smallholder farmers are female (FAO 2023); "
            "female BVN coverage 12pp below male (EFInA 2022)."
        ),
    },
    "arena": {
        "icon":  "⚔️",
        "title": "Strategic Social Reasoning Arena",
        "tagline": "Agents negotiate, deceive, and form coalitions — does AI play fair?",
        "colour": "#d97706",
        "description": (
            "A game-theoretic simulation where AI agents interact through negotiation, "
            "coalition formation, and strategic deception. Reveals how AI systems behave "
            "when they have strategic incentives — and whether those incentives align "
            "with fair outcomes for all participants."
        ),
        "how_to_use": [
            "Enable **Strategic Arena** in the sidebar before running.",
            "The arena runs a multi-round negotiation game between agent groups.",
            "Read the **Final Standings** table: which agent group won, and what "
            "strategies did they use?",
            "High **Deception Rate** means agents are finding it profitable to misrepresent "
            "their true preferences — a sign the AI reward structure is misaligned.",
            "The **Coalition Matrix** shows which groups allied — often, disadvantaged "
            "groups form coalitions but are outbid by wealthier individual agents.",
            "Use this module to test whether your AI's incentive structure encourages "
            "fair cooperation or adversarial behaviour.",
        ],
        "key_metrics": ["coalition_stability", "deception_rate", "social_welfare"],
        "research_note": (
            "Based on mechanism design and social choice theory (Arrow, 1951; Myerson, 1981). "
            "Coalition dynamics calibrated to Nigerian stakeholder analysis from NITDA "
            "AI Policy consultations 2022-2023."
        ),
    },
}

# ── Default challenge bank (admin can extend via UI) ──────────────────────────
DEFAULT_CHALLENGES: dict[str, list[dict]] = {
    "health": [
        {
            "id": "hc_mmr_1", "title": "🩺 Protect Every Mother",
            "story": "The NHIA maternal AI missed 31% of high-risk rural women in the 2022 FCT pilot. Configure the AI to keep maternal_access_gap below 15pp.",
            "target_metric": "maternal_access_gap", "target_value": 0.15,
            "pass_condition": "less_than", "points": 150,
            "hint": "Reduce geographic and socioeconomic bias together.",
            "difficulty": "Medium", "badge": "policy_pass",
        },
        {
            "id": "hc_oop_1", "title": "💸 End the Poverty Trap",
            "story": "An AI triage system is pushing 23% of poor households into catastrophic OOP spending. Bring catastrophic_expenditure_risk below 20%.",
            "target_metric": "catastrophic_expenditure_risk", "target_value": 0.20,
            "pass_condition": "less_than", "points": 120,
            "hint": "Wealth quintile bias is the primary driver.",
            "difficulty": "Easy", "badge": "policy_pass",
        },
        {
            "id": "hc_nhis_1", "title": "🏥 Universal Coverage",
            "story": "Only 4.5% of Nigerians have NHIA coverage. The AI is denying informal workers at 28pp higher rate. Bring insurance_denial_gap below 20pp.",
            "target_metric": "insurance_denial_gap", "target_value": 0.20,
            "pass_condition": "less_than", "points": 130,
            "hint": "Informal sector bias is the key. Historical bias amplifies it.",
            "difficulty": "Medium", "badge": "policy_pass",
        },
        {
            "id": "hc_hard_1", "title": "⚡ Expert: Zero Tolerance",
            "story": "EXPERT CHALLENGE. Achieve fairness_score above 0.85 AND maternal_access_gap below 10pp simultaneously. This requires carefully balancing multiple bias types.",
            "target_metric": "fairness_score", "target_value": 0.85,
            "pass_condition": "greater_than", "points": 300,
            "hint": "Use Balanced HGB algorithm and reduce ALL bias types to below 0.15.",
            "difficulty": "Expert", "badge": "low_bias",
        },
    ],
    "economic": [
        {
            "id": "ec_hiring_1", "title": "👔 Level the Playing Field",
            "story": "A Lagos tech company's hiring AI has a 24pp private/state university gap. Bring intersectional_worst_gap below 20pp before deployment.",
            "target_metric": "intersectional_worst_gap", "target_value": 0.20,
            "pass_condition": "less_than", "points": 130,
            "hint": "Historical bias encodes past discrimination. Target it first.",
            "difficulty": "Medium", "badge": "policy_pass",
        },
        {
            "id": "ec_gig_1", "title": "⚒️ Fair Gig Wages",
            "story": "58% of gig workers earn below minimum wage. AI wage-setting suppresses informal worker pay. Bring wage_suppression_index below 0.25.",
            "target_metric": "wage_suppression_index", "target_value": 0.25,
            "pass_condition": "less_than", "points": 110,
            "hint": "Both informal sector and socioeconomic bias compound here.",
            "difficulty": "Easy", "badge": "policy_pass",
        },
        {
            "id": "ec_market_1", "title": "🛒 SME Visibility",
            "story": "Female-led SMEs get 34% less marketplace visibility than male-led ones with identical products. Bring gender_outcome_gap below 15pp.",
            "target_metric": "gender_outcome_gap", "target_value": 0.15,
            "pass_condition": "less_than", "points": 120,
            "hint": "Gender bias is the direct driver here.",
            "difficulty": "Easy", "badge": "policy_pass",
        },
        {
            "id": "ec_hard_1", "title": "⚡ Expert: Triple Zero",
            "story": "EXPERT. Achieve: informal_sector_gap < 15pp AND gender_outcome_gap < 10pp AND economic_inclusion_score > 0.80 simultaneously.",
            "target_metric": "economic_inclusion_score", "target_value": 0.80,
            "pass_condition": "greater_than", "points": 350,
            "hint": "Use Soft Voting ensemble and set demographic + historical bias to 0.0.",
            "difficulty": "Expert", "badge": "beat_benchmark",
        },
    ],
    "judicial": [
        {
            "id": "jud_fpr_1", "title": "⚖️ Innocent Until Proven",
            "story": "A Lagos predictive policing AI has a 22pp racial FPR gap mirroring COMPAS. Bring demographic_parity_difference below 10pp.",
            "target_metric": "demographic_parity_difference", "target_value": 0.10,
            "pass_condition": "less_than", "points": 140,
            "hint": "Demographic and historical bias are entangled. Balanced HGB helps.",
            "difficulty": "Medium", "badge": "policy_pass",
        },
        {
            "id": "jud_liberty_1", "title": "🕊️ Liberty Score",
            "story": "The NJC requires liberty_score above 0.70 for any AI used in bail decisions. Achieve this while keeping detection quality above 0.65.",
            "target_metric": "liberty_score", "target_value": 0.70,
            "pass_condition": "greater_than", "points": 130,
            "hint": "Enable Governance Layer — oversight reduces false positives.",
            "difficulty": "Medium", "badge": "policy_pass",
        },
    ],
    "disinformation": [
        {
            "id": "dis_lang_1", "title": "🗳️ Protect Every Language",
            "story": "30 days to the 2027 election. Hausa posts removed at 24pp higher rate than English. Bring language_fpr_gap below 10pp.",
            "target_metric": "language_fpr_gap", "target_value": 0.10,
            "pass_condition": "less_than", "points": 160,
            "hint": "Linguistic bias is the direct driver. Reduce it.",
            "difficulty": "Medium", "badge": "policy_pass",
        },
        {
            "id": "dis_spr_1", "title": "🗣️ Silence No One",
            "story": "Speech suppression risk is above 0.30 — meaning 30% of the moderation environment is hostile to legitimate expression. Bring SSR below 0.20.",
            "target_metric": "speech_suppression_risk", "target_value": 0.20,
            "pass_condition": "less_than", "points": 140,
            "hint": "Both over_removal_rate and language gap contribute to SSR.",
            "difficulty": "Medium", "badge": "policy_pass",
        },
    ],
    "education": [
        {
            "id": "edu_gap_1", "title": "🎓 Unlock Every Student",
            "story": "JAMB AI shows 23pp urban-rural gap. Students with identical ability are excluded by geography. Bring demographic_parity_difference below 15pp.",
            "target_metric": "demographic_parity_difference", "target_value": 0.15,
            "pass_condition": "less_than", "points": 120,
            "hint": "Geographic bias encoding coaching access inequality.",
            "difficulty": "Easy", "badge": "policy_pass",
        },
    ],
    "financial": [
        {
            "id": "fin_inf_1", "title": "💳 Bank the Unbanked",
            "story": "64.9% of Nigeria's workforce is informal. Credit AI denies them at 33pp higher rate. Bring informal_sector_gap below 15pp.",
            "target_metric": "informal_sector_gap", "target_value": 0.15,
            "pass_condition": "less_than", "points": 130,
            "hint": "Socioeconomic bias encodes employment formality.",
            "difficulty": "Medium", "badge": "policy_pass",
        },
    ],
    "security": [
        {
            "id": "sec_lib_1", "title": "🛡️ Liberty and Security",
            "story": "Surveillance AI flags northern communities at 2.3x the rate of southern ones. Achieve liberty_score > 0.70 while keeping detection_rate > 0.65.",
            "target_metric": "liberty_score", "target_value": 0.70,
            "pass_condition": "greater_than", "points": 150,
            "hint": "Demographic and geographic bias interact. Enable Governance Layer.",
            "difficulty": "Hard", "badge": "policy_pass",
        },
    ],
    "agrotech": [
        {
            "id": "agro_gen_1", "title": "👩‍🌾 Equal Fields",
            "story": "52% of Nigerian farmers are women, yet AI recommends fertilizer to men at 28pp higher rate. Bring gender_outcome_gap below 10pp.",
            "target_metric": "gender_outcome_gap", "target_value": 0.10,
            "pass_condition": "less_than", "points": 120,
            "hint": "Gender bias is the direct driver. Enable Gender Audit module.",
            "difficulty": "Easy", "badge": "policy_pass",
        },
    ],
}

DIFFICULTY_COLOURS = {
    "Easy":   ("#dcfce7", "#166534", "🟢"),
    "Medium": ("#fef3c7", "#92400e", "🟡"),
    "Hard":   ("#fee2e2", "#991b1b", "🔴"),
    "Expert": ("#ede9fe", "#5b21b6", "⚡"),
}


# ── Challenge state management ─────────────────────────────────────────────────

def _get_challenges(domain: str) -> list[dict]:
    if "_admin_challenges" not in st.session_state:
        st.session_state["_admin_challenges"] = {}
    if domain not in st.session_state["_admin_challenges"]:
        st.session_state["_admin_challenges"][domain] = \
            DEFAULT_CHALLENGES.get(domain, [])[:]
    return st.session_state["_admin_challenges"][domain]


def _save_challenges(domain: str, challenges: list[dict]) -> None:
    if "_admin_challenges" not in st.session_state:
        st.session_state["_admin_challenges"] = {}
    st.session_state["_admin_challenges"][domain] = challenges


# ── Feature module sidebar controls ───────────────────────────────────────────

def feature_module_sidebar(domain: str) -> dict:
    """
    Render the feature module toggle panel in the sidebar.
    Returns dict of enabled modules. Replaces raw st.toggle calls.
    """
    st.markdown(
        "<p style='font-size:.78rem;font-weight:700;color:#6d28d9;"
        "letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px;'>"
        "🔬 Feature Modules</p>",
        unsafe_allow_html=True,
    )

    # Domain-specific module availability
    available = _domain_modules(domain)

    enabled = {}
    for mod_key, (label, default, icon) in available.items():
        meta = MODULE_META.get(mod_key, {})
        col1, col2 = st.columns([4, 1])
        with col1:
            enabled[mod_key] = st.toggle(
                f"{icon} {label}",
                value=default,
                key=f"_fm_{domain}_{mod_key}",
                help=meta.get("tagline", ""),
            )
        with col2:
            if st.button("ℹ️", key=f"_fm_info_{domain}_{mod_key}",
                          help=f"About {label}"):
                st.session_state[f"_fm_show_info_{mod_key}"] = True

        if st.session_state.get(f"_fm_show_info_{mod_key}"):
            with st.expander(f"📖 About: {label}", expanded=True):
                _module_info_card(mod_key)
                if st.button("✕ Close", key=f"_close_info_{mod_key}"):
                    st.session_state[f"_fm_show_info_{mod_key}"] = False

    return enabled


def _domain_modules(domain: str) -> dict:
    """Return available feature modules for a given domain."""
    base = {
        "governance":   ("Governance Layer",   True,  "🏛️"),
        "gender_audit": ("Gender Equity Audit", False, "👩"),
    }
    domain_extras = {
        "health":   {
            "agent_economy":      ("Agent Economy",       False, "💰"),
            "multimodal_redteam": ("Multimodal Red Team", False, "🛡️"),
            "arena":              ("Strategic Arena",      False, "⚔️"),
        },
        "agrotech": {
            "agent_economy":      ("Agent Economy",       False, "💰"),
            "multimodal_redteam": ("Multimodal Red Team", False, "🛡️"),
            "arena":              ("Strategic Arena",      False, "⚔️"),
        },
        "economic": {
            "agent_economy":      ("Agent Economy",       False, "💰"),
        },
        "security": {
            "multimodal_redteam": ("Multimodal Red Team", False, "🛡️"),
            "arena":              ("Strategic Arena",      False, "⚔️"),
        },
        "disinformation": {
            "multimodal_redteam": ("Multimodal Red Team", True,  "🛡️"),
        },
    }
    result = dict(base)
    result.update(domain_extras.get(domain, {}))
    return result


def _module_info_card(mod_key: str) -> None:
    """Render a rich info card for one feature module."""
    meta = MODULE_META.get(mod_key, {})
    if not meta:
        st.info("No documentation available for this module.")
        return

    colour = meta.get("colour", "#334155")
    st.markdown(
        f"<div style='background:#f8fafc;border-left:4px solid {colour};"
        f"border-radius:8px;padding:12px 14px;'>"
        f"<b style='font-size:.95rem;color:#0f172a;'>"
        f"{meta.get('icon','')} {meta.get('title','')}</b><br>"
        f"<i style='color:#64748b;font-size:.8rem;'>{meta.get('tagline','')}</i><br><br>"
        f"<p style='color:#374151;font-size:.8rem;line-height:1.6;'>"
        f"{meta.get('description','')}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown("**📖 How to use this module:**")
    for j, step in enumerate(meta.get("how_to_use", []), 1):
        st.markdown(f"**{j}.** {step}")

    note = meta.get("research_note", "")
    if note:
        st.caption(f"📚 Research basis: {note}")


# ── Feature modules results tab ────────────────────────────────────────────────

def feature_modules_tab(domain: str, run_results: list[dict],
                         feats: dict | None = None,
                         xai: dict | None = None,
                         governance: dict | None = None,
                         gender_audit: Any = None,
                         agent_economy: dict | None = None,
                         arena: dict | None = None,
                         redteam: dict | None = None) -> None:
    """
    Render the complete Feature Modules tab with rich UI, how-to guides,
    and module-specific deep-dives.

    Call this inside `with tab_feature_modules:`.
    """
    feats = feats or {}
    enabled_any = any([governance, gender_audit, agent_economy, arena, redteam,
                       feats.get("multimodal_redteam"), feats.get("agent_economy")])

    if not enabled_any and not run_results:
        _empty_state_panel(domain)
        return

    # ── Module overview cards ──────────────────────────────────────────────────
    _module_overview_cards(domain, governance, gender_audit, agent_economy,
                           arena, redteam, feats)

    st.markdown("---")

    # ── Individual module results ──────────────────────────────────────────────
    if governance or feats.get("governance"):
        _render_governance(governance or feats.get("governance", {}))

    if gender_audit:
        _render_gender_audit(gender_audit)

    if agent_economy or feats.get("agent_economy"):
        _render_agent_economy(agent_economy or feats.get("agent_economy", {}))

    if redteam or feats.get("multimodal_redteam"):
        _render_redteam(redteam or feats.get("multimodal_redteam", {}))

    if arena or feats.get("arena"):
        _render_arena(arena or feats.get("arena", {}))

    if not enabled_any:
        _empty_state_panel(domain)


def _empty_state_panel(domain: str) -> None:
    modules = _domain_modules(domain)
    names   = [f"{icon} {label}" for label, _, icon in modules.values()]
    st.markdown(
        f"<div style='background:#f8fafc;border:2px dashed #e2e8f0;"
        f"border-radius:12px;padding:32px;text-align:center;'>"
        f"<div style='font-size:2.5rem;margin-bottom:8px;'>🔬</div>"
        f"<h3 style='color:#334155;margin:0 0 8px;'>Feature Modules Not Yet Run</h3>"
        f"<p style='color:#64748b;font-size:.88rem;max-width:480px;margin:0 auto 16px;'>"
        f"Enable one or more feature modules in the sidebar, then click "
        f"<b>Run Simulation</b> to see deep-dive results here.</p>"
        f"<div style='display:flex;flex-wrap:wrap;gap:8px;justify-content:center;'>"
        + "".join(
            f"<span style='background:#f1f5f9;border:1px solid #e2e8f0;"
            f"border-radius:20px;padding:4px 12px;font-size:.78rem;color:#475569;'>"
            f"{n}</span>"
            for n in names
        )
        + f"</div></div>",
        unsafe_allow_html=True,
    )

    # Inline how-to for each available module
    for mod_key in _domain_modules(domain):
        meta = MODULE_META.get(mod_key, {})
        if meta:
            with st.expander(
                f"{meta.get('icon','')} How to use: {meta.get('title','')}"
            ):
                _module_info_card(mod_key)


def _module_overview_cards(domain, governance, gender_audit, agent_economy,
                            arena, redteam, feats) -> None:
    """Render a row of status cards, one per active module."""
    active = []
    if governance:
        flags = len(governance.get("ai_flags", []))
        active.append(("🏛️", "Governance", f"{flags} flags raised", "#7c3aed"))
    if gender_audit:
        gap = getattr(gender_audit, "overall_gender_gap", 0)
        active.append(("👩", "Gender Audit",
                        f"Gap: {gap:.1%}", "#16a34a" if gap < 0.10 else "#dc2626"))
    if agent_economy or feats.get("agent_economy"):
        ae = agent_economy or feats.get("agent_economy", {})
        gini = ae.get("gini_coefficient", ae.get("allocation_gini", 0))
        active.append(("💰", "Agent Economy",
                        f"Gini: {gini:.2f}", "#0891b2"))
    if redteam or feats.get("multimodal_redteam"):
        rt = redteam or feats.get("multimodal_redteam", {})
        bypass = rt.get("combined_bypass_rate", 0)
        active.append(("🛡️", "Red Team",
                        f"Bypass: {bypass:.1%}", "#dc2626" if bypass > 0.3 else "#d97706"))
    if arena or feats.get("arena"):
        active.append(("⚔️", "Arena", "Coalition game complete", "#d97706"))

    if not active:
        return

    cols = st.columns(len(active))
    for col, (icon, name, stat, colour) in zip(cols, active):
        col.markdown(
            f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
            f"border-left:4px solid {colour};border-radius:8px;"
            f"padding:12px;text-align:center;'>"
            f"<div style='font-size:1.6rem;'>{icon}</div>"
            f"<div style='font-weight:700;font-size:.82rem;color:#0f172a;"
            f"margin:4px 0 2px;'>{name}</div>"
            f"<div style='font-size:.78rem;color:{colour};font-weight:600;'>"
            f"{stat}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_governance(gov: dict) -> None:
    if not gov:
        return
    with st.expander("🏛️ Governance Layer — Full Results", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("AI Flags Raised",   len(gov.get("ai_flags", [])))
        col2.metric("Policy Votes",       gov.get("policy_votes_count", "N/A"))
        col3.metric("Ledger Entries",     gov.get("ledger_entries", 0))

        flags = gov.get("ai_flags", [])
        if flags:
            st.markdown("**Flagged decisions:**")
            for flag in flags[:5]:
                st.markdown(
                    f"<div style='background:#fef9c3;border-left:3px solid #d97706;"
                    f"border-radius:4px;padding:6px 10px;margin:3px 0;"
                    f"font-size:.8rem;'>⚠️ {flag}</div>",
                    unsafe_allow_html=True,
                )
            if len(flags) > 5:
                st.caption(f"...and {len(flags)-5} more flags")

        narrative = gov.get("narrative", "")
        if narrative:
            st.info(narrative)

        with st.expander("📖 How to interpret governance results"):
            _module_info_card("governance")


def _render_gender_audit(ga) -> None:
    if not ga:
        return
    with st.expander("👩 Gender Equity Audit — Full Results", expanded=True):
        gap  = getattr(ga, "overall_gender_gap",    0)
        rep  = getattr(ga, "representation_score",  0)
        digi = getattr(ga, "digital_inclusion_score", 0)
        verdict_pass = gap < 0.10 and rep >= 0.85

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gender Gap",        f"{gap:.1%}",  delta_color="inverse")
        c2.metric("Representation",    f"{rep:.2f}",  delta_color="normal")
        c3.metric("Digital Inclusion", f"{digi:.2f}", delta_color="normal")
        c4.markdown(
            f"<div style='background:{'#dcfce7' if verdict_pass else '#fee2e2'};"
            f"border-radius:8px;padding:10px;text-align:center;margin-top:4px;'>"
            f"<b style='color:{'#166534' if verdict_pass else '#991b1b'};font-size:.88rem;'>"
            f"{'✅ UNESCO PASS' if verdict_pass else '❌ UNESCO FAIL'}</b></div>",
            unsafe_allow_html=True,
        )

        # Radar chart
        cats   = ["Gender Gap\n(inverted)", "Representation", "Digital Inclusion"]
        values = [1 - gap, rep, digi]
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(22,163,74,0.15)",
            line=dict(color="#16a34a", width=2),
            name="Your result",
        ))
        fig.add_trace(go.Scatterpolar(
            r=[0.9, 0.85, 0.80, 0.9],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(14,165,233,0.08)",
            line=dict(color="#0ea5e9", width=1, dash="dash"),
            name="UNESCO target",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=300,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📖 How to interpret gender audit results"):
            _module_info_card("gender_audit")


def _render_agent_economy(ae: dict) -> None:
    if not ae:
        return
    with st.expander("💰 Agent Economy — Full Results", expanded=True):
        gini     = ae.get("gini_coefficient", ae.get("allocation_gini", 0))
        alloc_eq = ae.get("allocation_equity", 0)

        c1, c2 = st.columns(2)
        c1.metric("Gini Coefficient",  f"{gini:.3f}",
                  help="0=perfect equality, 1=complete inequality")
        c2.metric("Allocation Equity", f"{alloc_eq:.2f}" if alloc_eq else "N/A")

        # Agent summary table
        agents = ae.get("agent_summary", [])
        if agents:
            df_a = pd.DataFrame(agents)
            st.dataframe(df_a, use_container_width=True)

        # Auction results
        auction = ae.get("auction_results", [])
        if auction:
            st.markdown("**Auction round outcomes:**")
            df_au = pd.DataFrame(auction)
            if "winner_wealth" in df_au.columns and "resource" in df_au.columns:
                fig = px.bar(df_au, x="resource", y="winner_wealth",
                             title="Resource allocation by winner wealth",
                             color="winner_wealth",
                             color_continuous_scale="RdYlGn_r")
                fig.update_layout(height=280)
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("📖 How to interpret agent economy results"):
            _module_info_card("agent_economy")


def _render_redteam(rt: dict) -> None:
    if not rt:
        return
    with st.expander("🛡️ Multimodal Red Team — Full Results", expanded=True):
        bypass = rt.get("combined_bypass_rate", 0)
        risk   = rt.get("combined_sociotechnical_risk", 0)

        c1, c2 = st.columns(2)
        c1.metric("Combined Bypass Rate",        f"{bypass:.1%}",
                  delta=f"{'HIGH RISK' if bypass > 0.3 else 'Acceptable'}",
                  delta_color="inverse")
        c2.metric("Sociotechnical Risk",          f"{risk:.2f}",
                  delta=f"{'HIGH' if risk > 0.6 else 'Moderate'}",
                  delta_color="inverse")

        rows = rt.get("modality_results", [])
        if rows:
            df_rt = pd.DataFrame([{
                "Modality":       r.get("modality", ""),
                "Attack Vector":  r.get("attack_vector", ""),
                "Severity":       r.get("severity", ""),
                "Bypass Rate":    f"{r.get('bypass_rate', 0):.1%}",
                "Sociotech Risk": f"{r.get('sociotechnical_risk', 0):.2f}",
            } for r in rows])
            st.dataframe(df_rt, use_container_width=True)

        for r in rows:
            if r.get("vr_scenario"):
                st.markdown(
                    f"<div style='background:#fef3c7;border-left:3px solid #d97706;"
                    f"border-radius:6px;padding:10px 14px;margin:8px 0;'>"
                    f"<b>🥽 Deepfake Scenario:</b> {r.get("vr_scenario","")}</div>",
                    unsafe_allow_html=True,
                )
                break

        with st.expander("📖 How to interpret red team results"):
            _module_info_card("multimodal_redteam")


def _render_arena(arena: dict) -> None:
    if not arena:
        return
    with st.expander("⚔️ Strategic Arena — Full Results", expanded=True):
        standings = arena.get("final_standings", [])
        if standings:
            df_s = pd.DataFrame(standings)
            st.dataframe(df_s, use_container_width=True)

        deception = arena.get("deception_rate", 0)
        welfare   = arena.get("social_welfare", 0)
        c1, c2 = st.columns(2)
        c1.metric("Deception Rate",  f"{deception:.1%}",
                  help="Fraction of agents that misrepresented preferences")
        c2.metric("Social Welfare",  f"{welfare:.2f}",
                  help="Aggregate welfare score (higher = more equitable outcome)")

        with st.expander("📖 How to interpret arena results"):
            _module_info_card("arena")


# ── Multi-challenge panel ──────────────────────────────────────────────────────

def multi_challenge_panel(domain: str, latest_results: dict) -> None:
    """
    Show ALL active challenges for the domain simultaneously.
    Multiple challenges shown in a card grid — user can see progress on all at once.
    """
    challenges = _get_challenges(domain)
    if not challenges:
        return

    # Track completed challenges
    if "_completed_challenges" not in st.session_state:
        st.session_state["_completed_challenges"] = []

    st.markdown("---")
    st.markdown(
        "<p style='font-size:.82rem;font-weight:700;color:#7c3aed;"
        "letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px;'>"
        "🎯 ACTIVE CHALLENGES</p>",
        unsafe_allow_html=True,
    )

    # Group by difficulty
    difficulties = ["Easy", "Medium", "Hard", "Expert"]
    shown = 0

    for diff in difficulties:
        diff_challenges = [c for c in challenges if c.get("difficulty", "Medium") == diff]
        if not diff_challenges:
            continue

        bg, fg, dot = DIFFICULTY_COLOURS[diff]
        st.markdown(
            f"<p style='font-size:.72rem;font-weight:600;color:{fg};"
            f"margin:8px 0 4px;'>{dot} {diff.upper()} CHALLENGES</p>",
            unsafe_allow_html=True,
        )

        cols = st.columns(min(2, len(diff_challenges)))
        for col, challenge in zip(cols, diff_challenges[:2]):
            with col:
                _single_challenge_card(challenge, latest_results, bg, fg)
                shown += 1

    if shown == 0:
        st.info("Run the simulation to check challenge status.")


def _single_challenge_card(challenge: dict, results: dict,
                             bg: str, fg: str) -> None:
    """Render one challenge card with live status."""
    cid     = challenge["id"]
    metric  = challenge["target_metric"]
    target  = challenge["target_value"]
    cond    = challenge["pass_condition"]
    current = results.get(metric)
    done    = cid in st.session_state.get("_completed_challenges", [])

    # Status
    if current is not None:
        if cond == "less_than":
            passed = float(current) < target
            progress = min(1.0, target / max(float(current), 0.001)) if current else 0
        else:
            passed = float(current) > target
            progress = min(1.0, float(current) / max(target, 0.001))
    else:
        passed   = False
        progress = 0

    border_colour = "#16a34a" if passed else ("#e2e8f0" if not done else "#94a3b8")
    status_icon   = "✅" if passed else ("🔄" if current is not None else "⏳")

    st.markdown(
        f"<div style='background:{bg};border:2px solid {border_colour};"
        f"border-radius:10px;padding:12px;margin-bottom:8px;min-height:160px;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
        f"<b style='font-size:.85rem;color:#0f172a;'>{challenge.get("title","")}</b>"
        f"<span style='font-size:1.2rem;'>{status_icon}</span>"
        f"</div>"
        f"<p style='font-size:.75rem;color:#374151;margin:4px 0 6px;"
        f"line-height:1.4;'>{challenge.get("story","")}</p>"
        f"<div style='background:rgba(0,0,0,.08);border-radius:4px;height:5px;'>"
        f"<div style='background:{fg};border-radius:4px;height:5px;"
        f"width:{progress*100:.0f}%;'></div>"
        f"</div>"
        f"<div style='display:flex;justify-content:space-between;margin-top:6px;"
        f"font-size:.72rem;color:{fg};'>"
        # note: target_label pre-computed above
        f"<span>Target: {challenge.get('target_label', str(target))}</span>"
        f"<span>+{challenge.get("points",0)} pts</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if passed and not done:
        st.session_state["_completed_challenges"].append(cid)
        st.balloons()
        st.success(f"🏆 Challenge complete! +{challenge.get("points",0)} points!")
        try:
            from components.gags_interactive import award_points, award_badge
            award_points(challenge["points"], challenge["title"])
            award_badge(challenge.get("badge", "policy_pass"))
        except Exception:
            pass

    if not passed and current is not None:
        with st.expander("💡 Hint"):
            st.caption(challenge.get("hint", "Adjust bias parameters and try again."))


# ── Admin challenge management panel ──────────────────────────────────────────

def admin_challenge_panel(domain: str) -> None:
    """
    Password-protected admin panel for managing domain challenges.
    Allows adding, editing, and removing challenges without redeployment.
    """
    with st.expander("🔐 Admin: Manage Challenges", expanded=False):
        pwd = st.text_input("Admin password", type="password",
                            key=f"_admin_pwd_{domain}")

        if pwd != ADMIN_PASSWORD:
            st.info("Enter admin password to manage challenges.")
            return

        st.success("✅ Admin access granted")
        challenges = _get_challenges(domain)

        tab_view, tab_add, tab_edit = st.tabs(
            ["📋 View All", "➕ Add Challenge", "✏️ Edit / Remove"]
        )

        # ── View all ──────────────────────────────────────────────────────────
        with tab_view:
            if not challenges:
                st.info("No challenges configured for this domain.")
            else:
                df_ch = pd.DataFrame([{
                    "ID":         c["id"],
                    "Title":      c["title"],
                    "Metric":     c["target_metric"],
                    "Target":     c["target_value"],
                    "Condition":  c["pass_condition"],
                    "Difficulty": c.get("difficulty", "Medium"),
                    "Points":     c["points"],
                } for c in challenges])
                st.dataframe(df_ch, use_container_width=True)

        # ── Add new challenge ──────────────────────────────────────────────────
        with tab_add:
            st.markdown("**Add a new challenge:**")
            col1, col2 = st.columns(2)
            with col1:
                new_id    = st.text_input("Challenge ID (unique)",
                                          key=f"_new_id_{domain}",
                                          placeholder="e.g. hc_custom_1")
                new_title = st.text_input("Title",
                                          key=f"_new_title_{domain}",
                                          placeholder="e.g. 🩺 Custom Challenge")
                new_story = st.text_area("Story / Brief",
                                         key=f"_new_story_{domain}",
                                         height=80,
                                         placeholder="Describe the real-world scenario...")
                new_hint  = st.text_input("Hint",
                                          key=f"_new_hint_{domain}",
                                          placeholder="Hint for students...")
            with col2:
                new_metric = st.text_input("Target metric",
                                           key=f"_new_metric_{domain}",
                                           placeholder="e.g. fairness_score")
                new_target = st.number_input("Target value",
                                             key=f"_new_target_{domain}",
                                             min_value=0.0, max_value=1.0,
                                             value=0.15, step=0.01)
                new_cond   = st.selectbox("Pass condition",
                                          ["less_than", "greater_than"],
                                          key=f"_new_cond_{domain}")
                new_diff   = st.selectbox("Difficulty",
                                          ["Easy", "Medium", "Hard", "Expert"],
                                          key=f"_new_diff_{domain}")
                new_pts    = st.number_input("Points",
                                             key=f"_new_pts_{domain}",
                                             min_value=10, max_value=500,
                                             value=100, step=10)

            if st.button("➕ Add Challenge", key=f"_add_ch_{domain}",
                          type="primary"):
                if not new_id or not new_title or not new_metric:
                    st.error("ID, Title, and Target metric are required.")
                elif any(c["id"] == new_id for c in challenges):
                    st.error(f"Challenge ID '{new_id}' already exists.")
                else:
                    challenges.append({
                        "id":           new_id,
                        "title":        new_title,
                        "story":        new_story,
                        "hint":         new_hint,
                        "target_metric": new_metric,
                        "target_value":  float(new_target),
                        "target_label":  f"{new_cond.replace('_',' ')} {new_target}",
                        "pass_condition": new_cond,
                        "difficulty":   new_diff,
                        "points":       int(new_pts),
                        "badge":        "policy_pass",
                    })
                    _save_challenges(domain, challenges)
                    st.success(f"✅ Challenge '{new_title}' added!")
                    st.rerun()

        # ── Edit / Remove ──────────────────────────────────────────────────────
        with tab_edit:
            if not challenges:
                st.info("No challenges to edit.")
            else:
                ch_ids = [c["id"] for c in challenges]
                sel_id = st.selectbox("Select challenge to edit/remove",
                                      ch_ids, key=f"_sel_ch_{domain}")
                ch = next(c for c in challenges if c["id"] == sel_id)

                edit_title  = st.text_input("Title",    value=ch["title"],
                                            key=f"_e_title_{domain}")
                edit_story  = st.text_area("Story",     value=ch["story"],
                                           key=f"_e_story_{domain}", height=80)
                edit_hint   = st.text_input("Hint",     value=ch.get("hint",""),
                                            key=f"_e_hint_{domain}")
                edit_target = st.number_input("Target value",
                                              value=float(ch["target_value"]),
                                              key=f"_e_target_{domain}",
                                              step=0.01)
                edit_pts    = st.number_input("Points",
                                              value=int(ch["points"]),
                                              key=f"_e_pts_{domain}", step=10)
                edit_diff   = st.selectbox("Difficulty",
                                           ["Easy","Medium","Hard","Expert"],
                                           index=["Easy","Medium","Hard","Expert"]
                                           .index(ch.get("difficulty","Medium")),
                                           key=f"_e_diff_{domain}")

                col_save, col_del = st.columns(2)
                with col_save:
                    if st.button("💾 Save Changes", key=f"_save_ch_{domain}",
                                  type="primary"):
                        for i, c in enumerate(challenges):
                            if c["id"] == sel_id:
                                challenges[i].update({
                                    "title":        edit_title,
                                    "story":        edit_story,
                                    "hint":         edit_hint,
                                    "target_value": float(edit_target),
                                    "points":       int(edit_pts),
                                    "difficulty":   edit_diff,
                                    "target_label": (
                                        f"{c.get("pass_condition","").replace("_"," ")} "
                                        f"{edit_target}"
                                    ),
                                })
                                break
                        _save_challenges(domain, challenges)
                        st.success("✅ Challenge updated!")
                        st.rerun()

                with col_del:
                    if st.button("🗑️ Remove Challenge", key=f"_del_ch_{domain}",
                                  type="secondary"):
                        challenges = [c for c in challenges if c["id"] != sel_id]
                        _save_challenges(domain, challenges)
                        st.success(f"Removed '{sel_id}'")
                        st.rerun()

        # ── Reset to defaults ──────────────────────────────────────────────────
        st.markdown("---")
        if st.button("🔄 Reset to default challenges",
                     key=f"_reset_ch_{domain}"):
            _save_challenges(domain, DEFAULT_CHALLENGES.get(domain, [][:]))
            st.success("Reset to defaults.")
            st.rerun()
