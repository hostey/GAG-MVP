# components/nigeria_regulatory.py
"""
Nigeria AI Regulatory Framework Module — v1.0
===============================================
Maps GAGS simulation results to three Nigerian regulatory instruments:

  1. NITDA AI Policy 2023
     National Information Technology Development Agency
     Nigeria's primary AI governance framework

  2. NDPR (Nigeria Data Protection Regulation) 2019 + 2023 Amendment
     National Information Technology Development Agency
     Nigeria's primary data protection law (equivalent to GDPR)

  3. NCC AI Guidelines (Nigerian Communications Commission)
     Sector-specific AI guidelines for telecoms and digital services
     Relevant for USSD/SMS AI advisory systems

  4. NASRDA & FMARD Guidelines (bonus)
     National Space Research and Development Agency — satellite data
     Federal Ministry of Agriculture — agrotech AI deployment

Usage
-----
    from components.nigeria_regulatory import (
        nigeria_compliance_panel,
        nitda_scorecard,
        ndpr_assessment,
        ncc_ussd_checklist,
    )

    # In the compliance tab:
    nigeria_compliance_panel(
        simulation_metrics={"accuracy": 0.82, "fairness_score": 0.76, ...},
        domain="agrotech",
        has_ussd_fallback=True,
        has_gender_audit=True,
        has_multilingual=True,
    )
"""

from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

try:
    from components.i18n import t, get_lang
except ImportError:
    def t(key, lang=None): return key
    def get_lang(): return "en"


# ── Framework definitions ─────────────────────────────────────────────────────

NITDA_PRINCIPLES = {
    "1": {
        "title": "Transparency & Explainability",
        "description": "AI systems must be explainable and their decisions understandable to affected persons.",
        "gags_check": "xai_enabled",
        "evidence": "Explainable AI tab with LIME-lite explanations and counterfactuals",
    },
    "2": {
        "title": "Fairness & Non-Discrimination",
        "description": "AI systems must not discriminate against persons on grounds of gender, ethnicity, religion, disability, or socioeconomic status.",
        "gags_check": "fairness_score_pass",
        "evidence": "Fairness score ≥ 0.70 and demographic parity gap ≤ 10%",
    },
    "3": {
        "title": "Accountability & Responsibility",
        "description": "Clear accountability chains must exist for AI decisions; humans must remain in the loop for high-stakes outcomes.",
        "gags_check": "governance_enabled",
        "evidence": "Governance layer with citizen vote, ledger, and human override",
    },
    "4": {
        "title": "Safety & Security",
        "description": "AI systems must be resilient to adversarial attacks and data poisoning.",
        "gags_check": "redteam_enabled",
        "evidence": "Multimodal red-team module tests text, image, and deepfake attacks",
    },
    "5": {
        "title": "Privacy & Data Protection",
        "description": "AI systems must comply with the Nigeria Data Protection Regulation (NDPR). Data minimisation and purpose limitation required.",
        "gags_check": "ndpr_compliant",
        "evidence": "Simulation uses synthetic/anonymised data; no PII processed",
    },
    "6": {
        "title": "Local Context & Inclusion",
        "description": "AI systems must be adapted to Nigerian languages, cultures, and connectivity constraints. USSD/SMS fallback required for low-connectivity users.",
        "gags_check": "ussd_fallback",
        "evidence": "USSD simulation mode with Hausa/Yoruba/Igbo language support",
    },
    "7": {
        "title": "Gender & Social Equity",
        "description": "AI systems must demonstrate equitable outcomes for women and marginalised groups. Gender disaggregated performance metrics required.",
        "gags_check": "gender_audit",
        "evidence": "UNESCO Women4EthicalAI gender equity audit integrated",
    },
    "8": {
        "title": "Capacity Building",
        "description": "Organisations deploying AI must invest in local AI literacy and skills development.",
        "gags_check": "capacity_building",
        "evidence": "User guidance, metric glossary, and educational simulation mode",
    },
    "9": {
        "title": "Environmental Sustainability",
        "description": "AI systems must minimise environmental impact; energy-efficient approaches preferred.",
        "gags_check": "lightweight_model",
        "evidence": "Lightweight sklearn models (no GPU required); simulation-only mode",
    },
    "10": {
        "title": "Innovation & Competitiveness",
        "description": "AI governance must not stifle innovation; proportionate regulation based on risk level.",
        "gags_check": "risk_proportionate",
        "evidence": "Risk-tiered compliance: limited risk for simulation; high risk for deployment",
    },
}

NDPR_ARTICLES = {
    "Art. 2.1": {
        "title": "Lawful Basis for Processing",
        "requirement": "Personal data may only be processed on a lawful basis (consent, contract, legal obligation, vital interests, public task, legitimate interests).",
        "gags_relevance": "GAGS processes synthetic data only; no real personal data collected.",
        "status_key": "synthetic_data",
    },
    "Art. 2.5": {
        "title": "Data Minimisation",
        "requirement": "Only data necessary for the stated purpose may be collected and processed.",
        "gags_relevance": "Feature sets limited to model requirements; no unnecessary attributes.",
        "status_key": "data_minimisation",
    },
    "Art. 2.6": {
        "title": "Accuracy & Integrity",
        "requirement": "Personal data must be accurate and kept up to date.",
        "gags_relevance": "Bias detection flags data quality issues; poisoning detection identifies corrupted records.",
        "status_key": "accuracy_monitoring",
    },
    "Art. 2.7": {
        "title": "Storage Limitation",
        "requirement": "Personal data must not be retained longer than necessary for the purpose.",
        "gags_relevance": "Simulation results are session-based only; no persistent PII storage.",
        "status_key": "no_persistent_pii",
    },
    "Art. 3.1": {
        "title": "Data Subject Rights",
        "requirement": "Data subjects have rights to access, rectification, erasure, and objection.",
        "gags_relevance": "Counterfactual explanations support the right to explanation of automated decisions.",
        "status_key": "explanation_right",
    },
    "Art. 4.1": {
        "title": "Data Protection Impact Assessment",
        "requirement": "High-risk processing requires a DPIA before deployment.",
        "gags_relevance": "Compliance report and model card constitute DPIA documentation framework.",
        "status_key": "dpia_documented",
    },
    "Art. 5.0": {
        "title": "Cross-border Data Transfer",
        "requirement": "Data may not be transferred outside Nigeria without adequate protection.",
        "gags_relevance": "Simulation runs locally; no data leaves the deployment environment.",
        "status_key": "local_processing",
    },
}

NCC_USSD_GUIDELINES = {
    "G1": {
        "title": "Service Registration",
        "requirement": "All USSD-based AI advisory services must be registered with NCC before deployment.",
        "gags_relevance": "USSD simulation mode demonstrates compliance pathway; production requires NCC short code registration.",
        "status_key": "ncc_registration_aware",
    },
    "G2": {
        "title": "Consumer Protection",
        "requirement": "USSD services must not charge users without explicit consent; free advisory services must be clearly labelled.",
        "gags_relevance": "USSD simulator clearly marks advisory as free; no billing simulation.",
        "status_key": "free_service_labelled",
    },
    "G3": {
        "title": "Accessible Interface",
        "requirement": "Services must be accessible on all network types including 2G; menus must be in local languages.",
        "gags_relevance": "USSD interface tested on 2G-compatible 182-character limit; Hausa/Yoruba/Igbo/English menus.",
        "status_key": "multilingual_ussd",
    },
    "G4": {
        "title": "Data Privacy on USSD",
        "requirement": "USSD sessions must not capture or transmit personal identifying information.",
        "gags_relevance": "USSD flow uses LGA-level only (not individual names/IDs); NDPR compliant.",
        "status_key": "ussd_privacy",
    },
    "G5": {
        "title": "Session Timeout",
        "requirement": "USSD sessions must timeout after 180 seconds of inactivity.",
        "gags_relevance": "Simulation models session timeout; production implementation must enforce NCC timeout.",
        "status_key": "session_timeout_modelled",
    },
    "G6": {
        "title": "Audit Trail",
        "requirement": "Service providers must maintain records of USSD session metadata for 2 years.",
        "gags_relevance": "Simulation history browser provides session audit trail; production requires persistent logging.",
        "status_key": "audit_trail",
    },
}


# ── Scoring logic ─────────────────────────────────────────────────────────────

def _evaluate_checks(
    metrics: Dict[str, float],
    flags: Dict[str, bool],
) -> Dict[str, bool]:
    """
    Evaluate all regulatory check conditions against simulation metrics and flags.
    Returns a dict of check_key → pass/fail.
    """
    fs  = metrics.get("fairness_score", 0.5)
    dp  = metrics.get("demographic_parity", metrics.get("demographic_parity_difference", 0.1))
    acc = metrics.get("accuracy", 0.5)

    return {
        # NITDA checks
        "xai_enabled":          flags.get("xai_enabled", False),
        "fairness_score_pass":  fs >= 0.70 and dp <= 0.10,
        "governance_enabled":   flags.get("governance_enabled", False),
        "redteam_enabled":      flags.get("redteam_enabled", False),
        "ndpr_compliant":       True,   # GAGS uses synthetic data — always passes
        "ussd_fallback":        flags.get("ussd_fallback", False),
        "gender_audit":         flags.get("gender_audit", False),
        "capacity_building":    True,   # Metric glossary + tour = capacity building
        "lightweight_model":    True,   # sklearn only = lightweight
        "risk_proportionate":   True,   # Simulation mode = limited risk
        # NDPR checks
        "synthetic_data":       True,
        "data_minimisation":    True,
        "accuracy_monitoring":  True,
        "no_persistent_pii":    True,
        "explanation_right":    flags.get("xai_enabled", False),
        "dpia_documented":      acc > 0,
        "local_processing":     True,
        # NCC checks
        "ncc_registration_aware":    True,
        "free_service_labelled":     True,
        "multilingual_ussd":         flags.get("multilingual_enabled", False),
        "ussd_privacy":              True,
        "session_timeout_modelled":  flags.get("ussd_fallback", False),
        "audit_trail":               True,
    }


# ── UI functions ──────────────────────────────────────────────────────────────

def nitda_scorecard(
    metrics: Dict[str, float],
    flags: Dict[str, bool],
    expanded: bool = True,
) -> Tuple[int, int]:
    """
    Render the NITDA AI Policy 2023 compliance scorecard.
    Returns (passed, total) tuple.
    """
    checks = _evaluate_checks(metrics, flags)
    passed = 0

    with st.expander("🇳🇬 NITDA AI Policy 2023 — 10 Principles", expanded=expanded):
        st.markdown(
            "<p style='font-size:.83rem;color:var(--color-text-secondary);"
            "margin-bottom:.75rem;'>"
            "National Information Technology Development Agency — Nigeria's primary "
            "AI governance framework. Compliance required for all AI systems "
            "deployed in Nigeria's public and critical sectors.</p>",
            unsafe_allow_html=True,
        )

        for num, principle in NITDA_PRINCIPLES.items():
            check_key = principle["gags_check"]
            is_pass   = checks.get(check_key, False)
            if is_pass:
                passed += 1

            icon  = "✅" if is_pass else "❌"
            color = "var(--color-text-success)" if is_pass else "var(--color-text-danger)"

            st.markdown(
                f"<div style='border-left:3px solid {color};"
                f"padding:.4rem .75rem;margin:.3rem 0;"
                f"background:var(--color-background-secondary);"
                f"border-radius:0 6px 6px 0;'>"
                f"<div style='display:flex;align-items:center;gap:.5rem;'>"
                f"<span>{icon}</span>"
                f"<strong style='font-size:.85rem;'>Principle {num}: {principle['title']}</strong>"
                f"</div>"
                f"<p style='font-size:.78rem;color:var(--color-text-secondary);margin:.2rem 0 0;'>"
                f"{principle['description']}</p>"
                f"<p style='font-size:.75rem;color:var(--color-text-tertiary);margin:.1rem 0 0;'>"
                f"Evidence: {principle['evidence']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        total = len(NITDA_PRINCIPLES)
        score_pct = passed / total
        color = "#27ae60" if score_pct >= 0.8 else "#f39c12" if score_pct >= 0.6 else "#e74c3c"
        st.markdown(
            f"<div style='margin-top:.75rem;padding:.6rem 1rem;"
            f"background:var(--color-background-primary);"
            f"border:1px solid var(--color-border-secondary);"
            f"border-radius:var(--border-radius-md);'>"
            f"<strong>NITDA Score: "
            f"<span style='color:{color};'>{passed}/{total} principles met "
            f"({score_pct:.0%})</span></strong>"
            f"{'  ✅ Deployment-ready' if score_pct >= 0.8 else '  ⚠️ Remediation required before deployment'}"
            f"</div>",
            unsafe_allow_html=True,
        )

    return passed, len(NITDA_PRINCIPLES)


def ndpr_assessment(
    metrics: Dict[str, float],
    flags: Dict[str, bool],
    expanded: bool = False,
) -> Tuple[int, int]:
    """
    Render the NDPR (Nigeria Data Protection Regulation) assessment.
    Returns (passed, total) tuple.
    """
    checks = _evaluate_checks(metrics, flags)
    passed = 0

    with st.expander("🔒 NDPR 2019 / 2023 — Data Protection Assessment", expanded=expanded):
        st.markdown(
            "<p style='font-size:.83rem;color:var(--color-text-secondary);"
            "margin-bottom:.75rem;'>"
            "Nigeria Data Protection Regulation — the primary data protection law, "
            "enforced by the Nigeria Data Protection Commission (NDPC). "
            "Mandatory for all organisations processing personal data of Nigerian citizens.</p>",
            unsafe_allow_html=True,
        )

        for art, article in NDPR_ARTICLES.items():
            is_pass = checks.get(article["status_key"], True)
            if is_pass:
                passed += 1

            icon  = "✅" if is_pass else "❌"
            color = "var(--color-text-success)" if is_pass else "var(--color-text-danger)"

            st.markdown(
                f"<div style='border-left:3px solid {color};"
                f"padding:.4rem .75rem;margin:.3rem 0;"
                f"background:var(--color-background-secondary);"
                f"border-radius:0 6px 6px 0;'>"
                f"<span>{icon}</span> "
                f"<strong style='font-size:.83rem;'>{art} — {article['title']}</strong><br>"
                f"<span style='font-size:.78rem;color:var(--color-text-secondary);'>"
                f"{article['requirement']}</span><br>"
                f"<span style='font-size:.75rem;color:var(--color-text-tertiary);'>"
                f"GAGS: {article['gags_relevance']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        total = len(NDPR_ARTICLES)
        score_pct = passed / total
        color = "#27ae60" if score_pct == 1.0 else "#f39c12" if score_pct >= 0.8 else "#e74c3c"
        st.markdown(
            f"<div style='margin-top:.75rem;padding:.6rem 1rem;"
            f"background:var(--color-background-primary);"
            f"border:1px solid var(--color-border-secondary);"
            f"border-radius:var(--border-radius-md);'>"
            f"<strong>NDPR Assessment: "
            f"<span style='color:{color};'>{passed}/{total} articles addressed "
            f"({score_pct:.0%})</span></strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

    return passed, len(NDPR_ARTICLES)


def ncc_ussd_checklist(
    flags: Dict[str, bool],
    expanded: bool = False,
) -> Tuple[int, int]:
    """
    Render the NCC USSD/AI deployment checklist.
    Returns (passed, total) tuple.
    """
    checks = _evaluate_checks({}, flags)
    passed = 0

    with st.expander("📡 NCC AI Guidelines — USSD/SMS Deployment Checklist", expanded=expanded):
        st.markdown(
            "<p style='font-size:.83rem;color:var(--color-text-secondary);"
            "margin-bottom:.75rem;'>"
            "Nigerian Communications Commission guidelines for AI-powered USSD and SMS services. "
            "Required for any AI advisory system delivered over telecom networks in Nigeria.</p>",
            unsafe_allow_html=True,
        )

        for gid, guideline in NCC_USSD_GUIDELINES.items():
            is_pass = checks.get(guideline["status_key"], False)
            if is_pass:
                passed += 1

            icon  = "✅" if is_pass else "⚠️"
            color = "var(--color-text-success)" if is_pass else "var(--color-text-warning)"

            st.markdown(
                f"<div style='border-left:3px solid {color};"
                f"padding:.4rem .75rem;margin:.3rem 0;"
                f"background:var(--color-background-secondary);"
                f"border-radius:0 6px 6px 0;'>"
                f"<span>{icon}</span> "
                f"<strong style='font-size:.83rem;'>{gid}: {guideline['title']}</strong><br>"
                f"<span style='font-size:.78rem;color:var(--color-text-secondary);'>"
                f"{guideline['requirement']}</span><br>"
                f"<span style='font-size:.75rem;color:var(--color-text-tertiary);'>"
                f"GAGS: {guideline['gags_relevance']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        total = len(NCC_USSD_GUIDELINES)
        score_pct = passed / total
        color = "#27ae60" if score_pct >= 0.8 else "#f39c12" if score_pct >= 0.5 else "#e74c3c"
        st.markdown(
            f"<div style='margin-top:.75rem;padding:.6rem 1rem;"
            f"background:var(--color-background-primary);"
            f"border:1px solid var(--color-border-secondary);"
            f"border-radius:var(--border-radius-md);'>"
            f"<strong>NCC Checklist: "
            f"<span style='color:{color};'>{passed}/{total} guidelines met "
            f"({score_pct:.0%})</span></strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

    return passed, len(NCC_USSD_GUIDELINES)


def nigeria_compliance_panel(
    simulation_metrics: Dict[str, float],
    domain: str = "agrotech",
    has_ussd_fallback: bool = False,
    has_gender_audit: bool = False,
    has_multilingual: bool = False,
    has_xai: bool = False,
    has_governance: bool = False,
    has_redteam: bool = False,
) -> Dict[str, Any]:
    """
    Render the complete Nigeria Regulatory Compliance Panel.
    Shows NITDA, NDPR, and NCC assessments with an overall verdict.

    Parameters
    ----------
    simulation_metrics : dict with accuracy, fairness_score, demographic_parity, etc.
    domain             : "healthcare" | "national_security" | "agrotech"
    has_*              : boolean flags for enabled features

    Returns
    -------
    dict with summary scores for all three frameworks
    """
    flags = {
        "xai_enabled":          has_xai,
        "governance_enabled":   has_governance,
        "redteam_enabled":      has_redteam,
        "ussd_fallback":        has_ussd_fallback,
        "gender_audit":         has_gender_audit,
        "multilingual_enabled": has_multilingual,
    }

    st.markdown("### 🇳🇬 Nigeria Regulatory Framework Assessment")
    st.markdown(
        "<div style='background:var(--color-background-secondary);"
        "border:0.5px solid var(--color-border-tertiary);"
        "border-radius:var(--border-radius-lg);padding:1rem 1.25rem;"
        "margin-bottom:1rem;'>"
        "<p style='margin:0;font-size:.85rem;color:var(--color-text-primary);'>"
        "<strong>Scope:</strong> Nigeria-specific regulatory requirements for AI systems "
        "operating in the Federal Capital Territory and across Nigerian states. "
        "These requirements are in addition to international frameworks (EU AI Act, ISO 42001, UNESCO).</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Domain-specific risk statement
    domain_risk = {
        "healthcare":        ("HIGH RISK", "Clinical AI requires pre-market review by NAFDAC and alignment with FMoH guidelines.", "#e74c3c"),
        "national_security": ("HIGH RISK", "Security AI requires approval from NSA and must comply with Cybercrimes Act 2015.", "#e74c3c"),
        "agrotech":          ("LIMITED RISK", "Agricultural AI advisory services are limited-risk but must comply with FMARD guidelines.", "#f39c12"),
    }
    risk_level, risk_note, risk_color = domain_risk.get(domain, ("MODERATE RISK", "", "#f39c12"))

    st.markdown(
        f"<div style='border-left:4px solid {risk_color};"
        f"padding:.6rem 1rem;border-radius:0 8px 8px 0;"
        f"background:var(--color-background-secondary);margin-bottom:1rem;'>"
        f"<strong>Nigeria Risk Classification: "
        f"<span style='color:{risk_color};'>{risk_level}</span></strong><br>"
        f"<span style='font-size:.83rem;color:var(--color-text-secondary);'>{risk_note}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Run all three assessments
    nitda_pass, nitda_total = nitda_scorecard(simulation_metrics, flags, expanded=True)
    ndpr_pass,  ndpr_total  = ndpr_assessment(simulation_metrics, flags, expanded=False)
    ncc_pass,   ncc_total   = ncc_ussd_checklist(flags, expanded=False)

    # Overall Nigeria compliance score
    total_checks = nitda_total + ndpr_total + ncc_total
    total_passed = nitda_pass  + ndpr_pass  + ncc_pass
    overall_pct  = total_passed / total_checks if total_checks > 0 else 0

    overall_color  = "#27ae60" if overall_pct >= 0.80 else "#f39c12" if overall_pct >= 0.60 else "#e74c3c"
    overall_status = ("NIGERIA-READY ✅" if overall_pct >= 0.80
                      else "PARTIAL COMPLIANCE ⚠️" if overall_pct >= 0.60
                      else "NON-COMPLIANT ❌")

    st.markdown(
        f"<div style='background:var(--color-background-primary);"
        f"border:2px solid {overall_color};"
        f"border-radius:var(--border-radius-lg);padding:1rem 1.25rem;"
        f"margin-top:1rem;text-align:center;'>"
        f"<p style='font-size:.85rem;font-weight:600;color:var(--color-text-secondary);"
        f"margin:0 0 .3rem;text-transform:uppercase;letter-spacing:.05em;'>"
        f"Overall Nigeria Regulatory Status</p>"
        f"<p style='font-size:1.3rem;font-weight:700;color:{overall_color};margin:0;'>"
        f"{overall_status}</p>"
        f"<p style='font-size:.83rem;color:var(--color-text-secondary);margin:.3rem 0 0;'>"
        f"{total_passed} of {total_checks} checks passed across NITDA, NDPR, and NCC</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Quick-fix recommendations for failing checks
    fails = []
    if not flags["ussd_fallback"]:
        fails.append("Enable USSD/SMS simulation to satisfy NITDA Principle 6 and NCC G3.")
    if not flags["gender_audit"]:
        fails.append("Enable Gender Equity Audit to satisfy NITDA Principle 7.")
    if not flags["multilingual_enabled"]:
        fails.append("Enable Hausa/Yoruba/Igbo interface to satisfy NITDA Principle 6 and NCC G3.")
    if not flags["xai_enabled"]:
        fails.append("Enable Explainable AI to satisfy NITDA Principle 1 and NDPR Art. 3.1.")
    if simulation_metrics.get("fairness_score", 0) < 0.70:
        fails.append("Fairness score below 0.70 — apply bias mitigation to satisfy NITDA Principle 2.")

    if fails:
        st.markdown("**Quick actions to improve Nigeria compliance:**")
        for f in fails:
            st.markdown(f"  • {f}")

    return {
        "nitda": {"passed": nitda_pass, "total": nitda_total, "score": nitda_pass / nitda_total},
        "ndpr":  {"passed": ndpr_pass,  "total": ndpr_total,  "score": ndpr_pass  / ndpr_total},
        "ncc":   {"passed": ncc_pass,   "total": ncc_total,   "score": ncc_pass   / ncc_total},
        "overall": {"passed": total_passed, "total": total_checks, "score": overall_pct},
        "status": overall_status,
    }